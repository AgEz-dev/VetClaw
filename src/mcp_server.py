"""VetClaw Model Context Protocol (MCP) Server.

符合开放 MCP 规范 (Protocol Version: 2024-11-05)。
支持双模式：
1. 标准输入输出 (stdio)：供 Claude Desktop, Cursor, Antigravity 等 MCP Client 直接挂载。
2. HTTP / JSON-RPC 2.0：供 Spring Boot 医院中台与微服务以 MCP 协议标准调用。

能力边界：
- Tools: search_knowledge_base, check_drug_safety, get_inventory_summary, triage_patient, review_prescription
- Resources: vetclaw://knowledge/* (门诊指南、毒物手册、用药矩阵)
- Prompts: outpatient_preconsultation, soap_record_draft
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 动态加载工具与知识库
from tools import registry
import knowledge_tool  # noqa: F401
import hospital_tools  # noqa: F401
from triage import triage_note, extract_slots
from prescription_review import review_prescription

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


class VetClawMcpServer:
    def __init__(self, name: str = "vetclaw-clinical-mcp", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.protocol_version = "2024-11-05"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取所有已注册的 MCP Tools 规范元数据 (JSON Schema)。"""
        return [
            {
                "name": "search_knowledge_base",
                "description": "基于 BGE 稠密向量 + BM25 稀疏索引与 RRF 融合算法，检索小动物临床指南与用药切片。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索临床病症、用药或指南关键词"},
                        "top_k": {"type": "integer", "description": "返回最相关切片数，默认 3", "default": 3}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "check_drug_safety",
                "description": "校验特定犬猫药品针对物种、体重与日剂量的安全性，排查黑框用药禁忌与中毒红线。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "drug_name": {"type": "string", "description": "药品名称（如：对乙酰氨基酚、恩诺沙星）"},
                        "species": {"type": "string", "description": "物种（犬 / 猫）"},
                        "weight_kg": {"type": "number", "description": "患宠体重(kg)"},
                        "daily_dose_mg": {"type": "number", "description": "计划每日给药剂量(mg)"}
                    },
                    "required": ["drug_name", "species"]
                }
            },
            {
                "name": "get_inventory_summary",
                "description": "调阅全院药品物资进销存实时台账，排查极度紧缺与临界断货物资（门诊与运营决策核心工具）。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "threshold": {"type": "integer", "description": "库存警戒水位上限（默认 10）", "default": 10}
                    }
                }
            },
            {
                "name": "triage_patient",
                "description": "门诊急症四级分诊（P0~P3）智能判定与红线识别（如：细小休克、完全尿闭、误食催吐禁忌）。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chief_complaint": {"type": "string", "description": "宠主主诉与异常表现"},
                        "species": {"type": "string", "description": "物种（犬 / 猫）"}
                    },
                    "required": ["chief_complaint"]
                }
            },
            {
                "name": "review_prescription",
                "description": "全方处方安全审方引擎：核查药物相互作用(DDI)、理化中和沉淀禁忌与黑框警示。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patient_species": {"type": "string", "description": "患宠物种（犬 / 猫）"},
                        "patient_weight_kg": {"type": "number", "description": "体重(kg)"},
                        "medications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "drug_name": {"type": "string"},
                                    "dose": {"type": "string"},
                                    "route": {"type": "string"}
                                },
                                "required": ["drug_name"]
                            },
                            "description": "拟开具处方药品清单"
                        }
                    },
                    "required": ["patient_species", "medications"]
                }
            }
        ]

    def get_resource_definitions(self) -> List[Dict[str, Any]]:
        """获取所有公开的 MCP Resources 清单。"""
        resources = [
            {
                "uri": "vetclaw://telemetry/summary",
                "name": "telemetry_summary",
                "description": "全院 AI 智能体运行遥测统计指标（TTFB、Token 消耗、Fast-Path 节省估算）",
                "mimeType": "application/json"
            }
        ]
        if KNOWLEDGE_DIR.exists():
            for p in sorted(KNOWLEDGE_DIR.glob("*.md")):
                uri = f"vetclaw://knowledge/{p.stem}"
                resources.append({
                    "uri": uri,
                    "name": p.stem,
                    "description": f"小动物临床循证专科知识库切片集合: {p.name}",
                    "mimeType": "text/markdown"
                })
        return resources

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """读取指定 URI 的 Resource 内容。"""
        if uri == "vetclaw://telemetry/summary":
            from telemetry import get_telemetry_summary
            summary = get_telemetry_summary()
            return {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(summary, ensure_ascii=False, indent=2)
            }

        prefix = "vetclaw://knowledge/"
        if not uri.startswith(prefix):
            raise ValueError(f"Unknown resource URI scheme: {uri}")
        slug = uri[len(prefix):]
        target = KNOWLEDGE_DIR / f"{slug}.md"
        if not target.exists():
            raise FileNotFoundError(f"Knowledge resource not found: {slug}")
        content = target.read_text(encoding="utf-8")
        return {
            "uri": uri,
            "mimeType": "text/markdown",
            "text": content
        }

    def get_prompt_definitions(self) -> List[Dict[str, Any]]:
        """获取标准 MCP Prompts 模板定义。"""
        return [
            {
                "name": "outpatient_preconsultation",
                "description": "门诊宠主预问诊与主诉危险信号分流系统提示词",
                "arguments": [
                    {"name": "species", "description": "犬或猫", "required": False},
                    {"name": "symptom", "description": "主诉症状", "required": True}
                ]
            },
            {
                "name": "soap_record_draft",
                "description": "执业兽医门诊标准 SOAP 病历草稿起草提示词",
                "arguments": [
                    {"name": "chief_complaint", "description": "主诉 (Subjective)", "required": True},
                    {"name": "exam_findings", "description": "体格与化验发现 (Objective)", "required": True}
                ]
            }
        ]

    def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """渲染指定 Prompt 模板。"""
        args = arguments or {}
        if name == "outpatient_preconsultation":
            symptom = args.get("symptom", "未提供")
            species = args.get("species", "宠物")
            return {
                "description": "门诊预问诊分流 Prompt",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"请针对{species}的症状「{symptom}」进行门诊预问诊初筛：排查呼吸窘迫、抽搐昏迷、完全尿闭等急症红线，提供家庭观察要点与就医科室引导，严禁开具处方药。"
                        }
                    }
                ]
            }
        elif name == "soap_record_draft":
            cc = args.get("chief_complaint", "")
            exam = args.get("exam_findings", "")
            return {
                "description": "SOAP 病历起草 Prompt",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"请依据门诊主诉 S: {cc} 与客观检查 O: {exam}，起草规范的 A (评估诊断) 与 P (循证治疗计划) SOAP 病历结构。"
                        }
                    }
                ]
            }
        raise ValueError(f"Unknown prompt: {name}")

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """分发执行具体的 MCP Tool。"""
        if name == "search_knowledge_base":
            query = arguments.get("query", "")
            top_k = int(arguments.get("top_k", 3))
            func = registry.get("search_knowledge_base")
            if func:
                return str(func(query=query, top_k=top_k))
            return "知识库检索组件未就绪"

        elif name == "check_drug_safety":
            drug_name = arguments.get("drug_name", "")
            species = arguments.get("species", "犬")
            weight = arguments.get("weight_kg")
            dose = arguments.get("dose_mg_per_kg") or arguments.get("daily_dose_mg")
            func = registry.get("check_drug_safety")
            if func:
                res = func(drug_name=drug_name, species=species, weight_kg=weight, dose_mg_per_kg=dose)
                return json.dumps(res, ensure_ascii=False, indent=2)
            return "用药安全审查工具未就绪"

        elif name == "get_inventory_summary":
            threshold = int(arguments.get("threshold", 10))
            func = registry.get("get_inventory_summary")
            if func:
                return str(func(threshold=threshold))
            return "物资库存调阅工具未就绪"

        elif name == "triage_patient":
            chief = arguments.get("chief_complaint", "")
            slots = extract_slots(chief)
            note = triage_note(chief, slots)
            return note or f"【分诊结果】主诉「{chief}」未触及即刻生命红线，建议按常规门诊 (P3 级) 引导至相应专科就诊。"

        elif name == "review_prescription":
            species = arguments.get("patient_species", "犬")
            meds = arguments.get("medications", [])
            weight = arguments.get("patient_weight_kg", 5.0)
            res = review_prescription(medications=meds, patient_species=species, patient_weight_kg=weight)
            return json.dumps(res, ensure_ascii=False, indent=2)

        raise ValueError(f"Unknown tool: {name}")

    def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """同步处理一条 JSON-RPC 2.0 请求。"""
        msg_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version
                    }
                }
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": self.get_tool_definitions()
                }
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                res_text = self.call_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": res_text}
                        ],
                        "isError": False
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": f"Tool execution failed: {str(e)}"}
                        ],
                        "isError": True
                    }
                }

        elif method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "resources": self.get_resource_definitions()
                }
            }

        elif method == "resources/read":
            uri = params.get("uri")
            try:
                resource = self.read_resource(uri)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "contents": [resource]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": str(e)
                    }
                }

        elif method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "prompts": self.get_prompt_definitions()
                }
            }

        elif method == "prompts/get":
            prompt_name = params.get("name")
            prompt_args = params.get("arguments", {})
            try:
                prompt_data = self.get_prompt(prompt_name, prompt_args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": prompt_data
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": str(e)
                    }
                }

        elif method == "notifications/initialized":
            # 客户端握手确认，无回包
            return None

        # 未知方法
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }

    async def run_stdio(self):
        """标准 stdio 运行循环：从 sys.stdin 读取每行 JSON-RPC 请求并回写 sys.stdout。"""
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                break
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            try:
                msg = json.loads(line_str)
                resp = self.handle_message(msg)
                if resp is not None:
                    out = json.dumps(resp, ensure_ascii=False)
                    sys.stdout.write(out + "\n")
                    sys.stdout.flush()
            except Exception as err:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(err)}"}
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


# 单例实例
mcp_server = VetClawMcpServer()

if __name__ == "__main__":
    asyncio.run(mcp_server.run_stdio())

"""ReAct 调度核心：统一事件生成器 _run_events 同时驱动 run() 与 run_stream()（零 LangChain）。"""
import json
import logging
import re
import time
from types import SimpleNamespace

from case_context import reset_context, set_context
from memory import estimate_tokens, message_tokens
from memory_l3 import render_memory_block
from resilience import call_llm

logger = logging.getLogger(__name__)

CITATION_MARK = "【来源："
HIT_MARK = "【知识库检索结果】"
MISS_MARK = "【检索未命中】"
DEGRADE_ANSWER = (
    "该情况不在我已收录的官方说明书与安全矩阵范围内，我不会凭空推测用药剂量。"
    "请先做以下现场排查：1) 观察宠物牙龈颜色（粉红/发白/发青）与静息呼吸频率；"
    "2) 核对药盒成分表是否含对乙酰氨基酚、木糖醇、葱属精油等已知犬猫剧毒成分；"
    "3) 记录误食时间与估计剂量。请携带原药盒与上述体征数据尽快前往宠物医院急诊。"
)
CITATION_RE = re.compile(r"【来源：[^】]+】")

DEFAULT_SYSTEM_PROMPT = """你是 VetClaw，一个宠物健康分诊助手。你可以查阅宠物安全矩阵、家庭急救与官方用药说明书知识库，回答常见症状、用药禁忌与护理建议。严格遵守：

1. 闲聊或自我介绍（如"你好""你能做什么"）：直接自然回答，不要调用工具。
2. 涉及宠物健康、症状判断、用药剂量、禁忌食物、急救处理的问题：必须先调用
   search_knowledge_base，再仅依据检索结果回答。
3. 引用：凡依据检索结果给出的事实，句末必须标注来源，格式为【来源：<chunk_id>】，
   chunk_id 形如 knowledge/species_drug_safety_matrix.md#对乙酰氨基酚-2；多条依据分别标注。
4. 防幻觉：只能使用检索片段中的信息，不得补充片段之外的用药剂量、成分或推测。
   检索为空或收到【检索未命中】时，不要只说"无法回答"，而要进入风险排查引导：
   明确告知该成分不在已收录文档中、不推测剂量，然后引导用户自查牙龈颜色/呼吸频率、
   核对药盒是否含对乙酰氨基酚/木糖醇等已知剧毒成分，并建议带原药盒就医。
5. 安全声明：每次回答末尾必须提醒"以上建议仅供参考，不能替代执业兽医诊断，紧急情况请立即就医"。
6. 档案上下文里给出「档案ID」且问题涉及该宠物的疫苗、既往用药或完整档案时，可调用
   get_pet_profile / get_vaccination_history / get_medication_history（pet_id 填该档案ID）。
   这些工具返回 JSON，先看 ok：ok=false 时如实说明查不到，不得编造档案内容。
7. 需要医院医生跟进的复杂或高风险情况，可调用 escalate_to_human（reason 必填，写明原因）。
   涉及具体药品说明书或单药安全核对时，可用 lookup_drug_label / check_drug_safety；
   工具结论只作参考，最终以执业兽医判断为准。
8. 档案上下文里的「长期记忆」必须带来源引用（医院档案 / 宠主确认 / AI 推断（待确认））；
   标注 AI 推断的只能作为待确认线索，绝不能当作过敏史、用药禁忌或剂量依据。

示例：
问：狗狗能吃巧克力吗？
答：巧克力对狗狗有毒，含可可碱会引起呕吐、心跳加速甚至抽搐，必须立即远离并就医【来源：knowledge/species_drug_safety_matrix.md#对乙酰氨基酚-2】。
以上建议仅供参考，不能替代执业兽医诊断，紧急情况请立即就医。
"""


PATIENT_CONTEXT_HEADER = "宠物档案上下文（仅作背景参考，不要主动复述全部内容）："


def render_patient_context(patient_context) -> str:
    """把档案字段渲染为单段文本；无有效字段时返回 ''。

    纯函数、**双引擎共用**（`core_agent` / `graph_agent`），保证注入文本零漂移。
    接受 dict 或含同名属性的对象（如 Pydantic 模型）。
    注意：本函数**不参与任何判定**，只在系统提示里追加背景信息——
    Fast-Path 急症拦截仍只看用户提问本身，不受档案影响。
    """
    if not patient_context:
        return ""

    def _get(key):
        if isinstance(patient_context, dict):
            return patient_context.get(key)
        return getattr(patient_context, key, None)

    parts = []
    if _get("species"):
        seg = f"物种: {_get('species')}"
        if _get("breed"):
            seg += f"; 品种: {_get('breed')}"
        parts.append(seg)
    if _get("age_years") is not None:
        parts.append(f"年龄: {_get('age_years')}岁")
    if _get("weight_kg") is not None:
        parts.append(f"体重: {_get('weight_kg')}kg")
    if _get("allergies"):
        parts.append(f"过敏史: {_get('allergies')}")
    if _get("latest_diagnosis"):
        parts.append(f"最近一次诊断: {_get('latest_diagnosis')}")
    if _get("vaccine_alert_count") is not None:
        parts.append(f"疫苗逾期/将到期提醒数: {_get('vaccine_alert_count')}")
    # 档案 ID 只在已有其它字段时追加：仅一个 ID 不构成有效背景，
    # 但它是医院数据工具（get_* / escalate_to_human）的入参来源。
    if parts and _get("pet_id") is not None:
        parts.append(f"档案ID: {_get('pet_id')}")

    memory_text = render_memory_block(_get("ai_memories"))
    if memory_text:
        if parts:
            return (f"{PATIENT_CONTEXT_HEADER}\n- " + "；".join(parts)
                    + "\n" + memory_text)
        return memory_text

    return f"{PATIENT_CONTEXT_HEADER}\n- " + "；".join(parts) if parts else ""


def merge_usage(usage, *, messages, text_parts, tool_parts, chunk_usage=None):
    """把一轮 LLM 调用的用量并入累计 dict（双引擎共用，保证口径一致）。

    真实 usage 优先；流式接口未返回 usage 时按 ``memory.estimate_tokens`` 估算，
    并置 ``estimated=True`` —— 轨迹里必须能区分真实与估算，避免把估算值当账单。
    """
    prompt_tokens = int(getattr(chunk_usage, "prompt_tokens", 0) or 0) if chunk_usage else 0
    completion_tokens = int(getattr(chunk_usage, "completion_tokens", 0) or 0) if chunk_usage else 0
    if prompt_tokens or completion_tokens:
        usage["input_tokens"] = usage.get("input_tokens", 0) + prompt_tokens
        usage["output_tokens"] = usage.get("output_tokens", 0) + completion_tokens
        return
    usage["estimated"] = True
    usage["input_tokens"] = usage.get("input_tokens", 0) + sum(
        message_tokens(m) for m in messages)
    output_tokens = estimate_tokens("".join(text_parts)) if text_parts else 0
    for slot in (tool_parts or {}).values():
        output_tokens += estimate_tokens(slot.get("name", ""))
        output_tokens += estimate_tokens(slot.get("arguments", ""))
    usage["output_tokens"] = usage.get("output_tokens", 0) + output_tokens


def count_tool_calls(usage, count):
    usage["tool_count"] = usage.get("tool_count", 0) + int(count or 0)


def chunk_usage_of(chunk):
    """取流式 chunk 的 usage（部分兼容端点会在末尾发一个 choices 为空的用量帧）。"""
    return getattr(chunk, "usage", None)


class ReActAgent:
    def __init__(self, client, registry, model="gpt-4o-mini",
                 max_steps=5, system_prompt=DEFAULT_SYSTEM_PROMPT,
                 clock=None, total_timeout=30.0, fastpath_guard=None,
                 max_tokens=None):
        self.client = client
        self.registry = registry
        self.model = model
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self._clock = clock or time.monotonic
        self.total_timeout = total_timeout
        self.guard = fastpath_guard
        # 输出长度上界（VETCLAW_MAX_TOKENS，默认 800）：防 token 滥用，测试注入 None 表示不传
        self.max_tokens = max_tokens

    def run(self, prompt, history=None, patient_context=None, case_id="",
            role="consult"):
        # 消费统一事件流：done 取 answer、error 取 message，与流式行为 100% 一致。
        for ev in self._run_events(prompt, history, patient_context, case_id, role):
            if ev["event"] == "done":
                return ev["data"]["answer"]
            if ev["event"] == "error":
                return ev["data"]["message"]
        return ""

    def run_stream(self, prompt, history=None, patient_context=None, case_id="",
                   role="consult", on_usage=None):
        usage = {}
        yield from self._run_events(prompt, history, patient_context, case_id, role,
                                    usage=usage)
        # 用量通过回调交给编排层写轨迹，**不新增事件**——6 种 SSE 事件契约保持原样。
        usage.setdefault("tool_count", 0)
        if on_usage is not None:
            try:
                on_usage(usage)
            except Exception:
                logger.exception("usage callback error")

    def _run_events(self, prompt, history=None, patient_context=None, case_id="",
                    role="consult", usage=None):
        cite_state = None
        start = self._clock()

        # Fast-Path 规则守卫：P0 急症与未知处方药在入口秒级拦截，不调 LLM
        if self.guard is not None:
            hit = self.guard.check(prompt, patient_context)
            if hit["action"] == "emergency":
                msg = self.guard.emergency_message(
                    hit["toxin"], hit["species"], hit.get("chunk_id"),
                    hit.get("slots"))
                yield {"event": "done", "data": {"answer": msg, "citations": []}}
                return
            if hit["action"] == "refuse":
                msg = self.guard.refuse_message(hit["drug_hint"])
                yield {"event": "done", "data": {"answer": msg, "citations": []}}
                return

        # 档案上下文：仅追加到 system 提示，不参与任何判定（急症拦截仍只看 prompt）
        system_content = self.system_prompt
        ctx_text = render_patient_context(patient_context)
        if ctx_text:
            system_content = f"{system_content}\n\n{ctx_text}"
        messages = [{"role": "system", "content": system_content}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        yield {"event": "thought", "data": {"stage": "thinking"}}

        for _ in range(self.max_steps):
            if self._clock() - start > self.total_timeout:
                yield {"event": "error", "data": {"code": "MODEL_TIMEOUT",
                       "message": "模型响应超时，请稍后重试"}}
                return
            tool_parts, text_parts = {}, []
            turn_usage = None
            stream = None
            try:
                # 韧性：退避重试 + 熔断（与 graph 引擎共用同一实现，保证零漂移）
                llm_kwargs = dict(model=self.model, messages=messages,
                                  tools=self.registry.schemas(), stream=True)
                if self.max_tokens is not None:
                    llm_kwargs["max_tokens"] = self.max_tokens
                stream = call_llm(self.client, **llm_kwargs)
                for chunk in stream:
                    if self._clock() - start > self.total_timeout:
                        yield {"event": "error", "data": {"code": "MODEL_TIMEOUT",
                               "message": "模型响应超时，请稍后重试"}}
                        return
                    chunk_usage = chunk_usage_of(chunk)
                    if chunk_usage is not None:
                        turn_usage = chunk_usage
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue      # 仅含 usage 的末尾帧
                    delta = choices[0].delta
                    for p in (getattr(delta, "tool_calls", None) or []):
                        idx = getattr(p, "index", 0) or 0
                        slot = tool_parts.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if getattr(p, "id", None):
                            slot["id"] = p.id
                        fn = getattr(p, "function", None)
                        if fn:
                            if getattr(fn, "name", None) and not slot["name"]:
                                slot["name"] = fn.name  # 仅初次非空赋值，严禁 += 累加 name
                            if getattr(fn, "arguments", None):
                                slot["arguments"] += fn.arguments
                    piece = getattr(delta, "content", None)
                    if piece:
                        text_parts.append(piece)
                        yield {"event": "token", "data": {"text": piece}}
            except Exception as e:
                # SDK 层（httpx read timeout / 连接错误）兜底转 SSE error，不让 traceback 冒到 HTTP 层
                logger.exception("model stream error")
                code = "MODEL_TIMEOUT" if "timeout" in type(e).__name__.lower() else "INTERNAL_ERROR"
                yield {"event": "error", "data": {"code": code,
                       "message": "模型服务暂时不可用，请稍后重试"}}
                return
            finally:
                if stream is not None and hasattr(stream, "close"):
                    try:
                        stream.close()
                    except Exception:
                        pass

            if usage is not None:
                merge_usage(usage, messages=messages, text_parts=text_parts,
                            tool_parts=tool_parts, chunk_usage=turn_usage)
                if tool_parts:
                    count_tool_calls(usage, len(tool_parts))

            if tool_parts:
                calls = [{"id": v["id"], "type": "function",
                          "function": {"name": v["name"], "arguments": v["arguments"]}}
                         for _, v in sorted(tool_parts.items())]
                messages.append({"role": "assistant", "content": None, "tool_calls": calls})
                for c in calls:
                    yield {"event": "tool_call", "data": {"id": c["id"],
                           "name": c["function"]["name"],
                           "arguments": _safe_args(c["function"]["arguments"])}}
                    yield {"event": "thought", "data": {"stage": "executing_tool"}}
                    tc = SimpleNamespace(id=c["id"], function=SimpleNamespace(
                        name=c["function"]["name"], arguments=c["function"]["arguments"]))
                    content = self._run_tool(tc, case_id, role)
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
                    yield {"event": "tool_result", "data": {"id": c["id"], "content": content}}
                    if content.startswith(HIT_MARK):
                        cite_state = "hit"  # hit 优先级最高，不被后续工具/miss 覆盖
                    elif content.startswith(MISS_MARK) and cite_state != "hit":
                        cite_state = "miss"
                yield {"event": "thought", "data": {"stage": "thinking"}}
                continue

            final = "".join(text_parts)
            fix = self._compliance_fix(cite_state, final)
            if fix:
                messages.append({"role": "user", "content": fix})
                yield {"event": "thought", "data": {"stage": "thinking"}}
                continue
            yield {"event": "done", "data": {
                "answer": final, "citations": CITATION_RE.findall(final)}}
            return

        message = DEGRADE_ANSWER if cite_state in ("hit", "miss") \
            else f"已达到最大步数 {self.max_steps}，强制停止。"
        code = "DEGRADED" if cite_state in ("hit", "miss") else "MAX_STEPS"
        yield {"event": "error", "data": {"code": code, "message": message}}
        return

    def _compliance_fix(self, cite_state, final):
        # 委托模块级纯函数：LangGraph 引擎（graph_agent.py）复用同一份逻辑，
        # 保证两引擎行为零漂移（阶段四「事件序列 diff 为空」的前提）。
        return compliance_fix(cite_state, final)

    def _run_tool(self, tc, case_id="", role="consult"):
        return execute_tool(self.registry, tc, case_id, role)


def compliance_fix(cite_state, final):
    """合规校验：返回补正指令（无则 None）。纯函数，不碰 writer，双引擎共用。"""
    if cite_state == "hit" and CITATION_MARK not in final:
        return (f"【系统校验】回答缺少{CITATION_MARK}...】来源标注，"
                "请仅基于上述检索结果重写，并在每条事实句末标注来源。")
    if cite_state == "miss" and "不会凭空推测用药剂量" not in final:
        return (f"【系统校验】检索未命中，请直接回复：{DEGRADE_ANSWER}"
                " 不要用通用知识编造。")
    return None


def execute_tool(registry, tc, case_id="", role="consult"):
    """执行一次工具调用并返回字符串结果。纯函数，双引擎共用。

    工具白名单由 ``RoleScopedRegistry`` 在注册表边界拒绝（模型看不到越权工具，
    真调用也会抛 ``PermissionError``）；这里只负责把异常转成模型可读的失败文本。
    ``case_id`` / ``role`` 在**本次调用的同步栈内**写入上下文，供转人工等写工具
    把工单关联回病例；不能用模块级全局，两个流并发时会串号。
    """
    name = tc.function.name
    raw = tc.function.arguments
    args = {} if not raw else None
    if args is None:
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as e:
            return f"参数解析失败: {e}"
    if name.startswith("danger_"):
        return "工具执行失败: PermissionDenied: 该工具为高危操作，已触发安全拦截"
    role = getattr(registry, "role", role) or role
    token = set_context(case_id, role)
    try:
        return str(registry.call(name, args))
    except Exception as e:
        return f"工具执行失败: {type(e).__name__}: {e}"
    finally:
        reset_context(token)


def _safe_args(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}

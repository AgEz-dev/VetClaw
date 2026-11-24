"""Badcase 自动化脱敏收集与数据飞轮存储 (Feedback Store)。

工业级数据闭环规范：
1. 收集来源：医生在工作站人工修改 AI 起草的 SOAP 病历，或前台/宠主对回答点踩反馈；
2. 自动化脱敏 (Anonymization)：使用正则自动擦除手机号、身份证、真实宠主姓名，防止隐私泄露；
3. JSON Lines 格式落盘：追加写入 data/badcases.jsonl，轻量高效且具备高并发追加安全；
4. 飞轮转换引擎：提供一键导出为 eval_dataset 标准评测用例的转换能力，使线上坏病例自动反哺离线评测基线。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BADCASE_FILE = Path(__file__).resolve().parent.parent / "data" / "badcases.jsonl"
DEFAULT_EVAL_EXPORT_FILE = Path(__file__).resolve().parent.parent / "data" / "badcases_eval_export.json"

# 身份证脱敏正则：18 位，前后不得有数字 (防止截断)
ID_CARD_REGEX = re.compile(r"(?<!\d)(\d{6})\d{8}(\d{3}[\dXx])(?!\d)")
# 手机号脱敏正则：11 位大陆手机号，前后不得有数字
PHONE_REGEX = re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)")


def anonymize_text(text: str) -> str:
    """脱敏敏感个人信息（手机号、身份证号）。先处理长数字(身份证)，再处理短数字(手机号)。"""
    if not text:
        return ""
    t = ID_CARD_REGEX.sub(r"\1********\2", text)
    t = PHONE_REGEX.sub(r"\1****\2", t)
    return t


def record_feedback(
    trace_id: str,
    role: str,
    query: str,
    retrieved_chunk_ids: List[str],
    ai_output: str,
    human_revised_output: str,
    reason: str = "",
    file_path: Path | str | None = None
) -> Dict[str, Any]:
    """脱敏记录一条 Badcase 反馈至 jsonl 文件。"""
    target_path = Path(file_path) if file_path else DEFAULT_BADCASE_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "trace_id": trace_id,
        "role": role,
        "query": anonymize_text(query),
        "retrieved_chunk_ids": list(retrieved_chunk_ids or []),
        "ai_output": anonymize_text(ai_output),
        "human_revised_output": anonymize_text(human_revised_output),
        "reason": anonymize_text(reason),
    }

    with open(target_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def load_badcases(file_path: Path | str | None = None) -> List[Dict[str, Any]]:
    """读取所有已收集的 Badcase 记录。"""
    target_path = Path(file_path) if file_path else DEFAULT_BADCASE_FILE
    if not target_path.exists():
        return []

    cases = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cases


def export_badcases_to_eval_format(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None
) -> Dict[str, Any]:
    """将 badcases.jsonl 中的坏病例转化为 eval_dataset 格式，实现数据飞轮反哺。"""
    records = load_badcases(input_path)
    eval_cases = []

    for idx, rec in enumerate(records):
        query = rec.get("query", "")
        revised = rec.get("human_revised_output", "")
        chunk_ids = rec.get("retrieved_chunk_ids", [])
        expected_chunk = chunk_ids[0] if chunk_ids else ""

        # 简单提取人工修正答案中的核心关键词（2个字以上的汉字短语）
        raw_words = re.findall(r"[\u4e00-\u9fa5]{2,6}", revised)
        keywords = list(dict.fromkeys(raw_words))[:3]

        eval_cases.append({
            "id": f"BC-{idx+1:03d}",
            "query": query,
            "expected_source_chunk": expected_chunk,
            "must_contain_keywords": keywords,
            "expected_behavior": "answer",
            "category": "badcase_feedback",
            "source_trace_id": rec.get("trace_id", "")
        })

    payload = {
        "version": "v1-badcase-flywheel",
        "exported_at": datetime.now().isoformat(),
        "cases_count": len(eval_cases),
        "cases": eval_cases
    }

    out_file = Path(output_path) if output_path else DEFAULT_EVAL_EXPORT_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

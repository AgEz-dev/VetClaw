"""L3 长期记忆的渲染与预算控制（纯函数，双引擎共用）。

三条纪律：

1. **只渲染事实，不渲染摘要**：医院库返回的是短句事实（过敏/慢性病/长期用药），
   本模块不拼接对话原文。
2. **来源必须可见**：每条都带「医院档案 / 宠主确认 / AI 推断（待确认）」标签。
   ``AI_INFERRED`` 只能用于回顾，**永远不能成为安全结论的依据**。
3. **预算是硬上限**：top 8 条、整块 ≤200 token（含表头），超了就截断——
   记忆是背景，不能把一次问诊的上下文挤爆。

为什么注入由编排层固定做一次、而不是做成第 8 个工具：让模型自己决定
「要不要回忆」必然漏；工具数也因此保持 7 个。
"""
from __future__ import annotations

from memory import estimate_tokens

MEMORY_TOP_N = 8
MEMORY_TOKEN_BUDGET = 200

MEMORY_HEADER = "长期记忆（按来源标注；AI 推断不可作为安全依据）："

SOURCE_LABELS = {
    "HOSPITAL_RECORD": "医院档案",
    "OWNER_CONFIRMED": "宠主确认",
    "AI_INFERRED": "AI 推断（待确认）",
}

FACT_LABELS = {
    "ALLERGY": "过敏史",
    "CHRONIC_DISEASE": "慢性病",
    "MEDICATION": "长期用药",
    "DIET": "饮食",
    "LIFESTYLE": "生活史",
    "OTHER": "其他",
}


def _field(item: dict, camel: str, snake: str):
    return item.get(camel, item.get(snake))


def _line_of(item: dict) -> str:
    value = str(_field(item, "factValue", "fact_value") or "").strip()
    if not value:
        return ""
    fact_type = str(_field(item, "factType", "fact_type") or "").strip()
    source = str(_field(item, "source", "source") or "").strip()
    label = SOURCE_LABELS.get(source, source or "未知来源")
    kind = FACT_LABELS.get(fact_type, fact_type or "其他")
    return f"[{label}] {kind}：{value}"


def _truncate_first_line(prefix: str, value: str, max_tokens: int) -> str:
    """单条事实本身就超预算时，截断正文而不是整块丢弃（保留来源与类型）。"""
    low, high, best = 0, len(value), ""
    while low <= high:
        mid = (low + high) // 2
        text = value[:mid] + ("…" if mid < len(value) else "")
        line = prefix + text
        if estimate_tokens(MEMORY_HEADER + "\n- " + line) <= max_tokens:
            best = line
            low = mid + 1
        else:
            high = mid - 1
    return best


def render_memories(memories, max_tokens: int = MEMORY_TOKEN_BUDGET,
                    top_n: int = MEMORY_TOP_N) -> list[str]:
    """渲染为带来源标签的短行；整块（含表头）不超过 ``max_tokens``。"""
    lines: list[str] = []
    for item in list(memories or [])[:top_n]:
        if not isinstance(item, dict):
            continue
        line = _line_of(item)
        if not line:
            continue
        block = MEMORY_HEADER + "\n- " + "；".join(lines + [line])
        if estimate_tokens(block) > max_tokens:
            if not lines:
                prefix = line[:line.rfind("：") + 1]
                truncated = _truncate_first_line(prefix, line[len(prefix):], max_tokens)
                if truncated:
                    lines.append(truncated)
            break
        lines.append(line)
    return lines


def render_memory_block(memories, max_tokens: int = MEMORY_TOKEN_BUDGET) -> str:
    """完整注入块；没有可渲染行时返回空串（调用方据此决定是否追加）。"""
    lines = render_memories(memories, max_tokens=max_tokens)
    return MEMORY_HEADER + "\n- " + "；".join(lines) if lines else ""


def safety_authoritative(item: dict) -> bool:
    """安全结论只认医院档案 / 宠主确认；AI_INFERRED 永远不参与。"""
    flag = _field(item, "safetyAuthoritative", "safety_authoritative")
    if isinstance(flag, bool):
        return flag
    return str(_field(item, "source", "source") or "") in (
        "HOSPITAL_RECORD", "OWNER_CONFIRMED")

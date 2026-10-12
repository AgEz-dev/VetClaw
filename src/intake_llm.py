"""预问诊的 LLM 通道：提示词构造 + **严格解析**。

分工（与 ``intake.py`` 规则通道配合）：

- **规则通道**负责"不漏"：选项、频次、时间这类结构化信息，规则比模型更可靠、可测、零成本。
- **LLM 通道**负责"抓得住"：自由文本槽位（主诉表述 / 诊疗经过 / 一般情况 / 宠主补充）
  规则抓不住 —— 宠主说"它从昨晚开始一直吐，三四次吧，精神也没以前好"，
  一句里就藏着 3–4 个槽位。

**本模块只做提示词构造与响应解析**（纯函数、可离线单测）；真正的模型调用在 Agent 层完成，
便于注入与 mock —— **单元测试不依赖网络**。

解析必须**严格**：模型会幻觉出不存在的槽位名、或给出不在选项内的值，
一律丢弃而不是"尽量接受" —— 预问诊的数据要进医生的眼睛，**宁缺勿错**。
"""
from __future__ import annotations

import json
import re

VALID_STATES = ("filled", "vague")

# 诊断类措辞：AI 提示只描述"待排查方向"，出现诊断结论一律丢弃（红线）
_DIAGNOSIS_PATTERNS = ("确诊", "诊断为", "患有", "判断为", "可确定是", "肯定是")


# --------------------------------------------------------------------- 槽位抽取

def build_extract_messages(conversation_text: str, template: dict,
                           focus_keys: list[str] | None = None) -> list[dict]:
    """构造「从对话抽取槽位」的提示词。"""
    slots = list(template.get("base_slots", []))
    for branch in template.get("branches", {}).values():
        slots += list(branch.get("slots", []))
    slots += list(template.get("archive_slots", []))

    lines = []
    for s in slots:
        if focus_keys and s["key"] not in focus_keys:
            continue
        desc = "- %s（%s）" % (s["key"], s.get("name", s["key"]))
        if s.get("options"):
            desc += " 可选值：" + " / ".join(s["options"])
        elif s.get("free_text"):
            desc += " 自由文本，用宠主原话，不要自己组织语言"
        lines.append(desc)

    system = (
        "你是宠物门诊的预问诊信息抽取器。只做一件事：从对话里抽取下列槽位的值。\n"
        "规则：\n"
        "1. 只输出 JSON，不要任何解释文字。格式：{\"槽位名\": {\"state\": \"filled|vague\", \"value\": \"...\"}}\n"
        "2. `filled` = 明确说了；`vague` = 说了但含糊（如「吐了好几次」没说次数）。\n"
        "3. 有可选值的槽位**只能**填可选值里的一个；对不上就别输出这个槽位。\n"
        "4. 自由文本槽位**照抄宠主原话**，不要改写、不要总结、不要润色。\n"
        "5. 没提到的槽位**不要输出**，不要猜、不要编。\n"
        "6. 不要给任何诊断、判断或建议 —— 你只做信息抄录。\n\n"
        "槽位清单：\n" + "\n".join(lines)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "对话记录：\n" + conversation_text.strip()},
    ]


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return (match.group(1) if match else text).strip()


def parse_extract_response(raw: str, template: dict) -> dict:
    """**严格**解析模型输出：只接受模板内存在的槽位与合法取值，其余一律丢弃。

    :returns: ``{key: {"state", "value", "source": "llm"}}``
    """
    try:
        data = json.loads(_strip_code_fence(raw))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    specs = {}
    for s in (list(template.get("base_slots", [])) + list(template.get("archive_slots", []))):
        specs[s["key"]] = s
    for branch in template.get("branches", {}).values():
        for s in branch.get("slots", []):
            specs[s["key"]] = s

    out: dict = {}
    for key, payload in data.items():
        spec = specs.get(key)
        if spec is None:                       # 幻觉出的槽位 → 丢弃
            continue
        if not isinstance(payload, dict):
            continue
        state = str(payload.get("state", "")).strip()
        value = str(payload.get("value", "")).strip()
        if state not in VALID_STATES or not value:
            continue
        options = spec.get("options")
        if options:
            # 选项型槽位只认精确命中（模糊匹配交给规则通道的同义词表，模型不得自创）
            if value not in options:
                continue
            state = "filled"
        out[key] = {"state": state, "value": value, "source": "llm"}
    return out


def slots_without_options(template: dict) -> list[str]:
    """**LLM 通道的适用范围**：无 ``options`` 的槽位。

    分工原则（实测教训）：**选项型槽位归规则通道，无选项型归 LLM 通道**。
    实测：把选项型槽位也交给模型，它会"把空填满"—— 对话只说"昨天开始的"（没说急缓），
    模型却填了"突然出现"；只说"禁食了没好转"，模型推断成"越来越重"。
    规则通道在选项上更可靠、可测、零成本；LLM 的价值在于**抓自由文本**（主诉 / 诊疗经过 /
    一般情况 / 宠主补充）。
    """
    out = []
    for s in (list(template.get("base_slots", [])) + list(template.get("archive_slots", []))):
        if not s.get("options"):
            out.append(s["key"])
    for branch in template.get("branches", {}).values():
        for s in branch.get("slots", []):
            if not s.get("options"):
                out.append(s["key"])
    return out


def merge_llm_slots(state: dict, llm_slots: dict, only_missing: bool = True) -> dict:
    """把 LLM 抽到的槽位并入状态。

    :param only_missing: **规则通道优先** —— 默认只填补 ``missing`` 的槽位，
        不覆盖规则通道已确定的值（规则在结构化信息上更精确）。
    """
    for key, value in (llm_slots or {}).items():
        if only_missing and state.get(key, {}).get("state", "missing") != "missing":
            continue
        state[key] = value
    return state


# --------------------------------------------------------------------- 摘要生成

_SUMMARY_KEYS = ("chief_complaint", "present_illness", "past_history", "general",
                 "other_findings", "ai_hints", "suggested_questions")


def build_summary_messages(conversation_text: str, slots_status: list[dict]) -> list[dict]:
    """构造「生成医生端结构化摘要」的提示词（SOAP 的 S/O 骨架）。"""
    filled = [row for row in slots_status if row.get("state") != "missing"]
    slot_lines = "\n".join(
        "- %s（%s，来源=%s）：%s" % (r["key"], r.get("name", ""), r.get("source", ""), r.get("value", ""))
        for r in filled
    ) or "（无）"

    system = (
        "你在为宠物医生生成**预问诊摘要**。输出 JSON，字段固定为：\n"
        "{\n"
        '  "chief_complaint": "主诉：主要症状 + 持续时间，一句话",\n'
        '  "present_illness": {"onset": "", "characteristics": "", "progression": "", "associated": "", "treatment": ""},\n'
        '  "past_history": {"chronic": "", "surgery": "", "vaccination": "", "allergy": ""},\n'
        '  "general": {"appetite": "", "water": "", "weight_change": ""},\n'
        '  "other_findings": [{"text": "", "source": "free_text"}],\n'
        '  "ai_hints": [{"text": "", "citations": []}],\n'
        '  "suggested_questions": []\n'
        "}\n"
        "红线（务必遵守）：\n"
        "1. **不得给诊断结论**（不许出现「确诊」「诊断为」「患有」这类判断）；\n"
        "   `ai_hints` 只能写「待排查方向」，如「建议先排除误食与异物」。\n"
        "2. **不得编造**：槽位里没有的信息一律留空字符串或空数组。\n"
        "3. 宠主补充的原话**原样**放进 other_findings 的 text 或对应字段，不许改写。\n"
        "4. `suggested_questions` 写「现场该问但没问到的」，没有就空数组。\n"
        "5. 只输出 JSON，不要解释文字。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content":
            "已采集的槽位：\n" + slot_lines + "\n\n对话原文：\n" + conversation_text.strip()},
    ]


def parse_summary_response(raw: str) -> dict:
    """解析摘要；字段缺失补空，**含诊断结论的 ai_hints 一律丢弃**（红线）。"""
    empty = {
        "chief_complaint": "",
        "present_illness": {k: "" for k in ("onset", "characteristics", "progression",
                                            "associated", "treatment")},
        "past_history": {k: "" for k in ("chronic", "surgery", "vaccination", "allergy")},
        "general": {k: "" for k in ("appetite", "water", "weight_change")},
        "other_findings": [],
        "ai_hints": [],
        "suggested_questions": [],
    }
    try:
        data = json.loads(_strip_code_fence(raw))
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty

    out = dict(empty)
    if isinstance(data.get("chief_complaint"), str):
        out["chief_complaint"] = data["chief_complaint"].strip()
    for group, keys in (("present_illness", empty["present_illness"]),
                        ("past_history", empty["past_history"]),
                        ("general", empty["general"])):
        section = data.get(group)
        if isinstance(section, dict):
            out[group] = {k: str(section.get(k, "") or "").strip() for k in keys}

    findings = data.get("other_findings")
    if isinstance(findings, list):
        for item in findings:
            if isinstance(item, dict) and str(item.get("text", "")).strip():
                out["other_findings"].append({
                    "text": str(item["text"]).strip(),
                    "source": str(item.get("source", "free_text")),
                })

    hints = data.get("ai_hints")
    if isinstance(hints, list):
        for item in hints:
            text = str((item or {}).get("text", "")).strip() if isinstance(item, dict) else ""
            if not text:
                continue
            if any(p in text for p in _DIAGNOSIS_PATTERNS):   # 🔴 红线：诊断结论一律丢弃
                continue
            cites = item.get("citations") if isinstance(item, dict) else None
            out["ai_hints"].append({
                "text": text,
                "citations": [str(c) for c in cites] if isinstance(cites, list) else [],
            })

    questions = data.get("suggested_questions")
    if isinstance(questions, list):
        out["suggested_questions"] = [str(q).strip() for q in questions if str(q).strip()]

    return out


def contains_diagnosis(text: str) -> bool:
    """供上层做输出护栏复用（医生端渲染前也可再查一次）。"""
    return any(p in (text or "") for p in _DIAGNOSIS_PATTERNS)

"""病历起草的 LLM 通道：提示词构造 + **严格解析**（问诊辅助 · 第二批）。

定位：AI 只起草**叙述性内容**（SOAP 的 S 主诉 / O 建议查体清单 / P 医嘱草稿），
**A（确诊）永不输出** —— 本模块从两个层面钉死这条红线：

1. **schema 层**：输出的字段集合里**没有 diagnosis**，解析时未知键一律丢弃，
   模型想越界也没有落点；
2. **内容层**：任何字段文本命中诊断措辞（复用 ``intake_llm`` 的措辞表）一律丢弃。

与 ``intake_llm`` 同一条纪律：**本模块只做提示词构造与响应解析**（纯函数、可离线
单测），真正的模型调用由服务层注入 —— 单元测试不依赖网络。
"""
from __future__ import annotations

import json
import re

from intake_llm import _DIAGNOSIS_PATTERNS, _strip_code_fence

# 数据库列宽（pet_medical_record）：超长会直接让落库报错，必须在这里截断
_CHIEF_MAX = 255          # chief_complaint varchar(255)
_ADVICE_MAX = 500         # doctor_advice varchar(500)
_CHECKLIST_MAX_ITEMS = 8  # 建议查体清单条数上限
_CHECKLIST_ITEM_MAX = 80  # 单条上限

_EMERGENCY_WORDS = ("中毒", "误食", "吞食", "巧克力", "木糖醇", "百合", "葡萄",
                    "洋葱", "药物", "老鼠药")

_DISCLAIMER = "本草稿由 AI 根据预问诊内容生成，仅供医生参考，不构成诊断；出具以医生核对为准。"
_DISCLAIMER_URGENT = _DISCLAIMER + "宠主描述疑似涉及误食/中毒等急症情形，请优先按急症流程处置。"


# --------------------------------------------------------------------- 提示词构造

def build_draft_messages(summary: dict, dialogue: list[dict],
                         archive: dict | None, pet: dict | None) -> list[dict]:
    """构造「从预问诊成果起草病历草稿」的提示词。

    :param summary: 预问诊结构化摘要（契约 v1.2 §3.4 的 ``intake``）
    :param dialogue: 对话原文（``[{"role", "text"}]``）
    :param archive: 档案槽位（``vaccine`` / ``past_disease`` / ``medications`` / ``allergy``）
    :param pet: 宠物信号（物种 / 品种 / 年龄 / 体重 / 过敏），只有已知的才传
    """
    def _fmt_summary() -> str:
        lines = []
        if summary.get("chief_complaint"):
            lines.append("主诉：" + str(summary["chief_complaint"]))
        for group in ("present_illness", "past_history", "general"):
            section = summary.get(group)
            if isinstance(section, dict):
                pairs = "；".join("%s=%s" % (k, v) for k, v in section.items() if v)
                if pairs:
                    lines.append("%s：%s" % (group, pairs))
        for f in summary.get("other_findings") or []:
            if isinstance(f, dict) and f.get("text"):
                lines.append("宠主补充（原话）：" + str(f["text"]))
        hints = summary.get("ai_hints") or []
        if hints:
            lines.append("预问诊阶段已提示的待排查方向：" +
                         "；".join(str(h.get("text", "")) for h in hints if isinstance(h, dict)))
        return "\n".join(lines) or "（无）"

    dialogue_text = "\n".join(
        "%s：%s" % (d.get("role", "user"), d.get("text", ""))
        for d in dialogue if isinstance(d, dict) and d.get("text")) or "（无）"
    archive_text = "；".join("%s=%s" % (k, v) for k, v in (archive or {}).items()) or "（无）"
    pet_text = "；".join("%s=%s" % (k, v) for k, v in (pet or {}).items() if v) or "（无）"

    system = (
        "你在为宠物医生起草电子病历的**叙述性内容**（SOAP 中的 S 与 P 的草稿部分）。\n"
        "输出 JSON，字段固定为：\n"
        "{\n"
        '  "chief_complaint": "主诉：主要症状 + 持续时间，医学叙述，一句话",\n'
        '  "exam_checklist": ["建议到院核查的查体项目", "..."],\n'
        '  "doctor_advice": "医嘱草稿：家庭护理与复诊建议",\n'
        '  "disclaimer": "固定免责声明，原样照抄即可"\n'
        "}\n"
        "红线（务必遵守）：\n"
        "1. **不得给诊断、确诊、鉴别结论** —— 确诊是兽医的执业行为，AI 永远不做；\n"
        "   `exam_checklist` 只写「建议核查的查体项目」（如「评估脱水程度」），不是查体结果。\n"
        "2. **不得给处方、药品剂量** —— 用药由医生决定，草稿里不出现任何具体药物剂量。\n"
        "3. **不得编造**：对话与摘要里没有的信息一律留空，不要推测体重、体温等数值。\n"
        "4. 主诉以宠主描述为事实基础，做医学化的语言整理，不添油加醋。\n"
        "5. 只输出 JSON，不要解释文字。"
    )
    user = (
        "宠物：\n" + pet_text +
        "\n\n档案：\n" + archive_text +
        "\n\n预问诊摘要：\n" + _fmt_summary() +
        "\n\n对话原文：\n" + dialogue_text
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------- 严格解析

def _clean_text(value, limit: int) -> str:
    """压成单行空白并截断；非字符串一律空串。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def parse_draft_response(raw: str) -> dict:
    """**严格**解析草稿输出：未知键一律丢弃、诊断措辞一律丢弃、按列宽截断。

    :returns: ``{"chief_complaint", "exam_checklist", "doctor_advice", "disclaimer"}``
        解析失败返回全空草稿（空主诉由调用方判定为"本次起草失败"）。
    """
    empty = {"chief_complaint": "", "exam_checklist": [], "doctor_advice": "",
             "disclaimer": _DISCLAIMER}
    try:
        data = json.loads(_strip_code_fence(raw))
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty

    out = dict(empty)
    # 🔴 红线第 1 层：无论模型输出什么键，白名单之外的一律丢弃（diagnosis 在此被丢弃）
    if isinstance(data.get("chief_complaint"), str):
        out["chief_complaint"] = _clean_text(data["chief_complaint"], _CHIEF_MAX)
    if isinstance(data.get("doctor_advice"), str):
        advice = _clean_text(data["doctor_advice"], _ADVICE_MAX)
        # 🔴 红线第 2 层：内容层兜底 —— 命中诊断措辞的文本整段丢弃
        out["doctor_advice"] = "" if any(p in advice for p in _DIAGNOSIS_PATTERNS) else advice

    items = data.get("exam_checklist")
    if isinstance(items, list):
        for item in items:
            text = _clean_text(item, _CHECKLIST_ITEM_MAX)
            if not text or any(p in text for p in _DIAGNOSIS_PATTERNS):
                continue
            if len(out["exam_checklist"]) >= _CHECKLIST_MAX_ITEMS:
                break
            out["exam_checklist"].append(text)

    return out


def disclaimer_for(dialogue_text: str) -> str:
    """按对话内容给出免责声明：命中急症词 → 加强版（先按急症流程处置）。"""
    return _DISCLAIMER_URGENT if any(w in (dialogue_text or "") for w in _EMERGENCY_WORDS) \
        else _DISCLAIMER

"""预问诊槽位引擎：决定「问什么」。

设计要点（对齐跨服务契约 v1.2 §3.4 与《预问诊-问诊项与话术规范-v0.1》）：

- **槽位三态**：``filled`` / ``vague`` / ``missing`` —— **含糊也是缺口**。
  漏问多数不是"没问"，而是"问了没问清"，因此 ``vague`` 会被继续追问。
- **追问按影响度排序**（P0 安全 > P1 主诉定性 > P2 鉴别 > P3 补充），**不按模板顺序**；
  这样即使宠主中途不聊了，取到的也是最有价值的信息。
- **每轮最多 N 问**（默认取模板 ``meta.rules.max_questions_per_turn``），避免连环炮。
- **档案优先**：可由业务档案带出的槽位（``archive_slots``）**有值就不问**。
- **分支激活**：按主诉关键词激活分支槽位（如 ``vomiting``），通用清单问不细、分支才问得深。
- **收尾**：``required`` 全部非 ``missing``，或达轮数上限；后者把未采集项如实列入
  ``incomplete``（**不得空占位**）。

本模块**纯函数优先、零 LLM、零网络、零 I/O 副作用**（模板由调用方传入或显式加载），
可离线单测。语义抽槽（LLM 通道）与摘要生成在 ``intake_summary`` / Agent 中完成。
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "rules" / "intake_slots.json"

_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

# 「48h 摄入与接触史」的阳性/阴性词。
# ⚠️ 只用**具体**毒物/异物名，不用"药"这类笼统词 —— 实测事故："没给药"（回答"诊疗经过"
# 问题时）会被读成"疑似摄入药"。笼统词歧义大，宁缺勿滥。
_POSITIVE_INTAKE = (
    "巧克力", "骨头", "垃圾", "植物百合", "百合", "绿萝", "洋葱", "大蒜", "葡萄", "老鼠药",
    "布洛芬", "对乙酰氨基酚", "扑热息痛", "安眠药", "玩具", "袜子", "塑料袋",
    "新零食", "换粮", "变质",
)
_NEGATIVE_WORDS = ("没有", "没乱", "没吃", "未吃", "无", "都正常", "正常吃饭", "没给药", "没喂")


def load_template(path: str | Path | None = None) -> dict:
    """加载槽位模板（热更新：改 JSON 即生效，无需改代码）。"""
    return json.loads(Path(path or DEFAULT_TEMPLATE).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 槽位集合

def activate_branches(template: dict, conversation_text: str) -> list[str]:
    """按关键词激活分支；命中多个则全部激活（宠主可能同时说了不止一个症状）。"""
    text = conversation_text or ""
    return [
        name for name, branch in template.get("branches", {}).items()
        if any(k in text for k in branch.get("trigger_keywords", []))
    ]


def active_slots(template: dict, conversation_text: str = "") -> list[dict]:
    """当前生效的槽位 = base + 已激活分支 + archive。"""
    slots = list(template.get("base_slots", []))
    for name in activate_branches(template, conversation_text):
        slots += list(template["branches"][name].get("slots", []))
    slots += list(template.get("archive_slots", []))
    return slots


def empty_state(template: dict, conversation_text: str = "") -> dict:
    """初始槽位状态（全部 missing；archive 槽位由 ``merge_archive`` 填充）。"""
    return {
        s["key"]: {"state": "missing", "value": "", "source": s.get("source", "dialog")}
        for s in active_slots(template, conversation_text)
    }


# --------------------------------------------------------------------------- 抽取

def _norm(text: str) -> str:
    return (text or "").replace(" ", "").replace("\u3000", "")


_NEG_PREFIX = ("没", "不", "未", "无", "别", "非")


def _negated_at(text: str, index: int) -> bool:
    """命中位置前 1–2 字是否为否定词 —— 防"没好转"被读成"好转"。

    实测事故：回答治疗问题时说"禁食了一顿**没好转**"，里面的"好转"命中了
    「发展与演变」的选项，把 progression 填成"好转"（与事实相反）。
    """
    prefix = text[max(0, index - 2):index]
    return any(prefix.endswith(w) for w in _NEG_PREFIX)


def match_option(text: str, slot: dict, allow_synonyms: bool = True) -> str | None:
    """在该槽位的选项里匹配；返回**选项原词**。

    :param allow_synonyms: 是否允许口语同义词参与匹配。

    ⚠️ **口语同义词必须只在"本轮问过该槽位"时启用** —— 否则会跨槽位污染：
    实测事故：``vomit_freq`` 配了同义词"一直吐"，于是引擎问「主诉」、宠主答
    **"我家狗一直吐"** 时，这句话顺手把「频次」填成了"5 次以上"（"一直吐"是主诉描述，
    不是频次答案）。书面语选项歧义低，可全槽位扫描；口语同义词歧义高，需上下文门控。

    ⚠️ 命中处若紧跟否定词则**丢弃该命中**（"没好转" ≠ "好转"）。
    """
    t = _norm(text)
    if not t:
        return None
    synonyms = slot.get("synonyms") or {}
    for option in slot.get("options", []):
        candidates = [option] + (list(synonyms.get(option, [])) if allow_synonyms else [])
        for alt in candidates:
            if not alt:
                continue
            idx = t.find(_norm(alt))
            if idx >= 0 and not _negated_at(t, idx):
                return option
    return None


def _extract_intake(text: str) -> str | None:
    """「48h 摄入与接触史」专用判据：阳性优先于阴性（"没吃巧克力"里的"巧克力"不算阳性）。"""
    t = _norm(text)
    negated = any(w in t for w in _NEGATIVE_WORDS)
    if negated:
        return "无明确误食/未换粮"
    for word in _POSITIVE_INTAKE:
        if word in t:
            return f"疑似摄入/接触：{word}"
    return None


def _extract_chief(conversation_text: str, template: dict) -> str | None:
    """「主诉」无选项，规则通道靠**已激活分支的 symptom** 兜底（如 吐 → 呕吐）。

    值偏简（如"呕吐"），确切表述（"反复呕吐 1 天"）由摘要生成阶段用 LLM 组织；
    这里只保证"该槽位不会被漏掉"——规则通道的职责是**不漏**，不是**措辞好**。
    """
    for name in activate_branches(template, conversation_text):
        symptom = template["branches"][name].get("symptom")
        if symptom:
            return symptom
    return None


def extract_reply(reply: str, template: dict, conversation_text: str,
                  asked_keys: list[str] | None = None) -> dict:
    """从**本轮回复**抽取槽位。

    :param reply: 宠主本轮说的话
    :param conversation_text: 累积对话文本（用于分支激活 —— 主诉可能出现在第一句）
    :param asked_keys: 本轮问过的槽位 key。**口语同义词仅在这些槽位上启用**（防跨槽位污染，
        见 ``match_option`` 的说明）；为 ``None`` 时**全部槽位都只用书面语选项**匹配
        （保守默认 —— 与 exposure 的门控语义保持一致）。
        ⚠️ 实测事故：曾让 ``asked_keys=None`` 时仍启用同义词，结果"正常吃**饭**"里的"饭"
        命中了「呕吐物性状 / 未消化食物」的同义词，把本该是"黄水"的槽位填错。

    :returns: ``{key: {"state", "value", "source"}}``，仅含本轮抽到的槽位
    """
    asked = set(asked_keys) if asked_keys else None
    found: dict = {}
    for slot in active_slots(template, conversation_text):
        key = slot["key"]
        if slot.get("options"):
            allow_syn = asked is not None and key in asked
            hit = match_option(reply, slot, allow_synonyms=allow_syn)
            if hit:
                found[key] = {"state": "filled", "value": hit, "source": "dialog"}
                continue
        if key == "exposure":
            # **仅在"本轮问过该槽位"时抽取** —— 实测事故：回答"诊疗经过"时说的"没给药"
            # 被误判为"疑似摄入药"。安全兜底由 FastPath 急症守卫独立承担（不依赖本槽位），
            # 因此这里可以保守：宁可不抽，不可错抽。
            if asked is not None and key in asked:
                hit = _extract_intake(reply)
                if hit:
                    found[key] = {"state": "filled", "value": hit, "source": "dialog"}
        elif key == "chief":
            hit = _extract_chief(conversation_text, template)
            if hit:
                found[key] = {"state": "filled", "value": hit, "source": "dialog"}
    return found


def merge_archive(state: dict, archive: dict | None, template: dict) -> dict:
    """用业务档案补全 archive 槽位（**有值就不问** —— 这是"少问一半问题"的关键）。"""
    if not archive:
        return state
    for slot in template.get("archive_slots", []):
        key = slot["key"]
        value = archive.get(key)
        if value and state.get(key, {}).get("state", "missing") == "missing":
            state[key] = {"state": "filled", "value": str(value), "source": "archive"}
    return state


def apply_reply(state: dict, reply: str, template: dict, conversation_text: str,
                archive: dict | None = None, asked_keys: list[str] | None = None) -> dict:
    """一轮完整处理：抽取 → 自由文本兜底 → 档案补全（就地更新并返回 state）。

    :param asked_keys: 本轮**问过**的槽位 key。规则通道填不上的**自由文本**槽位
        （模板标 ``free_text: true``，如"诊疗经过""一般情况"）用**宠主原话**填充 ——
        没有这道兜底，这类槽位会"问了却永远填不上"。填充**不做加工**，交由摘要阶段组织。
    """
    state.update(extract_reply(reply, template, conversation_text, asked_keys=asked_keys))
    # 自由文本兜底：**仅在本轮只问了一个问题时**才用原话填充 ——
    # 一轮问多个时无法把一段回复可靠归属到多个自由文本槽位（会张冠李戴），
    # 此时宁可留空让它下轮再问，也不填错。选项型槽位不受此限（靠选项词精确归属）。
    if asked_keys and len(asked_keys) == 1 and (reply or "").strip():
        key = asked_keys[0]
        specs = {s["key"]: s for s in active_slots(template, conversation_text)}
        spec = specs.get(key, {})
        if (spec.get("free_text")
                and state.get(key, {}).get("state", "missing") == "missing"):
            state[key] = {"state": "filled", "value": reply.strip(), "source": "dialog"}
    return merge_archive(state, archive, template)


# --------------------------------------------------------------------------- 缺口与追问

def gaps(state: dict, template: dict, conversation_text: str = "") -> list[dict]:
    """未填 / 含糊的槽位，按 **(优先级, 是否必填, 模板顺序)** 排序。"""
    slots = active_slots(template, conversation_text)
    order = {s["key"]: i for i, s in enumerate(slots)}
    out = []
    for slot in slots:
        st = state.get(slot["key"], {}).get("state", "missing")
        if st in ("missing", "vague"):
            out.append({
                "key": slot["key"], "name": slot.get("name", slot["key"]),
                "priority": slot.get("priority", "P3"),
                "required": bool(slot.get("required")), "state": st,
            })
    out.sort(key=lambda g: (_PRIORITY_RANK.get(g["priority"], 9),
                            0 if g["required"] else 1, order[g["key"]]))
    return out


def next_questions(state: dict, template: dict, conversation_text: str = "",
                   max_n: int | None = None) -> list[dict]:
    """选出本轮要问的 1–2 个槽位（带话术与选项）。派生槽位（无 ``ask``）自动跳过。"""
    limit = max_n or template.get("meta", {}).get("rules", {}).get(
        "max_questions_per_turn", 2)
    specs = {s["key"]: s for s in active_slots(template, conversation_text)}
    picks: list[dict] = []
    for gap in gaps(state, template, conversation_text):
        if len(picks) >= limit:
            break
        slot = specs[gap["key"]]
        ask = slot.get("ask") or slot.get("ask_if_missing")
        if not ask:
            continue
        picks.append({
            "key": slot["key"], "name": slot.get("name"),
            "priority": slot.get("priority"), "state": gap["state"],
            "ask": ask, "options": slot.get("options"), "why": slot.get("why"),
        })
    # **自由文本型问题独占一轮**：一轮问两个、一个回复无法可靠归属到两个自由文本槽位
    # （实测：问"与进食关系"+"一般情况"，宠主答"吃完没多久就吐"，结果把"一般情况"也填了）。
    # 选项型问题不受影响 —— 它们能靠选项词精确归属，可以成对问。
    if any(specs[p["key"]].get("free_text") for p in picks):
        return [p for p in picks if specs[p["key"]].get("free_text")][:1]
    return picks


def is_complete(state: dict, template: dict, conversation_text: str = "") -> bool:
    """``required`` 槽位全部非 ``missing`` 即视为可收尾（``vague`` 不阻塞收尾）。"""
    return not incomplete_keys(state, template, conversation_text)


def incomplete_keys(state: dict, template: dict, conversation_text: str = "") -> list[str]:
    """未采集到的**必填**槽位 —— 如实进 ``intake.incomplete``，不得空占位。"""
    return [
        s["key"] for s in active_slots(template, conversation_text)
        if s.get("required") and state.get(s["key"], {}).get("state") == "missing"
    ]


def render_status(state: dict, template: dict, conversation_text: str = "") -> list[dict]:
    """槽位状态快照（供前端进度展示与调试）。"""
    return [{
        "key": s["key"], "name": s.get("name"), "priority": s.get("priority"),
        "state": state.get(s["key"], {}).get("state", "missing"),
        "value": state.get(s["key"], {}).get("value", ""),
        "source": state.get(s["key"], {}).get("source", s.get("source", "dialog")),
    } for s in active_slots(template, conversation_text)]

"""分诊槽位抽取与剂量分级。

设计约束（见 交付物/分诊剂量分级-方案与验收口径.md）：
- **纯函数**：不读写 ChromaDB、不调 LLM、不 import langgraph（守住冷启动与 3.2ms 口径）；
- 无第三方依赖，只用标准库；
- 数据源为 `rules/toxin_dose_table.json`（人读版在 `knowledge/toxic_substances_handbook.md`），
  本模块不另立真值。

分级只覆盖巧克力/可可碱：手册给出了按体重的 mg/kg 阈值与按类型的 mg/g 含量表。
其余毒物手册未给按体重阈值，故**不编造阈值**，只走「需补充信息」分支。
"""
import json
import re
from pathlib import Path

_DOSE_TABLE_PATH = Path(__file__).resolve().parents[1] / "rules" / "toxin_dose_table.json"

# 中文数字（仅需支持 0-99 与「半」；宠物场景不会出现更大的口语数字）
_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_NUM = r"(?:\d+(?:\.\d+)?|[零一二两三四五六七八九十]+|半)"

_WEIGHT_RE = re.compile(r"(" + _NUM + r")\s*(公斤|千克|kg|KG|Kg|斤)\s*(半)?")
_GRAM_RE = re.compile(r"(" + _NUM + r")\s*(?:克|g|G)")
_PORTION_RE = re.compile(r"(" + _NUM + r")\s*(?:整)?\s*(板|条|块|颗|粒|片)")
_HOUR_RE = re.compile(r"(" + _NUM + r")\s*(?:个)?\s*(?:小时|钟头|h)\s*(?:前|之前|以前)?")
_MIN_RE = re.compile(r"(" + _NUM + r")\s*分钟\s*(?:前|之前|以前)?")

# 顺序即优先级：可可粉 > 烘焙黑 > 黑 > 牛奶 > 白（含量高者先判，避免「烘焙黑」被判成「黑」）
_TYPE_PATTERNS = (
    (re.compile(r"可可粉"), "可可粉"),
    (re.compile(r"烘焙"), "烘焙黑"),
    (re.compile(r"黑巧"), "黑"),
    (re.compile(r"牛奶巧"), "牛奶"),
    (re.compile(r"白巧"), "白"),
)

_table_cache = None


def load_dose_table(path=None):
    """读取剂量表；阈值按 mg/kg 降序返回，便于首个命中即为最高档。"""
    global _table_cache
    if path is None and _table_cache is not None:
        return _table_cache
    p = Path(path) if path else _DOSE_TABLE_PATH
    table = json.loads(p.read_text(encoding="utf-8"))
    for key, spec in table.items():
        if key.startswith("_"):
            continue          # `_notes` 是说明文本，其同名键不是阈值表
        if isinstance(spec, dict) and isinstance(spec.get("thresholds"), list):
            spec["thresholds"] = sorted(spec["thresholds"],
                                        key=lambda t: t["mg_per_kg"], reverse=True)
    if path is None:
        _table_cache = table
    return table


def _to_number(token):
    """把「5」「5.5」「半」「十五」「二十」统一转成 float；无法解析返回 None。"""
    if token is None:
        return None
    token = token.strip()
    if not token:
        return None
    if token == "半":
        return 0.5
    try:
        return float(token)
    except ValueError:
        pass
    if "十" in token:
        tens_part, _, ones_part = token.partition("十")
        tens = _CN_DIGIT.get(tens_part, 1) if tens_part else 1
        ones = _CN_DIGIT.get(ones_part, 0) if ones_part else 0
        return float(tens * 10 + ones)
    if len(token) == 1 and token in _CN_DIGIT:
        return float(_CN_DIGIT[token])
    return None


def extract_weight_kg(text):
    """抽取体重并归一到 kg（1 斤 = 0.5 kg；「两斤半」= 2.5 斤 = 1.25 kg）。"""
    if not text:
        return None
    m = _WEIGHT_RE.search(text)
    if not m:
        return None
    value = _to_number(m.group(1))
    if value is None or value <= 0:
        return None
    if m.group(3):            # 尾缀「半」：X 公斤半 / X 斤半 → +0.5 个单位
        value += 0.5
    return value / 2.0 if m.group(2) == "斤" else value


def extract_chocolate_type(text):
    if not text:
        return None
    for pattern, name in _TYPE_PATTERNS:
        if pattern.search(text):
            return name
    return None


def extract_amount(text, table=None):
    """抽取摄入量。显式克数优先；否则按常见零售规格估算并置 assumed=True。"""
    if not text:
        return None
    m = _GRAM_RE.search(text)
    if m:
        value = _to_number(m.group(1))
        if value and value > 0:
            return {"grams": value, "assumed": False, "raw": m.group(0)}
    m = _PORTION_RE.search(text)
    if m:
        count = _to_number(m.group(1))
        unit = m.group(2)
        portions = (table or load_dose_table())["chocolate"]["portions_g"]
        per_unit = portions.get(unit)
        if count and count > 0 and per_unit:
            return {"grams": count * per_unit, "assumed": True, "raw": m.group(0),
                    "count": count, "unit": unit, "per_unit_g": per_unit}
    return None


def extract_elapsed_hours(text):
    """抽取「误食距今多久」，单位小时。"""
    if not text:
        return None
    m = _HOUR_RE.search(text)
    if m:
        value = _to_number(m.group(1))
        if value is not None:
            return value
    m = _MIN_RE.search(text)
    if m:
        value = _to_number(m.group(1))
        if value is not None:
            return value / 60.0
    if re.search(r"刚刚|刚才|这会儿|立马", text):
        return 0.0
    return None


def extract_slots(text, table=None):
    """抽取全部槽位；缺失项为 None。"""
    return {
        "weight_kg": extract_weight_kg(text),
        "chocolate_type": extract_chocolate_type(text),
        "amount": extract_amount(text, table),
        "elapsed_h": extract_elapsed_hours(text),
    }


def assess_chocolate(weight_kg, choc_type, amount_g, table=None):
    """按可可碱 mg/kg 分级。返回档位、依据文案与计算明细（详情用于输出可追溯）。"""
    spec = (table or load_dose_table())["chocolate"]
    mg_per_g = spec["mg_per_g"][choc_type]
    total_mg = mg_per_g * amount_g
    mg_per_kg = total_mg / weight_kg
    tier = None
    for candidate in spec["thresholds"]:       # 已按 mg_per_kg 降序
        if mg_per_kg >= candidate["mg_per_kg"]:
            tier = candidate
            break
    return {
        "mg_per_kg": mg_per_kg,
        "total_mg": total_mg,
        "mg_per_g": mg_per_g,
        "tier": tier["tier"] if tier else "below",
        "tier_label": tier["label"] if tier else spec["below_label"],
        "threshold": tier["mg_per_kg"] if tier else None,
        "advice": tier["advice"] if tier else spec["below_advice"],
    }


def _fmt(value):
    """数值格式化：整数不带小数点；小数保留有效位（mg/g 小到 0.03，不能用定点小数）。"""
    return f"{round(float(value), 2):g}"


def _amount_phrase(amount):
    if amount["assumed"]:
        return (f"摄入约 {_fmt(amount['grams'])} g"
                f"（按「{_fmt(amount['count'])} {amount['unit']} ≈ "
                f"{_fmt(amount['per_unit_g'])} g」估算，请以包装净含量为准）")
    return f"摄入约 {_fmt(amount['grams'])} g"


def triage_note(toxin, slots=None, table=None):
    """生成【严重程度评估】段。

    数据足够（巧克力 + 体重 + 类型 + 摄入量）→ 给出 mg/kg 分级；
    否则列出缺口。任何情况下都不给出「无需就医」类结论。
    """
    slots = slots or {}
    if toxin != "巧克力":
        return ("【严重程度评估】该毒物的严重程度取决于摄入量与个体状况，"
                "需结合体重、摄入量、误食时间与现行症状由兽医判断。")

    spec = (table or load_dose_table())["chocolate"]
    missing = []
    if not slots.get("weight_kg"):
        missing.append("体重")
    if not slots.get("chocolate_type"):
        missing.append("巧克力类型")
    if not slots.get("amount"):
        missing.append("摄入量")

    if missing:
        ask = "；".join({
            "体重": "宠物体重（kg）",
            "巧克力类型": f"巧克力类型（{spec['type_hint']}）",
            "摄入量": "摄入克数（包装净含量通常有标注；也可用几块/几板描述，我会按常见规格估算）",
        }[item] for item in missing)
        known = []
        if slots.get("weight_kg"):
            known.append(f"体重 {_fmt(slots['weight_kg'])} kg")
        if slots.get("chocolate_type"):
            known.append(f"类型 {slots['chocolate_type']}")
        if slots.get("amount"):
            known.append(_amount_phrase(slots["amount"]))
        prefix = ("已识别：" + "、".join(known) + "。" if known else "")
        return ("【严重程度评估】暂无法计算 —— 可可碱中毒是剂量依赖的"
                "（20 mg/kg 起出现症状，60 mg/kg 以上可能抽搐）。"
                f"{prefix}还需补充：{ask}。")

    result = assess_chocolate(slots["weight_kg"], slots["chocolate_type"],
                              slots["amount"]["grams"], table)
    basis = (f"体重 {_fmt(slots['weight_kg'])} kg × {slots['chocolate_type']}巧克力 "
             f"{_fmt(result['mg_per_g'])} mg/g × {_amount_phrase(slots['amount'])}")
    if result["tier"] == "below":
        head = (f"【严重程度评估】按可可碱计约 {_fmt(result['mg_per_kg'])} mg/kg"
                f"（{basis}）。分级：{result['tier_label']}")
    else:
        head = (f"【严重程度评估】按可可碱计约 {_fmt(result['mg_per_kg'])} mg/kg"
                f"（{basis}）。分级：{result['tier_label']}"
                f"（手册阈值 {result['threshold']} mg/kg 起）")
    return f"{head} —— {result['advice']}"

"""处方安全审核能力位（AI-4）：发药前的**确定性**闸门。

为什么这一步不用 LLM
--------------------
本项目已有的分层哲学是「用确定性规则兜底概率模型」（Fast-Path 守卫就是这么做的）。
用药安全闸门比它更极端：**审核结论必须是可复现的**——同一条处方，今天和明天、
这个模型和那个模型，必须给出同一个 BLOCK。因此本模块：

- **纯规则 + 表驱动**，不调用模型；结论由 `Verdict` 表达，可离线单测；
- 规则表（`rules/prescription_safety_table.json`）只镜像 `knowledge/` 里**已载明**的
  约束，每条带 `chunk_id` 供医生溯源；
- **表外药物不猜**：输出 WARN（未收录，需人工核定），而不是 PASS 也不是 BLOCK；
- `BLOCK` 由规则直接产出，任何上层（含未来的模型复核）都不能把它降级为 PASS
  （`contract.worst_verdict` 保证合并时取最严重者）。

输入/输出契约见 `docs/a2a-contract.md`；本模块只接受**结构化**字段，不解析自由文本剂量
——「看不懂就 WARN」比「猜一个剂量」安全。
"""
from __future__ import annotations

import json
from pathlib import Path

from a2a.contract import Verdict, worst_verdict

DEFAULT_SAFETY_TABLE = "rules/prescription_safety_table.json"
DEFAULT_TOXIN_RULES = "rules/fastpath_rules.json"

# 物种别名归一：医院侧 "犬"/"狗"/"dog" 都要落到同一 key
_SPECIES_ALIAS = {
    "犬": "犬", "狗": "犬", "dog": "犬", "canine": "犬",
    "猫": "猫", "cat": "猫", "feline": "猫",
    "兔": "兔", "rabbit": "兔",
    "豚鼠": "豚鼠", "guinea pig": "豚鼠",
    "仓鼠": "仓鼠", "hamster": "仓鼠",
}


def normalize_species(value) -> str:
    """把物种文本归一到安全表里的 key；未知物种原样返回（小写去空白）。"""
    text = str(value or "").strip().lower()
    return _SPECIES_ALIAS.get(text, text)


def _contains(haystack: str, needles) -> str | None:
    """返回第一个命中的 needle（用于给出可追溯的命中词）。"""
    low = (haystack or "").lower()
    for needle in needles:
        if needle and needle.lower() in low:
            return needle
    return None


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _Finding(dict):
    """一条审核发现（dict 子类，便于直接序列化进契约 payload）。"""


class PrescriptionReviewer:
    """处方安全审核器。表可注入（测试用），默认从 rules/ 读取。"""

    def __init__(self, safety_table: dict | None = None,
                 toxin_rules: dict | None = None,
                 safety_table_path: str = DEFAULT_SAFETY_TABLE,
                 toxin_rules_path: str = DEFAULT_TOXIN_RULES):
        self.safety_table = safety_table if safety_table is not None \
            else _load_json(safety_table_path)
        self.toxin_rules = toxin_rules if toxin_rules is not None \
            else _load_json(toxin_rules_path)
        self._drugs = self.safety_table.get("drugs", [])

    # ------------------------------------------------------------------ 匹配

    def _match_drug(self, drug_name: str) -> dict | None:
        for drug in self._drugs:
            if _contains(drug_name, drug.get("names", [])):
                return drug
        return None

    # ------------------------------------------------------------------ 查询入口
    # 供 Agent 工具（lookup_drug_label / check_drug_safety）复用同一份规则表，
    # 不允许工具层另抄一份别名或剂量区间 —— 两套数据必然漂移。

    def find_drug(self, drug_name: str) -> dict | None:
        return self._match_drug(drug_name or "")

    def find_toxin(self, drug_name: str, species: str = "") -> dict | None:
        return self._match_toxin(drug_name or "", normalize_species(species))

    def _match_toxin(self, drug_name: str, species: str) -> dict | None:
        """处方里出现 P0 剧毒物（如对乙酰氨基酚、布洛芬）—— 直接视为违规处方。"""
        for toxin in self.toxin_rules.get("p0_toxins", []):
            if not _contains(drug_name, toxin.get("keywords", [])):
                continue
            allowed = [normalize_species(s) for s in toxin.get("species", [])]
            if species and allowed and species not in allowed:
                continue
            return toxin
        return None

    # ------------------------------------------------------------------ 主入口

    def review(self, payload: dict) -> dict:
        """对一份处方给出审核结论。

        :param payload: ``{"patient": {...}, "prescription": {"items": [...]}}``
        :returns: ``{"verdict", "confidence", "reasons", "citations", "checks", "undetermined"}``
        """
        patient = payload.get("patient") or {}
        prescription = payload.get("prescription") or {}
        items = prescription.get("items") or []

        species = normalize_species(patient.get("species"))
        raw_allergies = str(patient.get("allergies") or "").strip()
        allergies = "" if raw_allergies.lower() in {"无", "无过敏史", "暂无", "否", "none", "n/a", "-", "未见异常", "健康"} else raw_allergies
        weight_kg = patient.get("weight_kg")
        age_weeks = patient.get("age_weeks")
        pregnant = bool(patient.get("pregnant"))

        findings: list[_Finding] = []
        citations: list[str] = []
        undetermined: list[str] = []
        checks: dict = {}
        matched_drugs: list[tuple[dict, dict]] = []   # (item, table entry)

        def add(level: Verdict, code: str, message: str, citation: str | None = None):
            finding = _Finding(level=level.value, code=code, message=message,
                               citation=citation or "")
            findings.append(finding)
            if citation and citation not in citations:
                citations.append(citation)
            return finding

        # ---- 逐条药品：剧毒 / 物种禁用 / 剂量 / 频次 / 年龄体重 / 未收录 ----
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("drug_name") or "").strip()
            if not name:
                undetermined.append("存在无药名的明细行")
                continue

            # C2 处方法定剧毒物（优先级最高：即使表内也拦）
            toxin = self._match_toxin(name, species)
            if toxin:
                add(Verdict.BLOCK, "TOXIN_IN_PRESCRIPTION",
                    f"处方含已知对{species or '该物种'}剧毒的成分「{toxin['name']}」，"
                    f"不得作为治疗用药开具。",
                    toxin.get("chunk_id"))
                checks.setdefault("toxin", []).append(toxin["name"])
                continue

            drug = self._match_drug(name)
            if drug is None:
                # C8 表外药物：不猜，交人工
                add(Verdict.WARN, "DRUG_NOT_IN_KNOWLEDGE",
                    f"「{name}」不在已收录的说明书范围内，系统无法核验其禁忌与剂量，"
                    f"请兽医按说明书核定后放行。")
                undetermined.append(name)
                continue

            matched_drugs.append((item, drug))

            # C1 过敏禁用
            hit_allergy = _contains(allergies, drug.get("allergy_groups", []))
            if hit_allergy:
                add(Verdict.BLOCK, "ALLERGY_CONTRAINDICATION",
                    f"宠主档案记录过敏史「{allergies}」，而{drug['label']}对"
                    f"「{hit_allergy}」类药物禁用。",
                    drug.get("citations", {}).get("allergy"))
                checks.setdefault("allergy", []).append(drug["label"])

            # C3 物种禁用
            if species and species in [normalize_species(s)
                                       for s in drug.get("species_banned", [])]:
                add(Verdict.BLOCK, "SPECIES_BANNED",
                    f"{drug['label']}禁用于{species}。",
                    drug.get("citations", {}).get("species_banned"))

            # C5 剂量超参考区间上限 → 复核（不代开处方，故不 BLOCK）
            dose = item.get("dose_mg_per_kg")
            rng = (drug.get("dose_mg_per_kg") or {}).get(species)
            if rng and isinstance(dose, (int, float)):
                low, high = rng[0], rng[1]
                if dose > high:
                    add(Verdict.WARN, "DOSE_ABOVE_REFERENCE",
                        f"{drug['label']}给{species}的参考剂量上限为 {high} mg/kg，"
                        f"本处方为 {dose} mg/kg，请兽医复核（肝肾功能异常者尤需调整）。",
                        drug.get("citations", {}).get("dose"))
                elif dose < low:
                    add(Verdict.PASS, "DOSE_BELOW_REFERENCE",
                        f"{drug['label']}本处方 {dose} mg/kg 低于参考区间下限 {low} mg/kg"
                        f"（低于上限不构成安全风险，仅提示可能剂量不足）。",
                        drug.get("citations", {}).get("dose"))
            elif item.get("dose_mg_per_kg") is None:
                undetermined.append(f"{name} 未给出 mg/kg 剂量")

            # 猫用固定剂量却按 kg 记录 → 口径冲突，交人工
            if species in [normalize_species(s)
                           for s in drug.get("fixed_dose_species", [])] \
                    and isinstance(dose, (int, float)):
                add(Verdict.WARN, "DOSAGE_BASIS_MISMATCH",
                    f"{drug['label']}在{species}应按固定剂量给药，"
                    f"而本处方按 {dose} mg/kg 记录，请确认剂量口径。",
                    drug.get("citations", {}).get("dose"))

            # C6 频次上限（猫用甲硝唑须减频次）
            max_freq = (drug.get("max_frequency_per_day") or {}).get(species)
            freq = item.get("frequency_per_day")
            if max_freq is not None and isinstance(freq, (int, float)) and freq > max_freq:
                add(Verdict.WARN, "FREQUENCY_EXCEEDS_LIMIT",
                    f"{drug['label']}{species}建议每日不超过 {max_freq} 次，"
                    f"本处方为每日 {freq} 次。",
                    drug.get("citations", {}).get("frequency"))

            # C7 年龄 / 体重下限
            min_weeks = drug.get("min_age_weeks")
            if min_weeks is not None and isinstance(age_weeks, (int, float)) \
                    and age_weeks < min_weeks:
                add(Verdict.BLOCK, "AGE_BELOW_MINIMUM",
                    f"{drug['label']}要求最小 {min_weeks} 周龄，本宠物 {age_weeks} 周龄。",
                    drug.get("citations", {}).get("usage"))
            min_weight = drug.get("min_weight_kg")
            if min_weight is not None and isinstance(weight_kg, (int, float)) \
                    and weight_kg < min_weight:
                add(Verdict.BLOCK, "WEIGHT_BELOW_MINIMUM",
                    f"{drug['label']}要求体重 ≥ {min_weight} kg，本宠物 {weight_kg} kg。",
                    drug.get("citations", {}).get("usage"))

            # C9 猫需专用剂型（无法从结构化字段确认时提示人工核对）
            if drug.get("cat_requires_specific_form") and species == "猫":
                form = str(item.get("form") or "")
                if "猫" not in form:
                    add(Verdict.WARN, "SPECIES_SPECIFIC_FORM",
                        f"{drug['label']}用于猫时须使用猫专用剂型（犬猫剂型不可混用），"
                        f"当前未提供可核验的剂型信息。",
                        drug.get("citations", {}).get("form"))

            # 猫用多西环素：给药方式提醒（说明书明确，属医嘱提示而非违规）
            if drug.get("cat_advisory") and species == "猫":
                add(Verdict.PASS, "ADMINISTRATION_ADVISORY",
                    f"{drug['label']}：{drug['cat_advisory']}",
                    drug.get("citations", {}).get("cat_warning"))

        # ---- C4 跨明细相互作用 ----
        all_names = " ".join(str(i.get("drug_name") or "") for i in items if isinstance(i, dict))
        for item, drug in matched_drugs:
            for combo in drug.get("forbidden_combo_keywords", []):
                if combo and combo.lower() in all_names.lower():
                    add(Verdict.BLOCK, "FORBIDDEN_COMBINATION",
                        f"{drug['label']}与「{combo}」不可同时使用（同一处方内检出）。",
                        drug.get("citations", {}).get("interaction"))

        # ---- C10 有过敏史但无法与任何已知药物交叉比对 ----
        if allergies and not checks.get("allergy"):
            add(Verdict.WARN, "ALLERGY_UNVERIFIABLE",
                f"档案记录过敏史「{allergies}」，但本处方中的药物无法与已知过敏分组"
                f"自动比对，请人工核对后再发药。")

        # ---- C11 妊娠状态（说明书：妊娠早期慎用 / 慎用） ----
        if pregnant and matched_drugs:
            add(Verdict.WARN, "PREGNANCY_CAUTION",
                "该宠物处于妊娠状态，已开具药物需按说明书「妊娠慎用」条款复核。")

        # ---- 汇总 ----
        verdict = worst_verdict(*[Verdict(f["level"]) for f in findings]) \
            if findings else Verdict.PASS
        if not items:
            verdict = Verdict.WARN
            add(Verdict.WARN, "NO_ITEMS", "处方无明细，无法审核。")

        confidence = self._confidence(verdict, undetermined)
        return {
            "verdict": verdict.value,
            "confidence": confidence,
            "reasons": [dict(f) for f in findings],
            "citations": citations,
            "checks": checks,
            "undetermined": undetermined,
        }

    @staticmethod
    def _confidence(verdict: Verdict, undetermined: list) -> float:
        """置信度只表达「**核验覆盖度**」，不表达「药没问题」的概率。

        - 每有一项无法自动核验的输入扣 0.2，下限 0.3；
        - BLOCK 由确定性规则命中得出，覆盖度不影响其确定性 → 不低于 0.9。
        """
        base = max(0.3, 1.0 - 0.2 * len(undetermined))
        if verdict is Verdict.BLOCK:
            return max(base, 0.9)
        return round(base, 2)


def review_prescription(payload: dict, reviewer: PrescriptionReviewer | None = None) -> dict:
    """模块级便捷入口（supervisor / 单测用）。"""
    return (reviewer or PrescriptionReviewer()).review(payload)

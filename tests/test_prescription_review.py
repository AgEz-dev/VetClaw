"""tests/test_prescription_review.py：处方安全审核（AI-4）—— 规则闸门逐条断言。

全部离线：审核器是纯规则 + 表驱动，不依赖模型、不依赖 Chroma。
表格里的每条约束都必须能指到 `knowledge/` 的 chunk（引用缺失即视为测试失败）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from prescription_review import PrescriptionReviewer, normalize_species, review_prescription


def _codes(out):
    return [r["code"] for r in out["reasons"]]


def _payload(items, **patient):
    return {"patient": patient, "prescription": {"items": items}}


# --------------------------------------------------------------------- 归一

def test_species_alias_normalized():
    assert normalize_species("狗") == "犬"
    assert normalize_species("Dog") == "犬"
    assert normalize_species("猫") == "猫"
    assert normalize_species("兔") == "兔"
    assert normalize_species("羊驼") == "羊驼"     # 未知物种原样返回


# --------------------------------------------------------------------- 阻断类（BLOCK）

def test_penicillin_allergy_blocks_amoxicillin():
    """说明书：对青霉素类过敏的动物禁用 → 档案写「青霉素」、处方开速诺，必须拦。"""
    out = review_prescription(_payload(
        [{"drug_name": "速诺", "dose_mg_per_kg": None}],
        species="猫", allergies="青霉素"))
    assert out["verdict"] == "block"
    assert "ALLERGY_CONTRAINDICATION" in _codes(out)
    assert out["citations"], "阻断必须带可溯源引用"
    assert "速诺" in out["citations"][0]


def test_species_banned_blocks_amoxicillin_in_rabbit():
    """说明书：兔、豚鼠、仓鼠等小型草食动物禁用（致命性肠道菌群紊乱）。"""
    out = review_prescription(_payload([{"drug_name": "阿莫西林"}], species="兔"))
    assert out["verdict"] == "block"
    assert "SPECIES_BANNED" in _codes(out)


def test_toxic_drug_in_prescription_is_blocked():
    """处方里出现 P0 剧毒物（对乙酰氨基酚）—— 犬猫皆毒，直接阻断并引用毒物矩阵。"""
    out = review_prescription(_payload([{"drug_name": "对乙酰氨基酚"}], species="犬"))
    assert out["verdict"] == "block"
    assert "TOXIN_IN_PRESCRIPTION" in _codes(out)
    assert any("species_drug_safety_matrix" in c for c in out["citations"])


def test_forbidden_combination_drontal_with_piperazine():
    """说明书：拜宠清与驱蛔药（如哌嗪）不可同时使用 → 同一处方内检出即拦。"""
    out = review_prescription(_payload(
        [{"drug_name": "拜宠清"}, {"drug_name": "哌嗪"}], species="犬"))
    assert out["verdict"] == "block"
    assert "FORBIDDEN_COMBINATION" in _codes(out)


def test_frontline_blocked_below_min_weight():
    """说明书：福来恩体重 < 2 kg 禁用。"""
    out = review_prescription(_payload([{"drug_name": "福来恩"}],
                                       species="猫", weight_kg=1.5))
    assert out["verdict"] == "block"
    assert "WEIGHT_BELOW_MINIMUM" in _codes(out)


def test_frontline_blocked_below_min_age():
    out = review_prescription(_payload([{"drug_name": "福来恩"}],
                                       species="猫", age_weeks=6, weight_kg=3))
    assert out["verdict"] == "block"
    assert "AGE_BELOW_MINIMUM" in _codes(out)


# --------------------------------------------------------------------- 复核类（WARN）

def test_cat_metronidazole_frequency_exceeds_limit():
    """说明书：猫需减频次，防神经毒性 → 每日 2 次应提示复核。"""
    out = review_prescription(_payload(
        [{"drug_name": "甲硝唑", "dose_mg_per_kg": 15, "frequency_per_day": 2}],
        species="猫"))
    assert out["verdict"] == "warn"
    assert "FREQUENCY_EXCEEDS_LIMIT" in _codes(out)


def test_dose_above_reference_gives_warn_not_block():
    """超参考区间上限 → WARN（说明书写明实际剂量由兽医核定），不是 BLOCK。"""
    out = review_prescription(_payload(
        [{"drug_name": "多西环素", "dose_mg_per_kg": 20}], species="犬"))
    assert out["verdict"] == "warn"
    assert "DOSE_ABOVE_REFERENCE" in _codes(out)


def test_unknown_drug_is_warn_not_pass():
    """表外药物**不猜**：既不放行(PASS)也不阻断(BLOCK)，交人工核定。"""
    out = review_prescription(_payload([{"drug_name": "某某新药"}], species="犬"))
    assert out["verdict"] == "warn"
    assert "DRUG_NOT_IN_KNOWLEDGE" in _codes(out)
    assert out["undetermined"], "未核验项必须被记录，置信度据此下调"


def test_allergy_present_but_unverifiable_is_warn():
    """有过敏史但无法与已知药物比对 → 不得给 PASS。"""
    out = review_prescription(_payload([{"drug_name": "甲硝唑"}],
                                       species="犬", allergies="海鲜"))
    assert out["verdict"] == "warn"
    assert "ALLERGY_UNVERIFIABLE" in _codes(out)


def test_pregnancy_caution_warns():
    out = review_prescription(_payload([{"drug_name": "甲硝唑"}],
                                       species="犬", pregnant=True))
    assert out["verdict"] == "warn"
    assert "PREGNANCY_CAUTION" in _codes(out)


# --------------------------------------------------------------------- 放行（PASS）

def test_standard_amoxicillin_dose_passes():
    out = review_prescription(_payload(
        [{"drug_name": "速诺", "dose_mg_per_kg": 12.5, "frequency_per_day": 2}],
        species="犬", weight_kg=10))
    assert out["verdict"] == "pass"
    assert "ALLERGY" not in "".join(_codes(out))


def test_low_dose_is_pass_with_hint():
    """低于参考区间下限不是安全风险，只提示可能剂量不足 → 结论仍为 pass。"""
    out = review_prescription(_payload(
        [{"drug_name": "多西环素", "dose_mg_per_kg": 2}], species="犬"))
    assert out["verdict"] == "pass"
    assert "DOSE_BELOW_REFERENCE" in _codes(out)


# --------------------------------------------------------------------- 结构不变量

def test_empty_prescription_is_warn_with_code():
    out = review_prescription({"patient": {"species": "犬"}, "prescription": {}})
    assert out["verdict"] == "warn"
    assert "NO_ITEMS" in _codes(out)


def test_block_confidence_never_low():
    """BLOCK 由确定性规则命中得出，覆盖度不足也不应把它的置信度压低。"""
    out = review_prescription(_payload(
        [{"drug_name": "对乙酰氨基酚"}, {"drug_name": "未知药A"}], species="犬"))
    assert out["verdict"] == "block"
    assert out["confidence"] >= 0.9


def test_all_table_citations_point_to_knowledge_chunks():
    """表内每条引用都必须指向 knowledge/ 下的真实 chunk id 形态。"""
    reviewer = PrescriptionReviewer()
    for drug in reviewer.safety_table["drugs"]:
        for key, cid in (drug.get("citations") or {}).items():
            assert cid.startswith("knowledge/"), f"{drug['key']}.{key} 不是知识库引用"
            assert "#" in cid, f"{drug['key']}.{key} 缺少 chunk 定位"


def test_worst_verdict_keeps_block():
    """合并结论取最严重者：规则层 BLOCK 永远不被模型层 PASS 覆盖。"""
    from a2a.contract import Verdict, worst_verdict
    assert worst_verdict(Verdict.BLOCK, Verdict.PASS) is Verdict.BLOCK
    assert worst_verdict(Verdict.WARN, Verdict.PASS) is Verdict.WARN
    assert worst_verdict(Verdict.PASS, Verdict.NONE) is Verdict.PASS


if __name__ == "__main__":
    raise SystemExit(sys.exit(__import__("pytest").main([__file__, "-q"])))

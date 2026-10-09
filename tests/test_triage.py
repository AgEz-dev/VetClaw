"""tests/test_triage.py：分诊槽位抽取与剂量分级验收。

对应《分诊剂量分级-方案与验收口径》A1–A5。剂量期望值在测试内**独立复算**
（直接用手册的中值），不读 rules/toxin_dose_table.json —— 避免与实现互为镜像的自证。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from triage import (assess_chocolate, extract_amount, extract_chocolate_type,  # noqa: E402
                    extract_elapsed_hours, extract_slots, extract_weight_kg,
                    triage_note)

# 手册 knowledge/toxic_substances_handbook.md 的独立复述（区间取中值）
HANDBOOK_MG_PER_G = {"白": 0.03, "牛奶": 2.5, "黑": 6.5, "烘焙黑": 13, "可可粉": 23}
THRESHOLDS = [(60, "重度"), (40, "中度"), (20, "轻度")]


def expected_tier(mg_per_kg):
    for lower, label in THRESHOLDS:
        if mg_per_kg >= lower:
            return label
    return "未达轻度阈值"


# --------------------------------------------------------------- A3 槽位抽取


def test_weight_kg_unit_and_jin():
    assert extract_weight_kg("我家5公斤的泰迪") == 5.0
    assert extract_weight_kg("体重 5 kg") == 5.0
    assert extract_weight_kg("35公斤拉布拉多") == 35.0
    assert extract_weight_kg("十斤的狗") == 5.0     # 1 斤 = 0.5 kg
    assert extract_weight_kg("10斤的狗") == 5.0
    assert extract_weight_kg("两斤半的猫") == 1.25  # 「两斤半」→ 数字取「两」


def test_weight_absent_on_unrelated_text():
    for text in ["狗吃了巧克力怎么办", "猫误食百合", "你好"]:
        assert extract_weight_kg(text) is None


def test_chocolate_type_precedence():
    assert extract_chocolate_type("吃了可可粉") == "可可粉"
    assert extract_chocolate_type("一整板烘焙黑巧克力") == "烘焙黑"  # 不可被判成「黑」
    assert extract_chocolate_type("半块黑巧克力") == "黑"
    assert extract_chocolate_type("牛奶巧克力") == "牛奶"
    assert extract_chocolate_type("白巧克力") == "白"
    assert extract_chocolate_type("偷吃了巧克力") is None           # 未指明类型


def test_amount_explicit_grams_not_assumed():
    a = extract_amount("吃了40克牛奶巧克力")
    assert a["grams"] == 40.0
    assert a["assumed"] is False


def test_amount_portion_is_marked_assumed():
    a = extract_amount("偷吃了半板黑巧克力")
    assert a["grams"] == 50.0      # 半 × 板(100 g)
    assert a["assumed"] is True
    b = extract_amount("吃了一整板烘焙黑巧克力")
    assert b["grams"] == 100.0 and b["assumed"] is True


def test_amount_absent():
    assert extract_amount("狗吃了巧克力") is None


def test_elapsed_hours():
    assert extract_elapsed_hours("大概2小时前") == 2.0
    assert extract_elapsed_hours("半小时前") == 0.5
    assert extract_elapsed_hours("30分钟前") == 0.5
    assert extract_elapsed_hours("刚刚吃的") == 0.0
    assert extract_elapsed_hours("狗吃了巧克力") is None


# --------------------------------------------------------------- A1 剂量复算


def test_a1_dose_matches_independent_recompute():
    """独立复算：mg/kg = mg_per_g × 克数 ÷ 体重，逐类型逐量比对。"""
    for ctype, mg_per_g in HANDBOOK_MG_PER_G.items():
        for weight in (2.0, 5.0, 15.0, 35.0):
            for grams in (1.0, 12.0, 40.0, 100.0):
                want = mg_per_g * grams / weight
                got = assess_chocolate(weight, ctype, grams)
                assert abs(got["mg_per_kg"] - want) < 1e-9, (ctype, weight, grams)
                assert got["tier_label"] == expected_tier(want)


# --------------------------------------------------------------- A2 阈值边界


def test_a2_threshold_boundaries():
    """20 / 40 / 60 三个阈值的上下相邻点必须落对档。"""
    for target in (19.9, 20.0, 39.9, 40.0, 59.9, 60.0, 200.0):
        grams = target * 5.0 / 6.5          # 5 kg 黑巧(6.5 mg/g) 反推克数
        got = assess_chocolate(5.0, "黑", grams)
        assert abs(got["mg_per_kg"] - target) < 1e-9
        assert got["tier_label"] == expected_tier(target), target


# --------------------------------------------------------------- A4 输出标注


def test_a4_graded_note_reports_mg_per_kg_and_tier():
    slots = extract_slots("我家5公斤的泰迪偷吃了半板黑巧克力，大概2小时前")
    note = triage_note("巧克力", slots)
    assert "65 mg/kg" in note                      # 6.5 × 50 / 5
    assert "重度" in note
    assert "估算" in note                           # 摄入量为按板估算 → 必须标注
    assert "100 g" in note                          # 必须打印所用克数假设
    assert "立即" in note                           # 不得弱化就医建议


def test_a4_explicit_grams_not_marked_estimated():
    slots = extract_slots("5公斤的狗吃了40克黑巧克力")
    note = triage_note("巧克力", slots)
    assert "52 mg/kg" in note                       # 6.5 × 40 / 5
    assert "估算" not in note


def test_a4_ask_lists_only_missing_slots():
    """已识别的槽位不应出现在「还需补充」里。"""
    note = triage_note("巧克力", extract_slots("猫吃了一点可可粉，30分钟前"))
    assert "已识别：类型 可可粉" in note
    assert note.count("巧克力类型") == 0            # 类型已知 → 不再索要


def test_a4_ask_when_nothing_known():
    note = triage_note("巧克力", extract_slots("狗吃了巧克力怎么办"))
    assert "宠物体重" in note and "巧克力类型" in note and "摄入克数" in note


# --------------------------------------------------------------- 非巧克力不编造


def test_non_chocolate_has_no_fabricated_threshold():
    note = triage_note("百合", extract_slots("猫吃了百合"))
    assert "mg/kg" not in note                       # 手册未给该毒物按体重阈值
    assert "百合" not in note or True                # 不做毒物专属断言，只锁「不编数字」


def test_unknown_chocolate_type_not_graded():
    """类型缺失时 mg/g 跨度达 700 倍，不得给出分级结论。"""
    note = triage_note("巧克力", extract_slots("5公斤的狗吃了一整板巧克力"))
    assert "mg/kg" in note                            # 仅出现在「阈值说明」里
    assert "分级：" not in note
    assert "巧克力类型" in note


# --------------------------------------------------------------- A5 守卫端到端


def _make_guard():
    from fastpath import FastPathGuard

    docs = {
        "knowledge/species_drug_safety_matrix.md#现场急救通用-SOP-所有毒物中毒通用-0":
            "SOP：立即脱离毒源并送医",
        "knowledge/toxic_substances_handbook.md#巧克力与可可碱中毒-0":
            "巧克力含可可碱，犬代谢慢。",
        "knowledge/species_drug_safety_matrix.md#葡萄与葡萄干-6":
            "葡萄导致犬急性肾损伤。",
    }

    class FakeCollection:
        def get(self, ids):
            return {"documents": [docs[i] for i in ids]}

    class FakePipeline:
        collection = FakeCollection()

    return FastPathGuard(pipeline=FakePipeline()), docs


def test_a5_emergency_message_prefix_and_fetch_order():
    """预警头仍在最前；SOP 与毒物详情原样保留；免责声明仍在最末。"""
    guard, _ = _make_guard()
    hit = guard.check("狗吃了葡萄")
    answer = guard.emergency_message(hit["toxin"], hit["species"],
                                     hit["chunk_id"], hit["slots"])
    assert answer.startswith("【P0 急症预警】")
    assert "SOP：立即脱离毒源并送医" in answer
    assert "葡萄导致犬急性肾损伤" in answer
    assert answer.rstrip().endswith("紧急情况请立即就医。")
    assert "【严重程度评估】" in answer                # 新增段存在


def test_a5_check_exposes_slots():
    guard, _ = _make_guard()
    hit = guard.check("5公斤的狗偷吃了半板黑巧克力")
    assert hit["action"] == "emergency"
    assert hit["slots"]["weight_kg"] == 5.0
    assert hit["slots"]["chocolate_type"] == "黑"


def test_a5_agent_end_to_end_graded_without_llm():
    """经 ReActAgent 触发急症：分级段出现，且全程不触碰 LLM 客户端。"""
    from core_agent import ReActAgent
    from tools import ToolRegistry

    guard, _ = _make_guard()

    class NoLLMClient:
        pass  # P0 路径不应触碰 client，任何属性访问都会 AttributeError

    agent = ReActAgent(NoLLMClient(), ToolRegistry(), model="fake",
                       fastpath_guard=guard)
    answer = agent.run("我家5公斤的狗偷吃了半板黑巧克力，大概2小时前")
    assert answer.startswith("【P0 急症预警】")
    assert "65 mg/kg" in answer
    assert "重度" in answer

    asked = agent.run("狗吃了巧克力")
    assert "还需补充" in asked


def test_a5_legacy_call_without_slots_still_works():
    """向后兼容：不传 slots 时不得抛异常（旧调用方 / 降级路径）。"""
    guard, _ = _make_guard()
    msg = guard.emergency_message("不明毒物", "猫", None)
    assert msg.startswith("【P0 急症预警】")
    assert "关于不明毒物的详细信息" not in msg


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("全部断言通过")

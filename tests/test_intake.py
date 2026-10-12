"""tests/test_intake.py：预问诊槽位引擎断言（纯函数、零 LLM、零网络、离线可跑）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intake import (  # noqa: E402
    _extract_intake, load_template, activate_branches, active_slots, empty_state,
    match_option, extract_reply, merge_archive, apply_reply, gaps, next_questions,
    is_complete, incomplete_keys, render_status,
)

TPL = load_template()
ARCHIVE = {
    "vaccine": "三针已完成", "past_disease": "无慢性病史",
    "medications": "无在用药", "allergy": "无已知过敏",
}


def _slot(template, key):
    for s in (template["base_slots"] + template["archive_slots"]
              + template["branches"]["vomiting"]["slots"]):
        if s["key"] == key:
            return s
    raise KeyError(key)


# --------------------------------------------------------------------- 模板完整性

def test_template_loads():
    assert TPL["version"] == "1.0"
    assert TPL["base_slots"] and TPL["archive_slots"]
    assert TPL["branches"]["vomiting"]["slots"]


def test_template_schema_integrity():
    """必填必须有话术、每条必须带依据、选项槽位必须配同义词 —— 防止模板退化"""
    allslots = (TPL["base_slots"] + TPL["archive_slots"]
                + TPL["branches"]["vomiting"]["slots"])
    for s in allslots:
        assert s.get("why"), "槽位缺依据: %s" % s["key"]
        if s.get("required"):
            assert s.get("ask"), "必填槽位缺话术: %s" % s["key"]
        if s.get("options"):
            assert s.get("synonyms"), "选项槽位缺同义词表: %s" % s["key"]


# --------------------------------------------------------------------- 分支与槽位集合

def test_branch_activation():
    assert activate_branches(TPL, "我家狗一直吐") == ["vomiting"]
    assert activate_branches(TPL, "它皮肤有点痒") == []


def test_active_slots_branch_delta():
    assert len(active_slots(TPL, "它一直吐")) - len(active_slots(TPL, "它皮肤痒")) == 6


def test_empty_state_is_all_missing():
    st = empty_state(TPL, "吐")
    assert all(v["state"] == "missing" for v in st.values())
    assert len(st) == 18


# --------------------------------------------------------------------- 选项匹配（含同义词）

def test_match_option_written_and_synonym():
    slot = _slot(TPL, "vomit_content")
    assert match_option("黄绿色液体", slot) == "黄绿色液体"
    assert match_option("吐的是黄水", slot) == "黄绿色液体"      # 口语同义词
    assert match_option("吐了咖啡色的", slot) == "咖啡渣样"       # 口语同义词


def test_match_option_no_false_positive():
    assert match_option("今天天气不错", _slot(TPL, "vomit_content")) is None


def test_intake_slot_negative_and_positive():
    assert _extract_intake("没有，就正常吃饭，也没乱啃东西") == "无明确误食/未换粮"
    assert _extract_intake("好像吃了巧克力") == "疑似摄入/接触：巧克力"


# --------------------------------------------------------------------- 抽取

def test_extract_multi_answer_in_one_reply():
    conv = "我家狗一直吐"
    found = extract_reply("有用力干呕，三四次吧", TPL, conv,
                          asked_keys=["vomit_kind", "vomit_freq"])
    assert found["vomit_kind"]["value"] == "有干呕用力"
    assert found["vomit_freq"]["value"] == "3–5 次"


def test_reply_without_asked_keys_uses_written_options_only():
    """保守默认：未告知"问过什么"时只用**书面语选项** —— 防口语同义词乱填。

    实测事故：「未消化食物」配了同义词"饭"，宠主说"正常吃**饭**"时被填成了呕吐物性状。
    """
    found = extract_reply("没有，就正常吃饭", TPL, "我家狗一直吐")   # 不传 asked_keys
    assert "vomit_content" not in found, "未问过该槽位时，同义词不得生效"
    # 书面语选项仍然可用（歧义低）
    found2 = extract_reply("吐的是黄绿色液体", TPL, "我家狗一直吐")
    assert found2["vomit_content"]["value"] == "黄绿色液体"


def test_extract_chief_from_branch_symptom():
    found = extract_reply("我家狗一直吐", TPL, "我家狗一直吐")
    assert found["chief"]["value"] == "呕吐"
    assert found["chief"]["source"] == "dialog"


def test_free_text_fallback_only_for_asked_slots():
    """自由文本槽位（general）只在"被问过"时才用原话填充 —— 避免从无关文本误填"""
    st = empty_state(TPL, "吐")
    apply_reply(st, "精神差，水喝得少", TPL, "吐")                      # 没问 → 不填
    assert st["general"]["state"] == "missing"
    apply_reply(st, "精神差，水喝得少", TPL, "吐", asked_keys=["general"])  # 问了 → 用原话
    assert st["general"]["state"] == "filled"
    assert st["general"]["value"] == "精神差，水喝得少"


# --------------------------------------------------------------------- 档案补全

def test_archive_fills_and_respects_existing_value():
    st = empty_state(TPL, "吐")
    merge_archive(st, ARCHIVE, TPL)
    assert st["vaccine"]["state"] == "filled"
    assert st["vaccine"]["source"] == "archive"

    st2 = empty_state(TPL, "吐")
    st2["vaccine"] = {"state": "filled", "value": "主人自述打过", "source": "dialog"}
    merge_archive(st2, ARCHIVE, TPL)
    assert st2["vaccine"]["value"] == "主人自述打过", "已有值不得被档案覆盖"


# --------------------------------------------------------------------- 缺口与追问

def test_gaps_put_safety_first():
    g = gaps(empty_state(TPL, "吐"), TPL, "吐")
    assert g[0]["priority"] == "P0", "P0 安全项必须排在最前"
    assert {g[0]["key"], g[1]["key"]} == {"exposure", "vomit_danger"}


def test_next_questions_capped_and_skip_derived():
    qs = next_questions(empty_state(TPL, "吐"), TPL, "吐")
    assert len(qs) == 2, "每轮最多 2 问"
    assert all(q["ask"] for q in qs)
    assert all(q["key"] != "other_findings" for q in qs), "派生槽位不问"
    assert qs[0]["priority"] == "P0"


def test_next_questions_respects_custom_cap():
    qs = next_questions(empty_state(TPL, "吐"), TPL, "吐", max_n=1)
    assert len(qs) == 1


# --------------------------------------------------------------------- 收尾

def test_completion_and_incomplete_listing():
    st = empty_state(TPL, "吐")
    merge_archive(st, ARCHIVE, TPL)
    assert not is_complete(st, TPL, "吐")
    assert len(incomplete_keys(st, TPL, "吐")) == 12, "呕吐分支下应有 12 个必填项"
    for s in active_slots(TPL, "吐"):
        if s.get("required"):
            st[s["key"]] = {"state": "filled", "value": "x", "source": "dialog"}
    assert is_complete(st, TPL, "吐")
    assert incomplete_keys(st, TPL, "吐") == []


def test_render_status_shape():
    rows = render_status(empty_state(TPL, "吐"), TPL, "吐")
    assert len(rows) == 18
    assert {"key", "name", "priority", "state", "value", "source"} <= set(rows[0])


# --------------------------------------------------------------------- 闭环模拟

def test_end_to_end_loop_until_complete():
    """模拟完整闭环：引擎提问 → 宠主按脚本作答 → 直到必填齐备（含轮数上限自洽）"""
    answers = {
        "chief": "我家狗一直吐",
        "onset": "慢慢变严重的",
        "exposure": "没有，就正常吃饭",
        "vomit_content": "黄水，有点泡沫",
        "vomit_relation": "吃完没多久就吐",
        "vomit_kind": "有用力干呕",
        "vomit_freq": "三四次吧",
        "vomit_danger": "肚子还好，没有一直干呕",
        "vomit_associated": "没拉稀，精神差点",
        "progression": "越来越重",
        "treatment": "没给药，禁食了一顿没好转",
        "general": "精神差、水喝得少，体重没称过",
    }
    conv, turns = "", 0
    st = merge_archive(empty_state(TPL, ""), ARCHIVE, TPL)
    max_turns = TPL["meta"]["rules"]["max_turns"]
    while not is_complete(st, TPL, conv) and turns < max_turns:
        qs = next_questions(st, TPL, conv)
        assert qs, "有缺口却问不出问题（疑似必填槽位缺 ask 话术）"
        turns += 1
        for q in qs:
            reply = answers.get(q["key"], "")
            conv += reply + "\n"
            apply_reply(st, reply, TPL, conv, asked_keys=[x["key"] for x in qs])
    assert is_complete(st, TPL, conv), (
        "达轮数上限仍未齐备，缺: %s" % incomplete_keys(st, TPL, conv))
    assert turns <= max_turns
    assert st["chief"]["value"] == "呕吐"
    assert st["vomit_freq"]["value"] == "3–5 次"
    assert st["vomit_danger"]["value"] == "都没有"
    assert st["exposure"]["value"] == "无明确误食/未换粮"
    assert st["vaccine"]["source"] == "archive"
    assert st["general"]["value"] == "精神差、水喝得少，体重没称过"
    print("      → 用 %d 轮完成，18 槽位全部落位" % turns)


# ----------------------------------------------------- 歧义防护（两条实测事故的回归）

def test_synonym_not_used_unless_asked():
    """回归事故①：问「主诉」答"我家狗一直吐"，曾把「频次」误填为"5 次以上"。

    口语同义词歧义高（"一直吐"是主诉描述，不是频次），必须只在**本轮问过该槽位**时启用。
    """
    conv = "我家狗一直吐"
    found = extract_reply("我家狗一直吐", TPL, conv, asked_keys=["chief"])
    assert "vomit_freq" not in found, "未问的槽位被口语同义词误填"
    found2 = extract_reply("一直吐", TPL, conv, asked_keys=["vomit_freq"])
    assert found2["vomit_freq"]["value"] == "5 次以上", "问了该槽位时同义词应可用"


def test_exposure_not_extracted_when_not_asked():
    """回归事故②：回答「诊疗经过」说的"没给药"，曾被读成「摄入史」= 疑似摄入药。

    自由判据型槽位（exposure）同样需要"本轮问过"才抽取 —— 安全兜底由 FastPath 独立承担。
    """
    conv = "我家狗一直吐"
    found = extract_reply("没给药，禁食了一顿没好转", TPL, conv, asked_keys=["treatment"])
    assert "exposure" not in found, "未问及摄入史却抽出了 exposure"
    found2 = extract_reply("没有，就正常吃饭", TPL, conv, asked_keys=["exposure"])
    assert found2["exposure"]["value"] == "无明确误食/未换粮"


def test_exposure_uses_specific_toxin_words_only():
    """笼统词不得作为摄入证据（"药"这种词会把"没给药"读成阳性）"""
    from intake import _POSITIVE_INTAKE
    assert "药" not in _POSITIVE_INTAKE, "'药'过于笼统，会造成误判"
    assert "巧克力" in _POSITIVE_INTAKE and "老鼠药" in _POSITIVE_INTAKE


def test_negated_option_is_rejected():
    """回归事故③：回答治疗说的"禁食了一顿没好转"，曾把「发展与演变」填成"好转"（与事实相反）"""
    prog = _slot(TPL, "progression")
    assert match_option("没好转", prog) is None, "否定语境不得判为阳性"
    assert match_option("好多了，见好", prog) == "好转", "肯定语境应正常匹配"

    danger = _slot(TPL, "vomit_danger")
    assert match_option("没有一直干呕", danger) == "都没有", "'没有…'应归到'都没有'而非'干呕无物'"


def test_synonyms_unique_across_slots():
    """数据质量守卫：同一口语同义词不得挂在两个槽位（否则命中时归属不确定）"""
    owner = {}
    allslots = (TPL["base_slots"] + TPL["archive_slots"]
                + TPL["branches"]["vomiting"]["slots"])
    for s in allslots:
        for opt, syns in (s.get("synonyms") or {}).items():
            for alt in [opt] + list(syns):
                key = alt.replace(" ", "")
                assert key not in owner or owner[key] == s["key"], (
                    "同义词 %r 同时挂在 %s 与 %s" % (alt, owner.get(key), s["key"]))
                owner[key] = s["key"]


if __name__ == "__main__":
    test_template_loads()
    test_template_schema_integrity()
    test_branch_activation()
    test_active_slots_branch_delta()
    test_empty_state_is_all_missing()
    test_match_option_written_and_synonym()
    test_match_option_no_false_positive()
    test_intake_slot_negative_and_positive()
    test_extract_multi_answer_in_one_reply()
    test_reply_without_asked_keys_uses_written_options_only()
    test_extract_chief_from_branch_symptom()
    test_free_text_fallback_only_for_asked_slots()
    test_archive_fills_and_respects_existing_value()
    test_gaps_put_safety_first()
    test_next_questions_capped_and_skip_derived()
    test_next_questions_respects_custom_cap()
    test_completion_and_incomplete_listing()
    test_render_status_shape()
    test_end_to_end_loop_until_complete()
    test_synonym_not_used_unless_asked()
    test_exposure_not_extracted_when_not_asked()
    test_exposure_uses_specific_toxin_words_only()
    test_negated_option_is_rejected()
    test_synonyms_unique_across_slots()
    print("全部断言通过")

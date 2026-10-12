"""tests/test_intake_llm.py：LLM 通道的提示词与**严格解析**断言（纯函数、零网络）。

重点验证"防幻觉"与"红线" —— 模型会输出不存在的槽位名、不在选项内的值、
甚至诊断结论；这些**必须被丢弃**，因为预问诊数据要进医生的眼睛。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intake import load_template  # noqa: E402
from intake_llm import (  # noqa: E402
    build_extract_messages, parse_extract_response,
    build_summary_messages, parse_summary_response, contains_diagnosis,
)

TPL = load_template()
CONV = "主人：我家狗一直吐，昨天开始的\n助手：它有没有可能吃到不该吃的东西？\n主人：没有，就正常吃饭"


# --------------------------------------------------------------------- 提示词构造

def test_extract_prompt_lists_slots_and_options():
    msgs = build_extract_messages(CONV, TPL)
    system = msgs[0]["content"]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert CONV in msgs[1]["content"]
    assert "chief" in system and "呕吐物性状" in system
    assert "未消化食物" in system, "选项型槽位必须把可选值列给模型"
    assert "照抄宠主原话" in system, "自由文本槽位必须要求照抄"


def test_extract_prompt_forbids_diagnosis():
    system = build_extract_messages(CONV, TPL)[0]["content"]
    assert "不要给任何诊断" in system


def test_extract_prompt_focus_keys():
    system = build_extract_messages(CONV, TPL, focus_keys=["chief"])[0]["content"]
    assert "chief" in system
    assert "呕吐物性状" not in system, "focus_keys 应缩小清单"


def test_summary_prompt_has_redlines():
    system = build_summary_messages(CONV, [])[0]["content"]
    assert "不得给诊断结论" in system
    assert "不得编造" in system
    assert "原样" in system, "宠主原话必须要求原样保留"


# --------------------------------------------------------------------- 严格解析：抽取

def test_parse_extract_plain_json():
    raw = '{"chief": {"state": "filled", "value": "呕吐"}, "vomit_freq": {"state": "filled", "value": "3–5 次"}}'
    got = parse_extract_response(raw, TPL)
    assert got["chief"]["value"] == "呕吐"
    assert got["chief"]["source"] == "llm"
    assert got["vomit_freq"]["value"] == "3–5 次"


def test_parse_extract_fenced_json():
    raw = '```json\n{"chief": {"state": "filled", "value": "呕吐"}}\n```'
    assert parse_extract_response(raw, TPL)["chief"]["value"] == "呕吐"


def test_parse_extract_drops_hallucinated_slot():
    raw = '{"chief": {"state": "filled", "value": "呕吐"}, "blood_type": {"state": "filled", "value": "A"}}'
    got = parse_extract_response(raw, TPL)
    assert "chief" in got
    assert "blood_type" not in got, "不存在的槽位必须丢弃"


def test_parse_extract_drops_value_outside_options():
    """选项型槽位只认精确命中 —— 模型不得自创选项值"""
    raw = '{"vomit_content": {"state": "filled", "value": "咖啡色带血丝"}}'
    got = parse_extract_response(raw, TPL)
    assert "vomit_content" not in got


def test_parse_extract_drops_invalid_state_or_empty_value():
    raw = '{"chief": {"state": "unknown", "value": "呕吐"}, "onset": {"state": "filled", "value": "  "}}'
    assert parse_extract_response(raw, TPL) == {}


def test_parse_extract_tolerates_vague_state():
    raw = '{"general": {"state": "vague", "value": "精神差，饮水情况没说"}}'
    assert parse_extract_response(raw, TPL)["general"]["state"] == "vague"


def test_parse_extract_bad_json_returns_empty():
    assert parse_extract_response("模型今天不想输出 JSON", TPL) == {}
    assert parse_extract_response("[1, 2, 3]", TPL) == {}


# --------------------------------------------------------------------- 严格解析：摘要

GOOD_SUMMARY = """
{
  "chief_complaint": "反复呕吐 1 天（24h 内 3–4 次）",
  "present_illness": {"onset": "昨晚起", "characteristics": "黄绿色液体伴泡沫",
                      "progression": "未见好转", "associated": "精神下降、饮水减少",
                      "treatment": "未用药，禁食一顿无效"},
  "past_history": {"chronic": "无", "surgery": "无", "vaccination": "三针已完成", "allergy": "无已知过敏"},
  "general": {"appetite": "下降", "water": "减少", "weight_change": "未测量"},
  "other_findings": [{"text": "最近老是舔自己的爪子，以前不这样", "source": "free_text"}],
  "ai_hints": [{"text": "呕吐伴精神下降，建议先排除误食与异物",
                "citations": ["knowledge/toxic_substances_handbook.md#x-1"]}],
  "suggested_questions": ["呕吐物是否含血", "腹部触诊与体温"]
}
"""


def test_parse_summary_normal():
    got = parse_summary_response(GOOD_SUMMARY)
    assert got["chief_complaint"].startswith("反复呕吐")
    assert got["present_illness"]["onset"] == "昨晚起"
    assert got["past_history"]["vaccination"] == "三针已完成"
    assert got["other_findings"][0]["text"].startswith("最近老是舔")
    assert got["ai_hints"][0]["citations"]
    assert got["suggested_questions"] == ["呕吐物是否含血", "腹部触诊与体温"]


def test_parse_summary_drops_diagnosis_hint():
    """🔴 红线：ai_hints 只描述待排查方向，出现诊断结论一律丢弃"""
    raw = """{"ai_hints": [
        {"text": "建议先排除误食与异物", "citations": []},
        {"text": "确诊为急性胃肠炎", "citations": []},
        {"text": "诊断为胰腺炎可能性大", "citations": []}
    ]}"""
    got = parse_summary_response(raw)
    assert len(got["ai_hints"]) == 1
    assert got["ai_hints"][0]["text"] == "建议先排除误食与异物"


def test_parse_summary_fills_missing_fields():
    got = parse_summary_response('{"chief_complaint": "呕吐 1 天"}')
    assert got["chief_complaint"] == "呕吐 1 天"
    assert got["present_illness"]["onset"] == ""
    assert got["ai_hints"] == [] and got["suggested_questions"] == []


def test_parse_summary_bad_json_returns_skeleton():
    got = parse_summary_response("不是 JSON")
    assert set(got) == {"chief_complaint", "present_illness", "past_history",
                        "general", "other_findings", "ai_hints", "suggested_questions"}
    assert got["chief_complaint"] == ""


def test_contains_diagnosis_helper():
    assert contains_diagnosis("确诊为胃炎")
    assert contains_diagnosis("建议诊断为异物")
    assert not contains_diagnosis("建议先排除误食与异物")


if __name__ == "__main__":
    test_extract_prompt_lists_slots_and_options()
    test_extract_prompt_forbids_diagnosis()
    test_extract_prompt_focus_keys()
    test_summary_prompt_has_redlines()
    test_parse_extract_plain_json()
    test_parse_extract_fenced_json()
    test_parse_extract_drops_hallucinated_slot()
    test_parse_extract_drops_value_outside_options()
    test_parse_extract_drops_invalid_state_or_empty_value()
    test_parse_extract_tolerates_vague_state()
    test_parse_extract_bad_json_returns_empty()
    test_parse_summary_normal()
    test_parse_summary_drops_diagnosis_hint()
    test_parse_summary_fills_missing_fields()
    test_parse_summary_bad_json_returns_skeleton()
    test_contains_diagnosis_helper()
    print("全部断言通过")

"""tests/test_record_draft.py：病历起草 —— 红线断言（无 diagnosis）+ 严格解析 + 端点行为。

全部离线：LLM 经 ``services.intake_service.set_llm`` 注入 Fake，不依赖网络。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi.testclient import TestClient

from app import app
from record_draft import build_draft_messages, disclaimer_for, parse_draft_response
from services.intake_service import set_llm


# --------------------------------------------------------------------- 解析层

def test_parse_keeps_whitelisted_fields_and_drops_diagnosis():
    """🔴 红线：模型幻觉出 diagnosis 键 → 必须被丢弃，输出里不存在这个字段。"""
    raw = json.dumps({
        "chief_complaint": "犬呕吐 1 天，逐渐加重",
        "diagnosis": "急性胃肠炎",                    # 越界输出 → 丢弃
        "exam_checklist": ["评估脱水程度", "腹部触诊"],
        "doctor_advice": "少量多次饮水，禁食 4-6 小时后观察",
    }, ensure_ascii=False)
    draft = parse_draft_response(raw)
    assert draft["chief_complaint"] == "犬呕吐 1 天，逐渐加重"
    assert draft["exam_checklist"] == ["评估脱水程度", "腹部触诊"]
    assert "diagnosis" not in draft
    assert "急性胃肠炎" not in json.dumps(draft, ensure_ascii=False)


def test_parse_drops_diagnosis_wording_in_content():
    """🔴 红线：内容层兜底 —— 医嘱/清单命中诊断措辞 → 整段丢弃。"""
    raw = json.dumps({
        "chief_complaint": "猫呕吐 2 天",
        "exam_checklist": ["评估脱水程度", "可确定是胰腺炎", "触诊腹部"],
        "doctor_advice": "建议确诊为胃肠炎后用药",
    }, ensure_ascii=False)
    draft = parse_draft_response(raw)
    assert draft["exam_checklist"] == ["评估脱水程度", "触诊腹部"]
    assert draft["doctor_advice"] == ""


def test_parse_garbage_returns_empty_not_crash():
    assert parse_draft_response("这不是JSON")["chief_complaint"] == ""
    assert parse_draft_response("[1,2,3]")["exam_checklist"] == []
    assert parse_draft_response("")["doctor_advice"] == ""


def test_parse_truncates_to_db_column_width():
    """主诉 > 255 字符 → 截断到列宽（否则落库直接报错）。"""
    raw = json.dumps({"chief_complaint": "吐" * 400,
                      "doctor_advice": "嘱" * 600}, ensure_ascii=False)
    draft = parse_draft_response(raw)
    assert len(draft["chief_complaint"]) == 255
    assert len(draft["doctor_advice"]) == 500


def test_parse_caps_checklist_items():
    raw = json.dumps({"exam_checklist": ["项目%d" % i for i in range(20)]},
                     ensure_ascii=False)
    assert len(parse_draft_response(raw)["exam_checklist"]) == 8


def test_disclaimer_stronger_on_emergency_words():
    assert "急症" in disclaimer_for("我家狗误食了巧克力")
    assert "急症" not in disclaimer_for("我家狗吐黄水")


def test_build_messages_contains_redline_and_data():
    msgs = build_draft_messages(
        summary={"chief_complaint": "呕吐 1 天",
                 "present_illness": {"onset": "昨晚", "progression": "越来越重"}},
        dialogue=[{"role": "user", "text": "它一直吐"}],
        archive={"allergy": "无已知过敏"},
        pet={"species": "犬", "breed": "泰迪"})
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    assert "不得给诊断" in system and "不得给处方" in system
    assert "越来越重" in user and "它一直吐" in user
    assert "allergy=无已知过敏" in user and "泰迪" in user


# --------------------------------------------------------------------- 端点层

def _summary_fixture():
    return {"chief_complaint": "犬呕吐 1 天，逐渐加重",
            "present_illness": {"onset": "昨晚", "characteristics": "吐黄水带泡沫",
                                 "progression": "越来越重", "associated": "", "treatment": ""}}


def test_endpoint_returns_draft_without_diagnosis():
    canned = json.dumps({
        "chief_complaint": "犬呕吐 1 天，逐渐加重，吐黄水带泡沫",
        "exam_checklist": ["评估脱水程度", "腹部触诊"],
        "doctor_advice": "禁食 4-6 小时后少量多次饮水",
        "diagnosis": "不该出现的字段",
    }, ensure_ascii=False)
    set_llm(lambda messages: canned)
    try:
        with TestClient(app) as c:
            r = c.post("/api/record/draft", json={
                "summary": _summary_fixture(),
                "dialogue": [{"role": "user", "text": "它从昨晚开始吐"}],
                "archive": {"allergy": "无"},
                "patient_context": {"species": "犬", "breed": "泰迪", "age_years": 3},
            })
            assert r.status_code == 200
            body = r.json()["data"]
            assert "diagnosis" not in body["draft"]
            assert body["draft"]["exam_checklist"] == ["评估脱水程度", "腹部触诊"]
            assert body["meta"]["generated_by"] == "vetclaw"
            assert body["meta"]["trace_id"]
    finally:
        set_llm(None)


def test_endpoint_503_when_llm_unavailable():
    set_llm(None)                                    # 关闭模型通道
    with TestClient(app) as c:
        r = c.post("/api/record/draft", json={"summary": _summary_fixture()})
        assert r.status_code == 503
        assert r.json()["code"] == "DRAFT_UNAVAILABLE"


def test_endpoint_503_when_llm_output_unparsable():
    set_llm(lambda messages: "抱歉，我无法输出 JSON。")   # 通道在跑但产出不可用
    try:
        with TestClient(app) as c:
            r = c.post("/api/record/draft", json={"summary": _summary_fixture()})
            assert r.status_code == 503
            assert "未成功" in r.json()["message"]
    finally:
        set_llm(None)


def test_endpoint_503_when_llm_raises():
    def _boom(messages):
        raise RuntimeError("network down")

    set_llm(_boom)
    try:
        with TestClient(app) as c:
            r = c.post("/api/record/draft", json={"summary": _summary_fixture()})
            assert r.status_code == 503
    finally:
        set_llm(None)


def test_endpoint_empty_summary_gets_empty_draft_rejected():
    """摘要/对话全空 → 模型即使输出也无法凭空起草：主诉为空 → 503（宁缺勿编）。"""
    set_llm(lambda messages: json.dumps(
        {"chief_complaint": "", "exam_checklist": [], "doctor_advice": ""},
        ensure_ascii=False))
    try:
        with TestClient(app) as c:
            assert c.post("/api/record/draft", json={"summary": {}}).status_code == 503
    finally:
        set_llm(None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

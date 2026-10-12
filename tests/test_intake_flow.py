"""tests/test_intake_flow.py：预问诊会话状态机断言（零网络 —— LLM 用可注入的假实现）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intake import load_template  # noqa: E402
from intake_flow import IntakeSession, build_summary_fallback, OPENING  # noqa: E402

TPL = load_template()
ARCHIVE = {"vaccine": "三针已完成", "past_disease": "无慢性病史",
           "medications": "无在用药", "allergy": "无已知过敏"}

ANSWERS = {
    "chief": "我家狗一直吐", "onset": "慢慢变严重的", "exposure": "没有，就正常吃饭",
    "vomit_content": "吐的是黄水，有点泡沫", "vomit_relation": "吃完没多久就吐",
    "vomit_kind": "有用力干呕", "vomit_freq": "三四次吧",
    "vomit_danger": "肚子还好，没有一直干呕", "vomit_associated": "没拉稀，精神差点",
    "progression": "越来越重", "treatment": "没给药，禁食了一顿没好转",
    "general": "精神差、水喝得少",
}
SUMMARY_JSON = """{
  "chief_complaint": "呕吐 1 天（24h 内 3–5 次）",
  "present_illness": {"onset": "昨晚起", "characteristics": "黄绿色液体伴泡沫",
                      "progression": "越来越重", "associated": "无腹泻、精神下降",
                      "treatment": "未用药，禁食一顿无效"},
  "past_history": {"chronic": "无", "surgery": "", "vaccination": "三针已完成", "allergy": "无已知过敏"},
  "general": {"appetite": "", "water": "减少", "weight_change": ""},
  "other_findings": [{"text": "最近老是舔自己的爪子", "source": "free_text"}],
  "ai_hints": [{"text": "建议先排除误食与异物", "citations": ["knowledge/x.md#y-1"]},
               {"text": "确诊为急性胃肠炎", "citations": []}],
  "suggested_questions": ["呕吐物是否含血"]
}"""


class FakeLLM:
    """按 system 提示词类型分流返回预置内容；可注入异常以验证降级。"""

    def __init__(self, extract="{}", summary=SUMMARY_JSON, boom=False):
        self.extract, self.summary, self.boom = extract, summary, boom
        self.calls = []

    def __call__(self, messages):
        system = messages[0]["content"]
        self.calls.append("summary" if "预问诊摘要" in system else "extract")
        if self.boom:
            raise RuntimeError("模型不可用")
        return self.summary if "预问诊摘要" in system else self.extract


class FakeGuard:
    """假守卫：命中给定关键词即返回 emergency。"""

    def __init__(self, keyword="巧克力"):
        self.keyword = keyword

    def check(self, query, patient_context=None):
        if self.keyword in query:
            return {"action": "emergency", "toxin": "巧克力", "species": "狗", "chunk_id": "x#1"}
        return {"action": "pass"}


def _drive(session, max_steps=12):
    """驱动整段会话直到 done：按引擎实际提问（结构化 questions 里的 key）作答。"""
    turns = 0
    out = session.opening()
    while not out.get("done") and turns < max_steps:
        turns += 1
        keys = [q["key"] for q in out.get("questions", [])]
        out = session.handle(" ".join(ANSWERS.get(k, "") for k in keys))
    return out, turns


# ------------------------------------------------------------------ 开场

def test_opening_has_message_and_first_questions():
    s = IntakeSession(TPL, llm=None, archive=ARCHIVE)
    out = s.opening()
    assert out["mode"] == "intake" and out.get("done") is not True
    assert "预问诊助手" in out["message"]
    assert "不构成诊断" in out["message"], "开场必须带免责"
    assert "可以像这样说" in out["message"], "开场必须给示例"
    assert out["questions"], "开场应带第一批问题"
    assert all("key" in q and "ask" in q for q in out["questions"]), \
        "问题必须是结构化的（含 key）—— 调用方要靠 key 驱动抽取门控"


# ------------------------------------------------------------------ 纯规则模式（无 LLM）

def test_rule_only_session_completes_with_fallback_summary():
    s = IntakeSession(TPL, llm=None, archive=ARCHIVE)
    out, turns = _drive(s)
    assert out["done"] is True
    assert turns <= TPL["meta"]["rules"]["max_turns"]
    intake = out["intake"]
    assert intake["incomplete"] == [], "必填应齐备"
    assert intake["interrupted"] is False
    assert intake["ai_hints"] == [], "无模型时不编造提示"
    assert intake["summary"]["chief_complaint"], "兜底摘要也要有主诉"


def test_fallback_summary_is_built_from_slots():
    s = IntakeSession(TPL, llm=None, archive=ARCHIVE)
    _drive(s)
    summary = build_summary_fallback(s.state, TPL, s.conversation)
    assert "呕吐" in summary["chief_complaint"]
    assert summary["past_history"]["vaccination"] == "三针已完成"
    assert summary["ai_hints"] == []


# ------------------------------------------------------------------ 带 LLM 通道

def test_llm_channel_fills_free_text_and_summary():
    llm = FakeLLM(extract='{"other_findings": {"state": "filled", "value": "最近老是舔爪子"}}')
    s = IntakeSession(TPL, llm=llm, archive=ARCHIVE)
    out, _ = _drive(s)
    assert out["done"] is True
    assert llm.calls.count("summary") == 1
    intake = out["intake"]
    assert intake["summary"]["chief_complaint"].startswith("呕吐")
    assert intake["summary"]["other_findings"][0]["text"].startswith("最近老是舔")


def test_llm_summary_drops_diagnosis_and_keeps_original_text():
    """红线在流程层同样成立：含诊断结论的 ai_hints 被丢，原话保留"""
    s = IntakeSession(TPL, llm=FakeLLM(), archive=ARCHIVE)
    out, _ = _drive(s)
    hints = out["intake"]["ai_hints"]
    assert len(hints) == 1 and "确诊" not in hints[0]["text"]
    assert "舔" in out["intake"]["summary"]["other_findings"][0]["text"]


def test_llm_failure_degrades_to_fallback():
    s = IntakeSession(TPL, llm=FakeLLM(boom=True), archive=ARCHIVE)
    out, _ = _drive(s)
    assert out["done"] is True, "模型挂了也要能收尾"
    assert out["intake"]["summary"]["chief_complaint"], "应退化为兜底摘要"


def test_llm_garbage_output_degrades():
    s = IntakeSession(TPL, llm=FakeLLM(extract="不是 JSON", summary="也不是 JSON"), archive=ARCHIVE)
    out, _ = _drive(s)
    assert out["done"] is True
    assert out["intake"]["summary"]["chief_complaint"]


# ------------------------------------------------------------------ 急症中断

def test_emergency_interrupts_and_skips_questions():
    s = IntakeSession(TPL, llm=None, archive=ARCHIVE, guard=FakeGuard())
    s.opening()
    out = s.handle("它好像吃了巧克力")
    assert out["done"] is True, "命中急症应立即收尾，不再追问"
    assert out["questions"] == []
    intake = out["intake"]
    assert intake["interrupted"] is True
    assert intake["emergency"]["toxin"] == "巧克力"
    assert "急症" in intake["ai_hints"][0]["text"]


def test_no_guard_no_interrupt():
    s = IntakeSession(TPL, llm=None, archive=ARCHIVE, guard=None)
    s.opening()
    out = s.handle("它好像吃了巧克力")
    assert out.get("done") is not True, "无守卫时不中断（由上层负责）"
    assert out["questions"]


# ------------------------------------------------------------------ 轮数上限与如实标注

def test_max_turns_marks_incomplete_honestly():
    """宠主一直答非所问 → 达上限时必须**如实**列出未采集项，而不是假装问全了"""
    s = IntakeSession(TPL, llm=None, archive=ARCHIVE, max_turns=2)
    s.opening()
    out = s.handle("不知道")
    out = s.handle("不清楚")
    assert out["done"] is True
    assert out["intake"]["incomplete"], "未采集的必填项必须如实列出"
    assert len(out["intake"]["incomplete"]) > 0


# ------------------------------------------------------------------ 契约载荷形状

def test_snapshot_matches_contract_shape():
    s = IntakeSession(TPL, llm=FakeLLM(), archive=ARCHIVE)
    _drive(s)
    snap = s.snapshot()
    assert set(snap) == {"slots", "summary", "ai_hints", "suggested_questions",
                         "free_text", "incomplete", "interrupted", "emergency"}
    assert set(snap["summary"]) >= {"chief_complaint", "present_illness", "past_history",
                                    "general", "other_findings", "ai_hints",
                                    "suggested_questions"}
    sample = next(iter(snap["slots"].values()))
    assert set(sample) == {"state", "value", "source"}


def test_state_round_trip_keeps_progress():
    """续传：本服务无状态 —— 导出 state、重建会话后必须能接着走，不丢进度。"""
    s = IntakeSession(TPL, llm=None, archive=ARCHIVE)
    s.opening()
    s.handle(ANSWERS["chief"])
    snapshot = s.state.get("chief", {}).get("value")

    restored = IntakeSession.from_state(s.to_state(), llm=None, archive=ARCHIVE)
    assert restored.state["chief"]["value"] == snapshot
    assert restored.conversation == s.conversation
    assert restored.turns == s.turns
    # 接着走：重建后的会话应能继续推进（不从头开始）
    out = restored.handle(ANSWERS["exposure"])
    assert out.get("done") is not True
    assert len(restored.conversation) > len(s.conversation)


def test_from_state_none_starts_fresh():
    s = IntakeSession.from_state(None, llm=None, archive=ARCHIVE)
    assert s.turns == 0 and s.conversation == ""
    assert s.state["chief"]["state"] == "missing"


def test_state_carries_emergency_interrupt():
    s = IntakeSession(TPL, llm=None, archive=ARCHIVE, guard=FakeGuard())
    s.opening()
    out = s.handle("它好像吃了巧克力")
    restored = IntakeSession.from_state(s.to_state(), llm=None, archive=ARCHIVE, guard=FakeGuard())
    assert restored.interrupted is True
    assert restored.emergency["toxin"] == "巧克力"
    assert restored.snapshot()["interrupted"] is True


if __name__ == "__main__":
    test_opening_has_message_and_first_questions()
    test_rule_only_session_completes_with_fallback_summary()
    test_fallback_summary_is_built_from_slots()
    test_llm_channel_fills_free_text_and_summary()
    test_llm_summary_drops_diagnosis_and_keeps_original_text()
    test_llm_failure_degrades_to_fallback()
    test_llm_garbage_output_degrades()
    test_emergency_interrupts_and_skips_questions()
    test_no_guard_no_interrupt()
    test_max_turns_marks_incomplete_honestly()
    test_snapshot_matches_contract_shape()
    print("全部断言通过")

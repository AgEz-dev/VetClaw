"""tests/test_agent_trace.py：B3 轨迹表 —— 事件覆盖、耗时、脱敏与按 case 隔离。

轨迹是**观测面**：它必须能回答"这次任务调了哪些工具、成败、多慢、最终结论"，
同时又不能把病例原文、工具结果正文或知识库片段抄进表里。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from a2a.agents import AgentContext, AgentOutcome, consult_stream  # noqa: E402
from a2a.contract import CaseState, CaseTask, Step, TaskStatus, Verdict  # noqa: E402
from a2a.store import CaseStore, digest_args  # noqa: E402
from a2a.stream import INTERNAL_OUTCOME  # noqa: E402
from a2a.supervisor import Supervisor  # noqa: E402
from a2a.trace import prompt_version  # noqa: E402
from case_context import get_case_id  # noqa: E402

QUERY = "我家猫误食了百合花怎么办"
SENSITIVE = "豆豆|青霉素|百合花"


def _task(case_id="case-trace-1", step=Step.CONSULT, prompt=QUERY):
    return CaseTask(step=step, case_id=case_id,
                    idempotency_key=f"idem-{case_id}", payload={"prompt": prompt})


def _run(streamer, task=None, ctx=None):
    store = CaseStore(":memory:")
    sup = Supervisor(store, ctx=ctx or AgentContext(),
                     stream_agents={Step.CONSULT: streamer})
    task = task or _task()
    events = list(sup.submit_stream(task))
    return events, store, task


def _dump(rows):
    return json.dumps(rows, ensure_ascii=False)


# --------------------------------------------------------------------- 工具事件

def test_tool_events_recorded_with_digest_and_elapsed():
    def streamer(case, task, ctx):
        yield {"event": "tool_call", "data": {
            "id": "c1", "name": "get_pet_profile",
            "arguments": {"pet_id": 7, "query": QUERY}}}
        yield {"event": "tool_result", "data": {
            "id": "c1",
            "content": json.dumps({"ok": True,
                                   "data": {"petName": "豆豆", "allergies": "青霉素"},
                                   "error_code": None, "elapsed_ms": 5},
                                  ensure_ascii=False)}}
        yield {"event": INTERNAL_OUTCOME, "data": AgentOutcome(
            status=TaskStatus.OK, payload={"answer": "请立即就医"})}

    events, store, task = _run(streamer)
    rows = store.list_trace(task.case_id)

    assert [r["event"] for r in rows] == ["tool_call", "tool_result", "verdict"]
    call, result, verdict = rows

    assert call["tool_name"] == "get_pet_profile"
    assert call["args_digest"] == digest_args({"pet_id": 7, "query": QUERY})
    assert len(call["args_digest"]) == 64

    assert result["tool_name"] == "get_pet_profile"
    assert result["ok"] == 1
    assert result["error_code"] is None
    assert result["elapsed_ms"] is not None and result["elapsed_ms"] >= 0

    assert verdict["verdict"] == "none"
    assert verdict["ok"] == 1
    assert [r["seq"] for r in rows] == [1, 2, 3]

    dumped = _dump(rows)
    assert QUERY not in dumped, "病例原文不得进入轨迹"
    assert "豆豆" not in dumped and "青霉素" not in dumped, "工具结果正文不得进入轨迹"


def test_tool_failure_envelope_recorded():
    def streamer(case, task, ctx):
        yield {"event": "tool_call", "data": {
            "id": "c9", "name": "get_medication_history", "arguments": {"pet_id": 7}}}
        yield {"event": "tool_result", "data": {
            "id": "c9",
            "content": json.dumps({"ok": False, "data": None,
                                   "error_code": "HOSPITAL_TIMEOUT",
                                   "elapsed_ms": 3001})}}
        yield {"event": INTERNAL_OUTCOME, "data": AgentOutcome(
            status=TaskStatus.OK, payload={"answer": "查不到"})}

    _, store, task = _run(streamer)
    result = store.list_trace(task.case_id)[1]
    assert result["event"] == "tool_result"
    assert result["ok"] == 0
    assert result["error_code"] == "HOSPITAL_TIMEOUT"


def test_legacy_tool_result_string_classified():
    def streamer(case, task, ctx):
        yield {"event": "tool_call", "data": {
            "id": "a", "name": "search_knowledge_base", "arguments": {"query": QUERY}}}
        yield {"event": "tool_result", "data": {
            "id": "a", "content": "工具执行失败: ValueError: boom"}}
        yield {"event": "tool_call", "data": {
            "id": "b", "name": "search_knowledge_base", "arguments": {"query": QUERY}}}
        yield {"event": "tool_result", "data": {
            "id": "b", "content": "【知识库检索结果】\n[1] knowledge/x.md#1\n正文"}}
        yield {"event": "tool_call", "data": {
            "id": "c", "name": "get_pet_profile", "arguments": {"pet_id": 7}}}
        yield {"event": "tool_result", "data": {
            "id": "c", "content": "工具执行失败: ValueError: 参数不在白名单: ['admin']"}}
        yield {"event": INTERNAL_OUTCOME, "data": AgentOutcome(
            status=TaskStatus.OK, payload={"answer": "好的"})}

    _, store, task = _run(streamer)
    rows = [r for r in store.list_trace(task.case_id) if r["event"] == "tool_result"]
    assert rows[0]["ok"] == 0 and rows[0]["error_code"] == "TOOL_FAILED"
    assert rows[1]["ok"] == 1 and rows[1]["error_code"] is None
    assert rows[2]["error_code"] == "BAD_ARGUMENTS", "模型编造参数必须可与工具故障区分"


# --------------------------------------------------------------------- 前置分诊

class _EmergencyGuard:
    def check(self, query, patient=None):
        return {"action": "emergency", "toxin": "百合", "species": "猫",
                "chunk_id": "knowledge/species_drug_safety_matrix.md#百合-1",
                "slots": {}}

    def emergency_message(self, toxin, species, chunk_id, slots):
        return "请立即就医"


def test_pretriage_handoff_traced_and_agent_short_circuited():
    called = {"n": 0}

    def streamer(case, task, ctx):
        called["n"] += 1
        yield {"event": INTERNAL_OUTCOME, "data": AgentOutcome(
            status=TaskStatus.OK, payload={"answer": "不该执行"})}

    ctx = AgentContext(guard=_EmergencyGuard())
    events, store, task = _run(streamer, ctx=ctx)
    rows = store.list_trace(task.case_id)

    assert called["n"] == 0, "BLOCK 必须短路，原能力位不得执行"
    assert [r["event"] for r in rows] == ["handoff", "verdict"]
    assert rows[0]["agent"] == "triage"
    assert rows[0]["verdict"] == "block"
    assert rows[1]["verdict"] == "block"
    assert QUERY not in _dump(rows)


# --------------------------------------------------------------------- 其它

def test_trace_seq_isolated_per_case():
    def streamer(case, task, ctx):
        yield {"event": INTERNAL_OUTCOME, "data": AgentOutcome(
            status=TaskStatus.OK, payload={"answer": "ok"})}

    store = CaseStore(":memory:")
    for case_id in ("case-a", "case-b"):
        sup = Supervisor(store, ctx=AgentContext(),
                         stream_agents={Step.CONSULT: streamer})
        list(sup.submit_stream(CaseTask(
            step=Step.CONSULT, case_id=case_id,
            idempotency_key=f"idem-{case_id}", payload={"prompt": QUERY})))

    assert [r["seq"] for r in store.list_trace("case-a")] == [1]
    assert [r["seq"] for r in store.list_trace("case-b")] == [1]


def test_error_path_traced():
    def broken(case, task, ctx):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    events, store, task = _run(broken)
    assert events[-1]["event"] == "error"
    row = store.list_trace(task.case_id)[-1]
    assert row["event"] == "error"
    assert row["error_code"] == "AGENT_ERROR"
    assert row["ok"] == 0


def test_execute_tool_sets_case_id_for_call_and_clears_after():
    from types import SimpleNamespace

    from core_agent import execute_tool
    from tools import ToolRegistry

    reg = ToolRegistry()

    @reg.tool
    def probe() -> str:
        """读取当前 case。"""
        return get_case_id()

    tc = SimpleNamespace(function=SimpleNamespace(name="probe", arguments="{}"))
    assert execute_tool(reg, tc, "case-x") == "case-x"
    assert get_case_id() == "", "工具调用结束必须清理，避免串到下一个任务"


def test_consult_stream_passes_case_id_to_engine():
    seen = {}

    def chat(prompt, history=None, patient_context=None, case_id=""):
        seen["case_id"] = case_id
        yield {"event": "done", "data": {"answer": "ok", "citations": []}}

    list(consult_stream(CaseState(case_id="case-x", patient={}),
                        _task(case_id="case-x"), AgentContext(chat_stream=chat)))
    assert seen["case_id"] == "case-x", "问答引擎必须拿到 case，写工具才能关联工单"


def test_prompt_version_is_short_hash_not_prompt_text():
    version = prompt_version("你是 VetClaw，一个宠物健康分诊助手。")
    assert version.startswith("p_") and len(version) == 10
    assert "VetClaw" not in version
    assert prompt_version(None) == ""


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("全部断言通过")

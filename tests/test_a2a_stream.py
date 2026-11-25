"""tests/test_a2a_stream.py：A2A 流式能力 —— 帧协议 / 问答流式能力位 / 端点语义。

全部离线：问答流由 `AgentContext.chat_stream` 注入替身生成器，不需要模型。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi.testclient import TestClient

from a2a.agents import AgentContext
from a2a.contract import CaseTask, Step, TaskStatus
from a2a.store import CaseStore
from a2a.stream import done_frame, error_frame, start_frame, to_sse
from a2a.supervisor import IdempotencyConflict, Supervisor
from prescription_review import PrescriptionReviewer

TERMINAL = {"done", "error"}


def _fake_chat(tokens=("你", "好", "呀"), citations=("knowledge/x.md#a-1",)):
    """替身问答流：形如 ReActAgent.run_stream 的事件序列。"""

    def stream(prompt, history, patient_context, case_id=""):
        yield {"event": "thought", "data": {"stage": "thinking"}}
        yield {"event": "tool_call", "data": {"id": "c1", "name": "search_knowledge_base",
                                              "arguments": {"query": prompt}}}
        yield {"event": "tool_result", "data": {"id": "c1", "content": "【知识库检索结果】…"}}
        for t in tokens:
            yield {"event": "token", "data": {"text": t}}
        yield {"event": "done", "data": {"answer": "".join(tokens),
                                         "citations": list(citations)}}

    return stream


def _ctx(**kwargs):
    base = dict(reviewer=PrescriptionReviewer(), chat_stream=_fake_chat())
    base.update(kwargs)
    return AgentContext(**base)


@pytest.fixture()
def store():
    s = CaseStore(":memory:")
    yield s
    s.close()


def _task(key="k1", case_id="appt-1", prompt="狗能吃巧克力吗", **kw):
    return CaseTask(step=Step.CONSULT, case_id=case_id,
                    idempotency_key=key, payload={"prompt": prompt}, **kw)


def _frames(store, task, ctx=None):
    sup = Supervisor(store, ctx or _ctx())
    return list(sup.submit_stream(task))


def _terminal_count(frames):
    return sum(1 for f in frames if f["event"] in TERMINAL)


# --------------------------------------------------------------------- 帧协议

def test_to_sse_escapes_newlines_as_continuation():
    """SSE 规范：data 内换行必须转续行，否则破坏空行分帧。"""
    out = to_sse("token", {"text": "第一行\n第二行"})
    assert "data: " in out
    assert out.count("\n\n") == 1, "一帧只应有一处分帧空行"
    assert "\ndata: " in out


def test_frame_stays_single_data_line_even_with_newlines():
    """分帧安全的核心不变量：**无论内容里有什么，一帧只产生一个 data 行**。

    json.dumps 会把内联换行转义成 `\\n`（两个字符），因此 SSE 的"空行分帧"
    不会被内容破坏 —— 这正是 `to_sse` 里那次 replace 存在的价值（对非 JSON 内容兜底）。
    """
    out = to_sse("token", {"text": "a\nb"})
    lines = [ln for ln in out.split("\n") if ln.startswith("data: ")]
    assert len(lines) == 1
    import json
    payload = json.loads(lines[0][len("data: "):])
    assert payload["text"] == "a\nb", "转义后必须能原样还原"


def test_start_frame_carries_binding_fields():
    task = _task()
    f = start_frame(task)
    assert f["event"] == "start"
    assert f["data"]["case_id"] == "appt-1"
    assert f["data"]["step"] == "consult"
    assert f["data"]["deduped"] is False


def test_done_frame_keeps_legacy_fields():
    """done 必须是旧 done 事件的**超集**：旧中继只读 answer/citations。"""
    from a2a.contract import TaskResult
    r = TaskResult(case_id="c", task_id="t", step=Step.CONSULT,
                   status=TaskStatus.OK, payload={"answer": "答复"},
                   citations=["k#1"])
    d = done_frame(r)["data"]
    assert d["answer"] == "答复" and d["citations"] == ["k#1"]
    assert d["case_id"] == "c" and d["verdict"] == "none"


def test_error_frame_has_code_and_trace():
    e = error_frame("X", "出错了", "trace-1")["data"]
    assert e == {"code": "X", "message": "出错了", "trace_id": "trace-1"}


# --------------------------------------------------------------------- 问答流式能力位

def test_stream_forwards_events_and_ends_with_done(store):
    frames = _frames(store, _task())
    names = [f["event"] for f in frames]
    assert names[0] == "start"
    assert names[-1] == "done"
    assert _terminal_count(frames) == 1, "必须有且只有一个终态"

    mids = names[1:-1]
    assert "thought" in mids and "tool_call" in mids and "tool_result" in mids
    assert any(f["event"] == "token" for f in frames)

    done = frames[-1]["data"]
    assert done["answer"] == "你好呀"
    assert done["citations"] == ["knowledge/x.md#a-1"]


def test_stream_records_case_and_idempotency(store):
    sup = Supervisor(store, _ctx())
    list(sup.submit_stream(_task(key="k-rec")))
    case = sup.get_case("appt-1")
    assert case is not None
    assert Step.CONSULT.value in case.steps
    assert store.get_result("k-rec") is not None


def test_stream_replay_emits_stored_answer_once(store):
    """回放语义：**不重放 token**，把已存答案整段吐出 + done(deduped)。"""
    first = _frames(store, _task(key="k-replay"))
    assert first[-1]["event"] == "done"
    assert first[-1]["data"]["deduped"] is False

    second = _frames(store, _task(key="k-replay"))
    events = [f["event"] for f in second]
    assert events[0] == "start" and second[0]["data"]["deduped"] is True
    assert events[-1] == "done"
    # 回放只有 1 个 token 帧（整段答案），不是原来的 3 个
    assert events.count("token") == 1
    assert second[-2]["data"]["text"] == "你好呀"
    assert _terminal_count(second) == 1


def test_stream_need_more_is_terminal_done_with_status(store):
    """`NEED_MORE` 不是失败：流式下以 `done` + `status=need_more` 收尾。

    这与同步端点一致（HTTP 200 + status=need_more），让客户端只需判断
    「终态是 done 还是 error」，不必为每种 status 写一条分支。
    """
    frames = _frames(store, _task(prompt=""))
    assert frames[-1]["event"] == "done"
    assert frames[-1]["data"]["status"] == "need_more"
    assert _terminal_count(frames) == 1


def test_stream_without_chat_is_unavailable(store):
    frames = _frames(store, _task(), ctx=_ctx(chat_stream=None))
    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["code"] == "CHAT_UNAVAILABLE"


def test_stream_maps_upstream_error_to_error_frame(store):
    def failing(prompt, history, patient_context, case_id=""):
        yield {"event": "thought", "data": {"stage": "thinking"}}
        yield {"event": "error", "data": {"code": "MODEL_TIMEOUT", "message": "超时"}}

    frames = _frames(store, _task(), ctx=_ctx(chat_stream=failing))
    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["code"] == "MODEL_TIMEOUT"
    assert _terminal_count(frames) == 1


def test_stream_without_terminal_event_is_failure(store):
    """上游没给终态 → 必须报错，绝不能假装成功（客户端会一直等）。"""

    def silent(prompt, history, patient_context, case_id=""):
        yield {"event": "token", "data": {"text": "半"}}

    frames = _frames(store, _task(), ctx=_ctx(chat_stream=silent))
    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["code"] == "NO_TERMINAL_EVENT"


def test_stream_upstream_exception_is_error_frame(store):
    def boom(prompt, history, patient_context, case_id=""):
        raise RuntimeError("boom")
        yield  # 让函数成为生成器

    frames = _frames(store, _task(), ctx=_ctx(chat_stream=boom))
    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["code"] == "AGENT_ERROR"


def test_stream_failed_result_is_not_memoized(store):
    """失败不记忆：同一幂等键的第二次调用必须真的重跑。"""
    sup = Supervisor(store, _ctx(chat_stream=None))
    list(sup.submit_stream(_task(key="k-fail")))
    assert store.get_result("k-fail") is None, "失败不得写入幂等表"


# --------------------------------------------------------------------- 端点语义

def test_stream_step_not_streamable(store):
    """同步专属能力位打到流式端点 → 明确指引，而不是含糊失败。"""
    sup = Supervisor(store, _ctx())
    task = CaseTask(step=Step.REVIEW, case_id="c1", idempotency_key="k-r",
                    payload={"prescription": {"items": [{"drug_name": "速诺"}]}})
    frames = list(sup.submit_stream(task))
    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["code"] == "STEP_NOT_STREAMABLE"


def test_sync_endpoint_rejects_stream_only_step(store):
    """反向：流式专属能力位打到同步端点，也要给明确指引。"""
    sup = Supervisor(store, _ctx())
    r = sup.submit(_task(key="k-sync"))
    assert r.status is TaskStatus.FAILED
    assert r.payload.get("code") == "STEP_NOT_STREAMABLE"


def test_assert_no_conflict_raises_before_stream(store):
    """冲突必须在**握手前**抛出，否则只能退化成流内 error 帧。"""
    sup = Supervisor(store, _ctx())
    list(sup.submit_stream(_task(key="k-conf")))
    with pytest.raises(IdempotencyConflict):
        sup.assert_no_conflict(_task(key="k-conf", prompt="换了内容"))


def test_stream_deadline_exceeded(store):
    import time
    frames = _frames(store, _task(key="k-dead", deadline_at=time.time() - 1))
    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["code"] == "DEADLINE_EXCEEDED"


# --------------------------------------------------------------------- HTTP

@pytest.fixture()
def client(store):
    from a2a.routes import set_supervisor
    from app import app
    set_supervisor(Supervisor(store, _ctx()))
    yield TestClient(app)
    set_supervisor(None)


def test_api_stream_returns_event_stream(client):
    body = {"step": "consult", "case_id": "appt-h", "idempotency_key": "api-s1",
            "payload": {"prompt": "狗能吃巧克力吗"}}
    resp = client.post("/a2a/task/stream", json=body)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    assert "event: start" in text
    assert "event: token" in text
    assert "event: done" in text
    assert text.count("event: done") == 1


def test_api_stream_conflict_is_http_409(client):
    body = {"step": "consult", "case_id": "appt-h2", "idempotency_key": "api-s2",
            "payload": {"prompt": "狗能吃巧克力吗"}}
    assert client.post("/a2a/task/stream", json=body).status_code == 200
    changed = dict(body)
    changed["payload"] = {"prompt": "换了问题"}
    resp = client.post("/a2a/task/stream", json=changed)
    assert resp.status_code == 409, "握手前必须能返回真 409"
    assert resp.json()["code"] == "IDEMPOTENCY_CONFLICT"


def test_api_stream_unknown_step_rejected(client):
    resp = client.post("/a2a/task/stream",
                       json={"step": "hack", "case_id": "c",
                             "idempotency_key": "k", "payload": {}})
    assert resp.status_code == 422

"""tests/test_usage_trace.py：E 批用量字段与反馈导出。

覆盖：
- 真实 usage 优先、缺失时估算并标记 estimated（双引擎共用口径）；
- on_usage 回调不新增 SSE 事件；
- supervisor 把 token / 工具次数 / 耗时写进 verdict 轨迹；
- 反馈 JSONL 导出客户端（导出即消费、绝不重试）与本地合并去重。
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from a2a.agents import AgentContext, AgentOutcome, consult_stream  # noqa: E402
from a2a.contract import CaseState, CaseTask, Step, TaskStatus  # noqa: E402
from a2a.store import CaseStore  # noqa: E402
from a2a.stream import INTERNAL_OUTCOME  # noqa: E402
from a2a.supervisor import Supervisor  # noqa: E402
from core_agent import ReActAgent, merge_usage  # noqa: E402
from hospital_client import HospitalClient, HospitalClientError  # noqa: E402
from scripts.export_feedback import merge_samples, parse_jsonl  # noqa: E402
from scripts.show_trace import main as show_trace_main  # noqa: E402
from tools import ToolRegistry  # noqa: E402


# --------------------------------------------------------------------- 用量归一

def _usage_obj(prompt, completion):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


def test_merge_usage_prefers_real_usage():
    usage = {}
    merge_usage(usage, messages=[{"role": "user", "content": "你好"}],
                text_parts=["答案"], tool_parts={},
                chunk_usage=_usage_obj(11, 7))
    assert usage == {"input_tokens": 11, "output_tokens": 7}
    assert "estimated" not in usage

    merge_usage(usage, messages=[], text_parts=[], tool_parts={},
                chunk_usage=_usage_obj(3, 2))
    assert usage == {"input_tokens": 14, "output_tokens": 9}


def test_merge_usage_estimates_and_flags():
    usage = {}
    merge_usage(usage, messages=[{"role": "system", "content": "系统提示"},
                                 {"role": "user", "content": "问题"}],
                text_parts=["一段回答"],
                tool_parts={0: {"name": "search_knowledge_base",
                                "arguments": '{"query": "x"}'}},
                chunk_usage=None)
    assert usage["estimated"] is True
    assert usage["input_tokens"] > 0
    # 文本 + 工具名 + 工具参数都要计入输出侧
    assert usage["output_tokens"] > len("一段回答") // 4


class _FakeCompletions:
    def __init__(self, turns):
        self._turns = [list(turn) for turn in turns]

    def create(self, **kwargs):
        if not self._turns:
            raise AssertionError("脚本已耗尽：模型被调用的次数超出预期")
        return iter(self._turns.pop(0))


class _FakeClient:
    def __init__(self, turns):
        self.chat = SimpleNamespace(completions=_FakeCompletions(turns))


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _chunk(content=None, tool_calls=None, usage=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=_delta(content, tool_calls))],
                           usage=usage)


def _usage_only_chunk(prompt, completion):
    return SimpleNamespace(choices=[], usage=_usage_obj(prompt, completion))


def _agent(turns):
    reg = ToolRegistry()

    @reg.tool
    def ping() -> str:
        """ping。"""
        return "pong"

    return ReActAgent(_FakeClient(turns), reg, model="fake", max_steps=3), reg


def test_react_on_usage_real_usage_and_no_extra_events():
    tool_call = SimpleNamespace(index=0, id="c1",
                                function=SimpleNamespace(name="ping", arguments="{}"))
    agent, _ = _agent([
        [_chunk(tool_calls=[tool_call], usage=_usage_obj(10, 2))],
        [_chunk(content="好了"), _usage_only_chunk(21, 9)],
    ])
    captured = {}
    events = list(agent.run_stream("hi", on_usage=captured.update))

    assert captured == {"input_tokens": 31, "output_tokens": 11, "tool_count": 1}
    kinds = {e["event"] for e in events}
    assert kinds <= {"thought", "token", "tool_call", "tool_result", "done", "error"}


def test_react_on_usage_estimated_when_stream_has_no_usage():
    agent, _ = _agent([[_chunk(content="直接回答")]])
    captured = {}
    list(agent.run_stream("你好", on_usage=captured.update))

    assert captured["estimated"] is True
    assert captured["input_tokens"] > 0
    assert captured["output_tokens"] > 0
    assert captured["tool_count"] == 0


# --------------------------------------------------------------------- 编排层轨迹

def _task(case_id="case-usage-1"):
    return CaseTask(step=Step.CONSULT, case_id=case_id,
                    idempotency_key=f"idem-{case_id}",
                    payload={"prompt": "我的猫吐了"})


def test_supervisor_writes_usage_fields_to_verdict_trace():
    def streamer(case, task, ctx):
        yield {"event": "tool_call", "data": {"id": "t1", "name": "get_pet_profile",
                                              "arguments": {"pet_id": 1}}}
        yield {"event": "tool_result", "data": {"id": "t1", "content": '{"ok": true}'}}
        yield {"event": INTERNAL_OUTCOME, "data": AgentOutcome(
            status=TaskStatus.OK, payload={"answer": "请就医"},
            usage={"input_tokens": 31, "output_tokens": 12,
                   "tool_count": 1, "estimated": True})}

    store = CaseStore(":memory:")
    sup = Supervisor(store, ctx=AgentContext(),
                     stream_agents={Step.CONSULT: streamer})
    task = _task()
    list(sup.submit_stream(task))

    verdict = [r for r in store.list_trace(task.case_id) if r["event"] == "verdict"][-1]
    assert verdict["input_tokens"] == 31
    assert verdict["output_tokens"] == 12
    assert verdict["tool_count"] == 1
    assert verdict["usage_estimated"] == 1
    assert verdict["elapsed_ms"] is not None and verdict["elapsed_ms"] >= 0


def test_consult_stream_estimates_when_engine_has_no_on_usage():
    def chat_stream(prompt, history, patient_context, case_id=""):
        yield {"event": "token", "data": {"text": "答案"}}
        yield {"event": "done", "data": {"answer": "答案", "citations": []}}

    case = CaseState(case_id="c1")
    task = CaseTask(step=Step.CONSULT, case_id="c1", idempotency_key="k1",
                    payload={"prompt": "你好"})
    ctx = AgentContext(chat_stream=chat_stream)
    outcomes = [ev["data"] for ev in consult_stream(case, task, ctx)
                if ev["event"] == INTERNAL_OUTCOME]
    assert outcomes[0].usage["estimated"] is True
    assert outcomes[0].usage["output_tokens"] > 0
    assert outcomes[0].usage["tool_count"] == 0


# --------------------------------------------------------------------- 反馈导出

SECRET = "test-secret"


def _client(handler):
    return HospitalClient(base_url="http://hospital.test", secret=SECRET,
                          transport=httpx.MockTransport(handler))


def test_export_feedback_client_returns_raw_jsonl_once():
    calls = []
    body = '{"id": "FB-1", "query": "猫吐了", "expected_behavior": "answer"}\n'

    def handler(request):
        calls.append(request)
        assert request.method == "POST"
        assert request.url.path == "/internal/ai/feedback/export"
        assert request.headers["X-Internal-Secret"] == SECRET
        return httpx.Response(200, text=body,
                              headers={"Content-Type": "application/x-ndjson"})

    client = _client(handler)
    assert client.export_feedback_jsonl() == body
    assert len(calls) == 1


def test_export_feedback_client_does_not_retry_5xx():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"message": "busy"})

    client = _client(handler)
    try:
        client.export_feedback_jsonl()
        raise AssertionError("5xx 必须抛出 HospitalClientError")
    except HospitalClientError as exc:
        assert exc.error_code == "HOSPITAL_UNAVAILABLE"
    assert len(calls) == 1, "导出即消费，失败不得重试"


def test_export_feedback_requires_secret():
    client = HospitalClient(base_url="http://hospital.test", secret="",
                            transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    try:
        client.export_feedback_jsonl()
        raise AssertionError("缺密钥必须失败")
    except HospitalClientError as exc:
        assert exc.error_code == "HOSPITAL_NOT_CONFIGURED"


def test_parse_and_merge_dedup(tmp_path):
    text = "\n".join([
        json.dumps({"id": "FB-1", "query": "q1", "expected_behavior": "answer"}),
        "",
        json.dumps({"id": "FB-2", "query": "q2", "expected_behavior": "refuse"}),
    ])
    samples = parse_jsonl(text)
    assert [s["id"] for s in samples] == ["FB-1", "FB-2"]

    out = tmp_path / "eval_feedback.jsonl"
    out.write_text(json.dumps({"id": "FB-1", "query": "old",
                               "expected_behavior": "answer"}) + "\n",
                   encoding="utf-8")
    assert merge_samples(out, samples) == 1
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [x["id"] for x in lines] == ["FB-1", "FB-2"]
    assert merge_samples(out, samples) == 0


def test_parse_jsonl_rejects_bad_line():
    text = (json.dumps({"id": "FB-1", "query": "q",
                        "expected_behavior": "answer"}) + "\nnot-json")
    try:
        parse_jsonl(text)
        raise AssertionError("非法行必须报错")
    except ValueError as exc:
        assert "第 2 行" in str(exc)


def test_parse_jsonl_rejects_missing_fields():
    try:
        parse_jsonl('{"id": "FB-1", "query": "q"}')
        raise AssertionError("缺字段必须报错")
    except ValueError as exc:
        assert "expected_behavior" in str(exc)


def test_show_trace_prints_usage_without_original_text(tmp_path, capsys):
    db = tmp_path / "trace.db"
    store = CaseStore(str(db))
    try:
        store.record_trace("case-1", step="consult", agent="supervisor",
                           event="verdict", ok=True, verdict="none",
                           elapsed_ms=42, input_tokens=31, output_tokens=12,
                           tool_count=1, usage_estimated=1)
    finally:
        store.close()

    assert show_trace_main(["--case", "case-1", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "verdict" in out and "42" in out and "consult" in out
    assert "31" in out and "12" in out
    assert "Y" in out

    assert show_trace_main(["--case", "missing", "--db", str(db)]) == 0
    assert "暂无轨迹" in capsys.readouterr().out

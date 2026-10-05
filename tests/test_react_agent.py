"""tests/test_react_agent.py：ReAct 状态机断言测试（流式 FakeClient，离线可跑）。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core_agent import ReActAgent
from tools import ToolRegistry


def text_msg(content):
    return SimpleNamespace(role="assistant", content=content, tool_calls=None)


def tool_msg(*calls):
    return SimpleNamespace(role="assistant", content=None, tool_calls=[
        SimpleNamespace(id=cid, type="function",
                         function=SimpleNamespace(name=name, arguments=args))
        for cid, name, args in calls])


def _tool_chunk(index, tc):
    function = SimpleNamespace(name=tc.function.name, arguments=tc.function.arguments)
    piece = SimpleNamespace(index=index, id=tc.id, type="function", function=function)
    delta = SimpleNamespace(content=None, tool_calls=[piece])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _text_chunk(content):
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def stream_chunks(message):
    """把一条完整 assistant 消息转成流式 delta chunks。"""
    if message.tool_calls:
        return [_tool_chunk(i, tc) for i, tc in enumerate(message.tool_calls)]
    return [_text_chunk(message.content)]


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class FakeCompletions:
    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        self.client.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        message = self.client.responses[len(self.client.calls) - 1]
        if kwargs.get("stream"):
            return FakeStream(stream_chunks(message))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls = responses, []
        self.chat = SimpleNamespace(completions=FakeCompletions(self))


def role_of(m):
    return m.role if hasattr(m, "role") else m["role"]


def test_direct_answer():
    client = FakeClient([text_msg("直接答案")])
    agent = ReActAgent(client, ToolRegistry(), model="fake")
    assert agent.run("你好") == "直接答案"
    assert len(client.calls) == 1
    assert [role_of(m) for m in client.calls[0]["messages"]] == ["system", "user"]


def test_single_tool_then_answer():
    reg = ToolRegistry()

    @reg.tool
    def query_error_doc(keyword: str) -> str:
        """查询错误文档。"""
        return f"doc about {keyword}"

    client = FakeClient([
        tool_msg(("call_1", "query_error_doc", '{"keyword": "timeout"}')),
        text_msg("最终答案"),
    ])
    assert ReActAgent(client, reg, model="fake").run("查一下") == "最终答案"
    msgs = client.calls[-1]["messages"]
    assert [role_of(m) for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[3]["tool_call_id"] == "call_1" and "timeout" in msgs[3]["content"]


def test_exception_backfill():
    reg = ToolRegistry()

    @reg.tool
    def boom(x: str) -> str:
        """总是报错。"""
        raise ValueError("炸了")

    client = FakeClient([tool_msg(("c1", "boom", '{"x": "a"}')),
                         text_msg("修复后答案")])
    assert ReActAgent(client, reg, model="fake").run("go") == "修复后答案"
    content = client.calls[-1]["messages"][3]["content"]
    assert "ValueError" in content and "炸了" in content


def test_max_steps_fuse_and_empty_args():
    reg = ToolRegistry()

    @reg.tool
    def ping() -> str:
        """ping。"""
        return "pong"

    client = FakeClient([tool_msg(("c1", "ping", "")),
                         tool_msg(("c2", "ping", None)),
                         tool_msg(("c3", "ping", ""))])
    result = ReActAgent(client, reg, model="fake", max_steps=3).run("loop")
    assert "步数" in result and "3" in result
    assert len(client.calls) == 3


def test_history_order():
    client = FakeClient([text_msg("ok")])
    history = [{"role": "user", "content": "旧问题"},
               {"role": "assistant", "content": "旧回答"}]
    ReActAgent(client, ToolRegistry(), model="fake").run("新问题", history=history)
    roles = [role_of(m) for m in client.calls[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_parallel_tool_calls():
    reg = ToolRegistry()

    @reg.tool
    def tool_a(q: str) -> str:
        """工具A。"""
        return f"A:{q}"

    @reg.tool
    def tool_b(q: str) -> str:
        """工具B。"""
        return f"B:{q}"

    client = FakeClient([
        tool_msg(("id_a", "tool_a", '{"q": "x"}'),
                 ("id_b", "tool_b", '{"q": "y"}')),
        text_msg("汇总答案"),
    ])
    assert ReActAgent(client, reg, model="fake").run("并行") == "汇总答案"
    tool_msgs = [m for m in client.calls[-1]["messages"]
                 if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "id_a" and tool_msgs[0]["content"] == "A:x"
    assert tool_msgs[1]["tool_call_id"] == "id_b" and tool_msgs[1]["content"] == "B:y"


def test_dangerous_tool_interception():
    agent = ReActAgent(client=None, registry=None)
    fake_tc = SimpleNamespace(id="call_999",
                              function=SimpleNamespace(name="danger_delete_db",
                                                       arguments="{}"))
    result = agent._run_tool(fake_tc)
    assert "PermissionDenied" in result and "安全拦截" in result


if __name__ == "__main__":
    test_direct_answer()
    test_single_tool_then_answer()
    test_exception_backfill()
    test_max_steps_fuse_and_empty_args()
    test_history_order()
    test_parallel_tool_calls()
    test_dangerous_tool_interception()
    print("全部断言通过")

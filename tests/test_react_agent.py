"""tests/test_react_agent.py：ReAct 状态机断言测试（FakeClient 模拟 SDK，离线可跑）。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core_agent import ReActAgent
from tools import ToolRegistry


def make_text(content):
    """SDK 风格：无工具调用的响应。"""
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def make_tool_calls(*calls):
    """SDK 风格：calls = [(id, name, arguments), ...]。"""
    tcs = [SimpleNamespace(id=cid,
                           function=SimpleNamespace(name=name, arguments=args))
           for cid, name, args in calls]
    msg = SimpleNamespace(role="assistant", content=None, tool_calls=tcs)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeCompletions:
    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        kwargs["messages"] = list(kwargs["messages"])  # 快照，防止后续 append 污染
        self.client.calls.append(kwargs)
        return self.client.responses[len(self.client.calls) - 1]


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.chat = SimpleNamespace(completions=FakeCompletions(self))


def role_of(m):
    return m.role if hasattr(m, "role") else m["role"]


def test_direct_answer():
    client = FakeClient([make_text("直接答案")])
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
        make_tool_calls(("call_1", "query_error_doc", '{"keyword": "timeout"}')),
        make_text("最终答案"),
    ])
    agent = ReActAgent(client, reg, model="fake")
    assert agent.run("查一下") == "最终答案"
    # 第二次 create 时模型看到的上下文（最终答案在 create 返回后才产生）
    msgs = client.calls[-1]["messages"]
    assert [role_of(m) for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[3]["tool_call_id"] == "call_1"
    assert "timeout" in msgs[3]["content"]


def test_exception_backfill():
    reg = ToolRegistry()

    @reg.tool
    def boom(x: str) -> str:
        """总是报错。"""
        raise ValueError("炸了")

    client = FakeClient([
        make_tool_calls(("c1", "boom", '{"x": "a"}')),
        make_text("修复后答案"),
    ])
    agent = ReActAgent(client, reg, model="fake")
    assert agent.run("go") == "修复后答案"
    content = client.calls[-1]["messages"][3]["content"]
    assert "ValueError" in content and "炸了" in content


def test_max_steps_fuse_and_empty_args():
    reg = ToolRegistry()

    @reg.tool
    def ping() -> str:
        """ping。"""
        return "pong"

    client = FakeClient([
        make_tool_calls(("c1", "ping", "")),     # 空字符串 → {}
        make_tool_calls(("c2", "ping", None)),   # None → {}
        make_tool_calls(("c3", "ping", "")),
    ])
    agent = ReActAgent(client, reg, model="fake", max_steps=3)
    result = agent.run("loop")
    assert "步数" in result and "3" in result
    assert len(client.calls) == 3


def test_history_order():
    client = FakeClient([make_text("ok")])
    history = [{"role": "user", "content": "旧问题"},
               {"role": "assistant", "content": "旧回答"}]
    agent = ReActAgent(client, ToolRegistry(), model="fake")
    agent.run("新问题", history=history)
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
        make_tool_calls(("id_a", "tool_a", '{"q": "x"}'),
                        ("id_b", "tool_b", '{"q": "y"}')),
        make_text("汇总答案"),
    ])
    agent = ReActAgent(client, reg, model="fake")
    assert agent.run("并行") == "汇总答案"
    tool_msgs = [m for m in client.calls[-1]["messages"]
                 if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "id_a" and tool_msgs[0]["content"] == "A:x"
    assert tool_msgs[1]["tool_call_id"] == "id_b" and tool_msgs[1]["content"] == "B:y"
"""tests/test_react_agent.py：ReAct 状态机断言测试（FakeClient 模拟 SDK，离线可跑）。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core_agent import ReActAgent
from tools import ToolRegistry


def make_text(content):
    """SDK 风格：无工具调用的响应。"""
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def make_tool_calls(*calls):
    """SDK 风格：calls = [(id, name, arguments), ...]。"""
    tcs = [SimpleNamespace(id=cid,
                           function=SimpleNamespace(name=name, arguments=args))
           for cid, name, args in calls]
    msg = SimpleNamespace(role="assistant", content=None, tool_calls=tcs)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeCompletions:
    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        kwargs["messages"] = list(kwargs["messages"])  # 快照，防止后续 append 污染
        self.client.calls.append(kwargs)
        return self.client.responses[len(self.client.calls) - 1]


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.chat = SimpleNamespace(completions=FakeCompletions(self))


def role_of(m):
    return m.role if hasattr(m, "role") else m["role"]


def test_direct_answer():
    client = FakeClient([make_text("直接答案")])
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
        make_tool_calls(("call_1", "query_error_doc", '{"keyword": "timeout"}')),
        make_text("最终答案"),
    ])
    agent = ReActAgent(client, reg, model="fake")
    assert agent.run("查一下") == "最终答案"
    # 第二次 create 时模型看到的上下文（最终答案在 create 返回后才产生）
    msgs = client.calls[-1]["messages"]
    assert [role_of(m) for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[3]["tool_call_id"] == "call_1"
    assert "timeout" in msgs[3]["content"]


def test_exception_backfill():
    reg = ToolRegistry()

    @reg.tool
    def boom(x: str) -> str:
        """总是报错。"""
        raise ValueError("炸了")

    client = FakeClient([
        make_tool_calls(("c1", "boom", '{"x": "a"}')),
        make_text("修复后答案"),
    ])
    agent = ReActAgent(client, reg, model="fake")
    assert agent.run("go") == "修复后答案"
    content = client.calls[-1]["messages"][3]["content"]
    assert "ValueError" in content and "炸了" in content


def test_max_steps_fuse_and_empty_args():
    reg = ToolRegistry()

    @reg.tool
    def ping() -> str:
        """ping。"""
        return "pong"

    client = FakeClient([
        make_tool_calls(("c1", "ping", "")),     # 空字符串 → {}
        make_tool_calls(("c2", "ping", None)),   # None → {}
        make_tool_calls(("c3", "ping", "")),
    ])
    agent = ReActAgent(client, reg, model="fake", max_steps=3)
    result = agent.run("loop")
    assert "步数" in result and "3" in result
    assert len(client.calls) == 3


def test_history_order():
    client = FakeClient([make_text("ok")])
    history = [{"role": "user", "content": "旧问题"},
               {"role": "assistant", "content": "旧回答"}]
    agent = ReActAgent(client, ToolRegistry(), model="fake")
    agent.run("新问题", history=history)
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
        make_tool_calls(("id_a", "tool_a", '{"q": "x"}'),
                        ("id_b", "tool_b", '{"q": "y"}')),
        make_text("汇总答案"),
    ])
    agent = ReActAgent(client, reg, model="fake")
    assert agent.run("并行") == "汇总答案"
    tool_msgs = [m for m in client.calls[-1]["messages"]
                 if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "id_a" and tool_msgs[0]["content"] == "A:x"
    assert tool_msgs[1]["tool_call_id"] == "id_b" and tool_msgs[1]["content"] == "B:y"


if __name__ == "__main__":
    test_direct_answer()
    test_single_tool_then_answer()
    test_exception_backfill()
    test_max_steps_fuse_and_empty_args()
    test_history_order()
    test_parallel_tool_calls()
    print("全部断言通过")
def test_dangerous_tool_interception():
    # 验证当模型尝试调用 danger_ 开头的工具时，会被核心状态机拦截
    agent = ReActAgent(client=None, registry=None)
    from types import SimpleNamespace
    fake_tc = SimpleNamespace(
        id="call_999",
        function=SimpleNamespace(name="danger_delete_db", arguments="{}")
    )
    result = agent._run_tool(fake_tc)
    assert "PermissionDenied" in result
    assert "安全拦截" in result

if __name__ == "__main__":
    test_direct_answer()
    test_single_tool_then_answer()
    test_exception_backfill()
    test_max_steps_fuse_and_empty_args()
    test_history_order()
    test_parallel_tool_calls()
    test_dangerous_tool_interception()
    print("全部断言通过")

"""tests/test_citation.py：引用约束与防幻觉降级断言测试（ScriptedClient，离线可跑）。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tools import ToolRegistry
from knowledge_tool import rag_provider, register_to
from core_agent import ReActAgent, DEGRADE_ANSWER


class FakePipeline:
    def __init__(self, results):
        self.results, self.calls = results, []

    def search(self, query, top_k=3):
        self.calls.append((query, top_k))
        return self.results


def chunk(content="docker compose up -d 拉起服务", source="knowledge/deploy.md",
          idx=1, distance=0.1):
    return {"content": content, "source": source,
            "chunk_id": f"{source}#{idx}", "distance": distance}


def chunks_of(message):
    """把完整 assistant 消息转成流式 delta chunks。"""
    chunks = []
    if message.tool_calls:
        for i, tc in enumerate(message.tool_calls):
            function = SimpleNamespace(name=tc.function.name,
                                      arguments=tc.function.arguments)
            piece = SimpleNamespace(index=i, id=tc.id, type="function",
                                   function=function)
            delta = SimpleNamespace(content=None, tool_calls=[piece])
            chunks.append(SimpleNamespace(choices=[SimpleNamespace(delta=delta)]))
        return chunks
    delta = SimpleNamespace(content=message.content, tool_calls=None)
    return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)])]


class _Stream:
    def __init__(self, chunks):
        self._c = chunks

    def __iter__(self):
        return iter(self._c)


class ScriptedClient:
    """按 create 调用次序返回预设消息（流式），并对每次 messages 做快照。"""

    def __init__(self, script):
        self.script, self.snapshots = script, []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model, messages, tools, stream=False):
        self.snapshots.append(list(messages))
        message = self.script[len(self.snapshots) - 1]
        return _Stream(chunks_of(message)) if stream else \
            SimpleNamespace(choices=[SimpleNamespace(message=message)])


def answer(content):
    return SimpleNamespace(content=content, tool_calls=None)


def tool_call(name, args, cid="call_1"):
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=cid,
                type="function",
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                ),
            )
        ],
    )


def make_agent(script, results, max_steps=5):
    reg = ToolRegistry()
    register_to(reg)
    pipe = FakePipeline(results)
    rag_provider.set(pipe)
    return ReActAgent(ScriptedClient(script), reg, model="fake",
                      max_steps=max_steps), pipe


def test_chitchat_no_tool():
    agent, pipe = make_agent(
        [answer("你好呀，我是 Sentinel-Agent，可以帮你查文档、排查问题。")], [])
    try:
        out = agent.run("你好")
        assert pipe.calls == []
        assert "你好" in out
        assert "【来源：" not in out and "无法提供确切解答" not in out
    finally:
        rag_provider.set(None)


def test_hit_with_citation():
    script = [tool_call("search_knowledge_base", {"query": "服务怎么启动"}),
              answer("执行 docker compose up -d 即可拉起全部服务【来源：knowledge/deploy.md#1】")]
    agent, _ = make_agent(script, [chunk()])
    try:
        assert "【来源：knowledge/deploy.md#1】" in agent.run("服务怎么启动")
    finally:
        rag_provider.set(None)


def test_miss_degrade():
    script = [tool_call("search_knowledge_base", {"query": "集群参数"}),
              answer(DEGRADE_ANSWER)]
    agent, pipe = make_agent(script, [])
    try:
        out = agent.run("集群参数怎么配")
        assert "无法提供确切解答" in out and "【来源：" not in out
        assert pipe.calls[0] == ("集群参数", 3)  # 检索词取自模型 tool_call，非用户原话
    finally:
        rag_provider.set(None)


def test_hit_postcheck_rewrite():
    script = [tool_call("search_knowledge_base", {"query": "启动"}),
              answer("执行 docker compose up -d 就行。"),  # 漏角标
              answer("执行 docker compose up -d 拉起服务【来源：knowledge/deploy.md#1】")]
    agent, _ = make_agent(script, [chunk()])
    try:
        out = agent.run("启动")
        assert "【来源：knowledge/deploy.md#1】" in out
        fix = [m for m in agent.client.snapshots[2]
               if isinstance(m, dict) and m.get("role") == "user"
               and "系统校验" in m.get("content", "")]
        assert fix and "【来源：" in fix[0]["content"]  # 补正指令确实下发
    finally:
        rag_provider.set(None)


def test_miss_postcheck_degrade():
    script = [tool_call("search_knowledge_base", {"query": "配额"}),
              answer("你可以把配额设成 100。"),  # 硬编
              answer(DEGRADE_ANSWER)]
    agent, _ = make_agent(script, [])
    try:
        out = agent.run("配额多少")
        assert "无法提供确切解答" in out and "100" not in out
    finally:
        rag_provider.set(None)


def test_timeout_hit_degrades():
    # 命中后连续 5 轮漏标：超时也必须优雅降级，而不是机械报“最大步数”
    script = [tool_call("search_knowledge_base", {"query": "启动"})] \
        + [answer("docker compose up 就行")] * 4
    agent, _ = make_agent(script, [chunk()], max_steps=5)
    try:
        out = agent.run("启动")
        assert out == DEGRADE_ANSWER and "最大步数" not in out
    finally:
        rag_provider.set(None)


def test_timeout_none_step_message():
    # 未涉及检索（cite_state=None）的工具循环超时，才返回步数熔断提示
    def echo(text: str) -> str:
        """回显输入文本。"""
        return text

    reg = ToolRegistry()
    reg.register(echo)
    script = [tool_call("echo", {"text": "x"}, cid=f"c{i}") for i in range(5)]
    out = ReActAgent(ScriptedClient(script), reg, model="fake",
                     max_steps=5).run("x")
    assert "最大步数 5" in out and "无法提供确切解答" not in out


def test_missing_distance_still_hit():
    r = {"content": "docker compose up -d", "source": "knowledge/deploy.md",
         "chunk_id": "knowledge/deploy.md#1"}  # 无 distance 字段
    script = [tool_call("search_knowledge_base", {"query": "启动"}),
              answer("docker compose up -d【来源：knowledge/deploy.md#1】")]
    agent, _ = make_agent(script, [r])
    try:
        assert "【来源：knowledge/deploy.md#1】" in agent.run("启动")
    finally:
        rag_provider.set(None)


def test_custom_hit_state_not_overwritten():
    # 同轮先检索命中、后调普通工具：cite_state 锁定为 hit，漏标耗尽后必须降级而非报步数
    reg = ToolRegistry()

    @reg.tool
    def search_knowledge_base(query: str) -> str:
        """检索知识库。"""
        return "【知识库检索结果】\n[1] doc.md#0\n核心配置参数"

    @reg.tool
    def dummy_tool() -> str:
        """普通工具。"""
        return "ok"

    both = SimpleNamespace(content=None, tool_calls=[
        SimpleNamespace(id="c1", type="function",
                        function=SimpleNamespace(name="search_knowledge_base",
                                                 arguments='{"query": "配置"}')),
        SimpleNamespace(id="c2", type="function",
                        function=SimpleNamespace(name="dummy_tool", arguments="")),
    ])
    agent = ReActAgent(ScriptedClient([both] + [answer("核心配置参数（漏标）")] * 4),
                       reg, model="fake", max_steps=5)
    out = agent.run("配置")
    assert out == DEGRADE_ANSWER and "最大步数" not in out
    tool_msgs = [m for m in agent.client.snapshots[1]
                 if isinstance(m, dict) and m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]  # 两工具同轮均回填


if __name__ == "__main__":
    test_chitchat_no_tool()
    test_hit_with_citation()
    test_miss_degrade()
    test_hit_postcheck_rewrite()
    test_miss_postcheck_degrade()
    test_timeout_hit_degrades()
    test_timeout_none_step_message()
    test_missing_distance_still_hit()
    test_custom_hit_state_not_overwritten()
    print("全部断言通过")

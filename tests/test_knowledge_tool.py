"""tests/test_knowledge_tool.py：知识库检索工具断言测试（FakePipeline，离线可跑）。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tools import ToolRegistry
import knowledge_tool
from knowledge_tool import rag_provider, register_to, format_results


class FakePipeline:
    def __init__(self, results):
        self.results, self.calls = results, []

    def search(self, query, top_k=3):
        self.calls.append((query, top_k))
        return self.results


def chunk(content, distance=0.1, source="knowledge/deploy.md", idx=3):
    return {"content": content, "source": source,
            "chunk_id": f"{source}#{idx}", "distance": distance}


def test_schema_structure():
    reg = ToolRegistry()
    register_to(reg)
    fn = reg.get("search_knowledge_base").schema["function"]
    assert fn["name"] == "search_knowledge_base"
    assert fn["description"].startswith("检索宠物健康与用药知识库")
    assert fn["parameters"]["properties"]["query"]["type"] == "string"
    assert fn["parameters"]["required"] == ["query"]


def test_hit_format():
    reg = ToolRegistry()
    register_to(reg)
    fake = FakePipeline([chunk("docker compose up -d 拉起服务", idx=3),
                         chunk("连接超时先查网络", source="knowledge/errors.md", idx=1)])
    rag_provider.set(fake)
    try:
        out = reg.call("search_knowledge_base", {"query": "部署"})
        assert "【知识库检索结果】" in out
        assert "[1]" in out and "[2]" in out
        assert "knowledge/deploy.md#3" in out
        assert "knowledge/errors.md#1" in out
    finally:
        rag_provider.set(None)


def test_empty_and_far():
    reg = ToolRegistry()
    register_to(reg)
    for results in ([], [chunk("无关内容", distance=0.95)]):
        rag_provider.set(FakePipeline(results))
        try:
            out = reg.call("search_knowledge_base", {"query": "天气"})
            assert "未检索到" in out and "[1]" not in out
        finally:
            rag_provider.set(None)


def test_missing_distance_and_long_content():
    r = {"content": "字" * 500, "source": "a.md", "chunk_id": "a.md#0"}  # 无 distance
    out = format_results([r], "x")
    assert "[1] a.md#0" in out          # 缺 distance 不崩，按相关处理
    assert "已截断" in out              # 超长正文被截断
    assert len(out) < 400


class FakeClient:
    """第一次要求查知识库，第二次给最终答案；记录每次 create 的 messages 快照。"""

    def __init__(self):
        self.snapshots = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model, messages, tools, stream=False):
        self.snapshots.append(list(messages))

        def gen(chunks):
            for c in chunks:
                yield c

        if len(self.snapshots) == 1:
            function = SimpleNamespace(name="search_knowledge_base",
                                      arguments='{"query": "部署怎么启动"}')
            piece = SimpleNamespace(index=0, id="call_1", type="function",
                                   function=function)
            delta = SimpleNamespace(content=None, tool_calls=[piece])
            return gen([SimpleNamespace(choices=[SimpleNamespace(delta=delta)])])
        delta = SimpleNamespace(content="（最终答案）", tool_calls=None)
        return gen([SimpleNamespace(choices=[SimpleNamespace(delta=delta)])])


def test_agent_link():
    from core_agent import ReActAgent
    reg = ToolRegistry()
    register_to(reg)
    fake_pipe = FakePipeline([chunk("docker compose up -d 一键拉起", idx=3)])
    rag_provider.set(fake_pipe)
    client = FakeClient()
    try:
        ReActAgent(client, reg, model="fake").run("部署怎么启动")
        assert fake_pipe.calls[0] == ("部署怎么启动", 3)       # query/top_k 透传
        second = client.snapshots[1]
        tool_msgs = [m for m in second if isinstance(m, dict) and m.get("role") == "tool"]
        assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_1"
        content = tool_msgs[0]["content"]
        assert "【知识库检索结果】" in content                 # 回填的是格式化片段
        assert "knowledge/deploy.md#3" in content
        assert "docker compose up -d" in content
    finally:
        rag_provider.set(None)


def test_custom_chunk_truncation():
    # 单块正文超过 300 字时强制截断并加标记
    from knowledge_tool import format_results
    super_long_content = "关键配置项" * 80  # 400 字符，超过 300
    fake_results = [{
        "content": super_long_content,
        "source": "knowledge/config.md",
        "chunk_id": "knowledge/config.md#0",
        "distance": 0.1,
    }]
    formatted = format_results(fake_results, "配置")
    assert "……（已截断）" in formatted
    assert "knowledge/config.md#0" in formatted


if __name__ == "__main__":
    test_schema_structure()
    test_hit_format()
    test_empty_and_far()
    test_missing_distance_and_long_content()
    test_agent_link()
    test_custom_chunk_truncation()
    print("全部断言通过")

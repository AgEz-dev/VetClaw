"""tests/test_api.py：FastAPI + SSE 端到端断言测试（注入流式 Fake，离线可跑）。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

from app import app
from tools import ToolRegistry
from core_agent import ReActAgent
from knowledge_tool import rag_provider, register_to
from services.agent_service import set_agent


class FakePipeline:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=3):
        return self.results


def chunk():
    return {"content": "docker compose up -d 拉起服务", "source": "knowledge/deploy.md",
            "chunk_id": "knowledge/deploy.md#1", "distance": 0.1}


def answer(content):
    return SimpleNamespace(content=content, tool_calls=None)


def tool_call(name, args, cid="call_1"):
    function = SimpleNamespace(name=name,
                               arguments=json.dumps(args, ensure_ascii=False))
    piece = SimpleNamespace(id=cid, type="function", function=function)
    return SimpleNamespace(content=None, tool_calls=[piece])


def parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name, data_lines = None, []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        events.append((name, json.loads("".join(data_lines))))
    return events


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_stream_headers_and_tool_sequence():
    reg = ToolRegistry()
    register_to(reg)
    rag_provider.set(FakePipeline([chunk()]))
    client = ScriptedClientLike([
        tool_call("search_knowledge_base", {"query": "启动"}),
        answer("执行 docker compose up -d【来源：knowledge/deploy.md#1】"),
    ])
    set_agent(ReActAgent(client, reg, model="fake"))
    try:
        with TestClient(app) as c:
            r = c.post("/api/chat/stream", json={"prompt": "启动"})
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            events = parse_sse(r.text)
            names = [n for n, _ in events]
            assert names == ["thought", "tool_call", "thought", "tool_result",
                             "thought", "token", "done"]
            assert events[1][1]["name"] == "search_knowledge_base"
            assert "docker compose up -d" in events[3][1]["content"]
            assert events[-1][1]["citations"] == ["【来源：knowledge/deploy.md#1】"]
    finally:
        rag_provider.set(None)
        set_agent(None)


def test_chitchat_only_token_done():
    set_agent(ReActAgent(ScriptedClientLike([answer("你好，我是 VetClaw。")]),
                         ToolRegistry(), model="fake"))
    try:
        with TestClient(app) as c:
            events = parse_sse(c.post("/api/chat/stream", json={"prompt": "你好"}).text)
            assert [n for n, _ in events] == ["thought", "token", "done"]
    finally:
        set_agent(None)


def test_invalid_request_422():
    with TestClient(app) as c:
        assert c.post("/api/chat/stream", json={"prompt": ""}).status_code == 422
        assert c.post("/api/chat/stream", json={}).status_code == 422


def test_timeout_ends_with_error_and_generator_exhausted():
    reg = ToolRegistry()

    @reg.tool
    def ping() -> str:
        """ping。"""
        return "pong"

    client = ScriptedClientLike(
        [tool_call("ping", {}, cid=f"c{i}") for i in range(5)])
    set_agent(ReActAgent(client, reg, model="fake", max_steps=5))
    try:
        with TestClient(app) as c:
            events = parse_sse(c.post("/api/chat/stream", json={"prompt": "loop"}).text)
            assert events[-1][0] == "error"          # 末尾必为 error
            assert "最大步数 5" in events[-1][1]["message"]
            assert len([n for n, _ in events if n == "error"]) == 1  # 其后再无事件
    finally:
        set_agent(None)


# 最小流式客户端（与其它测试中的 ScriptedClient 同构，独立放置避免跨文件依赖）
def _chunks_of(message):
    if message.tool_calls:
        out = []
        for i, tc in enumerate(message.tool_calls):
            function = SimpleNamespace(name=tc.function.name,
                                      arguments=tc.function.arguments)
            piece = SimpleNamespace(index=i, id=tc.id, type="function",
                                   function=function)
            delta = SimpleNamespace(content=None, tool_calls=[piece])
            out.append(SimpleNamespace(choices=[SimpleNamespace(delta=delta)]))
        return out
    delta = SimpleNamespace(content=message.content, tool_calls=None)
    return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)])]


class ScriptedClientLike:
    def __init__(self, script):
        self.script, self.n = script, 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, model, messages, tools, stream=False):
        message = self.script[self.n]
        self.n += 1
        return _Iterable(_chunks_of(message)) if stream else \
            SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Iterable:
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        return iter(self.chunks)


if __name__ == "__main__":
    test_health()
    test_stream_headers_and_tool_sequence()
    test_chitchat_only_token_done()
    test_invalid_request_422()
    test_timeout_ends_with_error_and_generator_exhausted()
    print("全部断言通过")

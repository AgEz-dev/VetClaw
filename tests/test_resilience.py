"""tests/test_resilience.py：生产弹性断言测试（超时守卫、SDK 异常兜底、CORS、全局异常）。"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
from app import app  # noqa: E402
from core_agent import ReActAgent  # noqa: E402
from tools import ToolRegistry  # noqa: E402
from api import to_sse  # noqa: E402


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


class FakeClock:
    """假时钟：第一次返回 0，之后每次 +999，让总时长守卫立即触发。"""
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 999.0
        return self.now


def _text_chunk(content):
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_fake_clock_total_timeout():
    """假时钟让总时长守卫瞬间触发，断言 error code=MODEL_TIMEOUT，0ms 跑完。"""
    clock = FakeClock()

    class SlowClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)
            self.closed = None

        def create(self, **kwargs):
            # 模型正常吐字，但假时钟已超过 total_timeout
            return FakeStream([_text_chunk("你好")])

    agent = ReActAgent(SlowClient(), ToolRegistry(), model="fake",
                       clock=clock, total_timeout=1.0)
    events = list(agent.run_stream("你好"))
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["code"] == "MODEL_TIMEOUT"
    assert "超时" in events[-1]["data"]["message"]


def test_sdk_exception_turned_into_error_event():
    """FakeClient.create 直接抛异常，断言转成 error 事件而非 traceback 冒出。"""
    class BoomClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            raise RuntimeError("boom from sdk")

    agent = ReActAgent(BoomClient(), ToolRegistry(), model="fake")
    events = list(agent.run_stream("你好"))
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["code"] == "INTERNAL_ERROR"
    assert "boom" not in events[-1]["data"]["message"]  # 不泄漏原始异常


def test_cors_preflight_headers():
    """OPTIONS 预检带 Origin，断言响应头含 Access-Control-Allow-Origin。"""
    with TestClient(app) as c:
        r = c.options(
            "/api/chat/stream",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.status_code in (200, 204)
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_global_exception_handler_returns_clean_json():
    """临时注册一个抛 ValueError 的测试路由，断言 500 + 统一 JSON + 不泄漏 traceback。"""
    @app.get("/_test_boom")
    def boom():
        raise ValueError("secret-detail-must-not-leak")

    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/_test_boom")
        assert r.status_code == 500
        body = r.json()
        assert body["code"] == "INTERNAL_ERROR"
        assert "secret-detail-must-not-leak" not in str(body)
        assert "Traceback" not in str(body)
    finally:
        app.router.routes = [
            route for route in app.router.routes
            if getattr(route, "path", None) != "/_test_boom"
        ]


def test_custom_clock_not_timeout_passes():
    # 假时钟在未超限时，生成器正常吐事件、以 done 收尾，不触发超时 error
    class SlowClock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            self.t += 1.0
            return self.t

    class MockClient:
        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    chunk = SimpleNamespace(
                        choices=[SimpleNamespace(
                            delta=SimpleNamespace(content="你好", tool_calls=None))])
                    return iter([chunk])

    agent = ReActAgent(client=MockClient(), registry=ToolRegistry(),
                       clock=SlowClock(), total_timeout=30)
    events = list(agent.run_stream("测试"))
    event_types = [e["event"] for e in events]
    assert "error" not in event_types
    assert event_types[-1] == "done"


if __name__ == "__main__":
    test_fake_clock_total_timeout()
    test_sdk_exception_turned_into_error_event()
    test_cors_preflight_headers()
    test_global_exception_handler_returns_clean_json()
    test_custom_clock_not_timeout_passes()
    print("全部断言通过")

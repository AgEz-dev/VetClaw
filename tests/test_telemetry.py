import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from telemetry import (
    init_telemetry,
    get_current_telemetry,
    get_telemetry_snapshot,
    track_stage,
    record_tokens,
    record_fallback,
    record_fastpath_hit,
    persist_telemetry,
    get_telemetry_summary,
    init_db
)
from mcp_server import VetClawMcpServer


def test_track_stage_decorator_sync():
    init_telemetry("trace-sync-1", "triage", "USER")

    @track_stage("fastpath")
    def sync_sample(a, b):
        """Sample sync function docstring."""
        import time
        time.sleep(0.01)
        return a + b

    assert sync_sample.__name__ == "sync_sample"
    assert "Sample sync function" in sync_sample.__doc__
    res = sync_sample(3, 4)
    assert res == 7

    telem = get_telemetry_snapshot()
    assert telem["fastpath_ms"] > 5.0
    assert "fastpath" in telem["stages"]


def test_track_stage_decorator_async():
    async def _run():
        init_telemetry("trace-async-1", "consult", "DOCTOR")

        @track_stage("rag_retrieval")
        async def async_sample(x):
            """Sample async docstring."""
            await asyncio.sleep(0.01)
            return x * 2

        assert async_sample.__name__ == "async_sample"
        assert "Sample async docstring" in async_sample.__doc__
        res = await async_sample(5)
        assert res == 10

        telem = get_telemetry_snapshot()
        assert telem["rag_retrieval_ms"] > 5.0

    asyncio.run(_run())


def test_async_context_isolation():
    """验证高并发协程中 ContextVar 隔离，trace_id 不串线。"""
    async def _run():
        async def worker(trace_id: str, step: str, role: str, sleep_s: float):
            init_telemetry(trace_id, step, role)
            await asyncio.sleep(sleep_s)
            snap = get_telemetry_snapshot()
            assert snap["trace_id"] == trace_id
            assert snap["step"] == step
            assert snap["role"] == role

        await asyncio.gather(
            worker("req-001", "triage", "USER", 0.02),
            worker("req-002", "draft", "DOCTOR", 0.01),
            worker("req-003", "review", "ADMIN", 0.015),
        )

    asyncio.run(_run())


def test_sqlite_persistence_and_summary(tmp_path):
    db_file = tmp_path / "test_telem.db"
    init_db(db_file)

    # 插入一条正常请求
    init_telemetry("req-normal", "draft", "DOCTOR")
    record_tokens(500, 200)
    telem1 = get_telemetry_snapshot()
    telem1["llm_ttfb_ms"] = 120.0
    telem1["llm_total_ms"] = 850.0
    row_id_1 = persist_telemetry(telem1, db_path=db_file)
    assert row_id_1 > 0

    # 插入一条 FastPath 拦截请求
    init_telemetry("req-fp", "triage", "USER")
    record_fastpath_hit(True)
    telem2 = get_telemetry_snapshot()
    telem2["fastpath_ms"] = 1.2
    telem2["fastpath_hit"] = True
    row_id_2 = persist_telemetry(telem2, db_path=db_file)
    assert row_id_2 > 0

    summary = get_telemetry_summary(db_path=db_file)
    assert summary["total_requests"] == 2
    assert summary["fastpath_hits"] == 1
    assert summary["fastpath_hit_rate"] == 0.5
    assert summary["estimated_saved_tokens"] == 800
    assert summary["total_prompt_tokens"] == 500
    assert summary["total_completion_tokens"] == 200


def test_mcp_resource_telemetry_summary():
    server = VetClawMcpServer()
    defs = server.get_resource_definitions()
    uris = [d["uri"] for d in defs]
    assert "vetclaw://telemetry/summary" in uris

    res = server.read_resource("vetclaw://telemetry/summary")
    assert res["uri"] == "vetclaw://telemetry/summary"
    assert "total_requests" in res["text"]

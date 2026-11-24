"""全链路异步遥测与 Token/延迟监控系统 (Telemetry System)。

对标 2026 工业级 AI Agent 生产监控规范：
1. 异步上下文隔离：使用 contextvars.ContextVar 绑定 trace_id / step / role，确保高并发下指标不串线；
2. 双模监控装饰器：@track_stage 兼容同步函数与 async def 协程，保留原生元信息 (@functools.wraps)；
3. 关键性能指标捕获：精准记录 fastpath_ms、rag_retrieval_ms、llm_ttfb_ms、llm_total_ms、Token 吞吐量；
4. 商业价值量化：量化 Fast-Path 拦截率与节省的 Token 成本；
5. 轻量化持久化与 MCP 暴露：SQLite 日志落盘与只读 MCP Resource vetclaw://telemetry/summary。
"""
from __future__ import annotations

import asyncio
import functools
import sqlite3
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# ---- 上下文变量 (ContextVars) ----
_trace_id_var: ContextVar[str] = ContextVar("telemetry_trace_id", default="")
_step_var: ContextVar[str] = ContextVar("telemetry_step", default="")
_role_var: ContextVar[str] = ContextVar("telemetry_role", default="")
_current_telemetry_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar("current_telemetry", default=None)

# 默认数据库路径
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "telemetry.db"


def init_telemetry(
    trace_id: str,
    step: str = "consult",
    role: str = "USER",
) -> Dict[str, Any]:
    """初始化当前请求的遥测数据结构并存入 ContextVar。"""
    telem: Dict[str, Any] = {
        "trace_id": trace_id,
        "step": step,
        "role": role,
        "fastpath_ms": 0.0,
        "rag_retrieval_ms": 0.0,
        "llm_ttfb_ms": 0.0,
        "llm_total_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "fallback_used": False,
        "fastpath_hit": False,
        "stages": {},
        "start_time": time.perf_counter(),
    }
    _trace_id_var.set(trace_id)
    _step_var.set(step)
    _role_var.set(role)
    _current_telemetry_var.set(telem)
    return telem


def get_current_telemetry() -> Optional[Dict[str, Any]]:
    """获取当前协程上下文中的遥测数据。"""
    return _current_telemetry_var.get()


def get_telemetry_snapshot() -> Dict[str, Any]:
    """获取当前遥测数据的只读浅拷贝；若未初始化则返回兜底空结构。"""
    telem = _current_telemetry_var.get()
    if telem is None:
        return {
            "trace_id": _trace_id_var.get() or "unknown",
            "step": _step_var.get() or "unknown",
            "role": _role_var.get() or "unknown",
            "fastpath_ms": 0.0,
            "rag_retrieval_ms": 0.0,
            "llm_ttfb_ms": 0.0,
            "llm_total_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "fallback_used": False,
            "fastpath_hit": False,
        }
    snapshot = dict(telem)
    if "start_time" in snapshot:
        snapshot["total_latency_ms"] = round((time.perf_counter() - snapshot["start_time"]) * 1000, 2)
    return snapshot


def record_tokens(prompt_tokens: int, completion_tokens: int) -> None:
    """累加当前请求消耗的 Token。"""
    telem = _current_telemetry_var.get()
    if telem is not None:
        telem["prompt_tokens"] += int(prompt_tokens)
        telem["completion_tokens"] += int(completion_tokens)


def record_fallback(used: bool = True) -> None:
    """标记当前请求是否触发了三级容灾降级。"""
    telem = _current_telemetry_var.get()
    if telem is not None:
        telem["fallback_used"] = bool(used)


def record_fastpath_hit(hit: bool = True) -> None:
    """标记当前请求是否命中了 Fast-Path 规则拦截。"""
    telem = _current_telemetry_var.get()
    if telem is not None:
        telem["fastpath_hit"] = bool(hit)


def track_stage(stage_name: str) -> Callable:
    """通用监控装饰器：精准统计函数或协程执行耗时，并更新当前遥测指标。

    兼容常规同步函数与 async def 异步协程，严格使用 @functools.wraps 保留原函数元信息。
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                    _record_stage_metric(stage_name, elapsed_ms)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                    _record_stage_metric(stage_name, elapsed_ms)
            return sync_wrapper
    return decorator


def _record_stage_metric(stage_name: str, elapsed_ms: float) -> None:
    telem = _current_telemetry_var.get()
    if telem is None:
        return
    telem["stages"][stage_name] = elapsed_ms
    if stage_name == "fastpath":
        telem["fastpath_ms"] = elapsed_ms
    elif stage_name == "rag_retrieval":
        telem["rag_retrieval_ms"] = elapsed_ms
    elif stage_name == "llm_ttfb":
        telem["llm_ttfb_ms"] = elapsed_ms
    elif stage_name in ("llm_total", "llm_call"):
        telem["llm_total_ms"] = elapsed_ms


# ---- SQLite 持久化与聚合统计 ----

def init_db(db_path: Path | str | None = None) -> Path:
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                step TEXT NOT NULL,
                role TEXT NOT NULL,
                fastpath_ms REAL DEFAULT 0,
                rag_retrieval_ms REAL DEFAULT 0,
                llm_ttfb_ms REAL DEFAULT 0,
                llm_total_ms REAL DEFAULT 0,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                fallback_used INTEGER DEFAULT 0,
                fastpath_hit INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_telem_trace ON ai_telemetry_logs (trace_id);")
        conn.commit()
    return target_path


def persist_telemetry(
    telem: Optional[Dict[str, Any]] = None,
    db_path: Path | str | None = None,
) -> int:
    """将一条遥测记录持久化至 SQLite 数据库，返回自增 ID。"""
    target = telem or get_telemetry_snapshot()
    path = init_db(db_path)
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ai_telemetry_logs (
                trace_id, step, role, fastpath_ms, rag_retrieval_ms,
                llm_ttfb_ms, llm_total_ms, prompt_tokens, completion_tokens,
                fallback_used, fastpath_hit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target.get("trace_id", "unknown"),
            target.get("step", "consult"),
            target.get("role", "USER"),
            float(target.get("fastpath_ms", 0.0)),
            float(target.get("rag_retrieval_ms", 0.0)),
            float(target.get("llm_ttfb_ms", 0.0)),
            float(target.get("llm_total_ms", 0.0)),
            int(target.get("prompt_tokens", 0)),
            int(target.get("completion_tokens", 0)),
            1 if target.get("fallback_used") else 0,
            1 if target.get("fastpath_hit") else 0,
        ))
        conn.commit()
        return cursor.lastrowid or 0


def get_telemetry_summary(db_path: Path | str | None = None) -> Dict[str, Any]:
    """聚合查询遥测大盘指标，输出请求总量、平均延迟、Token 消耗以及 Fast-Path 节省估算。"""
    path = init_db(db_path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        row = cur.execute("""
            SELECT
                COUNT(*) as total_requests,
                AVG(llm_ttfb_ms) as avg_ttfb_ms,
                AVG(fastpath_ms) as avg_fastpath_ms,
                AVG(rag_retrieval_ms) as avg_rag_ms,
                AVG(llm_total_ms) as avg_llm_ms,
                AVG(prompt_tokens) as avg_prompt_tokens,
                AVG(completion_tokens) as avg_completion_tokens,
                SUM(prompt_tokens) as total_prompt_tokens,
                SUM(completion_tokens) as total_completion_tokens,
                SUM(CASE WHEN fastpath_hit = 1 THEN 1 ELSE 0 END) as fastpath_hits,
                SUM(CASE WHEN fallback_used = 1 THEN 1 ELSE 0 END) as fallback_count
            FROM ai_telemetry_logs
        """).fetchone()

        total = row["total_requests"] if row and row["total_requests"] else 0
        hits = row["fastpath_hits"] if row and row["fastpath_hits"] else 0
        hit_rate = round(hits / total, 4) if total > 0 else 0.0
        # 每一次 FastPath 命中，避开大模型 Prompt (约 600 token) 与 Output (约 200 token)，节省约 800 token
        saved_tokens = hits * 800

        return {
            "total_requests": total,
            "avg_ttfb_ms": round(row["avg_ttfb_ms"] or 0.0, 2),
            "avg_fastpath_ms": round(row["avg_fastpath_ms"] or 0.0, 2),
            "avg_rag_ms": round(row["avg_rag_ms"] or 0.0, 2),
            "avg_llm_ms": round(row["avg_llm_ms"] or 0.0, 2),
            "avg_prompt_tokens": round(row["avg_prompt_tokens"] or 0.0, 1),
            "avg_completion_tokens": round(row["avg_completion_tokens"] or 0.0, 1),
            "total_prompt_tokens": row["total_prompt_tokens"] or 0,
            "total_completion_tokens": row["total_completion_tokens"] or 0,
            "fastpath_hits": hits,
            "fastpath_hit_rate": hit_rate,
            "estimated_saved_tokens": saved_tokens,
            "fallback_count": row["fallback_count"] or 0,
            "architecture_roi_summary": f"Fast-Path 规则硬拦截率 {hit_rate * 100:.1f}%，累计避免 {saved_tokens} 个 Token 的大模型算力消耗，平均规则耗时仅 {round(row['avg_fastpath_ms'] or 0.0, 2)}ms。"
        }

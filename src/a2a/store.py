"""case 级结构化状态存储（SQLite / 标准库）。

存两样东西，缺一不可：

1. **case 状态**（`a2a_case`）：一次就诊的累计上下文 —— 宠物事实 + 各能力位的产出。
   这是「四个能力位共享上下文」的物化；没有它，起草还得让调用方把预问诊摘要重传一遍。
2. **幂等回放表**（`a2a_result`）：`idempotency_key → (请求指纹, 结果)`。
   医院侧重试、网关重放、双写重发都会命中这里，保证**同一次任务只执行一次**。

为什么幂等表要存**请求指纹**而不只是结果：只存结果的话，调用方若用同一个幂等键
提交了**不同内容**（改了处方再重试），就会静默拿到上一次的旧结论 —— 这在用药安全
场景里是致命错误。指纹不同必须显式冲突（见 `supervisor`）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from a2a.contract import CaseState, TaskResult

DEFAULT_DB_PATH = os.getenv("VETCLAW_A2A_DB", "data/a2a.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS a2a_case (
    case_id     TEXT PRIMARY KEY,
    state_json  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    seq         INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS a2a_result (
    idempotency_key     TEXT PRIMARY KEY,
    request_fingerprint TEXT NOT NULL,
    result_json         TEXT NOT NULL,
    created_at          REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_trace (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    step            TEXT NOT NULL,
    agent           TEXT NOT NULL,
    event           TEXT NOT NULL,
    tool_name       TEXT,
    args_digest     TEXT,
    ok              INTEGER,
    error_code      TEXT,
    verdict         TEXT,
    elapsed_ms      INTEGER,
    prompt_version  TEXT,
    model           TEXT,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    tool_count      INTEGER NOT NULL DEFAULT 0,
    usage_estimated INTEGER NOT NULL DEFAULT 0,
    from_step       TEXT,
    to_step         TEXT,
    reason_code     TEXT,
    case_seq        INTEGER,
    payload_digest  TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_case ON agent_trace(case_id, seq);
CREATE INDEX IF NOT EXISTS idx_trace_event ON agent_trace(event);
"""

# 旧库升级：CREATE TABLE IF NOT EXISTS 不会给已存在的表补列，必须显式 ALTER。
# C 批新增 handoff 五列；D/E 批若再加列，继续往这里追加即可。
_TRACE_MIGRATIONS = (
    ("from_step", "TEXT"),
    ("to_step", "TEXT"),
    ("reason_code", "TEXT"),
    ("case_seq", "INTEGER"),
    ("payload_digest", "TEXT"),
    ("tool_count", "INTEGER NOT NULL DEFAULT 0"),
    ("usage_estimated", "INTEGER NOT NULL DEFAULT 0"),
)


class CaseStore:
    """线程安全的 case 存储。`path=":memory:"` 供单测使用。"""

    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        if path not in (":memory:", ""):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate_trace()
            self._conn.commit()

    def _migrate_trace(self) -> None:
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(agent_trace)")}
        for name, ddl in _TRACE_MIGRATIONS:
            if name not in columns:
                self._conn.execute(f"ALTER TABLE agent_trace ADD COLUMN {name} {ddl}")

    # ------------------------------------------------------------------ case

    def get_case(self, case_id: str) -> CaseState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM a2a_case WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            return None
        return CaseState.from_dict(json.loads(row["state_json"]))

    def create_case(self, case_id: str, patient: dict | None = None) -> CaseState:
        """创建 case；已存在则**返回既有状态**（幂等，不覆盖已积累的上下文）。"""
        existing = self.get_case(case_id)
        if existing is not None:
            return existing
        state = CaseState(case_id=case_id, patient=patient or {})
        self.save_case(state)
        return state

    def save_case(self, state: CaseState) -> None:
        payload = json.dumps(state.to_dict(), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO a2a_case(case_id, state_json, created_at, updated_at, seq)"
                " VALUES(?,?,?,?,?)"
                " ON CONFLICT(case_id) DO UPDATE SET"
                "   state_json=excluded.state_json, updated_at=excluded.updated_at,"
                "   seq=excluded.seq",
                (state.case_id, payload, state.created_at, state.updated_at, state.seq),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ 幂等

    def get_result(self, idempotency_key: str) -> tuple[str, TaskResult] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT request_fingerprint, result_json FROM a2a_result"
                " WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        if row is None:
            return None
        return row["request_fingerprint"], TaskResult.from_dict(json.loads(row["result_json"]))

    def put_result(self, idempotency_key: str, request_fingerprint: str,
                   result: TaskResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO a2a_result(idempotency_key, request_fingerprint,"
                " result_json, created_at) VALUES(?,?,?,?)"
                " ON CONFLICT(idempotency_key) DO NOTHING",
                (idempotency_key, request_fingerprint,
                 json.dumps(result.to_dict(), ensure_ascii=False), result.created_at),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ 轨迹

    def record_trace(self, case_id: str, *, step: str, agent: str, event: str,
                     tool_name: str | None = None, args: dict | None = None,
                     ok: bool | None = None, error_code: str | None = None,
                     verdict: str | None = None, elapsed_ms: int | None = None,
                     prompt_version: str = "", model: str = "",
                     input_tokens: int = 0, output_tokens: int = 0,
                     tool_count: int = 0, usage_estimated: int = 0,
                     from_step: str | None = None, to_step: str | None = None,
                     reason_code: str | None = None, case_seq: int | None = None,
                     payload_digest: str | None = None) -> None:
        """追加一条轨迹。``args`` / ``payload`` 只落摘要 —— 病例原文绝不进表。"""
        digest = digest_args(args)
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_trace WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            seq = int(row[0])
            self._conn.execute(
                "INSERT INTO agent_trace(case_id, seq, step, agent, event, tool_name,"
                " args_digest, ok, error_code, verdict, elapsed_ms, prompt_version,"
                " model, input_tokens, output_tokens, tool_count, usage_estimated,"
                " from_step, to_step,"
                " reason_code, case_seq, payload_digest, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (case_id, seq, step, agent, event, tool_name, digest,
                 None if ok is None else int(bool(ok)), error_code, verdict,
                 elapsed_ms, prompt_version, model, input_tokens, output_tokens,
                 int(tool_count or 0), int(bool(usage_estimated)),
                 from_step, to_step, reason_code, case_seq, payload_digest,
                 time.time()),
            )
            self._conn.commit()

    def list_trace(self, case_id: str, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM agent_trace WHERE case_id = ? ORDER BY seq ASC LIMIT ?",
                (case_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]


def digest_args(args: dict | None) -> str | None:
    """入参摘要：键排序后 SHA-256，保留可比对性但不泄漏内容。"""
    if args is None:
        return None
    material = json.dumps(args, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

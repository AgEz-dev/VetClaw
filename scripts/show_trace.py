"""打印一个 case 的脱敏执行轨迹（演示与排障共用）。

用法：
    python -m scripts.show_trace --case chat-<conversationId>
    python -m scripts.show_trace --case case-1 --db data/a2a.db --json

轨迹表只含参数摘要、工具名、耗时、token 用量与结论，不含病例原文与工具结果正文，
因此可以投屏演示。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from a2a.store import DEFAULT_DB_PATH, CaseStore  # noqa: E402

COLUMNS = ("seq", "event", "step", "agent", "tool_name", "ok",
           "elapsed_ms", "input_tokens", "output_tokens", "tool_count",
           "usage_estimated", "verdict", "reason_code")


def format_rows(rows: list[dict]) -> str:
    head = ("seq", "event", "step", "agent", "tool", "ok", "ms",
            "in", "out", "tools", "est", "verdict/reason")
    lines = ["  ".join(head)]
    for row in rows:
        verdict = row.get("verdict") or row.get("reason_code") or ""
        lines.append("  ".join(str(x) for x in (
            row.get("seq", ""), row.get("event", ""), row.get("step", ""),
            row.get("agent", ""), row.get("tool_name") or "-",
            "-" if row.get("ok") is None else ("ok" if row.get("ok") else "FAIL"),
            "-" if row.get("elapsed_ms") is None else row.get("elapsed_ms"),
            row.get("input_tokens", 0), row.get("output_tokens", 0),
            row.get("tool_count", 0), "Y" if row.get("usage_estimated") else "-",
            verdict)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="打印 case 脱敏执行轨迹")
    parser.add_argument("--case", required=True, help="case_id（对话为 chat-<conversationId>）")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"轨迹库路径（默认 {DEFAULT_DB_PATH}）")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--json", action="store_true", help="输出原始 JSON 行")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        print(f"轨迹库不存在: {args.db}", file=sys.stderr)
        return 1
    store = CaseStore(args.db)
    try:
        rows = store.list_trace(args.case, limit=args.limit)
    finally:
        store.close()
    if not rows:
        print(f"case {args.case} 暂无轨迹")
        return 0

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(format_rows(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

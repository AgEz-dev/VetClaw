"""A2A 流式帧协议：把内部事件序列化为 SSE。

为什么单独一个模块
------------------
流式有三个容易做错的地方，集中在一处才好钉住：

1. **帧格式必须与旧接口一致**。医院侧的 `VetClawAiProvider` 已经有一套成熟的中继
   （把 6 类事件按到达顺序转成前端帧、`done` 补发 delta、`error` 带 trace_id）。
   只要本层的帧与 `/api/chat/stream` 同构，将来切换就是改一个 URL，而不是重写中继。
2. **SSE 的 data 内换行必须转成续行** `data:`，否则会破坏空行分帧（这是 SSE 规范，
   不是可选优化）。
3. **一次请求有且只有一个终态**帧（`done` 或 `error`）。多一个终态会让客户端重复收尾，
   少一个会让客户端永远等下去。

事件词汇沿用旧接口的六个：`start` / `thought` / `tool_call` / `tool_result` /
`token` / `done` / `error`（`start` 为新增，用于让客户端在首帧绑定 task 与幂等状态）。
"""
from __future__ import annotations

import json

from a2a.contract import CaseTask, TaskResult

# ---- 事件名（与医院侧中继的 switch 对齐） ----
EVENT_START = "start"
EVENT_THOUGHT = "thought"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"

# ---- 内部标记：不对外发送。能力位用它把最终结论交回 supervisor ----
INTERNAL_OUTCOME = "_outcome"


def to_sse(event: str, data: dict) -> str:
    """序列化为一帧 SSE。data 内换行转续行（规范要求）。"""
    payload = json.dumps(data, ensure_ascii=False).replace("\n", "\ndata: ")
    return f"event: {event}\ndata: {payload}\n\n"


def frame(event: str, data: dict) -> dict:
    """内部事件（未序列化），便于在不碰字符串的前提下做单元测试。"""
    return {"event": event, "data": data}


def start_frame(task: CaseTask, deduped: bool = False) -> dict:
    """首帧：让客户端在收到任何内容前就能绑定 case/task 与「是不是回放」。"""
    return frame(EVENT_START, {
        "case_id": task.case_id,
        "task_id": task.task_id,
        "step": task.step.value,
        "trace_id": task.trace_id,
        "deduped": deduped,
    })


def token_frame(text: str) -> dict:
    return frame(EVENT_TOKEN, {"text": text})


def done_frame(result: TaskResult) -> dict:
    """终态帧。字段是旧 `done` 事件的**超集**：旧字段（answer/citations）原样保留，
    因此医院侧现有中继无需改动即可消费。"""
    return frame(EVENT_DONE, {
        "answer": result.payload.get("answer", ""),
        "citations": result.citations,
        # ---- A2A 扩展字段（旧中继会忽略未知字段）----
        "case_id": result.case_id,
        "task_id": result.task_id,
        "step": result.step.value,
        "status": result.status.value,
        "verdict": result.verdict.value,
        "confidence": result.confidence,
        "trace_id": result.trace_id,
        "deduped": result.deduped,
    })


def error_frame(code: str, message: str, trace_id: str = "") -> dict:
    return frame(EVENT_ERROR, {"code": code, "message": message, "trace_id": trace_id})

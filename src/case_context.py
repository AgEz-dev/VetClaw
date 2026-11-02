"""当前工具调用的上下文（ContextVar）：case_id + 角色。

工具参数由模型生成，不能把 case_id / 角色塞进工具签名——模型会照抄或编造。
执行层在**本次调用的同步栈内**设置，工具只读；没有 A2A case 的调用（单测、脚本）
取到空 case 与默认会话角色。

为什么用 ContextVar 而不是全局：两个流并发时全局会串号（A 病例的转人工工单
可能挂到 B 病例上）。且 set/reset 必须在同步的工具调用内完成——SSE 生成器会在
不同上下文里恢复，跨上下文 reset 会抛 ``Token was created in a different Context``。
"""
from __future__ import annotations

from contextvars import ContextVar, Token

_current: ContextVar[tuple[str, str]] = ContextVar(
    "current_tool_context", default=("", "consult"))


def set_context(case_id: str, role: str = "consult") -> Token:
    return _current.set((case_id or "", role or "consult"))


def reset_context(token: Token) -> None:
    _current.reset(token)


def set_case_id(case_id: str, role: str = "consult") -> Token:
    return set_context(case_id, role)


def reset_case_id(token: Token) -> None:
    reset_context(token)


def get_case_id() -> str:
    return _current.get()[0]


def get_role() -> str:
    return _current.get()[1]

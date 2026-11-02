"""Agent 轨迹的口径常量与脱敏工具。

轨迹只记录**结构化摘要**，不落任何病例原文：
- 工具入参只存 SHA-256 摘要（``args_digest``），不存 query / 过敏史 / 处方明细；
- 工具结果不存正文，只存 ``ok`` 与 ``error_code``；
- 模型与提示词用**版本号**标识（system prompt 的短哈希），便于对比不同版本的执行差异。
"""
from __future__ import annotations

import hashlib

# 事件词汇（与 SSE 帧名刻意区分：帧是给客户端的，轨迹是给工程复盘的）
EVENT_HANDOFF = "handoff"        # 前置分诊把任务交给能力位（或直接短路）
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_VERDICT = "verdict"        # 本次任务的最终结论
EVENT_ERROR = "error"

# 轨迹里的 agent 归属
AGENT_SUPERVISOR = "supervisor"
AGENT_TRIAGE = "triage"
AGENT_TOOL = "tool"


def prompt_version(system_prompt: str | None) -> str:
    """提示词版本 = 内容短哈希。改了 prompt 就会换版本，历史轨迹可分组对比。"""
    if not system_prompt:
        return ""
    digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    return f"p_{digest[:8]}"

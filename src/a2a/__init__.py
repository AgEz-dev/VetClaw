"""A2A 编排层：任务契约 + case 状态 + supervisor。

模块划分（单向依赖，contract 不依赖任何业务模块）：

- ``contract``：结构化消息与状态对象（纯数据、零依赖）
- ``store``：case 状态与幂等回放（SQLite / 标准库）
- ``agents``：四个能力位的适配器（包既有业务模块，不改它们的语义）
- ``supervisor``：编排器（加载 case → 幂等检查 → 派发 → 落库 → 返回结果）
"""
from a2a.contract import (
    CaseState,
    CaseTask,
    Step,
    TaskResult,
    TaskStatus,
    Verdict,
    fingerprint,
    new_id,
    worst_verdict,
)

__all__ = [
    "CaseState", "CaseTask", "Step", "TaskResult", "TaskStatus", "Verdict",
    "fingerprint", "new_id", "worst_verdict",
]

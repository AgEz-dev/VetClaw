"""A2A 请求/响应模型。

与既有的 `schemas/chat.py` 分开，因为 A2A 是**编排层**契约（case / 任务 / 幂等键），
而 chat 是**能力层**契约（一次问答）。两者生命周期不同：chat 请求无状态、可随意重放；
A2A 任务有 case 上下文、必须带幂等键。
"""
from typing import Literal

from pydantic import BaseModel, Field

StepName = Literal["triage", "intake", "draft", "review", "consult"]
"""`consult`（问答）**仅**支持流式端点 `/a2a/task/stream`；提交到同步端点会得到
`STEP_NOT_STREAMABLE` 语义的执行失败与 422 之外的明确提示。"""


class A2aCaseOpen(BaseModel):
    """建案请求。``case_id`` 由调用方派生（建议 ``appt-<预约id>``），保证同一就诊可重复建。"""

    case_id: str = Field(min_length=1)
    patient: dict = Field(default_factory=dict)
    """宠物事实：``species`` / ``weight_kg`` / ``age_weeks`` / ``allergies`` / ``pregnant``。
    **同一 case 只需录一次**，后续能力位自动继承。"""


class A2aTaskRequest(BaseModel):
    """提交一次能力位任务。

    ``idempotency_key`` 必填：医院侧重试、网关重放、双写重发都要靠它保证只执行一次。
    **同一个键提交不同内容会被拒绝（409）**，不会拿旧结论冒充新请求。
    """

    step: StepName
    case_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    payload: dict = Field(default_factory=dict)
    task_id: str = ""
    caller: str = ""
    deadline_at: float | None = None
    """epoch 秒。超时任务直接失败，不做无意义重计算。"""
    confidence_threshold: float | None = None
    """低于此置信度时由调用方转入 HITL（本服务只在结果里回传 confidence）。"""

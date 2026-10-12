"""请求/响应 Pydantic 模型。"""
from typing import Literal

from pydantic import BaseModel, Field


class PatientContext(BaseModel):
    """宠物档案上下文。

    由**调用方（业务系统）组装好的纯文本字段**，本服务只作为背景参考，
    **不反查业务库**——保持单向依赖，避免循环调用。
    字段全部可选：缺失即视为"无信息"，不参与渲染。
    """

    pet_id: int | None = None
    species: str | None = None          # 物种：犬 / 猫 / 异宠
    breed: str | None = None            # 品种
    age_years: int | None = None        # 年龄（岁）
    weight_kg: float | None = None      # 体重（kg）
    allergies: str | None = None        # 过敏史
    latest_diagnosis: str | None = None  # 最近一次诊断
    vaccine_alert_count: int | None = None  # 疫苗逾期/将到期提醒数


class ChatRequest(BaseModel):
    """对话请求。

    ``mode`` 区分两种产出（同一套对话能力，两种输出）：
    - ``consult``（默认）：对**宠主**说话，给建议 + 引用溯源 —— 既有行为，不变；
    - ``intake``：**预问诊** —— 对**医生**说话，输出结构化摘要（契约 v1.2 §3.4）。

    ``intake_state``：预问诊的**不透明续传状态**（由上一轮响应原样带回）。
    本服务保持无状态：不押会话、不需要 session id。
    """

    prompt: str = Field(min_length=1)
    history: list[dict] = Field(default_factory=list)
    patient_context: PatientContext | None = None
    mode: Literal["consult", "intake"] = "consult"
    intake_state: dict | None = None
    intake_archive: dict | None = None
    """预问诊的**档案槽位**（由业务系统组装好的短文本）：``vaccine`` / ``past_disease`` /
    ``medications`` / ``allergy``。**有值就不问宠主** —— 这是"少问一半问题"的来源。
    与 ``patient_context`` 的分工：后者进 LLM 提示，本字段进**槽位预填**。"""


class ChatResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)

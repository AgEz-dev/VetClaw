"""请求/响应 Pydantic 模型。"""
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
    prompt: str = Field(min_length=1)
    history: list[dict] = Field(default_factory=list)
    patient_context: PatientContext | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)

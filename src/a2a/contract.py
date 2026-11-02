"""A2A 任务契约：Agent 之间**只传结构化消息**，绝不传自然语言。

设计要点（对应「把编排权收进 VetClaw」这条路线）：

- **case 是编排的一等公民**：一次就诊 = 一个 case，四个能力位（预问诊 / 分诊 /
  起草 / 审核）共享同一份 case 状态。这是对旧契约的根本修正 —— 旧契约下医院侧
  每次调用都传 `history: []`（`VetClawAiProvider.buildBody` 的「单轮语义」），
  四个能力位彼此完全不知情。
- **幂等键 + 请求指纹**：`idempotency_key` 相同、`payload` 指纹也相同 → 回放上次
  结果；指纹**不同** → 拒绝（409 语义），绝不用旧结果冒充新请求。只做前者是
  常见事故：重试换了内容却拿到上一次的答案。
- **执行态与结论分离**：`TaskStatus` 说「任务跑没跑成」，`Verdict` 说「审核结论」。
  把这两件事混成一个字段，下游闸门就没法区分「模型挂了」与「药有问题」——
  前者该重试，后者该拦人。
- 本模块**零第三方依赖、不 import 任何业务模块**，便于两侧共用与离线单测。
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# --------------------------------------------------------------------- 枚举

class Step(str, Enum):
    """就诊链路上的能力位。

    前四个是**业务闭环**（顺序即业务顺序）；`CONSULT` 是**对话位**，单独一类：
    它需要逐字流式下发，走同步接口会丢掉流式体验，因此只支持流式端点
    （`POST /a2a/task/stream`）。

    ⚠️ 现状：`CONSULT` 的流式能力已实现并有测试，但**医院侧尚未切换**
    （问答仍走旧的 `/api/chat/stream`）。切换时只需医院侧把请求指向
    `/a2a/task/stream` 并补 `case_id` + 幂等键 —— 帧格式与旧接口同构。
    """

    TRIAGE = "triage"    # 分诊：P0 急症 / 未知处方药拦截
    INTAKE = "intake"    # 预问诊：多轮追问出结构化摘要
    DRAFT = "draft"      # 病历起草：SOAP 叙述性草稿
    REVIEW = "review"    # 处方安全审核：发药前闸门
    CONSULT = "consult"  # 问答：对宠主说话，逐字流式（**仅流式端点**）


class TaskStatus(str, Enum):
    """**执行**结果。"""

    OK = "ok"                # 跑完了，有结论
    NEED_MORE = "need_more"  # 输入不足，需调用方补字段（不是错误）
    FAILED = "failed"        # 执行失败（模型不可用 / 解析失败 / 内部错误）


class Verdict(str, Enum):
    """**业务**结论（仅 REVIEW 步有值，其余步为 NONE）。

    取值刻意与医院侧闸门语义一一对应，下游不需要再翻译：
    - PASS  → 可发药
    - WARN  → 可发药但需医生确认（HITL 提示）
    - BLOCK → 禁止发药，必须人工复核
    """

    NONE = "none"
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


# 结论的严重度排序：合并多个结论时取最严重者（BLOCK > WARN > PASS > NONE）
_SEVERITY = {Verdict.NONE: 0, Verdict.PASS: 1, Verdict.WARN: 2, Verdict.BLOCK: 3}


def worst_verdict(*verdicts: Verdict) -> Verdict:
    """合并结论：取最严重的一个。**规则层的 BLOCK 永远不会被模型层的 PASS 覆盖。**"""
    result = Verdict.NONE
    for v in verdicts:
        if v is None:
            continue
        if _SEVERITY[v] > _SEVERITY[result]:
            result = v
    return result


# --------------------------------------------------------------------- 工具函数

def now() -> float:
    return time.time()


def new_id(prefix: str = "") -> str:
    rid = uuid.uuid4().hex[:16]
    return f"{prefix}{rid}" if prefix else rid


def canonical_json(value: Any) -> str:
    """规范化 JSON：键排序 + 紧凑分隔符，保证同一内容指纹稳定。"""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(payload: Any) -> str:
    """请求载荷指纹：用于识别「同幂等键、不同内容」的冲突重试。"""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- 契约对象

@dataclass
class CaseTask:
    """一次能力位调用请求（由调用方提交）。

    医院侧只需把**结构化数据**放进来（宠物事实 / 处方明细 / 预问诊摘要），
    不需要描述「下一步做什么」——编排由 supervisor 负责，调用方不再承担编排。
    """

    step: Step
    case_id: str
    idempotency_key: str
    payload: dict = field(default_factory=dict)
    task_id: str = ""
    caller: str = ""
    trace_id: str = ""
    deadline_at: float | None = None          # epoch 秒；None = 不限
    confidence_threshold: float | None = None  # 低于此值触发 HITL 提示

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = new_id("task_")
        if not self.trace_id:
            self.trace_id = new_id("trace_")
        if not self.idempotency_key:
            raise ValueError("idempotency_key 必填：无幂等键的重试会导致重复执行")
        if not self.case_id:
            raise ValueError("case_id 必填：case 是编排的一等公民")

    @property
    def request_fingerprint(self) -> str:
        return fingerprint({"step": self.step.value, "case_id": self.case_id,
                            "payload": self.payload})

    def is_expired(self, clock: float | None = None) -> bool:
        if self.deadline_at is None:
            return False
        return (clock if clock is not None else now()) > self.deadline_at


@dataclass
class TaskResult:
    """一次能力位调用的结果（由 supervisor 返回，并写入 case 状态）。"""

    case_id: str
    task_id: str
    step: Step
    status: TaskStatus
    verdict: Verdict = Verdict.NONE
    payload: dict = field(default_factory=dict)
    confidence: float = 1.0
    reasons: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    trace_id: str = ""
    created_at: float = field(default_factory=now)
    deduped: bool = False      # True = 本次是幂等回放，未真实执行

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "task_id": self.task_id,
            "step": self.step.value, "status": self.status.value,
            "verdict": self.verdict.value, "payload": self.payload,
            "confidence": self.confidence, "reasons": self.reasons,
            "citations": self.citations, "trace_id": self.trace_id,
            "created_at": self.created_at, "deduped": self.deduped,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskResult:
        return cls(
            case_id=data["case_id"], task_id=data["task_id"],
            step=Step(data["step"]), status=TaskStatus(data["status"]),
            verdict=Verdict(data.get("verdict", "none")),
            payload=data.get("payload") or {}, confidence=data.get("confidence", 1.0),
            reasons=data.get("reasons") or [], citations=data.get("citations") or [],
            trace_id=data.get("trace_id", ""), created_at=data.get("created_at", 0.0),
            deduped=data.get("deduped", False),
        )

    def replay(self) -> TaskResult:
        """幂等回放：返回副本并标记 deduped。"""
        clone = TaskResult.from_dict(self.to_dict())
        clone.deduped = True
        return clone


@dataclass
class CaseState:
    """case 级累计状态：四个能力位共享的就诊上下文。

    这是「真 A2A」与「四个独立接口」的分水岭 —— 起草时不再需要调用方把
    预问诊摘要重传一遍，它已经在 `steps[INTAKE]` 里。
    """

    case_id: str
    patient: dict = field(default_factory=dict)   # 宠物事实（species/weight_kg/allergies…）
    steps: dict = field(default_factory=dict)     # {Step.value: TaskResult.to_dict()}
    handoffs: list = field(default_factory=list)  # 结构化交接记录（不含自然语言）
    memories: list = field(default_factory=list)  # L3 长期事实（来自医院库，只读快照）
    memory_loaded: bool = False                   # 本 case 是否已尝试加载过 L3（避免反复取数）
    created_at: float = field(default_factory=now)
    updated_at: float = field(default_factory=now)
    seq: int = 0                                  # 单调递增，供乐观并发与审计

    def record(self, result: TaskResult) -> CaseState:
        self.steps[result.step.value] = result.to_dict()
        self.updated_at = now()
        self.seq += 1
        return self

    def record_handoff(self, handoff: dict) -> CaseState:
        """追加交接记录；同一位置重复重试时不重复记账。

        ``seq`` 只为能力位结果递增 —— handoff 是结果的派生事件，拿``case_seq``
        指回它发生在哪一步之后，不单独占用一个就诊序号。
        """
        if not self.handoffs or self.handoffs[-1] != handoff:
            self.handoffs.append(dict(handoff))
            self.updated_at = now()
        return self

    def result_of(self, step: Step) -> TaskResult | None:
        raw = self.steps.get(step.value)
        return TaskResult.from_dict(raw) if raw else None

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "patient": self.patient, "steps": self.steps,
                "handoffs": self.handoffs, "memories": self.memories,
                "memory_loaded": self.memory_loaded,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "seq": self.seq}

    @classmethod
    def from_dict(cls, data: dict) -> CaseState:
        return cls(case_id=data["case_id"], patient=data.get("patient") or {},
                   steps=data.get("steps") or {},
                   handoffs=data.get("handoffs") or [],
                   memories=data.get("memories") or [],
                   memory_loaded=bool(data.get("memory_loaded", False)),
                   created_at=data.get("created_at", 0.0),
                   updated_at=data.get("updated_at", 0.0), seq=data.get("seq", 0))

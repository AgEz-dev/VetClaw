"""多 Agent 角色注册表：四份角色提示词 + 硬编码工具白名单 + 有限 handoff 边。

三条纪律写进代码，而不是写进提示词：

1. **工具白名单**：每个角色只能调用自己那份工具集。模型看不到白名单外的工具
   （``RoleScopedRegistry.schemas``），即使凭空编造工具名也会被执行层拒绝
   （``core_agent.execute_tool``）。提示词里写「你只能用某工具」约束不住模型，
   执行层拒绝才算数。
2. **handoff 只传结构化消息**：``Handoff`` 只有 from / to / reason_code / case_seq /
   payload_digest，没有自然语言。payload 只落 SHA-256 摘要，病例原文不进轨迹。
3. **边是有限集合**：只有 ``HANDOFF_EDGES`` 里的转移会被记录；未登记的边抛
   ``HandoffDenied``。跨系统的 ``draft→human`` 与 ``human→review`` 也显式建边，
   让「哪一段必须人工签字」在轨迹里可见。

角色是**同一进程内的四份提示词 + 四份工具集**，不是四个进程。它们共享
``CaseState`` 的事实，但不共享草稿。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from a2a.contract import Step, TaskStatus, Verdict, fingerprint
from a2a.trace import prompt_version

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- 角色与节点

ROLE_TRIAGE = Step.TRIAGE.value
ROLE_INTAKE = Step.INTAKE.value
ROLE_DRAFT = Step.DRAFT.value
ROLE_REVIEW = Step.REVIEW.value
ROLE_CONSULT = Step.CONSULT.value

NODE_HUMAN = "human"
NODE_DONE = "done"
NODE_FAILED = "failed"

ROLE_LABELS = {
    ROLE_TRIAGE: "分诊",
    ROLE_INTAKE: "预问诊",
    ROLE_DRAFT: "记录",
    ROLE_REVIEW: "审方",
    ROLE_CONSULT: "会话位",
}

ROLE_NODES = frozenset(ROLE_LABELS)

# --------------------------------------------------------------------- 工具白名单
#
# 与提示词里的「可用工具」一一对应。会话位需要转人工（唯一写工具），
# 因此 ⑦ 显式列入会话位；计划表的 ①②③④⑤⑥ 漏列了它，但演示与现有问答链路
# 都依赖 escalate_to_human，不能为了对齐表格把它拿掉。

ROLE_TOOLS: dict[str, frozenset[str]] = {
    ROLE_TRIAGE: frozenset({
        "search_knowledge_base", "check_drug_safety", "get_pet_profile"}),
    ROLE_INTAKE: frozenset({
        "get_pet_profile", "get_vaccination_history", "get_medication_history"}),
    ROLE_DRAFT: frozenset({
        "search_knowledge_base", "get_pet_profile", "get_medication_history"}),
    ROLE_REVIEW: frozenset({
        "lookup_drug_label", "check_drug_safety"}),
    ROLE_CONSULT: frozenset({
        "search_knowledge_base", "lookup_drug_label", "check_drug_safety",
        "get_pet_profile", "get_vaccination_history", "get_medication_history",
        "get_inventory_summary", "escalate_to_human"}),
}


def tool_allowed(role: str, tool_name: str) -> bool:
    """未知角色一律拒绝（白名单是默认拒绝，不是默认放行）。"""
    return tool_name in ROLE_TOOLS.get(role, frozenset())


class RoleScopedRegistry:
    """按角色过滤工具注册表：模型只看到本角色可用的工具。

    只代理 ``schemas`` / ``call`` / ``get`` 三个方法 —— 与 ``ToolRegistry`` 的
    双引擎用法完全一致。真正的拒绝仍由 ``execute_tool`` 兜底，形成「看不到 +
    调不动」双层约束。
    """

    def __init__(self, base, role: str):
        self._base = base
        self.role = role

    def schemas(self) -> list[dict]:
        return [s for s in self._base.schemas()
                if tool_allowed(self.role, s["function"]["name"])]

    def get(self, name: str):
        if not tool_allowed(self.role, name):
            raise PermissionError(f"角色 {self.role} 无权调用工具 {name}")
        return self._base.get(name)

    def call(self, name: str, arguments: dict):
        if not tool_allowed(self.role, name):
            raise PermissionError(f"角色 {self.role} 无权调用工具 {name}")
        return self._base.call(name, arguments)


# --------------------------------------------------------------------- 提示词

_PROMPTS_DIR = Path(os.getenv(
    "VETCLAW_PROMPTS_DIR",
    Path(__file__).resolve().parents[2] / "prompts" / "roles"))


@lru_cache(maxsize=None)
def role_prompt(role: str) -> str:
    """读取角色提示词正文；缺失时返回空串（由调用方决定兜底）。"""
    path = _PROMPTS_DIR / f"{role}.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("角色提示词缺失: %s", path)
        return ""


def role_prompt_version(role: str) -> str:
    """角色提示词版本 = 内容短哈希。改 prompt 必然换版本，历史轨迹可分组对比。"""
    return prompt_version(role_prompt(role))


# --------------------------------------------------------------------- handoff

# (from, to, reason_code)。任何角色执行失败统一走 to=failed / reason=FAILED。
HANDOFF_EDGES: frozenset[tuple[str, str, str]] = frozenset({
    (ROLE_TRIAGE, ROLE_INTAKE, "NEED_MORE"),
    (ROLE_INTAKE, ROLE_TRIAGE, "SLOTS_READY"),
    (ROLE_TRIAGE, ROLE_DRAFT, "PASS"),
    (ROLE_DRAFT, NODE_HUMAN, "DRAFT_READY"),
    (NODE_HUMAN, ROLE_REVIEW, "RX_SUBMITTED"),
    (ROLE_REVIEW, NODE_HUMAN, "BLOCK"),
    (ROLE_REVIEW, NODE_HUMAN, "WARN"),
    (ROLE_REVIEW, NODE_DONE, "PASS"),
    # 分诊的急症 / 拒答复核是「直接交给人」，不是交给下一个能力位
    (ROLE_TRIAGE, NODE_HUMAN, "BLOCK"),
    (ROLE_TRIAGE, NODE_HUMAN, "WARN"),
    # 预问诊中途命中急症：立即中断，交人
    (ROLE_INTAKE, NODE_HUMAN, "EMERGENCY_INTERRUPTED"),
})


class HandoffDenied(ValueError):
    """未登记的 handoff 边 —— 拒绝记录，避免上下文污染。"""


@dataclass(frozen=True)
class Handoff:
    """角色之间唯一的交接物：**没有自然语言字段**。"""

    from_role: str
    to_role: str
    reason_code: str
    case_seq: int
    payload_digest: str

    def to_dict(self) -> dict:
        return {
            "from": self.from_role,
            "to": self.to_role,
            "reason_code": self.reason_code,
            "case_seq": self.case_seq,
            "payload_digest": self.payload_digest,
        }


def is_allowed_handoff(from_role: str, to_role: str, reason_code: str) -> bool:
    if (from_role, to_role, reason_code) in HANDOFF_EDGES:
        return True
    # 计划里的「any → failed」：失败不是业务转移，任何角色都可以落到 failed
    return (to_role == NODE_FAILED and reason_code == "FAILED"
            and from_role in ROLE_NODES)


def make_handoff(from_role: str, to_role: str, reason_code: str, case_seq: int,
                 payload: dict | None = None) -> Handoff:
    if not is_allowed_handoff(from_role, to_role, reason_code):
        raise HandoffDenied(f"未登记的 handoff 边: {from_role}->{to_role} ({reason_code})")
    return Handoff(from_role=from_role, to_role=to_role,
                   reason_code=reason_code, case_seq=int(case_seq),
                   payload_digest=fingerprint(payload or {}))


def failed_handoff(step: Step | str, case_seq: int,
                   payload: dict | None = None) -> Handoff:
    role = step.value if isinstance(step, Step) else str(step)
    return make_handoff(role, NODE_FAILED, "FAILED", case_seq, payload)


def outcome_handoff(step: Step | str, status: TaskStatus, verdict: Verdict,
                    payload: dict | None, case_seq: int) -> Handoff | None:
    """一次能力位执行完成后，根据**执行态 + 结论**决定下一位是谁。

    规则来自计划 §6.2 的有限边集合：
    - 分诊：急症/拒答 → human；信息不足 → intake；信息齐 → draft
    - 预问诊：急症中断 → human；收尾 → triage 复检；追问中 → 留在本角色
    - 记录：草稿完成 → human（医生签字）
    - 审方：BLOCK/WARN → human；PASS → done
    """
    role = step.value if isinstance(step, Step) else str(step)
    payload = payload or {}

    if status is TaskStatus.FAILED:
        return failed_handoff(role, case_seq, payload)

    if role == ROLE_TRIAGE:
        if verdict is Verdict.BLOCK:
            return make_handoff(ROLE_TRIAGE, NODE_HUMAN, "BLOCK", case_seq, payload)
        if verdict is Verdict.WARN:
            return make_handoff(ROLE_TRIAGE, NODE_HUMAN, "WARN", case_seq, payload)
        if status is TaskStatus.NEED_MORE:
            return make_handoff(ROLE_TRIAGE, ROLE_INTAKE, "NEED_MORE", case_seq, payload)
        if status is TaskStatus.OK and verdict is Verdict.PASS:
            return make_handoff(ROLE_TRIAGE, ROLE_DRAFT, "PASS", case_seq, payload)
        return None

    if role == ROLE_INTAKE:
        snapshot = payload.get("intake") if isinstance(payload.get("intake"), dict) else {}
        if payload.get("interrupted") or snapshot.get("interrupted"):
            return make_handoff(ROLE_INTAKE, NODE_HUMAN,
                                "EMERGENCY_INTERRUPTED", case_seq, payload)
        if payload.get("done") is True or snapshot:
            return make_handoff(ROLE_INTAKE, ROLE_TRIAGE, "SLOTS_READY", case_seq, payload)
        return None

    if role == ROLE_DRAFT:
        if status is TaskStatus.OK and isinstance(payload.get("draft"), dict):
            return make_handoff(ROLE_DRAFT, NODE_HUMAN, "DRAFT_READY", case_seq, payload)
        return None

    if role == ROLE_REVIEW:
        if verdict is Verdict.BLOCK:
            return make_handoff(ROLE_REVIEW, NODE_HUMAN, "BLOCK", case_seq, payload)
        if verdict is Verdict.WARN:
            return make_handoff(ROLE_REVIEW, NODE_HUMAN, "WARN", case_seq, payload)
        if verdict is Verdict.PASS:
            return make_handoff(ROLE_REVIEW, NODE_DONE, "PASS", case_seq, payload)
        return None

    return None

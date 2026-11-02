"""supervisor：A2A 编排器。**编排权在这里，不在调用方。**

一次 `submit(task)` 的固定顺序：

1. **截止时间**：过期任务直接失败，不做无意义的重计算。
2. **幂等检查**：同 `idempotency_key` + 同请求指纹 → 回放上次结果（`deduped=True`）；
   指纹不同 → 抛 `IdempotencyConflict`（409 语义）。**绝不静默返回旧结论。**
3. **加载 case**：不存在则创建；已存在则**复用其累计上下文**（这是与旧契约的分水岭）。
4. **派发**到对应能力位，把 `case` 一并传下去（能力位可从中补全缺省输入）。
5. **落库**：成功/待补的结果写入 case 状态，并记入幂等表。
6. 返回结构化 `TaskResult`。

失败结果的记忆化策略（重要）
--------------------------
只有 `OK` / `NEED_MORE` 会被写入幂等表；`FAILED` **不记忆**。理由：幂等键的语义是
「同一次业务意图」，而模型超时、网关 5xx 属于**同一次意图的瞬时失败**，调用方用
**同一个键**重试是正确行为。若把失败也记忆进去，一次抖动就会让该任务永久失败，
调用方只能换键 —— 那等于放弃幂等保护。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from a2a.agents import AGENTS, STREAM_AGENTS, AgentContext, AgentOutcome, triage_agent
from a2a.contract import (
    CaseState,
    CaseTask,
    Step,
    TaskResult,
    TaskStatus,
    Verdict,
    worst_verdict,
)
from a2a.roles import (
    NODE_FAILED,
    NODE_HUMAN,
    ROLE_REVIEW,
    ROLE_TRIAGE,
    Handoff,
    failed_handoff,
    make_handoff,
    outcome_handoff,
    role_prompt_version,
)
from a2a.store import CaseStore
from a2a.stream import (
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    INTERNAL_OUTCOME,
    done_frame,
    error_frame,
    start_frame,
    token_frame,
)
from a2a.trace import (
    AGENT_SUPERVISOR,
    AGENT_TOOL,
    EVENT_ERROR,
    EVENT_HANDOFF,
    EVENT_TOOL_CALL as TRACE_TOOL_CALL,
    EVENT_TOOL_RESULT as TRACE_TOOL_RESULT,
    EVENT_VERDICT,
)

logger = logging.getLogger(__name__)

# 前置分诊覆盖的能力位：只覆盖**自由文本入口**。
#
# 起草（draft）刻意不在集合内：它的入参是预问诊产出的结构化摘要，
# 对摘要再扫一遍毒物词只会把「已在处置中的病例」误拦，收益为零；
# 而它的自由文本来源（预问诊/问答）都已被本闸门覆盖。
_PRETRIAGE_STEPS = {Step.INTAKE, Step.CONSULT}


def _elapsed_ms(start: float) -> int:
    """整步耗时（毫秒）：分诊 + 能力位执行 + 落库都计入，调用方看到的是端到端。"""
    return int((time.monotonic() - start) * 1000)


class IdempotencyConflict(RuntimeError):
    """同一幂等键被用于**不同内容**的请求 —— 必须让调用方显式处理。"""


@dataclass
class PreTriage:
    """编排层前置分诊的结论。

    ``outcome`` 非空 = **短路**：BLOCK 时原能力位根本不执行（不调 LLM、不推进会话）。
    ``WARN`` 只合并结论、照常执行 —— 拒答类命中仍由能力位自己给话术，
    编排层不替它说话，避免同一个问题出现两种口径。
    """

    verdict: Verdict = Verdict.NONE
    outcome: AgentOutcome | None = None
    reasons: list = field(default_factory=list)


class Supervisor:
    def __init__(self, store: CaseStore, ctx: AgentContext | None = None,
                 agents: dict | None = None, stream_agents: dict | None = None):
        self.store = store
        self.ctx = ctx or AgentContext()
        self.agents = agents or AGENTS
        self.stream_agents = stream_agents if stream_agents is not None else STREAM_AGENTS

    # ------------------------------------------------------------------ 主入口

    def submit(self, task: CaseTask) -> TaskResult:
        trace = task.trace_id

        # 1. 截止时间
        if task.is_expired():
            self._trace_error(task.case_id, task.step.value, "DEADLINE_EXCEEDED")
            return self._result(task, TaskStatus.FAILED, payload={
                "code": "DEADLINE_EXCEEDED",
                "message": "任务已超过截止时间，未执行"})

        # 2. 幂等
        existing = self.store.get_result(task.idempotency_key)
        if existing is not None:
            stored_fp, stored_result = existing
            if stored_fp != task.request_fingerprint:
                self._trace_error(task.case_id, task.step.value, "IDEMPOTENCY_CONFLICT")
                raise IdempotencyConflict(
                    f"幂等键 {task.idempotency_key} 已用于不同内容的请求"
                    f"（本次指纹 {task.request_fingerprint[:12]}…，"
                    f"已存指纹 {stored_fp[:12]}…）")
            logger.info("a2a 幂等回放 key=%s step=%s trace=%s",
                        task.idempotency_key, task.step.value, trace)
            return stored_result.replay()

        # 3. 加载 / 创建 case
        case = self.store.get_case(task.case_id)
        if case is None:
            patient = (task.payload or {}).get("patient") or {}
            case = self.store.create_case(task.case_id, patient)

        # 3.1 L3 长期记忆：同一 case 只取一次，失败降级为空
        self._ensure_memories(case, task)

        # 3.2 人 → 审方：处方是医生签字后才提交的，这条边必须留在轨迹里
        self._admit_handoff(case, task)

        # 3.5 前置分诊：BLOCK 在这里就结束，原能力位不执行
        task_started = time.monotonic()
        gate = self._pretriage(case, task)
        if gate.outcome is not None:
            return self._short_circuit(case, task, gate.outcome,
                                       elapsed_ms=_elapsed_ms(task_started))

        # 4. 派发
        agent = self.agents.get(task.step)
        if agent is None:
            # 流式专属能力位走错端点 → 给明确指引，而不是含糊的「未知能力位」
            if task.step in self.stream_agents:
                self._trace_error(task.case_id, task.step.value, "STEP_NOT_STREAMABLE")
                return self._result(task, TaskStatus.FAILED, payload={
                    "code": "STEP_NOT_STREAMABLE",
                    "message": f"能力位 {task.step.value} 只支持流式，"
                               f"请使用 POST /a2a/task/stream"})
            self._trace_error(task.case_id, task.step.value, "UNKNOWN_STEP")
            return self._result(task, TaskStatus.FAILED, payload={
                "code": "UNKNOWN_STEP", "message": f"未知能力位: {task.step.value}"})

        try:
            outcome: AgentOutcome = agent(case, task, self.ctx)
        except Exception as exc:
            # 能力位内部异常不得冒到 HTTP 层；转成可重试的执行失败
            logger.exception("a2a agent 执行失败 step=%s trace=%s",
                             task.step.value, trace)
            self._trace_failure(case, task)
            return self._result(task, TaskStatus.FAILED, payload={
                "code": "AGENT_ERROR",
                "message": f"{task.step.value} 执行异常: {type(exc).__name__}"})

        # 5. 落库（WARN 由前置分诊合并，绝不让模型结论盖过规则命中）
        result = self._result(task, outcome.status,
                              verdict=worst_verdict(gate.verdict, outcome.verdict),
                              payload=outcome.payload, confidence=outcome.confidence,
                              reasons=[*gate.reasons, *(outcome.reasons or [])],
                              citations=outcome.citations)
        handoff = self._record_result(case, task, result)
        self._trace_handoff(task.case_id, handoff, task.step.value)
        self._trace_result(task.case_id, result,
                           elapsed_ms=_elapsed_ms(task_started), usage=outcome.usage)
        logger.info("a2a 执行完成 case=%s step=%s status=%s verdict=%s trace=%s",
                    task.case_id, task.step.value, outcome.status.value,
                    outcome.verdict.value, trace)
        return result

    # ------------------------------------------------------------------ 流式入口

    def assert_no_conflict(self, task: CaseTask) -> None:
        """流开始**之前**做一次同步冲突检查。

        为什么必须单独一步：SSE 握手一旦完成，HTTP 状态码就已经发出去了，
        再想返回 409 已经来不及。把冲突检查前置，客户端才能拿到真正的 HTTP 409。
        （生成器内部仍会再查一次，用于兜住检查与迭代之间的竞态。）
        """
        existing = self.store.get_result(task.idempotency_key)
        if existing is None:
            return
        stored_fp, _ = existing
        if stored_fp != task.request_fingerprint:
            raise IdempotencyConflict(
                f"幂等键 {task.idempotency_key} 已用于不同内容的请求")

    def submit_stream(self, task: CaseTask):
        """流式执行：产出 A2A 事件帧（dict），**有且只有一个终态**（done 或 error）。

        与同步入口的三处刻意差异：

        1. **回放不是重放 token**。流无法真正"续传"，所以回放时把已存的答案作为
           **一整段 token** 下发，再接 `done(deduped=true)`。客户端拿到的是完整内容，
           不是半截流。这一点必须写进契约，否则调用方会误以为能"从断点继续"。
        2. **失败不记忆**（与同步一致），因此断流后用同一个键重试会**重新执行**。
        3. **客户端提前断开则结果不落库**：执行已发生但结论没写进幂等表，
           重试会重跑一次。这是取舍——把半截结果落库比重跑更糟。
        """
        trace = task.trace_id

        if task.is_expired():
            self._trace_error(task.case_id, task.step.value, "DEADLINE_EXCEEDED")
            yield error_frame("DEADLINE_EXCEEDED", "任务已超过截止时间，未执行", trace)
            return

        existing = self.store.get_result(task.idempotency_key)
        if existing is not None:
            stored_fp, stored = existing
            if stored_fp != task.request_fingerprint:
                self._trace_error(task.case_id, task.step.value, "IDEMPOTENCY_CONFLICT")
                yield error_frame("IDEMPOTENCY_CONFLICT",
                                  f"幂等键 {task.idempotency_key} 已用于不同内容的请求",
                                  trace)
                return
            replay = stored.replay()
            yield start_frame(task, deduped=True)
            text = (replay.payload or {}).get("answer")
            if text:
                yield token_frame(str(text))
            yield done_frame(replay)
            return

        streamer = self.stream_agents.get(task.step)
        if streamer is None:
            self._trace_error(task.case_id, task.step.value, "STEP_NOT_STREAMABLE")
            yield error_frame(
                "STEP_NOT_STREAMABLE",
                f"能力位 {task.step.value} 不支持流式，请使用 POST /a2a/task", trace)
            return

        case = self.store.get_case(task.case_id)
        if case is None:
            case = self.store.create_case(
                task.case_id, (task.payload or {}).get("patient") or {})

        self._ensure_memories(case, task)

        self._admit_handoff(case, task)

        # 前置分诊先于任何一帧内容：BLOCK 时客户端只会看到 start + done，
        # 不会先收到半句"正在检索知识库"再被判急症。
        task_started = time.monotonic()
        gate = self._pretriage(case, task)

        yield start_frame(task)

        if gate.outcome is not None:
            yield done_frame(self._short_circuit(
                case, task, gate.outcome, elapsed_ms=_elapsed_ms(task_started)))
            return

        outcome: AgentOutcome | None = None
        # 工具耗时的唯一可信来源是 tool_call → tool_result 的间隔：
        # 工具结果帧的 content 格式是对外契约，不能为了埋点往里塞字段。
        pending_tools: dict[str, tuple[float, str]] = {}
        try:
            for ev in streamer(case, task, self.ctx):
                name = ev.get("event")
                data = ev.get("data") or {}
                if name == EVENT_TOOL_CALL:
                    call_id = str(data.get("id") or "")
                    pending_tools[call_id] = (time.monotonic(), str(data.get("name") or ""))
                    self._trace_tool_call(case.case_id, task.step.value, data)
                elif name == EVENT_TOOL_RESULT:
                    call_id = str(data.get("id") or "")
                    started, tool_name = pending_tools.pop(call_id, (time.monotonic(), ""))
                    self._trace_tool_result(case.case_id, task.step.value, tool_name,
                                            data, int((time.monotonic() - started) * 1000))
                if name == INTERNAL_OUTCOME:
                    outcome = data
                    continue
                yield ev
        except Exception:
            logger.exception("a2a 流式能力位异常 step=%s trace=%s",
                             task.step.value, trace)
            self._trace_failure(case, task)
            yield error_frame("AGENT_ERROR",
                              f"{task.step.value} 执行异常", trace)
            return

        if outcome is None:
            self._trace_failure(case, task, error_code="NO_OUTCOME")
            yield error_frame("NO_OUTCOME", "能力位未返回结论", trace)
            return

        result = self._result(task, outcome.status,
                              verdict=worst_verdict(gate.verdict, outcome.verdict),
                              payload=outcome.payload, confidence=outcome.confidence,
                              reasons=[*gate.reasons, *(outcome.reasons or [])],
                              citations=outcome.citations)
        handoff = self._record_result(case, task, result)
        self._trace_handoff(task.case_id, handoff, task.step.value)
        self._trace_result(task.case_id, result,
                           elapsed_ms=_elapsed_ms(task_started),
                           usage=(outcome.usage if outcome is not None else None))

        if outcome.status is not TaskStatus.FAILED:
            logger.info("a2a 流式完成 case=%s step=%s status=%s trace=%s",
                        task.case_id, task.step.value, outcome.status.value, trace)
            yield done_frame(result)
            return

        yield error_frame((outcome.payload or {}).get("code") or "AGENT_ERROR",
                          (outcome.payload or {}).get("message") or "执行失败", trace)

    # ------------------------------------------------------------------ 查询

    def get_case(self, case_id: str) -> CaseState | None:
        return self.store.get_case(case_id)

    def open_case(self, case_id: str, patient: dict | None = None) -> CaseState:
        case = self.store.create_case(case_id, patient)
        self._ensure_memories(case)
        return case

    # ------------------------------------------------------------------ L3 记忆

    def _ensure_memories(self, case: CaseState, task: CaseTask | None = None) -> None:
        """病例内首次执行时固定取一次 L3 记忆，失败静默降级为空。

        只快照一次：同一次就诊中途新增记忆不应让前后两轮看到的背景不一致。
        记忆是背景增强，医院侧不可用时绝不拖垮问答/起草主流程。
        """
        if case.memory_loaded or self.ctx.memory_provider is None:
            return
        extra_patient = (((task.payload or {}).get("patient") or {})
                         if task is not None else {})
        patient = {**(case.patient or {}), **extra_patient}
        # 契约只在 patient 内携带宠物事实；顶层 pet_id 不读取，避免两个来源打架
        pet_id = patient.get("pet_id")
        if pet_id in (None, ""):
            return      # 还没拿到档案 ID：不标记 loaded，等后续任务带上再取
        try:
            case.memories = list(self.ctx.memory_provider(pet_id) or [])
        except Exception:
            logger.warning("L3 记忆加载失败 pet=%s case=%s", pet_id, case.case_id,
                           exc_info=True)
            case.memories = []
        case.memory_loaded = True
        self.store.save_case(case)

    # ------------------------------------------------------------------ 前置分诊

    def _pretriage(self, case: CaseState, task: CaseTask) -> PreTriage:
        """编排层的安全闸门：在能力位执行前跑一次规则分诊。

        为什么放在 supervisor 而不是各能力位内部：能力位只会越加越多，逐个记得调守卫
        迟早会漏；闸门放在唯一入口上，新增能力位才**默认受保护**。
        判定逻辑直接复用 `triage_agent`，不另外维护一套毒物规则。
        """
        if self.ctx.guard is None or task.step not in _PRETRIAGE_STEPS:
            return PreTriage()

        payload = task.payload or {}
        query = str(payload.get("query") or payload.get("prompt") or "").strip()
        if not query:
            return PreTriage()      # 无自由文本（如预问诊开场）→ 没有可拦的内容

        probe = CaseTask(step=Step.TRIAGE, case_id=task.case_id,
                         idempotency_key=f"{task.idempotency_key}:pretriage",
                         payload={"query": query,
                                  "patient": payload.get("patient") or {}},
                         trace_id=task.trace_id)
        outcome = triage_agent(case, probe, self.ctx)

        if outcome.verdict is Verdict.BLOCK:
            return PreTriage(verdict=outcome.verdict, outcome=outcome,
                             reasons=outcome.reasons)
        if outcome.verdict is Verdict.WARN:
            return PreTriage(verdict=outcome.verdict, reasons=outcome.reasons)
        return PreTriage()

    def _short_circuit(self, case: CaseState, task: CaseTask,
                       outcome: AgentOutcome, elapsed_ms: int | None = None) -> TaskResult:
        """记录短路结论并返回 —— 原能力位不执行（不调 LLM、不推进会话）。"""
        payload = dict(outcome.payload or {})
        payload["pretriage"] = True
        if task.step is Step.INTAKE:
            # 预问诊调用方按 {message,questions,slots,state} 取字段：
            # 短路时给"零追问"的完整外形，而不是让它去猜哪些字段缺失
            payload.setdefault("message", payload.get("answer", ""))
            payload.setdefault("questions", [])
            payload.setdefault("slots", {})

        result = self._result(task, outcome.status, verdict=outcome.verdict,
                              payload=payload, confidence=outcome.confidence,
                              reasons=outcome.reasons, citations=outcome.citations)
        # 短路是**分诊**做的决定：交接来源是 triage，不是被拦下的目标能力位
        case.record(result)
        handoff = make_handoff(ROLE_TRIAGE, NODE_HUMAN, outcome.verdict.name,
                               case.seq, result.payload)
        case.record_handoff(handoff.to_dict())
        self.store.save_case(case)
        self.store.put_result(task.idempotency_key, task.request_fingerprint, result)
        self._trace_handoff(case.case_id, handoff, task.step.value)
        self._trace_result(task.case_id, result, elapsed_ms=elapsed_ms,
                           usage=outcome.usage)
        # 交接目标恒定是 human；被拦下的能力位另写 step，避免日志读成 triage->consult
        logger.info("a2a handoff triage->human verdict=%s step=%s case=%s trace=%s",
                    outcome.verdict.value, task.step.value, task.case_id, task.trace_id)
        return result

    # ------------------------------------------------------------------ 落库 / handoff

    def _admit_handoff(self, case: CaseState, task: CaseTask) -> None:
        """审方任务只可能由「医生签字后的处方」触发，进来先记 human→review。"""
        if task.step is not Step.REVIEW:
            return
        handoff = make_handoff(NODE_HUMAN, ROLE_REVIEW, "RX_SUBMITTED",
                               case.seq, task.payload)
        case.record_handoff(handoff.to_dict())
        self.store.save_case(case)
        self._trace_handoff(case.case_id, handoff, task.step.value)

    def _record_result(self, case: CaseState, task: CaseTask,
                       result: TaskResult) -> Handoff | None:
        """落 case / 幂等表，并把本步的 handoff 写进 case 与轨迹。"""
        if result.status is TaskStatus.FAILED:
            handoff = failed_handoff(task.step, case.seq, result.payload)
            case.record_handoff(handoff.to_dict())
            self.store.save_case(case)
            return handoff

        case.record(result)
        handoff = outcome_handoff(task.step, result.status, result.verdict,
                                  result.payload, case.seq)
        if handoff is not None:
            case.record_handoff(handoff.to_dict())
        self.store.save_case(case)
        self.store.put_result(task.idempotency_key, task.request_fingerprint, result)
        return handoff

    def _record_failed_handoff(self, case: CaseState, task: CaseTask) -> Handoff:
        handoff = failed_handoff(task.step, case.seq, task.payload)
        case.record_handoff(handoff.to_dict())
        self.store.save_case(case)
        return handoff

    def _trace_failure(self, case: CaseState, task: CaseTask,
                       error_code: str = "AGENT_ERROR") -> None:
        """异常失败：先记 failed 交接，再记 error（error 保持在轨迹末尾）。"""
        handoff = self._record_failed_handoff(case, task)
        self._trace_handoff(case.case_id, handoff, task.step.value)
        self._trace_error(task.case_id, task.step.value, error_code)

    # ------------------------------------------------------------------ 轨迹

    def _trace(self, case_id: str, *, step: str, agent: str, event: str,
               prompt_version_override: str | None = None, **fields) -> None:
        """轨迹写入失败绝不影响主流程：它是观测面，不是业务面。"""
        try:
            self.store.record_trace(
                case_id, step=step, agent=agent, event=event,
                prompt_version=(prompt_version_override
                                if prompt_version_override is not None
                                else self.ctx.prompt_version),
                model=self.ctx.model, **fields)
        except Exception:
            logger.exception("agent_trace 写入失败 case=%s event=%s", case_id, event)

    def _prompt_version_for(self, role: str) -> str:
        """角色提示词版本；``consult``/``human`` 没有角色文件时回落到引擎版本。"""
        if role == NODE_HUMAN:
            return ""
        return role_prompt_version(role) or self.ctx.prompt_version

    def _trace_handoff(self, case_id: str, handoff: Handoff | None,
                       step: str) -> None:
        if handoff is None:
            return
        # verdict 只对 PASS/WARN/BLOCK 三种业务结论有值；NEED_MORE 等交接没有结论
        verdict = (handoff.reason_code.lower()
                   if handoff.reason_code in ("PASS", "WARN", "BLOCK") else None)
        self._trace(case_id, step=step, agent=handoff.from_role,
                    event=EVENT_HANDOFF, ok=handoff.to_role != NODE_FAILED,
                    prompt_version_override=self._prompt_version_for(handoff.from_role),
                    from_step=handoff.from_role, to_step=handoff.to_role,
                    reason_code=handoff.reason_code, case_seq=handoff.case_seq,
                    payload_digest=handoff.payload_digest, verdict=verdict)

    def _trace_tool_call(self, case_id: str, step: str, data: dict) -> None:
        args = data.get("arguments")
        self._trace(case_id, step=step, agent=AGENT_TOOL, event=TRACE_TOOL_CALL,
                    prompt_version_override=self._prompt_version_for(step),
                    tool_name=str(data.get("name") or ""),
                    args=args if isinstance(args, dict) else None)

    def _trace_tool_result(self, case_id: str, step: str, tool_name: str,
                           data: dict, elapsed_ms: int) -> None:
        ok, error_code = self._tool_status(data.get("content"))
        self._trace(case_id, step=step, agent=AGENT_TOOL, event=TRACE_TOOL_RESULT,
                    prompt_version_override=self._prompt_version_for(step),
                    tool_name=tool_name or None, ok=ok, error_code=error_code,
                    elapsed_ms=elapsed_ms)

    def _trace_result(self, case_id: str, result: TaskResult,
                      elapsed_ms: int | None = None,
                      usage: dict | None = None) -> None:
        """结论轨迹：顺带落 token / 工具次数 / 耗时。

        这些是**观测字段**，不参与任何判定；``usage_estimated`` 显式区分模型回传
        的真实用量与本层估算，避免下游把估算值当计费数据。
        """
        failed = result.status is TaskStatus.FAILED
        usage = usage or {}
        self._trace(case_id, step=result.step.value, agent=AGENT_SUPERVISOR,
                    event=EVENT_VERDICT, ok=not failed,
                    prompt_version_override=self._prompt_version_for(result.step.value),
                    verdict=result.verdict.value,
                    error_code=(result.payload or {}).get("code") if failed else None,
                    elapsed_ms=elapsed_ms,
                    input_tokens=int(usage.get("input_tokens", 0) or 0),
                    output_tokens=int(usage.get("output_tokens", 0) or 0),
                    tool_count=int(usage.get("tool_count", 0) or 0),
                    usage_estimated=int(bool(usage.get("estimated"))))

    def _trace_error(self, case_id: str, step: str, error_code: str) -> None:
        self._trace(case_id, step=step, agent=AGENT_SUPERVISOR,
                    event=EVENT_ERROR, ok=False,
                    prompt_version_override=self._prompt_version_for(step),
                    error_code=error_code)

    @staticmethod
    def _tool_status(content) -> tuple[bool, str | None]:
        """从工具结果判定成功与否；结果正文本身绝不入轨迹。"""
        text = str(content or "")
        try:
            body = json.loads(text)
        except ValueError:
            body = None
        if isinstance(body, dict) and "ok" in body:
            return bool(body.get("ok")), body.get("error_code")
        if "无权调用工具" in text:
            return False, "ROLE_DENIED"
        if "参数不在白名单" in text:
            return False, "BAD_ARGUMENTS"
        if text.startswith("工具执行失败"):
            return False, "TOOL_FAILED"
        if text.startswith("参数解析失败"):
            return False, "BAD_ARGUMENTS"
        return True, None

    # ------------------------------------------------------------------ 内部

    def _result(self, task: CaseTask, status: TaskStatus,
                verdict: Verdict = Verdict.NONE, payload: dict | None = None,
                confidence: float = 1.0, reasons: list | None = None,
                citations: list | None = None) -> TaskResult:
        return TaskResult(
            case_id=task.case_id, task_id=task.task_id, step=task.step,
            status=status, verdict=verdict, payload=payload or {},
            confidence=confidence, reasons=reasons or [],
            citations=citations or [], trace_id=task.trace_id,
        )

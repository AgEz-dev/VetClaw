"""四个能力位的适配器：把既有业务模块包成统一的 A2A Agent 接口。

刻意**不改**被包装模块的语义（`fastpath` / `intake_flow` / `record_draft`
的行为一个字都没动，186 个既有测试因此不受影响）——适配器只做三件事：

1. 从 **case 状态**补全缺省输入（这是与旧契约的本质差异：起草不再要求调用方重传
   预问诊摘要，它已在 `case.steps['intake']` 里）；
2. 把产出归一成 `AgentOutcome`（执行态 + 结论 + 置信度 + 引用）；
3. 输入不足时返回 `NEED_MORE` 并**列出缺哪几个字段**，而不是抛异常或猜。

依赖全部走 `AgentContext` 注入：单测不需要网络、不需要 BGE、不需要 Chroma。
"""
from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from a2a.contract import CaseState, CaseTask, Step, TaskStatus, Verdict
from a2a.stream import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_THOUGHT,
    EVENT_TOKEN,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    INTERNAL_OUTCOME,
)
from memory import estimate_tokens, message_tokens


@dataclass
class AgentOutcome:
    """能力位的归一化产出。"""

    status: TaskStatus
    payload: dict = field(default_factory=dict)
    verdict: Verdict = Verdict.NONE
    confidence: float = 1.0
    reasons: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    # 用量（不参与业务判定）：input/output tokens 与工具次数，供 agent_trace 落库。
    usage: dict = field(default_factory=dict)


@dataclass
class AgentContext:
    """能力位的外部依赖（全部可注入，默认由路由层懒装配）。"""

    guard: Any = None                  # FastPathGuard
    llm: Callable | None = None        # 病历起草的模型通道（messages -> str）
    reviewer: Any = None               # PrescriptionReviewer
    intake_session_factory: Any = None  # 预问诊会话工厂
    chat_stream: Callable | None = None
    """问答流：``(prompt, history, patient_context, case_id="") -> Iterator[事件dict]``。

    直接注入既有 ``ReActAgent.run_stream`` —— 本层不重写问答逻辑，只做事件转发与
    结论归集。这样 React / LangGraph 两个引擎的事件序列不变，零漂移口径不受影响。
    ``case_id`` 只供工具写操作关联工单，不进入提示词。
    """

    # L3 长期记忆：``pet_id -> [事实dict]``。由 Supervisor 在 case 开始时固定取一次，
    # 不做成工具（让模型自己决定是否回忆必然漏），失败时静默降级为空。
    memory_provider: Callable | None = None

    # 轨迹元数据（不参与任何判定）：模型名与 system prompt 的短哈希版本。
    model: str = ""
    prompt_version: str = ""


def _need_more(message: str, missing: list[str], **extra) -> AgentOutcome:
    return AgentOutcome(status=TaskStatus.NEED_MORE,
                        payload={"message": message, "missing": missing, **extra},
                        confidence=0.0)


# --------------------------------------------------------------------- 分诊

_DURATION_RE = re.compile(
    r"\d+\s*(?:天|日|小时|分钟|周|个月|月|年)"
    r"|[一二三四五六七八九十两半]\s*(?:天|日|小时|分钟|周|个月|月|年)")


def _missing_triage_facts(patient: dict, payload: dict, query: str) -> list[str]:
    """分诊要判断「信息够不够」，只认结构化事实，不从自由文本猜体重/物种。"""
    missing = []
    if not patient.get("species"):
        missing.append("species")
    if patient.get("weight_kg") in (None, ""):
        missing.append("weight_kg")
    if not (payload.get("duration") or payload.get("duration_text")
            or _DURATION_RE.search(query)):
        missing.append("duration")
    return missing


def triage_agent(case: CaseState, task: CaseTask, ctx: AgentContext) -> AgentOutcome:
    """急症 / 未知处方药前置拦截 + 信息完备性判定（包 `fastpath.FastPathGuard`）。"""
    query = str((task.payload or {}).get("query") or "").strip()
    if not query:
        return _need_more("分诊需要主诉文本", ["query"])
    if ctx.guard is None:
        return AgentOutcome(status=TaskStatus.FAILED,
                            payload={"message": "分诊守卫未装配（FastPathGuard 缺失）"})

    patient = {**(case.patient or {}), **((task.payload or {}).get("patient") or {})}
    hit = ctx.guard.check(query, patient or None)
    action = hit.get("action")

    if action == "emergency":
        return AgentOutcome(
            status=TaskStatus.OK,
            payload={**hit, "answer": ctx.guard.emergency_message(
                hit.get("toxin"), hit.get("species"), hit.get("chunk_id"), hit.get("slots"))},
            verdict=Verdict.BLOCK,
            confidence=1.0,
            reasons=[{"level": "block", "code": "P0_EMERGENCY",
                      "message": f"命中 P0 急症毒物：{hit.get('toxin')}"}],
            citations=[c for c in [hit.get("chunk_id")] if c],
        )
    if action == "refuse":
        return AgentOutcome(
            status=TaskStatus.OK,
            payload={**hit, "answer": ctx.guard.refuse_message(hit.get("drug_hint"))},
            verdict=Verdict.WARN,
            confidence=1.0,
            reasons=[{"level": "warn", "code": "UNLISTED_DRUG_DOSAGE",
                      "message": f"未收录处方药剂量咨询：{hit.get('drug_hint')}"}],
        )

    missing = _missing_triage_facts(patient, task.payload or {}, query)
    if missing:
        # 不是错误：分诊已经排除急症，但下游起草/审方需要的事实还没齐 → 交给预问诊
        return AgentOutcome(
            status=TaskStatus.NEED_MORE,
            payload={**hit, "missing": missing,
                     "message": "为完成分诊，还需补充：" + "、".join(missing)},
            verdict=Verdict.PASS,
            confidence=1.0,
        )
    return AgentOutcome(status=TaskStatus.OK, payload=hit, verdict=Verdict.PASS,
                        confidence=1.0)


# --------------------------------------------------------------------- 预问诊

def intake_agent(case: CaseState, task: CaseTask, ctx: AgentContext) -> AgentOutcome:
    """多轮预问诊（包 `intake_flow.IntakeSession`）。一轮任务 = 一轮对话。"""
    payload = task.payload or {}
    prompt = str(payload.get("prompt") or "").strip()

    if ctx.intake_session_factory is None:
        return AgentOutcome(status=TaskStatus.FAILED,
                            payload={"message": "预问诊依赖未装配"})

    archive = payload.get("archive") or {}
    patient = {**(case.patient or {}), **(payload.get("patient") or {})}

    # 开场：显式标记 → 一律用**全新会话**取开场白 + 首批追问。
    # 为什么不继承 case 里已有的状态：开场白的语义是「这个就诊还没开始聊」，
    # 若继承已进行到第 5 轮的状态，会得到一句与当前进度矛盾的问候。
    # 重复调用是安全的：同一 case 用同一幂等键 → 回放同一个开场，不会覆盖已有进度。
    if payload.get("opening"):
        session = ctx.intake_session_factory(None, archive, patient)
        result = session.opening()
        return AgentOutcome(status=TaskStatus.OK, payload=result,
                            verdict=Verdict.NONE, confidence=1.0)

    # 续传状态优先级：本次显式传入 > case 内上一轮预问诊的产出。
    # **后者是本层的价值**：旧契约要求调用方每轮把 state 原样带回（`to_state` 往返），
    # 现在 state 已在 case 里，调用方只传本轮用户说了什么。
    previous = case.result_of(Step.INTAKE)
    previous_state = ((previous.payload or {}).get("state") if previous else None)
    state = payload.get("intake_state") or previous_state
    if not prompt:
        return _need_more("预问诊轮次需要本轮宠主发言", ["prompt"])

    session = ctx.intake_session_factory(state, archive, patient)
    result = session.handle(prompt)

    return AgentOutcome(
        status=TaskStatus.OK,
        payload=result,
        verdict=Verdict.NONE,
        confidence=1.0,
        citations=[c.get("chunk_id") for c in (result.get("citations") or [])
                   if isinstance(c, dict) and c.get("chunk_id")],
    )


# --------------------------------------------------------------------- 病历起草

def draft_agent(case: CaseState, task: CaseTask, ctx: AgentContext) -> AgentOutcome:
    """SOAP 叙述性草稿（包 `record_draft`）。**从 case 状态取预问诊成果**。"""
    if ctx.llm is None:
        return AgentOutcome(status=TaskStatus.FAILED,
                            payload={"message": "起草模型通道未配置"})

    from record_draft import build_draft_messages, disclaimer_for, parse_draft_response

    payload = task.payload or {}
    # 关键：默认取 case 里预问诊那一步的产出，调用方不必重传
    intake = case.result_of(Step.INTAKE) or None
    intake_payload = (intake.payload or {}) if intake else {}
    summary = payload.get("summary") or intake_payload.get("summary") or {}
    dialogue = payload.get("dialogue") or intake_payload.get("dialogue") or []
    archive = payload.get("archive") or intake_payload.get("archive") or {}
    patient = {**(case.patient or {}), **(payload.get("patient") or {})}

    if not summary and not dialogue:
        return _need_more("起草需要预问诊摘要或对话原文（本次 case 内也没有）",
                          ["summary", "dialogue"])

    try:
        raw = ctx.llm(build_draft_messages(summary, dialogue, archive, patient,
                                           memories=case.memories))
    except Exception as exc:  # 模型通道故障 → 执行失败，可重试
        return AgentOutcome(status=TaskStatus.FAILED,
                            payload={"message": f"起草模型调用失败: {type(exc).__name__}"})

    draft = parse_draft_response(raw)
    dialogue_text = "\n".join(str(d.get("text", "")) for d in dialogue
                              if isinstance(d, dict))
    draft["disclaimer"] = disclaimer_for(dialogue_text)
    if not draft.get("chief_complaint"):
        # 通道在跑但产出不可解析 —— 绝不给医生一张空草稿
        return AgentOutcome(status=TaskStatus.FAILED,
                            payload={"message": "本次起草未产出可解析内容"})
    return AgentOutcome(status=TaskStatus.OK, payload={"draft": draft},
                        verdict=Verdict.NONE, confidence=0.9)


# --------------------------------------------------------------------- 处方审核

def review_agent(case: CaseState, task: CaseTask, ctx: AgentContext) -> AgentOutcome:
    """处方安全审核（包 `prescription_review.PrescriptionReviewer`）。

    宠物事实以 **case 状态为准**、payload 覆盖（同一 case 里档案只应录一次）。
    """
    payload = {**(task.payload or {})}
    if not payload.get("patient"):
        payload["patient"] = {**(case.patient or {})}
    else:
        payload["patient"] = {**(case.patient or {}), **payload["patient"]}

    items = ((payload.get("prescription") or {}).get("items")) or []
    if not items:
        return _need_more("处方审核需要处方明细", ["prescription.items"])
    if ctx.reviewer is None:
        return AgentOutcome(status=TaskStatus.FAILED,
                            payload={"message": "审核器未装配"})

    out = ctx.reviewer.review(payload)
    return AgentOutcome(
        status=TaskStatus.OK,
        payload=out,
        verdict=Verdict(out.get("verdict", "none")),
        confidence=float(out.get("confidence", 1.0)),
        reasons=list(out.get("reasons") or []),
        citations=list(out.get("citations") or []),
    )


AGENTS: dict[Step, Callable[[CaseState, CaseTask, AgentContext], AgentOutcome]] = {
    Step.TRIAGE: triage_agent,
    Step.INTAKE: intake_agent,
    Step.DRAFT: draft_agent,
    Step.REVIEW: review_agent,
}


# --------------------------------------------------------------------- 流式能力位
#
# 流式能力位是**生成器**：边跑边 `yield` 事件帧，最后用 `INTERNAL_OUTCOME` 把结论
# 交回 supervisor（supervisor 不会把它下发给客户端）。这样「同步」与「流式」两种
# 能力位共用同一套结果落库/幂等逻辑，不需要两套编排。

def consult_stream(case: CaseState, task: CaseTask, ctx: AgentContext):
    """问答（对宠主说话，逐字流式）。

    **不重写问答逻辑**：直接把既有 `ReActAgent.run_stream` 的事件转发出去，
    只做两件事——① 转发 `token/thought/tool_call/tool_result`；② 归集最终答案与引用。
    因此 React / LangGraph 两个引擎的事件序列**一字未改**，零漂移口径不受影响。
    """
    payload = task.payload or {}
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        yield _outcome_event(_need_more("问答需要 prompt", ["prompt"]))
        return
    if ctx.chat_stream is None:
        yield _outcome_event(AgentOutcome(
            status=TaskStatus.FAILED,
            payload={"code": "CHAT_UNAVAILABLE",
                     "message": "问答能力未装配（ReActAgent 不可用）"}))
        return

    history = payload.get("history") or []
    patient = {**(case.patient or {}), **(payload.get("patient") or {})}
    if case.memories:
        # 只作为背景注入 system 上下文；不进入 Fast-Path 判定，不改变事件序列
        patient["ai_memories"] = list(case.memories)
    answer_parts: list[str] = []
    citations: list[str] = []
    terminal = False
    failure: dict | None = None
    usage: dict = {}
    tool_count = 0

    # 引擎通过可选回调交回 token 用量（不新增 SSE 事件）；测试桩没有该参数时自动跳过。
    stream_kwargs = {}
    try:
        if "on_usage" in inspect.signature(ctx.chat_stream).parameters:
            stream_kwargs["on_usage"] = lambda u: usage.update(u or {})
    except (TypeError, ValueError):
        stream_kwargs = {}

    try:
        for ev in ctx.chat_stream(prompt, history, patient or None,
                                  case_id=case.case_id, **stream_kwargs):
            name = ev.get("event")
            data = ev.get("data") or {}
            if name == EVENT_TOKEN:
                answer_parts.append(str(data.get("text") or ""))
                yield _frame(EVENT_TOKEN, data)
            elif name in (EVENT_THOUGHT, EVENT_TOOL_CALL, EVENT_TOOL_RESULT):
                if name == EVENT_TOOL_CALL:
                    tool_count += 1
                yield _frame(name, data)
            elif name == EVENT_DONE:
                terminal = True
                citations = list(data.get("citations") or [])
                if not answer_parts and data.get("answer"):
                    answer_parts.append(str(data["answer"]))
            elif name == EVENT_ERROR:
                terminal = True
                failure = {"code": data.get("code") or "INTERNAL_ERROR",
                           "message": data.get("message") or "问答执行失败"}
    except Exception as exc:
        yield _outcome_event(AgentOutcome(
            status=TaskStatus.FAILED,
            payload={"code": "AGENT_ERROR",
                     "message": f"问答执行异常: {type(exc).__name__}"},
            usage=usage))
        return

    if failure is not None:
        yield _outcome_event(AgentOutcome(status=TaskStatus.FAILED, payload=failure,
                                          usage=usage))
        return
    if not terminal:
        # 上游没给终态：绝不能假装成功（客户端会一直等下去）
        yield _outcome_event(AgentOutcome(
            status=TaskStatus.FAILED,
            payload={"code": "NO_TERMINAL_EVENT",
                     "message": "问答未返回终态事件"},
            usage=usage))
        return

    answer = "".join(answer_parts)
    if not usage:
        # 测试桩/兼容端点没有用量：按同一启发式估算，并在轨迹里标记 estimated。
        usage = {
            "input_tokens": (sum(message_tokens(m) for m in (history or []))
                             + estimate_tokens(prompt)),
            "output_tokens": estimate_tokens(answer),
            "estimated": True,
        }
    usage.setdefault("input_tokens", 0)
    usage.setdefault("output_tokens", 0)
    usage["tool_count"] = tool_count

    yield _outcome_event(AgentOutcome(
        status=TaskStatus.OK,
        payload={"answer": answer},
        verdict=Verdict.NONE,
        confidence=1.0,
        citations=citations,
        usage=usage,
    ))


def _frame(name: str, data: dict) -> dict:
    return {"event": name, "data": data}


def _outcome_event(outcome: AgentOutcome) -> dict:
    """把结论包成内部事件；supervisor 识别后**不下发**给客户端。"""
    return {"event": INTERNAL_OUTCOME, "data": outcome}


STREAM_AGENTS: dict[Step, Callable[[CaseState, CaseTask, AgentContext], Any]] = {
    Step.CONSULT: consult_stream,
}

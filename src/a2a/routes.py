"""A2A 路由：建案 → 提交任务 → 查询 case 状态 →（流式）提交任务。

编排权在服务端的体现：调用方**只提交结构化数据与幂等键**，不再决定「下一步做什么」，
也不再需要把上一步的产出重新传回来（case 里已经有了）。

两种提交形态：
- ``POST /a2a/task``：同步 JSON。适合「要结论」的能力位（分诊 / 预问诊 / 起草 / 审核）。
- ``POST /a2a/task/stream``：SSE 流式。适合「边说边给」的能力位（问答）。
  帧格式与既有 ``/api/chat/stream`` **同构**，因此医院侧现有 SSE 中继无需重写即可切换。

错误语义（与既有接口保持一致的 ``{code,message}`` 外形）：

- ``400``：请求字段非法（Pydantic）/ 契约校验失败（Step 未知等由 supervisor 转 FAILED）
- ``404``：case 不存在
- ``409``：**幂等键冲突** —— 同一个键被用于不同内容的请求
- ``500``：内部错误（不泄漏 traceback）

> 流式端点的 ``409`` 在**握手前**同步判定并返回真 HTTP 409；一旦开始推帧，
> 后续错误只能以 ``error`` 帧表达（这是 SSE 的固有限制，不是设计选择）。
"""
from __future__ import annotations

import logging
import pydantic

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from a2a.agents import AgentContext
from a2a.contract import CaseTask, Step
from a2a.stream import to_sse
from a2a.supervisor import IdempotencyConflict, Supervisor
from observability import get_request_id
from schemas.a2a import A2aCaseOpen, A2aTaskRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/a2a", tags=["a2a"])

_supervisor: Supervisor | None = None


def build_agent_context() -> AgentContext:
    """装配能力位依赖。**懒加载 + 逐项容错**：缺哪一项，只有依赖它的能力位会失败，
    其余能力位仍可用（避免"因为没配 Key，连处方审核都挂了"）。"""
    ctx = AgentContext()

    try:
        from services.agent_service import get_guard
        ctx.guard = get_guard()
    except Exception as exc:
        logger.warning("a2a: FastPathGuard 未装配（分诊将不可用）: %s", exc)

    try:
        from services.intake_service import get_llm
        ctx.llm = get_llm()
    except Exception as exc:
        logger.warning("a2a: 模型通道未装配（起草将不可用）: %s", exc)

    # 审核器是纯规则，永远可用 —— 用药安全闸门不依赖任何外部服务
    from prescription_review import PrescriptionReviewer
    ctx.reviewer = PrescriptionReviewer()

    def _intake_factory(state, archive, patient):
        from intake_flow import IntakeSession
        from services.intake_service import get_guard, get_llm
        return IntakeSession.from_state(state, llm=get_llm(), archive=archive,
                                        guard=get_guard(), patient_context=patient)

    ctx.intake_session_factory = _intake_factory

    # 问答流：直接注入既有 ReActAgent.run_stream（双引擎共用同一事件序列，零漂移）
    try:
        from a2a.trace import prompt_version
        from services.agent_service import get_agent
        agent = get_agent()
        ctx.chat_stream = agent.run_stream
        # 轨迹元数据：模型名 + system prompt 短哈希（不落提示词正文）
        ctx.model = getattr(agent, "model", "")
        ctx.prompt_version = prompt_version(getattr(agent, "system_prompt", ""))
    except Exception as exc:
        logger.warning("a2a: 问答能力未装配（consult 将不可用）: %s", exc)

    # L3 长期记忆：case 开局取一次（医院侧只读）。未配密钥时关闭，不影响其余能力。
    try:
        from hospital_client import hospital_provider
        hospital = hospital_provider.get()
        if hospital.secret:
            ctx.memory_provider = hospital.list_memories
        else:
            logger.info("a2a: 未配置医院密钥，L3 记忆注入关闭")
    except Exception as exc:
        logger.warning("a2a: L3 记忆通道未装配（问答/起草将无长期记忆）: %s", exc)

    return ctx


def get_supervisor() -> Supervisor:
    global _supervisor
    if _supervisor is None:
        from a2a.store import CaseStore
        _supervisor = Supervisor(CaseStore(), build_agent_context())
    return _supervisor


def set_supervisor(value: Supervisor | None) -> None:
    """测试注入。"""
    global _supervisor
    _supervisor = value


@router.post("/case")
def open_case(req: A2aCaseOpen):
    """建案（幂等：已存在则返回既有状态，不覆盖已积累的上下文）。"""
    state = get_supervisor().open_case(req.case_id, req.patient)
    return {"code": 0, "message": "ok", "data": state.to_dict()}


@router.get("/case/{case_id}")
def get_case(case_id: str):
    state = get_supervisor().get_case(case_id)
    if state is None:
        return JSONResponse(status_code=404,
                            content={"code": "CASE_NOT_FOUND",
                                     "message": f"case 不存在: {case_id}"})
    return {"code": 0, "message": "ok", "data": state.to_dict()}


@router.post("/task")
def submit_task(req: A2aTaskRequest):
    try:
        task = CaseTask(
            step=Step(req.step), case_id=req.case_id,
            idempotency_key=req.idempotency_key, payload=req.payload or {},
            task_id=req.task_id, caller=req.caller,
            deadline_at=req.deadline_at,
            confidence_threshold=req.confidence_threshold,
            trace_id=get_request_id() or "",
        )
    except ValueError as exc:
        return JSONResponse(status_code=400,
                            content={"code": "INVALID_TASK", "message": str(exc)})

    try:
        result = get_supervisor().submit(task)
    except IdempotencyConflict as exc:
        return JSONResponse(status_code=409,
                            content={"code": "IDEMPOTENCY_CONFLICT",
                                     "message": str(exc)})
    except Exception:
        logger.exception("a2a submit 失败 case=%s step=%s", req.case_id, req.step)
        return JSONResponse(status_code=500,
                            content={"code": "INTERNAL_ERROR",
                                     "message": "服务内部错误，请稍后重试"})
    return {"code": 0, "message": "ok", "data": result.to_dict()}


@router.post("/task/stream")
def submit_task_stream(req: A2aTaskRequest):
    """流式提交（SSE）。仅支持**已登记流式能力位**（当前为 `consult` / 问答）。

    冲突检查在握手前同步完成 —— 能返回真 HTTP 409 就绝不退化成流内 error 帧。
    """
    try:
        task = CaseTask(
            step=Step(req.step), case_id=req.case_id,
            idempotency_key=req.idempotency_key, payload=req.payload or {},
            task_id=req.task_id, caller=req.caller,
            deadline_at=req.deadline_at,
            confidence_threshold=req.confidence_threshold,
            trace_id=get_request_id() or "",
        )
    except ValueError as exc:
        return JSONResponse(status_code=400,
                            content={"code": "INVALID_TASK", "message": str(exc)})

    supervisor = get_supervisor()
    try:
        supervisor.assert_no_conflict(task)
    except IdempotencyConflict as exc:
        return JSONResponse(status_code=409,
                            content={"code": "IDEMPOTENCY_CONFLICT",
                                     "message": str(exc)})

    def generate():
        try:
            for ev in supervisor.submit_stream(task):
                yield to_sse(ev["event"], ev["data"])
        except Exception:
            # 生成器内部异常不得冒到 HTTP 层（握手后状态码已发）
            logger.exception("a2a 流式生成器异常 case=%s step=%s", req.case_id, req.step)
            yield to_sse("error", {"code": "INTERNAL_ERROR",
                                   "message": "服务内部错误，请稍后重试",
                                   "trace_id": task.trace_id})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class A2aFeedbackRequest(pydantic.BaseModel):
    trace_id: str = ""
    role: str = "DOCTOR"
    query: str = ""
    retrieved_chunk_ids: list[str] = []
    ai_output: str = ""
    human_revised_output: str = ""
    reason: str = ""


@router.post("/feedback")
def submit_feedback(req: A2aFeedbackRequest):
    """接收人工修正或点踩反馈 (Badcase)，脱敏落盘驱动数据飞轮。"""
    from feedback_store import record_feedback
    tid = req.trace_id or get_request_id() or "unknown"
    record = record_feedback(
        trace_id=tid,
        role=req.role,
        query=req.query,
        retrieved_chunk_ids=req.retrieved_chunk_ids,
        ai_output=req.ai_output,
        human_revised_output=req.human_revised_output,
        reason=req.reason
    )
    return {"code": 0, "message": "feedback recorded", "data": record}

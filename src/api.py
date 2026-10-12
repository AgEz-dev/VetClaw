"""HTTP 路由：SSE 流式对话接口。"""
import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from observability import get_request_id, set_request_id
from schemas.chat import ChatRequest
from services.agent_service import get_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


def to_sse(event, data):
    # SSE 规范：data 内的换行必须拆成续行 data:，否则会破坏空行分帧
    payload = json.dumps(data, ensure_ascii=False).replace("\n", "\ndata: ")
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    agent = get_agent()
    # 在请求线程捕获 request_id（context 由 RequestIdMiddleware 注入），供生成器内日志
    # 与兜底 error 事件使用——生成器执行上下文与请求上下文不保证一致，故显式透传。
    rid = get_request_id()

    def generate():
        set_request_id(rid)
        # 架构边界：StreamingResponse 握手后 HTTP 状态码已发 200，全局异常处理器
        # 再也捕获不到生成器内部异常。因此这里必须自己兜住，把异常转成 error 事件再关流。
        try:
            for ev in agent.run_stream(req.prompt, req.history, req.patient_context):
                yield to_sse(ev["event"], ev["data"])
        except Exception:
            logger.exception("chat stream generator error")
            yield to_sse("error", {"code": "INTERNAL_ERROR",
                                    "message": "服务内部错误，请稍后重试",
                                    "trace_id": rid})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/intake/stream")
def intake_stream(req: ChatRequest):
    """预问诊：**一次请求 = 一轮对话**。

    本服务**无状态** —— 会话状态由调用方（业务系统）保存并在下次请求经
    ``intake_state`` 原样带回，因此不需要 session id、不需要 TTL、不强依赖内存缓存。

    终态事件为 ``done``，其 ``data.intake`` 携带本轮追问或最终结构化摘要
    （契约 v1.2 §3.4）；``data.intake.state`` 是**下一次请求要原样带回**的续传状态。
    """
    from intake_flow import IntakeSession

    from services.intake_service import build_archive, get_guard, get_llm

    rid = get_request_id()

    def generate():
        set_request_id(rid)
        try:
            session = IntakeSession.from_state(
                req.intake_state,
                llm=get_llm(),
                archive=build_archive(req.intake_archive),
                guard=get_guard(),
                patient_context=(req.patient_context.model_dump()
                                 if req.patient_context else None),
            )
            result = session.opening() if not req.intake_state else session.handle(req.prompt)

            if result.get("message"):                       # 开场白
                yield to_sse("token", {"text": result["message"]})
            for q in result.get("questions", []):           # 本轮追问
                yield to_sse("token", {"text": "\n\n" + q["ask"]})
            yield to_sse("done", {"answer": "", "citations": [], "intake": result})
        except Exception:
            logger.exception("intake stream generator error")
            yield to_sse("error", {"code": "INTERNAL_ERROR",
                                   "message": "服务内部错误，请稍后重试",
                                   "trace_id": rid})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

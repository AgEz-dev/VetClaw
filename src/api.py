"""HTTP 路由：SSE 流式对话接口。"""
import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

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

    def generate():
        # 架构边界：StreamingResponse 握手后 HTTP 状态码已发 200，全局异常处理器
        # 再也捕获不到生成器内部异常。因此这里必须自己兜住，把异常转成 error 事件再关流。
        try:
            for ev in agent.run_stream(req.prompt, req.history):
                yield to_sse(ev["event"], ev["data"])
        except Exception:
            logger.exception("chat stream generator error")
            yield to_sse("error", {"code": "INTERNAL_ERROR",
                                    "message": "服务内部错误，请稍后重试"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

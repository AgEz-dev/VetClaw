"""HTTP 路由：SSE 流式对话接口。"""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas.chat import ChatRequest
from services.agent_service import get_agent

router = APIRouter(prefix="/api", tags=["chat"])


def to_sse(event, data):
    # SSE 规范：data 内的换行必须拆成续行 data:，否则会破坏空行分帧
    payload = json.dumps(data, ensure_ascii=False).replace("\n", "\ndata: ")
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    agent = get_agent()

    def generate():
        for ev in agent.run_stream(req.prompt, req.history):
            yield to_sse(ev["event"], ev["data"])

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

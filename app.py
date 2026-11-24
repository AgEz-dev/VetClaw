"""VetClaw FastAPI 入口。启动：uvicorn app:app --reload"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from api import router  # noqa: E402
from a2a.routes import router as a2a_router  # noqa: E402
from core.config import settings  # noqa: E402
from observability import RequestIdMiddleware, install_logging  # noqa: E402
from mcp_server import mcp_server  # noqa: E402

# 结构化日志 + request_id 注入（维度 6），取代裸 logging.basicConfig
install_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动预热：把 BGE 模型加载、jieba 字典构建、ChromaDB 连接建立都消化在启动阶段
    logging.info("Warming up: loading BGE model + jieba dict + ChromaDB...")
    try:
        from rag_pipeline import RAGPipeline, BGEEmbeddingFunction
        from services.agent_service import get_agent
        RAGPipeline(embedding_function=BGEEmbeddingFunction())
        get_agent()  # 触发 build_agent()，加载模型 + 构建 BM25
        logging.info("Warm-up complete.")
    except Exception as e:
        logging.warning("Warm-up skipped: %s", e)
    yield


app = FastAPI(title="VetClaw", version="2.0.0", lifespan=lifespan)

# CORS：allow_credentials=True 与 allow_origins=["*"] 冲突，白名单含 "*" 时自动关闭 credentials。
allow_credentials = "*" not in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# request_id 中间件（纯 ASGI）：最后注册 → 位于最外层，SSE 响应也会回写 X-Trace-Id
app.add_middleware(RequestIdMiddleware)

app.include_router(router)
# A2A 编排层（case 级任务契约）：编排放服务端，调用方只提交结构化数据 + 幂等键
app.include_router(a2a_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # 全局兜底：未捕获异常打日志，对外只返回干净 JSON，绝不泄漏 traceback/str(exc)。
    # 注意：此 handler 只能捕获"握手前"的异常（路由层、依赖注入层）；
    # StreamingResponse 生成器内部异常由 api.py 的 generate() 自行兜住。
    logging.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": "服务内部错误，请稍后重试"},
    )


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "VetClaw Clinical AI Agent",
        "version": "2.2.0",
        "description": "专科宠物医院数字化系统 · 临床AI决策与处方安全审查智能体",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "chat_stream": "/chat/stream",
            "a2a_task_stream": "/a2a/task/stream",
            "prescription_review": "/prescription/review",
            "mcp_jsonrpc": "/mcp"
        }
    }


@app.post("/mcp")
async def handle_mcp_jsonrpc(request: Request):
    """MCP (Model Context Protocol) 开放协议端点 (JSON-RPC 2.0)."""
    try:
        body = await request.json()
        resp = mcp_server.handle_message(body)
        return JSONResponse(content=resp)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/prescription/review")
@app.post("/api/prescription/review")
def direct_prescription_review(req: dict):
    from prescription_review import review_prescription
    return review_prescription(req)

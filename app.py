"""Sentinel-Agent FastAPI 入口。启动：uvicorn app:app --reload"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from api import router  # noqa: E402
from core.config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Sentinel-Agent", version="0.1.0")

# CORS：allow_credentials=True 与 allow_origins=["*"] 冲突，白名单含 "*" 时自动关闭 credentials。
allow_credentials = "*" not in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


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


@app.get("/health")
def health():
    return {"status": "ok"}

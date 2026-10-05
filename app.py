"""Sentinel-Agent FastAPI 入口。启动：uvicorn app:app --reload"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI  # noqa: E402
from api import router  # noqa: E402

app = FastAPI(title="Sentinel-Agent", version="0.1.0")
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}

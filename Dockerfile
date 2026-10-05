FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖（chromadb 运行需要的最小集，无编译工具）
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 先拷依赖声明，利用 Docker 层缓存
COPY pyproject.toml ./
# 再拷源码（setuptools 需要包源码才能 install）
COPY src/ ./src/
COPY schemas/ ./schemas/
COPY app.py ./

# 非 root 用户，UID 固定 1000，与 compose 的 user: "1000:1000" 对齐
RUN useradd -u 1000 -m appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

RUN pip install --no-cache-dir .

USER appuser

EXPOSE 8000

# 健康检查：用 curl 探 /health，纯静态接口不触发任何懒加载
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

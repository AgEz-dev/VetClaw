# 部署与回滚

> 维度 7「容灾与韧性」的配套文档：把部署方式、健康检查与回滚步骤写清楚，避免依赖记忆操作。

## 1. 部署形态

单容器：FastAPI 应用 + 本地持久化向量库（ChromaDB，落在容器内挂载的 `./data`）。

仓库提供：`Dockerfile`、`docker-compose.yml`、`.env.example`。

## 2. 构建与启动

```bash
cp .env.example .env      # 填入 OPENAI_API_KEY 等
docker compose up -d --build
```

默认端口 `8000`。查看状态：

```bash
docker compose ps
docker compose logs -f vetclaw
```

## 3. 环境变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `OPENAI_API_KEY` | 模型服务密钥 | 必填 |
| `OPENAI_BASE_URL` | OpenAI 兼容地址（直连可留空） | 空 |
| `VETCLAW_MODEL` | 模型标识 | `gpt-4o-mini` |
| `VETCLAW_TIMEOUT` | SDK 层单次请求读超时（秒） | `15` |
| `VETCLAW_TOTAL_TIMEOUT` | Agent 生命周期总时长守卫（秒） | `30` |
| `VETCLAW_MAX_STEPS` | ReAct 最大步数（工具循环与补正循环共用预算） | `5` |
| `VETCLAW_ENGINE` | 引擎：`react` \| `langgraph` | `react` |
| `CORS_ORIGINS` | 跨域白名单，逗号分隔 | 本地开发端口 |

## 4. 健康检查与观测

- **健康检查**：`GET /health` → `{"status":"ok"}`
- **请求 ID**：响应头回写 `X-Request-Id`（透传或生成），日志为单行 JSON 且每条含 `request_id`，
  便于按 request 粒度检索与跨服务对齐。
- **事件流**：`POST /api/chat/stream` 为标准 SSE，事件类型 `thought` / `tool_call` / `tool_result`
  / `token` / `done` / `error`；`error` 事件携带 `trace_id`。

## 5. 韧性

- 模型调用由 `src/resilience.py` 统一包裹：**指数退避 + 抖动重试**（仅瞬时错误），
  以及**进程级熔断**（连续失败达阈值后打开，期间快速失败、不发出真实请求）。
- 熔断/重试对 `react` 与 `langgraph` 两个引擎共用同一实现，行为一致。

## 6. 回滚方案

| 场景 | 操作 |
|---|---|
| **镜像回滚** | 构建时按提交打标签：`docker build -t vetclaw:$(git rev-parse --short HEAD) .`；回滚即用上一标签启动 |
| **代码回滚** | `git checkout <上一个 tag>` → `docker compose up -d --build` |
| **配置回滚** | `.env` 纳入版本管理外管理（备份/凭据管理），回滚时同步还原；密钥轮换后旧值即失效 |
| **数据回滚** | 向量库在 `./data`：**变更前先整体备份该目录**（`tar` 打包即可），回滚 = 还原目录后重启容器 |

> ⚠️ 注意：`VETCLAW_ENGINE` 切换（react ↔ langgraph）不涉及数据变更，可随时回退；
> 但若知识库已 `ingest --rebuild`，回滚代码时需确认 `rules/` 与 `knowledge/` 的版本一致。

## 7. 已知限制

- 单实例部署，无水平扩展与灰度能力。
- 向量库为进程内/本地持久化，非共享存储 → 多副本会有一致性问题，当前不适用。
- 无 metrics 端点（当前仅结构化日志 + request_id）。

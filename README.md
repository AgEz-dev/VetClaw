# VetClaw

宠物健康分诊 Agent：用户输入宠物误食 / 症状 / 用药问题，系统通过 **Fast-Path 规则守卫 + RAG 混合检索 + 原生 ReAct 调度**，输出带引用来源的专业回答，P0 急症 3ms 级拦截。

> 零 LangChain：从零手写 ReAct 状态机、tool_calls 流式解析与 cite_state 防幻觉状态机，吃透 Agent 底层原理。

## 核心能力

- **Fast-Path 前置守卫**：P0 致命毒物（百合 / 木糖醇 / 对乙酰氨基酚 / 葱属 / 布洛芬 / 葡萄）秒级拦截，基于 ChromaDB ID 点查直取急救 SOP + 毒物详情；未收录处方药（头孢 / 阿莫西林等）剂量咨询 100% 触发分诊引导
- **RAG 混合检索**：结构感知 Markdown 切片 + BGE-small-zh-v1.5 本地向量 + BM25(jieba 医疗词典) 稀疏召回，RRF(k=60) 融合重排
- **原生 ReAct 调度**：`_run_events()` 统一事件生成器同时驱动同步与流式；tool_calls 流式分片重组；双层超时（SDK read timeout + Agent 总时长守卫）
- **防幻觉三层防护**：Prompt 契约 → `cite_state` 局部状态机（hit 不可被 miss 覆盖）→ 补正反馈；回答强制标注【来源：chunk_id】
- **滑动窗口记忆**：按「assistant + tool_calls 原子块」裁剪，system prompt 预算保护

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| Web 框架 | FastAPI + Uvicorn | SSE 流式推送 |
| Agent 调度 | 纯手写 ReAct（零 LangChain） | `_run_events()` 统一事件生成器 |
| 向量库 | ChromaDB PersistentClient | 本地 SQLite 持久化，ID 点查 <3ms |
| Embedding | BGE-small-zh-v1.5 | 本地模型，query/doc 前缀分离 |
| 稀疏检索 | rank_bm25 + jieba | 中文分词，医疗实体词典 |
| 融合 | RRF（k=60） | Dense + Sparse 双路融合 |
| 规则守卫 | FastPathGuard | P0 急症拦截 + 未知处方药拒答 |
| 记忆 | SlidingWindowMemory | 原子块裁剪，system prompt 保护 |
| 部署 | Docker Compose | 数据卷挂载 `./data` |

## 架构

三层架构（接入层 / 调度层 / 能力层）与关键技术决策见 [docs/architecture.md](docs/architecture.md)。

## 项目结构

```text
VetClaw/
├── app.py                    # FastAPI 入口，lifespan 预热
├── src/
│   ├── core_agent.py         # ReAct 调度核心（_run_events 生成器）
│   ├── tools.py              # inspect 反射 ToolRegistry
│   ├── memory.py             # 滑动窗口原子块裁剪
│   ├── rag_pipeline.py       # 切片 + BGE + BM25 + RRF 混合检索
│   ├── fastpath.py           # Fast-Path 规则守卫
│   ├── knowledge_tool.py     # RAG 工具适配层（依赖注入）
│   ├── api.py                # POST /api/chat/stream SSE 路由
│   ├── core/config.py        # Settings + .env 加载
│   └── services/agent_service.py  # Agent 单例装配
├── rules/fastpath_rules.json # P0 毒物库 + 药品白名单 + watchlist
├── knowledge/                # 权威医疗文档（17 chunks）
├── tests/                    # 单元测试 + eval_dataset.json
├── scripts/
│   ├── ingest_docs.py        # 入库脚本（默认 upsert，--rebuild 重建）
│   └── evaluate_rag.py       # 评测脚本（Recall@3 / MRR / 拒答率）
├── web/index.html            # 单文件原生前端
└── data/chroma/              # ChromaDB 持久化（.gitignore 忽略）
```

## 当前成绩单（V2.0）

| 指标 | 数值 |
|---|---|
| 单元测试 | **59 全绿** |
| Pure RAG Recall@3 | **100%** (4/4) |
| 处方药拒答率 | **100%** (3/3) |
| System Pass Rate | **12/12 (100%)** |
| Fast-Path 拦截 TTFB | **3.2ms** |
| BM25 冷启动重建 | 13ms（17 chunks） |

## 快速开始

```powershell
# 0. 设置 HuggingFace 镜像（国内拉取 BGE 模型必须）
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 1. 安装依赖（Python 3.11+）
pip install -e .

# 2. 配置模型 API：复制 .env.example 为 .env，填入 OPENAI_API_KEY / OPENAI_BASE_URL

# 3. 启动后端
uvicorn app:app --port 8000 --reload

# 4. 运行测试 / 评测
pytest tests/ -q
python -m scripts.evaluate_rag

# 5. 知识库入库（默认 upsert；--rebuild 清空重灌）
python -m scripts.ingest_docs
```

## 部署

```powershell
docker compose up -d
```

## 仓库

<https://github.com/AgEz-dev/VetClaw>

# Sentinel-Agent

面向开发调试场景的轻量自愈 Agent 与异步服务网关。

## 概述

在开发与调试场景中，定位、复现和修复问题往往需要跨多个信息源：本地代码、文档知识库、运行日志。Sentinel-Agent 将大模型接入一个自主调度循环，使其能够按需调用工具、检索私有知识库、观察执行结果并自我纠错，同时通过异步服务网关以流式方式对外提供能力，为构建智能开发助手类应用提供可扩展的底座。

## 技术栈

Python 3.11+ · FastAPI · 原生 ReAct 状态机 · ChromaDB · SSE · Ollama / 硅基流动云端 API

## 核心模块

- **ReAct 循环调度器**：Thought → Action → Observation 的自主推理-行动循环
- **Tool-Calling 注册表**：工具动态注册、JSON Schema 自动提取与安全确认机制
- **RAG 混合检索**：BM25 + 向量召回，本地知识库语义检索
- **滑动窗口记忆**：Token 预算控制，自动剔除旧轮次，防止上下文溢出
- **SSE 流式推送**：思考过程、工具调用与最终答案逐字输出
- **会话互斥锁**：并发会话下共享状态的隔离与保护

## 项目结构

```text
ai-agent-study/
├── README.md            # 项目说明
├── .gitignore
├── pyproject.toml       # 依赖与工程配置
├── src/                 # 核心实现（Agent 调度器）
│   └── core_agent.py
├── schemas/             # Pydantic 请求/响应模型
├── tests/               # 测试
├── docs/
│   └── architecture.md  # 架构说明
├── datasets/            # 数据集
├── benchmarks/          # 评测
├── knowledge/           # 知识库语料
└── data/                # 运行时数据（向量库等）
```

## 架构

三层架构设计与关键技术决策见 [docs/architecture.md](docs/architecture.md)。

## 开发状态

- [x] 工程脚手架
- [ ] 核心调度层（ReAct 循环、工具注册表、滑动窗口记忆）
- [ ] 能力层（RAG 检索、代码阅读工具）
- [ ] 接入层（FastAPI 网关、SSE 流式）
- [ ] 容器化部署

## 快速开始

```powershell
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活（Windows）
.venv\Scripts\activate

# 3. 安装依赖（含开发依赖）
pip install -e ".[dev]"

# 4. 配置模型 API
#    Ollama 本地：设置 OPENAI_BASE_URL=http://localhost:11434/v1
#    硅基流动：设置 OPENAI_BASE_URL=https://api.siliconflow.cn/v1 与 OPENAI_API_KEY
#    也可在项目根目录创建 .env 文件（已被 .gitignore 忽略）

# 5. 运行测试
pytest
```

> 运行入口将在核心调度层完成后补充。

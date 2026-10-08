# VetClaw 架构说明

## 三层架构

VetClaw 是一个宠物健康分诊 Agent，分为三层，职责单一、可独立演进。

### 1. 接入层（FastAPI 工程网关）

对外提供 HTTP 接口与 SSE 流式通道。

- Pydantic V2 校验层：拦截非法入参（空 prompt 等）返回 422
- SSE 流式推送：`thought / tool_call / tool_result / token / done / error` 六类事件
- 全局异常兜底：握手前异常由全局 handler 兜住；流式生成器内部异常由 `api.py` 的 `generate()` 自行兜底转 `error` 事件（StreamingResponse 握手后状态码已发 200，外层处理器拦不到）

### 2. 调度层（原生 ReAct 调度中枢，零第三方框架）

- 统一事件生成器 `_run_events()`：同时驱动 `run()`（同步）与 `run_stream()`（流式），行为 100% 一致
- Tool-Calling 注册表：`inspect` 反射自动提取 JSON Schema，危险操作前缀拦截
- 流式 tool_calls 分片重组：`name` 仅初次赋值、`arguments` 累加
- `cite_state` 三态状态机（None / hit / miss）：同轮内一旦命中不可被后续工具覆盖，驱动引用补正与降级
- 双层超时：SDK read timeout（单次请求字节间隔）+ Agent 总时长守卫（整个 ReAct 生命周期）
- 可切换引擎：默认 `react`（即上列实现）；设 `VETCLAW_ENGINE=langgraph` 可切到另一个可选引擎（默认不装），见 §5

### 3. 能力层（工具集）

- **Fast-Path 前置规则守卫**：P0 致命毒物秒级拦截（ChromaDB ID 点查 SOP + 毒物 chunk，<3ms）；未收录处方药剂量咨询 100% 触发分诊引导
- **RAG 混合检索**：结构感知 Markdown 切片（表格整块保护）+ BGE-small-zh-v1.5 本地向量 + BM25(jieba) 稀疏召回，RRF(k=60) 融合重排
- **滑动窗口记忆**：按「assistant + tool_calls 原子块」裁剪，system prompt 预算保护

## 关键决策

### 1. 为什么不用 LangChain

- 框架重度黑盒，推理链路不透明，问题难以定位
- 框架抽象带来额外学习成本，掩盖 Agent 本质
- 原生实现依赖少、可控性强，便于理解原理与按需扩展
- 代价：需要自行处理状态管理、错误恢复与兼容性细节

> 以上针对**默认调度路径**。另有一个可选的 LangGraph 对照引擎（默认不装、不启用），见 §5。

### 2. 为什么 Fast-Path 前置拦截

- P0 急症（猫误食百合）走 LLM 需 2-3 秒，规则匹配 <5ms，急症不能等
- 安全生杀大权不能交给模型「愿不愿意调工具」，必须入口前置、确定性熔断
- 未收录处方药直接分诊引导，不调 LLM 编剂量

### 3. 为什么 BM25 + Dense 混合检索

- 纯向量检索对抽象语义（「送医时间窗口」）与专有名词（「塞拉菌素」）召回差
- BM25 补关键词召回，RRF 融合重排，两路优势互补

### 4. 为什么选 ChromaDB

- 本地嵌入式向量库，零运维、无需独立服务
- Python 原生 API，与调度层集成简单；`collection.get(ids=[...])` 为纯键值点查，毫秒级
- 代价：单机内存依赖，大规模场景需评估

### 5. 为什么另起一个 LangGraph 对照引擎

**不是为了替换手写实现，而是为了有一份可对比的参照。**

**三个理由**

1. **交叉验证** —— 同一套事件契约下有两套实现，可以互相校验：同一批用例分别跑两条路径、比对事件序列，这个不变量可以机器断言；改动其中一套时，另一套就是现成的回归参照。
2. **手写版本在变复杂** —— `_run_events()` 一个生成器里塞了六段逻辑（守卫 / 组装消息 / LLM 流式 / 工具执行 / 合规校验 / 收尾降级）。要继续加记忆持久化、断点续跑会越来越难改；LangGraph 的 checkpointer 自带 thread 级持久化。
3. **流程本来不是「链」** —— `tools ⇄ llm` 是**回环**，合规校验不过还要回到 `llm` 重写。「链」（LCEL 那类）表达不了回环，「图 + 条件边 + 循环」才是合适的形态。

**代价**

| 维度 | 实测 |
|---|---|
| 新增依赖 | 16 包 / 下载 6.9 MB / 磁盘 ≈38.7 MB |
| `import langgraph` 冷启动 | **≈859 ms** |

> 859 ms 会破坏 P0「3.2 ms / 冷启动」的口径，所以所有 langgraph 符号都放在**函数体内**懒加载，只在 `VETCLAW_ENGINE=langgraph` 时才 import；并由一条 **AST 静态检查**在 CI 中守住这条规则。

**边界**

- **不删除、不替换** `ReActAgent` —— 两个引擎并存，手写版作为对照基准；
- **默认仍是 `react`** —— 线上行为不变，六种 SSE 事件契约不变；
- **验收 = 事件序列 diff 为空**：同一输入下两引擎逐事件、逐字段一致（`tests/test_graph_agent.py`，15/15 通过）。

> 选型上：LCEL 是「链」不是「图」，表达不了回环；LlamaIndex 偏 RAG 索引侧；AutoGen / CrewAI 面向多 Agent 协作。本项目是单 Agent + 工具，且流程有回环。

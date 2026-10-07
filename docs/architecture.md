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

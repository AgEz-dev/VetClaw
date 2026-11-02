# HANDOFF · VetClaw 真 A2A（2026-10-17）

> 面向**接手的 agent**。本文档不依赖任何对话上下文，读完即可接手。
> 作者：deepseek-flash。交接时两个仓库的改动**全部未提交**（见 §2）。

---

## 1. 一句话现状

「真 A2A」已落地并验证：**编排权在 VetClaw**，医院侧通过 `POST /a2a/task`（同步）与
`POST /a2a/task/stream`（SSE）提交结构化任务；四个业务能力位（分诊/预问诊/起草/处方审核）
全部走 case 契约，问答位（`consult`）的流式能力已就绪但**医院侧尚未切换**。

| 仓库 | 测试 | 状态 |
|---|---|---|
| `D:\Forwork\VetClaw` | **258 passed**（186 既有全绿 + 72 新增） | 未提交 |
| `D:\workspace\pet-hospital-system` | **198 passed**（189 集成 + 9 单元）`BUILD SUCCESS` | 未提交 |
| 医院侧前端 | `npm run build` exit 0 | 未提交 |

---

## 2. ⚠️ 未提交改动（**接手第一件事：先看这里，别 reset**）

两个仓库都是**工作区脏、HEAD 干净**的状态，没有产生任何 commit。

- VetClaw HEAD = `3923a0d`，医院侧 HEAD = `ea20891`
- 建议接手后先 `git -C <repo> status --short` 确认，再决定是否提交
- **禁止** `git checkout -- .` / `git reset --hard` / `git clean`（会抹掉整轮工作）

未提交清单（VetClaw）：`README.md`、`app.py` 修改；`docs/a2a-contract.md`、`rules/prescription_safety_table.json`、
`schemas/a2a.py`、`src/a2a/`、`src/prescription_review.py`、`tests/test_a2a.py`、
`tests/test_a2a_stream.py`、`tests/test_prescription_review.py` 新增。

未提交清单（医院侧）：13 个修改（含 `PrescriptionController`、`AiIntakeController`、
`IntakeTurnClient`、`RecordDraftClient`、`PetPrescriptionServiceImpl`、`PetPrescription`、
`ResultCode`、前端 3 个文件）+ 8 个新增（见 `git status`）。

---

## 3. 本机环境（有坑，照着来）

| 项 | 事实 |
|---|---|
| Python | `D:\python`，3.13.0，`python -m pytest` 可用 |
| Maven | **不在 PATH**。用 `& 'D:\APP\data\Java Program\宠物安\apache-maven-3.9.16\bin\mvn.cmd'`（路径含**中文与空格**，必须整段引号） |
| Java | 11.0.0.2 |
| MySQL | 8.0.39，服务 `MySQL80` 已运行；本地开发凭据在 `backend/src/main/resources/application.yml`（`root` / `123456`） |
| 数据库 | `pet_hospital`（应用库）、`pet_hospital_test`（测试库） |
| Node | v22.23.2，`frontend/node_modules` 已安装 |
| Shell | ⚠️ `pwsh` 实际是 **PowerShell 5.1 语义**（没有 `-SkipHttpErrorCheck`；`.GetResponseStream()` 不存在） |

### 必须知道的四条命令陷阱

1. **HTTP 联调用 Python `urllib`，不要用 PowerShell cmdlet**（版本差异 + 中文编码两重坑）。
2. **`| Select-Object -First N` 会掐断上游进程**：对 `uvicorn` 这类长驻服务用它 ≈ 运行 N 行后杀掉。
   要么不接管道（后台 job 自己收集输出），要么 `Tee-Object` 落盘再读文件。
3. **控制台是 GBK，中文显示成乱码**。读写含中文的文件用 Python `io.open(..., encoding='utf-8')`
   或 `[IO.File]::ReadAllText($p, [Text.Encoding]::UTF8)`，不要走 PowerShell 文本管道。
4. **`mvn` 必须带 `-o`（离线）**：`~/.m2` 已有全部依赖；不加 `-o` 可能尝试联网而卡住。

---

## 4. 已完成 & 已验证（按能力位）

### 4.1 VetClaw 侧（编排核心）

| 文件 | 职责 |
|---|---|
| `src/a2a/contract.py` | `CaseTask` / `TaskResult` / `CaseState`；`Step`（triage/intake/draft/review/consult）；`TaskStatus`（ok/need_more/failed）× `Verdict`（none/pass/warn/block）；`worst_verdict` 合并取最严重 |
| `src/a2a/store.py` | SQLite：`a2a_case`（case 状态）+ `a2a_result`（幂等回放表，存**请求指纹**） |
| `src/a2a/agents.py` | 4 个同步能力位 + `consult_stream` 流式能力位；`AGENTS` / `STREAM_AGENTS` 注册表 |
| `src/a2a/supervisor.py` | `submit()`（同步）/ `submit_stream()`（流式）/ `assert_no_conflict()`（握手前 409） |
| `src/a2a/stream.py` | 流式帧协议（`start`/`thought`/`tool_call`/`tool_result`/`token`/`done`/`error`）+ `to_sse` |
| `src/a2a/routes.py` | `POST /a2a/case`、`GET /a2a/case/{id}`、`POST /a2a/task`、`POST /a2a/task/stream` |
| `src/prescription_review.py` | **AI-4 处方安全审核**（纯规则表驱动，9 类闸门，每条带 `chunk_id`） |
| `rules/prescription_safety_table.json` | 安全约束镜像（**只镜像 `knowledge/` 已载明内容**） |
| `docs/a2a-contract.md` | **对外契约（§9 是流式）** —— 医院侧对接的唯一依据 |

### 4.2 医院侧

| 文件 | 职责 |
|---|---|
| `service/ai/A2aTaskClient.java` | 通用任务契约客户端（同步 JSON） |
| `service/impl/PrescriptionReviewServiceImpl.java` | 送审 + 落库 + 清旧放行（HITL） |
| `service/impl/PetPrescriptionServiceImpl.java` | 发药闸门 `assertReviewAllows()` |
| `controller/PrescriptionController.java` | `/{id}/review`、`/{id}/review/override` |
| `service/ai/IntakeTurnClient.java` | 预问诊 opening/turn → `/a2a/task`（**状态不再往返**） |
| `service/ai/RecordDraftClient.java` | 起草 → step=draft |
| `resources/sql/migration_v1.2_a2a_review.sql` | 幂等迁移（7 个 `review_*` 字段），**已应用到两个库** |
| `frontend/src/views/doctor/Workstation.vue` | 审核面板 + 发药置灰 + 放行弹窗 |

### 4.3 验证过的行为（可直接复跑，见 §8）

- **预问诊连续性**（真实 HTTP，7/7）：开场 → 第 1 轮（只传 prompt）→ 第 2 轮，槽位逐轮推进且**不重复提问**；同键同内容重放 `deduped=True` 且**轮次不推进**；同键改内容 → 409。
- **处方闸门**（集成测试 7 项）：`block` 未放行 → 发药 409 且库存不扣；人工放行（留痕）后可发药；放行不填理由 → 400；`pass` 无阻断可推翻 → 409；审核不可用 → 503 且不污染结论；详情接口透出审核字段。
- **流式端点**（真实 uvicorn，帧序）：`start → thought → error`，**终态恰好 1 个**；`start` 携带 `case/step/deduped`。

---

## 5. 绝不能破坏的 6 条不变量

1. **双引擎零漂移**：`execute_tool` / `compliance_fix` / `call_llm` 是 `core_agent` 与 `graph_agent` 共享的纯函数。
   改它们必须同时过 `tests/test_graph_agent.py`（断言事件序列 diff 为空）。
2. **冷启动口径**：`src/` 顶层禁止 import 重依赖；`tests/test_engine_switch.py:87` 是 AST 检查。
3. **幂等三态**：同键同内容 → 回放；同键**异**内容 → 409；**`FAILED` 不记忆**（否则一次抖动让任务永久失败）。
   ⚠️ 推论：**冲突检测只对「已存记录」生效** —— 上一次失败后，同键换内容会照常执行。
4. **`status` × `verdict` 双轴**：`status` 说「跑没跑成」，`verdict` 说「业务结论」。合并会让下游分不清「该重试」与「该拦人」。
5. **闸门不对称**：`block` **始终**拦截，不受 `ai.a2a.review-required` 影响（否则等于留一条"关开关就能发毒药"的路）；该开关只决定「未审核」是否拦。
6. **规则表只镜像文档**：`rules/prescription_safety_table.json` 每条必须带 `chunk_id`；
   有测试 `test_all_table_citations_point_to_knowledge_chunks` 强制。**不要凭医学常识往里加规则**。

---

## 6. 已知陷阱（都是踩过的）

| 陷阱 | 说明 |
|---|---|
| MyBatis-Plus `updateById` 忽略 null | 想清空列必须用 `Wrappers.lambdaUpdate().set(col, null)`；重审清旧放行就靠这个 |
| SSE 握手后改不了状态码 | 幂等冲突必须在**握手前**同步判定（`assert_no_conflict`），否则只能退化成 `error` 帧 |
| `/opening` 必须走契约且带 `opening:true` | 否则它会作为一次 intake 步**覆盖** case 里已积累的预问诊成果；这也是首轮 `turn` 不重复提问的前提 |
| `frontend/dist/` | 已被 `.gitignore` 忽略，构建不会污染仓库 |
| Vue 未知全局组件不阻断构建 | 用全局注册的 Element Plus 图标前先 grep 包的 `.d.ts`，或改显式 import |

### 旧接口现状（已核实）

- `/api/intake/stream` —— **已无调用方**（`IntakeTurnClient` 的 SSE 代码已随 opening 迁入契约而删除，
  现在类里只有 `A2aTaskClient` 一条传输）。端点在 VetClaw 侧仍存在（`api.py` 的 `/api/intake/stream`），
  但医院侧不再使用；**保留它是为了回滚安全，不要急着删**。
- `/api/chat/stream` —— **仍在用**，调用方是 `VetClawAiProvider.java:69`，是唯一未迁移的能力位。
- `/api/record/draft` —— 医院侧已改用 `/a2a/task`（step=draft），该端点同理保留。

---

## 7. 未完成的工作（按建议顺序）

### 7.1 `ai.a2a.review-required` 置 `true`（当前 `false`）

现状：`block` 已硬拦；「未审核」（`review_*` 为 NULL）默认放行并打 WARN 日志。
置 `true` 后存量处方会被拦 —— **前置条件**：先补一个存量补审工具。

建议顺序：**补审工具 → 观察一轮 → 再置 true**，不要直接开开关。
补审工具可复用 `PrescriptionReviewService.review(id)`，逐条待发药处方跑一遍；
注意 `review()` 依赖 VetClaw 可达，未配 `ai.vetclaw.base-url` 会 503。

### 7.2 问答切到流式契约

能力已就绪（`Step.CONSULT` + `consult_stream` + `tests/test_a2a_stream.py` 21 项）。
医院侧切换只需改三处：

1. `VetClawAiProvider` 的 URL：`/api/chat/stream` → `/a2a/task/stream`；
2. 请求体补 `case_id`（沿用 `appt-<id>`）与 `idempotency_key`（建议含问题内容指纹）；
3. `thought/tool_call/tool_result` 仍按「仅后端留痕」处理，**不需要改**。

中继的 `switch`、`done` 补 delta、`error` 带 trace_id **都不用动**（`done` 帧是旧 `done` 的超集）。
切换前复跑 `test_engine_switch` + 医院侧全量测试。

### 7.3 完全没做的

- **前端浏览器实机点验**：只验证到「能构建 + 图标存在 + 后端字段到位」。布局/tooltip 位置/文案换行**未经人眼确认**。
- **真实模型 E2E**：全程无 `OPENAI_API_KEY`，问答与起草的真实调用路径没有端到端跑过（用替身/降级验证）。
- **上一轮评估里的 A2/A3**：span 树 + token 计量、轮次质量判定 —— 未开始（契约里也没有）。
- **Langfuse / OTel**：未接入。

---

## 8. 回归命令（可直接复制）

```powershell
# VetClaw（258 expected）
cd D:\Forwork\VetClaw
python -m pytest -q

# 医院侧后端（198 expected）
cd D:\workspace\pet-hospital-system\backend
& 'D:\APP\data\Java Program\宠物安\apache-maven-3.9.16\bin\mvn.cmd' -o test

# 医院侧前端（exit 0 expected）
cd D:\workspace\pet-hospital-system\frontend
npm run build

# 只跑本次新增的两组（快）
cd D:\Forwork\VetClaw
python -m pytest tests/test_a2a.py tests/test_a2a_stream.py tests/test_prescription_review.py -q
```

起 VetClaw 做手工联调（无 API Key 也可跑审核/预问诊规则路径）：

```powershell
cd D:\Forwork\VetClaw
$env:OPENAI_API_KEY = 'sk-dummy-e2e'
$env:OPENAI_BASE_URL = 'http://127.0.0.1:9/v1'   # 不可路由 → 模型调用快速失败，走规则兜底
python -m uvicorn app:app --port 8124
# 探活：Invoke-RestMethod http://127.0.0.1:8124/health
```

---

## 9. 建议的阅读顺序（30 分钟上手）

1. `docs/a2a-contract.md` —— 对外契约，先看懂它，其余都是实现细节
2. `src/a2a/contract.py` —— 数据结构与 `worst_verdict`
3. `src/a2a/supervisor.py` —— 编排顺序（截止时间 → 幂等 → 载 case → 派发 → 落库）
4. `src/prescription_review.py` —— 唯一的医学判定处（规则表驱动）
5. `service/impl/PetPrescriptionServiceImpl.java` 的 `assertReviewAllows()` —— 闸门
6. `frontend/src/views/doctor/Workstation.vue` 里 `reviewMeta` 计算属性 —— 展示层

---

## 10. 过程复盘（如需了解踩过的坑）

`D:\Forwork\agent任务价值素材清单\deepseek-flash-复盘\2026-10-17-VetClaw生产级缺口核实.md`

其中 P3（严重度：高）是「只读被调用方仓库就断言无真实分工」——直接导致最初误判 A2A 无价值；
P5（高）是「差点凭医学常识编造药物交叉过敏表」。接手时值得一读，避免重犯。

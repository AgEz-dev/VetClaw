# A2A 任务契约 v1

> 面向调用方（`pet-hospital-system`）：编排权在 VetClaw，调用方只提交**结构化数据 + 幂等键**。

## 1. 为什么改

旧契约下，医院侧对 VetClaw 是「四个互不相干的 HTTP 端点」，每次调用显式传空 history
（`VetClawAiProvider.buildBody` 的「单轮语义」）。后果是**同一次就诊的四段 AI 互不知情**：

- 预问诊问了 12 轮，起草时调用方还得把摘要与对话**重新组装再传一遍**；
- 预问诊的续传状态要**每轮往返**（`to_state` → Java 落库 → 下次带回）；
- 没有任何地方能回答「这次就诊 AI 做完了哪些判断」。

新契约把编排收进 VetClaw：**一次就诊 = 一个 case**，四个能力位共享 case 状态；
调用方不再决定「下一步做什么」，也不再搬运上下文。

## 2. 端点

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/a2a/case` | 建案（幂等，已存在则返回既有状态） |
| GET | `/a2a/case/{case_id}` | 查询 case 全量状态（四步产出 + 宠物事实 + seq） |
| POST | `/a2a/task` | 提交一次能力位任务 |

`case_id` 建议由医院侧按业务键派生，如 `appt-<apptId>`，保证同一预约可重复建案。

## 3. 契约对象

### CaseTask（请求体）

```json
{
  "step": "review",
  "case_id": "appt-1024",
  "idempotency_key": "appt-1024:review:rx-88:v3",
  "payload": { },
  "task_id": "",
  "caller": "pet-hospital-system",
  "deadline_at": 1793000000.0,
  "confidence_threshold": 0.6
}
```

- `idempotency_key` **必填**，缺失直接 400。语义见 §5。
- `deadline_at`（epoch 秒）过期 → 任务失败、不执行。
- `confidence_threshold` 仅回传参考，闸门由调用方执行。

### TaskResult（响应 `data`）

```json
{
  "case_id": "appt-1024",
  "task_id": "task_ab12...",
  "step": "review",
  "status": "ok",
  "verdict": "block",
  "payload": { },
  "confidence": 0.95,
  "reasons": [{"level": "block", "code": "ALLERGY_CONTRAINDICATION", "message": "…", "citation": "knowledge/…#…"}],
  "citations": ["knowledge/prescription_and_otc_drugs.md#速诺-阿莫西林克拉维酸钾-2"],
  "trace_id": "trace_9f…",
  "created_at": 1793000000.0,
  "deduped": false
}
```

**`status` 与 `verdict` 必须分开读**（这是本契约的核心设计）：

| 字段 | 取值 | 含义 | 调用方应对 |
|---|---|---|---|
| `status` | `ok` | 跑完了 | 看 verdict |
| | `need_more` | 输入不足，`payload.missing` 列出缺的字段 | 补字段后用**新**幂等键重提 |
| | `failed` | 执行失败（模型不可用/解析失败/超时） | **同一幂等键**重试 |
| `verdict` | `pass` / `warn` / `block` / `none` | 业务结论（仅 review 步有值） | 见 §6 |

把两者混成一个字段，下游就没法区分「模型挂了（该重试）」与「药有问题（该拦人）」。

## 4. 四个能力位的 payload

### triage（分诊）

```json
{ "query": "猫误食了百合花怎么办", "patient": { "species": "猫" } }
```
产出 `payload: {action: emergency|refuse|pass, toxin, species, chunk_id, slots, answer}`。
`emergency` → `verdict=block`，`refuse` → `warn`。

### intake（预问诊）

**开场**（进入页面时取问候 + 首批追问）：

```json
{ "opening": true, "archive": {"allergy": "青霉素"} }
```
`opening:true` 表示「用**全新会话**返回开场白」，**不继承** case 里已有的进度
（否则会得到一句与当前进度矛盾的问候）。幂等键建议只由 `case_id + 档案` 决定，
这样重复进入页面只会回放同一段开场，不会覆盖已积累的预问诊成果。

**业务轮次**（宠主说了一句话）：

```json
{ "prompt": "昨天开始吐，今天两次", "archive": {"allergy": "青霉素"} }
```
只需 `prompt`：续传状态**自动从 case 取**，不再要求调用方往返 `state`。
幂等键建议为 `case_id:intake:t<轮次>:<本轮内容指纹>` —— 同一轮重试只会回放，
不会把会话多推进一格。

产出 `payload: {mode, done, message, questions:[{key,ask}], slots, state, intake:{summary,…}}`。

### draft（病历起草）

```json
{ "summary": {}, "dialogue": [], "archive": {}, "patient": {} }
```
**以上字段全部可空**：缺省时自动取 case 内 `intake` 那一步的产出。产出
`payload: {draft: {chief_complaint, exam_checklist, doctor_advice, disclaimer}}`。
红线不变：**输出白名单里没有 diagnosis**。

### review（处方安全审核）

```json
{
  "patient": { "species": "猫", "weight_kg": 3.2, "age_weeks": 52,
               "allergies": "青霉素", "pregnant": false },
  "prescription": {
    "rx_id": 88,
    "items": [
      { "drug_name": "速诺", "dose_mg_per_kg": 12.5,
        "frequency_per_day": 2, "duration_days": 7, "quantity": 14,
        "specification": "250mg", "form": "猫专用" }
    ]
  }
}
```

- `patient` 缺省时**自动取 case 内宠物事实**（同一 case 只应录一次）。
- 剂量以 `dose_mg_per_kg`（数值）表达，**不接受自由文本**——看不懂就 WARN，不猜。
- 产出 `payload: {verdict, confidence, reasons[], citations[], checks, undetermined[]}`。

## 5. 幂等语义

| 情形 | 行为 |
|---|---|
| 同键 + 同内容指纹 | **回放**上次结果，`deduped=true`，不重复执行 |
| 同键 + **不同**内容指纹 | **409 `IDEMPOTENCY_CONFLICT`** |
| 同键 + 上次 `failed` | **重新执行**（失败不记忆） |
| 同键 + 上次 `need_more` / `ok` | 回放 |

两点必须理解：

1. **不同内容绝不回放**。若用同键改了处方再重试，静默返回旧结论会直接造成用药安全事故。
2. **失败不记忆**。模型超时、网关 5xx 属同一次业务意图的瞬时失败，用同键重试是正确行为；
   若把失败也记进幂等表，一次抖动就会让该任务永久失败。

## 6. Verdict → 医院侧闸门

| verdict | 建议动作 |
|---|---|
| `pass` | 可发药 |
| `warn` | 可发药，但需医生确认（转 HITL 提示） |
| `block` | **禁止发药**，必须人工复核后方可放行 |
| `none` | 该能力位无审核结论（triage/intake/draft） |

> 规则层命中的 `block` 由确定性规则直接产出，任何上层（含未来的模型复核）都不能把它降级。

## 7. 错误码

| HTTP | code | 场景 |
|---|---|---|
| 400 | `INVALID_TASK` | 契约校验失败 |
| 404 | `CASE_NOT_FOUND` | case 不存在 |
| 409 | `IDEMPOTENCY_CONFLICT` | 幂等键被用于不同内容 |
| 422 | — | 请求体不符合 Pydantic 模型 |
| 500 | `INTERNAL_ERROR` | 内部错误（不泄漏 traceback） |

## 8. 与旧契约的差异

| | 旧 | 新 |
|---|---|---|
| 编排权 | 调用方（Java 逐段编排） | VetClaw（supervisor） |
| 上下文 | 每次空 history，摘要重传 | case 级共享，自动继承 |
| 预问诊状态 | 每轮往返 `state` | 存 case，调用方只传本轮发言 |
| 重试安全 | 无幂等概念 | 幂等键 + 内容指纹 + 冲突拒绝 |
| 结论语义 | 只有 `done`/`error` | `status`（执行）× `verdict`（业务）二维 |
| 处方安全 | 无服务端校验 | `review` 能力位（规则闸门 + 引用溯源） |

## 9. 流式端点（`POST /a2a/task/stream`）

请求体与 `/a2a/task` **完全相同**，响应为 `text/event-stream`。

### 9.1 帧词汇（与旧 `/api/chat/stream` 同构）

| 帧 | 何时 | data |
|---|---|---|
| `start` | 首帧，收到任何内容前 | `case_id` / `task_id` / `step` / `trace_id` / `deduped` |
| `thought` | 阶段变化 | `stage` |
| `tool_call` | 调用工具 | `id` / `name` / `arguments` |
| `tool_result` | 工具返回 | `id` / `content` |
| `token` | 逐字正文 | `text` |
| `done` | **终态**（成功或待补） | `answer` / `citations` + `case_id` / `task_id` / `step` / `status` / `verdict` / `confidence` / `trace_id` / `deduped` |
| `error` | **终态**（失败） | `code` / `message` / `trace_id` |

`done` 是旧 `done` 事件的**超集**：`answer` / `citations` 原样保留，因此医院侧现有
SSE 中继（`VetClawAiProvider`）**无需改写**即可消费，切换只需改 URL 并补 `case_id` + 幂等键。

### 9.2 终态保证

**一次请求有且只有一个终态帧。**

- `status = ok` / `need_more` → `done`（`need_more` 不是失败，与同步端点的 HTTP 200 一致）
- `status = failed` → `error`

客户端只需判断「终态是 done 还是 error」，不必为每种 status 分支。

### 9.3 回放语义（与同步端点**不同**，务必注意）

流无法真正"续传"。同一幂等键 + 同内容重试时：

1. 首帧 `start` 带 `deduped=true`；
2. 已存答案作为**一整段** `token` 下发（不是逐字重放）；
3. 接 `done`。

即：**重试拿到的是完整内容，不是从断点继续**。要实现"断点续传"，需要额外的游标协议，当前不做。

### 9.4 两条已知边界

| 情形 | 行为 | 原因 |
|---|---|---|
| 幂等键冲突 | **握手前**返回真 HTTP `409` | 一旦开始推帧，状态码已发出，只能退化成 `error` 帧 |
| 客户端中途断开 | 结果**不落库**，重试会**重新执行** | 把半截结果写进幂等表比重跑更糟（会永久返回残缺答案） |

> ⚠️ **冲突检测只对「已成功（或待补）的结果」生效。** 因为失败不记忆（§5），
> 若上一次尝试的终态是 `error`，幂等表里**没有记录**，那么用同一个键提交**不同内容**
> 会**照常执行**、不会返回 409。这是刻意取舍：一次网络抖动不该把该键永久钉死。
> 若调用方需要「无论成败都不允许换内容」，得自己在业务侧记住内容指纹。

### 9.5 能力位支持情况

| 能力位 | 同步 `/a2a/task` | 流式 `/a2a/task/stream` |
|---|---|---|
| `triage` / `intake` / `draft` / `review` | ✅ | ❌ `STEP_NOT_STREAMABLE` |
| `consult`（问答） | ❌ `STEP_NOT_STREAMABLE` | ✅ |

> 双向都会给出明确的 `STEP_NOT_STREAMABLE` 指引，不会返回含糊的「未知能力位」。

**当前状态**：`consult` 的流式能力已实现并测试，但**医院侧尚未切换**（问答仍走
`/api/chat/stream`）。切换的前提是先落地医生端的审核结论展示等体验项，
再一次性把 `VetClawAiProvider` 指向本端点。

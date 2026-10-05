"""ReAct 调度核心：统一事件生成器 _run_events 同时驱动 run() 与 run_stream()（零 LangChain）。"""
import json
import re
from types import SimpleNamespace

CITATION_MARK = "【来源："
HIT_MARK = "【知识库检索结果】"
MISS_MARK = "【检索未命中】"
DEGRADE_ANSWER = "知识库中未检索到相关信息，无法提供确切解答，建议补充文档或联系维护者。"
CITATION_RE = re.compile(r"【来源：[^】]+】")

DEFAULT_SYSTEM_PROMPT = """你是 Sentinel-Agent，一个可以使用工具的技术助手。严格遵守：

1. 闲聊或自我介绍（如“你好”“你能做什么”）：直接自然回答，不要调用工具。
2. 涉及项目技术细节的问题（错误排查、部署、配置、生产参数）：必须先调用
   search_knowledge_base，再仅依据检索结果回答。
3. 引用：凡依据检索结果给出的事实，句末必须标注来源，格式为【来源：<chunk_id>】，
   chunk_id 形如 knowledge/deploy.md#1；多条依据分别标注。
4. 防幻觉：只能使用检索片段中的信息，不得补充片段之外的技术参数或推测。
   检索为空或收到【检索未命中】时，直接回复：
   “知识库中未检索到相关信息，无法提供确切解答，建议补充文档或联系维护者。”

示例：
问：服务怎么启动？
答：执行 docker compose up -d 即可拉起全部服务【来源：knowledge/deploy.md#1】
"""


class ReActAgent:
    def __init__(self, client, registry, model="gpt-4o-mini",
                 max_steps=5, system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.client = client
        self.registry = registry
        self.model = model
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def run(self, prompt, history=None):
        # 消费统一事件流：done 取 answer、error 取 message，与流式行为 100% 一致。
        for ev in self._run_events(prompt, history):
            if ev["event"] == "done":
                return ev["data"]["answer"]
            if ev["event"] == "error":
                return ev["data"]["message"]
        return ""

    def run_stream(self, prompt, history=None):
        yield from self._run_events(prompt, history)

    def _run_events(self, prompt, history=None):
        # cite_state 为生成器内局部状态、随每次调用新建，严禁挂 self，杜绝跨请求污染。
        # 已知边界：技术问题若模型自行跳过检索（cite_state 始终 None），单 Agent 无法
        # 拦截，属 Prompt 规划层问题；后续靠微调或独立 Router 分类解决。
        cite_state = None
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        yield {"event": "thought", "data": {"stage": "thinking"}}

        for _ in range(self.max_steps):
            tool_parts, text_parts = {}, []
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages,
                tools=self.registry.schemas(), stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta
                for p in (getattr(delta, "tool_calls", None) or []):
                    idx = getattr(p, "index", 0) or 0
                    slot = tool_parts.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if getattr(p, "id", None):
                        slot["id"] = p.id
                    fn = getattr(p, "function", None)
                    if fn:
                        if getattr(fn, "name", None) and not slot["name"]:
                            slot["name"] = fn.name  # 仅初次非空赋值，严禁 += 累加 name
                        if getattr(fn, "arguments", None):
                            slot["arguments"] += fn.arguments
                piece = getattr(delta, "content", None)
                if piece:
                    text_parts.append(piece)
                    yield {"event": "token", "data": {"text": piece}}

            if tool_parts:
                calls = [{"id": v["id"], "type": "function",
                          "function": {"name": v["name"], "arguments": v["arguments"]}}
                         for _, v in sorted(tool_parts.items())]
                messages.append({"role": "assistant", "content": None, "tool_calls": calls})
                for c in calls:
                    yield {"event": "tool_call", "data": {"id": c["id"],
                           "name": c["function"]["name"],
                           "arguments": _safe_args(c["function"]["arguments"])}}
                    yield {"event": "thought", "data": {"stage": "executing_tool"}}
                    tc = SimpleNamespace(id=c["id"], function=SimpleNamespace(
                        name=c["function"]["name"], arguments=c["function"]["arguments"]))
                    content = self._run_tool(tc)
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
                    yield {"event": "tool_result", "data": {"id": c["id"], "content": content}}
                    if content.startswith(HIT_MARK):
                        cite_state = "hit"  # hit 优先级最高，不被后续工具/miss 覆盖
                    elif content.startswith(MISS_MARK) and cite_state != "hit":
                        cite_state = "miss"
                yield {"event": "thought", "data": {"stage": "thinking"}}
                continue

            final = "".join(text_parts)
            fix = self._compliance_fix(cite_state, final)
            if fix:
                messages.append({"role": "user", "content": fix})
                yield {"event": "thought", "data": {"stage": "thinking"}}
                continue
            yield {"event": "done", "data": {
                "answer": final, "citations": CITATION_RE.findall(final)}}
            return

        message = DEGRADE_ANSWER if cite_state in ("hit", "miss") \
            else f"已达到最大步数 {self.max_steps}，强制停止。"
        yield {"event": "error", "data": {"message": message}}
        return

    def _compliance_fix(self, cite_state, final):
        if cite_state == "hit" and CITATION_MARK not in final:
            return (f"【系统校验】回答缺少{CITATION_MARK}...】来源标注，"
                    "请仅基于上述检索结果重写，并在每条事实句末标注来源。")
        if cite_state == "miss" and "无法提供确切解答" not in final:
            return (f"【系统校验】检索未命中，请直接回复：{DEGRADE_ANSWER}"
                    " 不要用通用知识编造。")
        return None

    def _run_tool(self, tc):
        raw = tc.function.arguments
        args = {} if not raw else None
        if args is None:
            try:
                args = json.loads(raw)
            except json.JSONDecodeError as e:
                return f"参数解析失败: {e}"
        if tc.function.name.startswith("danger_"):
            return "工具执行失败: PermissionDenied: 该工具为高危操作，已触发安全拦截"
        try:
            return str(self.registry.call(tc.function.name, args))
        except Exception as e:
            return f"工具执行失败: {type(e).__name__}: {e}"


def _safe_args(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}

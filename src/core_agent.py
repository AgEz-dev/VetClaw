"""ReAct 调度核心：手写 Thought-Action-Observation 循环（零 LangChain）。"""
import json

DEFAULT_SYSTEM_PROMPT = (
    "你是一个可以使用工具的助手。需要外部信息时调用工具；"
    "工具报错时根据报错修正参数或换工具；得到足够信息后直接回答。"
)


class ReActAgent:
    def __init__(self, client, registry, model="gpt-4o-mini",
                 max_steps=5, system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.client = client
        self.registry = registry
        self.model = model
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def run(self, prompt, history=None):
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.registry.schemas(),
            )
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return msg.content

            for tc in msg.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": self._run_tool(tc)})

        return f"已达到最大步数 {self.max_steps}，强制停止。"

    def _run_tool(self, tc):
        raw = tc.function.arguments
        # 空字符串 / None 兜底为无参数 {}
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
            # ★ 异常不中断循环，格式化为 tool 消息回填驱动模型反思
            return f"工具执行失败: {type(e).__name__}: {e}"
    
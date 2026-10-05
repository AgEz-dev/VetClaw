"""ReAct 调度核心：手写 Thought-Action-Observation 循环（零 LangChain）。"""
import json

CITATION_MARK = "【来源："
HIT_MARK = "【知识库检索结果】"
MISS_MARK = "【检索未命中】"
DEGRADE_ANSWER = "知识库中未检索到相关信息，无法提供确切解答，建议补充文档或联系维护者。"

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
        # cite_state 必须是 run 内局部变量，禁止挂 self，避免多次 run 之间状态污染。
        # 已知边界：技术问题若模型自行跳过检索（cite_state 始终 None），当前单 Agent
        # 无法拦截，属 Prompt 规划层问题；后续靠微调或独立 Router 分类解决。
        cite_state = None
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=self.registry.schemas())
            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content = self._run_tool(tc)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                    # hit 优先级最高，一旦命中不被后续其他工具或 miss 覆盖
                    if content.startswith(HIT_MARK):
                        cite_state = "hit"
                    elif content.startswith(MISS_MARK) and cite_state != "hit":
                        cite_state = "miss"
                continue

            final = msg.content or ""
            # 补正统一用 role="user" 并带【系统校验】前缀，不插入 system 消息（避免非标 API 400）
            if cite_state == "hit" and CITATION_MARK not in final:
                messages.append({"role": "user", "content":
                    f"【系统校验】回答缺少{CITATION_MARK}...】来源标注，"
                    "请仅基于上述检索结果重写，并在每条事实句末标注来源。"})
                continue
            if cite_state == "miss" and "无法提供确切解答" not in final:
                messages.append({"role": "user", "content":
                    f"【系统校验】检索未命中，请直接回复：{DEGRADE_ANSWER} 不要用通用知识编造。"})
                continue
            return final

        # 熔断体验：技术检索流程超时/补正失败 → 优雅降级；仅纯闲聊（None）才用步数提示
        return DEGRADE_ANSWER if cite_state in ("hit", "miss") \
            else f"已达到最大步数 {self.max_steps}，强制停止。"

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

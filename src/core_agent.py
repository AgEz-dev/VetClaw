"""Agent 调度器（不依赖 LangChain 等框架）。

核心模块：
- 01：用 inspect 把普通函数自动提取为 JSON Schema（Tool-Calling 反射）
- 02：ReAct while 状态机（Thought -> Action -> Observation），
     函数报错时 try-except 捕获并回填上下文，驱动模型自我反思重试
- 03：滑动窗口记忆，超过 Token 阈值自动弹出最早的非系统消息

当前状态：脚手架占位，待实现上述模块。
"""

# TODO(01): 实现 function_to_schema(func) -> dict
# TODO(02): 实现 ReAct 循环：解析 model 输出的 tool_calls 并执行
# TODO(03): 实现滑动窗口：estimate_tokens(msgs) + trim_oldest(msgs, budget)


def main():
    raise NotImplementedError("Agent 调度器尚未实现，从 TODO(01) 开始")


if __name__ == "__main__":
    main()

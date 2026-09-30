"""W1 战场：从零手搓 Agent 调度器（不依赖 LangChain 等框架）。

第一周目标（逐天推进，见 README 计划表）：
- Day 1-2：用 inspect 把普通函数自动提取为 JSON Schema（Tool-Calling 反射解析器）
- Day 3-4：手写 ReAct while 状态机（Thought -> Action -> Observation），
          函数报错时 try-except 捕获并回填上下文，驱动模型自我反思重试
- Day 5-7：手写滑动窗口记忆，超 Token 阈值自动弹出最早的非系统消息

当前状态：脚手架占位，等你 Day 1 开工填充。
"""

# TODO(Day 1): 实现 function_to_schema(func) -> dict
# TODO(Day 3): 实现 ReAct 循环：while 解析 model 输出的 tool_calls 并执行
# TODO(Day 5): 实现滑动窗口：estimate_tokens(msgs) + trim_oldest(msgs, budget)


def main():
    raise NotImplementedError("W1 还没开工，Day 1 从这里开始写")


if __name__ == "__main__":
    main()

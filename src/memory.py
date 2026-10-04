"""滑动窗口记忆：Token 预算控制与原子块裁剪（零第三方依赖）。"""
import re

_CJK = r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]"


def estimate_tokens(text) -> int:
    """启发式：汉字 2 token，其他字符每 4 个 1 token；+1 保证短消息非零。"""
    text = str(text or "")
    cjk = len(re.findall(_CJK, text))
    other = len(re.sub(_CJK + r"|\s", "", text))
    return cjk * 2 + other // 4 + 1


def message_tokens(msg: dict) -> int:
    total = estimate_tokens(msg.get("content"))
    for tc in msg.get("tool_calls", []):
        # ★ tool_calls 的函数名与 arguments 字符串在此计入预算
        total += estimate_tokens(tc["function"]["name"])
        total += estimate_tokens(tc["function"]["arguments"])
    return total


class SlidingWindowMemory:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt
        self.messages = []

    def add_message(self, role, content, **extra):
        self.messages.append({"role": role, "content": content, **extra})

    def add(self, message):
        self.messages.append(message)

    def _chunks(self):
        chunks, i = [], 0
        while i < len(self.messages):
            m, chunk = self.messages[i], [self.messages[i]]
            i += 1
            if m.get("role") == "assistant" and m.get("tool_calls"):
                while i < len(self.messages) and self.messages[i].get("role") == "tool":
                    chunk.append(self.messages[i])
                    i += 1
            chunks.append(chunk)
        return chunks

    def get_messages(self, max_tokens):
        head, budget = [], max_tokens
        if self.system_prompt:
            head = [{"role": "system", "content": self.system_prompt}]
            budget -= estimate_tokens(self.system_prompt)
            budget = max(0, budget)  # 防御：超长 system 把预算减成负数时归零
        kept, used = [], 0
        for chunk in reversed(self._chunks()):
            size = sum(message_tokens(m) for m in chunk)
            if used + size <= budget:
                kept.append(chunk)
                used += size
        return head + [m for chunk in reversed(kept) for m in chunk]

"""tests/test_memory.py：滑动窗口记忆断言测试（离线、零依赖）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory import SlidingWindowMemory, message_tokens


def tool_call(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


def test_within_budget_unchanged():
    mem = SlidingWindowMemory("你是助手")
    mem.add_message("user", "你好")
    mem.add_message("assistant", "你好，有什么可以帮你")
    out = mem.get_messages(1000)
    assert [m["role"] for m in out] == ["system", "user", "assistant"]


def test_system_protected():
    # 场景一：正常 system + 大量消息挤占
    mem = SlidingWindowMemory("你是助手")
    for i in range(20):
        mem.add_message("user", f"这是第{i}条较长消息内容占预算abc")
    out = mem.get_messages(50)
    assert out[0]["role"] == "system" and out[0]["content"] == "你是助手"
    # 场景二：超长 system + 极小预算，完整保留不截断
    long_sys = "系统指令" * 100
    mem2 = SlidingWindowMemory(long_sys)
    mem2.add_message("user", "hi")
    out2 = mem2.get_messages(5)
    assert out2[0]["role"] == "system" and out2[0]["content"] == long_sys


def test_atomic_tool_chunk():
    def build():
        m = SlidingWindowMemory("sys")
        m.add_message("user", "旧问题")
        m.add_message("assistant", None,
                      tool_calls=[tool_call("c1", "search", '{"q": "x"}')])
        m.add_message("tool", "搜索结果内容", tool_call_id="c1")
        m.add_message("user", "新问题")
        return m

    out = build().get_messages(15)   # 只够 system + 两个 user，工具块整体丢弃
    roles = [m["role"] for m in out]
    assert "tool" not in roles and not any(m.get("tool_calls") for m in out)
    out = build().get_messages(1000)  # 预算充足，整块保留
    assert any(m.get("tool_calls") for m in out)
    assert any(m["role"] == "tool" and m["tool_call_id"] == "c1" for m in out)


def test_budget_total_respected():
    mem = SlidingWindowMemory(None)
    for i in range(30):
        mem.add_message("user", f"第{i}条：" + "内容" * 20)
    out = mem.get_messages(200)
    assert sum(message_tokens(m) for m in out) <= 200


def test_dangling_tool_calls():
    # assistant 发起 tool_calls 后异常中断，没有 tool 消息跟随
    mem = SlidingWindowMemory("sys")
    mem.add_message("user", "q1")
    mem.add_message("assistant", None, tool_calls=[tool_call("c1", "search", "{}")])
    mem.add_message("user", "q2")
    out = mem.get_messages(1000)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    out2 = mem.get_messages(20)  # 残缺块整体丢弃，不崩
    assert out2[0]["role"] == "system"


def test_custom_super_long_user_message_pruned():
    # 单条用户消息本身超过预算时，整块丢弃，只留 system（不做半截截断）
    mem = SlidingWindowMemory(system_prompt="系统设定")
    mem.add_message("user", "超长输入内容" * 100)
    result = mem.get_messages(max_tokens=20)
    assert len(result) == 1
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "系统设定"


if __name__ == "__main__":
    test_within_budget_unchanged()
    test_system_protected()
    test_atomic_tool_chunk()
    test_budget_total_respected()
    test_dangling_tool_calls()
    test_custom_super_long_user_message_pruned()
    print("全部断言通过")

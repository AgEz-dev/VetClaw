"""tests/test_graph_agent.py：LangGraph 引擎对照测试（阶段四）。

**唯一裁判 = 事件序列 diff 为空**：同一 ScriptedClient 脚本喂给两个引擎，
`list(run_stream(...))` 必须逐事件、逐字段相等（含 `run()` 返回值）。

覆盖路径：守卫直通 / 纯文本 / 单工具 / 并行工具 / 工具异常回填 /
补正重写 / 命中降级 / 步数熔断 / 超时 / SDK 异常 / history 顺序。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core_agent import DEGRADE_ANSWER, ReActAgent  # noqa: E402
from graph_agent import build_graph_agent  # noqa: E402
from tools import ToolRegistry  # noqa: E402

# --------------------------------------------------------------------------- shims


def text_msg(content):
    return SimpleNamespace(role="assistant", content=content, tool_calls=None)


def tool_msg(*calls):
    return SimpleNamespace(role="assistant", content=None, tool_calls=[
        SimpleNamespace(id=cid, type="function",
                        function=SimpleNamespace(name=name, arguments=args))
        for cid, name, args in calls])


def _tool_chunk(index, tc):
    function = SimpleNamespace(name=tc.function.name, arguments=tc.function.arguments)
    piece = SimpleNamespace(index=index, id=tc.id, type="function", function=function)
    delta = SimpleNamespace(content=None, tool_calls=[piece])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _text_chunk(content):
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def stream_chunks(message):
    """把一条完整 assistant 消息转成流式 delta chunks。"""
    if message.tool_calls:
        return [_tool_chunk(i, tc) for i, tc in enumerate(message.tool_calls)]
    return [_text_chunk(message.content)]


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class FakeCompletions:
    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        self.client.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        message = self.client.responses[len(self.client.calls) - 1]
        if kwargs.get("stream"):
            return FakeStream(stream_chunks(message))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls = responses, []
        self.chat = SimpleNamespace(completions=FakeCompletions(self))


class FakeClock:
    """第一次返回 0，之后每次 +999 → 总时长守卫瞬间触发。"""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 999.0
        return self.now


class BoomClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        raise RuntimeError("boom from sdk")


def make_guard(**overrides):
    """确定性 FastPathGuard（rules_dict 直构，pipeline=None → 不查向量库）。"""
    from fastpath import FastPathGuard

    rules = {
        "consult_pattern": "咨询",
        "action_regex": "(误食|吃了|舔了|咬了|吞了|啃了)",
        "p0_toxins": [{"name": "百合", "keywords": ["百合"],
                       "species": ["猫", "狗"], "chunk_id": "knowledge/x.md#1"}],
        "dosage_words": ["剂量", "多少毫克"],
        "drug_whitelist": ["多西环素"],
        "unlisted_drugs_watchlist": ["头孢"],
        "sop_chunk_id": "knowledge/sop.md#1",
    }
    rules.update(overrides)
    return FastPathGuard(rules_dict=rules, pipeline=None)


def text_answer(content):
    return text_msg(content)


def make_registry():
    reg = ToolRegistry()

    @reg.tool
    def query_doc(keyword: str) -> str:
        """查询文档。"""
        return f"doc about {keyword}"

    @reg.tool
    def boom(x: str) -> str:
        """总是抛异常。"""
        raise ValueError("炸了")

    @reg.tool
    def search_knowledge_base(query: str) -> str:
        """检索知识库（离线桩：恒命中）。"""
        return "【知识库检索结果】\n[1] knowledge/deploy.md#1\ndocker compose up -d"

    @reg.tool
    def miss_search(query: str) -> str:
        """检索知识库（离线桩：恒未命中）。"""
        return "【检索未命中】知识库中没有相关内容，请进入风险排查引导。"

    @reg.tool
    def ping() -> str:
        """ping。"""
        return "pong"

    return reg


def events_of(agent, prompt, history=None):
    return list(agent.run_stream(prompt, history))


def both(script, prompt, history=None, *, registry=None, guard=None,
         max_steps=5, clock=None, total_timeout=30.0):
    """同一脚本跑两引擎，返回 (react_events, graph_events, react_answer, graph_answer)。

    传入 clock 时**每个引擎各持一份副本**，避免两引擎共享同一假时钟互相推进。
    """
    reg = registry or make_registry()
    import copy

    def _clock():
        return copy.deepcopy(clock) if clock is not None else None

    # 每次调用都用**全新** engine + FakeClient（脚本单次消费）——
    # 否则 events 与 run 会争抢同一份 responses 而误报 IndexError。
    def mk_react():
        return ReActAgent(FakeClient(script), reg, model="fake", max_steps=max_steps,
                          fastpath_guard=guard, clock=_clock(),
                          total_timeout=total_timeout)

    def mk_graph():
        return build_graph_agent(FakeClient(script), reg, None, guard, model="fake",
                                 max_steps=max_steps, clock=_clock(),
                                 total_timeout=total_timeout)

    return (events_of(mk_react(), prompt, history), events_of(mk_graph(), prompt, history),
            mk_react().run(prompt, history), mk_graph().run(prompt, history))


def assert_parity(script, prompt, history=None, **kw):
    react_ev, graph_ev, react_ans, graph_ans = both(script, prompt, history, **kw)
    assert react_ev == graph_ev, (
        f"\nREACT: {react_ev}\nGRAPH: {graph_ev}\n引擎事件序列不一致")
    assert react_ans == graph_ans, (react_ans, graph_ans)
    return react_ev


# --------------------------------------------------------------------------- 对照用例


def test_direct_answer_parity():
    ev = assert_parity([text_answer("直接答案")], "你好")
    assert [e["event"] for e in ev] == ["thought", "token", "done"]


def test_single_tool_then_answer_parity():
    ev = assert_parity(
        [tool_msg(("call_1", "query_doc", '{"keyword": "timeout"}')),
         text_answer("最终答案")], "查一下")
    assert [e["event"] for e in ev] == [
        "thought", "tool_call", "thought", "tool_result", "thought", "token", "done"]
    assert ev[1]["data"]["arguments"] == {"keyword": "timeout"}


def test_parallel_tool_calls_parity():
    ev = assert_parity(
        [tool_msg(("id_a", "query_doc", '{"keyword": "x"}'),
                  ("id_b", "ping", "")),
         text_answer("汇总答案")], "并行")
    ids = [e["data"]["id"] for e in ev if e["event"] == "tool_call"]
    assert ids == ["id_a", "id_b"]


def test_tool_exception_backfill_parity():
    ev = assert_parity(
        [tool_msg(("c1", "boom", '{"x": "a"}')), text_answer("修复后答案")], "go")
    result = next(e for e in ev if e["event"] == "tool_result")
    assert "ValueError" in result["data"]["content"] and "炸了" in result["data"]["content"]


def test_max_steps_fuse_parity():
    ev = assert_parity(
        [tool_msg(("c1", "ping", "")), tool_msg(("c2", "ping", "")),
         tool_msg(("c3", "ping", ""))], "loop", max_steps=3)
    assert ev[-1]["event"] == "error"
    assert ev[-1]["data"]["code"] == "MAX_STEPS"
    assert "步数" in ev[-1]["data"]["message"] and "3" in ev[-1]["data"]["message"]


def test_hit_missing_citation_rewrite_parity():
    ev = assert_parity(
        [tool_msg(("c1", "search_knowledge_base", '{"query": "启动"}')),
         text_answer("执行 docker compose up -d 就行。"),
         text_answer("执行 docker compose up -d【来源：knowledge/deploy.md#1】")],
        "启动")
    assert ev[-1]["event"] == "done"
    assert "【来源：knowledge/deploy.md#1】" in ev[-1]["data"]["answer"]
    thoughts = [e for e in ev if e["event"] == "thought"
                and e["data"]["stage"] == "thinking"]
    assert len(thoughts) == 3  # 初始 + 工具后 + 补正后


def test_miss_then_degrade_answer_parity():
    ev = assert_parity(
        [tool_msg(("c1", "miss_search", '{"query": "配额"}')),
         text_answer("你可以把配额设成 100。"), text_answer(DEGRADE_ANSWER)], "配额")
    # 未命中 → 首轮硬编触发补正 → 次轮给出 DEGRADE_ANSWER → 合规 → done
    assert ev[-1]["event"] == "done"
    assert ev[-1]["data"]["answer"] == DEGRADE_ANSWER
    assert "100" not in ev[-1]["data"]["answer"]


def test_hit_exhausted_degrades_parity():
    script = [tool_msg(("c1", "search_knowledge_base", '{"query": "启动"}'))] \
        + [text_answer("docker compose up 就行")] * 4
    ev = assert_parity(script, "启动", max_steps=5)
    assert ev[-1]["event"] == "error"
    assert ev[-1]["data"]["code"] == "DEGRADED"
    assert ev[-1]["data"]["message"] == DEGRADE_ANSWER


def test_guard_emergency_parity():
    ev = assert_parity([], "猫误食了百合花怎么办", guard=make_guard())
    assert [e["event"] for e in ev] == ["done"]
    assert "P0 急症预警" in ev[-1]["data"]["answer"]


def test_guard_refuse_parity():
    ev = assert_parity([], "头孢给狗吃多少毫克", guard=make_guard())
    assert [e["event"] for e in ev] == ["done"]
    assert "分诊引导" in ev[-1]["data"]["answer"]


def test_guard_pass_goes_to_llm_parity():
    # 白名单词短路 → 不进拒答 → 正常走 LLM
    ev = assert_parity([text_answer("ok")], "多西环素给狗吃多少毫克", guard=make_guard())
    assert [e["event"] for e in ev] == ["thought", "token", "done"]


def test_history_order_parity():
    history = [{"role": "user", "content": "旧问题"},
               {"role": "assistant", "content": "旧回答"}]
    assert_parity([text_answer("ok")], "新问题", history=history)


def test_total_timeout_parity():
    ev = assert_parity([text_answer("你好")], "你好", clock=FakeClock(), total_timeout=1.0)
    assert [e["event"] for e in ev] == ["thought", "error"]
    assert ev[-1]["data"]["code"] == "MODEL_TIMEOUT"


def test_sdk_exception_turned_into_error_parity():
    reg = make_registry()
    react = ReActAgent(BoomClient(), reg, model="fake")
    graph = build_graph_agent(BoomClient(), reg, None, None, model="fake")
    react_ev, graph_ev = events_of(react, "你好"), events_of(graph, "你好")
    assert react_ev == graph_ev
    assert graph_ev[-1]["event"] == "error"
    assert graph_ev[-1]["data"]["code"] == "INTERNAL_ERROR"
    assert "boom" not in graph_ev[-1]["data"]["message"]


# --------------------------------------------------------------------------- 装配契约


def test_build_graph_agent_interface_equivalent():
    """S4.0 调用点契约：返回对象必须提供 run / run_stream。"""
    # 两次调用各消费一条脚本 → 断言图状态不跨次污染
    agent = build_graph_agent(FakeClient([text_answer("hi"), text_answer("hi")]),
                              make_registry(), None, None, model="fake")
    assert callable(agent.run) and callable(agent.run_stream)
    assert agent.run("你好") == "hi"
    assert agent.run("你好") == "hi"

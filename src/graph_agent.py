"""LangGraph 状态机引擎：与 ReActAgent **接口等价、事件序列逐字节一致** 的对照实现（阶段四）。

设计约束（见 `阶段四细化计划.md` §三）：
- 6 种 SSE 事件契约**原样保留**——本模块换的是调度实现，不是契约；
- 事件一律经 `get_stream_writer()` 写入 custom 流，对外由 `GraphAgent` 薄适配器**透传**；
- `done` / `error` 终态由 `finalize` / `degrade` 节点产出。

🔴 冷启动纪律（见 `分支记录-S4.0`）：
- `import langgraph` 冷启动 ≈859ms → **本模块顶层不得出现 langgraph import**；
- 全部 langgraph 符号在 `build_graph_agent()` 函数体内（含 `_writer()`）**懒加载**；
- `services/agent_service.py` 只在 `VETCLAW_ENGINE=langgraph` 分支里 `from graph_agent import ...`。

节点契约：每个节点返回**状态增量**（dict），并通过 `state["route"]` 显式声明下一跳。
⚠️ 共享纯函数（`compliance_fix` / `execute_tool` / `guard`）**绝不碰 writer**——图外调用
`get_stream_writer()` 会抛 `RuntimeError`（阶段四实测坑 A）。
"""
import logging
from functools import partial
from types import SimpleNamespace
from typing import TypedDict

from core_agent import (
    CITATION_RE,
    DEFAULT_SYSTEM_PROMPT,
    DEGRADE_ANSWER,
    HIT_MARK,
    MISS_MARK,
    _safe_args,
    compliance_fix,
    execute_tool,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """图状态：字段与 `ReActAgent._run_events` 的局部变量一一对应（计划 §3.1）。"""
    prompt: str
    history: list
    messages: list            # 累积对话消息
    cite_state: str | None    # None | "hit" | "miss"
    step: int                 # 已执行的 llm 轮数
    start: float              # 时钟起点（注入 clock，便于测试）
    tool_parts: dict          # 本轮 tool_calls 分片累积（每轮在 llm 节点重置）
    text_parts: list          # 本轮纯文本分片（每轮重置）
    answer: str
    citations: list
    error: dict | None
    route: str                # 条件边路由键（节点显式声明）


class _Ctx:
    """节点闭包共享上下文（client / registry / guard / 超时与步数上限）。"""

    def __init__(self, client, registry, guard, model, max_steps,
                 system_prompt, clock, total_timeout):
        self.client = client
        self.registry = registry
        self.guard = guard
        self.model = model
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.clock = clock
        self.total_timeout = total_timeout


def _writer():
    """取当前运行的 custom 流写入器。

    ⚠️ 必须 **函数体内** import langgraph —— 顶层 import 会毁掉 P0「3.2ms / 冷启动」口径
    （AST 静态检查 `test_no_module_level_langgraph_import_in_src` 会拦下）。
    """
    from langgraph.config import get_stream_writer

    return get_stream_writer()


# --------------------------------------------------------------------------- nodes

def guard_node(state, ctx):
    """Fast-Path 前置拦截（不调 LLM）：emergency / refuse 直接给答案并跳 finalize。"""
    start = ctx.clock()
    if ctx.guard is not None:
        hit = ctx.guard.check(state["prompt"])
        if hit["action"] == "emergency":
            return {"answer": ctx.guard.emergency_message(
                        hit["toxin"], hit["species"], hit.get("chunk_id")),
                    "citations": [], "start": start, "route": "finalize"}
        if hit["action"] == "refuse":
            return {"answer": ctx.guard.refuse_message(hit["drug_hint"]),
                    "citations": [], "start": start, "route": "finalize"}

    messages = [{"role": "system", "content": ctx.system_prompt}]
    if state.get("history"):
        messages.extend(state["history"])
    messages.append({"role": "user", "content": state["prompt"]})
    return {"messages": messages, "start": start, "step": 0, "route": "llm"}


def llm_node(state, ctx):
    """一轮 LLM 流式调用：先发 thought(thinking)，逐 token 写流，累积 tool_calls。"""
    writer = _writer()
    # ⚠️ 顺序与 ReActAgent 一致：thought(thinking) 在超时检查**之前**发出
    writer({"event": "thought", "data": {"stage": "thinking"}})
    start = state["start"]
    timeout_err = {"code": "MODEL_TIMEOUT", "message": "模型响应超时，请稍后重试"}

    if ctx.clock() - start > ctx.total_timeout:
        return {"error": timeout_err, "tool_parts": {}, "route": "finalize"}

    tool_parts, text_parts = {}, []
    stream = None
    try:
        stream = ctx.client.chat.completions.create(
            model=ctx.model, messages=state["messages"],
            tools=ctx.registry.schemas(), stream=True)
        for chunk in stream:
            if ctx.clock() - start > ctx.total_timeout:
                return {"error": timeout_err, "tool_parts": {}, "route": "finalize"}
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
                writer({"event": "token", "data": {"text": piece}})
    except Exception as e:
        # SDK 层（httpx read timeout / 连接错误）兜底转 error 事件，不让 traceback 冒到适配器
        logger.exception("model stream error")
        code = "MODEL_TIMEOUT" if "timeout" in type(e).__name__.lower() else "INTERNAL_ERROR"
        return {"error": {"code": code, "message": "模型服务暂时不可用，请稍后重试"},
                "tool_parts": {}, "route": "finalize"}
    finally:
        if stream is not None and hasattr(stream, "close"):
            try:
                stream.close()
            except Exception:
                pass

    route = "tools" if tool_parts else "compliance"
    return {"tool_parts": tool_parts, "text_parts": text_parts,
            "step": state.get("step", 0) + 1, "route": route}


def tools_node(state, ctx):
    """执行本轮全部 tool_calls：逐条发 tool_call / thought(executing_tool) / tool_result，
    并维护 cite_state 三态机（hit 优先级最高，不被后续 miss 覆盖）。"""
    writer = _writer()
    calls = [{"id": v["id"], "type": "function",
              "function": {"name": v["name"], "arguments": v["arguments"]}}
             for _, v in sorted(state["tool_parts"].items())]
    messages = list(state["messages"])
    messages.append({"role": "assistant", "content": None, "tool_calls": calls})
    cite_state = state.get("cite_state")

    for c in calls:
        writer({"event": "tool_call", "data": {"id": c["id"],
               "name": c["function"]["name"],
               "arguments": _safe_args(c["function"]["arguments"])}})
        writer({"event": "thought", "data": {"stage": "executing_tool"}})
        tc = SimpleNamespace(id=c["id"], function=SimpleNamespace(
            name=c["function"]["name"], arguments=c["function"]["arguments"]))
        content = execute_tool(ctx.registry, tc)
        messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
        writer({"event": "tool_result", "data": {"id": c["id"], "content": content}})
        if content.startswith(HIT_MARK):
            cite_state = "hit"  # hit 优先级最高，不被后续工具/miss 覆盖
        elif content.startswith(MISS_MARK) and cite_state != "hit":
            cite_state = "miss"

    # ⚠️ ReActAgent 在每次迭代末尾**无条件**发 thought(thinking)（第 154 行）后才 continue；
    # 若这是最后一次迭代（步数耗尽），该事件先于 degrade 出现。走 llm 时由 llm_node 补发，
    # 故此处只在熔断分支显式补上，避免重复。
    if state.get("step", 0) >= ctx.max_steps:
        writer({"event": "thought", "data": {"stage": "thinking"}})
        route = "degrade"
    else:
        route = "llm"
    return {"messages": messages, "cite_state": cite_state, "route": route}


def compliance_node(state, ctx):
    """合规校验：需要补正则回 llm 重写，否则定稿（answer + citations）。"""
    final = "".join(state.get("text_parts") or [])
    fix = compliance_fix(state.get("cite_state"), final)
    if fix:
        messages = list(state["messages"])
        messages.append({"role": "user", "content": fix})
        # ⚠️ 补正重写同样受步数上限约束：ReActAgent 的 `for _ in range(max_steps)` 把
        # 「工具循环」与「补正循环」**共用同一个预算**，二者都必须越界即熔断，否则
        # LangGraph 侧会多跑一轮 LLM（本次对照测试抓到的真实偏差）。
        # 同 tools_node：ReActAgent 第 161 行在 continue 前无条件补发 thought(thinking)，
        # 熔断分支必须显式补上（走 llm 时由 llm_node 发出）。
        if state.get("step", 0) >= ctx.max_steps:
            _writer()({"event": "thought", "data": {"stage": "thinking"}})
            route = "degrade"
        else:
            route = "llm"
        return {"messages": messages, "route": route}
    return {"answer": final, "citations": CITATION_RE.findall(final), "route": "finalize"}


def finalize_node(state, ctx):
    """终态：error 优先，否则 done。"""
    writer = _writer()
    if state.get("error"):
        writer({"event": "error", "data": state["error"]})
    else:
        writer({"event": "done", "data": {"answer": state.get("answer", ""),
                                          "citations": state.get("citations") or []}})
    return {"route": "__end__"}


def degrade_node(state, ctx):
    """循环上限熔断：命中过检索则给分诊引导（DEGRADED），否则报步数（MAX_STEPS）。"""
    writer = _writer()
    if state.get("cite_state") in ("hit", "miss"):
        code, message = "DEGRADED", DEGRADE_ANSWER
    else:
        code, message = "MAX_STEPS", f"已达到最大步数 {ctx.max_steps}，强制停止。"
    writer({"event": "error", "data": {"code": code, "message": message}})
    return {"route": "__end__"}


# --------------------------------------------------------------------------- graph

def build_graph(client, registry, guard, *, model, max_steps, system_prompt,
                clock, total_timeout):
    """编译状态图。langgraph 符号在本函数体内懒加载（冷启动纪律）。"""
    from langgraph.graph import END, START, StateGraph

    ctx = _Ctx(client, registry, guard, model, max_steps, system_prompt,
               clock, total_timeout)
    graph = StateGraph(AgentState)
    graph.add_node("guard", partial(guard_node, ctx=ctx))
    graph.add_node("llm", partial(llm_node, ctx=ctx))
    graph.add_node("tools", partial(tools_node, ctx=ctx))
    graph.add_node("compliance", partial(compliance_node, ctx=ctx))
    graph.add_node("finalize", partial(finalize_node, ctx=ctx))
    graph.add_node("degrade", partial(degrade_node, ctx=ctx))

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", lambda s: s["route"],
                                {"llm": "llm", "finalize": "finalize"})
    graph.add_conditional_edges("llm", lambda s: s["route"],
                                {"tools": "tools", "compliance": "compliance",
                                 "finalize": "finalize"})
    graph.add_conditional_edges("tools", lambda s: s["route"],
                                {"llm": "llm", "degrade": "degrade"})
    graph.add_conditional_edges("compliance", lambda s: s["route"],
                                {"llm": "llm", "finalize": "finalize",
                                 "degrade": "degrade"})
    graph.add_edge("finalize", END)
    graph.add_edge("degrade", END)
    return graph.compile()


class GraphAgent:
    """与 `ReActAgent` 接口等价（`run` / `run_stream`），供装配层无感替换。

    适配器只消费 `stream_mode="custom"` 并**原样透传** writer 载荷——因此事件序列天然
    与手写引擎一致；`updates` 一律忽略（阶段四实测：custom 事件按序先于同节点 updates）。
    """

    def __init__(self, graph):
        self._graph = graph

    def run(self, prompt, history=None):
        # 与 ReActAgent.run 语义一致：done 取 answer、error 取 message。
        for ev in self.run_stream(prompt, history):
            if ev["event"] == "done":
                return ev["data"]["answer"]
            if ev["event"] == "error":
                return ev["data"]["message"]
        return ""

    def run_stream(self, prompt, history=None):
        init = {"prompt": prompt, "history": list(history) if history else []}
        try:
            yield from self._graph.stream(init, stream_mode="custom")
        except Exception:
            # 阶段四实测坑 C：节点内异常会冒到调用方 → 适配器兜底转 error 事件
            logger.exception("graph stream error")
            yield {"event": "error", "data": {"code": "INTERNAL_ERROR",
                                              "message": "服务内部错误，请稍后重试"}}


def build_graph_agent(client, registry, pipeline, guard, *, model="gpt-4o-mini",
                      total_timeout=30.0, max_steps=5,
                      system_prompt=DEFAULT_SYSTEM_PROMPT, clock=None):
    """装配层入口（`services/agent_service.py` 的 langgraph 分支调用）。

    `pipeline` 在此仅作接口占位——Fast-Path 需要的 `RAGPipeline` 已注入 `guard`，
    调度层本身不直接检索（与 `ReActAgent` 一致）。
    """
    import time

    graph = build_graph(client, registry, guard, model=model, max_steps=max_steps,
                        system_prompt=system_prompt,
                        clock=clock or time.monotonic, total_timeout=total_timeout)
    return GraphAgent(graph)

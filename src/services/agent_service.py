"""Agent 装配与单例管理：生产环境懒加载，测试可注入替身。"""
from tools import registry
import knowledge_tool  # noqa: F401  导入即注册 search_knowledge_base
from core_agent import ReActAgent

_agent = None


def build_agent():
    from openai import OpenAI
    from core.config import settings
    from fastpath import FastPathGuard
    from rag_pipeline import RAGPipeline, BGEEmbeddingFunction
    from knowledge_tool import rag_provider
    kwargs = {"timeout": settings.timeout}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    pipeline = RAGPipeline(embedding_function=BGEEmbeddingFunction())
    rag_provider.set(pipeline)
    guard = FastPathGuard(pipeline=pipeline)

    # ⚠️ langgraph 只允许在**本分支内** import：实测 `import langgraph` 冷启动 ≈859ms，
    # 顶层 import 会直接毁掉 P0「3.2ms / 冷启动」口径（见 HANDOVER §九）。
    if settings.engine == "langgraph":
        try:
            from graph_agent import build_graph_agent
        except ImportError as exc:
            raise RuntimeError(
                "VETCLAW_ENGINE=langgraph，但 LangGraph 引擎不可用。"
                "请确认已安装可选依赖：pip install -e '.[graph]'"
            ) from exc
        return build_graph_agent(
            OpenAI(**kwargs), registry, pipeline, guard,
            model=settings.model, total_timeout=settings.total_timeout,
            max_steps=settings.max_steps,
        )

    return ReActAgent(OpenAI(**kwargs), registry, model=settings.model,
                      total_timeout=settings.total_timeout,
                      max_steps=settings.max_steps,
                      fastpath_guard=guard)


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def set_agent(agent):
    global _agent
    _agent = agent

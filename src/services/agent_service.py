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
    return ReActAgent(OpenAI(**kwargs), registry, model=settings.model,
                      total_timeout=settings.total_timeout,
                      fastpath_guard=guard)


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def set_agent(agent):
    global _agent
    _agent = agent

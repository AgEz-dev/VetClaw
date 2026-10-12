"""预问诊装配：复用既有守卫，懒建 LLM 调用。

两个刻意的取舍：

1. **守卫复用**：走 ``agent_service.get_guard()``，**不重复构建 RAGPipeline** ——
   否则 BGE 模型与 Chroma 会被加载两份，内存直接翻倍（本项目跑在小机器上，这点很重要）。
2. **LLM 客户端懒建**：``OpenAI`` 本身只是 HTTP 客户端，开销可忽略；但仍做成懒加载，
   让"未配置密钥"的环境**不因为 import 就报错**（预问诊在纯规则模式下应仍可用）。

``set_llm(None)`` 可显式关闭模型通道（测试与离线验证用）。
"""
from __future__ import annotations

from typing import Callable

_llm: Callable[[list[dict]], str] | None = None
_llm_ready = False


def build_llm() -> Callable[[list[dict]], str]:
    from openai import OpenAI

    from core.config import settings

    kwargs = {"timeout": settings.timeout}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    client = OpenAI(**kwargs)

    def call(messages: list[dict]) -> str:
        resp = client.chat.completions.create(model=settings.model, messages=messages,
                                              temperature=0,
                                              max_tokens=settings.max_tokens)
        return resp.choices[0].message.content or ""

    return call


def get_llm() -> Callable[[list[dict]], str] | None:
    """返回 LLM 调用；**不可用时返回 ``None``**（调用方据此退化为纯规则模式，而不是崩掉）。"""
    global _llm, _llm_ready
    if _llm_ready:
        return _llm
    try:
        _llm = build_llm()
    except Exception:          # 缺密钥 / SDK 异常 → 纯规则模式
        _llm = None
    _llm_ready = True
    return _llm


def set_llm(fn: Callable[[list[dict]], str] | None) -> None:
    """测试注入（传 ``None`` 即关闭模型通道）。"""
    global _llm, _llm_ready
    _llm = fn
    _llm_ready = True


def get_guard():
    """复用 agent 的守卫（同一个 RAGPipeline，不重复加载模型）。不可用时返回 None。"""
    try:
        from services.agent_service import get_guard as _get

        return _get()
    except Exception:
        return None


def build_archive(intake_archive: dict | None) -> dict:
    """把业务系统给的档案槽位规整为引擎要的形状（只保留已知键，缺失即不填）。"""
    if not intake_archive:
        return {}
    allowed = ("vaccine", "past_disease", "medications", "allergy")
    return {k: str(intake_archive[k]).strip()
            for k in allowed
            if intake_archive.get(k) and str(intake_archive[k]).strip()}

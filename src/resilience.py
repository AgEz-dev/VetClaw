"""LLM 调用的韧性包装：指数退避重试 + 轻量熔断（维度 7 容灾与韧性）。

两个引擎（core_agent / graph_agent）共用本模块的 `call_llm`，避免行为漂移——
这点与 `compliance_fix` / `execute_tool` 同源：任何跨引擎共享逻辑都只留一份。

边界说明：
- 退避重试只覆盖「建立流」这一步（`create`）。**流中途**失败不重试——
  已推送的 token 无法回滚，重试会造成重复输出；这类错误仍由调用方转 error 事件。
- 熔断只对**瞬时错误**（超时/连接/限流/5xx）计数；业务性错误（如参数错误）不计入，
  否则会把"用户问错"误判成"服务挂了"。
- ⚠️ 重试会叠加时延（最坏 3 次尝试）。Agent 层 `total_timeout` 只在循环边界检查，
  不覆盖此处；连续超时的场景由熔断器快速接管（打开后零真实请求）。
"""
from __future__ import annotations

import logging
import threading
import time

from tenacity import (retry, retry_if_exception, stop_after_attempt,
                      wait_exponential_jitter)

logger = logging.getLogger(__name__)

FAIL_THRESHOLD = 5     # 连续失败阈值（达到即打开熔断）
OPEN_SECONDS = 30.0    # 打开后的快速失败时长
MAX_ATTEMPTS = 3       # 单次调用最多尝试次数
INITIAL_BACKOFF = 0.5  # 退避初值（秒）
MAX_BACKOFF = 4.0      # 退避上限（秒）

_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}


class CircuitOpenError(RuntimeError):
    """熔断打开期间的快速失败信号：调用方据此降级，且不发出真实请求。"""


class _CircuitBreaker:
    """进程级、线程安全的轻量熔断器（closed → open → 半开探测）。"""

    def __init__(self, threshold: int = FAIL_THRESHOLD,
                 open_seconds: float = OPEN_SECONDS):
        self.threshold = threshold
        self.open_seconds = open_seconds
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    def before(self) -> None:
        with self._lock:
            if self._opened_at:
                if time.monotonic() - self._opened_at < self.open_seconds:
                    raise CircuitOpenError("LLM 服务连续失败，熔断器打开，快速失败")
                # 打开时长已过 → 置为半开，放行本次调用做探测
                self._opened_at = 0.0

    def on_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0

    def on_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_at = time.monotonic()
                logger.warning("LLM 熔断器打开：连续失败 %d 次", self._failures)

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0


breaker = _CircuitBreaker()


def is_transient(exc: BaseException) -> bool:
    """判定是否为可重试的瞬时错误（超时 / 连接 / 限流 / 5xx）。"""
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name or "ratelimit" in name:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in _TRANSIENT_STATUS


@retry(stop=stop_after_attempt(MAX_ATTEMPTS),
       wait=wait_exponential_jitter(initial=INITIAL_BACKOFF, max=MAX_BACKOFF),
       retry=retry_if_exception(is_transient),
       reraise=True)
def _create_with_retry(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def call_llm(client, **kwargs):
    """带熔断 + 退避重试的 LLM 流式调用，返回 SDK 流对象。

    - 熔断打开 → 立即抛 `CircuitOpenError`（零真实请求）；
    - 瞬时错误 → 指数退避 + 抖动，最多 `MAX_ATTEMPTS` 次；
    - 重试耗尽 / 非瞬时错误 → 原样抛出，由调用方转 error 事件或降级。
    """
    breaker.before()
    try:
        stream = _create_with_retry(client, **kwargs)
    except Exception as exc:
        if is_transient(exc):
            breaker.on_failure()
        raise
    breaker.on_success()
    return stream

"""医院业务系统内部接口客户端（机机通信，共享密钥）。

为什么不用 JWT
--------------
VetClaw 是**服务**，不是医生账号的持有者。让它登录拿医生 token，等于把人工账号
交给一个外部进程；改由两侧各配一份共享密钥（``X-Internal-Secret``）更小、更可控。
医院侧对 ``/internal/ai/**`` **默认拒绝**：密钥为空时直接 503，不存在"忘配就裸奔"。

超时与重试口径（与计划稿一致）
------------------------------
- 读接口（档案/接种/用药）：3s，失败重试 **1 次**，指数退避（0.2s）；
- 写接口（转人工）：5s，**永不重试** —— 重试由医院侧的幂等键兜底，
  客户端盲目重发只会把"已成功但响应丢了"变成重复工单。

错误一律转成 :class:`HospitalClientError`（带稳定 ``error_code``），
由工具层包装为 ``{ok, data, error_code, elapsed_ms}`` 交给模型与轨迹。
"""
from __future__ import annotations

import time

import httpx

from core.config import settings

READ_TIMEOUT = 3.0
WRITE_TIMEOUT = 5.0
RETRY_BASE_DELAY = 0.2
HEADER = "X-Internal-Secret"


class HospitalClientError(Exception):
    """带稳定错误码的内部接口异常。"""

    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code


class HospitalClient:
    def __init__(self, base_url: str | None = None, secret: str | None = None,
                 read_timeout: float = READ_TIMEOUT, write_timeout: float = WRITE_TIMEOUT,
                 retry_base_delay: float = RETRY_BASE_DELAY,
                 transport: httpx.BaseTransport | None = None,
                 sleep=time.sleep):
        self.base_url = (base_url if base_url is not None
                         else settings.hospital_base_url).rstrip("/")
        self.secret = secret if secret is not None else settings.hospital_secret
        self.read_timeout = read_timeout
        self.write_timeout = write_timeout
        self.retry_base_delay = retry_base_delay
        self._sleep = sleep
        self._http = httpx.Client(base_url=self.base_url, transport=transport)

    # ------------------------------------------------------------------ 读

    def get_pet_profile(self, pet_id: int) -> dict:
        return self._request("GET", f"/internal/ai/pet/{pet_id}/profile",
                             timeout=self.read_timeout, retryable=True)

    def get_vaccinations(self, pet_id: int) -> list:
        return self._request("GET", f"/internal/ai/pet/{pet_id}/vaccinations",
                             timeout=self.read_timeout, retryable=True)

    def get_medications(self, pet_id: int) -> list:
        return self._request("GET", f"/internal/ai/pet/{pet_id}/medications",
                             timeout=self.read_timeout, retryable=True)

    def list_memories(self, pet_id: int, limit: int = 8) -> list:
        """L3 长期记忆（医院侧已按安全事实优先排序并过滤过期）。"""
        return self._request("GET", f"/internal/ai/pet/{pet_id}/memories",
                             timeout=self.read_timeout, retryable=True,
                             params={"limit": int(limit)})

    # ------------------------------------------------------------------ 写

    def add_memory(self, pet_id: int, fact_type: str, fact_value: str,
                   source: str = "OWNER_CONFIRMED", confidence: float | None = None,
                   expires_at: str | None = None, actor: str = "VETCLAW") -> dict:
        body = {"petId": pet_id, "factType": fact_type, "factValue": fact_value,
                "source": source, "confidence": confidence,
                "expiresAt": expires_at, "actor": actor}
        return self._request("POST", "/internal/ai/memory",
                             timeout=self.write_timeout, retryable=False, json_body=body)

    def delete_memory(self, memory_id: int, reason: str) -> dict:
        body = {"reason": reason, "actor": "VETCLAW"}
        return self._request("POST", f"/internal/ai/memory/{memory_id}/delete",
                             timeout=self.write_timeout, retryable=False, json_body=body)

    def escalate(self, pet_id: int, case_id: str, reason: str,
                 idempotency_key: str, source: str = "VETCLAW") -> dict:
        body = {"petId": pet_id, "caseId": case_id or None, "reason": reason,
                "source": source, "idempotencyKey": idempotency_key}
        return self._request("POST", "/internal/ai/escalation",
                             timeout=self.write_timeout, retryable=False, json_body=body)

    def export_feedback_jsonl(self) -> str:
        """拉取已标注反馈（JSONL 原文）。**导出即消费，故绝不重试**：
        第一次调用成功但响应丢失时，重发只会拿到空批次，而样本已被推进到
        EXPORTED —— 宁可让运维重放本地文件，也不静默丢样本。
        """
        if not self.secret:
            raise HospitalClientError(
                "HOSPITAL_NOT_CONFIGURED",
                "未配置 VETCLAW_HOSPITAL_SECRET，医院数据工具不可用")
        try:
            resp = self._http.post("/internal/ai/feedback/export",
                                   timeout=self.write_timeout,
                                   headers={HEADER: self.secret})
        except httpx.TimeoutException:
            raise HospitalClientError(
                "HOSPITAL_TIMEOUT", f"医院内部接口超时（{self.write_timeout:g}s）")
        except httpx.HTTPError as exc:
            raise HospitalClientError(
                "HOSPITAL_UNAVAILABLE", f"医院内部接口不可达: {type(exc).__name__}")
        if resp.status_code >= 500:
            raise HospitalClientError("HOSPITAL_UNAVAILABLE",
                                      f"医院内部接口 {resp.status_code}")
        if resp.status_code >= 400:
            raise self._http_error(resp)
        return resp.text

    # ------------------------------------------------------------------ 内核

    def _request(self, method: str, path: str, *, timeout: float,
                 retryable: bool, json_body: dict | None = None,
                 params: dict | None = None):
        if not self.secret:
            raise HospitalClientError(
                "HOSPITAL_NOT_CONFIGURED",
                "未配置 VETCLAW_HOSPITAL_SECRET，医院数据工具不可用")

        attempts = 2 if retryable else 1
        last_error: HospitalClientError | None = None
        for attempt in range(attempts):
            try:
                resp = self._http.request(
                    method, path, timeout=timeout, json=json_body, params=params,
                    headers={HEADER: self.secret})
            except httpx.TimeoutException:
                last_error = HospitalClientError(
                    "HOSPITAL_TIMEOUT", f"医院内部接口超时（{timeout:g}s）")
            except httpx.HTTPError as exc:
                last_error = HospitalClientError(
                    "HOSPITAL_UNAVAILABLE", f"医院内部接口不可达: {type(exc).__name__}")
            else:
                if resp.status_code >= 500:
                    last_error = HospitalClientError(
                        "HOSPITAL_UNAVAILABLE",
                        f"医院内部接口 {resp.status_code}")
                elif resp.status_code >= 400:
                    raise self._http_error(resp)
                else:
                    return self._unwrap(resp)

            if attempt < attempts - 1:
                self._sleep(self.retry_base_delay * (2 ** attempt))

        assert last_error is not None
        raise last_error

    @staticmethod
    def _http_error(resp: httpx.Response) -> HospitalClientError:
        code = {401: "HOSPITAL_AUTH_FAILED", 403: "HOSPITAL_AUTH_FAILED",
                404: "HOSPITAL_NOT_FOUND", 409: "HOSPITAL_CONFLICT"}.get(
                    resp.status_code, f"HOSPITAL_HTTP_{resp.status_code}")
        return HospitalClientError(code, _message_of(resp, fallback=f"HTTP {resp.status_code}"))

    @staticmethod
    def _unwrap(resp: httpx.Response):
        try:
            body = resp.json()
        except ValueError as exc:
            raise HospitalClientError(
                "HOSPITAL_BAD_RESPONSE", "医院内部接口返回非 JSON") from exc
        code = body.get("code") if isinstance(body, dict) else None
        if code not in (0, 200, None):
            raise HospitalClientError(
                f"HOSPITAL_{code}",
                str(body.get("message") or f"业务码 {code}"))
        return body.get("data") if isinstance(body, dict) else body

    def close(self) -> None:
        self._http.close()


def _message_of(resp: httpx.Response, fallback: str) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("message"):
            return str(body["message"])
    except ValueError:
        pass
    return fallback


class HospitalClientProvider:
    """懒装配单例；测试可注入 MockTransport 客户端。"""

    def __init__(self):
        self._client: HospitalClient | None = None

    def set(self, client: HospitalClient | None) -> None:
        self._client = client

    def get(self) -> HospitalClient:
        if self._client is None:
            self._client = HospitalClient()
        return self._client


hospital_provider = HospitalClientProvider()

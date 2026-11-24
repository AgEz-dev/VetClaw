"""tests/test_hospital_tools.py：B2 七工具、工具治理与医院内部接口客户端断言。

全部离线：httpx.MockTransport 冒充医院接口，不依赖 Java 服务、不依赖模型。
覆盖：
- 工具注册与参数白名单；
- 读/写超时口径、读重试一次、写不重试；
- 共享密钥缺失/错误、5xx、业务码错误的稳定 error_code；
- 转人工幂等键稳定；
- 药品说明书查询与单药安全核对复用同一份规则表。
"""
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import hospital_tools  # noqa: E402
import knowledge_tool  # noqa: E402,F401  与生产装配一致：检索工具 + 本批六工具 = 7
from case_context import reset_case_id, set_case_id  # noqa: E402
from hospital_client import HospitalClient, hospital_provider  # noqa: E402
from tools import ToolRegistry, registry  # noqa: E402


def _client(handler, secret="test-secret", **kwargs):
    return HospitalClient(base_url="http://hospital.test", secret=secret,
                          transport=httpx.MockTransport(handler), **kwargs)


def _ok(data):
    return httpx.Response(200, json={"code": 200, "message": "ok", "data": data})


def _body(text):
    return json.loads(text)


def _call_with(client, fn, *args, **kwargs):
    hospital_provider.set(client)
    try:
        return _body(fn(*args, **kwargs))
    finally:
        hospital_provider.set(None)


# --------------------------------------------------------------------- 注册与治理

def test_eight_tools_registered():
    names = {s["function"]["name"] for s in registry.schemas()}
    assert {"search_knowledge_base", "lookup_drug_label", "check_drug_safety",
            "get_pet_profile", "get_vaccination_history", "get_medication_history",
            "get_inventory_summary", "escalate_to_human"} <= names
    assert len(names) == 8


def test_escalation_requires_pet_id_and_reason():
    fn = next(s["function"] for s in registry.schemas()
              if s["function"]["name"] == "escalate_to_human")
    assert fn["parameters"]["required"] == ["pet_id", "reason"]


def test_argument_whitelist_rejects_extra_param():
    reg = ToolRegistry()

    @reg.tool
    def ping(name: str) -> str:
        """打招呼。"""
        return name

    assert reg.call("ping", {"name": "a"}) == "a"
    try:
        reg.call("ping", {"name": "a", "admin": "true"})
        raise AssertionError("多余参数必须被白名单拒绝")
    except ValueError as exc:
        assert "admin" in str(exc)


# --------------------------------------------------------------------- 读接口

def test_profile_success_and_secret_header():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["secret"] = request.headers.get("X-Internal-Secret")
        seen["timeout"] = request.extensions.get("timeout")
        return _ok({"petId": 7, "petName": "豆豆", "allergies": "青霉素"})

    out = _call_with(_client(handler), hospital_tools.get_pet_profile, 7)
    assert out["ok"] is True
    assert out["data"]["petName"] == "豆豆"
    assert out["error_code"] is None
    assert out["elapsed_ms"] >= 0
    assert seen["path"] == "/internal/ai/pet/7/profile"
    assert seen["secret"] == "test-secret"
    assert seen["timeout"]["read"] == 3.0, "读接口超时必须是 3s"


def test_read_retries_once_with_backoff():
    calls = {"n": 0}
    sleeps = []

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("read timeout", request=request)
        return _ok([{"vaccineName": "卫佳伍"}])

    client = _client(handler, sleep=sleeps.append)
    out = _call_with(client, hospital_tools.get_vaccination_history, 7)
    assert out["ok"] is True
    assert out["data"][0]["vaccineName"] == "卫佳伍"
    assert calls["n"] == 2, "读接口失败应重试一次"
    assert sleeps == [0.2], "退避基数必须是 0.2s"


def test_read_5xx_retried_then_fails():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, json={"code": 500, "message": "系统繁忙"})

    out = _call_with(_client(handler), hospital_tools.get_medication_history, 7)
    assert out["ok"] is False
    assert out["error_code"] == "HOSPITAL_UNAVAILABLE"
    assert calls["n"] == 2


def test_missing_secret_fails_closed_without_network():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _ok({})

    out = _call_with(_client(handler, secret=""), hospital_tools.get_pet_profile, 7)
    assert out["ok"] is False
    assert out["error_code"] == "HOSPITAL_NOT_CONFIGURED"
    assert calls["n"] == 0, "密钥为空时不允许发出任何请求"


def test_wrong_secret_maps_to_auth_failed():
    def handler(request):
        return httpx.Response(401, json={"code": 401, "message": "内部接口密钥无效"})

    out = _call_with(_client(handler), hospital_tools.get_pet_profile, 7)
    assert out["ok"] is False
    assert out["error_code"] == "HOSPITAL_AUTH_FAILED"
    assert "密钥无效" in out["message"]


def test_not_found_maps_to_hospital_not_found():
    def handler(request):
        return httpx.Response(404, json={"code": 404, "message": "宠物档案不存在"})

    out = _call_with(_client(handler), hospital_tools.get_pet_profile, 999)
    assert out["ok"] is False
    assert out["error_code"] == "HOSPITAL_NOT_FOUND"


# --------------------------------------------------------------------- 写接口

def test_write_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ReadTimeout("write timeout", request=request)

    out = _call_with(_client(handler), hospital_tools.escalate_to_human, 7, "需要医生复核")
    assert out["ok"] is False
    assert out["error_code"] == "HOSPITAL_TIMEOUT"
    assert calls["n"] == 1, "写接口不得重试"


def test_escalation_stable_idempotency_key_and_write_timeout():
    bodies = []

    def handler(request):
        assert request.extensions["timeout"]["read"] == 5.0, "写接口超时必须是 5s"
        bodies.append(json.loads(request.content))
        return _ok({"id": 1, "caseId": None, "petId": 7,
                    "status": "PENDING", "deduped": len(bodies) > 1})

    client = _client(handler)
    hospital_provider.set(client)
    try:
        first = _body(hospital_tools.escalate_to_human(7, "持续呕吐，需要医生介入"))
        second = _body(hospital_tools.escalate_to_human(7, "持续呕吐，需要医生介入"))
    finally:
        hospital_provider.set(None)

    assert first["ok"] and second["ok"]
    assert first["data"]["deduped"] is False
    assert second["data"]["deduped"] is True
    assert bodies[0]["idempotencyKey"] == bodies[1]["idempotencyKey"]
    assert len(bodies[0]["idempotencyKey"]) == 32
    assert bodies[0]["reason"] == "持续呕吐，需要医生介入"
    assert bodies[0]["source"] == "VETCLAW"


def test_escalation_carries_case_and_key_is_case_scoped():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _ok({"id": 1, "deduped": False})

    hospital_provider.set(_client(handler))
    try:
        token = set_case_id("chat-u7-p3")
        try:
            in_case = _body(hospital_tools.escalate_to_human(7, "需要医生复核"))
        finally:
            reset_case_id(token)
        no_case = _body(hospital_tools.escalate_to_human(7, "需要医生复核"))
    finally:
        hospital_provider.set(None)

    assert in_case["ok"] and no_case["ok"]
    assert bodies[0]["caseId"] == "chat-u7-p3", "工单必须关联当前病例"
    assert bodies[1]["caseId"] is None, "无 case 的调用不得编造关联"
    assert bodies[0]["idempotencyKey"] != bodies[1]["idempotencyKey"], \
        "不同 case 的同一原因必须是不同工单"


def test_escalation_requires_reason():
    out = _body(hospital_tools.escalate_to_human(7, "  "))
    assert out["ok"] is False
    assert out["error_code"] == "TOOL_ERROR"


def test_escalation_conflict_maps_409():
    def handler(request):
        return httpx.Response(409, json={"code": 409, "message": "同一幂等键提交了不同内容"})

    out = _call_with(_client(handler), hospital_tools.escalate_to_human, 7, "原因")
    assert out["ok"] is False
    assert out["error_code"] == "HOSPITAL_CONFLICT"


# --------------------------------------------------------------------- 本地药品规则

def test_lookup_known_drug():
    out = _body(hospital_tools.lookup_drug_label("速诺"))
    assert out["ok"] is True
    data = out["data"]
    assert "速诺" in data["label"]
    assert "兔" in data["species_banned"]
    assert data["dose_mg_per_kg"]["犬"] == [12.5, 12.5]
    assert data["citations"]


def test_lookup_unknown_drug_is_structured_failure():
    out = _body(hospital_tools.lookup_drug_label("不存在的药"))
    assert out["ok"] is False
    assert out["error_code"] == "DRUG_NOT_IN_TABLE"


def test_check_drug_safety_blocks_penicillin_allergy():
    out = _body(hospital_tools.check_drug_safety(
        "速诺", species="猫", allergies="青霉素"))
    assert out["ok"] is True
    assert out["data"]["verdict"] == "block"
    assert "ALLERGY_CONTRAINDICATION" in [r["code"] for r in out["data"]["reasons"]]
    assert out["data"]["citations"]


def test_check_drug_safety_species_banned():
    out = _body(hospital_tools.check_drug_safety("阿莫西林", species="兔"))
    assert out["data"]["verdict"] == "block"
    assert "SPECIES_BANNED" in [r["code"] for r in out["data"]["reasons"]]


def test_check_drug_safety_unknown_drug_warns_not_passes():
    out = _body(hospital_tools.check_drug_safety("某新药", species="犬"))
    assert out["data"]["verdict"] == "warn"
    assert "DRUG_NOT_IN_KNOWLEDGE" in [r["code"] for r in out["data"]["reasons"]]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("全部断言通过")

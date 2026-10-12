"""tests/test_patient_context.py：档案上下文注入 + 链路头（X-Trace-Id）对齐断言。"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402
from core_agent import ReActAgent, render_patient_context  # noqa: E402
from services.agent_service import set_agent  # noqa: E402
from tools import ToolRegistry  # noqa: E402

CTX = {
    "species": "犬", "breed": "金毛寻回犬", "age_years": 3, "weight_kg": 28.5,
    "allergies": "青霉素", "latest_diagnosis": "慢性肾病", "vaccine_alert_count": 1,
}


class CaptureClient:
    """记录 create() 收到的 messages，并返回一段固定文本流（不产生工具调用）。"""

    def __init__(self):
        self.seen = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.seen.append(kwargs.get("messages"))
        delta = SimpleNamespace(content="好的", tool_calls=None)
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=delta)])])


def _agent(client):
    return ReActAgent(client, ToolRegistry(), model="fake")


# ---------------------------------------------------------------- 渲染函数

def test_render_fields():
    text = render_patient_context(CTX)
    for frag in ("物种: 犬", "品种: 金毛寻回犬", "年龄: 3岁", "体重: 28.5kg",
                 "过敏史: 青霉素", "最近一次诊断: 慢性肾病", "疫苗逾期/将到期提醒数: 1"):
        assert frag in text, frag


def test_render_empty_inputs():
    assert render_patient_context(None) == ""
    assert render_patient_context({}) == ""
    assert render_patient_context({"pet_id": 7}) == ""   # 只有 id、无有效字段 → 不注入


# ---------------------------------------------------------------- 注入

def test_context_injected_into_system_prompt():
    client = CaptureClient()
    list(_agent(client).run_stream("它最近怎么样", patient_context=CTX))
    system = client.seen[0][0]
    assert system["role"] == "system"
    assert "宠物档案上下文" in system["content"]
    assert "金毛寻回犬" in system["content"]


def test_without_context_prompt_unchanged():
    client = CaptureClient()
    list(_agent(client).run_stream("你好"))
    assert "宠物档案上下文" not in client.seen[0][0]["content"]


# ---------------------------------------------------------------- 链路头对齐

def test_trace_header_passthrough_and_generate():
    set_agent(_agent(CaptureClient()))
    with TestClient(app) as c:
        r1 = c.post("/api/chat/stream", json={"prompt": "你好"},
                    headers={"X-Trace-Id": "trace1234567890"})
        assert r1.headers.get("x-trace-id") == "trace1234567890"
        # 兼容历史头名 X-Request-Id
        r2 = c.post("/api/chat/stream", json={"prompt": "你好"},
                    headers={"X-Request-Id": "legacy12345678"})
        assert r2.headers.get("x-trace-id") == "legacy12345678"
        # 缺失则生成
        r3 = c.post("/api/chat/stream", json={"prompt": "你好"})
        assert r3.headers.get("x-trace-id")


def test_endpoint_passes_patient_context():
    client = CaptureClient()
    set_agent(_agent(client))
    with TestClient(app) as c:
        c.post("/api/chat/stream", json={"prompt": "它怎么样", "patient_context": CTX})
    assert any("宠物档案上下文" in m[0]["content"] for m in client.seen)


# ---------------------------------------------------------------- 档案补缺分诊

from fastpath import FastPathGuard  # noqa: E402
from triage import extract_slots, merge_patient_context  # noqa: E402


def test_merge_fills_weight_from_archive():
    slots = merge_patient_context(extract_slots("狗狗误食了巧克力怎么办"), {"weight_kg": 28.5})
    assert slots["weight_kg"] == 28.5


def test_prompt_takes_priority_over_archive():
    slots = merge_patient_context(extract_slots("10公斤的金毛误食了巧克力"), {"weight_kg": 28.5})
    assert slots["weight_kg"] == 10.0


def test_archive_bad_value_ignored():
    slots = merge_patient_context(extract_slots("狗狗误食了巧克力"), {"weight_kg": "很重"})
    # extract_slots 的键恒存在、值为 None；档案值非法时应保持 None（按缺失处理）
    assert slots["weight_kg"] is None


def test_no_archive_returns_slots_untouched():
    raw = extract_slots("10公斤的金毛误食了巧克力")
    assert merge_patient_context(raw, None) == raw


def test_fastpath_check_prefills_weight_from_archive():
    hit = FastPathGuard().check("狗狗误食了巧克力怎么办", {"weight_kg": 28.5})
    assert hit["action"] == "emergency"
    assert hit["slots"]["weight_kg"] == 28.5


def test_fastpath_check_without_archive_still_asks():
    hit = FastPathGuard().check("狗狗误食了巧克力怎么办")
    assert hit["action"] == "emergency"
    assert not hit["slots"].get("weight_kg")


if __name__ == "__main__":
    test_render_fields()
    test_render_empty_inputs()
    test_context_injected_into_system_prompt()
    test_without_context_prompt_unchanged()
    test_trace_header_passthrough_and_generate()
    test_endpoint_passes_patient_context()
    test_merge_fills_weight_from_archive()
    test_prompt_takes_priority_over_archive()
    test_archive_bad_value_ignored()
    test_no_archive_returns_slots_untouched()
    test_fastpath_check_prefills_weight_from_archive()
    test_fastpath_check_without_archive_still_asks()
    print("全部断言通过")

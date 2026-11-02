"""tests/test_memory_l3.py：D 批 L3 长期记忆 —— 渲染预算 / 注入口径 / 安全隔离。

全部离线：记忆由 ``AgentContext.memory_provider`` 注入，不依赖医院服务与模型。
核心断言：``AI_INFERRED`` 记忆**只能出现在背景文本**，绝不进入安全判定入口。
"""
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from a2a.agents import AgentContext, consult_stream
from a2a.contract import CaseState, CaseTask, Step, Verdict
from a2a.store import CaseStore
from a2a.supervisor import Supervisor
from core_agent import render_patient_context
from hospital_client import HospitalClient
from memory import estimate_tokens
from memory_l3 import (
    MEMORY_HEADER,
    MEMORY_TOKEN_BUDGET,
    render_memories,
    render_memory_block,
    safety_authoritative,
)
from prescription_review import PrescriptionReviewer
from record_draft import build_draft_messages

HOSPITAL = {"factType": "ALLERGY", "factValue": "青霉素", "source": "HOSPITAL_RECORD"}
OWNER = {"factType": "CHRONIC_DISEASE", "factValue": "慢性肾病", "source": "OWNER_CONFIRMED"}
INFERRED = {"factType": "DIET", "factValue": "可能对鸡肉不耐受", "source": "AI_INFERRED"}


# --------------------------------------------------------------------- 渲染

def test_render_lines_carry_source_and_fact_labels():
    block = render_memory_block([HOSPITAL, OWNER, INFERRED])
    assert block.startswith(MEMORY_HEADER)
    assert "[医院档案] 过敏史：青霉素" in block
    assert "[宠主确认] 慢性病：慢性肾病" in block
    assert "[AI 推断（待确认）] 饮食：可能对鸡肉不耐受" in block


def test_render_accepts_snake_case_payload():
    block = render_memory_block([{"fact_type": "MEDICATION", "fact_value": "苯巴比妥",
                                  "source": "OWNER_CONFIRMED"}])
    assert "长期用药：苯巴比妥" in block


def test_render_skips_invalid_items():
    assert render_memory_block(None) == ""
    assert render_memory_block([None, {}, {"factValue": "  "}]) == ""
    assert render_memory_block("not-a-list") == ""


def test_render_keeps_top_eight():
    items = [{"factType": "OTHER", "factValue": f"事实{i}", "source": "OWNER_CONFIRMED"}
             for i in range(12)]
    assert len(render_memories(items, max_tokens=10000)) == 8


def test_render_respects_token_budget():
    items = [{"factType": "OTHER", "factValue": "很长的历史事实" * 12,
              "source": "HOSPITAL_RECORD"} for _ in range(20)]
    block = render_memory_block(items)
    assert block, "单条超预算时应截断正文而不是整块丢弃"
    assert estimate_tokens(block) <= MEMORY_TOKEN_BUDGET
    assert "…" in block and "医院档案" in block


def test_safety_authoritative_only_counts_verified_sources():
    assert safety_authoritative(HOSPITAL) is True
    assert safety_authoritative(OWNER) is True
    assert safety_authoritative(INFERRED) is False
    # 医院侧显式下发的布尔值优先（后端是唯一权威）
    assert safety_authoritative({**HOSPITAL, "safetyAuthoritative": False}) is False
    assert safety_authoritative({**INFERRED, "safety_authoritative": True}) is True


# --------------------------------------------------------------------- 注入

def test_draft_messages_include_memories():
    messages = build_draft_messages({"chief_complaint": "呕吐"},
                                    [{"role": "user", "text": "吐了两天"}],
                                    {}, {"species": "犬"}, memories=[HOSPITAL])
    user = messages[-1]["content"]
    assert "长期记忆" in user
    assert "过敏史：青霉素" in user


def test_draft_messages_without_memories_unchanged():
    messages = build_draft_messages({"chief_complaint": "呕吐"}, [], {}, {"species": "犬"})
    assert "长期记忆：" not in messages[-1]["content"]


def test_patient_context_appends_memory_block():
    text = render_patient_context({"species": "犬", "pet_id": 7, "ai_memories": [HOSPITAL]})
    assert "物种: 犬" in text and "档案ID: 7" in text
    assert MEMORY_HEADER in text and "过敏史：青霉素" in text


def test_patient_context_unchanged_without_memories():
    assert "长期记忆" not in render_patient_context({"species": "犬"})
    assert render_patient_context({"ai_memories": []}) == ""


def test_consult_stream_passes_memories_as_background():
    captured = {}

    def fake_chat(prompt, history, patient_context, case_id=""):
        captured["patient"] = patient_context
        yield {"event": "done", "data": {"answer": "好的", "citations": []}}

    case = CaseState(case_id="c-mem", patient={"species": "犬", "pet_id": 7},
                     memories=[HOSPITAL], memory_loaded=True)
    task = CaseTask(step=Step.CONSULT, case_id="c-mem", idempotency_key="k-mem",
                    payload={"prompt": "它最近怎么样"})
    list(consult_stream(case, task, AgentContext(chat_stream=fake_chat)))
    assert captured["patient"]["ai_memories"] == [HOSPITAL]
    assert captured["patient"]["species"] == "犬"


# --------------------------------------------------------------------- 编排层加载

def _store():
    return CaseStore(":memory:")


def _review_task(key="k1", case_id="c1", patient=None):
    return CaseTask(step=Step.REVIEW, case_id=case_id, idempotency_key=key,
                    payload={"patient": patient or {"species": "犬", "pet_id": 7},
                             "prescription": {"items": [
                                 {"drug_name": "速诺", "dose_mg_per_kg": 12.5}]}})


def test_case_loads_memories_once():
    calls = []

    def provider(pet_id):
        calls.append(pet_id)
        return [HOSPITAL]

    store = _store()
    sup = Supervisor(store, AgentContext(reviewer=PrescriptionReviewer(),
                                         memory_provider=provider))
    try:
        first = sup.submit(_review_task("k1"))
        second = sup.submit(_review_task("k2"))
        case = sup.get_case("c1")
    finally:
        store.close()

    assert first.verdict is Verdict.PASS and second.verdict is Verdict.PASS
    assert calls == [7], "同一个 case 只允许取一次 L3 记忆"
    assert case.memory_loaded is True
    assert case.memories == [HOSPITAL]


def test_memory_provider_failure_degrades_silently():
    def provider(pet_id):
        raise RuntimeError("医院不可达")

    store = _store()
    sup = Supervisor(store, AgentContext(reviewer=PrescriptionReviewer(),
                                         memory_provider=provider))
    try:
        result = sup.submit(_review_task())
        case = sup.get_case("c1")
    finally:
        store.close()

    assert result.status.value == "ok", "记忆失败不得拖垮审核主流程"
    assert result.verdict is Verdict.PASS
    assert case.memories == [] and case.memory_loaded is True


def test_case_without_pet_id_defers_loading():
    calls = []
    store = _store()
    sup = Supervisor(store, AgentContext(reviewer=PrescriptionReviewer(),
                                         memory_provider=lambda pid: calls.append(pid) or []))
    try:
        case = sup.open_case("c1", {"species": "犬"})
        assert calls == [] and case.memory_loaded is False
        sup.submit(_review_task("k1", patient={"species": "犬", "pet_id": 7}))
        assert calls == [7]
    finally:
        store.close()


def test_open_case_loads_memories():
    store = _store()
    sup = Supervisor(store, AgentContext(memory_provider=lambda pid: [OWNER]))
    try:
        case = sup.open_case("c1", {"species": "犬", "pet_id": 9})
        reloaded = sup.get_case("c1")
    finally:
        store.close()

    assert case.memory_loaded is True
    assert reloaded.memories == [OWNER]


# --------------------------------------------------------------------- 安全隔离

def test_ai_inferred_memory_never_affects_review():
    """AI 推断的「过敏」不能当过敏史用：审核器只看 case.patient。"""
    store = _store()
    sup = Supervisor(store, AgentContext(
        reviewer=PrescriptionReviewer(),
        memory_provider=lambda pid: [{"factType": "ALLERGY", "factValue": "青霉素",
                                      "source": "AI_INFERRED"}],
    ))
    try:
        result = sup.submit(_review_task(patient={"species": "犬", "weight_kg": 10,
                                                  "pet_id": 7}))
        case = sup.get_case("c1")
    finally:
        store.close()

    assert result.verdict is Verdict.PASS, "AI_INFERRED 记忆不得触发禁忌拦截"
    assert case.memories[0]["source"] == "AI_INFERRED"
    assert "allergies" not in case.patient, "记忆不得被合并进宠物事实"


# --------------------------------------------------------------------- 医院客户端

def _client(handler, secret="test-secret"):
    return HospitalClient(base_url="http://hospital.test", secret=secret,
                          transport=httpx.MockTransport(handler))


def test_list_memories_request_shape():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["limit"] = request.url.params.get("limit")
        return httpx.Response(200, json={"code": 200, "data": [HOSPITAL]})

    out = _client(handler).list_memories(7, limit=8)
    assert out == [HOSPITAL]
    assert seen["path"] == "/internal/ai/pet/7/memories"
    assert seen["limit"] == "8"


def test_memory_write_bodies_and_no_retry():
    bodies = []

    def handler(request):
        bodies.append((request.url.path, __import__("json").loads(request.content)))
        return httpx.Response(200, json={"code": 200, "data": {"id": 1}})

    client = _client(handler)
    client.add_memory(7, "ALLERGY", "青霉素", source="OWNER_CONFIRMED", actor="VETCLAW")
    client.delete_memory(3, "宠主要求删除")
    assert bodies[0] == ("/internal/ai/memory", {
        "petId": 7, "factType": "ALLERGY", "factValue": "青霉素",
        "source": "OWNER_CONFIRMED", "confidence": None,
        "expiresAt": None, "actor": "VETCLAW"})
    assert bodies[1] == ("/internal/ai/memory/3/delete",
                         {"reason": "宠主要求删除", "actor": "VETCLAW"})


def test_memory_read_retries_once():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(200, json={"code": 200, "data": []})

    assert _client(handler).list_memories(7) == []
    assert calls["n"] == 2

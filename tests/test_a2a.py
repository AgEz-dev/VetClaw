"""tests/test_a2a.py：A2A 编排层 —— 契约 / case 存储 / 幂等 / supervisor / HTTP 路由。

全部离线：能力位依赖由 `AgentContext` 注入，不需要网络、BGE 或 Chroma。
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi.testclient import TestClient

from a2a.agents import AgentContext
from a2a.contract import (
    CaseState,
    CaseTask,
    Step,
    TaskResult,
    TaskStatus,
    Verdict,
    fingerprint,
    worst_verdict,
)
from a2a.store import CaseStore
from a2a.supervisor import IdempotencyConflict, Supervisor
from prescription_review import PrescriptionReviewer

# --------------------------------------------------------------------- fixtures

@pytest.fixture()
def store():
    s = CaseStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def supervisor(store):
    return Supervisor(store, AgentContext(reviewer=PrescriptionReviewer()))


def _review_task(key="k1", case_id="appt-1", items=None, **kw):
    payload = kw.pop("payload", None)
    if payload is None:
        payload = {"patient": {"species": "犬"},
                   "prescription": {"items": items or [{"drug_name": "速诺",
                                                        "dose_mg_per_kg": 12.5}]}}
    return CaseTask(step=Step.REVIEW, case_id=case_id, idempotency_key=key,
                    payload=payload, **kw)


# --------------------------------------------------------------------- 契约

def test_task_requires_idempotency_key():
    with pytest.raises(ValueError, match="idempotency_key"):
        CaseTask(step=Step.REVIEW, case_id="c1", idempotency_key="")


def test_task_requires_case_id():
    with pytest.raises(ValueError, match="case_id"):
        CaseTask(step=Step.REVIEW, case_id="", idempotency_key="k")


def test_task_autofills_ids():
    t = CaseTask(step=Step.REVIEW, case_id="c1", idempotency_key="k")
    assert t.task_id and t.trace_id
    assert t.task_id.startswith("task_") and t.trace_id.startswith("trace_")


def test_fingerprint_is_order_independent():
    """键顺序不同但内容相同 → 指纹必须一致，否则会误判为幂等冲突。"""
    a = fingerprint({"a": 1, "b": 2})
    b = fingerprint({"b": 2, "a": 1})
    assert a == b


def test_fingerprint_changes_with_content():
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_deadline_expiry():
    t = CaseTask(step=Step.REVIEW, case_id="c", idempotency_key="k",
                 deadline_at=time.time() - 10)
    assert t.is_expired()
    t2 = CaseTask(step=Step.REVIEW, case_id="c", idempotency_key="k")
    assert not t2.is_expired()


def test_worst_verdict_severity_order():
    assert worst_verdict(Verdict.NONE, Verdict.PASS) is Verdict.PASS
    assert worst_verdict(Verdict.PASS, Verdict.WARN) is Verdict.WARN
    assert worst_verdict(Verdict.WARN, Verdict.BLOCK) is Verdict.BLOCK
    assert worst_verdict() is Verdict.NONE


def test_result_roundtrip_dict():
    r = TaskResult(case_id="c", task_id="t", step=Step.REVIEW,
                   status=TaskStatus.OK, verdict=Verdict.BLOCK,
                   payload={"x": 1}, citations=["k#1"], reasons=[{"code": "C"}])
    back = TaskResult.from_dict(r.to_dict())
    assert back.verdict is Verdict.BLOCK and back.payload == {"x": 1}
    assert back.step is Step.REVIEW


def test_case_state_record_and_lookup():
    case = CaseState(case_id="c")
    r = TaskResult(case_id="c", task_id="t", step=Step.INTAKE,
                   status=TaskStatus.OK, payload={"summary": {"a": 1}})
    case.record(r)
    assert case.seq == 1
    got = case.result_of(Step.INTAKE)
    assert got is not None and got.payload == {"summary": {"a": 1}}
    assert case.result_of(Step.REVIEW) is None


# --------------------------------------------------------------------- 存储

def test_store_create_is_idempotent_and_keeps_context(store):
    first = store.create_case("c1", {"species": "犬"})
    result = TaskResult(case_id="c1", task_id="t", step=Step.TRIAGE,
                        status=TaskStatus.OK)
    first.record(result)
    store.save_case(first)

    again = store.create_case("c1", {"species": "猫"})   # 不应覆盖
    assert again.patient == {"species": "犬"}
    assert Step.TRIAGE.value in again.steps


def test_store_get_missing_returns_none(store):
    assert store.get_case("nope") is None


# --------------------------------------------------------------------- supervisor

def test_submit_review_returns_block(supervisor):
    r = supervisor.submit(_review_task(items=[{"drug_name": "对乙酰氨基酚"}]))
    assert r.status is TaskStatus.OK
    assert r.verdict is Verdict.BLOCK
    assert r.citations


def test_idempotent_replay_same_key_same_content(supervisor):
    t1 = _review_task(key="key-replay")
    r1 = supervisor.submit(t1)
    r2 = supervisor.submit(_review_task(key="key-replay"))
    assert r2.deduped is True, "同键同内容必须回放，不得重复执行"
    assert r2.task_id == r1.task_id, "回放应返回第一次的结果"


def test_idempotency_conflict_on_different_content(supervisor):
    supervisor.submit(_review_task(key="key-conflict"))
    with pytest.raises(IdempotencyConflict):
        supervisor.submit(_review_task(
            key="key-conflict", items=[{"drug_name": "甲硝唑"}]))


def test_deadline_exceeded_does_not_execute(supervisor):
    r = supervisor.submit(_review_task(key="k-dead",
                                       deadline_at=time.time() - 1))
    assert r.status is TaskStatus.FAILED
    assert r.payload.get("code") == "DEADLINE_EXCEEDED"


def test_case_context_is_inherited_by_later_step(supervisor):
    """**核心不变量**：建案时录的宠物事实，后续能力位不必重传。

    第一步只建案（species=兔），第二步处方审核的 payload 里**完全不提 patient**，
    仍必须因为「阿莫西林禁用于兔」而被阻断。
    """
    supervisor.open_case("appt-rabbit", {"species": "兔"})
    r = supervisor.submit(_review_task(
        key="k-rabbit", case_id="appt-rabbit",
        payload={"prescription": {"items": [{"drug_name": "阿莫西林"}]}}))
    assert r.verdict is Verdict.BLOCK
    assert any(x["code"] == "SPECIES_BANNED" for x in r.reasons)


def test_need_more_is_memoized(supervisor):
    payload = {"patient": {"species": "犬"}, "prescription": {}}
    r1 = supervisor.submit(_review_task(key="k-need", payload=payload))
    assert r1.status is TaskStatus.NEED_MORE
    r2 = supervisor.submit(_review_task(key="k-need", payload=payload))
    assert r2.deduped is True


def test_failed_result_is_not_memoized(store):
    """**瞬时失败必须可用同一个键重试** —— 若记忆化，一次抖动会让任务永久失败。"""
    sup = Supervisor(store, AgentContext())      # 无 llm → 起草必失败
    task = CaseTask(step=Step.DRAFT, case_id="c-draft", idempotency_key="k-fail",
                    payload={"summary": {"chief_complaint": "x"}})
    r1 = sup.submit(task)
    assert r1.status is TaskStatus.FAILED
    r2 = sup.submit(CaseTask(step=Step.DRAFT, case_id="c-draft",
                             idempotency_key="k-fail",
                             payload={"summary": {"chief_complaint": "x"}}))
    assert r2.deduped is False, "失败的执行不得被幂等表记忆"


def test_agent_exception_becomes_failed_result(store):
    def boom(case, task, ctx):
        raise RuntimeError("simulated")

    sup = Supervisor(store, AgentContext(),
                     agents={Step.REVIEW: boom})
    r = sup.submit(_review_task(key="k-boom"))
    assert r.status is TaskStatus.FAILED
    assert r.payload.get("code") == "AGENT_ERROR"


def test_draft_agent_pulls_inputs_from_case(store):
    """起草从 case 里的预问诊产出取输入 —— 调用方不必重传摘要与对话。"""
    captured = {}

    def fake_llm(messages):
        captured["messages"] = messages
        return '{"chief_complaint":"呕吐 2 天","exam_checklist":[],"doctor_advice":""}'

    sup = Supervisor(store, AgentContext(llm=fake_llm))
    sup.open_case("c-draft2", {"species": "犬", "weight_kg": 8})

    intake = TaskResult(case_id="c-draft2", task_id="t-in", step=Step.INTAKE,
                        status=TaskStatus.OK,
                        payload={"summary": {"chief_complaint": "呕吐 2 天"},
                                 "dialogue": [{"role": "user", "text": "昨天开始吐"}],
                                 "state": {"slots": {}, "turns": 1}})
    case = sup.get_case("c-draft2")
    case.record(intake)
    store.save_case(case)

    r = sup.submit(CaseTask(step=Step.DRAFT, case_id="c-draft2",
                            idempotency_key="k-draft", payload={}))
    assert r.status is TaskStatus.OK
    assert r.payload["draft"]["chief_complaint"] == "呕吐 2 天"
    joined = "".join(m["content"] for m in captured["messages"])
    assert "呕吐 2 天" in joined, "case 内的预问诊摘要必须进入起草提示词"


def test_intake_state_inherited_from_case(store):
    """预问诊续传状态从 case 取：调用方不必每轮往返 state。"""
    seen = {}

    class FakeSession:
        def __init__(self, state):
            seen["state"] = state

        def handle(self, prompt):
            return {"questions": [], "state": {"turns": 2}, "prompt": prompt}

    sup = Supervisor(store, AgentContext(
        intake_session_factory=lambda state, archive, patient: FakeSession(state)))
    sup.open_case("c-intake", {})
    first = TaskResult(case_id="c-intake", task_id="t1", step=Step.INTAKE,
                       status=TaskStatus.OK, payload={"state": {"turns": 1}})
    case = sup.get_case("c-intake")
    case.record(first)
    store.save_case(case)

    r = sup.submit(CaseTask(step=Step.INTAKE, case_id="c-intake",
                            idempotency_key="k-intake", payload={"prompt": "还吐吗"}))
    assert r.status is TaskStatus.OK
    assert seen["state"] == {"turns": 1}, "预问诊应继承 case 内的续传状态"


# --------------------------------------------------------------------- HTTP 路由

@pytest.fixture()
def client(supervisor):
    from a2a.routes import set_supervisor
    from app import app
    set_supervisor(supervisor)
    yield TestClient(app)
    set_supervisor(None)


def test_api_open_and_get_case(client):
    resp = client.post("/a2a/case", json={"case_id": "appt-9",
                                          "patient": {"species": "犬"}})
    assert resp.status_code == 200
    assert resp.json()["data"]["case_id"] == "appt-9"

    got = client.get("/a2a/case/appt-9")
    assert got.status_code == 200
    assert got.json()["data"]["patient"]["species"] == "犬"


def test_api_missing_case_returns_404(client):
    assert client.get("/a2a/case/does-not-exist").status_code == 404


def test_api_review_task_returns_block(client):
    body = {"step": "review", "case_id": "appt-r", "idempotency_key": "api-k1",
            "payload": {"patient": {"species": "猫", "allergies": "青霉素"},
                        "prescription": {"items": [{"drug_name": "速诺"}]}}}
    resp = client.post("/a2a/task", json=body)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["verdict"] == "block"
    assert data["status"] == "ok"


def test_api_idempotency_conflict_returns_409(client):
    base = {"step": "review", "case_id": "appt-409", "idempotency_key": "api-409",
            "payload": {"patient": {"species": "犬"},
                        "prescription": {"items": [{"drug_name": "速诺"}]}}}
    assert client.post("/a2a/task", json=base).status_code == 200
    changed = dict(base)
    changed["payload"] = {"patient": {"species": "犬"},
                          "prescription": {"items": [{"drug_name": "甲硝唑"}]}}
    resp = client.post("/a2a/task", json=changed)
    assert resp.status_code == 409
    assert resp.json()["code"] == "IDEMPOTENCY_CONFLICT"


def test_api_unknown_step_rejected(client):
    resp = client.post("/a2a/task", json={"step": "hack", "case_id": "c",
                                          "idempotency_key": "k", "payload": {}})
    assert resp.status_code == 422


def test_api_missing_idempotency_key_rejected(client):
    resp = client.post("/a2a/task", json={"step": "review", "case_id": "c",
                                          "payload": {}})
    assert resp.status_code == 422


def test_api_case_chain_works_end_to_end(client):
    """建案 → 审核 → 查 case：三个端点串起来可用。"""
    assert client.post("/a2a/case", json={"case_id": "appt-chain",
                                          "patient": {"species": "犬"}}).status_code == 200
    body = {"step": "review", "case_id": "appt-chain", "idempotency_key": "chain-1",
            "payload": {"prescription": {"items": [{"drug_name": "速诺",
                                                    "dose_mg_per_kg": 12.5}]}}}
    assert client.post("/a2a/task", json=body).json()["data"]["verdict"] == "pass"
    got = client.get("/a2a/case/appt-chain").json()["data"]
    assert got["seq"] >= 1 and "review" in got["steps"]


# --------------------------------------------------------------------- 预问诊：开场 vs 轮次

def test_intake_opening_forces_fresh_session(store):
    """**关键不变量**：`opening:true` 必须用全新会话，不继承 case 里已进行到的进度。

    否则用户在第 5 轮刷新页面，会收到一句"我们从头开始吧"式的矛盾问候。
    """
    seen = {}

    class FakeSession:
        def __init__(self, state):
            seen["state"] = state

        def opening(self):
            return {"message": "开场白",
                    "questions": [{"key": "vomit", "ask": "吐了几次"}],
                    "state": {"turns": 0}}

        def handle(self, prompt):
            return {"message": "已处理", "state": {"turns": 9}}

    sup = Supervisor(store, AgentContext(
        intake_session_factory=lambda state, archive, patient: FakeSession(state)))

    case = sup.open_case("c-open", {})
    case.record(TaskResult(case_id="c-open", task_id="t-prev", step=Step.INTAKE,
                           status=TaskStatus.OK, payload={"state": {"turns": 3}}))
    store.save_case(case)

    r = sup.submit(CaseTask(step=Step.INTAKE, case_id="c-open",
                            idempotency_key="k-open", payload={"opening": True}))
    assert r.status is TaskStatus.OK
    assert seen["state"] is None, "开场必须传 None 给会话工厂（全新会话）"
    assert r.payload["message"] == "开场白"


def test_intake_turn_requires_prompt(store):
    """有 case、但本轮没给 prompt → NEED_MORE（不能默默再开场一次）。"""
    sup = Supervisor(store, AgentContext(
        intake_session_factory=lambda state, archive, patient: None))
    sup.open_case("c-noprompt", {})
    r = sup.submit(CaseTask(step=Step.INTAKE, case_id="c-noprompt",
                            idempotency_key="k-noprompt", payload={}))
    assert r.status is TaskStatus.NEED_MORE
    assert r.payload.get("missing") == ["prompt"]


# --------------------------------------------------------------------- 分诊

class _FakeGuard:
    """替身守卫：只实现 supervisor 需要的那几个方法。"""

    def __init__(self, action):
        self.action = action
        self.seen = None

    def check(self, query, patient_context=None):
        self.seen = (query, patient_context)
        if self.action == "emergency":
            return {"action": "emergency", "toxin": "百合", "species": "猫",
                    "chunk_id": "knowledge/species_drug_safety_matrix.md#百合属-Lilium-Hemerocallis-所有部位-4"}
        if self.action == "refuse":
            return {"action": "refuse", "drug_hint": "头孢"}
        return {"action": "pass"}

    def emergency_message(self, toxin, species, chunk_id=None, slots=None):
        return f"急症：{species}接触{toxin}"

    def refuse_message(self, drug_hint):
        return f"拒答：{drug_hint}未收录"


def test_triage_emergency_is_block_with_citation(store):
    """P0 急症 → verdict=block，且引用指向毒物矩阵（下游据此走急诊流程）。"""
    guard = _FakeGuard("emergency")
    sup = Supervisor(store, AgentContext(guard=guard))
    r = sup.submit(CaseTask(step=Step.TRIAGE, case_id="appt-em",
                            idempotency_key="k-tri-em",
                            payload={"query": "猫误食了百合花怎么办"}))
    assert r.status is TaskStatus.OK
    assert r.verdict is Verdict.BLOCK
    assert any("species_drug_safety_matrix" in c for c in r.citations)
    assert "百合" in r.payload.get("answer", "")


def test_triage_pass_is_not_block(store):
    guard = _FakeGuard("pass")
    sup = Supervisor(store, AgentContext(guard=guard))
    r = sup.submit(CaseTask(step=Step.TRIAGE, case_id="appt-ok",
                            idempotency_key="k-tri-ok",
                            payload={"query": "狗狗最近食欲怎么样"}))
    assert r.verdict is Verdict.PASS


def test_triage_uses_case_patient_when_payload_omits_it(store):
    """分诊也要吃 case 里的宠物事实：payload 不带 patient 时用 case 的。"""
    guard = _FakeGuard("pass")
    sup = Supervisor(store, AgentContext(guard=guard))
    sup.open_case("appt-tri", {"species": "猫", "weight_kg": 3.1})
    sup.submit(CaseTask(step=Step.TRIAGE, case_id="appt-tri",
                        idempotency_key="k-tri-inherit",
                        payload={"query": "它吐了"}))
    assert guard.seen[1] == {"species": "猫", "weight_kg": 3.1}

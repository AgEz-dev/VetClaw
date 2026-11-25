"""tests/test_a2a_pretriage.py：编排层前置分诊（A 批 A2）。

要钉住的三件事：
1. BLOCK → 原能力位**一次都不执行**（不调 LLM、不推进预问诊轮次）；
2. WARN → 照常执行，但结论合并为 WARN（规则命中不被模型结论盖过）；
3. 无主诉 / 未装配守卫 → 不拦（开场白、离线单测两条路径都不能被误伤）。

全部离线：守卫与问答流都是替身，不需要网络、BGE 或 Chroma。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from a2a.agents import AgentContext
from a2a.contract import CaseTask, Step, TaskStatus, Verdict
from a2a.store import CaseStore
from a2a.supervisor import Supervisor


class _Guard:
    """替身守卫：只回放预置结论，不读规则表。"""

    def __init__(self, action):
        self.action = action

    def check(self, query, patient_context=None):
        if self.action == "emergency":
            return {"action": "emergency", "toxin": "巧克力", "species": "犬",
                    "chunk_id": "knowledge/toxic.md#chocolate", "slots": {}}
        if self.action == "refuse":
            return {"action": "refuse", "drug_hint": "布洛芬"}
        return {"action": "pass"}

    def emergency_message(self, toxin, species, chunk_id=None, slots=None):
        return f"【急症警报】{species}疑似误食{toxin}，请立即就近就医。"

    def refuse_message(self, drug_hint):
        return f"【分诊引导】{drug_hint}不在已收录范围，不提供剂量建议。"


class _Session:
    def __init__(self):
        self.handled = []

    def opening(self):
        return {"message": "先说说它怎么了", "questions": [{"slot": "chief"}],
                "slots": {}, "state": {"turns": 0}}

    def handle(self, prompt):
        self.handled.append(prompt)
        return {"message": "还有别的症状吗", "questions": [{"slot": "vomit"}],
                "slots": {}, "state": {"turns": 1}, "done": False, "citations": []}


class _Spy:
    """记录被调用次数的预问诊工厂。"""

    def __init__(self):
        self.calls = 0
        self.session = _Session()

    def __call__(self, state, archive, patient):
        self.calls += 1
        return self.session


def _fake_chat(calls):
    def stream(prompt, history, patient_context, case_id=""):
        calls.append(prompt)
        yield {"event": "token", "data": {"text": "正常回答"}}
        yield {"event": "done", "data": {"answer": "正常回答", "citations": []}}

    return stream


@pytest.fixture()
def store():
    s = CaseStore(":memory:")
    yield s
    s.close()


def _intake_task(key="k1", prompt="狗吃了巧克力怎么办"):
    return CaseTask(step=Step.INTAKE, case_id="appt-9", idempotency_key=key,
                    payload={"prompt": prompt})


def _consult_task(key="k2", prompt="狗吃了巧克力怎么办"):
    return CaseTask(step=Step.CONSULT, case_id="chat-1", idempotency_key=key,
                    payload={"prompt": prompt})


# --------------------------------------------------------------------- BLOCK

def test_intake_emergency_short_circuits_without_running_agent(store):
    factory = _Spy()
    sup = Supervisor(store, AgentContext(guard=_Guard("emergency"),
                                         intake_session_factory=factory))

    result = sup.submit(_intake_task())

    assert factory.calls == 0, "BLOCK 之后预问诊不得执行"
    assert result.status is TaskStatus.OK
    assert result.verdict is Verdict.BLOCK
    assert result.payload["pretriage"] is True
    assert result.payload["message"].startswith("【急症警报】")
    assert result.payload["questions"] == []
    assert result.payload["slots"] == {}


def test_short_circuit_is_recorded_in_case_but_does_not_pose_as_triage(store):
    factory = _Spy()
    sup = Supervisor(store, AgentContext(guard=_Guard("emergency"),
                                         intake_session_factory=factory))
    sup.submit(_intake_task())

    case = sup.get_case("appt-9")
    assert case.steps["intake"]["verdict"] == "block"
    # 前置分诊不写 triage 步：显式分诊调用才有资格占用这个位置
    assert "triage" not in case.steps


def test_consult_emergency_stream_has_no_token_and_no_agent_call(store):
    calls = []
    sup = Supervisor(store, AgentContext(guard=_Guard("emergency"),
                                         chat_stream=_fake_chat(calls)))

    frames = list(sup.submit_stream(_consult_task()))

    assert calls == [], "BLOCK 之后问答不得执行"
    assert [f["event"] for f in frames] == ["start", "done"]
    assert frames[-1]["data"]["verdict"] == "block"
    assert frames[-1]["data"]["answer"].startswith("【急症警报】")


def test_replay_after_short_circuit_is_deduped(store):
    factory = _Spy()
    sup = Supervisor(store, AgentContext(guard=_Guard("emergency"),
                                         intake_session_factory=factory))
    task = _intake_task()
    first = sup.submit(task)
    second = sup.submit(task)

    assert first.verdict is Verdict.BLOCK
    assert second.deduped is True
    assert factory.calls == 0


# --------------------------------------------------------------------- WARN

def test_refuse_warn_still_executes_and_merges_verdict(store):
    factory = _Spy()
    sup = Supervisor(store, AgentContext(guard=_Guard("refuse"),
                                         intake_session_factory=factory))

    result = sup.submit(_intake_task(prompt="布洛芬给狗吃多少"))

    assert factory.calls == 1, "WARN 只合结论，不短路"
    assert result.verdict is Verdict.WARN
    assert result.payload["questions"], "仍应返回预问诊的追问"
    assert any(r.get("code") == "UNLISTED_DRUG_DOSAGE" for r in result.reasons)


# --------------------------------------------------------------------- 不拦

def test_opening_without_free_text_is_not_gated(store):
    factory = _Spy()
    sup = Supervisor(store, AgentContext(guard=_Guard("emergency"),
                                         intake_session_factory=factory))
    task = CaseTask(step=Step.INTAKE, case_id="appt-9", idempotency_key="k-open",
                    payload={"opening": True})

    result = sup.submit(task)

    assert factory.calls == 1
    assert result.payload["message"] == "先说说它怎么了"


def test_no_guard_means_no_pretriage(store):
    factory = _Spy()
    sup = Supervisor(store, AgentContext(intake_session_factory=factory))

    result = sup.submit(_intake_task())

    assert factory.calls == 1
    assert result.verdict is Verdict.NONE


def test_draft_is_out_of_scope(store):
    """起草入参是结构化摘要，不在自由文本闸门范围内。"""
    from a2a.agents import AgentOutcome

    called = []

    def draft_agent(case, task, ctx):
        called.append(task.step)
        return AgentOutcome(status=TaskStatus.OK, payload={"draft": {"chief_complaint": "x"}})

    sup = Supervisor(store, AgentContext(guard=_Guard("emergency")),
                     agents={Step.DRAFT: draft_agent})
    task = CaseTask(step=Step.DRAFT, case_id="appt-9", idempotency_key="k-draft",
                    payload={"summary": {"chief_complaint": "吃了巧克力"}})

    result = sup.submit(task)

    assert called == [Step.DRAFT]
    assert result.verdict is Verdict.NONE

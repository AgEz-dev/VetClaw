"""tests/test_a2a_roles.py：C 批 —— 角色提示词 / 工具白名单 / 有限 handoff / 轨迹。

三道约束各有一组断言：

1. **提示词是文件、版本是内容哈希**：改 prompt 必然换版本，历史轨迹可分组对比；
   工具清单只存在代码里（``ROLE_TOOLS``），提示词里不抄一份，避免两处漂移。
2. **白名单默认拒绝**：未知角色看不到任何工具；越权调用在执行层抛
   ``PermissionError``，再被 ``execute_tool`` 转成模型可读的失败文本。
3. **handoff 是有限边**：只有 ``HANDOFF_EDGES`` 里的转移能落 case / 落轨迹，
   交接物只有 from / to / reason_code / case_seq / payload_digest，没有自然语言。
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

import hospital_tools  # noqa: E402,F401  导入即注册医院远端工具
import knowledge_tool  # noqa: E402,F401  导入即注册 search_knowledge_base
from a2a.agents import AgentContext, AgentOutcome  # noqa: E402
from a2a.contract import CaseState, CaseTask, Step, TaskResult, TaskStatus, Verdict, fingerprint  # noqa: E402
from a2a.roles import (  # noqa: E402
    HANDOFF_EDGES,
    NODE_DONE,
    NODE_FAILED,
    NODE_HUMAN,
    ROLE_CONSULT,
    ROLE_DRAFT,
    ROLE_INTAKE,
    ROLE_LABELS,
    ROLE_REVIEW,
    ROLE_TOOLS,
    ROLE_TRIAGE,
    HandoffDenied,
    RoleScopedRegistry,
    failed_handoff,
    is_allowed_handoff,
    make_handoff,
    outcome_handoff,
    role_prompt,
    role_prompt_version,
    tool_allowed,
)
from a2a.store import CaseStore  # noqa: E402
from a2a.stream import INTERNAL_OUTCOME  # noqa: E402
from a2a.supervisor import Supervisor  # noqa: E402
from a2a.trace import prompt_version  # noqa: E402
from case_context import get_case_id, get_role  # noqa: E402
from core_agent import execute_tool  # noqa: E402
from prescription_review import PrescriptionReviewer  # noqa: E402
from record_draft import build_draft_messages  # noqa: E402
from tools import ToolRegistry, registry as global_registry  # noqa: E402


# --------------------------------------------------------------------- 提示词

ROLE_PROMPTED = (ROLE_TRIAGE, ROLE_INTAKE, ROLE_DRAFT, ROLE_REVIEW)


def test_every_prompted_role_has_prompt_and_content_hash_version():
    assert set(ROLE_LABELS) == {ROLE_TRIAGE, ROLE_INTAKE, ROLE_DRAFT,
                                ROLE_REVIEW, ROLE_CONSULT}
    for role in ROLE_PROMPTED:
        text = role_prompt(role)
        assert text.strip(), f"{role} 角色提示词缺失"
        version = role_prompt_version(role)
        assert version == prompt_version(text)
        assert version.startswith("p_") and len(version) == 10
        assert "VetClaw" not in version, "版本号不能泄漏提示词正文"
    assert role_prompt(ROLE_CONSULT) == "", "会话位沿用引擎 system prompt，不另设角色文件"


def test_tool_names_live_only_in_code_not_prompts():
    """提示词写工具名约束不住模型，还会与白名单漂移 —— 清单一律只在代码里。"""
    for role in ROLE_PROMPTED:
        text = role_prompt(role)
        for tool_name in ROLE_TOOLS[role]:
            assert tool_name not in text, f"{role} 提示词不应抄工具名 {tool_name}"


def test_draft_prompt_is_the_single_source_for_record_draft():
    messages = build_draft_messages(
        {"chief_complaint": "呕吐 2 天"},
        [{"role": "user", "text": "昨天开始吐"}], None, None)
    system = messages[0]["content"]
    assert system == role_prompt(ROLE_DRAFT)
    assert "不得给诊断" in system and "不得给处方" in system


# --------------------------------------------------------------------- 工具白名单

def _probe_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def search_knowledge_base(query: str) -> str:
        """检索知识库。"""
        return "kb:" + query

    @reg.tool
    def lookup_drug_label(drug_name: str) -> str:
        """查药品说明书。"""
        return "label:" + drug_name

    @reg.tool
    def probe() -> str:
        """返回当前工具上下文。"""
        return f"{get_case_id()}:{get_role()}"

    return reg


def test_unknown_role_is_deny_by_default():
    assert not tool_allowed("admin", "search_knowledge_base")
    scoped = RoleScopedRegistry(_probe_registry(), "admin")
    assert scoped.schemas() == []
    with pytest.raises(PermissionError, match="无权调用工具"):
        scoped.call("search_knowledge_base", {"query": "x"})


def test_role_registry_hides_and_denies_out_of_scope_tool():
    scoped = RoleScopedRegistry(_probe_registry(), ROLE_TRIAGE)
    names = {s["function"]["name"] for s in scoped.schemas()}
    assert "search_knowledge_base" in names
    assert "lookup_drug_label" not in names, "模型不应看到白名单外的工具"

    assert scoped.call("search_knowledge_base", {"query": "百合"}) == "kb:百合"
    with pytest.raises(PermissionError, match="角色 triage 无权调用工具 lookup_drug_label"):
        scoped.call("lookup_drug_label", {"drug_name": "速诺"})
    with pytest.raises(PermissionError):
        scoped.get("lookup_drug_label")


def test_execute_tool_turns_role_denial_into_model_readable_failure():
    scoped = RoleScopedRegistry(_probe_registry(), ROLE_REVIEW)
    tc = SimpleNamespace(function=SimpleNamespace(
        name="search_knowledge_base", arguments='{"query":"x"}'))
    out = execute_tool(scoped, tc, "case-deny")
    assert out.startswith("工具执行失败: PermissionError")
    assert "无权调用工具" in out, "轨迹据此归类为 ROLE_DENIED"
    assert get_case_id() == "" and get_role() == "consult", "调用结束必须清理上下文"


def test_execute_tool_passes_role_into_tool_context():
    tc = SimpleNamespace(function=SimpleNamespace(name="probe", arguments="{}"))
    assert execute_tool(_probe_registry(), tc, "case-r", "review") == "case-r:review"


def test_every_whitelisted_tool_exists_in_real_registry():
    """白名单里的名字必须都能在真实注册表里找到，拼错即失败。"""
    registered = {s["function"]["name"] for s in global_registry.schemas()}
    for role, allowed in ROLE_TOOLS.items():
        missing = allowed - registered
        assert not missing, f"{role} 白名单引用了未注册工具: {sorted(missing)}"


def test_consult_whitelist_covers_read_tools_and_escalation():
    assert ROLE_TOOLS[ROLE_CONSULT] == {
        "search_knowledge_base", "lookup_drug_label", "check_drug_safety",
        "get_pet_profile", "get_vaccination_history", "get_medication_history",
        "get_inventory_summary", "escalate_to_human",
    }


# --------------------------------------------------------------------- handoff 边

def test_allowed_edges_cover_the_planned_paths():
    planned = {
        (ROLE_TRIAGE, ROLE_INTAKE, "NEED_MORE"),
        (ROLE_INTAKE, ROLE_TRIAGE, "SLOTS_READY"),
        (ROLE_TRIAGE, ROLE_DRAFT, "PASS"),
        (ROLE_DRAFT, NODE_HUMAN, "DRAFT_READY"),
        (NODE_HUMAN, ROLE_REVIEW, "RX_SUBMITTED"),
        (ROLE_REVIEW, NODE_HUMAN, "BLOCK"),
        (ROLE_REVIEW, NODE_HUMAN, "WARN"),
        (ROLE_REVIEW, NODE_DONE, "PASS"),
    }
    assert planned <= HANDOFF_EDGES
    for edge in planned:
        assert is_allowed_handoff(*edge)


def test_any_role_can_fail_but_unknown_role_cannot():
    for role in ROLE_LABELS:
        assert is_allowed_handoff(role, NODE_FAILED, "FAILED")
        assert failed_handoff(role, 3).to_role == NODE_FAILED
    assert not is_allowed_handoff("admin", NODE_FAILED, "FAILED")


def test_unregistered_edge_is_denied():
    for edge in [
        (ROLE_DRAFT, ROLE_REVIEW, "DRAFT_READY"),
        (ROLE_CONSULT, NODE_HUMAN, "PASS"),
        (ROLE_REVIEW, NODE_HUMAN, "PASS"),
        (ROLE_TRIAGE, ROLE_INTAKE, "PASS"),
    ]:
        assert not is_allowed_handoff(*edge)
        with pytest.raises(HandoffDenied, match="未登记的 handoff 边"):
            make_handoff(*edge, case_seq=0, payload={"query": "x"})


def test_handoff_carries_digest_only_no_case_text():
    payload = {"query": "猫误食百合花", "patient": {"name": "豆豆"}}
    handoff = make_handoff(ROLE_TRIAGE, ROLE_INTAKE, "NEED_MORE", 2, payload)
    body = handoff.to_dict()
    assert set(body) == {"from", "to", "reason_code", "case_seq", "payload_digest"}
    assert body["payload_digest"] == fingerprint(payload)
    assert len(body["payload_digest"]) == 64
    dumped = json.dumps(body, ensure_ascii=False)
    assert "百合" not in dumped and "豆豆" not in dumped


def test_outcome_handoff_mapping():
    need = outcome_handoff(Step.TRIAGE, TaskStatus.NEED_MORE, Verdict.PASS, {}, 1)
    assert (need.from_role, need.to_role, need.reason_code) == (
        ROLE_TRIAGE, ROLE_INTAKE, "NEED_MORE")

    go_draft = outcome_handoff(Step.TRIAGE, TaskStatus.OK, Verdict.PASS, {}, 1)
    assert (go_draft.from_role, go_draft.to_role) == (ROLE_TRIAGE, ROLE_DRAFT)

    for verdict in (Verdict.BLOCK, Verdict.WARN):
        to_human = outcome_handoff(Step.TRIAGE, TaskStatus.OK, verdict, {}, 1)
        assert (to_human.from_role, to_human.to_role) == (ROLE_TRIAGE, NODE_HUMAN)

    interrupted = outcome_handoff(Step.INTAKE, TaskStatus.OK, Verdict.NONE,
                                  {"interrupted": True}, 2)
    assert (interrupted.from_role, interrupted.to_role,
            interrupted.reason_code) == (ROLE_INTAKE, NODE_HUMAN, "EMERGENCY_INTERRUPTED")

    slots_ready = outcome_handoff(Step.INTAKE, TaskStatus.OK, Verdict.NONE,
                                  {"intake": {"chief_complaint": "呕吐"}}, 2)
    assert (slots_ready.from_role, slots_ready.to_role) == (ROLE_INTAKE, ROLE_TRIAGE)

    still_asking = outcome_handoff(Step.INTAKE, TaskStatus.OK, Verdict.NONE,
                                   {"questions": []}, 2)
    assert still_asking is None, "追问中的轮次不发生角色转移"

    draft_ready = outcome_handoff(Step.DRAFT, TaskStatus.OK, Verdict.NONE,
                                  {"draft": {"chief_complaint": "x"}}, 3)
    assert (draft_ready.from_role, draft_ready.to_role,
            draft_ready.reason_code) == (ROLE_DRAFT, NODE_HUMAN, "DRAFT_READY")
    assert outcome_handoff(Step.DRAFT, TaskStatus.OK, Verdict.NONE, {}, 3) is None

    for verdict, node in ((Verdict.BLOCK, NODE_HUMAN), (Verdict.WARN, NODE_HUMAN),
                          (Verdict.PASS, NODE_DONE)):
        reviewed = outcome_handoff(Step.REVIEW, TaskStatus.OK, verdict, {}, 4)
        assert reviewed.to_role == node and reviewed.reason_code == verdict.name

    failed = outcome_handoff(Step.REVIEW, TaskStatus.FAILED, Verdict.NONE, {}, 4)
    assert (failed.to_role, failed.reason_code) == (NODE_FAILED, "FAILED")


def test_case_handoff_dedupes_only_identical_tail():
    case = CaseState(case_id="c")
    handoff = make_handoff(ROLE_TRIAGE, ROLE_INTAKE, "NEED_MORE", 0, {"q": 1})
    case.record_handoff(handoff.to_dict())
    case.record_handoff(handoff.to_dict())
    assert len(case.handoffs) == 1, "同一交接重复写不得刷屏"
    assert case.seq == 0, "handoff 不是能力位结果，不得推进 seq"

    second = make_handoff(ROLE_TRIAGE, ROLE_DRAFT, "PASS", 0, {"q": 1})
    case.record_handoff(second.to_dict())
    assert [h["to"] for h in case.handoffs] == [ROLE_INTAKE, ROLE_DRAFT]


# --------------------------------------------------------------------- supervisor 集成

class _PassGuard:
    def check(self, query, patient_context=None):
        return {"action": "pass"}

    def emergency_message(self, toxin, species, chunk_id=None, slots=None):
        return "急症"

    def refuse_message(self, drug_hint):
        return "拒答"


def _store():
    return CaseStore(":memory:")


def _triage_result_for(sup, case_id, query, patient=None):
    payload = {"query": query}
    if patient:
        payload["patient"] = patient
    return sup.submit(CaseTask(step=Step.TRIAGE, case_id=case_id,
                               idempotency_key=f"idem-{case_id}", payload=payload))


def test_triage_missing_facts_hands_off_to_intake_with_trace():
    store = _store()
    sup = Supervisor(store, AgentContext(guard=_PassGuard()))
    result = _triage_result_for(sup, "case-missing", "狗狗最近食欲怎么样")

    assert result.status is TaskStatus.NEED_MORE
    assert set(result.payload["missing"]) == {"species", "weight_kg", "duration"}

    case = sup.get_case("case-missing")
    assert len(case.handoffs) == 1
    handoff = case.handoffs[0]
    assert (handoff["from"], handoff["to"], handoff["reason_code"]) == (
        ROLE_TRIAGE, ROLE_INTAKE, "NEED_MORE")
    assert handoff["case_seq"] == case.seq

    rows = store.list_trace("case-missing")
    assert [r["event"] for r in rows] == ["handoff", "verdict"]
    assert rows[0]["from_step"] == ROLE_TRIAGE
    assert rows[0]["to_step"] == ROLE_INTAKE
    assert rows[0]["reason_code"] == "NEED_MORE"
    assert rows[0]["payload_digest"] == handoff["payload_digest"]
    assert rows[0]["prompt_version"] == role_prompt_version(ROLE_TRIAGE)
    assert "食欲" not in json.dumps(rows, ensure_ascii=False)


def test_triage_complete_hands_off_to_draft():
    store = _store()
    sup = Supervisor(store, AgentContext(guard=_PassGuard()))
    result = _triage_result_for(sup, "case-complete", "狗狗吐了 2 天",
                                {"species": "犬", "weight_kg": 12})
    assert result.status is TaskStatus.OK and result.verdict is Verdict.PASS
    handoff = sup.get_case("case-complete").handoffs[-1]
    assert (handoff["from"], handoff["to"], handoff["reason_code"]) == (
        ROLE_TRIAGE, ROLE_DRAFT, "PASS")


def test_pretriage_block_hands_off_triage_to_human():
    class _EmergencyGuard(_PassGuard):
        def check(self, query, patient_context=None):
            return {"action": "emergency", "toxin": "百合", "species": "猫",
                    "chunk_id": "knowledge/species_drug_safety_matrix.md#百合"}

        def emergency_message(self, toxin, species, chunk_id=None, slots=None):
            return "请立即就医"

    store = _store()
    sup = Supervisor(store, AgentContext(guard=_EmergencyGuard()))

    def should_not_run(case, task, ctx):
        raise AssertionError("BLOCK 必须短路")

    sup.stream_agents = {Step.CONSULT: should_not_run}
    task = CaseTask(step=Step.CONSULT, case_id="case-block",
                    idempotency_key="idem-block", payload={"prompt": "猫吃了百合"})
    list(sup.submit_stream(task))

    case = sup.get_case("case-block")
    handoff = case.handoffs[-1]
    assert (handoff["from"], handoff["to"], handoff["reason_code"]) == (
        ROLE_TRIAGE, NODE_HUMAN, "BLOCK")
    rows = store.list_trace("case-block")
    assert [r["event"] for r in rows] == ["handoff", "verdict"]
    assert rows[0]["verdict"] == "block" and rows[1]["verdict"] == "block"


def test_draft_ready_hands_off_to_human_for_signature():
    store = _store()

    def fake_llm(messages):
        return ('{"chief_complaint":"呕吐 2 天","exam_checklist":[],'
                '"doctor_advice":"","disclaimer":"仅供参考"}')

    sup = Supervisor(store, AgentContext(llm=fake_llm))
    case = sup.open_case("case-draft", {"species": "犬", "weight_kg": 8})
    case.record(TaskResult(case_id="case-draft", task_id="t-in", step=Step.INTAKE,
                           status=TaskStatus.OK,
                           payload={"summary": {"chief_complaint": "呕吐 2 天"},
                                    "dialogue": [{"role": "user", "text": "昨天开始吐"}],
                                    "state": {"slots": {}, "turns": 1}}))
    store.save_case(case)

    result = sup.submit(CaseTask(step=Step.DRAFT, case_id="case-draft",
                                 idempotency_key="idem-draft", payload={}))
    assert result.status is TaskStatus.OK
    handoff = sup.get_case("case-draft").handoffs[-1]
    assert (handoff["from"], handoff["to"], handoff["reason_code"]) == (
        ROLE_DRAFT, NODE_HUMAN, "DRAFT_READY")

    rows = store.list_trace("case-draft")
    assert rows[0]["event"] == "handoff"
    assert rows[0]["prompt_version"] == role_prompt_version(ROLE_DRAFT)


@pytest.mark.parametrize("drug,expected_to,expected_reason", [
    ("速诺", NODE_DONE, "PASS"),
    ("对乙酰氨基酚", NODE_HUMAN, "BLOCK"),
])
def test_review_admits_human_then_hands_off(drug, expected_to, expected_reason):
    store = _store()
    sup = Supervisor(store, AgentContext(reviewer=PrescriptionReviewer()))
    result = sup.submit(CaseTask(
        step=Step.REVIEW, case_id=f"case-{drug}", idempotency_key=f"idem-{drug}",
        payload={"patient": {"species": "犬"},
                 "prescription": {"items": [{"drug_name": drug,
                                             "dose_mg_per_kg": 12.5}]}}))

    case = sup.get_case(f"case-{drug}")
    assert [(h["from"], h["to"], h["reason_code"]) for h in case.handoffs] == [
        (NODE_HUMAN, ROLE_REVIEW, "RX_SUBMITTED"),
        (ROLE_REVIEW, expected_to, expected_reason),
    ]
    assert result.verdict.name == expected_reason

    rows = store.list_trace(f"case-{drug}")
    assert [r["event"] for r in rows] == ["handoff", "handoff", "verdict"]
    assert rows[0]["from_step"] == NODE_HUMAN and rows[0]["prompt_version"] == ""
    assert rows[1]["prompt_version"] == role_prompt_version(ROLE_REVIEW)
    assert rows[2]["verdict"] == expected_reason.lower()


def test_agent_failure_hands_off_to_failed_before_error_trace():
    store = _store()

    def boom(case, task, ctx):
        raise RuntimeError("simulated")

    sup = Supervisor(store, AgentContext(), agents={Step.REVIEW: boom})
    result = sup.submit(CaseTask(
        step=Step.REVIEW, case_id="case-fail", idempotency_key="idem-fail",
        payload={"patient": {"species": "犬"},
                 "prescription": {"items": [{"drug_name": "速诺"}]}}))
    assert result.status is TaskStatus.FAILED

    case = sup.get_case("case-fail")
    assert (case.handoffs[-1]["from"], case.handoffs[-1]["to"],
            case.handoffs[-1]["reason_code"]) == (ROLE_REVIEW, NODE_FAILED, "FAILED")

    rows = store.list_trace("case-fail")
    assert [r["event"] for r in rows] == ["handoff", "handoff", "error"]
    assert rows[-2]["to_step"] == NODE_FAILED
    assert rows[-1]["error_code"] == "AGENT_ERROR"


def test_stream_handoff_and_tool_trace_share_role_version():
    """SSE 链路与同步链路写同一套字段：工具事件也要带角色提示词版本。"""
    store = _store()

    def streamer(case, task, ctx):
        yield {"event": "tool_call", "data": {
            "id": "c1", "name": "search_knowledge_base",
            "arguments": {"query": "犬呕吐"}}}
        yield {"event": "tool_result", "data": {
            "id": "c1", "content": json.dumps({"ok": True, "data": "..."})}}
        yield {"event": INTERNAL_OUTCOME, "data": AgentOutcome(
            status=TaskStatus.OK, payload={"answer": "ok"})}

    ctx = AgentContext(model="m", prompt_version="p_engine")
    sup = Supervisor(store, ctx=ctx,
                     stream_agents={Step.CONSULT: streamer})
    list(sup.submit_stream(CaseTask(
        step=Step.CONSULT, case_id="case-consult", idempotency_key="idem-c",
        payload={"prompt": "犬呕吐"})))

    rows = store.list_trace("case-consult")
    assert [r["event"] for r in rows] == ["tool_call", "tool_result", "verdict"]
    assert all(r["prompt_version"] == "p_engine" for r in rows), (
        "consult 没有角色文件，必须回落到引擎提示词版本而不是空串")


def test_graph_adapter_propagates_role_and_case():
    from graph_agent import GraphAgent

    class _StubGraph:
        init = None

        def stream(self, init, stream_mode="custom"):
            self.init = init
            yield {"event": "done", "data": {"answer": "ok"}}

    stub = _StubGraph()
    agent = GraphAgent(stub)
    assert agent.run("hi", case_id="case-g", role="review") == "ok"
    assert stub.init["case_id"] == "case-g"
    assert stub.init["role"] == "review", "LangGraph 引擎必须把角色带进工具节点"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("全部断言通过")

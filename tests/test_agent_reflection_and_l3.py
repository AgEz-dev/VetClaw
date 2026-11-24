import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from intake_flow import IntakeSession, detect_memory_conflict
from core_agent import critic_review


def test_l3_memory_conflict_detection():
    """测试 L3 档案冲突检测逻辑：青霉素过敏与慢性肾病。"""
    l3_profile = {
        "allergies": "青霉素过敏",
        "chronic_diseases": "慢性肾衰竭"
    }

    # 1. 触发过敏冲突
    conflicts_allergy = detect_memory_conflict("狗狗伤口发炎化脓，吃什么消炎药好？", l3_profile)
    assert len(conflicts_allergy) >= 1
    assert conflicts_allergy[0]["type"] == "ALLERGY_CONFLICT"
    assert "青霉素" in conflicts_allergy[0]["high_priority_question"]

    # 2. 触发慢性肾衰呕吐冲突
    conflicts_renal = detect_memory_conflict("猫咪一直吐黄水，已经两天不吃东西了", l3_profile)
    assert len(conflicts_renal) >= 1
    assert conflicts_renal[0]["type"] == "CHRONIC_RENAL_CONFLICT"
    assert "慢性肾功能不全" in conflicts_renal[0]["high_priority_question"]


def test_intake_session_preemption_on_conflict():
    """测试 IntakeSession 槽位被打断，高优插入 [历史档案风险提示] 追问。"""
    session = IntakeSession(l3_profile={"allergies": "青霉素类抗生素过敏"})
    session.opening()

    # 首轮输入包含感染/消炎药
    res = session.handle("我家狗打架耳朵咬破感染化脓了，想给它开点消炎药吃")
    assert res["done"] is False
    questions = res["questions"]
    assert len(questions) == 1
    # 验证打断槽位，首个问题为高危冲突追问
    assert questions[0]["key"] == "memory_conflict_preemption"
    assert "历史档案风险提示" in questions[0]["ask"]

    # 验证最终摘要中置顶注入医生警示
    snap = session.snapshot()
    ai_hints = snap.get("ai_hints", [])
    assert any("过敏史" in h.get("text", "") for h in ai_hints)


def test_critic_review_user_diagnosis_correction():
    """测试 Critic 守卫对宠主端越权确诊与处方剂量的修正。"""
    raw_answer = "根据描述，我确诊为你得了犬细小病毒，建议口服阿莫西林 500mg 治疗。"
    passed, corrected, details = critic_review(raw_answer, role="USER")
    assert passed is False
    assert "宠主端越权确诊" in details["violations"]
    assert "宠主端开具具体处方药剂量" in details["violations"]
    assert "确诊为你得了" not in corrected
    assert "500mg" not in corrected
    assert "非确诊，须执业兽医面诊" in corrected


def test_critic_review_doctor_potassium_speed_limit():
    """测试 Critic 守卫对医生端补液方案遗漏补钾上限的自动注入。"""
    raw_doctor_plan = "严重细小病毒休克期，立即静脉注射乳酸林格氏液进行扩容，并给予氯化钾纠正代酸。"
    passed, corrected, details = critic_review(raw_doctor_plan, role="DOCTOR")
    assert passed is False
    assert any("补钾速度安全上限" in v for v in details["violations"])
    assert "0.5" in corrected
    assert "mEq/kg/hour" in corrected


def test_critic_review_hallucinated_citation_cleanup():
    """测试 Critic 守卫对召回结果中不存在的虚假引用角标的剔除。"""
    raw_text = "根据最新文献【来源: knowledge/fake_paper.md#fake-01】，建议使用新型止吐药。"
    valid_chunks = [
        {"chunk_id": "knowledge/outpatient_clinical_guidelines.md#呕吐鉴别-1", "content": "..."}
    ]
    passed, corrected, details = critic_review(raw_text, role="DOCTOR", retrieved_chunks=valid_chunks)
    assert passed is False
    assert any("引用幻觉切片" in v for v in details["violations"])
    assert "fake_paper.md" not in corrected

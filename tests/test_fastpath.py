"""tests/test_fastpath.py：FastPathGuard 5 组基准验证。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastpath import FastPathGuard


def make_guard():
    return FastPathGuard()


def test_01_chitchat_lily_not_emergency():
    g = make_guard()
    assert g.check("猫能吃百合吗")["action"] == "pass"


def test_02_probiotic_not_false_refuse():
    g = make_guard()
    assert g.check("狗吃益生菌多少")["action"] == "pass"


def test_03_grape_emergency_no_le():
    """狗吃葡萄（无"了"）也必须拦截——宁严勿漏。"""
    g = make_guard()
    r = g.check("狗吃葡萄")
    assert r["action"] == "emergency"
    assert r["toxin"] == "葡萄"


def test_04_whitelist_drug_pass():
    g = make_guard()
    assert g.check("大宠爱多大能用")["action"] == "pass"


def test_05_unlisted_drug_refuse():
    g = make_guard()
    r = g.check("阿莫西林给猫吃多少")
    assert r["action"] == "refuse"
    assert r["drug_hint"] == "阿莫西林"


def test_06_grape_with_le():
    g = make_guard()
    r = g.check("狗吃了葡萄怎么办")
    assert r["action"] == "emergency"


def test_07_can_eat_pattern_not_emergency():
    g = make_guard()
    assert g.check("狗可以吃葡萄吗")["action"] == "pass"


def test_core_agent_p0_emergency_calls_guard_with_chunk_id():
    """回归：经 core_agent 触发 P0 急症时，chunk_id 正确透传至 emergency_message，
    返回 SOP + 毒物详情拼接文本，且全程不调 LLM；chunk_id=None 时优雅降级不抛异常。"""
    from core_agent import ReActAgent
    from tools import ToolRegistry

    fetched = []
    docs = {
        "knowledge/species_drug_safety_matrix.md#现场急救通用-SOP-所有毒物中毒通用-0":
            "SOP：立即脱离毒源并催吐送医",
        "knowledge/species_drug_safety_matrix.md#葡萄与葡萄干-6":
            "葡萄导致狗急性肾衰竭，误食后 6 小时内为黄金洗胃窗口",
    }

    class FakeCollection:
        def get(self, ids):
            fetched.extend(ids)
            return {"documents": [docs[i] for i in ids]}

    class FakePipeline:
        collection = FakeCollection()

    class NoLLMClient:
        pass  # P0 路径不应触碰 client，任何属性访问都会 AttributeError

    guard = FastPathGuard(pipeline=FakePipeline())
    agent = ReActAgent(NoLLMClient(), ToolRegistry(), model="fake",
                       fastpath_guard=guard)
    answer = agent.run("狗吃了葡萄怎么办")

    # check() 命中 emergency 必须携带 chunk_id，且为规则表中的葡萄 chunk
    grape_chunk = [t for t in guard.rules["p0_toxins"]
                   if t["name"] == "葡萄"][0]["chunk_id"]
    assert guard.check("狗吃了葡萄怎么办")["chunk_id"] == grape_chunk
    # 点查顺序：先 SOP 后毒物详情，证明 chunk_id 正确透传
    assert fetched == [guard.rules["sop_chunk_id"], grape_chunk]
    # 拼接文本完整：预警头 + SOP + 详情 + 免责声明
    assert answer.startswith("【P0 急症预警】")
    assert "SOP：立即脱离毒源并催吐送医" in answer
    assert "葡萄导致狗急性肾衰竭" in answer
    assert "仅供参考" in answer

    # 防御降级：chunk_id 为 None 时不抛异常，退化为通用警报（无详情段）
    generic = guard.emergency_message("不明毒物", "猫", None)
    assert generic.startswith("【P0 急症预警】")
    assert "关于不明毒物的详细信息" not in generic


if __name__ == "__main__":
    for f in [test_01_chitchat_lily_not_emergency,
              test_02_probiotic_not_false_refuse,
              test_03_grape_emergency_no_le,
              test_04_whitelist_drug_pass,
              test_05_unlisted_drug_refuse,
              test_06_grape_with_le,
              test_07_can_eat_pattern_not_emergency,
              test_core_agent_p0_emergency_calls_guard_with_chunk_id]:
        f()
    print("全部断言通过")

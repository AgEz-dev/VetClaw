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


if __name__ == "__main__":
    for f in [test_01_chitchat_lily_not_emergency,
              test_02_probiotic_not_false_refuse,
              test_03_grape_emergency_no_le,
              test_04_whitelist_drug_pass,
              test_05_unlisted_drug_refuse,
              test_06_grape_with_le,
              test_07_can_eat_pattern_not_emergency]:
        f()
    print("全部断言通过")

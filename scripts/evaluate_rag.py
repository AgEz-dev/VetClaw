"""RAG 基线评估脚本：Recall@3 / MRR@3 / 拒答触发率 / 急症毒物一致性 / 关键词断言。

用法：
    python -m scripts.evaluate_rag

判定规则（v3，2026-10-09）：
  1. 拒答用例 —— 规则化判定：守卫 action == "refuse" 才算 PASS。
  2. 急症用例 —— 守卫须命中 emergency，且 chunk_id 必须等于用例的
     expected_source_chunk（毒物一致性断言）。
  3. 检索用例 —— expected chunk 命中 top3 计入 Recall，再断言 must_contain_keywords。
  4. **expected_fail 用例（v3 新增）**：标记为已知失败的用例（如守卫物种盲区）——
     失败记为 XFAIL（不计入任何指标分母），意外通过记为 XPASS（须当场核查原因）。
     旧 50 组的判定语义与 v2 完全一致。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_pipeline import RAGPipeline, BGEEmbeddingFunction
from fastpath import FastPathGuard


def judge(case, gp, pipe):
    """对单条用例判定，返回 (status, detail, rank)。

    status ∈ PASS / FAIL / MISS / GUARD-MISS / GUARD-MISMATCH / KW-FAIL / FALSE-GUARD
    rank 仅检索命中时非 None。
    """
    expect = case["expected_behavior"]
    cat = case.get("category", "")
    expected_chunk = case.get("expected_source_chunk")
    action = gp.get("action")

    if expect == "refuse":
        if action == "refuse":
            return "PASS", f"guard refused: {gp.get('drug_hint')}", None
        return "FAIL", f"expected refuse, guard action={action}", None

    if cat == "emergency":
        if action != "emergency":
            return "GUARD-MISS", f"guard action={action}, expected emergency", None
        if gp.get("chunk_id") != expected_chunk:
            return "GUARD-MISMATCH", f"toxin={gp.get('toxin')} chunk != expected", None
        return "PASS", f"P0 emergency: {gp.get('toxin')}", None

    if action != "pass":
        return "FALSE-GUARD", f"guard action={action} blocked an answer case", None

    results = pipe.search(case["query"], top_k=3)
    found = [r for r in results if r["chunk_id"] == expected_chunk]
    if not found:
        return "MISS", "expected chunk not in top3", None

    rank = [r["chunk_id"] for r in results].index(expected_chunk) + 1
    content = results[rank - 1].get("content", "")
    missing_kw = [kw for kw in case.get("must_contain_keywords", [])
                  if kw not in content]
    if missing_kw:
        return "KW-FAIL", f"rank={rank} missing={missing_kw}", rank
    return "PASS", f"rank={rank}", rank


def main():
    ds = json.loads(Path("tests/eval_dataset.json").read_text(encoding="utf-8"))
    pipe = RAGPipeline("data/chroma", "knowledge",
                       embedding_function=BGEEmbeddingFunction())
    guard = FastPathGuard()

    hits = 0                 # chunk 命中 top3 的检索用例数
    reciprocal_sum = 0.0
    kw_fail = 0              # chunk 命中但关键词缺失
    refuse_pass = 0
    refuse_total = 0
    emergency_pass = 0
    emergency_total = 0
    guard_mismatch = 0       # 急症漏命中 / 毒物 chunk 不一致
    false_guard = 0          # answer 用例被守卫误拦
    rag_answer_cases = 0
    xfail_ok = 0             # 已知失败：如预期失败
    xpass = 0                # 已知失败用例意外通过（须核查）
    rows = []

    for case in ds["cases"]:
        q = case["query"]
        gp = guard.check(q)
        status, detail, rank = judge(case, gp, pipe)

        # ---- expected_fail：已知失败用例，单列不进指标 ----
        if case.get("expected_fail"):
            if status == "PASS":
                xpass += 1
                rows.append((case["id"], "XPASS",
                             "预期失败但意外通过，请核查守卫是否已修复: " + detail, q))
            else:
                xfail_ok += 1
                rows.append((case["id"], "XFAIL",
                             f"已知失败[{case.get('fail_reason', '')}] {detail}", q))
            continue

        cid = case["id"]
        if case["expected_behavior"] == "refuse":
            refuse_total += 1
            if status == "PASS":
                refuse_pass += 1
            rows.append((cid, status, detail, q))
            continue

        if case.get("category") == "emergency":
            emergency_total += 1
            if status == "PASS":
                emergency_pass += 1
            else:
                guard_mismatch += 1
            rows.append((cid, status, detail, q))
            continue

        # 检索用例
        if status == "FALSE-GUARD":
            false_guard += 1
            rows.append((cid, status, detail, q))
            continue

        rag_answer_cases += 1
        if rank:
            hits += 1
            reciprocal_sum += 1.0 / rank
        if status == "KW-FAIL":
            kw_fail += 1
        rows.append((cid, status, detail, q))

    # ---------- 报告 ----------
    print("=" * 70)
    print("VetClaw RAG 基线评估报告（BGE-small-zh-v1.5）")
    print("=" * 70)
    for cid, status, detail, q in rows:
        mark = {"PASS": "✅", "XFAIL": "⚪", "XPASS": "🎉"}.get(status, "❌")
        print(f"{mark} {cid}  [{status:13}] {detail:44} | {q}")
    print("-" * 70)
    recall = hits / rag_answer_cases if rag_answer_cases else 0
    mrr = reciprocal_sum / rag_answer_cases if rag_answer_cases else 0
    refuse_rate = refuse_pass / refuse_total if refuse_total else 0
    em_rate = emergency_pass / emergency_total if emergency_total else 0
    print(f"检索用例 (RAG):  {rag_answer_cases}")
    print(f"Recall@3:       {recall:.2%}  ({hits}/{rag_answer_cases})")
    print(f"MRR@3:          {mrr:.3f}")
    print(f"关键词断言:      {'全部通过' if kw_fail == 0 else f'{kw_fail} 条 KW-FAIL'}")
    print(f"急症用例:        {emergency_pass}/{emergency_total} 毒物 chunk 一致 ({em_rate:.2%})")
    print(f"拒答用例:        {refuse_pass}/{refuse_total} 守卫命中 ({refuse_rate:.2%})")
    if xfail_ok or xpass:
        print(f"已知失败 (XFAIL): {xfail_ok}   意外通过 (XPASS): {xpass}")
    if guard_mismatch:
        print(f"⚠ 守卫异常:      {guard_mismatch} 条（急症漏命中 / 毒物不一致）")
    if false_guard:
        print(f"⚠ 误拦:         {false_guard} 条（answer 用例被守卫拦截）")
    print("=" * 70)


if __name__ == "__main__":
    main()

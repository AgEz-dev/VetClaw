"""RAG 基线评估脚本：Recall@3 / MRR@3 / 拒答触发率 / 急症毒物一致性 / 关键词断言。

用法：
    python -m scripts.evaluate_rag

判定规则（v2，2026-10-07 修正三处缺陷）：
  1. 拒答用例 —— **规则化判定**：守卫 action == "refuse" 才算 PASS，
     不再依赖「top3 距离全 > MAX_DISTANCE」这一脆弱启发式。
  2. 急症用例 —— 守卫须命中 emergency，**且其 chunk_id 必须等于用例的
     expected_source_chunk**（毒物一致性断言），杜绝"任意 emergency 都判 PASS"的假过。
  3. 检索用例 —— expected chunk 命中 top3 计入 Recall；再断言
     must_contain_keywords 全部出现在命中 chunk 中（原 kw_ok 死代码已启用）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_pipeline import RAGPipeline, BGEEmbeddingFunction
from fastpath import FastPathGuard


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
    rows = []

    for case in ds["cases"]:
        q = case["query"]
        expect = case["expected_behavior"]
        cat = case.get("category", "")
        expected_chunk = case.get("expected_source_chunk")
        gp = guard.check(q)
        action = gp.get("action")

        # ---------- 1) 拒答用例：规则化判定 ----------
        if expect == "refuse":
            refuse_total += 1
            if action == "refuse":
                refuse_pass += 1
                rows.append((case["id"], "PASS", f"guard refused: {gp.get('drug_hint')}", q))
            else:
                rows.append((case["id"], "FAIL",
                             f"expected refuse, guard action={action}", q))
            continue

        # ---------- 2) 急症用例：守卫命中 + 毒物 chunk 一致 ----------
        if cat == "emergency":
            emergency_total += 1
            if action != "emergency":
                guard_mismatch += 1
                rows.append((case["id"], "GUARD-MISS",
                             f"guard action={action}, expected emergency", q))
            elif gp.get("chunk_id") != expected_chunk:
                guard_mismatch += 1
                rows.append((case["id"], "GUARD-MISMATCH",
                             f"toxin={gp.get('toxin')} chunk != expected", q))
            else:
                emergency_pass += 1
                rows.append((case["id"], "PASS", f"P0 emergency: {gp.get('toxin')}", q))
            continue

        # ---------- 3) 检索用例：RAG 检索 + 关键词断言 ----------
        if action != "pass":
            false_guard += 1
            rows.append((case["id"], "FALSE-GUARD",
                         f"guard action={action} blocked an answer case", q))
            continue

        rag_answer_cases += 1
        results = pipe.search(q, top_k=3)
        found = [r for r in results if r["chunk_id"] == expected_chunk]
        if not found:
            rows.append((case["id"], "MISS", "expected chunk not in top3", q))
            continue

        rank = [r["chunk_id"] for r in results].index(expected_chunk) + 1
        hits += 1
        reciprocal_sum += 1.0 / rank

        content = results[rank - 1].get("content", "")
        missing_kw = [kw for kw in case.get("must_contain_keywords", [])
                      if kw not in content]
        if missing_kw:
            kw_fail += 1
            rows.append((case["id"], "KW-FAIL", f"rank={rank} missing={missing_kw}", q))
        else:
            rows.append((case["id"], "PASS", f"rank={rank}", q))

    # ---------- 报告 ----------
    print("=" * 70)
    print("VetClaw RAG 基线评估报告（BGE-small-zh-v1.5）")
    print("=" * 70)
    for cid, status, detail, q in rows:
        mark = "✅" if status == "PASS" else "❌"
        print(f"{mark} {cid}  [{status:13}] {detail:34} | {q}")
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
    if guard_mismatch:
        print(f"⚠ 守卫异常:      {guard_mismatch} 条（急症漏命中 / 毒物不一致）")
    if false_guard:
        print(f"⚠ 误拦:         {false_guard} 条（answer 用例被守卫拦截）")
    print("=" * 70)


if __name__ == "__main__":
    main()

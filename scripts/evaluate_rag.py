"""RAG 基线评估脚本：Recall@3 / MRR@3 / 拒答触发率。

用法：
    python -m scripts.evaluate_rag
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_pipeline import RAGPipeline, BGEEmbeddingFunction

MAX_DISTANCE = 0.45  # cosine 距离 > 此值视为未命中


def main():
    ds = json.loads(Path("tests/eval_dataset.json").read_text(encoding="utf-8"))
    pipe = RAGPipeline("data/chroma", "knowledge",
                       embedding_function=BGEEmbeddingFunction())

    hits = 0          # Recall@3 分子
    reciprocal_sum = 0.0
    refuse_pass = 0
    total = len(ds["cases"])
    rows = []

    for case in ds["cases"]:
        q = case["query"]
        expect = case["expected_behavior"]
        results = pipe.search(q, top_k=3)

        if expect == "answer":
            expected = case["expected_source_chunk"]
            found = [r for r in results if r["chunk_id"] == expected]
            if found:
                rank = [r["chunk_id"] for r in results].index(expected) + 1
                hits += 1
                reciprocal_sum += 1.0 / rank
                # 关键词检查
                kw_ok = all(kw in results[rank-1]["content"] for kw in case["must_contain_keywords"])
                rows.append((case["id"], "PASS", f"rank={rank}", q))
            else:
                rank = "-"
                rows.append((case["id"], "MISS", "expected chunk not in top3", q))
        else:
            # refuse case：top3 里所有 distance 都 > MAX_DISTANCE 视为未命中
            all_far = all(r.get("distance", 1.0) > MAX_DISTANCE for r in results)
            if all_far:
                refuse_pass += 1
                rows.append((case["id"], "PASS", "correctly refused", q))
            else:
                top = results[0] if results else {}
                rows.append((case["id"], "FALSE-HIT",
                             f"top1 dist={top.get('distance',0):.2f}", q))

    # 打印报告
    print("=" * 70)
    print("VetClaw RAG 基线评估报告（BGE-small-zh-v1.5）")
    print("=" * 70)
    for cid, status, detail, q in rows:
        mark = "✅" if status == "PASS" else "❌"
        print(f"{mark} {cid}  [{status:8}] {detail:28} | {q}")
    print("-" * 70)
    answer_cases = [c for c in ds["cases"] if c["expected_behavior"] == "answer"]
    refuse_cases = [c for c in ds["cases"] if c["expected_behavior"] == "refuse"]
    recall = hits / len(answer_cases) if answer_cases else 0
    mrr = reciprocal_sum / len(answer_cases) if answer_cases else 0
    refuse_rate = refuse_pass / len(refuse_cases) if refuse_cases else 0
    print(f"Answer cases:  {len(answer_cases)}")
    print(f"Recall@3:      {recall:.2%}  ({hits}/{len(answer_cases)})")
    print(f"MRR@3:         {mrr:.3f}")
    print(f"Refuse cases:  {len(refuse_cases)}")
    print(f"Refuse rate:   {refuse_rate:.2%}  ({refuse_pass}/{len(refuse_cases)})")
    print("=" * 70)


if __name__ == "__main__":
    main()

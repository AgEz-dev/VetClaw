"""知识库入库脚本：扫描 knowledge/*.md 写入 ChromaDB。

用法：
    python -m scripts.ingest_docs              # 默认 upsert（同 id 覆盖，幂等）
    python -m scripts.ingest_docs --rebuild     # 显式重建：drop collection 后全量入库
    python -m scripts.ingest_docs --db data/chroma --collection knowledge
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_pipeline import RAGPipeline, BGEEmbeddingFunction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/chroma")
    ap.add_argument("--collection", default="knowledge")
    ap.add_argument("--dir", default="knowledge")
    ap.add_argument("--rebuild", action="store_true", help="drop collection 后全量重建")
    args = ap.parse_args()

    kb_dir = Path(args.dir)
    if not kb_dir.exists():
        print(f"[!] 目录不存在：{kb_dir}")
        sys.exit(1)

    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        print(f"[!] {kb_dir} 下没有 .md 文件")
        sys.exit(1)

    ef = BGEEmbeddingFunction()
    pipe = RAGPipeline(args.db, args.collection, embedding_function=ef)

    if args.rebuild:
        print(f"[rebuild] drop collection: {args.collection}")
        pipe.reset()

    total = 0
    for f in md_files:
        n = pipe.ingest(str(f))
        print(f"  {f.name}: {n} chunks")
        total += n

    print(f"[done] 共 {len(md_files)} 个文件，{total} chunks，collection 总数 {pipe.count()}")


if __name__ == "__main__":
    main()

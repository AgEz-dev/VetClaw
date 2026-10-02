"""轻量 RAG 流水线：递归切片 + ChromaDB 持久化检索（零重型包装）。"""
from pathlib import Path

import chromadb

SEPARATORS = ["\n\n", "\n", "。", " "]


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """递归字符切片：优先在段落/行/句号边界切，相邻块带 chunk_overlap 重叠。"""
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in SEPARATORS:
                pos = text.rfind(sep, start, end)
                if pos >= start + chunk_size // 2:  # 块足够大才在边界切，避免碎块
                    end = pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        # ★ 修复：start 必须严格递增；end 过小（开头附近就有分隔符）时至少前进 1
        start = max(end - chunk_overlap, start + 1)
    return [c for c in chunks if c]


class RAGPipeline:
    def __init__(self, db_path: str = "data/chroma", collection: str = "knowledge",
                 embedding_function=None):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            collection,
            embedding_function=embedding_function,  # None → chroma 内置 ONNX 模型
            metadata={"hnsw:space": "cosine"},      # 余弦距离，文本检索更合理
        )

    def ingest(self, file_path: str) -> int:
        """读取 md/txt → 切片 → upsert 入库；返回新增 chunk 数。"""
        path = Path(file_path)
        chunks = split_text(path.read_text(encoding="utf-8"))
        if not chunks:  # 空文件/纯空白：无内容可入库，直接返回 0，避免空 upsert 报错
            return 0
        ids = [f"{path.as_posix()}#{i}" for i in range(len(chunks))]
        metas = [{"source": path.as_posix(), "index": i} for i in range(len(chunks))]
        self.collection.upsert(ids=ids, documents=chunks, metadatas=metas)
        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """语义检索，返回 content/source/chunk_id/distance 元数据片段。"""
        res = self.collection.query(query_texts=[query], n_results=top_k)
        return [
            {"content": res["documents"][0][i],
             "source": res["metadatas"][0][i]["source"],
             "chunk_id": res["ids"][0][i],
             "distance": res["distances"][0][i]}
            for i in range(len(res["documents"][0]))
        ]

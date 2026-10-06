"""轻量 RAG 流水线：结构感知 Markdown 切片 + ChromaDB 持久化检索。

- split_text（旧，保留兼容）：递归字符切片，测试用
- split_markdown（新）：按 ## 二级标题分块，表格整体保护
- BGEEmbeddingFunction：本地 BGE-small-zh-v1.5，query/doc 前缀分离
"""
import re
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction

SEPARATORS = ["\n\n", "\n", "。", " "]


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """旧递归字符切片：保留不动，兼容现有单元测试。"""
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in SEPARATORS:
                pos = text.rfind(sep, start, end)
                if pos >= start + chunk_size // 2:
                    end = pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return [c for c in chunks if c]


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|")


def _slug(title: str) -> str:
    return re.sub(r"[^\w一-鿿]+", "-", title.strip()).strip("-") or "root"


def split_markdown(text: str, max_chunk: int = 800) -> list[dict]:
    """按 ## 二级标题分块；连续 | 表格行作为不可分割整体。

    返回 [{"title": "...", "content": "..."}]。
    - # 一级标题：作为文档级标题，不参与切分
    - ## 二级标题：主体切分点
    - ### 三级标题：留在所属 ## 块内
    - 无 ## 时整篇作为一个块（降级）
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    # 提取一级标题作为文档级标题
    doc_title = ""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        doc_title = m.group(1).strip()

    # 按 ## 二级标题切分
    sections = re.split(r"\n(?=##\s+)", text)
    chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        # 提取该 section 的标题（## 开头那行）
        hm = re.match(r"^##\s+(.+)$", sec, re.MULTILINE)
        title = hm.group(1).strip() if hm else (doc_title or "未命名章节")
        body = sec[hm.end():].strip() if hm else sec

        # section 内部逐行切分：表格整块保护，普通段落按 max_chunk 累积
        sub_chunks = _split_body(body, title, max_chunk)
        chunks.extend(sub_chunks)

    return chunks


def _split_body(body: str, title: str, max_chunk: int) -> list[dict]:
    lines = body.split("\n")
    results = []
    buf_lines = []
    buf_len = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_table_line(line):
            # 先把累积的普通段落 flush
            if buf_lines:
                results.append({"title": title, "content": "\n".join(buf_lines).strip()})
                buf_lines, buf_len = [], 0
            # 吞掉连续表格行（含表头分隔行）
            table = [line]
            j = i + 1
            while j < len(lines) and _is_table_line(lines[j]):
                table.append(lines[j])
                j += 1
            results.append({"title": title, "content": "\n".join(table).strip()})
            i = j
        else:
            buf_lines.append(line)
            buf_len += len(line) + 1
            if buf_len >= max_chunk:
                results.append({"title": title, "content": "\n".join(buf_lines).strip()})
                buf_lines, buf_len = [], 0
            i += 1
    if buf_lines:
        results.append({"title": title, "content": "\n".join(buf_lines).strip()})
    return [c for c in results if c["content"]]


class BGEEmbeddingFunction(EmbeddingFunction):
    """本地 BGE-small-zh-v1.5 向量化（懒加载，断网可用）。

    - embed_documents：入库时调用，文本不加前缀
    - embed_query：检索时调用，文本前加官方中文指令
    - __call__：兼容 chroma 默认协议，按 documents 处理
    """

    QUERY_INSTRUCTION = "为这个句子生成表示用于检索相关文章："

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def __call__(self, input):
        return self.embed_documents(list(input))

    def embed_documents(self, texts):
        self._ensure_model()
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text: str):
        self._ensure_model()
        vec = self._model.encode(self.QUERY_INSTRUCTION + text, normalize_embeddings=True)
        return vec.tolist()


class RAGPipeline:
    def __init__(self, db_path: str = "data/chroma", collection: str = "knowledge",
                 embedding_function=None):
        self.client = chromadb.PersistentClient(path=db_path)
        self._ef = embedding_function
        self.collection = self.client.get_or_create_collection(
            collection,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, file_path: str) -> int:
        """读取 md/txt → split_markdown → upsert 入库；返回新增 chunk 数。"""
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        sections = split_markdown(text)
        if not sections:
            return 0
        ids, docs, metas = [], [], []
        for i, sec in enumerate(sections):
            chunk_id = f"{path.as_posix()}#{_slug(sec['title'])}-{i}"
            ids.append(chunk_id)
            docs.append(sec["content"])
            metas.append({"source": path.as_posix(), "index": i,
                          "title": sec["title"]})
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(sections)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """语义检索：BGE 模型对 query 加前缀后用 query_embeddings 检索。"""
        # 用 QUERY_INSTRUCTION 属性区分 BGE（需要 query 前缀）vs Fake/其他（走 query_texts）
        ef = self._ef
        if ef is not None and getattr(ef, "QUERY_INSTRUCTION", None):
            vec = ef.embed_query(query)
            res = self.collection.query(query_embeddings=[vec], n_results=top_k)
        else:
            res = self.collection.query(query_texts=[query], n_results=top_k)
        return [
            {"content": res["documents"][0][i],
             "source": res["metadatas"][0][i]["source"],
             "chunk_id": res["ids"][0][i],
             "distance": res["distances"][0][i]}
            for i in range(len(res["documents"][0]))
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """drop collection（仅 --rebuild 时调用）。"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            self.collection.name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

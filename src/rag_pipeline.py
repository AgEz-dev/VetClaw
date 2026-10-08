"""轻量 RAG 流水线：结构感知 Markdown 切片 + ChromaDB 持久化检索。

- split_text（旧，保留兼容）：递归字符切片，测试用
- split_markdown（新）：按 ## 二级标题分块，表格整体保护 + 表格 chunk 上下文化
- BGEEmbeddingFunction：本地 BGE-small-zh-v1.5，query/doc 前缀分离
"""
import re
import time
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction

SEPARATORS = ["\n\n", "\n", "。", " "]

# 医疗专有名词词典（模块级常量，幂等注册一次）
_MED_TERMS = ["对乙酰氨基酚", "塞拉菌素", "莫昔克丁", "吡虫啉",
              "大宠爱", "爱沃克", "木糖醇", "葡萄", "百合",
              "驱虫药", "急救", "送医", "误食"]


def _ensure_jieba_terms():
    """幂等注册医疗专有名词到 jieba 词典（只注册一次，重复调用无副作用）。"""
    try:
        import jieba
    except ImportError:
        return
    for w in _MED_TERMS:
        jieba.add_word(w)


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


# 表格上下文化：表格 chunk 自身不含章节标题与实体名（如「福来恩」只出现在同节正文里），
# 直接入库会让它在稠密与稀疏检索上都"语义不可见"。这里给表格 chunk 补一段上下文头部。
# ⚠️ 只改 content，不改切片边界与顺序 → chunk_id 保持稳定。
TABLE_CONTEXT_MAX = 150


def _last_paragraph(text: str) -> str:
    """取文本最后一段非引用块正文（跳过 `>` 出处/免责声明），单行化后截断。"""
    if not text:
        return ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for para in reversed(paras):
        if para.startswith(">"):
            continue
        flat = " ".join(ln.strip() for ln in para.splitlines() if ln.strip())
        if flat:
            return flat[:TABLE_CONTEXT_MAX]
    return ""


def _table_context(title: str, lead_in: str) -> str:
    """表格 chunk 的上下文化头部：所属章节标题（+ 紧邻其前的正文首段）。"""
    head = [f"## {title}"]
    para = _last_paragraph(lead_in)
    if para:
        head.append(para)
    return "\n".join(head)


def split_markdown(text: str, max_chunk: int = 800) -> list[dict]:
    """按 ## 二级标题分块；连续 | 表格行作为不可分割整体。

    返回 [{"title": "...", "content": "..."}]。
    - # 一级标题：作为文档级标题，不参与切分
    - 文件开头的 > 引用块（数据来源/免责声明）：剥离，不单独成 chunk
    - ## 二级标题：主体切分点
    - ### 三级标题：留在所属 ## 块内
    - 无 ## 时整篇作为一个块（降级）
    - ⚠️ 表格 chunk（连续 `|` 行）会在 content 前补一段上下文化头部：
      `## <章节标题>` + 紧邻其前的正文首段（截断 150 字）。
      原因是表格本身常不含实体名（如「福来恩」只在同节正文里），
      不加头部会让表格 chunk 在稠密/稀疏检索上都"语义不可见"。
      **只改 content，不增删 chunk、不改下标 → chunk_id 保持稳定。**
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    # 提取一级标题作为文档级标题
    doc_title = ""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        doc_title = m.group(1).strip()

    # 剥离文件开头的 > 引用块（数据来源/免责声明），不参与切片
    text = re.sub(r"^#\s+.+\n+", "", text, count=1)  # 去掉一级标题行
    text = re.sub(r"^(>\s.*\n?)+", "", text).strip()  # 去掉连续 > 引用块

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
            # 先把累积的普通段落 flush（内容逐字节保持原样，切片边界不动）
            # lead_in 即"紧邻表格之前的正文"，稍后用于给表格 chunk 补上下文
            lead_in = "\n".join(buf_lines).strip()
            if buf_lines:
                results.append({"title": title, "content": lead_in})
                buf_lines, buf_len = [], 0
            # 吞掉连续表格行（含表头分隔行）
            table = [line]
            j = i + 1
            while j < len(lines) and _is_table_line(lines[j]):
                table.append(lines[j])
                j += 1
            table_body = "\n".join(table).strip()
            results.append({"title": title,
                            "content": f"{_table_context(title, lead_in)}\n\n{table_body}"})
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
        self._build_bm25()

    def _build_bm25(self):
        """从 ChromaDB 加载全量文档，构建 BM25 稀疏索引。"""
        t0 = time.monotonic()
        try:
            from rank_bm25 import BM25Okapi
            import jieba
        except ImportError:
            self._bm25 = None
            self._bm25_ids = []
            self._jieba = None
            return
        _ensure_jieba_terms()
        res = self.collection.get()
        self._bm25_docs = res["documents"]
        self._bm25_ids = res["ids"]
        corpus_tokens = [list(jieba.cut(d)) for d in self._bm25_docs]
        try:
            self._bm25 = BM25Okapi(corpus_tokens)
        except (ZeroDivisionError, ValueError):
            self._bm25 = None
        self._jieba = jieba
        self._bm25_build_ms = (time.monotonic() - t0) * 1000

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
        self._build_bm25()
        return len(sections)

    def _bm25_search(self, query: str, top_k: int = 3) -> list[dict]:
        if self._bm25 is None or not self._bm25_ids:
            return []
        tokens = list(self._jieba.cut(query))
        try:
            scores = self._bm25.get_scores(tokens)
        except (ZeroDivisionError, ValueError):
            return []
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [{"chunk_id": self._bm25_ids[i],
                 "content": self._bm25_docs[i]} for i in ranked if scores[i] > 0]

    @staticmethod
    def _rrf_fuse(dense: list, sparse: list, k: int = 60, top_k: int = 3) -> list[dict]:
        """RRF 融合：Score(d) = Σ 1/(k+rank(d))。

        dense 的 source/distance 字段优先保留，sparse 只补 chunk_id/content。
        """
        scores = {}
        meta = {}
        # 先放 dense（带 source/distance），再放 sparse（只补缺失字段）
        for hits in (dense, sparse):
            for rank, h in enumerate(hits, 1):
                cid = h["chunk_id"]
                scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank)
                if cid not in meta:
                    meta[cid] = dict(h)
                else:
                    # dense 已有的 source/distance 不被 sparse 覆盖
                    meta[cid].update({k: v for k, v in h.items()
                                      if k not in meta[cid]})
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [{**meta[cid], "rrf_score": s} for cid, s in ranked]

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """混合检索：BGE 稠密 + BM25 稀疏，RRF 融合重排。"""
        ef = self._ef
        if ef is not None and getattr(ef, "QUERY_INSTRUCTION", None):
            vec = ef.embed_query(query)
            res = self.collection.query(query_embeddings=[vec], n_results=top_k)
        else:
            res = self.collection.query(query_texts=[query], n_results=top_k)
        dense = [
            {"content": res["documents"][0][i],
             "source": res["metadatas"][0][i]["source"],
             "chunk_id": res["ids"][0][i],
             "distance": res["distances"][0][i]}
            for i in range(len(res["documents"][0]))
        ]
        sparse = self._bm25_search(query, top_k)
        return self._rrf_fuse(dense, sparse, top_k=top_k)

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

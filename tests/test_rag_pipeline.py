"""tests/test_rag_pipeline.py：RAG 流水线断言测试（FakeEmbedding，离线可跑）。"""
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chromadb import EmbeddingFunction
from rag_pipeline import RAGPipeline, split_text

DIM = 64


class FakeEmbedding(EmbeddingFunction):
    """确定性词袋向量：中文按单字、英文按词，hash 到固定维度。"""

    def __init__(self):
        pass

    def __call__(self, input):
        return [self._embed(t) for t in input]

    def _embed(self, text):
        vec = [0.0] * DIM
        for ch in text:
            if "一" <= ch <= "鿿":
                vec[hash(ch) % DIM] += 1.0
        for token in re.findall(r"[a-z]+", text.lower()):
            vec[hash(token) % DIM] += 1.0
        return vec


def test_split_text_overlap():
    text = "段落" * 300
    chunks = split_text(text, 500, 50)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[1][:50] == chunks[0][-50:]


def test_persistence():
    db, f = Path("data/_test_chroma"), Path("data/_test_doc.md")
    f.write_text("# 错误排查\n\n" + "超时" * 200, encoding="utf-8")
    try:
        p = RAGPipeline(str(db), embedding_function=FakeEmbedding())
        assert p.ingest(str(f)) >= 1
        p2 = RAGPipeline(str(db), embedding_function=FakeEmbedding())
        assert len(p2.search("超时", 2)) >= 1
    finally:
        shutil.rmtree(db, ignore_errors=True)
        f.unlink(missing_ok=True)


def test_search_metadata():
    db, f = Path("data/_test_chroma2"), Path("data/_test_meta.md")
    f.write_text("# 文档\n\nRedis 缓存击穿问题与解决" * 10, encoding="utf-8")
    try:
        p = RAGPipeline(str(db), embedding_function=FakeEmbedding())
        p.ingest(str(f))
        res = p.search("Redis 缓存", top_k=2)
        assert 1 <= len(res) <= 2
        for r in res:
            assert "content" in r and "chunk_id" in r
            assert r["source"] == f.as_posix()
    finally:
        shutil.rmtree(db, ignore_errors=True)
        f.unlink(missing_ok=True)


def test_multi_file_isolation():
    db = Path("data/_test_chroma3")
    fa, fb = Path("data/_a.md"), Path("data/_b.md")
    fa.write_text("数据库连接超时排查方案" * 20, encoding="utf-8")
    fb.write_text("前端跨域 CORS 配置说明" * 20, encoding="utf-8")
    try:
        p = RAGPipeline(str(db), embedding_function=FakeEmbedding())
        p.ingest(str(fa))
        p.ingest(str(fb))
        res = p.search("数据库超时", top_k=1)
        assert res
        assert res[0]["source"] == fa.as_posix()
    finally:
        shutil.rmtree(db, ignore_errors=True)
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


def test_early_separator_no_deadloop():
    text = "标题\n" + "正文内容" * 200
    chunks = split_text(text, 500, 50)
    assert len(chunks) == 2
    assert all(c for c in chunks)


def test_rrf_fuse_basic():
    dense = [
        {"chunk_id": "a", "content": "A", "distance": 0.1},
        {"chunk_id": "b", "content": "B", "distance": 0.2},
    ]
    sparse = [
        {"chunk_id": "c", "content": "C"},
        {"chunk_id": "a", "content": "A"},
    ]
    fused = RAGPipeline._rrf_fuse(dense, sparse, k=60, top_k=3)
    ids = [r["chunk_id"] for r in fused]
    assert "a" in ids
    assert "c" in ids
    assert "rrf_score" in fused[0]
    # dense 的 distance 字段不被 sparse 覆盖
    a_result = [r for r in fused if r["chunk_id"] == "a"][0]
    assert "distance" in a_result


def test_bm25_keyword_search():
    """BM25 关键词召回。语料必须 >= 3 篇：N=2 且 df=1 时 BM25Okapi 的
    IDF 恒为 ln(1.5)-ln(1.5)=0，得分必为 0，属数学退化而非代码缺陷。"""
    db = Path("data/_test_bm25")
    f1, f2 = Path("data/_bm25a.md"), Path("data/_bm25b.md")
    f3 = Path("data/_bm25c.md")
    f1.write_text("# 文档A\n\n宠物误食毒物后送医时间窗口与急救方案", encoding="utf-8")
    f2.write_text("# 文档B\n\n犬猫体温正常区间与发烧判断标准", encoding="utf-8")
    f3.write_text("# 文档C\n\n前端浏览器兼容性与渲染性能优化", encoding="utf-8")
    try:
        p = RAGPipeline(str(db), embedding_function=FakeEmbedding())
        p.ingest(str(f1))
        p.ingest(str(f2))
        p.ingest(str(f3))
        sparse = p._bm25_search("送医", top_k=3)
        assert len(sparse) >= 1
        assert "送医" in sparse[0]["content"]
    finally:
        shutil.rmtree(db, ignore_errors=True)
        f1.unlink(missing_ok=True)
        f2.unlink(missing_ok=True)
        f3.unlink(missing_ok=True)


def test_bm25_recovery_after_restart():
    """冷启动：重新 new RAGPipeline 后无需 ingest 即可检索。"""
    db = Path("data/_test_recovery")
    f1, f2 = Path("data/_rec_a.md"), Path("data/_rec_b.md")
    f1.write_text("# A\n\n宠物误食毒物后送医时间窗口与急救方案", encoding="utf-8")
    f2.write_text("# B\n\n犬猫体温正常区间与发烧判断标准", encoding="utf-8")
    try:
        p1 = RAGPipeline(str(db), embedding_function=FakeEmbedding())
        p1.ingest(str(f1))
        p1.ingest(str(f2))
        p2 = RAGPipeline(str(db), embedding_function=FakeEmbedding())
        assert p2._bm25 is not None
        res = p2.search("送医", top_k=2)
        assert len(res) >= 1
        assert "送医" in res[0]["content"]
    finally:
        shutil.rmtree(db, ignore_errors=True)
        f1.unlink(missing_ok=True)
        f2.unlink(missing_ok=True)


def test_top_k_truncation():
    """top_k 在 dense/sparse/fused 三层都正确截断。"""
    db = Path("data/_test_topk")
    f1, f2 = Path("data/_tk_a.md"), Path("data/_tk_b.md")
    f1.write_text("# A\n\n宠物误食毒物后送医时间窗口与急救方案", encoding="utf-8")
    f2.write_text("# B\n\n犬猫体温正常区间与发烧判断标准", encoding="utf-8")
    try:
        p = RAGPipeline(str(db), embedding_function=FakeEmbedding())
        p.ingest(str(f1))
        p.ingest(str(f2))
        res = p.search("送医", top_k=1)
        assert len(res) == 1
    finally:
        shutil.rmtree(db, ignore_errors=True)
        f1.unlink(missing_ok=True)
        f2.unlink(missing_ok=True)


if __name__ == "__main__":
    test_split_text_overlap()
    test_persistence()
    test_search_metadata()
    test_multi_file_isolation()
    test_early_separator_no_deadloop()
    test_rrf_fuse_basic()
    test_bm25_keyword_search()
    test_bm25_recovery_after_restart()
    test_top_k_truncation()
    print("全部断言通过")

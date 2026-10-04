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
    text = "段落" * 300  # 1200 字符、无分隔符，硬切
    chunks = split_text(text, 500, 50)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[1][:50] == chunks[0][-50:]  # 相邻块 50 字符重叠


def test_persistence():
    db, f = Path("data/_test_chroma"), Path("data/_test_doc.md")
    f.write_text("# 错误排查\n\n" + "超时" * 200, encoding="utf-8")
    try:
        p = RAGPipeline(str(db), embedding_function=FakeEmbedding())
        assert p.ingest(str(f)) >= 1
        assert any(db.rglob("*.sqlite*"))  # 落盘文件存在
        p2 = RAGPipeline(str(db), embedding_function=FakeEmbedding())
        assert len(p2.search("超时", 2)) >= 1  # 重开不 ingest 也能检索
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
            assert set(r) == {"content", "source", "chunk_id", "distance"}
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
        res = p.search("数据库超时", top_k=1)  # 最相关一条
        assert res
        assert res[0]["source"] == fa.as_posix()
    finally:
        shutil.rmtree(db, ignore_errors=True)
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


def test_early_separator_no_deadloop():
    # 开头 50 字符内就有换行：旧代码 start 变负数、切出碎块甚至死循环
    text = "标题\n" + "正文内容" * 200
    chunks = split_text(text, 500, 50)
    assert len(chunks) == 2
    assert all(c for c in chunks)
    assert chunks[1][:50] == chunks[0][-50:]  # 重叠正常，无重复碎块


if __name__ == "__main__":
    test_split_text_overlap()
    test_persistence()
    test_search_metadata()
    test_multi_file_isolation()
    test_early_separator_no_deadloop()
    print("全部断言通过")

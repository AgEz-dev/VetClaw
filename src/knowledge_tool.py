"""知识库检索工具：把 RAGPipeline 封装为 Agent 可调用工具。"""
from rag_pipeline import RAGPipeline
from tools import registry

MAX_DISTANCE = 0.8     # chroma cosine: distance = 1 - similarity，范围 [0,2]；0.8 即相似度 0.2
MAX_CHUNK_CHARS = 300  # 单块正文截断上限，防止回填 Token 膨胀


class RAGProvider:
    def __init__(self):
        self._pipeline = None

    def set(self, pipeline):
        self._pipeline = pipeline

    def get(self):
        if self._pipeline is None:
            # 暂不加锁：当前单线程/单事件循环场景够用；多线程并发初始化需加锁
            self._pipeline = RAGPipeline()
        return self._pipeline


rag_provider = RAGProvider()


def format_results(results, query):
    hits = [r for r in results if r.get("distance", 0.0) <= MAX_DISTANCE]
    if not hits:
        return (f"知识库中未检索到与「{query}」相关的内容，请基于通用知识回答，"
                "或明确告知缺少相关文档，不要编造。")
    lines = ["【知识库检索结果】"]
    for i, r in enumerate(hits, 1):
        content = r.get("content", "")
        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS] + "……（已截断）"
        loc = r.get("chunk_id", r.get("source", "未知来源"))
        lines.append(f"[{i}] {loc}\n{content}")
    return "\n".join(lines)


def search_knowledge_base(query: str) -> str:
    """检索项目技术知识库（错误排查、部署、配置等文档），用于回答需查阅文档的问题；query 为关键词或完整问题。"""
    return format_results(rag_provider.get().search(query, top_k=3), query)


def register_to(reg):
    reg.register(search_knowledge_base)


register_to(registry)

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi.testclient import TestClient
from app import app
from rag_pipeline import RAGPipeline, BGEEmbeddingFunction
from feedback_store import (
    record_feedback,
    load_badcases,
    export_badcases_to_eval_format,
    anonymize_text
)


def test_async_search_non_blocking():
    """测试 async_search 使用 asyncio.to_thread 进行异步线程隔离。"""
    pipe = RAGPipeline("data/chroma", "knowledge", embedding_function=BGEEmbeddingFunction())

    async def _search():
        res = await pipe.async_search("猫咪百合花中毒", top_k=2)
        assert len(res) > 0
        assert "chunk_id" in res[0]
        return res

    results = asyncio.run(_search())
    assert len(results) > 0


def test_anonymize_text_privacy():
    """测试隐私脱敏正则：手机号与身份证号。"""
    raw = "宠主李女士电话 13912345678，身份证 110101199003072345，带猫来看病"
    masked = anonymize_text(raw)
    assert "139****5678" in masked
    assert "13912345678" not in masked
    assert "110101********2345" in masked


def test_record_feedback_and_export(tmp_path):
    """测试 Badcase 记录与评测集转换飞轮。"""
    bc_file = tmp_path / "badcases.jsonl"
    eval_file = tmp_path / "eval_export.json"

    # 1. 记录反馈
    rec = record_feedback(
        trace_id="tr-test-101",
        role="DOCTOR",
        query="宠主手机号 13888889999 咨询细小补液",
        retrieved_chunk_ids=["knowledge/outpatient_clinical_guidelines.md#休克-1"],
        ai_output="建议补充生理盐水 100ml",
        human_revised_output="休克期首选乳酸林格氏液 20ml/kg 快速扩容并补钾",
        reason="AI 未按指南首选平衡晶体液并遗漏补钾",
        file_path=bc_file
    )
    assert "138****9999" in rec["query"]

    # 2. 读取验证
    cases = load_badcases(bc_file)
    assert len(cases) == 1
    assert cases[0]["trace_id"] == "tr-test-101"

    # 3. 飞轮导出为 eval_dataset
    export_res = export_badcases_to_eval_format(input_path=bc_file, output_path=eval_file)
    assert export_res["cases_count"] == 1
    ec = export_res["cases"][0]
    assert ec["id"] == "BC-001"
    assert ec["expected_source_chunk"] == "knowledge/outpatient_clinical_guidelines.md#休克-1"
    assert ec["category"] == "badcase_feedback"


def test_api_feedback_endpoint():
    """测试 POST /a2a/feedback API 接口。"""
    client = TestClient(app)
    payload = {
        "trace_id": "tr-api-01",
        "role": "DOCTOR",
        "query": "猫吃百合呕吐",
        "retrieved_chunk_ids": ["knowledge/species_drug_safety_matrix.md#百合属-Lilium-Hemerocallis-所有部位-4"],
        "ai_output": "建议在家观察",
        "human_revised_output": "百合对猫有急性致命肾毒性，必须立即洗胃催吐并导流急诊！",
        "reason": "AI 未识别出致命毒物急诊"
    }
    resp = client.post("/a2a/feedback", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["code"] == 0
    assert data["message"] == "feedback recorded"
    assert data["data"]["trace_id"] == "tr-api-01"

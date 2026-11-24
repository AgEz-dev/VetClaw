import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mcp_server import VetClawMcpServer


@pytest.fixture
def mcp():
    return VetClawMcpServer()


def test_mcp_initialize(mcp):
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        }
    }
    resp = mcp.handle_message(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert "tools" in result["capabilities"]
    assert "resources" in result["capabilities"]
    assert "prompts" in result["capabilities"]
    assert result["serverInfo"]["name"] == "vetclaw-clinical-mcp"


def test_mcp_tools_list(mcp):
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = mcp.handle_message(req)
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "check_drug_safety" in tool_names
    assert "search_knowledge_base" in tool_names
    assert "get_inventory_summary" in tool_names
    assert "triage_patient" in tool_names
    assert "review_prescription" in tool_names


def test_mcp_tool_call_drug_safety(mcp):
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "check_drug_safety",
            "arguments": {
                "drug_name": "对乙酰氨基酚",
                "species": "猫"
            }
        }
    }
    resp = mcp.handle_message(req)
    assert resp["id"] == 3
    assert not resp["result"]["isError"]
    content = resp["result"]["content"][0]["text"]
    assert "对乙酰氨基酚" in content or "BLOCK" in content or "拒绝" in content or "禁" in content


def test_mcp_tool_call_triage(mcp):
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "triage_patient",
            "arguments": {
                "chief_complaint": "猫咪误食百合花并张口呼吸",
                "species": "猫"
            }
        }
    }
    resp = mcp.handle_message(req)
    assert resp["id"] == 4
    content = resp["result"]["content"][0]["text"]
    assert "严重程度评估" in content or "兽医" in content or "急" in content


def test_mcp_resources_list_and_read(mcp):
    req_list = {"jsonrpc": "2.0", "id": 5, "method": "resources/list"}
    resp_list = mcp.handle_message(req_list)
    resources = resp_list["result"]["resources"]
    assert len(resources) > 0
    first_uri = resources[0]["uri"]

    req_read = {"jsonrpc": "2.0", "id": 6, "method": "resources/read", "params": {"uri": first_uri}}
    resp_read = mcp.handle_message(req_read)
    assert resp_read["id"] == 6
    contents = resp_read["result"]["contents"]
    assert len(contents) == 1
    assert len(contents[0]["text"]) > 10


def test_mcp_prompts_list_and_get(mcp):
    req_list = {"jsonrpc": "2.0", "id": 7, "method": "prompts/list"}
    resp_list = mcp.handle_message(req_list)
    prompts = resp_list["result"]["prompts"]
    prompt_names = {p["name"] for p in prompts}
    assert "outpatient_preconsultation" in prompt_names
    assert "soap_record_draft" in prompt_names

    req_get = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "prompts/get",
        "params": {
            "name": "outpatient_preconsultation",
            "arguments": {"species": "幼犬", "symptom": "频繁呕吐便血"}
        }
    }
    resp_get = mcp.handle_message(req_get)
    msg = resp_get["result"]["messages"][0]["content"]["text"]
    assert "幼犬" in msg
    assert "频繁呕吐便血" in msg

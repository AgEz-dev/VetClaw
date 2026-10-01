"""tests/test_tools.py：工具注册表断言测试（全部用 assert，不依赖 pytest 也能跑）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tools import ToolRegistry, tool


def test_schema_structure():
    """验证自动生成的 JSON Schema 结构、required 判定、无注解参数降级。"""
    @tool
    def query_error_doc(keyword: str, limit: int = 5, tag=None) -> list:
        """按关键词查询错误文档。"""
        return [keyword]

    s = query_error_doc.schema
    assert s["type"] == "function"
    assert s["function"]["name"] == "query_error_doc"
    assert s["function"]["description"] == "按关键词查询错误文档。"
    params = s["function"]["parameters"]
    assert params["type"] == "object"
    props = params["properties"]
    assert props["keyword"] == {"type": "string"}
    assert props["limit"] == {"type": "integer"}
    # tag 没写类型注解，降级为 string；tag 有默认值 None，不进 required
    assert props["tag"] == {"type": "string"}
    # ★ required 里只能有 keyword：limit、tag 都有默认值
    assert params["required"] == ["keyword"]


def test_call_executes_real_function():
    """验证 call 真实执行函数、缺省参数生效、未知工具抛 KeyError。"""
    reg = ToolRegistry()

    @reg.tool
    def greet(name: str, excited: bool = False) -> str:
        msg = f"hello {name}"
        return msg.upper() if excited else msg

    assert reg.call("greet", {"name": "world", "excited": True}) == "HELLO WORLD"
    assert reg.call("greet", {"name": "world"}) == "hello world"
    try:
        reg.call("nope", {})
        raise AssertionError("未知工具名应抛 KeyError")
    except KeyError:
        pass

def test_custom_server_status():
    @tool
    def get_server_status(port: int = 8080) -> str:
        """查询服务器状态。"""
        return f"port {port} ok"

    # 执行真跑
    assert get_server_status(port=9090) == "port 9090 ok"
    assert get_server_status() == "port 8080 ok"

    # 验证提取的 Schema 结构（你自己写的断言）
    schema = get_server_status.schema
    params = schema["function"]["parameters"]
    
    assert params["properties"]["port"]["type"] == "integer"
    assert "port" not in params["required"]


if __name__ == "__main__":
    test_schema_structure()
    test_call_executes_real_function()
    test_custom_server_status() 
    print("全部断言通过")

"""工具注册表：普通函数自动包装为 LLM 可调用工具（OpenAI Function Calling 格式）。"""
import inspect
from typing import Any, Callable, Dict, List

_TYPE_MAP = {str: "string", int: "integer", float: "number",
             bool: "boolean", list: "array", dict: "object"}


def function_to_schema(func: Callable) -> dict:
    sig = inspect.signature(func)
    doc = (func.__doc__ or "").strip().split("\n")[0]
    props, required = {}, []
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        props[name] = {"type": _TYPE_MAP.get(p.annotation, "string")}
        # ★ 必填判定：无默认值时 default 是哨兵 inspect.Parameter.empty
        if p.default == inspect.Parameter.empty:
            required.append(name)
    return {"type": "function",
            "function": {"name": func.__name__, "description": doc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


class Tool:
    def __init__(self, func: Callable, schema: dict):
        self.func, self.schema = func, schema

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]

    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, func: Callable) -> Tool:
        tool = Tool(func, function_to_schema(func))
        self._tools[tool.name] = tool
        return tool

    def tool(self, func: Callable = None):
        return self.register(func) if func else self.register

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"工具不存在: {name}")
        return self._tools[name]

    def call(self, name: str, arguments: dict) -> Any:
        return self.get(name).func(**arguments)

    def schemas(self) -> List[dict]:
        return [t.schema for t in self._tools.values()]


registry = ToolRegistry()
tool = registry.tool

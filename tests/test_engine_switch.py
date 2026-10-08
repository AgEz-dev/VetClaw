"""tests/test_engine_switch.py：引擎开关骨架（S4.0）单测。

覆盖三件事：
1. `Settings.engine` 默认 `react`、非法值 **fail-fast**、取值归一化；
2. **懒加载隔离**：`react` 引擎下导入装配层，不得把 `langgraph` 带进 `sys.modules`
   （实测 `import langgraph` 冷启动 ≈859ms，顶层 import 会毁掉 P0「3.2ms / 冷启动」口径）；
3. 开关默认值不改变线上行为（`react` 仍是默认路径）。
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def _fresh_settings(monkeypatch, engine=None):
    """在受控环境变量下构造一个全新的 Settings（不依赖模块级单例）。"""
    monkeypatch.delenv("VETCLAW_ENGINE", raising=False)
    if engine is not None:
        monkeypatch.setenv("VETCLAW_ENGINE", engine)
    from core.config import Settings

    return Settings()


def test_engine_defaults_to_react(monkeypatch):
    assert _fresh_settings(monkeypatch).engine == "react"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("react", "react"),
        ("langgraph", "langgraph"),
        ("  LANGGRAPH  ", "langgraph"),
        ("React", "react"),
    ],
)
def test_engine_value_normalized(monkeypatch, raw, expected):
    assert _fresh_settings(monkeypatch, raw).engine == expected


@pytest.mark.parametrize("bad", ["tensorflow", "langgraph2", "graph", ""])
def test_engine_invalid_value_fails_fast(monkeypatch, bad):
    """非法值必须抛错，不得静默回落 react —— 否则 CI 会"假绿"。"""
    with pytest.raises(ValueError, match="VETCLAW_ENGINE"):
        _fresh_settings(monkeypatch, bad)


def test_react_engine_does_not_import_langgraph():
    """冷启动隔离：只导入装配层，`langgraph` 不得进入 sys.modules。

    子进程执行，避免 pytest 进程里其它 import 造成的污染。
    失败时打印「是谁 import 的」栈，便于定位。
    """
    code = (
        "import sys, builtins, traceback\n"
        "_orig = builtins.__import__\n"
        "def _trace(name, *a, **k):\n"
        "    if name.split('.')[0] == 'langgraph':\n"
        "        print('--- import ' + name + ' triggered by: ---', file=sys.stderr)\n"
        "        traceback.print_stack(file=sys.stderr)\n"
        "    return _orig(name, *a, **k)\n"
        "builtins.__import__ = _trace\n"
        "sys.path.insert(0, " + repr(str(SRC)) + ")\n"
        "import services.agent_service\n"
        "print('RESULT=' + ('LOADED' if 'langgraph' in sys.modules else 'CLEAN'))\n"
        "print('LANGMODS=' + ','.join(sorted(m for m in sys.modules "
        "if m.split('.')[0] == 'langgraph')))\n"
    )
    env = dict(os.environ, VETCLAW_ENGINE="react")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, f"导入装配层失败：{proc.stderr[-800:]}"
    assert "RESULT=CLEAN" in proc.stdout, (
        f"stdout={proc.stdout!r}\n--- stderr(stack) ---\n{proc.stderr[-1800:]}\n"
        "装配层（或其 import 链）把 langgraph 带进了 sys.modules；"
        "冷启动会多出 ≈859ms，P0「3.2ms」口径被破坏"
    )


def test_no_module_level_langgraph_import_in_src():
    """静态保障（**确定性**，永不 flaky）：`src/**/*.py` 模块级不得 import langgraph。

    与上面那条运行时子进程检查互为补充：
    - 运行时检查能发现"间接被拖进来"，但依赖进程/文件系统状态（曾出现 1 次
      一次性失败，疑与 pip 刚装完的首次冷启动有关，未能复现）；
    - 这条基于 AST，只认源码事实。

    允许的写法：**函数体内**的 import（即 `build_agent()` 的懒加载分支）。
    """
    import ast

    offenders = []

    def _scan(node, inside_func):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                _scan(child, True)
                continue
            if isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name.split(".")[0] == "langgraph" and not inside_func:
                        offenders.append((str(py), child.lineno))
            elif isinstance(child, ast.ImportFrom):
                if (child.module or "").split(".")[0] == "langgraph" and not inside_func:
                    offenders.append((str(py), child.lineno))
            _scan(child, inside_func)

    for py in sorted(SRC.rglob("*.py")):
        _scan(ast.parse(py.read_text(encoding="utf-8")), False)

    assert not offenders, (
        f"以下位置在**模块级** import 了 langgraph（冷启动 +859ms）：{offenders}。"
        "请改为在函数体内懒加载。"
    )

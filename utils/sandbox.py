"""Restricted execution of LLM-generated pandas code.

Defence in layers:
1. Static check (AST): reject imports (other than pandas/numpy/plotly), dunder access, file/OS functions, and file I/O methods.
2. Restricted globals: only a whitelist of safe builtins plus df, pd, np, px.
3. Works on a copy of the data, with a time limit that really stops the code.

Note: in-process Python sandboxing is never bulletproof. This is suitable for a personal
or demo app; a multi-user production service should run generated code in an isolated
container or subprocess.
"""

from __future__ import annotations

import ast
import builtins
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TIME_LIMIT_SECONDS = 15
MAX_CONSTANT_EXPONENT = 100  # blocks things like 10**10**10 that freeze the CPU

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance",
        "len", "list", "map", "max", "min", "range", "reversed", "round", "set", "sorted", "str",
        "sum", "tuple", "zip", "ValueError", "KeyError", "TypeError", "Exception",
    )
}
SAFE_BUILTINS["print"] = lambda *args, **kwargs: None  # swallow prints

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview",
    "os", "sys", "subprocess", "shutil", "pathlib", "importlib", "builtins",
}
# Methods that read/write files or evaluate strings as code.
BLOCKED_ATTRIBUTES = {
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_json", "to_html", "to_sql", "to_hdf",
    "to_feather", "to_stata", "to_clipboard", "to_latex", "to_xml", "to_orc", "write_html",
    "write_image", "write_json", "show", "eval", "tofile", "save", "savez", "savetxt", "load",
    "loadtxt", "fromfile", "genfromtxt", "system", "popen",
}


# Imports of libraries that are already provided are allowed and simply skipped.
ALLOWED_MODULES = {"pandas": pd, "numpy": np, "plotly.express": px, "plotly.graph_objects": go}


class UnsafeCodeError(Exception):
    pass


class _Timeout(BaseException):
    """BaseException so generated `except Exception:` blocks can't swallow it."""


@dataclass
class ExecutionResult:
    result: Any = None
    fig: go.Figure | None = None
    error: str | None = None


def _strip_safe_imports(tree: ast.Module) -> dict[str, Any]:
    """Remove top-level imports of provided libraries; return the names they bind."""
    bindings: dict[str, Any] = {}
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any((a.asname or "").startswith("_") for a in node.names):
            kept.append(node)  # suspicious alias: leave it so the check below blocks it
            continue
        if isinstance(node, ast.Import) and all(a.name in ALLOWED_MODULES for a in node.names):
            for a in node.names:
                if a.asname or "." not in a.name:
                    bindings[a.asname or a.name] = ALLOWED_MODULES[a.name]
            continue
        if (isinstance(node, ast.ImportFrom) and node.module == "plotly"
                and all(f"plotly.{a.name}" in ALLOWED_MODULES for a in node.names)):
            for a in node.names:
                bindings[a.asname or a.name] = ALLOWED_MODULES[f"plotly.{a.name}"]
            continue
        kept.append(node)
    tree.body = kept
    return bindings


def check_code(code: str) -> tuple[ast.Module, dict[str, Any]]:
    """Parse the code and raise UnsafeCodeError if it uses anything disallowed."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Syntax error: {e.msg} (line {e.lineno})")
    bindings = _strip_safe_imports(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("Imports are not allowed.")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.While, ast.AsyncFunctionDef, ast.Await)):
            raise UnsafeCodeError(f"`{type(node).__name__}` statements are not allowed.")
        if isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or node.id.startswith("__")):
            raise UnsafeCodeError(f"Use of `{node.id}` is not allowed.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsafeCodeError(f"Access to private attribute `{node.attr}` is not allowed.")
            if node.attr in BLOCKED_ATTRIBUTES or node.attr.startswith("read_"):
                raise UnsafeCodeError(f"`.{node.attr}()` is not allowed (no file or code access).")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exp = node.right
            if (isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float))
                    and abs(exp.value) > MAX_CONSTANT_EXPONENT) or any(
                    isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) for n in ast.walk(exp)):
                raise UnsafeCodeError("Very large powers are not allowed.")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise UnsafeCodeError("Strings containing '__' are not allowed.")
    return tree, bindings


def run_code(code: str, df: pd.DataFrame) -> ExecutionResult:
    """Validate and execute generated code against a copy of `df`."""
    try:
        tree, bindings = check_code(code)
    except UnsafeCodeError as e:
        return ExecutionResult(error=f"Blocked unsafe code: {e}")

    namespace = {"__builtins__": SAFE_BUILTINS, "df": df.copy(), "pd": pd, "np": np, "px": px, "go": go,
                 **bindings}

    outcome: dict[str, BaseException] = {}
    deadline = time.monotonic() + TIME_LIMIT_SECONDS

    def tracer(frame, event, arg):
        # Called on every function call (and every line of the generated code):
        # once the deadline passes, raise inside the running code to stop it.
        if time.monotonic() > deadline:
            raise _Timeout()
        return tracer if frame.f_code.co_filename == "<generated>" else None

    def _execute():
        sys.settrace(tracer)
        try:
            exec(compile(tree, "<generated>", "exec"), namespace)
        except BaseException as e:
            outcome["error"] = e
        finally:
            sys.settrace(None)

    worker = threading.Thread(target=_execute, daemon=True)
    worker.start()
    worker.join(TIME_LIMIT_SECONDS + 2)
    error = outcome.get("error")
    if worker.is_alive() or isinstance(error, _Timeout):
        return ExecutionResult(error=f"The code took longer than {TIME_LIMIT_SECONDS}s and was stopped.")
    if error is not None:
        return ExecutionResult(error=f"{type(error).__name__}: {error}")

    fig = namespace.get("fig")
    if not isinstance(fig, go.Figure):
        fig = None
    result = namespace.get("result")
    if isinstance(result, go.Figure):  # the model put the chart in `result`
        fig, result = fig or result, None
    if "result" not in namespace and fig is None:
        return ExecutionResult(error="The code did not set a `result` variable.")
    return ExecutionResult(result=result, fig=fig)


def result_to_text(result: Any, max_rows: int = 60) -> str:
    """Render a result compactly for display or for sending back to the LLM."""
    if isinstance(result, (pd.DataFrame, pd.Series)):
        text = result.head(max_rows).to_string()
        if len(result) > max_rows:
            text += f"\n... ({len(result)} rows total)"
        return text
    if isinstance(result, float):
        return f"{result:,.4f}".rstrip("0").rstrip(".")
    return str(result)

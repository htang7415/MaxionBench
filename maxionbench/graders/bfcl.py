"""Offline BFCL AST grader for the single-turn categories (simple, multiple, parallel, parallel_multiple).

Follows the BFCL AST checker's rules: the function name must match; required parameters must be
present; no parameter outside the possible answer; values must match one acceptable option, where
"" marks an optional parameter that may be omitted, strings compare after BFCL's normalization
(case, spaces and ,./-_*^ ignored), integers are accepted for float parameters, and parallel calls
match in any order. Dotted BFCL names are sent to OpenAI-style APIs with underscores.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

from maxionbench.datasets.loaders.v03 import BfclCase

Call = dict[str, dict[str, Any]]  # {function_name: {param: value}}

_TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}
_STRIP = re.compile(r"[ ,./\-_*^]")


@dataclass(frozen=True)
class BfclGrade:
    correct: bool
    error: str | None = None


def openai_name(name: str) -> str:
    return name.replace(".", "_")


def to_openai_tools(functions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """BFCL function docs -> OpenAI `tools` (JSON Schema types, API-safe names)."""
    return [
        {"type": "function", "function": {
            "name": openai_name(f["name"]), "description": f.get("description", ""),
            "parameters": _schema(f["parameters"]),
        }}
        for f in functions
    ]


def _schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_schema(n) for n in node]
    if not isinstance(node, Mapping):
        return node
    out = {k: _schema(v) for k, v in node.items() if k != "optional"}
    if isinstance(node.get("type"), str):
        out["type"] = _TYPES.get(node["type"], node["type"])
    return out


def parse_openai_tool_calls(tool_calls: Sequence[Mapping[str, Any]] | None) -> list[Call]:
    """OpenAI `message.tool_calls` -> [{name: args}]; raises ValueError on non-JSON arguments."""
    calls = []
    for tc in tool_calls or ():
        fn = tc["function"]
        args = fn.get("arguments") or "{}"
        try:
            parsed = json.loads(args) if isinstance(args, str) else dict(args)
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments for {fn['name']!r} are not JSON: {exc}") from None
        calls.append({fn["name"]: parsed})
    return calls


def grade(case: BfclCase, calls: Sequence[Call]) -> BfclGrade:
    docs = {f["name"]: f for f in case.functions}
    by_api_name = {openai_name(n): n for n in docs}
    named: list[tuple[str, dict[str, Any]]] = []
    for call in calls:
        if len(call) != 1:
            return BfclGrade(False, "each call must name exactly one function")
        ((name, args),) = call.items()
        named.append((name if name in docs else by_api_name.get(name, name), args))
    truth = [next(iter(t.items())) for t in case.ground_truth]
    if len(named) != len(truth):
        return BfclGrade(False, f"expected {len(truth)} call(s), got {len(named)}")
    if len(truth) == 1:  # simple / multiple: report the specific mismatch
        error = _check(docs.get(truth[0][0]), named[0], truth[0])
        return BfclGrade(error is None, error)
    if _match_all(docs, named, truth, used=frozenset()):
        return BfclGrade(True)
    return BfclGrade(False, "no one-to-one match between calls and possible answers")


def _match_all(docs: Mapping[str, Any], named: list, truth: list, used: frozenset[int]) -> bool:
    if not truth:
        return True
    head, rest = truth[0], truth[1:]
    return any(
        i not in used and _check(docs.get(head[0]), call, head) is None and _match_all(docs, named, rest, used | {i})
        for i, call in enumerate(named)
    )


def _check(doc: Mapping[str, Any] | None, call: tuple[str, dict[str, Any]], truth: tuple[str, dict[str, list]]) -> str | None:
    name, args = call
    want_name, options = truth
    if name != want_name:
        return f"wrong function {name!r}, expected {want_name!r}"
    params = (doc or {}).get("parameters", {})
    props = params.get("properties", {})
    for p in params.get("required", []):
        if p not in args:
            return f"missing required parameter {p!r}"
    for p, value in args.items():
        if p not in options:
            return f"unexpected parameter {p!r}"
        if value is None:
            if None not in options[p]:
                return f"parameter {p!r} is null"
            continue
        if not _type_ok(value, props.get(p, {}).get("type")):
            return f"wrong type for {p!r}: {type(value).__name__}"
        if not any(_match(value, o) for o in options[p] if not _is_omit(o)):
            return f"wrong value for {p!r}: {value!r}"
    for p, opts in options.items():
        if p not in args and not any(_is_omit(o) for o in opts):
            return f"missing parameter {p!r}"
    return None


def _is_omit(option: Any) -> bool:
    return isinstance(option, str) and option == ""


def _type_ok(value: Any, expected: str | None) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected in ("array", "tuple"):
        return isinstance(value, list)
    if expected == "dict":
        return isinstance(value, dict)
    return True  # "any" or undocumented parameter


def _match(value: Any, option: Any) -> bool:
    if option is None:
        return value is None
    if isinstance(option, dict):  # values are lists of acceptable options
        if not isinstance(value, dict) or not set(value) <= set(option):
            return False
        for k, opts in option.items():
            if k not in value:
                if not any(_is_omit(o) for o in opts):
                    return False
            elif not any(_match(value[k], o) for o in opts if not _is_omit(o)):
                return False
        return True
    if isinstance(option, list):
        return isinstance(value, list) and len(value) == len(option) and all(map(_match, value, option))
    if isinstance(option, str):
        return isinstance(value, str) and _norm(value) == _norm(option)
    if isinstance(option, bool) or isinstance(value, bool):
        return type(value) is type(option) and value == option
    if isinstance(option, (int, float)):
        return isinstance(value, (int, float)) and float(value) == float(option)
    return bool(value == option)


def _norm(text: str) -> str:
    return _STRIP.sub("", text).lower().replace("'", '"')

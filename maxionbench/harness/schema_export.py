"""Generate the JSON Schema for harness results from the result dataclasses.

The schema is the contract consumed by the TypeScript dashboard (types are generated from it), so it
is derived from the Python model rather than maintained by hand.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from maxionbench.harness.results import RESULT_SCHEMA_VERSION, ExperimentResult

SCHEMA_PATH = Path(__file__).with_name("result.schema.json")


def result_json_schema() -> dict[str, Any]:
    defs: dict[str, Any] = {}
    _schema_for(ExperimentResult, defs)
    root = defs.pop(ExperimentResult.__name__)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://maxionbench.local/schemas/{RESULT_SCHEMA_VERSION}.json",
        "title": "ExperimentResult",
        **root,
        "$defs": defs,
    }


def _schema_for(tp: Any, defs: dict[str, Any]) -> dict[str, Any]:
    origin = get_origin(tp)
    if is_dataclass(tp):
        name = tp.__name__
        if name not in defs:
            defs[name] = {}  # reserve before recursing
            hints = get_type_hints(tp)
            props = {f.name: _schema_for(hints[f.name], defs) for f in fields(tp)}
            if name == ExperimentResult.__name__:
                props["schema_version"] = {"const": RESULT_SCHEMA_VERSION}
            defs[name] = {
                "type": "object",
                "properties": props,
                "required": [f.name for f in fields(tp)],
                "additionalProperties": False,
            }
        return {"$ref": f"#/$defs/{name}"}
    if origin is list:
        return {"type": "array", "items": _schema_for(get_args(tp)[0], defs)}
    if origin is dict:
        return {"type": "object", "additionalProperties": _schema_for(get_args(tp)[1], defs)}
    if origin is Literal:
        return {"enum": list(get_args(tp))}
    args = get_args(tp)
    if args and type(None) in args:
        inner = next(a for a in args if a is not type(None))
        return {"anyOf": [_schema_for(inner, defs), {"type": "null"}]}
    if tp is Any:
        return {}
    return {str: {"type": "string"}, int: {"type": "integer"}, float: {"type": "number"}, bool: {"type": "boolean"}}[tp]


def schema_text() -> str:
    return json.dumps(result_json_schema(), indent=2, sort_keys=True) + "\n"

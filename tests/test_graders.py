from __future__ import annotations

import json
import random
from typing import Any

import pytest

from maxionbench.datasets import sources
from maxionbench.datasets.loaders.v03 import BFCL_CATEGORIES, BfclCase, load_bfcl
from maxionbench.graders import bfcl
from maxionbench.graders.qa import agent_success, crag_label, crag_score, grade_qa

FN = {
    "name": "weather.get",
    "parameters": {
        "type": "dict",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "scale": {"type": "float"},
            "units": {"type": "string"},
            "opts": {"type": "dict", "properties": {"hourly": {"type": "boolean"}}},
        },
        "required": ["city", "days"],
    },
}
TRUTH = {"weather.get": {"city": ["New York", "NYC"], "days": [3], "scale": [1.0, ""], "units": ["", "metric"],
                         "opts": ["", {"hourly": [True]}]}}


def _case(category: str = "simple", truth: tuple = (TRUTH,)) -> BfclCase:
    return BfclCase("t_0", category, ({"role": "user", "content": "q"},), (FN,), truth)


@pytest.mark.parametrize(
    "args, error",
    [
        ({"city": "new-york", "days": 3}, None),  # BFCL string normalization; optional params omitted
        ({"city": "NYC", "days": 3, "scale": 1, "units": "Metric", "opts": {"hourly": True}}, None),  # int ok for float
        ({"city": "Boston", "days": 3}, "wrong value for 'city'"),
        ({"city": "NYC"}, "missing required parameter 'days'"),
        ({"city": "NYC", "days": "3"}, "wrong type for 'days'"),
        ({"city": "NYC", "days": 3.0}, "wrong type for 'days'"),
        ({"city": "NYC", "days": 3, "lang": "en"}, "unexpected parameter 'lang'"),
        ({"city": "NYC", "days": 3, "opts": {"hourly": 1}}, "wrong value for 'opts'"),
        ({"city": "NYC", "days": True}, "wrong type for 'days'"),
    ],
)
def test_single_call_rules(args: dict[str, Any], error: str | None) -> None:
    result = bfcl.grade(_case(), [{"weather.get": args}])
    assert result.correct is (error is None)
    assert error is None or error in (result.error or "")


def test_wrong_function_and_call_count() -> None:
    assert bfcl.grade(_case(), [{"other": {}}]).error.startswith("wrong function")
    assert not bfcl.grade(_case(), []).correct
    assert bfcl.grade(_case(), [{"weather_get": {"city": "NYC", "days": 3}}]).correct  # API-safe name accepted


def test_parallel_calls_match_in_any_order() -> None:
    other = {"weather.get": {**TRUTH["weather.get"], "city": ["Paris"]}}
    case = _case("parallel", (TRUTH, other))
    paris, nyc = {"weather.get": {"city": "Paris", "days": 3}}, {"weather.get": {"city": "NYC", "days": 3}}
    assert bfcl.grade(case, [paris, nyc]).correct and bfcl.grade(case, [nyc, paris]).correct
    assert not bfcl.grade(case, [nyc, nyc]).correct
    assert not bfcl.grade(case, [nyc]).correct


def test_openai_tool_round_trip() -> None:
    (tool,) = bfcl.to_openai_tools([FN])
    assert tool["function"]["name"] == "weather_get"
    props = tool["function"]["parameters"]["properties"]
    assert tool["function"]["parameters"]["type"] == "object" and props["scale"]["type"] == "number"
    calls = bfcl.parse_openai_tool_calls(
        [{"id": "1", "type": "function", "function": {"name": "weather_get", "arguments": '{"city": "NYC", "days": 3}'}}]
    )
    assert bfcl.grade(_case(), calls).correct
    with pytest.raises(ValueError, match="not JSON"):
        bfcl.parse_openai_tool_calls([{"function": {"name": "f", "arguments": "{bad"}}])


def _canonical(options: list[Any]) -> tuple[bool, Any]:
    """(include?, value): the first acceptable option, built recursively; omit optional-only params."""
    for opt in options:
        if not (isinstance(opt, str) and opt == ""):
            return True, _value(opt)
    return False, None


def _value(option: Any) -> Any:
    if isinstance(option, dict):  # values are lists of options
        inner = {k: _canonical(v) for k, v in option.items()}
        return {k: v for k, (inc, v) in inner.items() if inc}
    if isinstance(option, list):
        return [_value(x) for x in option]
    return option


def _reference_calls(case: BfclCase) -> list[dict[str, Any]]:
    """The reference answer as an OpenAI tool_calls payload (reversed, to exercise order matching)."""
    calls = []
    for truth in reversed(case.ground_truth):
        ((name, params),) = truth.items()
        args = {p: v for p, (inc, v) in ((p, _canonical(o)) for p, o in params.items()) if inc}
        calls.append({"function": {"name": bfcl.openai_name(name), "arguments": json.dumps(args)}})
    return calls


_HAVE_BFCL = (sources.DATASET_ROOT / "bfcl").exists()
# Reference answers that violate their own function schema, so a strict type check rejects them:
# simple_307 passes venue=True to a string parameter; parallel_multiple_21 passes strings to array params.
KNOWN_SCHEMA_CONFLICTS = {"simple_307", "parallel_multiple_21"}


@pytest.mark.skipif(not _HAVE_BFCL, reason="BFCL not fetched")
@pytest.mark.parametrize("category", BFCL_CATEGORIES)
def test_bfcl_reference_answers_grade_correct(category: str) -> None:
    cases = load_bfcl(category)
    failures = {
        c.id: g.error for c in cases
        if not (g := bfcl.grade(c, bfcl.parse_openai_tool_calls(_reference_calls(c)))).correct
    }
    assert set(failures) == KNOWN_SCHEMA_CONFLICTS & {c.id for c in cases}, failures
    assert all("wrong type" in e for e in failures.values() if not e.startswith("no one-to-one"))


@pytest.mark.skipif(not _HAVE_BFCL, reason="BFCL not fetched")
@pytest.mark.parametrize("category", BFCL_CATEGORIES)
def test_bfcl_corrupted_reference_answers_fail(category: str) -> None:
    rng = random.Random(0)
    checked = 0
    for case in load_bfcl(category):
        calls = bfcl.parse_openai_tool_calls(_reference_calls(case))
        (name, args), *_ = calls[0].items()
        # simple_363's reference names `find_closest`, which is not among its declared functions
        doc = next((f for f in case.functions if bfcl.openai_name(f["name"]) == name), None)
        required = [p for p in (doc or {}).get("parameters", {}).get("required", []) if p in args]
        if not required:
            continue
        p = rng.choice(required)
        dropped = [{name: {k: v for k, v in args.items() if k != p}}, *calls[1:]]
        assert not bfcl.grade(case, dropped).correct, case.id
        assert not bfcl.grade(case, [*calls, calls[0]]).correct, case.id  # extra call
        renamed = [{name + "_x": args}, *calls[1:]]
        assert not bfcl.grade(case, renamed).correct, case.id
        checked += 1
    assert checked > 0.8 * len(load_bfcl(category))


def test_qa_grades_take_best_gold() -> None:
    assert grade_qa("The Eiffel Tower", ["eiffel tower", "tour eiffel"]).em == 1.0
    g = grade_qa("Paris, France", ["Paris"])
    assert g.em == 0.0 and g.f1 == pytest.approx(2 / 3)


def test_crag_labels_and_score() -> None:
    labels = [crag_label("Paris", "paris"), crag_label("I don't know", "Paris"), crag_label("Lyon", "Paris"),
              crag_label("1.5", "one point five", alt_ans=["1.5"])]
    assert labels == ["correct", "missing", "incorrect", "correct"]
    assert crag_score(labels) == pytest.approx((1 + 0 - 1 + 1) / 4)


@pytest.mark.parametrize(
    "prediction, gold, ok",
    [
        ("Chief of Protocol", "Chief of Protocol", True),
        ("Chief of Protocol of the United States", "Chief of Protocol", True),
        ("yes", "Yes", True),
        ("yes, and no", "yes", False),  # yes/no needs exact match
        ("Paris or London or Rome or Berlin or Madrid or Lisbon or Vienna", "Paris", False),  # hedging
        ("Protocol Chief", "Chief of Protocol", False),
        (None, "x", False),
    ],
)
def test_agent_success(prediction: str | None, gold: str, ok: bool) -> None:
    assert agent_success(prediction, gold) is ok

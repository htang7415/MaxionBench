from __future__ import annotations

import json


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int, str]) -> None:
    print(json.dumps({"event": "conformance_test_start", "nodeid": nodeid}, sort_keys=True), flush=True)


def pytest_runtest_logreport(report) -> None:  # type: ignore[no-untyped-def]
    if report.when != "call":
        return
    print(
        json.dumps(
            {
                "event": "conformance_test_end",
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "duration_s": round(float(report.duration), 6),
            },
            sort_keys=True,
        ),
        flush=True,
    )

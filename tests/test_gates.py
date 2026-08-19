from __future__ import annotations

import json

from market_analogues.gates import GateReport, require_passed


def test_gate_keeps_immutable_history_and_latest_pointer(tmp_path) -> None:
    first = GateReport("task", True, {"run": 1})
    first.write(tmp_path)
    second = GateReport("task", True, {"run": 2})
    second.write(tmp_path)
    history = list((tmp_path / "history" / "task").glob("*.json"))
    assert len(history) == 2
    assert json.loads((tmp_path / "task.json").read_text())["metrics"]["run"] == 2
    require_passed(tmp_path, "task")

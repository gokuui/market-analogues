from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).parents[1]
        / "experiments" / "m04r" / "packed_bound_1pct.py"
    )
    spec = importlib.util.spec_from_file_location("m04r_packed_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_scan_payload_aggregates_every_pass_without_post_scan_failure() -> None:
    module = _module()
    cold = {"observed_peak_rss_mb": 210.0}
    first = [{"observed_peak_rss_mb": 240.0}]
    second = [{"observed_peak_rss_mb": 230.0}]
    payload = module._full_scan_payload(True, cold, first, second)
    assert payload["cold_scan"] is cold
    assert payload["warm_scans_first"] is first
    assert payload["warm_scans_second"] is second
    assert payload["scan_peak_rss_mb"] == 240.0
    assert payload["scan_io_mode"] == (
        "bounded positional reads over raw immutable pack"
    )

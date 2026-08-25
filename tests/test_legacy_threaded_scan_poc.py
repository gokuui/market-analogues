from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "experiments" / "m04r" / "legacy_threaded_scan_poc.py"
    spec = importlib.util.spec_from_file_location("legacy_threaded_scan_poc", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_ordered_map_preserves_order() -> None:
    module = _module()
    with ThreadPoolExecutor(max_workers=4) as executor:
        observed = list(module._bounded_ordered_map(
            executor, lambda value: value * value, range(20), 4,
        ))
    assert observed == [value * value for value in range(20)]

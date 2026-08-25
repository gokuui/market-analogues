from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "experiments" / "m04r" / "legacy_parallel_scan_poc.py"
    spec = importlib.util.spec_from_file_location("legacy_parallel_scan_poc", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partitions_cover_each_block_once() -> None:
    module = _module()
    partitions = module._partitions(41, 10, 2)
    assert sorted(value for part in partitions for value in part) == [0, 10, 20, 30, 40]
    assert set(partitions[0]).isdisjoint(partitions[1])

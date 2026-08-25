from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys


def _module():
    directory = Path(__file__).parents[1] / "experiments" / "m04r"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "m04r_parallel_certified_gate",
            directory / "parallel_certified_batch_gate.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))


MODULE = _module()


def _checkpoint() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": MODULE.GROUP_SCHEMA,
        "group_index": 0,
        "group_id": "0" * 16,
        "query_episode_ids": ["1" * 24],
        "elapsed_seconds": 100.0,
        "proposal_seconds": 70.0,
        "exact_total_seconds": 30.0,
        "peak_rss_mb": 500.0,
        "cases": [],
        "created_at": "2026-08-25T00:00:00+00:00",
    }
    payload["result_digest"] = MODULE.group_digest(payload)
    payload["checkpoint_integrity_digest"] = (
        MODULE.checkpoint_integrity_digest(payload)
    )
    return payload


def test_checkpoint_integrity_binds_runtime_but_result_digest_does_not() -> None:
    original = _checkpoint()
    changed = copy.deepcopy(original)
    changed["elapsed_seconds"] = 99.0

    assert MODULE.group_digest(changed) == original["result_digest"]
    assert (
        MODULE.checkpoint_integrity_digest(changed)
        != original["checkpoint_integrity_digest"]
    )


def test_checkpoint_integrity_ignores_only_creation_time_and_itself() -> None:
    original = _checkpoint()
    changed = copy.deepcopy(original)
    changed["created_at"] = "2027-01-01T00:00:00+00:00"
    changed["checkpoint_integrity_digest"] = "f" * 64

    assert (
        MODULE.checkpoint_integrity_digest(changed)
        == original["checkpoint_integrity_digest"]
    )


def test_group_id_binds_membership_and_controls() -> None:
    controls = {"processes": 8, "numba_threads_per_process": 1}
    arguments = (0, ["1" * 24], "2" * 64, "3" * 64, "4" * 64, controls)
    group_id = MODULE._group_id(*arguments)

    assert group_id == MODULE._group_id(*arguments)
    assert group_id != MODULE._group_id(
        0, ["5" * 24], "2" * 64, "3" * 64, "4" * 64, controls,
    )
    assert group_id != MODULE._group_id(
        0, ["1" * 24], "2" * 64, "3" * 64, "4" * 64,
        {**controls, "processes": 4},
    )


def test_real_checkpoint_fails_closed_on_timing_or_semantic_corruption() -> None:
    root = Path(__file__).parents[1]
    checkpoint_path = next((
        root / "config" / "data" / "analogues" / "poc" / "m04r"
        / "parallel-certified-batch-gate" / "groups"
    ).glob("00-*.json"))
    checkpoint = json.loads(checkpoint_path.read_text())
    baseline_payload = json.loads((
        root / "config" / "data" / "analogues" / "poc" / "m04r"
        / "certified-batch-gate" / "certified-batch-gate.json"
    ).read_text())
    baselines = {
        case["query_episode_id"]: case for case in baseline_payload["cases"]
    }
    arguments = {
        "index": checkpoint["group_index"],
        "group_id": checkpoint["group_id"],
        "query_ids": checkpoint["query_episode_ids"],
        "registry_digest": checkpoint["registry_digest"],
        "generation_id": checkpoint["generation_id"],
        "baseline_digest": checkpoint["baseline_batch_digest"],
        "controls": checkpoint["controls"],
        "baselines": baselines,
    }
    assert MODULE.valid_group(checkpoint, **arguments)

    timing_corruption = copy.deepcopy(checkpoint)
    timing_corruption["elapsed_seconds"] -= 1.0
    assert not MODULE.valid_group(timing_corruption, **arguments)

    semantic_corruption = copy.deepcopy(checkpoint)
    semantic_corruption["cases"][0]["matches"][0]["total_distance"] += 0.01
    semantic_corruption["result_digest"] = MODULE.group_digest(semantic_corruption)
    semantic_corruption["checkpoint_integrity_digest"] = (
        MODULE.checkpoint_integrity_digest(semantic_corruption)
    )
    assert not MODULE.valid_group(semantic_corruption, **arguments)

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import (  # noqa: E402
    m04r14_t14_10_wf03d_exclusion_repair_full as subject,
)


def test_full_repair_frozen_workload_is_complete() -> None:
    assert subject.EXPECTED_QUERIES == 3_936
    assert subject.TOP_K == 20
    assert subject.SUPERSET_K == 21
    assert sum(subject.EXPECTED_AFFECTED.values()) == 212
    assert subject.EXPECTED_AFFECTED_UNION == 173
    assert set(subject.EXPECTED_AFFECTED) == set(subject.METHODS)


def test_full_repair_uses_all_twelve_cores_without_oversubscription() -> None:
    assert subject.COMPOSITE_PROCESSES * subject.COMPOSITE_THREADS == 12
    assert subject.PRICE_PROCESSES * subject.PRICE_THREADS == 12


def test_affected_inventory_rejects_wrong_count() -> None:
    identifiers = {
        "composite": range(0, 78),
        "price_only": (*range(0, 39), *range(1_000, 1_055)),
        "deterministic_random": range(2_000, 2_016),
        "recent_return_volatility": range(3_000, 3_024),
    }
    value = {"methods": {
        method: {"affected_query_ids": [f"{index:024x}" for index in values]}
        for method, values in identifiers.items()
    }}
    assert {key: len(rows) for key, rows in subject._affected(value).items()} \
        == subject.EXPECTED_AFFECTED
    value["methods"]["composite"]["affected_query_ids"].pop()
    try:
        subject._affected(value)
    except subject.ExclusionRepairFullError:
        pass
    else:
        raise AssertionError("wrong affected inventory was accepted")

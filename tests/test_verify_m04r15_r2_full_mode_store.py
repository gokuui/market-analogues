from __future__ import annotations

import inspect
import json
from pathlib import Path

from experiments.m04r import verify_m04r15_r2_full_mode_store as verifier


ROOT = Path(__file__).resolve().parents[1]


def test_partitioning_reconstructs_frozen_128_groups() -> None:
    prereg = json.loads((ROOT / verifier.PREREG).read_text())
    results = [json.loads(line) for line in (ROOT / verifier.STORE / "RESULTS.jsonl").read_text().splitlines()]
    groups = verifier._groups([row["query_case_id"] for row in results], 128)
    assert len(groups) == 128
    assert [len(group) for group in groups] == prereg["partition_query_counts"]
    assert [verifier._stable(list(group)) for group in groups] == prereg["partition_query_digests"]


def test_independent_store_parser_reconstructs_seal_and_sparse_abstentions() -> None:
    prereg = json.loads((ROOT / verifier.PREREG).read_text())
    query_ids, actual, seal, coverage, hashes = verifier._load_actual(
        ROOT / verifier.STORE, prereg,
    )
    assert len(query_ids) == len(actual) == 3270 and len(hashes) == 128
    assert verifier._stable(hashes) == seal["partition_sha256_digest"]
    assert coverage["mode_status_counts"]["abstain_insufficient_complete_primary_members"] == 6
    for symbol in ("RELIW", "UOKA", "WINVW"):
        result = actual[f"nasdaq-{symbol}-shadow-current-252"]
        assert all(view["complete_members"] == 2 and
                   view["selection"]["member_to_mode"] == {}
                   for view in result["views"].values())


def test_verifier_source_does_not_import_production_mode_modules() -> None:
    source = inspect.getsource(verifier)
    assert "from market_analogues" not in source
    assert "import market_analogues" not in source
    assert "from experiments.m04r import m04r15_r2_full_mode_store" not in source

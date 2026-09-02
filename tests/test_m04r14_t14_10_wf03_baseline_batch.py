from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_baseline_batch as subject
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import verify_m04r14_t14_10_wf03_baseline_batch as verifier


def test_case_path_accepts_only_episode_identifiers(tmp_path: Path) -> None:
    query_id = "a" * 24
    assert subject._case_path(tmp_path, query_id) == tmp_path / f"{query_id}.json"
    with pytest.raises(subject.BaselineBatchError, match="query ID"):
        subject._case_path(tmp_path, "../unsafe")


def test_query_symbol_lookup_matches_packed_absent_symbol_semantics() -> None:
    assert subject._query_symbol_id(("A", "B"), "B") == 1
    assert subject._query_symbol_id(("A", "B"), "MISSING") is None


def test_verifier_oracle_sample_is_stratified_and_deterministic() -> None:
    rows = [
        {"fold_id": fold, "episode_id": f"{fold * 10 + value:024x}"}
        for fold in range(3) for value in range(7)
    ]
    first = verifier.select_oracle_sample(rows)
    assert first == verifier.select_oracle_sample(list(reversed(rows)))
    assert len(first) == 3 * verifier.SAMPLE_PER_FOLD


def test_valid_case_requires_seal_binding_and_neighbor_counts(tmp_path: Path) -> None:
    row = {"episode_id": "b" * 24, "case_id": "case-b"}
    path = subject._case_path(tmp_path, row["episode_id"])
    state = {
        "schema_version": "m04r14-wf03-baseline-batch-case-v2",
        "query_id": row["episode_id"], "case_id": row["case_id"],
        "feature_generation_id": "generation",
        "outcomes_or_labels_used": False,
        "random_neighbors": [{} for _ in range(20)],
        "rank_neighbors": [{} for _ in range(20)],
    }
    value = base._sealed(state, "case_digest")
    base._atomic(path, value)
    assert subject._valid_case(path, row, "generation") == value
    assert subject._valid_case(path, row, "changed") is None

"""Run the synthetic T14-09 evidence-card aggregation gate."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.m04r.m04r14_t14_09_evidence_card_oracle import reference_card
from market_analogues.evidence_cards import build_evidence_card
from market_analogues.types import stable_hash


SCHEMA = "m04r14-t14-09-evidence-card-synthetic-gate-v1"
CONTRACT = Path("config/m04r14-t14-09-evidence-card-contract.json")
CONTRACT_RECEIPT = Path(
    "config/data/analogues/m04r14/t14-09-evidence-card-contract-verification-v1/VERIFIED.json"
)
OUTPUT = Path(
    "config/data/analogues/m04r14/t14-09-evidence-card-synthetic-gate-v1"
)
HORIZONS = (5, 10, 20, 40, 60, 126)


class SyntheticEvidenceError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if type(value) is not dict:
        raise SyntheticEvidenceError(f"JSON object required: {path}")
    return value


def _fixture(kind: str) -> list[dict[str, Any]]:
    rows = []
    for rank in range(1, 21):
        symbol = f"S{rank}"
        if kind in {"same_and_duplicate", "mixed"} and rank == 1:
            symbol = "QUERY"
        if kind in {"same_and_duplicate", "mixed"} and rank == 3:
            symbol = "S2"
        eligibility = {}
        outcomes = {}
        for horizon in HORIZONS:
            complete = not (kind in {"censored", "mixed"} and rank in {4, 12} and horizon >= 20)
            eligible = complete
            reason = "eligible" if complete else "incomplete_horizon"
            if kind in {"embargo", "mixed"} and rank == 5 and horizon >= 20:
                eligible = False
                reason = "outcome_not_yet_observable"
            value: float | None = (rank - 10) / 50.0
            if kind in {"missing_measure", "mixed"} and rank == 7:
                value = None
            label = (
                "censored" if not complete else
                "ambiguous_same_first_touch_bar" if rank == 6 else
                "no_touch" if rank % 5 == 0 else
                "favorable_first" if rank % 2 == 0 else "adverse_first"
            )
            eligibility[str(horizon)] = {"eligible": eligible, "reason": reason}
            outcomes[str(horizon)] = {
                "complete": complete,
                "status": "complete" if complete else "source_end_before_horizon",
                "close_return": value,
                "benchmark_relative_return": None if rank == 8 else value,
                "maximum_favorable_excursion": None if value is None else value + .2,
                "maximum_adverse_excursion": None if value is None else value - .2,
                "mfe_atr": None if value is None else value * 4,
                "mae_atr": None if value is None else value * 2,
                "barrier_label": label if horizon == 20 else None,
            }
        rows.append({
            "query_case_id": f"synthetic-{kind}",
            "query_episode_id": f"query-{kind}", "query_symbol": "QUERY",
            "query_cutoff": "2026-03-30T00:00:00", "match_rank": rank,
            "matched_episode_id": f"{kind}-episode-{rank}",
            "matched_symbol": symbol, "matched_cutoff": f"2020-01-{rank:02d}",
            "total_distance": rank / 20.0,
            "component_distances": {"price": rank / 40.0}, "quality_tier": "A",
            "eligibility_by_horizon": eligibility, "outcomes_by_horizon": outcomes,
            "future_path_episode_reference": f"{kind}-episode-{rank}",
        })
    return rows


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    contract = _read(repository / CONTRACT)
    receipt = _read(repository / CONTRACT_RECEIPT)
    if not all((
        receipt.get("passed") is True,
        receipt.get("contract_digest") == contract.get("contract_digest"),
        receipt.get("query_level_outcome_aggregation_opened") is False,
    )):
        raise SyntheticEvidenceError("evidence contract prerequisite differs")
    started = perf_counter()
    families = (
        "baseline", "same_and_duplicate", "censored",
        "embargo", "missing_measure", "mixed",
    )
    digests = {}
    mutation_checks = {}
    for family in families:
        rows = _fixture(family)
        bindings = {
            "contract_digest": contract["contract_digest"],
            "provenance": {"fixture": family, "source": "synthetic_only"},
        }
        produced = build_evidence_card(rows, **bindings)
        expected = reference_card(rows, **bindings)
        if produced != expected:
            raise SyntheticEvidenceError(f"production/reference mismatch: {family}")
        if build_evidence_card(list(reversed(rows)), **bindings) != produced:
            raise SyntheticEvidenceError(f"row-order instability: {family}")
        changed = deepcopy(rows)
        changed[9]["outcomes_by_horizon"]["20"]["close_return"] = -99.0
        mutated = build_evidence_card(changed, **bindings)
        ranks_before = [row["match_rank"] for row in produced["raw_analogue_rows"]]
        ranks_after = [row["match_rank"] for row in mutated["raw_analogue_rows"]]
        if ranks_before != ranks_after or produced["card_digest"] == mutated["card_digest"]:
            raise SyntheticEvidenceError(f"outcome/rank isolation differs: {family}")
        encoded = json.dumps(produced, sort_keys=True, separators=(",", ":"), allow_nan=False)
        frame = pd.DataFrame({"family": [family], "card_json": [encoded]})
        temporary = repository / OUTPUT.parent / f".{OUTPUT.name}-{family}-{os.getpid()}.parquet"
        try:
            frame.to_parquet(temporary, index=False)
            restored = pd.read_parquet(temporary).iloc[0].card_json
            if restored != encoded:
                raise SyntheticEvidenceError(f"Parquet roundtrip differs: {family}")
        finally:
            temporary.unlink(missing_ok=True)
        digests[family] = produced["card_digest"]
        mutation_checks[family] = True
    state = {
        "schema_version": SCHEMA, "status": "passed", "passed": True,
        "contract_digest": contract["contract_digest"],
        "family_count": len(families), "family_card_digests": digests,
        "exact_production_reference_equality": True,
        "row_order_invariant": True,
        "outcome_mutation_preserved_ranks": all(mutation_checks.values()),
        "outcome_mutation_changed_summary_digest": all(mutation_checks.values()),
        "parquet_semantic_roundtrip": True,
        "query_level_real_outcome_aggregation_opened": False,
        "production_promotion_authorized": False,
    }
    return {
        **state, "elapsed_seconds": perf_counter() - started,
        "result_digest": stable_hash(state),
    }


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SyntheticEvidenceError("synthetic evidence gate root exists")
    path.mkdir(parents=False)
    descriptor = os.open(path / "RESULT.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({
            **value, "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    result = execute(repository)
    if not args.dry_run:
        _publish(repository / OUTPUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

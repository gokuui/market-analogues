"""Finalize the interrupted M04R-11 v1 producer as failed, without opening truth."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from m04r11_candidate_matrix import (
    FROZEN_GENERATION_ID, FROZEN_PROPOSAL_CONTRACT_DIGEST,
    FROZEN_REGISTRY_DIGEST, _checkpoint_valid,
)
from market_analogues.adapters import file_fingerprint, source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.m04r_validation_registry import validate_m04r_validation_registry
from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.representation import represent, representation_input_digest
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash


SCHEMA = "candidate-recall-producer-terminal-failure-v1"
EXPECTED_COMPLETED_CASES = 7
EXPECTED_FAILED_CASE_ID = "nasdaq-GABC-current-252"
EXPECTED_FALSE_GATES = ("cold_scan_at_most_120_seconds",)
EXPECTED_PRODUCER_CONTRACT_DIGEST = "e6d971081f288375df3ad66baad0300eb2d452860b464d6d9c111a6c75d657ad"
OMITTED = {"created_at", "failure_digest"}


def failure_digest(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key not in OMITTED})


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _validate_frozen_launch_contract(contract: dict[str, Any]) -> tuple[str, ...]:
    failures: list[str] = []
    try:
        manifest_path = Path(str(contract["physical_manifest_path"]))
        if not all((
            contract.get("schema_version") == "candidate-recall-producer-contract-v2",
            contract.get("contract_digest") == EXPECTED_PRODUCER_CONTRACT_DIGEST,
            contract.get("contract_digest") == stable_hash({
                key: value for key, value in contract.items()
                if key != "contract_digest"
            }),
            contract.get("registry_digest") == FROZEN_REGISTRY_DIGEST,
            contract.get("generation_id") == FROZEN_GENERATION_ID,
            contract.get("proposal_contract_digest")
            == FROZEN_PROPOSAL_CONTRACT_DIGEST,
            contract.get("real_forward_outcomes_accessed") is False,
            manifest_path.is_file(),
            file_fingerprint(manifest_path)
            == contract.get("physical_manifest_sha256"),
            contract.get("implementation_manifest", {}).get("digest")
            == stable_hash(contract.get("implementation_manifest", {}).get("files")),
        )):
            failures.append("frozen launch contract differs")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        failures.append(f"malformed launch contract:{type(exc).__name__}:{exc}")
    return tuple(failures)


def _require_exact_checkpoint_prefix(
    expected_names: set[str], observed_names: set[str],
) -> None:
    if observed_names != expected_names:
        raise ValueError("partial checkpoint filenames differ from registry prefix")


def _terminal_summary(
    index: int, case: dict[str, Any], payload: dict[str, Any],
) -> dict[str, Any]:
    false_gates = tuple(sorted(
        name for name, passed in payload["gates"].items() if passed is not True
    ))
    expected_pass = index < EXPECTED_COMPLETED_CASES - 1
    if payload.get("passed") is not expected_pass:
        raise ValueError(f"partial checkpoint pass status differs:{case['case_id']}")
    if expected_pass and false_gates or not expected_pass and (
        case["case_id"] != EXPECTED_FAILED_CASE_ID
        or false_gates != EXPECTED_FALSE_GATES
        or float(payload["cold_seconds"]) <= 120.0
        or float(payload["warm_second_seconds"]) > 60.0
        or float(payload["peak_rss_mb"]) > 1_024.0
    ):
        raise ValueError(f"terminal failure classification differs:{case['case_id']}")
    return {
        "ordinal": index,
        "registry_case_id": case["case_id"],
        "query_episode_id": case["episode_id"],
        "checkpoint_digest": payload["result_digest"],
        "checkpoint_integrity_digest": payload["checkpoint_integrity_digest"],
        "passed": payload["passed"],
        "false_gates": list(false_gates),
        "cold_seconds": payload["cold_seconds"],
        "warm_second_seconds": payload["warm_second_seconds"],
        "peak_rss_mb": payload["peak_rss_mb"],
    }


def _require_comparison_absent(comparison_root: Path) -> None:
    forbidden = (
        "RESULTS_OPENED.json", "candidate-comparison.json", "SEALED.json",
    )
    if any((comparison_root / name).exists() for name in forbidden):
        raise ValueError("candidate/authority comparison was already opened")


def _existing_failure_matches(
    existing: dict[str, Any], deterministic: dict[str, Any],
) -> bool:
    observed = {
        key: value for key, value in existing.items() if key not in OMITTED
    }
    return all((
        observed == deterministic,
        existing.get("failure_digest") == stable_hash(deterministic),
        existing.get("schema_version") == SCHEMA,
        existing.get("resume_authorized") is False,
        existing.get("authority_results_opened") is False,
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    expected_root = config.artifact_dir / "m04r11" / "candidate-pools-v1"
    if args.candidate_root.resolve() != expected_root.resolve():
        raise ValueError("failed v1 candidate root differs")
    if any((args.candidate_root / name).exists() for name in (
        "SEALED.json", "candidate-matrix.json",
    )):
        raise ValueError("terminal failure finalizer requires an unsealed partial run")
    registry = json.loads(args.registry.read_text())
    contract = json.loads((args.candidate_root / "candidate-contract.json").read_text())
    source = source_from_spec(config.datasets["nasdaq"])
    registry_failures = validate_m04r_validation_registry(source, args.registry.parent)
    contract_failures = _validate_frozen_launch_contract(contract)
    if registry_failures or contract_failures:
        raise ValueError(
            f"frozen registry/producer contract differs:"
            f"{registry_failures}:{contract_failures}"
        )
    generation = load_packed_generation(
        args.full_root / "store", FROZEN_GENERATION_ID,
        expected_provenance_digest=str(
            registry["search_contract"]["packed_provenance_digest"]
        ),
        verify_content=False, validate_records=False,
    )
    physical_rows = len(generation.rows) + len(generation.overflow)
    cases = list(registry["cases_data"])
    expected_completed = cases[:EXPECTED_COMPLETED_CASES]
    expected_paths = {
        f"{case['episode_id']}.json" for case in expected_completed
    }
    observed_paths = {
        path.name for path in (args.candidate_root / "cases").glob("*.json")
    }
    _require_exact_checkpoint_prefix(expected_paths, observed_paths)
    comparison_root = config.artifact_dir / "m04r11" / "candidate-comparison-v1"
    _require_comparison_absent(comparison_root)

    summaries: list[dict[str, Any]] = []
    for index, raw in enumerate(expected_completed):
        case = {**raw, "registry_digest": FROZEN_REGISTRY_DIGEST}
        payload = json.loads((
            args.candidate_root / "cases" / f"{case['episode_id']}.json"
        ).read_text())
        episode = build_episode(
            source, InstrumentKey("nasdaq", str(case["symbol"])),
            str(case["cutoff"]), int(case["lookback"]),
            str(case["representation_version"]),
        )
        if not _checkpoint_valid(
            payload, case=case, registry_digest=FROZEN_REGISTRY_DIGEST,
            generation_id=FROZEN_GENERATION_ID,
            proposal_contract_digest=FROZEN_PROPOSAL_CONTRACT_DIGEST,
            producer_contract_digest=str(contract["contract_digest"]),
            route_quotas=contract["route_quotas"], physical_rows=physical_rows,
            expected_query_start_ns=int(pd.Timestamp(
                episode.bars.timestamp.iloc[0],
            ).value),
            expected_latest_eligible_ns=int(
                latest_eligible_cutoff(episode, 60).value
            ),
            expected_representation_digest=representation_input_digest(
                represent(episode),
            ),
        ):
            raise ValueError(f"partial checkpoint differs:{case['case_id']}")
        summaries.append(_terminal_summary(index, case, payload))

    deterministic: dict[str, Any] = {
        "schema_version": SCHEMA,
        "registry_digest": FROZEN_REGISTRY_DIGEST,
        "producer_contract_digest": contract["contract_digest"],
        "generation_id": FROZEN_GENERATION_ID,
        "proposal_contract_digest": FROZEN_PROPOSAL_CONTRACT_DIGEST,
        "status": "terminal_performance_failure_after_interruption",
        "completed_cases": len(summaries),
        "completed_case_summaries": summaries,
        "failed_case_id": EXPECTED_FAILED_CASE_ID,
        "failed_gates": list(EXPECTED_FALSE_GATES),
        "remaining_query_episode_ids": [
            case["episode_id"] for case in cases[EXPECTED_COMPLETED_CASES:]
        ],
        "candidate_pools_sealed": False,
        "candidate_checkpoints_validated": True,
        "authority_results_opened": False,
        "candidate_authority_comparison_opened": False,
        "comparison_absence_policy": [
            str((comparison_root / name).resolve()) for name in (
                "RESULTS_OPENED.json", "candidate-comparison.json", "SEALED.json",
            )
        ],
        "production_promotion_authorized": False,
        "resume_authorized": False,
        "claims_policy": {
            "original_combined_gate_passed": False,
            "performance_exposed_cases": EXPECTED_COMPLETED_CASES,
            "remaining_untouched_performance_cases": len(cases) - EXPECTED_COMPLETED_CASES,
            "recall_holdout_cases_still_blind": len(cases),
        },
    }
    failure_path = args.candidate_root / "FAILED.json"
    if failure_path.exists():
        existing = json.loads(failure_path.read_text())
        if not _existing_failure_matches(existing, deterministic):
            raise ValueError("existing terminal failure evidence differs")
        if not (args.candidate_root / "FAILED.html").is_file():
            raise ValueError("existing terminal failure HTML is missing")
        print("terminal candidate failure is already finalized and revalidated")
        return 0
    payload = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "failure_digest": stable_hash(deterministic),
    }
    _atomic_text(
        args.candidate_root / "FAILED.html",
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-11 candidate producer failure</title></head><body>"
        "<h1>M04R-11 v1: terminal performance failure</h1>"
        "<p>No candidate pool seal was written and no authority result was opened.</p>"
        f"<pre>{escape(json.dumps(payload, indent=2, sort_keys=True))}</pre>"
        "</body></html>"
    )
    # The machine-readable terminal marker is published last.  If HTML writing
    # is interrupted, no FAILED.json exists and the complete validation can rerun.
    _atomic_json(failure_path, payload)
    print(json.dumps({
        "status": payload["status"], "completed_cases": len(summaries),
        "failed_case_id": payload["failed_case_id"],
        "failure_digest": payload["failure_digest"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

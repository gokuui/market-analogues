"""Durably count price-component threshold-closure admissions after frozen overflow."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from market_analogues.packed_bound_search import (
    PackedBoundQuery,
    packed_component_threshold_scan_contract,
    scan_packed_bound_threshold,
)
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_price_poc as price


SCHEMA = "m04r14-t14-10-wf03b-price-threshold-diagnostic-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03b-price-threshold-diagnostic-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03b_price_threshold_diagnostic_preregistered.json"
)
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03b_price_threshold_diagnostic.py",
    "src/market_analogues/packed_bound_search.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/quantized_bound.py",
)


class DiagnosticError(RuntimeError):
    pass


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repository, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise DiagnosticError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _failed_cases(repository: Path) -> list[dict[str, Any]]:
    root = repository / price.OUTPUT_RELATIVE
    result = base._read(root / "RESULT.json")
    base._validate_seal(result, "result_digest")
    if result.get("passed") is not False \
            or result.get("case_statuses") != ["frontier_overflow"] * 3:
        raise DiagnosticError("price overflow prerequisite differs")
    rows = []
    for ordinal, (label, query_id, _symbol, _cutoff) in enumerate(base.PROBES):
        case = root / "cases" / f"{ordinal:03d}-{label}-{query_id}"
        failed = base._read(case / "FAILED.json")
        base._validate_seal(failed, "failure_digest")
        proposal = base._read(case / "PROPOSAL_FORWARD.json")
        base._validate_seal(proposal)
        if failed.get("status") != "frontier_overflow" \
                or failed.get("frontier_rows") != base.MAXIMUM_FRONTIER \
                or proposal["proposal"]["query_episode_id"] != query_id:
            raise DiagnosticError("price overflow case differs")
        rows.append({
            "ordinal": ordinal, "label": label, "query_id": query_id,
            "threshold": failed["threshold"],
            "eligible_candidates": failed["eligible_candidates"],
            "exact_evaluated": failed["exact_evaluated"],
            "failed_sha256": base._sha(case / "FAILED.json"),
            "proposal_sha256": base._sha(case / "PROPOSAL_FORWARD.json"),
            "excluded_episode_ids": [
                row["episode_id"]
                for row in proposal["proposal"]["candidates"][:base.MAXIMUM_FRONTIER]
            ],
        })
    return rows


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise DiagnosticError("diagnostic preregistration requires a clean commit")
    cases = _failed_cases(repository)
    state = {
        "schema_version": SCHEMA, "status": "frozen_before_threshold_counts",
        "implementation_commit": _git(repository, "rev-parse", "HEAD"),
        "runtime_files": {path: base._sha(repository / path) for path in RUNTIME_FILES},
        "price_result_sha256": base._sha(
            repository / price.OUTPUT_RELATIVE / "RESULT.json"
        ),
        "cases": cases,
        "contract": packed_component_threshold_scan_contract("price"),
        "execution": {
            "component": "price", "branch_aware": True,
            "forward_block_rows": 4_096, "reverse_block_rows": 4_097,
            "exclude_exact_frontier": True,
            "consume": "count only; retain no candidate payload",
            "output_root": str((repository / OUTPUT_RELATIVE).resolve()),
        },
        "claims": {
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "authority_or_outcome_paths_accepted": False,
            "production_promotion_authorized": False,
        },
    }
    return base._sealed(state, "preregistration_digest")


def validate_preregistration(repository: Path, value: Mapping[str, Any]) -> None:
    base._validate_seal(value, "preregistration_digest")
    if value.get("schema_version") != SCHEMA or value.get("cases") != _failed_cases(repository):
        raise DiagnosticError("threshold diagnostic preregistration differs")
    head = value.get("implementation_commit")
    if type(head) is not str:
        raise DiagnosticError("threshold implementation commit differs")
    _git(repository, "merge-base", "--is-ancestor", head, "HEAD")
    for path, digest in value["runtime_files"].items():
        if base._sha(repository / path) != digest:
            raise DiagnosticError(f"threshold runtime differs: {path}")
        blob = subprocess.run(["git", "show", f"{head}:{path}"], cwd=repository,
                              capture_output=True, check=False)
        if blob.returncode or sha256(blob.stdout).hexdigest() != digest:
            raise DiagnosticError(f"threshold implementation binding differs: {path}")


def execute(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    validate_preregistration(repository, preregistration)
    _registry, by_id = base._registry(repository)
    resident = base._resident()
    root = repository / OUTPUT_RELATIVE
    if root.exists() or root.is_symlink():
        raise DiagnosticError("threshold diagnostic output exists")
    root.mkdir(parents=True)
    base._atomic(root / "CONTRACT.json", preregistration)
    outputs = []
    for case in preregistration["cases"]:
        source, episode, request, _packed = base._context(
            repository, by_id[case["query_id"]],
        )
        packed = PackedBoundQuery(
            episode.key.id, episode.key.instrument.source_symbol,
            int(episode.bars.timestamp.iloc[0].value),
            int(latest_eligible_cutoff(episode, 60).value), represent(episode),
            request.quality_tiers,
        )
        excluded = frozenset(case["excluded_episode_ids"])
        reports = []
        counts = []
        for block_rows, order in ((4_096, "forward"), (4_097, "reverse")):
            count = [0]
            report = scan_packed_bound_threshold(
                Path(resident["store_root"]), base.GENERATION_ID, packed,
                component="price", branch_aware=True,
                upper_inclusive=case["threshold"],
                excluded_episode_ids=excluded, block_rows=block_rows,
                block_order=order, verify_content=False,
                expected_provenance_digest=base.PROVENANCE_DIGEST,
                consume=lambda rows, value=count: value.__setitem__(0, value[0] + len(rows)),
            )
            if count[0] != report.admitted_rows:
                raise DiagnosticError("threshold callback accounting differs")
            reports.append(asdict(report)); counts.append(count[0])
        omitted = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
        semantic = lambda row: {key: value for key, value in row.items() if key not in omitted}
        if semantic(reports[0]) != semantic(reports[1]) or counts[0] != counts[1]:
            raise DiagnosticError("threshold traversal parity differs")
        total_below = counts[0] + case["exact_evaluated"]
        outputs.append({
            "ordinal": case["ordinal"], "label": case["label"],
            "query_id": case["query_id"], "threshold": case["threshold"],
            "eligible_candidates": case["eligible_candidates"],
            "already_exact": case["exact_evaluated"],
            "additional_admitted": counts[0],
            "total_at_or_below_threshold": total_below,
            "admission_fraction": total_below / case["eligible_candidates"],
            "forward_report": reports[0], "reverse_report": reports[1],
        })
    state = {
        "schema_version": "m04r14-t14-10-wf03b-price-threshold-diagnostic-result-v1",
        "status": "complete", "passed": True, "cases": outputs,
        "case_digest": stable_hash(outputs),
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    result = base._sealed(state, "result_digest")
    base._atomic(root / "RESULT.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    path = repository / PREREGISTRATION_RELATIVE
    if args.mode == "preregister":
        base._atomic(path, build_preregistration(repository)); return 0
    result = execute(repository, base._read(path))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

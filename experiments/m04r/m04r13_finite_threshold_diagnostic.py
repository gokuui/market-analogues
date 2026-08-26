"""Create-only, truth-blind finite-threshold preflight for M04R-13.

This development diagnostic accepts no authority, comparison, or outcome path.
It runs the four frozen searches serially in fresh child processes, using a
private temporary producer tree, and publishes only compact semantic summaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import m04r13_threaded_certified_exposed as producer
from market_analogues.types import stable_hash


SCHEMA = "m04r13-truth-blind-finite-threshold-diagnostic-v1"
CASE_SCHEMA = "m04r13-truth-blind-finite-threshold-case-v1"
DIAGNOSTIC_RELATIVE = Path(
    "config/data/analogues/m04r13/finite-threshold-diagnostic-v2/DIAGNOSTIC.json"
)


class DiagnosticError(ValueError):
    pass


def _optional_hex(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) is not float or not isfinite(value):
        raise DiagnosticError("threshold is not a finite float")
    return value.hex()


def summarize_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    certificate = payload.get("certificate")
    if type(certificate) is not dict:
        raise DiagnosticError("certificate is absent")
    try:
        json.dumps(certificate, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DiagnosticError("certificate is not strict-JSON serializable") from exc
    rounds = certificate.get("rounds")
    closures = certificate.get("threshold_closure_passes")
    if type(rounds) is not list or not rounds or type(closures) is not list:
        raise DiagnosticError("certificate threshold states differ")
    round_summaries = []
    for row in rounds:
        if type(row) is not dict or not all((
            type(row.get("frontier_rows")) is int,
            type(row.get("selected_rows")) is int,
            type(row.get("certified")) is bool,
        )):
            raise DiagnosticError("round state differs")
        round_summaries.append({
            "frontier_rows": row["frontier_rows"],
            "selected_rows": row["selected_rows"],
            "constrained_threshold_hex": _optional_hex(
                row.get("constrained_threshold")
            ),
            "next_lower_bound_hex": _optional_hex(row.get("next_lower_bound")),
            "certified": row["certified"],
        })
    closure_summaries = []
    for row in closures:
        if type(row) is not dict or not all((
            type(row.get("admitted_rows")) is int,
            type(row.get("selected_rows")) is int,
            type(row.get("certified")) is bool,
        )):
            raise DiagnosticError("closure state differs")
        closure_summaries.append({
            "lower_exclusive_hex": _optional_hex(row.get("lower_exclusive")),
            "upper_inclusive_hex": _optional_hex(row.get("upper_inclusive")),
            "admitted_rows": row["admitted_rows"],
            "selected_rows": row["selected_rows"],
            "resulting_threshold_hex": _optional_hex(
                row.get("resulting_threshold")
            ),
            "minimum_packed_unclassified_bound_hex": _optional_hex(
                row.get("minimum_packed_unclassified_bound")
            ),
            "minimum_native_pruned_bound_hex": _optional_hex(
                row.get("minimum_native_pruned_bound")
            ),
            "certified": row["certified"],
        })
    deterministic = {
        "schema_version": CASE_SCHEMA,
        "ordinal": producer.FROZEN_QUERY_IDS.index(payload.get("query_episode_id")),
        "registry_case_id": payload.get("registry_case_id"),
        "query_episode_id": payload.get("query_episode_id"),
        "forward_candidate_digest": payload.get("forward_proposal", {}).get(
            "candidate_digest"
        ),
        "forward_result_digest": payload.get("forward_proposal", {}).get(
            "result_digest"
        ),
        "reverse_candidate_digest": payload.get("reverse_proposal", {}).get(
            "candidate_digest"
        ),
        "reverse_result_digest": payload.get("reverse_proposal", {}).get(
            "result_digest"
        ),
        "certificate_result_digest": certificate.get("result_digest"),
        "case_result_digest": payload.get("result_digest"),
        "query_binding": payload.get("query_binding"),
        "stop_threshold_hex": _optional_hex(certificate.get("stop_threshold")),
        "next_lower_bound_hex": _optional_hex(certificate.get("next_lower_bound")),
        "minimum_native_pruned_bound_hex": _optional_hex(
            certificate.get("minimum_native_pruned_bound")
        ),
        "rounds": round_summaries,
        "threshold_closure_passes": closure_summaries,
        "semantic_passed": payload.get("semantic_passed"),
        "performance_passed_diagnostic": payload.get("performance_passed"),
        "all_thresholds_finite": True,
        "real_forward_outcomes_accessed": False,
    }
    digests = (
        "forward_candidate_digest", "forward_result_digest",
        "reverse_candidate_digest", "reverse_result_digest",
        "certificate_result_digest", "case_result_digest",
    )
    ordinal = deterministic["ordinal"]
    if not all((
        deterministic["registry_case_id"] == producer.FROZEN_CASE_IDS[ordinal],
        deterministic["semantic_passed"] is True,
        type(deterministic["performance_passed_diagnostic"]) is bool,
        payload.get("proposal_semantic_exact") is True,
        payload.get("real_forward_outcomes_accessed") is False,
        all(producer._is_digest(deterministic[key]) for key in digests),
        deterministic["forward_candidate_digest"]
        == deterministic["reverse_candidate_digest"],
    )):
        raise DiagnosticError("case semantic evidence differs")
    return {**deterministic, "result_digest": stable_hash(deterministic)}


def diagnostic_payload(
    *, repository: Path, implementation_git: Mapping[str, Any],
    registry_digest: str, registry_cases_digest: str,
    resident: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if [row.get("query_episode_id") for row in cases] != list(
        producer.FROZEN_QUERY_IDS
    ):
        raise DiagnosticError("diagnostic case order differs")
    deterministic = {
        "schema_version": SCHEMA,
        "status": "truth_blind_finite_thresholds_verified",
        "development_only": True,
        "authority_or_outcome_paths_accepted": False,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "implementation_git": dict(implementation_git),
        "environment": producer._environment_binding(),
        "execution": producer._execution_policy(),
        "config_sha256": producer._sha(repository / producer.CONFIG_RELATIVE),
        "diagnostic_path": str((repository / DIAGNOSTIC_RELATIVE).resolve()),
        "roots": {
            "config_path": str((repository / producer.CONFIG_RELATIVE).resolve()),
            "registry_root": str((repository / producer.REGISTRY_RELATIVE).resolve()),
            "source_full_root": str(
                (repository / producer.SOURCE_FULL_RELATIVE).resolve()
            ),
            "resident_root": str(producer.RESIDENT_ROOT.resolve()),
        },
        "generation_id": producer.GENERATION_ID,
        "provenance_digest": producer.PROVENANCE_DIGEST,
        "registry_digest": registry_digest,
        "registry_cases_digest": registry_cases_digest,
        "query_ids": list(producer.FROZEN_QUERY_IDS),
        "case_ids": list(producer.FROZEN_CASE_IDS),
        "resident": {
            "content_digest": resident["content_digest"],
            "ready_digest": resident["ready_digest"],
            "identity_digest": resident["identity_digest"],
        },
        "cases": [dict(row) for row in cases],
        "all_thresholds_finite": True,
        "all_semantic_checks_passed": True,
    }
    return {
        **deterministic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }


def _protected_paths(repository: Path) -> tuple[Path, ...]:
    return (
        repository,
        repository / producer.CONFIG_RELATIVE,
        repository / producer.REGISTRY_RELATIVE,
        repository / producer.SOURCE_FULL_RELATIVE,
        producer.RESIDENT_ROOT,
        repository / producer.DIAGNOSTIC_RELATIVE,
        repository / producer.OUTPUT_RELATIVE,
        repository / producer.PREREG_RELATIVE,
    )


def _unalias(path: Path, *, require_directory: bool) -> Path:
    lexical = path.absolute()
    if lexical.resolve() != lexical:
        raise DiagnosticError("diagnostic path ancestry is aliased")
    cursor = lexical
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise DiagnosticError("diagnostic path ancestry contains a symlink")
        cursor = cursor.parent
    if require_directory and (not lexical.is_dir() or lexical.is_symlink()):
        raise DiagnosticError("private scratch root differs")
    return lexical


def validate_scratch_root(repository: Path, scratch: Path) -> Path:
    scratch = _unalias(scratch, require_directory=True)
    for protected in _protected_paths(repository):
        resolved = protected.absolute().resolve()
        if (scratch == resolved or scratch.is_relative_to(resolved)
                or resolved.is_relative_to(scratch)):
            raise DiagnosticError("private scratch overlaps protected input")
    return scratch


def validate_diagnostic_output_path(repository: Path) -> Path:
    expected = (repository / DIAGNOSTIC_RELATIVE).absolute()
    if expected.parent != (
        repository
        / "config/data/analogues/m04r13/finite-threshold-diagnostic-v2"
    ).absolute():
        raise DiagnosticError("finite-threshold diagnostic path differs")
    _unalias(expected.parent, require_directory=expected.parent.exists())
    if expected.exists() or expected.is_symlink():
        raise DiagnosticError("finite-threshold diagnostic is create-only")
    return expected


def _inputs(
    repository: Path, scratch_output: Path, git: Mapping[str, Any],
) -> tuple[producer.Inputs, dict[str, Any], dict[str, Any]]:
    registry_digest, cases = producer._registry_cases(
        repository, repository / producer.REGISTRY_RELATIVE,
    )
    if registry_digest != producer.REGISTRY_DIGEST:
        raise DiagnosticError("registry digest differs")
    resident = producer.resident_full(
        repository / producer.SOURCE_FULL_RELATIVE / "store",
        producer.RESIDENT_ROOT, producer.GENERATION_ID,
        producer.PROVENANCE_DIGEST, producer.RESIDENT_RESERVE_BYTES,
    )
    contract = stable_hash({
        "diagnostic_schema": SCHEMA, "implementation_git": dict(git),
        "environment": producer._environment_binding(),
        "execution": producer._execution_policy(),
        "registry_digest": registry_digest,
        "query_ids": list(producer.FROZEN_QUERY_IDS),
        "resident_identity_digest": resident["identity_digest"],
    })
    inputs = producer.Inputs(
        repository, repository / producer.CONFIG_RELATIVE,
        repository / producer.REGISTRY_RELATIVE,
        repository / producer.SOURCE_FULL_RELATIVE / "store",
        producer.RESIDENT_ROOT, scratch_output, producer.GENERATION_ID,
        producer.PROVENANCE_DIGEST, producer.RESIDENT_RESERVE_BYTES,
        registry_digest, cases, contract,
    )
    return inputs, {"diagnostic_contract_digest": contract}, resident


def run_child(repository: Path, scratch_root: Path, ordinal: int) -> dict[str, Any]:
    scratch_root = validate_scratch_root(repository, scratch_root)
    git = producer._implementation_git(repository)
    output = scratch_root / "case-output"
    output.mkdir()
    (output / "cases").mkdir()
    inputs, contract, resident = _inputs(repository, output, git)
    payload = producer.run_case(
        inputs, contract, resident, inputs.cases[ordinal],
    )
    producer.validate_case(payload, inputs, inputs.cases[ordinal], resident)
    return summarize_case(payload)


def run_diagnostic(
    repository: Path, *, runner: Any = subprocess.run,
) -> dict[str, Any]:
    repository = repository.resolve()
    path = validate_diagnostic_output_path(repository)
    git = producer._implementation_git(repository)
    registry_digest, registry_cases = producer._registry_cases(
        repository, repository / producer.REGISTRY_RELATIVE,
    )
    resident = producer.resident_full(
        repository / producer.SOURCE_FULL_RELATIVE / "store",
        producer.RESIDENT_ROOT, producer.GENERATION_ID,
        producer.PROVENANCE_DIGEST, producer.RESIDENT_RESERVE_BYTES,
    )
    summaries: list[dict[str, Any]] = []
    temporary = Path(tempfile.mkdtemp(prefix="m04r13-finite-threshold-"))
    try:
        validate_scratch_root(repository, temporary)
        for ordinal in range(len(producer.FROZEN_QUERY_IDS)):
            case_root = temporary / f"case-{ordinal:02d}"
            case_root.mkdir()
            result_path = case_root / "SUMMARY.json"
            command = [
                sys.executable, str(Path(__file__).resolve()), "_case-child",
                "--scratch-root", str(case_root), "--case-ordinal", str(ordinal),
                "--result-path", str(result_path),
            ]
            completed = runner(command, check=False)
            if completed.returncode != 0:
                raise DiagnosticError(f"finite-threshold child failed: {ordinal}")
            summaries.append(producer._read_json(result_path))
    finally:
        shutil.rmtree(temporary)
    payload = diagnostic_payload(
        repository=repository, implementation_git=git,
        registry_digest=registry_digest,
        registry_cases_digest=stable_hash([
            case.registry_case for case in registry_cases
        ]), resident=resident, cases=summaries,
    )
    validation_inputs = producer.Inputs(
        repository, repository / producer.CONFIG_RELATIVE,
        repository / producer.REGISTRY_RELATIVE,
        repository / producer.SOURCE_FULL_RELATIVE / "store",
        producer.RESIDENT_ROOT, repository / producer.OUTPUT_RELATIVE,
        producer.GENERATION_ID, producer.PROVENANCE_DIGEST,
        producer.RESIDENT_RESERVE_BYTES, registry_digest, registry_cases,
        "finite-threshold-diagnostic",
    )
    producer.validate_finite_threshold_diagnostic(
        payload, repository=repository, expected_git=git,
        registry_cases_digest=stable_hash([
            case.registry_case for case in registry_cases
        ]), resident=resident,
        expected_query_bindings=producer.diagnostic_query_bindings(
            validation_inputs
        ),
    )
    validate_diagnostic_output_path(repository)
    producer._atomic(path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    child = commands.add_parser("_case-child")
    child.add_argument("--scratch-root", type=Path, required=True)
    child.add_argument("--case-ordinal", type=int, required=True)
    child.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    if args.command == "run":
        payload = run_diagnostic(repository)
    else:
        if args.case_ordinal not in range(len(producer.FROZEN_QUERY_IDS)):
            raise DiagnosticError("case ordinal differs")
        scratch = args.scratch_root.absolute()
        result_path = args.result_path.absolute()
        if scratch.is_symlink() or result_path.parent != scratch or result_path.exists():
            raise DiagnosticError("private child path differs")
        payload = run_child(repository, scratch, args.case_ordinal)
        producer._atomic(result_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

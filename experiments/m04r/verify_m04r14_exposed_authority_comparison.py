"""Independent verifier for the M04R-14 exposed-authority comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_all60_contract as contract


SCHEMA = "m04r14-exposed-authority-comparison-verification-v1"
COMPARISON = Path("config/data/analogues/m04r14/exposed-authority-comparison-v1")
CANDIDATE = Path("config/data/analogues/m04r14/throughput-development-poc-v1")
AUTHORITY = Path("config/data/analogues/m04r11/authorities-sealed-v4")
AUTHORITY_VERIFICATION = Path(
    "config/data/analogues/m04r11/authority-verification-v4/"
    "m04r11-authority-verification.json"
)
VERIFICATION = Path(
    "config/data/analogues/m04r14/exposed-authority-comparison-v1-verification"
)
COMPARATOR_SOURCE = Path("experiments/m04r/m04r14_exposed_authority_comparison.py")
STABLE_CERTIFICATE_FIELDS = (
    "contract_digest", "eligible_candidates", "exact_evaluated", "generation_id",
    "input_digest", "maximum_quantized_bound_excess", "minimum_native_pruned_bound",
    "native_bound_accounting", "query_episode_id", "safely_pruned", "schema_version",
    "stop_threshold", "stopped_early", "threshold_closure_passes",
)


class VerificationError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"regular file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise VerificationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                VerificationError(f"non-finite JSON: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise VerificationError(f"JSON object required: {path}")
    return value, raw


def _digest(value: Any) -> str:
    return contract.stable_digest(value)


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _git(repository: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True, check=False,
    )
    if completed.returncode:
        raise VerificationError(f"git validation failed: {' '.join(args)}")
    return completed.stdout


def _case_map(root: Path) -> tuple[dict[str, tuple[dict[str, Any], str]], list[dict[str, str]]]:
    case_root = root / "cases"
    if case_root.is_symlink() or not case_root.is_dir():
        raise VerificationError(f"case directory required: {case_root}")
    rows: dict[str, tuple[dict[str, Any], str]] = {}
    manifest: list[dict[str, str]] = []
    paths = sorted(case_root.glob("*.json"))
    if len(paths) != 60 or any(path.is_symlink() for path in paths):
        raise VerificationError(f"case tree differs: {root}")
    for path in paths:
        value, raw = _read(path)
        case_id = value.get("registry_case_id")
        if type(case_id) is not str or case_id in rows:
            raise VerificationError(f"case identity differs: {path}")
        if path.stem != value.get("query_episode_id"):
            raise VerificationError(f"case filename differs: {path}")
        digest = _sha(raw)
        rows[case_id] = (value, digest)
        manifest.append({"path": path.relative_to(root).as_posix(), "sha256": digest})
    return rows, manifest


def _gates(candidate: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, bool]:
    candidate_certificate = candidate.get("certificate", {})
    authority_certificate = authority.get("certificate", {})
    return {
        "case_id_equal": candidate.get("registry_case_id") == authority.get("registry_case_id"),
        "query_id_equal": candidate.get("query_episode_id") == authority.get("query_episode_id"),
        "ordered_matches_equal": candidate.get("matches") == authority.get("matches"),
        "stock_prefix_equal": candidate.get("query_stock_prefix") == authority.get("query_stock_prefix"),
        "benchmark_prefix_equal": candidate.get("query_benchmark_prefix") == authority.get("query_benchmark_prefix"),
        "stable_certificate_fields_equal": all(
            candidate_certificate.get(key) == authority_certificate.get(key)
            for key in STABLE_CERTIFICATE_FIELDS
        ),
        "candidate_gate_passed": candidate.get("gate_passed") is True,
        "authority_gate_passed": authority.get("gate_passed") is True,
    }


def verify(comparison_root: Path, *, repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    root = comparison_root.resolve(strict=True)
    if root != (repository / COMPARISON).resolve(strict=True):
        raise VerificationError("comparison root differs")
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != {"RESULTS_OPENED.json", "COMPARISON.json"} \
            or any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise VerificationError("comparison tree differs")

    marker, marker_raw = _read(root / "RESULTS_OPENED.json")
    comparison, comparison_raw = _read(root / "COMPARISON.json")
    marker_keys = {
        "schema_version", "status", "git_head", "candidate_result_digest",
        "candidate_verification_digest", "authority_root",
        "authority_or_outcome_read_before_marker", "marker_digest", "created_at",
    }
    if set(marker) != marker_keys:
        raise VerificationError("marker keys differ")
    marker_state = {key: value for key, value in marker.items() if key not in {"created_at", "marker_digest"}}
    if marker.get("marker_digest") != _digest(marker_state):
        raise VerificationError("marker digest differs")
    if not all((marker.get("schema_version") == "m04r14-exposed-authority-comparison-v1",
                marker.get("status") == "results_opened",
                marker.get("authority_or_outcome_read_before_marker") is False,
                marker.get("authority_root") == str((repository / AUTHORITY).resolve()))):
        raise VerificationError("marker contract differs")

    marker_head = marker.get("git_head")
    if type(marker_head) is not str or len(marker_head) != 40:
        raise VerificationError("marker Git head differs")
    _git(repository, "cat-file", "-e", f"{marker_head}^{{commit}}")
    committed_source = _git(repository, "show", f"{marker_head}:{COMPARATOR_SOURCE.as_posix()}")
    if committed_source != (repository / COMPARATOR_SOURCE).read_bytes():
        raise VerificationError("marker-bound comparator source differs")

    candidate_result, candidate_result_raw = _read(repository / CANDIDATE / "RESULT.json")
    candidate_verified, _ = _read(
        repository / "config/data/analogues/m04r14/throughput-development-poc-v1-verification/VERIFIED.json"
    )
    authority_verified, authority_verified_raw = _read(repository / AUTHORITY_VERIFICATION)
    if not all((marker["candidate_result_digest"] == candidate_result.get("result_digest"),
                marker["candidate_verification_digest"] == candidate_verified.get("result_digest"),
                candidate_verified.get("passed") is True,
                authority_verified.get("failures") == [],
                authority_verified.get("result_digest") ==
                "99d11756ed7542714635eca3fb75a43faed93881121d1f22d8d87426f4d6b190")):
        raise VerificationError("upstream evidence binding differs")

    candidates, candidate_manifest = _case_map(repository / CANDIDATE)
    authorities, authority_manifest = _case_map(repository / AUTHORITY)
    if set(candidates) != set(authorities):
        raise VerificationError("case sets differ")
    rows: list[dict[str, Any]] = []
    ordered_matches = 0
    for case_id in sorted(candidates):
        candidate, candidate_sha = candidates[case_id]
        authority, authority_sha = authorities[case_id]
        gates = _gates(candidate, authority)
        matches = candidate.get("matches")
        if type(matches) is not list or len(matches) != 20:
            raise VerificationError(f"match count differs: {case_id}")
        ordered_matches += len(matches)
        rows.append({
            "case_id": case_id, "query_id": candidate.get("query_episode_id"),
            "matches": len(matches), "candidate_sha256": candidate_sha,
            "authority_sha256": authority_sha, "gates": gates,
            "passed": all(gates.values()),
        })

    expected_keys = {
        "schema_version", "status", "marker_digest", "git_head",
        "candidate_result_digest", "candidate_verification_digest",
        "authority_verification_digest", "authority_verification_sha256", "cases",
        "ordered_matches", "all_passed", "rows", "production_promotion_authorized",
        "result_digest", "created_at",
    }
    if set(comparison) != expected_keys:
        raise VerificationError("comparison keys differ")
    result_input = {key: value for key, value in comparison.items() if key not in {"created_at", "result_digest"}}
    if not all((comparison.get("result_digest") == _digest(result_input),
                comparison.get("schema_version") == "m04r14-exposed-authority-comparison-v1",
                comparison.get("status") == "complete",
                comparison.get("marker_digest") == marker.get("marker_digest"),
                comparison.get("git_head") == marker_head,
                comparison.get("candidate_result_digest") == candidate_result.get("result_digest"),
                comparison.get("candidate_verification_digest") == candidate_verified.get("result_digest"),
                comparison.get("authority_verification_digest") == authority_verified.get("result_digest"),
                comparison.get("authority_verification_sha256") == _sha(authority_verified_raw),
                comparison.get("cases") == 60, comparison.get("ordered_matches") == ordered_matches == 1200,
                comparison.get("all_passed") is True,
                comparison.get("production_promotion_authorized") is False,
                comparison.get("rows") == rows, all(row["passed"] for row in rows))):
        raise VerificationError("comparison reconstruction differs")

    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "comparison_root": str(root), "comparison_result_digest": comparison["result_digest"],
        "comparison_result_sha256": _sha(comparison_raw), "marker_sha256": _sha(marker_raw),
        "marker_git_head": marker_head, "candidate_result_sha256": _sha(candidate_result_raw),
        "authority_verification_sha256": _sha(authority_verified_raw),
        "verified_cases": 60, "verified_ordered_matches": 1200,
        "candidate_manifest": candidate_manifest,
        "candidate_manifest_digest": _digest(candidate_manifest),
        "authority_manifest": authority_manifest,
        "authority_manifest_digest": _digest(authority_manifest),
        "production_promotion_authorized": False,
    }
    state["result_digest"] = _digest(state)
    return state


def _publish(path: Path, state: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise VerificationError("verification root exists")
    path.mkdir(parents=False)
    receipt = {**state, "created_at": datetime.now(timezone.utc).isoformat()}
    target = path / "VERIFIED.json"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(contract.canonical_bytes(receipt) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--verification-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    comparison = args.comparison_root or repository / COMPARISON
    output = args.verification_root or repository / VERIFICATION
    state = verify(comparison, repository=repository)
    if not args.dry_run:
        _publish(output, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

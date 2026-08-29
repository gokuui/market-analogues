"""Independent verifier for the M04R-14 grouped-throughput evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r14_all60_contract as contract


SCHEMA = "m04r14-throughput-verification-v1"
CANDIDATE = Path("config/data/analogues/m04r14/throughput-development-poc-v1")
BASELINE = Path("config/data/analogues/m04r14/all60-certified-development-v3")
VERIFICATION = Path(
    "config/data/analogues/m04r14/throughput-development-poc-v1-verification"
)
CASE_RESULT_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds",
    "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
    "peak_rss_mb", "result_digest", "checkpoint_integrity_digest",
}
CONTROLS = {
    "block_rows": 4_096, "deferred_alignments": True,
    "exact_workers_per_process": 1, "initial_frontier_rows": 1_000,
    "maximum_frontier_rows": 16_384, "numba_threads_per_process": 1,
    "requested_positions": True, "seed_rows": 512,
    "sorted_joined_iqr_merge": True, "vector_lower_bounds": True,
}


class VerificationError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"regular file required: {path}")
    raw = path.read_bytes()
    pairs_seen: list[str] = []

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise VerificationError(f"duplicate JSON key: {key}")
            result[key] = value
        pairs_seen.extend(result)
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


def _finite(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and value >= 0


def _case_result_digest(value: Mapping[str, Any]) -> str:
    return _digest({k: v for k, v in value.items() if k not in CASE_RESULT_OMITTED})


def _case_integrity_digest(value: Mapping[str, Any]) -> str:
    return _digest({k: v for k, v in value.items()
                    if k not in {"created_at", "checkpoint_integrity_digest"}})


def _baseline(repository: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = repository / BASELINE
    complete, _ = _read(root / "COMPLETE.json")
    semantics, _ = _read(root / "SEMANTICS.json")
    measurements, _ = _read(root / "MEASUREMENTS.json")
    if not all((complete.get("status") == "complete",
                complete.get("semantic_passed") is True,
                semantics.get("semantic_passed") is True,
                semantics.get("forward_reverse_equal") is True,
                measurements.get("summary", {}).get("cases") == 60)):
        raise VerificationError("V3 baseline differs")
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "cases").glob("*/CASE.json")):
        value, _ = _read(path)
        semantic, measurement = value.get("semantic"), value.get("measurement")
        if type(semantic) is not dict or type(measurement) is not dict:
            raise VerificationError("V3 case differs")
        case_id = semantic.get("case_id")
        if type(case_id) is not str or case_id in rows:
            raise VerificationError("V3 case ID differs")
        rows[case_id] = {"semantic": semantic, "measurement": measurement}
    if len(rows) != 60:
        raise VerificationError("V3 case count differs")
    binding = {
        "complete_digest": complete["complete_digest"],
        "semantic_digest": semantics["semantic_digest"],
        "measurement_digest": measurements["measurement_digest"],
        "serial_total_end_to_end_seconds": measurements["summary"][
            "total_end_to_end_seconds"
        ],
    }
    return rows, binding


def verify(candidate_root: Path, *, repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    root = candidate_root.resolve(strict=True)
    if root != (repository / CANDIDATE).resolve(strict=True):
        raise VerificationError("candidate root differs")
    directories = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir()}
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if directories != {"cases"} or len(files) != 61 or "RESULT.json" not in files \
            or any(p.is_symlink() for p in root.rglob("*")):
        raise VerificationError("candidate tree differs")
    result, result_raw = _read(root / "RESULT.json")
    expected_result_keys = {
        "all_semantics_equal", "authority_or_outcome_accessed", "baseline",
        "baseline_serial_service_seconds", "cases", "comparisons", "controls",
        "created_at", "development_only", "groups", "registry_digest",
        "result_digest", "schema_version", "selected_case_ids",
        "speedup_vs_serial_service_sum", "status", "wall_seconds",
    }
    if set(result) != expected_result_keys:
        raise VerificationError("result keys differ")
    deterministic = {k: v for k, v in result.items() if k != "created_at"}
    stored_digest = deterministic.pop("result_digest")
    deterministic["result_digest"] = stored_digest
    digest_input = {k: v for k, v in result.items() if k not in {"created_at", "result_digest"}}
    if stored_digest != _digest(digest_input):
        raise VerificationError("result digest differs")
    baseline, binding = _baseline(repository)
    registry, _ = _read(repository /
        "config/data/analogues/m04r10/nasdaq-untouched-authority-registry/query-registry.json")
    registry_rows = registry.get("cases_data")
    if type(registry_rows) is not list or len(registry_rows) != 60:
        raise VerificationError("registry differs")
    registry_by_case = {row["case_id"]: row for row in registry_rows}
    selected = result.get("selected_case_ids")
    if type(selected) is not list or len(selected) != 60 or set(selected) != set(baseline) \
            or len(set(selected)) != 60:
        raise VerificationError("selected cases differ")
    expected_selected = [case_id for case_id, _ in sorted(baseline.items(), key=lambda item: (
        -float(item[1]["measurement"]["exact_stage_seconds"]), item[0]))]
    if selected != expected_selected:
        raise VerificationError("selected order differs")
    manifest: list[dict[str, str]] = [{"path": "RESULT.json", "sha256": _sha(result_raw)}]
    observed: dict[str, dict[str, Any]] = {}
    for relative in sorted(files - {"RESULT.json"}):
        value, raw = _read(root / relative)
        manifest.append({"path": relative, "sha256": _sha(raw)})
        case_id = value.get("registry_case_id")
        if type(case_id) is not str or case_id in observed or case_id not in baseline:
            raise VerificationError("observed case ID differs")
        row = registry_by_case[case_id]; semantic = baseline[case_id]["semantic"]
        expected_gates = {
            "case_id_equal": case_id == semantic["case_id"],
            "query_id_equal": value.get("query_episode_id") == semantic["query_id"],
            "proposal_equal": value.get("proposal_result_digest")
                == semantic["forward_proposal"]["result_digest"],
            "matches_equal": value.get("matches") == semantic["matches"],
            "certificate_equal": value.get("certificate") == semantic["certificate"],
            "engine_gate_passed": value.get("gate_passed") is True,
        }
        if not all(expected_gates.values()) or value.get("status") != "completed" \
                or value.get("schema_version") != "m04r11-certified-authority-case-v4" \
                or value.get("real_forward_outcomes_accessed") is not False \
                or value.get("registry_digest") != registry["registry_digest"] \
                or value.get("generation_id") != "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483" \
                or value.get("query_symbol") != row["symbol"] \
                or value.get("query_cutoff") != row["cutoff"] \
                or value.get("query_stock_prefix") != row["stock_prefix"] \
                or value.get("query_benchmark_prefix") != row["benchmark_prefix"] \
                or not all(value.get("gates", {}).values()) \
                or value.get("result_digest") != _case_result_digest(value) \
                or value.get("checkpoint_integrity_digest") != _case_integrity_digest(value) \
                or not all(_finite(value.get(k)) for k in (
                    "proposal_seconds", "amortized_proposal_seconds", "exact_seconds",
                    "final_exact_seconds", "peak_rss_mb")):
            raise VerificationError(f"case reconstruction differs: {case_id}")
        if Path(relative).name != f"{value['query_episode_id']}.json":
            raise VerificationError("case path differs")
        observed[case_id] = {"value": value, "gates": expected_gates}
    comparisons = result.get("comparisons")
    expected_comparisons = [{"case_id": case_id, "gates": observed[case_id]["gates"],
                             "passed": True} for case_id in selected]
    group_ids = [query for group in result.get("groups", [])
                 for query in group.get("query_episode_ids", [])]
    if comparisons != expected_comparisons or len(observed) != 60 \
            or len(result.get("groups", [])) != 8 or len(group_ids) != 60 \
            or set(group_ids) != {v["value"]["query_episode_id"] for v in observed.values()} \
            or len(set(group_ids)) != 60:
        raise VerificationError("comparison/group coverage differs")
    serial_service = sum(
        float(row["measurement"]["forward_proposal_seconds"])
        + float(row["measurement"]["exact_stage_seconds"])
        for row in baseline.values()
    )
    expected_controls = {**CONTROLS, "processes": 8, "effective_cpus": list(range(8)),
        "grouping": "LPT-by-V3-exact-stage-seconds", "service_scans_per_query_group": 1,
        "reverse_scan_role": "separate-audit-not-service-throughput"}
    if not all((result["schema_version"] == "m04r14-throughput-development-poc-v1",
                result["status"] == "complete", result["cases"] == 60,
                result["development_only"] is True,
                result["authority_or_outcome_accessed"] is False,
                result["all_semantics_equal"] is True, result["baseline"] == binding,
                result["registry_digest"] == registry["registry_digest"],
                result["controls"] == expected_controls,
                result["baseline_serial_service_seconds"] == serial_service,
                _finite(result["wall_seconds"]),
                result["speedup_vs_serial_service_sum"]
                == serial_service / result["wall_seconds"])):
        raise VerificationError("aggregate reconstruction differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "candidate_root": str(root), "candidate_result_digest": result["result_digest"],
        "candidate_result_sha256": _sha(result_raw), "verified_cases": 60,
        "terminal_manifest": manifest, "terminal_manifest_digest": _digest(manifest),
        "baseline_complete_digest": binding["complete_digest"],
        "all_semantics_equal": True, "authority_or_outcome_accessed": False,
        "wall_seconds": result["wall_seconds"],
        "speedup_vs_serial_service_sum": result["speedup_vs_serial_service_sum"]}
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
        handle.flush(); os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--verification-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    candidate = args.candidate_root or repository / CANDIDATE
    output = args.verification_root or repository / VERIFICATION
    state = verify(candidate, repository=repository)
    if not args.dry_run:
        _publish(output, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

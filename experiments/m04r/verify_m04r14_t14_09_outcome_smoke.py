"""Independently recompute and verify every T14-09 smoke outcome and path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.m04r import m04r14_shadow_run as retrieval
from experiments.m04r.m04r14_t14_09_outcome_oracle import reference_episode
from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.types import InstrumentKey, stable_hash


SCHEMA = "m04r14-t14-09-outcome-smoke-verification-v1"
PREREGISTRATION = Path("experiments/m04r/m04r14_t14_09_outcome_smoke_preregistered.json")
OUTPUT_ROOT = Path("config/data/analogues/m04r14/t14-09-outcome-smoke-v1")
VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-outcome-smoke-v1-verification"
)
CONTRACT = Path("config/m04r14-t14-09-outcome-contract.json")
SYNTHETIC = Path(
    "config/data/analogues/m04r14/t14-09-synthetic-outcome-gate-v1/RESULT.json"
)
REGISTRY = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1")


class SmokeVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False, check: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=check,
    )
    return result.stdout if raw else result.stdout.strip()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise SmokeVerificationError(f"regular JSON file required: {path}")
    raw = path.read_bytes()
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise SmokeVerificationError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result
    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            SmokeVerificationError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise SmokeVerificationError(f"JSON object required: {path}")
    return value, raw


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SmokeVerificationError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def _frame_records(frame: pd.DataFrame, order: Sequence[str]) -> list[dict[str, Any]]:
    return _plain(frame.sort_values(list(order), kind="stable").reset_index(drop=True).to_dict("records"))


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    children: list[str] = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if values and values[0] == h0:
            children.extend(values[1:])
    accepted: list[str] = []
    for child in sorted(set(children)):
        lineage = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
        changed = str(_git(
            repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
        )).splitlines()
        if lineage != [child, h0] or changed != [PREREGISTRATION.as_posix()]:
            continue
        blob = _git(repository, "show", f"{child}:{PREREGISTRATION.as_posix()}", raw=True)
        if blob == raw:
            accepted.append(child)
    if len(accepted) != 1:
        raise SmokeVerificationError("smoke preregistration lifecycle differs")
    return accepted[0]


def _base_links(repository: Path, sample: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for item in sample:
        query_id = str(item["episode_id"])
        case, _ = _read(repository / retrieval.OUTPUT / "cases" / f"{query_id}.json")
        if not all((
            case.get("query_episode_id") == query_id,
            case.get("registry_case_id") == item["case_id"],
            case.get("gate_passed") is True,
            case.get("result_digest") == retrieval._deterministic_case_digest(case),
            case.get("checkpoint_integrity_digest") == retrieval._integrity_digest(case),
            type(case.get("matches")) is list and len(case["matches"]) == 20,
        )):
            raise SmokeVerificationError(f"candidate case differs: {item['case_id']}")
        for rank, match in enumerate(case["matches"], 1):
            links.append({
                "query_case_id": item["case_id"], "query_episode_id": query_id,
                "query_symbol": item["symbol"], "query_cutoff": str(case["query_cutoff"]),
                "match_rank": rank, "matched_episode_id": str(match["episode_id"]),
                "matched_symbol": str(match["symbol"]),
                "matched_cutoff": str(match["cutoff"]),
                "total_distance": float(match["total_distance"]),
                "match_digest": stable_hash(match),
                "candidate_case_result_digest": str(case["result_digest"]),
            })
    if len(links) != 240:
        raise SmokeVerificationError("candidate link count differs")
    return links


def _requests(links: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for link in links:
        row = {
            "episode_id": str(link["matched_episode_id"]),
            "symbol": str(link["matched_symbol"]), "cutoff": str(link["matched_cutoff"]),
        }
        if row["episode_id"] in result and result[row["episode_id"]] != row:
            raise SmokeVerificationError("conflicting episode metadata")
        result[row["episode_id"]] = row
    return [result[key] for key in sorted(result)]


def _bind_source_fingerprints(
    repository: Path, requests: Sequence[dict[str, Any]], registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lock = registry["source_lock"]
    quality_path = Path(str(lock["quality_path"]))
    if _sha(quality_path) != lock.get("quality_sha256"):
        raise SmokeVerificationError("quality metadata drifted")
    quality = pd.read_parquet(quality_path)
    if not {"symbol", "source_hash"}.issubset(quality.columns) \
            or quality.symbol.astype(str).duplicated().any():
        raise SmokeVerificationError("quality fingerprint schema differs")
    fingerprints = {
        str(row.symbol): str(row.source_hash) for row in quality.itertuples(index=False)
    }
    return [{
        **request, "expected_source_fingerprint": fingerprints[request["symbol"]],
    } for request in requests]


def _eligibility(completion: Any, query_cutoff: Any, complete: bool) -> dict[str, Any]:
    if not complete or completion is None:
        return {"eligible": False, "reason": "incomplete_horizon"}
    end = pd.Timestamp(completion)
    query = pd.Timestamp(query_cutoff)
    if end > query:
        return {"eligible": False, "reason": "outcome_not_yet_observable"}
    return {"eligible": True, "reason": "eligible"}


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, prereg_raw = _read(repository / PREREGISTRATION)
    prereg_state = {
        key: value for key, value in prereg.items() if key != "preregistration_digest"
    }
    if prereg.get("schema_version") != "m04r14-t14-09-outcome-smoke-preregistration-v1" \
            or prereg.get("preregistration_digest") != stable_hash(prereg_state):
        raise SmokeVerificationError("preregistration seal differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_child(repository, prereg_raw, h0)
    for name, expected in prereg.get("runtime_files", {}).items():
        if sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected \
                or _sha(repository / name) != expected:
            raise SmokeVerificationError(f"runtime source drifted: {name}")
    contract, contract_raw = _read(repository / CONTRACT)
    synthetic, synthetic_raw = _read(repository / SYNTHETIC)
    if not all((
        prereg.get("contract_digest") == contract.get("contract_digest"),
        prereg.get("contract_sha256") == sha256(contract_raw).hexdigest(),
        prereg.get("synthetic_result_digest") == synthetic.get("result_digest"),
        prereg.get("synthetic_sha256") == sha256(synthetic_raw).hexdigest(),
        synthetic.get("passed") is True,
        synthetic.get("real_forward_outcomes_accessed") is False,
    )):
        raise SmokeVerificationError("contract/synthetic prerequisite differs")
    registry, _ = _read(repository / REGISTRY / "query-registry.json")
    base_links = _base_links(repository, registry["audit_sample"])
    requests = _bind_source_fingerprints(repository, _requests(base_links), registry)
    if any((
        prereg.get("link_digest") != stable_hash(base_links),
        prereg.get("unique_episode_request_digest") != stable_hash(requests),
        prereg.get("link_count") != 240,
        prereg.get("unique_episode_count") != len(requests),
    )):
        raise SmokeVerificationError("preregistered link/request binding differs")

    root = repository / OUTPUT_ROOT
    names = {
        "RUN_STARTED.json", "episode-outcomes.parquet", "future-paths.parquet",
        "query-match-links.parquet", "COVERAGE.json", "SEALED.json",
    }
    if {path.name for path in root.iterdir()} != names \
            or any(path.is_symlink() for path in root.iterdir()):
        raise SmokeVerificationError("smoke output inventory differs")
    started, _ = _read(root / "RUN_STARTED.json")
    seal, seal_raw = _read(root / "SEALED.json")
    deterministic = {
        key: value for key, value in seal.items()
        if key not in {"elapsed_seconds", "result_digest", "created_at"}
    }
    if not all((
        seal.get("result_digest") == stable_hash(deterministic),
        seal.get("status") == "sealed", seal.get("passed") is True,
        seal.get("preregistration_h1") == h1,
        seal.get("preregistration_digest") == prereg["preregistration_digest"],
        seal.get("contract_digest") == contract["contract_digest"],
        seal.get("source_content_digest") == contract["retrieval_inputs"]["source_content_digest"],
        seal.get("query_links") == 240,
        seal.get("unique_episodes") == len(requests),
        seal.get("outcome_rows") == len(requests) * 6,
        seal.get("real_forward_outcomes_accessed") is True,
        seal.get("outcomes_affected_retrieval") is False,
        started.get("real_forward_outcomes_accessed") is False,
    )):
        raise SmokeVerificationError("smoke seal differs")
    files = [root / row["path"] for row in seal.get("file_manifest", [])]
    expected_manifest_paths = [
        "episode-outcomes.parquet", "future-paths.parquet",
        "query-match-links.parquet", "COVERAGE.json",
    ]
    if [row.get("path") for row in seal.get("file_manifest", [])] != expected_manifest_paths \
            or any(path.parent != root or path.is_symlink() for path in files):
        raise SmokeVerificationError("smoke manifest path closure differs")
    observed_manifest = [{
        "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
        "sha256": _sha(path),
    } for path in files]
    if observed_manifest != seal.get("file_manifest"):
        raise SmokeVerificationError("smoke file manifest differs")
    outcomes = pd.read_parquet(root / "episode-outcomes.parquet")
    paths = pd.read_parquet(root / "future-paths.parquet")
    links = pd.read_parquet(root / "query-match-links.parquet")
    coverage, _ = _read(root / "COVERAGE.json")

    config = load_config(repository / retrieval.CONFIG)
    source = source_from_spec(config.datasets["nasdaq"])
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise SmokeVerificationError("benchmark unavailable during verification")
    if source.benchmark_fingerprint() != registry["source_lock"]["benchmark_sha256"]:
        raise SmokeVerificationError("benchmark drifted during verification")
    if benchmark.attrs.get("source_timestamp_reordered") \
            or benchmark.attrs.get("source_duplicate_timestamps"):
        raise SmokeVerificationError("benchmark order differs during verification")
    expected_outcomes: list[dict[str, Any]] = []
    expected_paths: list[dict[str, Any]] = []
    identities: dict[str, str] = {}
    for request in requests:
        key = InstrumentKey("nasdaq", request["symbol"])
        stock = source.load(key)
        numeric = stock[["open", "high", "low", "close"]]
        if stock.attrs.get("source_timestamp_reordered") \
                or stock.attrs.get("source_duplicate_timestamps") \
                or not numeric.apply(lambda column: column.map(math.isfinite)).all().all() \
                or (numeric <= 0).any().any() \
                or (stock["high"] < stock[["open", "close", "low"]].max(axis=1)).any() \
                or (stock["low"] > stock[["open", "close", "high"]].min(axis=1)).any():
            raise SmokeVerificationError(f"invalid source during verification: {request['symbol']}")
        episode = build_episode(source, key, request["cutoff"], 252, "dense-v1")
        if episode.key.id != request["episode_id"]:
            raise SmokeVerificationError("episode identity differs during verification")
        fingerprint = source.fingerprint(key)
        if fingerprint != request["expected_source_fingerprint"]:
            raise SmokeVerificationError("stock source drifted during verification")
        identities[request["episode_id"]] = fingerprint
        rows, path_rows = reference_episode(
            stock, benchmark, episode_id=request["episode_id"],
            cutoff=request["cutoff"], source_fingerprint=fingerprint,
            contract_digest=contract["contract_digest"],
            source_content_digest=contract["retrieval_inputs"]["source_content_digest"],
        )
        expected_outcomes.extend(rows)
        expected_paths.extend(path_rows)
    observed_outcomes = _frame_records(outcomes, ("episode_id", "horizon_sessions"))
    observed_paths = _frame_records(paths, ("episode_id", "step"))
    expected_outcomes = sorted(expected_outcomes, key=lambda row: (
        row["episode_id"], row["horizon_sessions"],
    ))
    expected_paths = sorted(expected_paths, key=lambda row: (row["episode_id"], row["step"]))
    if observed_outcomes != expected_outcomes:
        raise SmokeVerificationError("independent outcome recomputation differs")
    if observed_paths != expected_paths:
        raise SmokeVerificationError("independent path recomputation differs")
    outcome_by_key = {
        (row["episode_id"], int(row["horizon_sessions"])): row
        for row in expected_outcomes
    }
    expected_links: list[dict[str, Any]] = []
    for link in base_links:
        eligibility = {
            str(horizon): _eligibility(
                outcome_by_key[(link["matched_episode_id"], horizon)]["completion_timestamp"],
                link["query_cutoff"],
                bool(outcome_by_key[(link["matched_episode_id"], horizon)]["complete"]),
            ) for horizon in (5, 10, 20, 40, 60, 126)
        }
        expected_links.append({
            **link, "source_fingerprint": identities[link["matched_episode_id"]],
            "outcome_eligibility_json": json.dumps(
                eligibility, sort_keys=True, separators=(",", ":"),
            ),
        })
    observed_links = _frame_records(links, ("query_episode_id", "match_rank"))
    expected_links = sorted(expected_links, key=lambda row: (
        row["query_episode_id"], row["match_rank"],
    ))
    if observed_links != expected_links:
        raise SmokeVerificationError("independent query-link reconstruction differs")
    semantic = {
        "outcome_digest": stable_hash(observed_outcomes),
        "path_digest": stable_hash(observed_paths),
        "link_digest": stable_hash(observed_links),
        "coverage_result_digest": coverage.get("result_digest"),
    }
    coverage_state = {
        key: value for key, value in coverage.items() if key != "result_digest"
    }
    expected_coverage_state = {
        "schema_version": "m04r14-t14-09-outcome-smoke-v1",
        "status": "complete", "query_links": len(observed_links),
        "unique_episodes": len(requests),
        "outcome_rows": len(observed_outcomes), "path_rows": len(observed_paths),
        "complete_by_horizon": {
            str(horizon): sum(
                bool(row["complete"]) for row in observed_outcomes
                if int(row["horizon_sessions"]) == horizon
            ) for horizon in (5, 10, 20, 40, 60, 126)
        },
        "status_counts": {
            key: sum(row["status"] == key for row in observed_outcomes)
            for key in sorted({str(row["status"]) for row in observed_outcomes})
        },
        "barrier_counts": {
            key: sum(
                int(row["horizon_sessions"]) == 20 and str(row["barrier_label"]) == key
                for row in observed_outcomes
            ) for key in sorted({
                str(row["barrier_label"]) for row in observed_outcomes
                if int(row["horizon_sessions"]) == 20
            })
        },
        "real_forward_outcomes_accessed": True,
    }
    if coverage_state != expected_coverage_state \
            or semantic != seal.get("semantic_digests") \
            or coverage.get("result_digest") != stable_hash(coverage_state):
        raise SmokeVerificationError("semantic or coverage digest differs")
    result_state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "smoke_result_digest": seal["result_digest"],
        "smoke_result_sha256": sha256(seal_raw).hexdigest(),
        "contract_digest": contract["contract_digest"],
        "query_links": 240, "unique_episodes": len(requests),
        "verified_outcome_rows": len(expected_outcomes),
        "verified_path_rows": len(expected_paths),
        "real_forward_outcomes_accessed": True,
        "outcomes_affected_retrieval": False,
        "full_run_authorized": True,
        "production_promotion_authorized": False,
    }
    return {**result_state, "result_digest": stable_hash(result_state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SmokeVerificationError("smoke verification root exists")
    path.mkdir(parents=False)
    descriptor = os.open(path / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
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
    value = verify(repository)
    if not args.dry_run:
        _publish(repository / VERIFICATION, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Preregister and build the 12-query T14-09 real-outcome smoke store."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.m04r import m04r14_shadow_run as retrieval
from experiments.m04r import verify_m04r14_t14_09_outcome_contract as contract_verifier
from market_analogues.adapters import source_from_spec
from market_analogues.causal_outcomes import compute_episode_outcomes, outcome_embargo
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.types import InstrumentKey, stable_hash


SCHEMA = "m04r14-t14-09-outcome-smoke-v1"
PREREG_SCHEMA = "m04r14-t14-09-outcome-smoke-preregistration-v1"
PREREGISTRATION = Path("experiments/m04r/m04r14_t14_09_outcome_smoke_preregistered.json")
OUTPUT = Path("config/data/analogues/m04r14/t14-09-outcome-smoke-v1")
CONTRACT = Path("config/m04r14-t14-09-outcome-contract.json")
CONTRACT_RECEIPT = contract_verifier.OUTPUT / "VERIFIED.json"
SYNTHETIC = Path(
    "config/data/analogues/m04r14/t14-09-synthetic-outcome-gate-v1/RESULT.json"
)
SEMANTIC = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-interrupted-verification-v1/VERIFIED.json"
)
AUDIT = Path("config/data/analogues/m04r14/nasdaq-shadow-audit-comparison-v1/RESULT.json")
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_09_outcome_smoke.py",
    "experiments/m04r/verify_m04r14_t14_09_outcome_smoke.py",
    "experiments/m04r/m04r14_t14_09_outcome_oracle.py",
    "src/market_analogues/causal_outcomes.py",
)


class OutcomeSmokeError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False, check: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=check,
    )
    return result.stdout if raw else result.stdout.strip()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise OutcomeSmokeError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise OutcomeSmokeError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            OutcomeSmokeError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise OutcomeSmokeError(f"JSON object required: {path}")
    return value, raw


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise OutcomeSmokeError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise OutcomeSmokeError(f"create-only target exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False,
        ).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise OutcomeSmokeError(f"create-only target exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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


def _frame_digest(frame: pd.DataFrame, order: Sequence[str]) -> str:
    ordered = frame.sort_values(list(order), kind="stable").reset_index(drop=True)
    return stable_hash(_plain(ordered.to_dict("records")))


def _receipt_valid(value: Mapping[str, Any], *, timing: bool = False) -> bool:
    omitted = {"result_digest", "created_at"}
    if timing:
        omitted.add("elapsed_seconds")
    return value.get("result_digest") == stable_hash({
        key: item for key, item in value.items() if key not in omitted
    })


def _extract_links(repository: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry, _ = _read(repository / retrieval.REGISTRY / "query-registry.json")
    semantic, _ = _read(repository / SEMANTIC)
    audit, _ = _read(repository / AUDIT)
    if not all((
        semantic.get("semantic_passed") is True,
        semantic.get("verified_cases") == 3270,
        semantic.get("verified_matches") == 65400,
        audit.get("passed") is True, audit.get("matching_positions") == 240,
        registry.get("registry_digest") == semantic.get("registry_digest"),
        registry.get("audit_sample_digest") == audit.get("audit_sample_digest"),
    )):
        raise OutcomeSmokeError("retrieval evidence does not authorize smoke extraction")
    links: list[dict[str, Any]] = []
    for sample in registry["audit_sample"]:
        query_id = str(sample["episode_id"])
        case, _ = _read(repository / retrieval.OUTPUT / "cases" / f"{query_id}.json")
        if not all((
            case.get("query_episode_id") == query_id,
            case.get("registry_case_id") == sample["case_id"],
            case.get("gate_passed") is True,
            case.get("result_digest") == retrieval._deterministic_case_digest(case),
            case.get("checkpoint_integrity_digest") == retrieval._integrity_digest(case),
            type(case.get("matches")) is list and len(case["matches"]) == 20,
        )):
            raise OutcomeSmokeError(f"retrieval case differs: {sample['case_id']}")
        for rank, match in enumerate(case["matches"], start=1):
            links.append({
                "query_case_id": sample["case_id"],
                "query_episode_id": query_id,
                "query_symbol": sample["symbol"],
                "query_cutoff": str(case["query_cutoff"]),
                "match_rank": rank,
                "matched_episode_id": str(match["episode_id"]),
                "matched_symbol": str(match["symbol"]),
                "matched_cutoff": str(match["cutoff"]),
                "total_distance": float(match["total_distance"]),
                "match_digest": stable_hash(match),
                "candidate_case_result_digest": str(case["result_digest"]),
            })
    if len(links) != 240 or len({
        (row["query_episode_id"], row["match_rank"]) for row in links
    }) != 240:
        raise OutcomeSmokeError("smoke link inventory differs")
    return links, registry


def _requests(links: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for link in links:
        row = {
            "episode_id": str(link["matched_episode_id"]),
            "symbol": str(link["matched_symbol"]),
            "cutoff": str(link["matched_cutoff"]),
        }
        prior = by_id.get(row["episode_id"])
        if prior is not None and prior != row:
            raise OutcomeSmokeError("matched episode identity has conflicting metadata")
        by_id[row["episode_id"]] = row
    return [by_id[key] for key in sorted(by_id)]


def _bind_source_fingerprints(
    repository: Path, requests: Sequence[dict[str, Any]], registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lock = registry["source_lock"]
    quality_path = Path(str(lock["quality_path"]))
    if _sha(quality_path) != lock.get("quality_sha256"):
        raise OutcomeSmokeError("quality metadata drifted from the source lock")
    quality = pd.read_parquet(quality_path)
    if not {"symbol", "source_hash"}.issubset(quality.columns) \
            or quality.symbol.astype(str).duplicated().any():
        raise OutcomeSmokeError("quality source-fingerprint schema differs")
    fingerprints = {
        str(row.symbol): str(row.source_hash) for row in quality.itertuples(index=False)
    }
    result: list[dict[str, Any]] = []
    for request in requests:
        expected = fingerprints.get(request["symbol"])
        if not expected:
            raise OutcomeSmokeError(f"matched symbol lacks a source lock: {request['symbol']}")
        result.append({**request, "expected_source_fingerprint": expected})
    return result


def _manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(
        repository, "ls-tree", "-r", "--name-only", head,
    )).splitlines())
    names = sorted({
        name for name in tracked
        if name.startswith("src/market_analogues/") and name.endswith(".py")
    } | set(RUNTIME_FILES))
    if any(name not in tracked for name in names):
        raise OutcomeSmokeError("runtime manifest contains an uncommitted file")
    return {
        name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest()
        for name in names
    }


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise OutcomeSmokeError("globally clean Git worktree required")
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    contract, contract_raw = _read(repository / CONTRACT)
    contract_receipt, contract_receipt_raw = _read(repository / CONTRACT_RECEIPT)
    synthetic, synthetic_raw = _read(repository / SYNTHETIC)
    if not all((
        contract_receipt.get("passed") is True,
        contract_receipt.get("contract_digest") == contract.get("contract_digest"),
        contract_receipt.get("real_forward_outcomes_accessed") is False,
        _receipt_valid(contract_receipt),
        synthetic.get("passed") is True,
        synthetic.get("contract_digest") == contract.get("contract_digest"),
        synthetic.get("real_forward_outcomes_accessed") is False,
        _receipt_valid(synthetic, timing=True),
    )):
        raise OutcomeSmokeError("contract or synthetic gate differs")
    links, registry = _extract_links(repository)
    requests = _bind_source_fingerprints(repository, _requests(links), registry)
    state = {
        "schema_version": PREREG_SCHEMA,
        "status": "frozen_before_real_outcome_smoke",
        "implementation_h0": h0,
        "runtime_files": _manifest(repository, h0),
        "contract_digest": contract["contract_digest"],
        "contract_sha256": sha256(contract_raw).hexdigest(),
        "contract_verification_result_digest": contract_receipt["result_digest"],
        "contract_verification_sha256": sha256(contract_receipt_raw).hexdigest(),
        "synthetic_result_digest": synthetic["result_digest"],
        "synthetic_sha256": sha256(synthetic_raw).hexdigest(),
        "registry_digest": registry["registry_digest"],
        "audit_sample_digest": registry["audit_sample_digest"],
        "link_count": 240,
        "link_digest": stable_hash(links),
        "unique_episode_count": len(requests),
        "unique_episode_request_digest": stable_hash(requests),
        "processes": 12,
        "output": str((repository / OUTPUT).resolve()),
        "retrieval_opened": True,
        "real_forward_outcomes_accessed": False,
        "production_promotion_authorized": False,
    }
    return {**state, "preregistration_digest": stable_hash(state)}


def _sole_child(repository: Path, prereg_raw: bytes, h0: str) -> str:
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
        if blob == prereg_raw:
            accepted.append(child)
    if len(accepted) != 1:
        raise OutcomeSmokeError("expected one exact preregistration-only child")
    return accepted[0]


def _validate_preregistration(
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise OutcomeSmokeError("globally clean Git worktree required")
    prereg, prereg_raw = _read(repository / PREREGISTRATION)
    state = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("schema_version") != PREREG_SCHEMA \
            or prereg.get("preregistration_digest") != stable_hash(state):
        raise OutcomeSmokeError("preregistration seal differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_child(repository, prereg_raw, h0)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository,
    ).returncode:
        raise OutcomeSmokeError("HEAD does not descend from outcome preregistration")
    for name, expected in prereg.get("runtime_files", {}).items():
        if sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected \
                or _sha(repository / name) != expected:
            raise OutcomeSmokeError(f"runtime source drifted: {name}")
    contract, contract_raw = _read(repository / CONTRACT)
    contract_receipt, contract_receipt_raw = _read(repository / CONTRACT_RECEIPT)
    synthetic, synthetic_raw = _read(repository / SYNTHETIC)
    if prereg.get("contract_digest") != contract.get("contract_digest") \
            or prereg.get("contract_sha256") != sha256(contract_raw).hexdigest() \
            or prereg.get("contract_verification_result_digest") != contract_receipt.get("result_digest") \
            or prereg.get("contract_verification_sha256") != sha256(contract_receipt_raw).hexdigest() \
            or not _receipt_valid(contract_receipt) \
            or contract_receipt.get("passed") is not True \
            or prereg.get("synthetic_result_digest") != synthetic.get("result_digest") \
            or prereg.get("synthetic_sha256") != sha256(synthetic_raw).hexdigest() \
            or not _receipt_valid(synthetic, timing=True) \
            or synthetic.get("passed") is not True \
            or prereg.get("real_forward_outcomes_accessed") is not False:
        raise OutcomeSmokeError("outcome contract drifted")
    links, registry = _extract_links(repository)
    requests = _bind_source_fingerprints(repository, _requests(links), registry)
    if any((
        prereg.get("registry_digest") != registry.get("registry_digest"),
        prereg.get("audit_sample_digest") != registry.get("audit_sample_digest"),
        prereg.get("link_count") != len(links),
        prereg.get("link_digest") != stable_hash(links),
        prereg.get("unique_episode_count") != len(requests),
        prereg.get("unique_episode_request_digest") != stable_hash(requests),
        prereg.get("processes") != 12,
        prereg.get("output") != str((repository / OUTPUT).resolve()),
    )):
        raise OutcomeSmokeError("preregistered smoke inputs differ")
    return prereg, contract, links, requests, h1


def _worker(
    repository: str, requests: tuple[dict[str, Any], ...],
    contract_digest: str, source_content_digest: str,
) -> dict[str, Any]:
    root = Path(repository)
    config = load_config(root / retrieval.CONFIG)
    source = source_from_spec(config.datasets["nasdaq"])
    benchmark = source.load_benchmark()
    if benchmark is None:
        raise OutcomeSmokeError("NASDAQ benchmark is unavailable")
    registry, _ = _read(root / retrieval.REGISTRY / "query-registry.json")
    if source.benchmark_fingerprint() != registry["source_lock"]["benchmark_sha256"]:
        raise OutcomeSmokeError("benchmark drifted from the source lock")
    if benchmark.attrs.get("source_timestamp_reordered") \
            or benchmark.attrs.get("source_duplicate_timestamps"):
        raise OutcomeSmokeError("benchmark session order is non-canonical")
    outcomes: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for request in requests:
        key = InstrumentKey("nasdaq", request["symbol"])
        stock = source.load(key)
        if stock.attrs.get("source_timestamp_reordered") \
                or stock.attrs.get("source_duplicate_timestamps"):
            raise OutcomeSmokeError(f"non-canonical source order: {request['symbol']}")
        episode = build_episode(source, key, request["cutoff"], 252, "dense-v1")
        if episode.key.id != request["episode_id"]:
            raise OutcomeSmokeError(f"matched episode identity differs: {request['episode_id']}")
        fingerprint = source.fingerprint(key)
        if fingerprint != request["expected_source_fingerprint"]:
            raise OutcomeSmokeError(f"stock source drifted from lock: {request['symbol']}")
        bundle = compute_episode_outcomes(
            stock, benchmark, episode_id=request["episode_id"],
            cutoff=request["cutoff"], source_fingerprint=fingerprint,
            contract_digest=contract_digest,
            source_content_digest=source_content_digest,
        )
        outcomes.extend(_plain(bundle.outcomes.to_dict("records")))
        paths.extend(_plain(bundle.paths.to_dict("records")))
        identities.append({**request, "source_fingerprint": fingerprint})
    return {"outcomes": outcomes, "paths": paths, "identities": identities}


def _groups(requests: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], ...]]:
    values: list[list[dict[str, Any]]] = [[] for _ in range(12)]
    for row in requests:
        index = int(sha256(row["symbol"].encode()).hexdigest(), 16) % 12
        values[index].append(dict(row))
    return [tuple(sorted(group, key=lambda row: row["episode_id"])) for group in values if group]


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, contract, links, requests, h1 = _validate_preregistration(repository)
    root = repository / OUTPUT
    if root.exists() or root.is_symlink():
        raise OutcomeSmokeError("outcome smoke root exists")
    if os.statvfs(repository).f_bavail * os.statvfs(repository).f_frsize < 5 * 1024 ** 3:
        raise OutcomeSmokeError("less than 5 GiB free before outcome smoke")
    root.mkdir(parents=True)
    started = perf_counter()
    _atomic_json(root / "RUN_STARTED.json", {
        "schema_version": SCHEMA, "status": "running",
        "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "contract_digest": contract["contract_digest"],
        "link_digest": prereg["link_digest"],
        "unique_episode_request_digest": prereg["unique_episode_request_digest"],
        "real_forward_outcomes_accessed": False, "created_at": _now(),
    })
    source_content_digest = str(contract["retrieval_inputs"]["source_content_digest"])
    context = multiprocessing.get_context("spawn")
    groups = _groups(requests)
    with ProcessPoolExecutor(max_workers=12, mp_context=context) as pool:
        results = list(pool.map(
            _worker, [str(repository)] * len(groups), groups,
            [str(contract["contract_digest"])] * len(groups),
            [source_content_digest] * len(groups),
        ))
    outcome_rows = [row for result in results for row in result["outcomes"]]
    path_rows = [row for result in results for row in result["paths"]]
    identities = [row for result in results for row in result["identities"]]
    identity_by_episode = {row["episode_id"]: row for row in identities}
    if len(identity_by_episode) != len(requests):
        raise OutcomeSmokeError("computed episode identity coverage differs")
    outcome_by_key = {
        (row["episode_id"], int(row["horizon_sessions"])): row for row in outcome_rows
    }
    enriched_links: list[dict[str, Any]] = []
    for link in links:
        identity = identity_by_episode[link["matched_episode_id"]]
        eligibility: dict[str, dict[str, Any]] = {}
        for horizon in (5, 10, 20, 40, 60, 126):
            outcome = outcome_by_key[(link["matched_episode_id"], horizon)]
            eligible, reason = outcome_embargo(
                outcome["completion_timestamp"], link["query_cutoff"],
                complete=bool(outcome["complete"]),
            )
            eligibility[str(horizon)] = {"eligible": eligible, "reason": reason}
        enriched_links.append({
            **link, "source_fingerprint": identity["source_fingerprint"],
            "outcome_eligibility_json": json.dumps(
                eligibility, sort_keys=True, separators=(",", ":"),
            ),
        })
    outcomes = pd.DataFrame(outcome_rows).sort_values(
        ["episode_id", "horizon_sessions"], kind="stable",
    ).reset_index(drop=True)
    outcomes["horizon_sessions"] = outcomes["horizon_sessions"].astype("int64")
    outcomes["available_sessions"] = outcomes["available_sessions"].astype("int64")
    outcomes["complete"] = outcomes["complete"].astype("bool")
    for column in ("time_to_mfe", "time_to_mae", "barrier_touch_offset"):
        outcomes[column] = outcomes[column].astype("Int64")
    paths = pd.DataFrame(path_rows).sort_values(
        ["episode_id", "step"], kind="stable",
    ).reset_index(drop=True)
    paths["step"] = paths["step"].astype("int64")
    paths["expected_session_match"] = paths["expected_session_match"].astype("bool")
    link_frame = pd.DataFrame(enriched_links).sort_values(
        ["query_episode_id", "match_rank"], kind="stable",
    ).reset_index(drop=True)
    link_frame["match_rank"] = link_frame["match_rank"].astype("int64")
    _atomic_parquet(root / str(contract["artifacts"]["episode_outcomes"]), outcomes)
    _atomic_parquet(root / str(contract["artifacts"]["future_paths"]), paths)
    _atomic_parquet(root / str(contract["artifacts"]["query_match_links"]), link_frame)
    coverage_state = {
        "schema_version": SCHEMA, "status": "complete",
        "query_links": len(link_frame), "unique_episodes": len(requests),
        "outcome_rows": len(outcomes), "path_rows": len(paths),
        "complete_by_horizon": {
            str(horizon): int(outcomes.loc[
                outcomes.horizon_sessions == horizon, "complete"
            ].sum()) for horizon in (5, 10, 20, 40, 60, 126)
        },
        "status_counts": {
            str(key): int(value) for key, value in outcomes.status.value_counts().sort_index().items()
        },
        "barrier_counts": {
            str(key): int(value) for key, value in outcomes.loc[
                outcomes.horizon_sessions == 20, "barrier_label"
            ].value_counts(dropna=False).sort_index().items()
        },
        "real_forward_outcomes_accessed": True,
    }
    coverage = {**coverage_state, "result_digest": stable_hash(coverage_state)}
    _atomic_json(root / str(contract["artifacts"]["coverage"]), coverage)
    files = [
        root / str(contract["artifacts"][name])
        for name in ("episode_outcomes", "future_paths", "query_match_links", "coverage")
    ]
    semantic = {
        "outcome_digest": _frame_digest(outcomes, ("episode_id", "horizon_sessions")),
        "path_digest": _frame_digest(paths, ("episode_id", "step")),
        "link_digest": _frame_digest(link_frame, ("query_episode_id", "match_rank")),
        "coverage_result_digest": coverage["result_digest"],
    }
    final_state = {
        "schema_version": SCHEMA, "status": "sealed", "passed": True,
        "preregistration_h1": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "contract_digest": contract["contract_digest"],
        "source_content_digest": source_content_digest,
        "query_links": len(link_frame), "unique_episodes": len(requests),
        "outcome_rows": len(outcomes), "path_rows": len(paths),
        "semantic_digests": semantic,
        "file_manifest": [{
            "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": _sha(path),
        } for path in files],
        "elapsed_seconds": perf_counter() - started,
        "real_forward_outcomes_accessed": True,
        "outcomes_affected_retrieval": False,
        "production_promotion_authorized": False,
    }
    deterministic = {
        key: value for key, value in final_state.items() if key != "elapsed_seconds"
    }
    final = {**final_state, "result_digest": stable_hash(deterministic), "created_at": _now()}
    _atomic_json(root / str(contract["artifacts"]["seal"]), final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-preregistration", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    try:
        value = build_preregistration(repository) if args.build_preregistration else execute(repository)
    except BaseException as exc:
        root = repository / OUTPUT
        if root.is_dir() and not (root / "SEALED.json").exists() \
                and not (root / "FAILED.json").exists():
            _atomic_json(root / "FAILED.json", {
                "schema_version": SCHEMA, "status": "failed",
                "error_type": type(exc).__name__, "message": str(exc),
                "resume_authorized": False, "created_at": _now(),
            })
        raise
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

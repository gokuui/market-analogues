"""Preregister and build the outcome-blind WF-03D cross-store manifest."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_exclusion_repair_full as repair
from experiments.m04r import verify_m04r14_t14_10_wf03d_exclusion_repair_full as repair_verifier


SCHEMA = "m04r14-t14-10-wf03d-cross-store-preregistration-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-cross-store-manifest-v1"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03d_cross_store_manifest_v1_preregistered.json"
)
LINK_SCHEMA = "m04r14-t14-10-wf03d-raw-link-table-v1"
REQUEST_SCHEMA = "m04r14-t14-10-wf03d-episode-request-table-v1"
MANIFEST_SCHEMA = "m04r14-t14-10-wf03d-cross-store-manifest-v1"
RESULT_SCHEMA = "m04r14-t14-10-wf03d-cross-store-result-v1"
LINK_FILE = "raw_links.parquet"
REQUEST_FILE = "episode_requests.parquet"
EXPECTED_QUERIES = 3_936
EXPECTED_METHOD_LANES = 15_744
EXPECTED_LINKS = 314_880
METHODS = repair.METHODS
QUALITY_CODES = {1: "A", 2: "B"}
SEMANTIC_CHUNK_ROWS = 16_384
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_10_wf03d_cross_store_manifest.py",
    "experiments/m04r/verify_m04r14_t14_10_wf03d_cross_store_manifest.py",
    "experiments/m04r/m04r14_t14_10_wf03d_exclusion_repair_full.py",
    "experiments/m04r/verify_m04r14_t14_10_wf03d_exclusion_repair_full.py",
    "experiments/m04r/m04r14_t14_10_wf03_feasibility.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/types.py",
    "pyproject.toml",
)
LINK_COLUMNS = (
    "query_id", "query_case_id", "query_symbol", "query_cutoff",
    "fold_id", "fold_role", "method", "rank", "matched_episode_id",
    "matched_symbol", "matched_cutoff", "quality_tier", "distance_hex",
    "latest_eligible_ns", "match_digest", "effective_matches_digest",
    "resolution_kind", "source_artifact_path", "source_artifact_sha256",
    "source_artifact_digest",
)
REQUEST_COLUMNS = (
    "episode_id", "dataset_id", "symbol", "cutoff", "quality_tier",
)


class CrossStoreManifestError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if result.returncode:
        message = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise CrossStoreManifestError(message.strip() or "git command failed")
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CrossStoreManifestError(f"regular file required: {path}")
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


def _semantic_digest(records: Sequence[Mapping[str, Any]]) -> str:
    digest = sha256()
    digest.update(f"canonical-json-record-chunks-v1\0{len(records)}\0".encode())
    for start in range(0, len(records), SEMANTIC_CHUNK_ROWS):
        payload = json.dumps(
            _plain(list(records[start:start + SEMANTIC_CHUNK_ROWS])),
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(
        repository, "ls-tree", "-r", "--name-only", head,
    )).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES):
        raise CrossStoreManifestError("runtime manifest contains uncommitted files")
    return {
        name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest()
        for name in RUNTIME_FILES
    }


def _read_verified_inputs(
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = repository / repair.OUTPUT_RELATIVE
    result_path = root / "RESULT.json"
    manifest_path = root / "MANIFEST.json"
    verified_path = repository / repair_verifier.OUTPUT_RELATIVE / "VERIFIED.json"
    result = base._read(result_path)
    manifest = base._read(manifest_path)
    verified = base._read(verified_path)
    base._validate_seal(result)
    base._validate_seal(manifest, "manifest_digest")
    base._validate_seal(verified, "verification_digest")
    registry, _ = base._registry(repository)
    rows = registry["queries_data"]
    if not all((
        result.get("passed") is True,
        result.get("manifest_digest") == manifest.get("manifest_digest"),
        result.get("manifest_sha256") == _sha(manifest_path),
        verified.get("passed") is True,
        verified.get("producer_result_digest") == result.get("result_digest"),
        verified.get("manifest_digest") == manifest.get("manifest_digest"),
        verified.get("cross_store_manifest_construction_authorized") is True,
        verified.get("effective_neighbour_links") == EXPECTED_LINKS,
        result.get("outcomes_or_labels_used") is False,
        result.get("historical_walk_forward_query_outcomes_opened") is False,
        result.get("final_period_result_opened") is False,
        len(rows) == EXPECTED_QUERIES,
    )):
        raise CrossStoreManifestError("verified exclusion-repair input differs")
    return result, manifest, verified, rows


def _source_case_key(method: str) -> str:
    if method == "composite":
        return "composite"
    if method == "price_only":
        return "price_only"
    return "baselines"


def _source_matches(case: Mapping[str, Any], method: str) -> list[dict[str, Any]]:
    try:
        if method == "composite":
            value = case["retrieval"]["matches"]
        elif method == "price_only":
            value = case["matches"]
        else:
            value = case[
                "random_neighbors" if method == "deterministic_random"
                else "rank_neighbors"
            ]
    except (KeyError, TypeError) as exc:
        raise CrossStoreManifestError("upstream match layout differs") from exc
    if type(value) is not list or len(value) != repair.TOP_K:
        raise CrossStoreManifestError("upstream top-20 differs")
    return value


def _distance_hex(method: str, match: Mapping[str, Any]) -> str | None:
    if method == "composite":
        try:
            return float(match["total_distance"]).hex()
        except (KeyError, TypeError, ValueError) as exc:
            raise CrossStoreManifestError("composite distance differs") from exc
    value = match.get("distance_hex")
    if value is not None and type(value) is not str:
        raise CrossStoreManifestError("baseline/price distance differs")
    return value


def _packed_identities(
    repository: Path, identifiers: Sequence[str],
) -> dict[str, tuple[str, str, str]]:
    resident = base._resident()
    packed = load_packed_generation(
        Path(resident["store_root"]), base.GENERATION_ID,
        expected_provenance_digest=base.PROVENANCE_DIGEST,
        verify_content=False, validate_records=False,
    )
    ids = np.concatenate((packed.rows["episode_id"], packed.overflow["episode_id"]))
    cutoff = np.concatenate((packed.rows["cutoff_ns"], packed.overflow["cutoff_ns"]))
    symbol_id = np.concatenate((packed.rows["symbol_id"], packed.overflow["symbol_id"]))
    quality = np.concatenate((packed.rows["quality_tier"], packed.overflow["quality_tier"]))
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if len(np.unique(sorted_ids)) != len(sorted_ids):
        raise CrossStoreManifestError("packed episode IDs are not unique")
    try:
        requested = np.asarray(
            [np.void(bytes.fromhex(value)) for value in identifiers], dtype="V12",
        )
    except ValueError as exc:
        raise CrossStoreManifestError("invalid matched episode ID") from exc
    positions = np.searchsorted(sorted_ids, requested)
    if np.any(positions >= len(sorted_ids)) \
            or not np.array_equal(sorted_ids[positions], requested):
        raise CrossStoreManifestError("matched episode absent from packed inventory")
    selected = order[positions]
    result: dict[str, tuple[str, str, str]] = {}
    for identifier, index in zip(identifiers, selected, strict=True):
        try:
            tier = QUALITY_CODES[int(quality[index])]
            symbol = packed.symbols[int(symbol_id[index])]
        except (KeyError, IndexError) as exc:
            raise CrossStoreManifestError("packed identity code differs") from exc
        result[identifier] = (
            symbol, pd.Timestamp(int(cutoff[index]), unit="ns").isoformat(), tier,
        )
    return result


def _logical_inventory(
    repository: Path, manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if type(manifest.get("queries")) is not list \
            or len(manifest["queries"]) != len(rows):
        raise CrossStoreManifestError("repair manifest query inventory differs")
    preliminary: list[dict[str, Any]] = []
    for row, entry in zip(rows, manifest["queries"], strict=True):
        query_id = str(row["episode_id"])
        if not all((
            entry.get("query_id") == query_id,
            entry.get("case_id") == row.get("case_id"),
            entry.get("symbol") == row.get("symbol"),
            entry.get("cutoff") == row.get("cutoff"),
            type(entry.get("methods")) is list,
            [value.get("method") for value in entry["methods"]] == list(METHODS),
        )):
            raise CrossStoreManifestError("repair manifest query binding differs")
        paths = repair._case_paths(repository, query_id)
        baseline_case = base._read(paths["baselines"])
        base._validate_seal(baseline_case, "case_digest")
        latest_ns = baseline_case.get("latest_eligible_ns")
        if type(latest_ns) is not int:
            raise CrossStoreManifestError("latest eligible cutoff differs")
        loaded_cases: dict[str, dict[str, Any]] = {"baselines": baseline_case}
        for method_entry in entry["methods"]:
            method = str(method_entry["method"])
            if method_entry.get("kind") == "top21_drop_query_symbol":
                artifact_path = repair.OUTPUT_RELATIVE / str(
                    method_entry.get("repair_receipt", "")
                )
                receipt = base._read(repository / artifact_path)
                base._validate_seal(receipt, "receipt_digest")
                matches = receipt.get("corrected_matches")
                artifact_digest = receipt.get("receipt_digest")
                if not all((
                    _sha(repository / artifact_path)
                        == method_entry.get("repair_receipt_sha256"),
                    artifact_digest == method_entry.get("repair_receipt_digest"),
                    receipt.get("query_id") == query_id,
                    receipt.get("method") == method,
                    type(matches) is list,
                )):
                    raise CrossStoreManifestError("repair receipt binding differs")
            elif method_entry.get("kind") == "upstream_top20_unchanged_over_subset":
                key = _source_case_key(method)
                artifact_path = paths[key].relative_to(repository)
                case = loaded_cases.get(key)
                if case is None:
                    case = base._read(repository / artifact_path)
                    base._validate_seal(case, "case_digest")
                    loaded_cases[key] = case
                if _sha(repository / artifact_path) \
                        != method_entry.get("source_case_sha256", {}).get(key):
                    raise CrossStoreManifestError("upstream case SHA differs")
                matches = _source_matches(case, method)
                artifact_digest = case.get("case_digest")
            else:
                raise CrossStoreManifestError("unknown repair resolution")
            if len(matches) != repair.TOP_K \
                    or stable_hash(matches) != method_entry.get("effective_matches_digest"):
                raise CrossStoreManifestError("effective match digest differs")
            artifact_sha = _sha(repository / artifact_path)
            for rank, match in enumerate(matches, 1):
                preliminary.append({
                    "query_id": query_id,
                    "query_case_id": str(row["case_id"]),
                    "query_symbol": str(row["symbol"]),
                    "query_cutoff": str(row["cutoff"]),
                    "fold_id": str(row["fold_id"]),
                    "fold_role": str(row["fold_role"]),
                    "method": method,
                    "rank": rank,
                    "matched_episode_id": str(match["episode_id"]),
                    "matched_symbol": str(match["symbol"]),
                    "_declared_match_cutoff": match.get("cutoff"),
                    "distance_hex": _distance_hex(method, match),
                    "latest_eligible_ns": latest_ns,
                    "match_digest": stable_hash(match),
                    "effective_matches_digest": str(
                        method_entry["effective_matches_digest"]
                    ),
                    "resolution_kind": str(method_entry["kind"]),
                    "source_artifact_path": artifact_path.as_posix(),
                    "source_artifact_sha256": artifact_sha,
                    "source_artifact_digest": str(artifact_digest),
                })
    identifiers = sorted({row["matched_episode_id"] for row in preliminary})
    identities = _packed_identities(repository, identifiers)
    links = []
    for row in preliminary:
        symbol, cutoff, tier = identities[row["matched_episode_id"]]
        if symbol != row["matched_symbol"] \
                or row["_declared_match_cutoff"] is not None \
                    and pd.Timestamp(str(row["_declared_match_cutoff"])).isoformat() != cutoff \
                or pd.Timestamp(cutoff).value > row["latest_eligible_ns"] \
                or symbol == row["query_symbol"]:
            raise CrossStoreManifestError("effective link identity/eligibility differs")
        links.append({
            **{key: value for key, value in row.items()
               if key != "_declared_match_cutoff"},
            "matched_symbol": symbol, "matched_cutoff": cutoff,
            "quality_tier": tier,
        })
    requests = [{
        "episode_id": identifier, "dataset_id": "nasdaq",
        "symbol": identities[identifier][0], "cutoff": identities[identifier][1],
        "quality_tier": identities[identifier][2],
    } for identifier in identifiers]
    _validate_logical_inventory(links, requests)
    return links, requests


def _validate_logical_inventory(
    links: Sequence[Mapping[str, Any]], requests: Sequence[Mapping[str, Any]],
    *, expected_queries: int = EXPECTED_QUERIES,
    expected_methods: Sequence[str] = METHODS, top_k: int = repair.TOP_K,
) -> None:
    expected_links = expected_queries * len(expected_methods) * top_k
    if len(links) != expected_links:
        raise CrossStoreManifestError("raw-link count differs")
    lanes: dict[tuple[str, str], list[int]] = {}
    queries: set[str] = set()
    for row in links:
        key = (str(row.get("query_id")), str(row.get("method")))
        lanes.setdefault(key, []).append(int(row.get("rank", 0)))
        queries.add(key[0])
        if row.get("matched_symbol") == row.get("query_symbol") \
                or pd.Timestamp(str(row.get("matched_cutoff"))).value \
                    > int(row.get("latest_eligible_ns", -1)):
            raise CrossStoreManifestError("raw-link causal/symbol rule differs")
    if len(queries) != expected_queries \
            or len(lanes) != expected_queries * len(expected_methods) \
            or {method for _, method in lanes} != set(expected_methods) \
            or any(sorted(ranks) != list(range(1, top_k + 1))
                   for ranks in lanes.values()):
        raise CrossStoreManifestError("query/method/rank coverage differs")
    request_by_id = {str(row.get("episode_id")): dict(row) for row in requests}
    if len(request_by_id) != len(requests) \
            or set(request_by_id) != {str(row.get("matched_episode_id")) for row in links}:
        raise CrossStoreManifestError("episode-request coverage differs")
    for row in links:
        request = request_by_id[str(row["matched_episode_id"])]
        if (request.get("symbol"), request.get("cutoff"), request.get("quality_tier")) \
                != (row.get("matched_symbol"), row.get("matched_cutoff"), row.get("quality_tier")):
            raise CrossStoreManifestError("episode-request identity differs")


def _inputs(repository: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result, manifest, verified, rows = _read_verified_inputs(repository)
    links, requests = _logical_inventory(repository, manifest, rows)
    root = repository / repair.OUTPUT_RELATIVE
    verified_path = repository / repair_verifier.OUTPUT_RELATIVE / "VERIFIED.json"
    state = {
        "repair_result_digest": result["result_digest"],
        "repair_result_sha256": _sha(root / "RESULT.json"),
        "repair_manifest_digest": manifest["manifest_digest"],
        "repair_manifest_sha256": _sha(root / "MANIFEST.json"),
        "repair_verification_digest": verified["verification_digest"],
        "repair_verification_sha256": _sha(verified_path),
        "repair_effective_inventory_digest": verified["effective_inventory_digest"],
        "registry_sha256": _sha(repository / base.REGISTRY_FILE),
        "resident_content_digest": base._resident()["content_digest"],
        "packed_generation_id": base.GENERATION_ID,
        "packed_provenance_digest": base.PROVENANCE_DIGEST,
    }
    return state, links, requests


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise CrossStoreManifestError("globally clean Git worktree required")
    if (repository / OUTPUT_RELATIVE).exists():
        raise CrossStoreManifestError("cross-store output must be absent")
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    inputs, links, requests = _inputs(repository)
    state = {
        "schema_version": SCHEMA,
        "status": "frozen_before_cross_store_publication",
        "implementation_h0": h0,
        "runtime_files": _runtime_manifest(repository, h0),
        "inputs": inputs,
        "query_count": EXPECTED_QUERIES,
        "method_lane_count": EXPECTED_METHOD_LANES,
        "link_count": len(links),
        "raw_link_semantic_digest": _semantic_digest(links),
        "unique_episode_count": len(requests),
        "episode_request_semantic_digest": _semantic_digest(requests),
        "methods": list(METHODS),
        "top_k": repair.TOP_K,
        "output": str((repository / OUTPUT_RELATIVE).resolve()),
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    return base._sealed(state, "preregistration_digest")


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    accepted: list[str] = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0:
            continue
        for child in values[1:]:
            parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(
                repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
            )).splitlines()
            if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw:
                accepted.append(child)
    if len(set(accepted)) != 1:
        raise CrossStoreManifestError("expected one exact preregistration-only child")
    return accepted[0]


def validate_preregistration(
    repository: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise CrossStoreManifestError("globally clean Git worktree required")
    path = repository / PREREGISTRATION_RELATIVE
    raw = path.read_bytes()
    value = base._read(path)
    base._validate_seal(value, "preregistration_digest")
    if value.get("schema_version") != SCHEMA:
        raise CrossStoreManifestError("cross-store preregistration schema differs")
    h0 = str(value.get("implementation_h0"))
    h1 = _sole_child(repository, raw, h0)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository,
    ).returncode:
        raise CrossStoreManifestError("HEAD does not descend from preregistration")
    for name, expected in value.get("runtime_files", {}).items():
        if _sha(repository / name) != expected \
                or sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected:
            raise CrossStoreManifestError(f"runtime source drifted: {name}")
    inputs, links, requests = _inputs(repository)
    if any((
        value.get("inputs") != inputs,
        value.get("query_count") != EXPECTED_QUERIES,
        value.get("method_lane_count") != EXPECTED_METHOD_LANES,
        value.get("link_count") != len(links),
        value.get("raw_link_semantic_digest") != _semantic_digest(links),
        value.get("unique_episode_count") != len(requests),
        value.get("episode_request_semantic_digest") != _semantic_digest(requests),
        value.get("outcomes_or_labels_used") is not False,
    )):
        raise CrossStoreManifestError("cross-store frozen inventory drifted")
    return value, links, requests


def _frame(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records, columns=list(columns))
    if "rank" in frame:
        frame["rank"] = frame["rank"].astype("int16")
    if "latest_eligible_ns" in frame:
        frame["latest_eligible_ns"] = frame["latest_eligible_ns"].astype("int64")
    return frame


def _table_entry(path: Path, frame: pd.DataFrame, digest: str, schema: str) -> dict[str, Any]:
    return {
        "path": path.name, "schema_version": schema, "rows": len(frame),
        "columns": list(frame.columns), "bytes": path.stat().st_size,
        "sha256": _sha(path), "semantic_digest": digest,
    }


def _existing(repository: Path, preregistration: Mapping[str, Any]) -> dict[str, Any] | None:
    root = repository / OUTPUT_RELATIVE
    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir() \
            or {path.name for path in root.iterdir()} \
                != {"CONTRACT.json", LINK_FILE, REQUEST_FILE, "MANIFEST.json", "RESULT.json"}:
        raise CrossStoreManifestError("cross-store output layout differs")
    if base._read(root / "CONTRACT.json") != preregistration:
        raise CrossStoreManifestError("cross-store resume contract differs")
    manifest = base._read(root / "MANIFEST.json")
    result = base._read(root / "RESULT.json")
    base._validate_seal(manifest, "manifest_digest")
    base._validate_seal(result)
    if not all((
        result.get("passed") is True,
        result.get("manifest_digest") == manifest.get("manifest_digest"),
        result.get("manifest_sha256") == _sha(root / "MANIFEST.json"),
        result.get("preregistration_digest") == preregistration.get("preregistration_digest"),
    )):
        raise CrossStoreManifestError("cross-store terminal publication differs")
    return result


def execute(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    preregistration, links, requests = validate_preregistration(repository)
    prior = _existing(repository, preregistration)
    if prior is not None:
        return prior
    root = repository / OUTPUT_RELATIVE
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        base._atomic(temporary / "CONTRACT.json", preregistration)
        link_frame = _frame(links, LINK_COLUMNS)
        request_frame = _frame(requests, REQUEST_COLUMNS)
        link_frame.to_parquet(
            temporary / LINK_FILE, index=False, engine="pyarrow", compression="zstd",
        )
        request_frame.to_parquet(
            temporary / REQUEST_FILE, index=False, engine="pyarrow", compression="zstd",
        )
        link_digest = _semantic_digest(_plain(link_frame.to_dict("records")))
        request_digest = _semantic_digest(_plain(request_frame.to_dict("records")))
        if link_digest != preregistration["raw_link_semantic_digest"] \
                or request_digest != preregistration["episode_request_semantic_digest"]:
            raise CrossStoreManifestError("published table semantics differ")
        tables = [
            _table_entry(temporary / LINK_FILE, link_frame, link_digest, LINK_SCHEMA),
            _table_entry(temporary / REQUEST_FILE, request_frame, request_digest, REQUEST_SCHEMA),
        ]
        manifest = base._sealed({
            "schema_version": MANIFEST_SCHEMA, "status": "complete",
            "preregistration_digest": preregistration["preregistration_digest"],
            "query_count": EXPECTED_QUERIES,
            "method_lane_count": EXPECTED_METHOD_LANES,
            "link_count": len(links), "unique_episode_count": len(requests),
            "tables": tables,
            "file_inventory": ["CONTRACT.json", LINK_FILE, REQUEST_FILE],
            "outcomes_or_labels_used": False,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
        }, "manifest_digest")
        base._atomic(temporary / "MANIFEST.json", manifest)
        result = base._sealed({
            "schema_version": RESULT_SCHEMA, "status": "complete", "passed": True,
            "preregistration_digest": preregistration["preregistration_digest"],
            "manifest_digest": manifest["manifest_digest"],
            "manifest_sha256": _sha(temporary / "MANIFEST.json"),
            "query_count": EXPECTED_QUERIES,
            "method_lane_count": EXPECTED_METHOD_LANES,
            "link_count": len(links), "unique_episode_count": len(requests),
            "raw_link_semantic_digest": link_digest,
            "episode_request_semantic_digest": request_digest,
            "elapsed_seconds": perf_counter() - started,
            "outcomes_or_labels_used": False,
            "historical_walk_forward_query_outcomes_opened": False,
            "final_period_result_opened": False,
            "production_promotion_authorized": False,
            "independent_verification_authorized": True,
        })
        base._atomic(temporary / "RESULT.json", result)
        os.rename(temporary, root)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    if args.mode == "preregister":
        base._atomic(
            repository / PREREGISTRATION_RELATIVE,
            build_preregistration(repository),
        )
        return 0
    print(json.dumps(execute(repository), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

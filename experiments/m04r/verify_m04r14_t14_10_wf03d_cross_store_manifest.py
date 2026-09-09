"""Independently verify the outcome-blind WF-03D cross-store manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_exclusion_repair_full as repair
from experiments.m04r import m04r14_t14_10_wf03d_cross_store_manifest as producer
from experiments.m04r import verify_m04r14_t14_10_wf03d_exclusion_repair_full as repair_verifier


SCHEMA = "m04r14-t14-10-wf03d-cross-store-verification-v1"
OUTPUT_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03d-cross-store-manifest-v1-verification"
)
QUALITY_CODES = {1: "A", 2: "B"}


class CrossStoreVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=False,
    )
    if result.returncode:
        message = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise CrossStoreVerificationError(message.strip() or "git command failed")
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CrossStoreVerificationError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise CrossStoreVerificationError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise CrossStoreVerificationError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            CrossStoreVerificationError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise CrossStoreVerificationError(f"JSON object required: {path}")
    return value, raw


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
    for start in range(0, len(records), producer.SEMANTIC_CHUNK_ROWS):
        payload = json.dumps(
            _plain(list(records[start:start + producer.SEMANTIC_CHUNK_ROWS])),
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


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
            if parents == [child, h0] \
                    and changed == [producer.PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{producer.PREREGISTRATION_RELATIVE}", raw=True) == raw:
                accepted.append(child)
    if len(set(accepted)) != 1:
        raise CrossStoreVerificationError("preregistration lifecycle differs")
    return accepted[0]


def _source_case_key(method: str) -> str:
    return method if method in ("composite", "price_only") else "baselines"


def _source_matches(case: Mapping[str, Any], method: str) -> list[dict[str, Any]]:
    try:
        if method == "composite":
            result = case["retrieval"]["matches"]
        elif method == "price_only":
            result = case["matches"]
        else:
            result = case[
                "random_neighbors" if method == "deterministic_random"
                else "rank_neighbors"
            ]
    except (KeyError, TypeError) as exc:
        raise CrossStoreVerificationError("upstream match layout differs") from exc
    if type(result) is not list or len(result) != repair.TOP_K:
        raise CrossStoreVerificationError("upstream match inventory differs")
    return result


def _distance_hex(method: str, match: Mapping[str, Any]) -> str | None:
    if method == "composite":
        try:
            return float(match["total_distance"]).hex()
        except (KeyError, TypeError, ValueError) as exc:
            raise CrossStoreVerificationError("composite distance differs") from exc
    result = match.get("distance_hex")
    if result is not None and type(result) is not str:
        raise CrossStoreVerificationError("method distance differs")
    return result


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
    symbols = np.concatenate((packed.rows["symbol_id"], packed.overflow["symbol_id"]))
    qualities = np.concatenate((packed.rows["quality_tier"], packed.overflow["quality_tier"]))
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if len(np.unique(sorted_ids)) != len(sorted_ids):
        raise CrossStoreVerificationError("packed episode IDs are not unique")
    try:
        requested = np.asarray(
            [np.void(bytes.fromhex(value)) for value in identifiers], dtype="V12",
        )
    except ValueError as exc:
        raise CrossStoreVerificationError("invalid episode ID") from exc
    positions = np.searchsorted(sorted_ids, requested)
    if np.any(positions >= len(sorted_ids)) \
            or not np.array_equal(sorted_ids[positions], requested):
        raise CrossStoreVerificationError("episode absent from packed inventory")
    selected = order[positions]
    result: dict[str, tuple[str, str, str]] = {}
    for identifier, index in zip(identifiers, selected, strict=True):
        try:
            result[identifier] = (
                packed.symbols[int(symbols[index])],
                pd.Timestamp(int(cutoff[index]), unit="ns").isoformat(),
                QUALITY_CODES[int(qualities[index])],
            )
        except (KeyError, IndexError) as exc:
            raise CrossStoreVerificationError("packed identity code differs") from exc
    return result


def _verify_lifecycle(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg, raw = _read(repository / producer.PREREGISTRATION_RELATIVE)
    state = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("schema_version") != producer.SCHEMA \
            or prereg.get("preregistration_digest") != stable_hash(state):
        raise CrossStoreVerificationError("preregistration seal differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_child(repository, raw, h0)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository,
    ).returncode:
        raise CrossStoreVerificationError("HEAD does not descend from preregistration")
    for name, expected in prereg.get("runtime_files", {}).items():
        if _sha(repository / name) != expected \
                or sha256(_git(repository, "show", f"{h0}:{name}", raw=True)).hexdigest() != expected \
                or sha256(_git(repository, "show", f"{h1}:{name}", raw=True)).hexdigest() != expected:
            raise CrossStoreVerificationError(f"runtime source drifted: {name}")
    return prereg, {"h0": h0, "h1": h1}


def verify(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise CrossStoreVerificationError("cross-store verifier requires clean commit")
    verifier_commit = str(_git(repository, "rev-parse", "HEAD"))
    prereg, lineage = _verify_lifecycle(repository)
    repair_root = repository / repair.OUTPUT_RELATIVE
    repair_result, _ = _read(repair_root / "RESULT.json")
    repair_manifest, _ = _read(repair_root / "MANIFEST.json")
    repair_verified_path = repository / repair_verifier.OUTPUT_RELATIVE / "VERIFIED.json"
    repair_verified, _ = _read(repair_verified_path)
    for value, key in (
        (repair_result, "result_digest"),
        (repair_manifest, "manifest_digest"),
        (repair_verified, "verification_digest"),
    ):
        expected = stable_hash({name: item for name, item in value.items() if name != key})
        if value.get(key) != expected:
            raise CrossStoreVerificationError("repair prerequisite seal differs")
    expected_inputs = {
        "repair_result_digest": repair_result["result_digest"],
        "repair_result_sha256": _sha(repair_root / "RESULT.json"),
        "repair_manifest_digest": repair_manifest["manifest_digest"],
        "repair_manifest_sha256": _sha(repair_root / "MANIFEST.json"),
        "repair_verification_digest": repair_verified["verification_digest"],
        "repair_verification_sha256": _sha(repair_verified_path),
        "repair_effective_inventory_digest": repair_verified["effective_inventory_digest"],
        "registry_sha256": _sha(repository / base.REGISTRY_FILE),
        "resident_content_digest": base._resident()["content_digest"],
        "packed_generation_id": base.GENERATION_ID,
        "packed_provenance_digest": base.PROVENANCE_DIGEST,
    }
    if prereg.get("inputs") != expected_inputs \
            or repair_verified.get("cross_store_manifest_construction_authorized") is not True:
        raise CrossStoreVerificationError("repair prerequisite identity differs")

    root = repository / producer.OUTPUT_RELATIVE
    if root.is_symlink() or not root.is_dir() \
            or {path.name for path in root.iterdir()} != {
                "CONTRACT.json", producer.LINK_FILE, producer.REQUEST_FILE,
                "MANIFEST.json", "RESULT.json",
            }:
        raise CrossStoreVerificationError("cross-store file closure differs")
    contract, _ = _read(root / "CONTRACT.json")
    manifest, _ = _read(root / "MANIFEST.json")
    result, _ = _read(root / "RESULT.json")
    for value, key in ((manifest, "manifest_digest"), (result, "result_digest")):
        expected = stable_hash({name: item for name, item in value.items() if name != key})
        if value.get(key) != expected:
            raise CrossStoreVerificationError("terminal seal differs")
    if contract != prereg or not all((
        manifest.get("schema_version") == producer.MANIFEST_SCHEMA,
        manifest.get("preregistration_digest") == prereg["preregistration_digest"],
        manifest.get("query_count") == producer.EXPECTED_QUERIES,
        manifest.get("method_lane_count") == producer.EXPECTED_METHOD_LANES,
        manifest.get("link_count") == producer.EXPECTED_LINKS,
        manifest.get("unique_episode_count") == prereg.get("unique_episode_count"),
        result.get("schema_version") == producer.RESULT_SCHEMA,
        result.get("passed") is True,
        result.get("manifest_digest") == manifest.get("manifest_digest"),
        result.get("manifest_sha256") == _sha(root / "MANIFEST.json"),
        result.get("query_count") == producer.EXPECTED_QUERIES,
        result.get("method_lane_count") == producer.EXPECTED_METHOD_LANES,
        result.get("link_count") == producer.EXPECTED_LINKS,
        result.get("unique_episode_count") == prereg.get("unique_episode_count"),
        result.get("outcomes_or_labels_used") is False,
        result.get("historical_walk_forward_query_outcomes_opened") is False,
        result.get("final_period_result_opened") is False,
        result.get("production_promotion_authorized") is False,
    )):
        raise CrossStoreVerificationError("terminal publication differs")
    tables = manifest.get("tables")
    if type(tables) is not list or len(tables) != 2 \
            or manifest.get("file_inventory") \
                != ["CONTRACT.json", producer.LINK_FILE, producer.REQUEST_FILE]:
        raise CrossStoreVerificationError("table manifest differs")
    table_by_path = {entry.get("path"): entry for entry in tables}
    if set(table_by_path) != {producer.LINK_FILE, producer.REQUEST_FILE}:
        raise CrossStoreVerificationError("table inventory differs")
    for name in (producer.LINK_FILE, producer.REQUEST_FILE):
        path = root / name
        entry = table_by_path[name]
        if entry.get("bytes") != path.stat().st_size or entry.get("sha256") != _sha(path):
            raise CrossStoreVerificationError("table physical identity differs")
    if not all((
        table_by_path[producer.LINK_FILE].get("schema_version") == producer.LINK_SCHEMA,
        table_by_path[producer.LINK_FILE].get("rows") == producer.EXPECTED_LINKS,
        table_by_path[producer.LINK_FILE].get("columns") == list(producer.LINK_COLUMNS),
        table_by_path[producer.REQUEST_FILE].get("schema_version") == producer.REQUEST_SCHEMA,
        table_by_path[producer.REQUEST_FILE].get("rows") == prereg.get("unique_episode_count"),
        table_by_path[producer.REQUEST_FILE].get("columns") == list(producer.REQUEST_COLUMNS),
    )):
        raise CrossStoreVerificationError("table declaration differs")

    link_frame = pd.read_parquet(root / producer.LINK_FILE, engine="pyarrow")
    request_frame = pd.read_parquet(root / producer.REQUEST_FILE, engine="pyarrow")
    if tuple(link_frame.columns) != producer.LINK_COLUMNS \
            or tuple(request_frame.columns) != producer.REQUEST_COLUMNS \
            or len(link_frame) != producer.EXPECTED_LINKS:
        raise CrossStoreVerificationError("table schema/count differs")
    link_records = _plain(link_frame.to_dict("records"))
    request_records = _plain(request_frame.to_dict("records"))
    link_digest = _semantic_digest(link_records)
    request_digest = _semantic_digest(request_records)
    if not all((
        link_digest == prereg.get("raw_link_semantic_digest"),
        request_digest == prereg.get("episode_request_semantic_digest"),
        table_by_path[producer.LINK_FILE].get("semantic_digest") == link_digest,
        table_by_path[producer.REQUEST_FILE].get("semantic_digest") == request_digest,
        result.get("raw_link_semantic_digest") == link_digest,
        result.get("episode_request_semantic_digest") == request_digest,
    )):
        raise CrossStoreVerificationError("table semantic identity differs")

    registry, _ = base._registry(repository)
    rows = registry["queries_data"]
    entries = repair_manifest.get("queries")
    if type(entries) is not list or len(entries) != len(rows):
        raise CrossStoreVerificationError("repair query inventory differs")
    identifiers = sorted({str(row["matched_episode_id"]) for row in link_records})
    identities = _packed_identities(repository, identifiers)
    position = 0
    lanes = 0
    source_artifacts: set[str] = set()
    for query, entry in zip(rows, entries, strict=True):
        query_id = str(query["episode_id"])
        if (entry.get("query_id"), entry.get("case_id"), entry.get("symbol"), entry.get("cutoff")) \
                != (query_id, query.get("case_id"), query.get("symbol"), query.get("cutoff")):
            raise CrossStoreVerificationError("repair query binding differs")
        paths = repair._case_paths(repository, query_id)
        baseline, _ = _read(paths["baselines"])
        latest_ns = baseline.get("latest_eligible_ns")
        loaded: dict[str, dict[str, Any]] = {"baselines": baseline}
        for method_entry in entry["methods"]:
            method = str(method_entry["method"])
            if method_entry.get("kind") == "top21_drop_query_symbol":
                artifact = repair.OUTPUT_RELATIVE / str(method_entry["repair_receipt"])
                receipt, _ = _read(repository / artifact)
                matches = receipt.get("corrected_matches")
                artifact_digest = receipt.get("receipt_digest")
                if artifact_digest != stable_hash({
                    key: value for key, value in receipt.items() if key != "receipt_digest"
                }) or artifact_digest != method_entry.get("repair_receipt_digest") \
                        or _sha(repository / artifact) != method_entry.get("repair_receipt_sha256") \
                        or receipt.get("query_id") != query_id \
                        or receipt.get("method") != method:
                    raise CrossStoreVerificationError("repair receipt identity differs")
            elif method_entry.get("kind") == "upstream_top20_unchanged_over_subset":
                key = _source_case_key(method)
                artifact = paths[key].relative_to(repository)
                case = loaded.get(key)
                if case is None:
                    case, _ = _read(repository / artifact)
                    loaded[key] = case
                artifact_digest = case.get("case_digest")
                if artifact_digest != stable_hash({
                    name: value for name, value in case.items() if name != "case_digest"
                }) or _sha(repository / artifact) \
                        != method_entry.get("source_case_sha256", {}).get(key):
                    raise CrossStoreVerificationError("upstream source identity differs")
                matches = _source_matches(case, method)
            else:
                raise CrossStoreVerificationError("resolution kind differs")
            if type(matches) is not list or len(matches) != repair.TOP_K \
                    or stable_hash(matches) != method_entry.get("effective_matches_digest"):
                raise CrossStoreVerificationError("effective match inventory differs")
            artifact_sha = _sha(repository / artifact)
            source_artifacts.add(artifact.as_posix())
            for rank, match in enumerate(matches, 1):
                actual = link_records[position]
                matched_id = str(match["episode_id"])
                symbol, cutoff, tier = identities[matched_id]
                if match.get("symbol") != symbol \
                        or match.get("cutoff") is not None \
                            and pd.Timestamp(str(match["cutoff"])).isoformat() != cutoff:
                    raise CrossStoreVerificationError("source match identity differs")
                expected = {
                    "query_id": query_id, "query_case_id": str(query["case_id"]),
                    "query_symbol": str(query["symbol"]),
                    "query_cutoff": str(query["cutoff"]),
                    "fold_id": str(query["fold_id"]),
                    "fold_role": str(query["fold_role"]),
                    "method": method, "rank": rank,
                    "matched_episode_id": matched_id, "matched_symbol": symbol,
                    "matched_cutoff": cutoff, "quality_tier": tier,
                    "distance_hex": _distance_hex(method, match),
                    "latest_eligible_ns": latest_ns,
                    "match_digest": stable_hash(match),
                    "effective_matches_digest": str(method_entry["effective_matches_digest"]),
                    "resolution_kind": str(method_entry["kind"]),
                    "source_artifact_path": artifact.as_posix(),
                    "source_artifact_sha256": artifact_sha,
                    "source_artifact_digest": str(artifact_digest),
                }
                if actual != expected or symbol == query["symbol"] \
                        or pd.Timestamp(cutoff).value > int(latest_ns):
                    raise CrossStoreVerificationError("raw-link reconstruction differs")
                position += 1
            lanes += 1
    if position != producer.EXPECTED_LINKS or lanes != producer.EXPECTED_METHOD_LANES:
        raise CrossStoreVerificationError("raw-link coverage differs")
    expected_requests = [{
        "episode_id": identifier, "dataset_id": "nasdaq",
        "symbol": identities[identifier][0], "cutoff": identities[identifier][1],
        "quality_tier": identities[identifier][2],
    } for identifier in identifiers]
    if request_records != expected_requests \
            or len(request_records) != prereg.get("unique_episode_count"):
        raise CrossStoreVerificationError("episode-request reconstruction differs")

    state = {
        "schema_version": SCHEMA,
        "status": "verified_cross_store_manifest",
        "passed": True,
        "verifier_commit": verifier_commit,
        "verifier_runtime_sha256": _sha(Path(__file__).resolve()),
        "implementation_h0": lineage["h0"],
        "preregistration_h1": lineage["h1"],
        "preregistration_digest": prereg["preregistration_digest"],
        "producer_result_digest": result["result_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "raw_link_semantic_digest": link_digest,
        "episode_request_semantic_digest": request_digest,
        "query_count": producer.EXPECTED_QUERIES,
        "method_lane_count": lanes, "link_count": position,
        "unique_episode_count": len(expected_requests),
        "source_artifact_count": len(source_artifacts),
        "gates": {
            "file_manifest_closed": True,
            "all_table_bytes_and_semantics_verified": True,
            "all_query_method_rank_positions_reconstructed": True,
            "all_source_artifact_bindings_verified": True,
            "all_packed_episode_identities_verified": True,
            "all_query_symbols_excluded": True,
            "all_causal_gaps_verified": True,
            "episode_requests_exactly_deduplicated": True,
            "outcome_blind": True,
        },
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "walk_forward_outcome_store_construction_authorized": True,
        "elapsed_seconds": perf_counter() - started,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {**state, "verification_digest": stable_hash(state)}


def publish(repository: Path) -> Path:
    value = verify(repository)
    root = repository.resolve(strict=True) / OUTPUT_RELATIVE
    root.mkdir(parents=True, exist_ok=True)
    path = root / "VERIFIED.json"
    if path.exists():
        prior, _ = _read(path)
        omitted = {"verification_digest", "created_at", "elapsed_seconds"}
        if {k: v for k, v in prior.items() if k not in omitted} \
                != {k: v for k, v in value.items() if k not in omitted}:
            raise CrossStoreVerificationError("verification replay differs")
        return path
    base._atomic(path, value)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args(argv)
    print(publish(args.repository))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create and validate the immutable M04R-13 evidence catalog.

The catalog is descriptive development evidence.  In particular, the frozen
M04R-13 ``performance_passed`` field covered proposal latency and process RSS;
it did not preregister either an exact-stage or an end-to-end SLO.  This module
keeps those claims separate and never upgrades either run into production
authorization.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4


SCHEMA_VERSION = "m04r14-evidence-catalog-v1"
STATUS = "immutable_m04r13_evidence_catalog_validated"
QUERY_IDS = (
    "3307023dbe2164d025e788da",
    "3618af07dedd52fb3bdb1ccd",
    "9d7365581643bd93e85beb67",
    "99a0838725a09570b4a075ff",
)
EVIDENCE_RELATIVE = Path("config/data/analogues/m04r13")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/evidence-catalog-v1/catalog.json")
RUNTIME_FILES = ("experiments/m04r/m04r14_evidence_catalog.py",)
ROOT_ORDER = (
    "finite-threshold-diagnostic-v1",
    "threaded-certified-exposed-v1",
    "finite-threshold-diagnostic-v2",
    "threaded-certified-exposed-v2",
    "threaded-certified-exposed-verification-v2",
    "threaded-certified-exposed-verification-v2-replay",
)

# These hashes freeze the evidence bytes that existed at the T14-00 boundary.
KNOWN_ROOT_FILES: dict[str, dict[str, str]] = {
    "finite-threshold-diagnostic-v1": {
        "DIAGNOSTIC.json": "ad75e05065fdc0a9236f9ba65064b30b2f0701d8fa9b27a6488c4accec89546f",
    },
    "finite-threshold-diagnostic-v2": {
        "DIAGNOSTIC.json": "8b2cda9a1f31c8f7e57fd4da425fbad562f83725561a4d8b0ca8b669e8d28a1a",
    },
    "threaded-certified-exposed-v1": {
        "CONTRACT.json": "7c20cb8b04bb12979cae0ccdaec51da662a46d2d102cecbf6f273b33c4d30ec8",
        "PRODUCER_SEALED.json": "0983bd1c38705b91874c68597b405259581ba08e78b171f314921aa08941cb15",
        "RESIDENT.json": "dafdda9d066607a35adf4e5962438adfe8cbc6aaceef4b89c9c2f147d04bfa7b",
        "RUN_STARTED.json": "7b6aee59c1a48f04f3ec2fe00671cb5f582c0178e972f4ca8b9e28b6bb052de0",
        "cases/00-3307023dbe2164d025e788da.json": "f1b27771d3c4f44f2191a7f9f6eb9167495b7248a501dffd4c182c04442b417e",
        "cases/01-3618af07dedd52fb3bdb1ccd.json": "aafbfe30bfb9a76203d490bcb4388750c82e7bf14ec2c0038d8a357ba4b77580",
        "cases/02-9d7365581643bd93e85beb67.json": "c0e38b32c03f0316771f3d95512a71431d9ea9c05ddde49a5642caeff736cab1",
        "cases/03-99a0838725a09570b4a075ff.json": "22207fdc3103b16db2010e9c9f10e196930b3ed07de913dd4e3324802536bda5",
    },
    "threaded-certified-exposed-v2": {
        "COMPARISON.json": "69ca55798347c628653391b5600bed9dae1287fd3f4e6d9043fe2e6eac7796ef",
        "COMPARISON_SEALED.json": "00752d0c9c20cfaf930437d269161228cd0ff4fe02de00dad4eb1882910f2aa7",
        "CONTRACT.json": "6fd69e283532174406c3edd1acd4b1d9d53c9894d5a072f0ce6be968840e596a",
        "PRODUCER_SEALED.json": "de2abc1ec65e833e8ffdc6345291df01d6a398f11e9092bead374810a916c025",
        "RESIDENT.json": "dafdda9d066607a35adf4e5962438adfe8cbc6aaceef4b89c9c2f147d04bfa7b",
        "RESULTS_OPENED.json": "0d50e2745320f590067789ec711a09ed2ca0808380a5f898589b5b5a96ab1752",
        "RUN_STARTED.json": "e1c894df2084c565b20e4398f70f5556587ec3c06d2a966bb13d230700ae3e73",
        "cases/00-3307023dbe2164d025e788da.json": "8da3e6057f257a7a5b5ecf3a2d57b59724aed25f27e696fe7ec30edb1213d5d3",
        "cases/01-3618af07dedd52fb3bdb1ccd.json": "6bb07f1a2d6a6d66f1b8e0a6347ae1808ffa1481462f6a1ca69d6f7db004992f",
        "cases/02-9d7365581643bd93e85beb67.json": "bd115cb4b229f97b6ce2d9f41406a0fc135d42d41b048817bf43249399308e23",
        "cases/03-99a0838725a09570b4a075ff.json": "fbd953942b9cda85d47a45e96a63cb5fc52531109905492a3e40fccde154d07d",
    },
    "threaded-certified-exposed-verification-v2": {
        "verification.html": "69df37c0ad03c86028efc71a5d650304738f7e45c08d6503fab195503fa45b9b",
        "verification.json": "c7f37ecd921dece60a72c7f3b6c94e49999dba06d178db0a81beba75d4de0fce",
    },
    "threaded-certified-exposed-verification-v2-replay": {
        "verification.html": "0472737a2c4607c8ad71e3813ce04ea033b129decde5c153683953f8f75002ac",
        "verification.json": "bd3c30d9eec9b53c839d26ab3c20578512876bbfb2b77456e5afef3912d9219a",
    },
}


class CatalogError(ValueError):
    """The preserved M04R-13 evidence or its catalog differs."""


def _stable_hash(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
        allow_nan=False,
    ).encode()).hexdigest()


def _without(value: Mapping[str, Any], omitted: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in omitted}


def _read_bytes(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CatalogError(f"cannot open evidence file: {path}") from exc
    try:
        before = os.fstat(descriptor)
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
        item.st_ctime_ns, item.st_mode,
    )
    if not stat.S_ISREG(before.st_mode) or identity(before) != identity(after):
        raise CatalogError(f"evidence file identity changed: {path}")
    raw = b"".join(blocks)
    return raw, sha256(raw).hexdigest()


def _strict_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise CatalogError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> Any:
        raise CatalogError(f"non-finite JSON value in {label}: {value}")

    try:
        payload = json.loads(
            raw, object_pairs_hook=pairs, parse_constant=invalid,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"invalid JSON evidence: {label}") from exc
    if type(payload) is not dict:
        raise CatalogError(f"JSON evidence is not an object: {label}")
    _reject_nonfinite(payload, label)
    return payload


def _reject_nonfinite(value: Any, label: str) -> None:
    if type(value) is float and not isfinite(value):
        raise CatalogError(f"non-finite JSON value in {label}")
    if type(value) is dict:
        for item in value.values():
            _reject_nonfinite(item, label)
    elif type(value) is list:
        for item in value:
            _reject_nonfinite(item, label)


def _timestamp(value: Any, label: str) -> datetime:
    if type(value) is not str:
        raise CatalogError(f"{label} timestamp differs")
    try:
        observed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CatalogError(f"{label} timestamp differs") from exc
    if observed.tzinfo is None or observed.utcoffset() is None \
            or observed.utcoffset().total_seconds() != 0:
        raise CatalogError(f"{label} timestamp is not UTC")
    return observed


def _is_digest(value: Any) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def _require_digest(value: Mapping[str, Any], field: str, omitted: set[str], label: str) -> None:
    expected = _stable_hash(_without(value, omitted))
    if value.get(field) != expected:
        raise CatalogError(f"{label} stable digest differs")


def _git(repository: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            ("git", "-C", str(repository), *arguments), input=input_bytes,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CatalogError(f"Git runtime binding failed: {' '.join(arguments)}") from exc
    return result.stdout


def _git_runtime_manifest(repository: Path) -> dict[str, Any]:
    """Bind this catalog implementation to a clean, committed HEAD blob."""
    if _git(repository, "diff", "--name-only") or _git(
        repository, "diff", "--cached", "--name-only",
    ):
        raise CatalogError("tracked Git worktree and index must be clean")
    commit = _git(repository, "rev-parse", "HEAD").decode().strip()
    files: dict[str, str] = {}
    for relative in RUNTIME_FILES:
        _git(repository, "ls-files", "--error-unmatch", "--", relative)
        head_bytes = _git(repository, "show", f"HEAD:{relative}")
        current, current_sha = _read_bytes(repository / relative)
        head_sha = sha256(head_bytes).hexdigest()
        if current != head_bytes or current_sha != head_sha:
            raise CatalogError(f"runtime file differs from committed HEAD: {relative}")
        files[relative] = head_sha
    deterministic = {
        "implementation_commit": commit,
        "files": files,
        "files_digest": _stable_hash(files),
    }
    return {**deterministic, "digest": _stable_hash(deterministic)}


def _validate_stored_runtime_manifest(
    repository: Path, value: Mapping[str, Any],
) -> None:
    """Validate a historical build binding from a later clean descendant HEAD."""
    _validate_runtime_manifest(value)
    if _git(repository, "diff", "--name-only") or _git(
        repository, "diff", "--cached", "--name-only",
    ):
        raise CatalogError("tracked Git worktree and index must be clean")
    stored_commit = value["implementation_commit"]
    current_commit = _git(repository, "rev-parse", "HEAD").decode().strip()
    _git(repository, "merge-base", "--is-ancestor", stored_commit, current_commit)
    for relative in RUNTIME_FILES:
        _git(repository, "ls-files", "--error-unmatch", "--", relative)
        stored_bytes = _git(repository, "show", f"{stored_commit}:{relative}")
        current_head_bytes = _git(repository, "show", f"{current_commit}:{relative}")
        worktree_bytes, worktree_sha = _read_bytes(repository / relative)
        expected_sha = value["files"][relative]
        if not all((
            sha256(stored_bytes).hexdigest() == expected_sha,
            sha256(current_head_bytes).hexdigest() == expected_sha,
            worktree_sha == expected_sha,
            worktree_bytes == current_head_bytes == stored_bytes,
        )):
            raise CatalogError(f"stored runtime blob differs: {relative}")


def _validate_runtime_manifest(value: Any) -> None:
    commit = value.get("implementation_commit") if type(value) is dict else None
    try:
        commit_valid = type(commit) is str and len(bytes.fromhex(commit)) == 20
    except ValueError:
        commit_valid = False
    if type(value) is not dict or set(value) != {
        "implementation_commit", "files", "files_digest", "digest",
    } or type(value.get("files")) is not dict \
            or set(value["files"]) != set(RUNTIME_FILES) \
            or not all(_is_digest(item) for item in value["files"].values()) \
            or not commit_valid \
            or value["files_digest"] != _stable_hash(value["files"]) \
            or value["digest"] != _stable_hash(_without(value, {"digest"})):
        raise CatalogError("catalog runtime manifest differs")


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw, digest = _read_bytes(path)
    return _strict_json_bytes(raw, str(path)), digest


def _validate_root(
    repository: Path, root: Path, expected: Mapping[str, str],
) -> dict[str, Any]:
    if root.is_symlink() or root.resolve() != root.absolute() or not root.is_dir():
        raise CatalogError(f"evidence root differs: {root}")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        relative = str(path.relative_to(root))
        if stat.S_ISLNK(mode):
            raise CatalogError(f"evidence root contains a symlink: {root}")
        if stat.S_ISREG(mode):
            observed_files.add(relative)
        elif stat.S_ISDIR(mode):
            observed_directories.add(relative)
        else:
            raise CatalogError(f"evidence root contains a special entry: {root}")
    expected_directories = {
        str(parent) for relative in expected
        for parent in Path(relative).parents
        if str(parent) != "."
    }
    if observed_files != set(expected) or observed_directories != expected_directories:
        raise CatalogError(f"evidence root inventory differs: {root}")
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for relative, frozen_sha in sorted(expected.items()):
        path = root / relative
        if path.suffix == ".json":
            documents[relative], observed_sha = _read_json(path)
        else:
            _raw, observed_sha = _read_bytes(path)
        if observed_sha != frozen_sha:
            raise CatalogError(f"immutable evidence hash differs: {root.name}/{relative}")
        hashes[relative] = observed_sha
    return {
        "path": str(root.relative_to(repository)), "files": hashes,
        "files_digest": _stable_hash(hashes), "documents": documents,
    }


def _resnapshot_root(root: Path, expected: Mapping[str, str]) -> None:
    observed: dict[str, str] = {}
    directories: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        relative = str(path.relative_to(root))
        if stat.S_ISREG(mode):
            _raw, observed[relative] = _read_bytes(path)
        elif stat.S_ISDIR(mode):
            directories.add(relative)
        else:
            raise CatalogError(f"evidence tree changed during catalog build: {root}")
    expected_directories = {
        str(parent) for relative in expected for parent in Path(relative).parents
        if str(parent) != "."
    }
    if observed != dict(expected) or directories != expected_directories:
        raise CatalogError(f"evidence tree changed during catalog build: {root}")


def _case_documents(root: Mapping[str, Any]) -> list[dict[str, Any]]:
    documents = root["documents"]
    return [documents[f"cases/{ordinal:02d}-{query_id}.json"]
            for ordinal, query_id in enumerate(QUERY_IDS)]


def _candidate_digest(rows: Any) -> str:
    keys = {
        "episode_id", "symbol", "cutoff_ns", "quality_tier",
        "lower_bound_hex", "routes", "overflow_fallback",
    }
    if type(rows) is not list or not all(
        type(row) is dict and set(row) == keys for row in rows
    ):
        raise CatalogError("proposal candidate schema differs")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        try:
            bound = float.fromhex(row["lower_bound_hex"])
            identifier = bytes.fromhex(row["episode_id"])
        except (TypeError, ValueError) as exc:
            raise CatalogError("proposal candidate encoding differs") from exc
        if not all((
            len(identifier) == 12, type(row["symbol"]) is str and row["symbol"],
            type(row["cutoff_ns"]) is int and row["cutoff_ns"] >= 0,
            row["quality_tier"] in {"A", "B"},
            row["routes"] == sorted(set(row["routes"])),
            all(type(route) is str for route in row["routes"]),
            type(row["overflow_fallback"]) is bool,
            isfinite(bound) and bound >= 0 and bound.hex() == row["lower_bound_hex"],
        )):
            raise CatalogError("proposal candidate semantics differ")
        normalized.append(dict(row))
    order = [(float.fromhex(row["lower_bound_hex"]), row["episode_id"])
             for row in rows]
    if order != sorted(order) or len({row["episode_id"] for row in rows}) != len(rows):
        raise CatalogError("proposal candidate order differs")
    return _stable_hash(normalized)


def _validate_proposal(value: Any, query_id: str, generation_id: str) -> None:
    required = {
        "schema_version", "generation_id", "query_episode_id", "candidates",
        "rows_scanned", "eligible_rows", "eligible_main_rows",
        "eligible_overflow_rows", "route_counts", "route_quotas", "block_rows",
        "block_order", "elapsed_seconds", "peak_rss_mb", "candidate_digest",
        "result_digest", "contract_digest", "input_digest",
    }
    if type(value) is not dict or set(value) != required:
        raise CatalogError("proposal fields differ")
    candidates = value["candidates"]
    candidate_digest = _candidate_digest(candidates)
    deterministic = {
        "schema_version": value["schema_version"],
        "contract_digest": value["contract_digest"],
        "generation_id": value["generation_id"],
        "query_episode_id": value["query_episode_id"],
        "rows_scanned": value["rows_scanned"],
        "eligible_rows": value["eligible_rows"],
        "eligible_main_rows": value["eligible_main_rows"],
        "eligible_overflow_rows": value["eligible_overflow_rows"],
        "route_counts": value["route_counts"], "route_quotas": value["route_quotas"],
        "candidate_digest": value["candidate_digest"],
        "real_forward_outcomes_accessed": False,
        "input_digest": value["input_digest"],
    }
    numeric = (
        "rows_scanned", "eligible_rows", "eligible_main_rows",
        "eligible_overflow_rows", "block_rows",
    )
    if not all((
        value["generation_id"] == generation_id,
        value["query_episode_id"] == query_id,
        value["candidate_digest"] == candidate_digest,
        value["result_digest"] == _stable_hash(deterministic),
        all(type(value[key]) is int and value[key] >= 0 for key in numeric),
        value["eligible_rows"]
        == value["eligible_main_rows"] + value["eligible_overflow_rows"],
        type(value["elapsed_seconds"]) in {int, float}
        and isfinite(value["elapsed_seconds"]) and value["elapsed_seconds"] >= 0,
        type(value["peak_rss_mb"]) in {int, float}
        and isfinite(value["peak_rss_mb"]) and value["peak_rss_mb"] >= 0,
    )):
        raise CatalogError("proposal reconstruction differs")


def _validate_certificate(value: Any, matches: Any, query_id: str, generation_id: str) -> None:
    required = {
        "schema_version", "contract_digest", "generation_id", "query_episode_id",
        "input_digest", "eligible_candidates", "exact_evaluated", "safely_pruned",
        "stopped_early", "stop_threshold", "next_lower_bound",
        "maximum_quantized_bound_excess", "materialization_groups", "sparse_symbols",
        "batch_symbols", "rounds", "result_digest", "elapsed_seconds",
        "native_bound_accounting", "minimum_native_pruned_bound",
        "threshold_closure_passes",
    }
    if type(value) is not dict or set(value) != required or type(matches) is not list:
        raise CatalogError("certificate fields differ")
    try:
        deterministic = {
            "schema_version": value["schema_version"],
            "contract_digest": value["contract_digest"],
            "generation_id": value["generation_id"],
            "query_episode_id": value["query_episode_id"],
            "input_digest": value["input_digest"],
            "eligible_candidates": value["eligible_candidates"],
            "exact_evaluated": value["exact_evaluated"],
            "safely_pruned": value["safely_pruned"],
            "stopped_early": value["stopped_early"],
            "stop_threshold_hex": value["stop_threshold"].hex(),
            "next_lower_bound_hex": (
                value["next_lower_bound"].hex()
                if value["next_lower_bound"] is not None else None
            ),
            "maximum_quantized_bound_excess_hex":
                value["maximum_quantized_bound_excess"].hex(),
            "rounds": value["rounds"],
            "matches": [{
                "episode_id": row["episode_id"],
                "total_hex": row["total_distance"].hex(),
                "components": {
                    key: component.hex()
                    for key, component in sorted(row["component_distances"].items())
                },
                "alignment": row["alignment"],
            } for row in matches],
            "real_forward_outcomes_accessed": False,
            "native_bound_accounting": value["native_bound_accounting"],
            "minimum_native_pruned_bound_hex": (
                value["minimum_native_pruned_bound"].hex()
                if value["minimum_native_pruned_bound"] is not None else None
            ),
            "threshold_closure_passes": value["threshold_closure_passes"],
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise CatalogError("certificate numeric encoding differs") from exc
    if not all((
        value["generation_id"] == generation_id,
        value["query_episode_id"] == query_id,
        value["result_digest"] == _stable_hash(deterministic),
        value["rounds"] and type(value["rounds"]) is list,
        type(value["threshold_closure_passes"]) is list,
    )):
        raise CatalogError("certificate reconstruction differs")


def _validate_diagnostic(
    value: Mapping[str, Any], expected_digest: str,
) -> dict[str, Mapping[str, Any]]:
    cases = value.get("cases")
    if not all((
        value.get("schema_version")
        == "m04r13-truth-blind-finite-threshold-diagnostic-v1",
        value.get("status") == "truth_blind_finite_thresholds_verified",
        value.get("development_only") is True,
        value.get("production_promotion_authorized") is False,
        value.get("real_forward_outcomes_accessed") is False,
        value.get("query_ids") == list(QUERY_IDS),
        value.get("result_digest") == expected_digest,
        value.get("result_digest")
        == _stable_hash(_without(value, {"created_at", "result_digest"})),
        type(cases) is list and len(cases) == 4,
    )):
        raise CatalogError("finite-threshold diagnostic boundary differs")
    _timestamp(value.get("created_at"), "diagnostic")
    result: dict[str, Mapping[str, Any]] = {}
    for query_id, case in zip(QUERY_IDS, cases, strict=True):
        if not all((
            type(case) is dict, case.get("query_episode_id") == query_id,
            case.get("semantic_passed") is True,
            case.get("all_thresholds_finite") is True,
            case.get("real_forward_outcomes_accessed") is False,
            type(case.get("certificate_result_digest")) is str,
            len(case["certificate_result_digest"]) == 64,
            _is_digest(case.get("case_result_digest")),
            case.get("result_digest")
            == _stable_hash(_without(case, {"result_digest"})),
        )):
            raise CatalogError("finite-threshold diagnostic case differs")
        result[query_id] = case
    return result


def _validate_run(
    version: str, root: Mapping[str, Any], diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    documents = root["documents"]
    contract = documents["CONTRACT.json"]
    seal = documents["PRODUCER_SEALED.json"]
    started = documents["RUN_STARTED.json"]
    resident = documents["RESIDENT.json"]
    cases = _case_documents(root)
    diagnostic_binding = contract.get("finite_threshold_diagnostic")
    diagnostic_payload = diagnostic["documents"]["DIAGNOSTIC.json"]
    diagnostic_sha = diagnostic["files"]["DIAGNOSTIC.json"]
    git_binding = contract.get("git")
    if not all((
        type(git_binding) is dict,
        git_binding.get("files_digest") == _stable_hash(git_binding.get("files")),
        git_binding.get("digest") == _stable_hash(_without(git_binding, {"digest"})),
        contract.get("preregistration_digest")
        == _stable_hash(_without(contract, {"preregistration_digest"})),
        type(contract.get("roots")) is dict,
        Path(contract["roots"]["output_root"]).name
        == f"threaded-certified-exposed-{version}",
    )):
        raise CatalogError(f"M04R-13 {version} preregistration digest differs")
    if not all((
        contract.get("schema_version")
        == "m04r13-threaded-certified-exposed-preregistration-v1",
        contract.get("status") == "frozen_before_producer",
        contract.get("development_only") is True,
        contract.get("production_promotion_authorized") is False,
        contract.get("real_forward_outcomes_accessed") is False,
        contract.get("query_ids") == list(QUERY_IDS),
        type(diagnostic_binding) is dict,
        diagnostic_binding.get("sha256") == diagnostic_sha,
        diagnostic_binding.get("result_digest")
        == diagnostic_payload.get("result_digest"),
        seal.get("schema_version")
        == "m04r13-threaded-certified-exposed-producer-seal-v1",
        seal.get("status") == "truth_blind_producer_complete",
        seal.get("development_only") is True,
        seal.get("truth_opened") is False,
        seal.get("production_promotion_authorized") is False,
        seal.get("real_forward_outcomes_accessed") is False,
        seal.get("query_ids") == list(QUERY_IDS),
        seal.get("semantic_passed") is True,
        seal.get("performance_passed") is True,
    )):
        raise CatalogError(f"M04R-13 {version} producer boundary differs")
    started_expected = {
        "schema_version": "m04r13-run-started-v1",
        "preregistration_digest": contract["preregistration_digest"],
        "case_order": list(QUERY_IDS), "parent_max_workers": 1,
    }
    if _without(started, {"created_at"}) != started_expected:
        raise CatalogError(f"M04R-13 {version} run-start binding differs")
    _timestamp(started.get("created_at"), "run start")
    lease = resident.get("lease")
    if not all((
        type(lease) is dict,
        lease.get("lease_digest") == _stable_hash(_without(lease, {"lease_digest"})),
        resident.get("identity_digest")
        == _stable_hash(_without(resident, {"identity_digest"})),
        resident.get("content_digest") == contract.get("resident_content_digest"),
        resident.get("ready_digest") == contract.get("resident_ready_digest"),
        resident.get("identity_digest") == contract.get("resident_identity_digest"),
        lease.get("content_digest") == resident.get("content_digest"),
        lease.get("ready_digest") == resident.get("ready_digest"),
        lease.get("ready_file_sha256") == resident.get("ready_file_sha256"),
    )):
        raise CatalogError(f"M04R-13 {version} resident binding differs")
    diagnostic_certificates = _validate_diagnostic(
        diagnostic_payload, diagnostic_payload["result_digest"],
    )
    case_digests: list[str] = []
    certificate_digests: list[str] = []
    proposal_resource_passed = True
    for query_id, case in zip(QUERY_IDS, cases, strict=True):
        metrics = case.get("metrics")
        certificate = case.get("certificate")
        if not all((
            case.get("query_episode_id") == query_id,
            case.get("development_only") is True,
            case.get("truth_opened") is False,
            case.get("production_promotion_authorized") is False,
            case.get("real_forward_outcomes_accessed") is False,
            case.get("semantic_passed") is True,
            case.get("performance_passed") is True,
            type(metrics) is dict, type(certificate) is dict,
            all(type(metrics.get(key)) in {int, float}
                and isfinite(metrics[key]) and metrics[key] >= 0
                for key in (
                    "forward_proposal_seconds", "reverse_proposal_seconds",
                    "exact_task_wall_seconds", "case_task_wall_seconds",
                    "process_rss_mb",
                )),
            type(case.get("result_digest")) is str,
            type(certificate.get("result_digest")) is str,
            certificate.get("result_digest")
            == diagnostic_certificates[query_id]["certificate_result_digest"],
            case.get("preregistration_digest") == contract["preregistration_digest"],
            case.get("resident_identity_digest") == resident["identity_digest"],
            case.get("lease_digests") == [lease["lease_digest"]] * 5,
            case.get("result_digest")
            == _stable_hash(_without(case, {"created_at", "result_digest"})),
        )):
            raise CatalogError(f"M04R-13 {version} case boundary differs")
        _timestamp(case.get("created_at"), "case")
        binding = case.get("query_binding")
        diagnostic_case = diagnostic_certificates[query_id]
        if type(binding) is not dict or not all((
            binding.get("packed_provenance_digest") == contract["provenance_digest"],
            _is_digest(binding.get("packed_query_input_digest")),
            _is_digest(binding.get("certified_input_digest")),
            binding == diagnostic_case.get("query_binding"),
            case.get("registry_case_id") == diagnostic_case.get("registry_case_id"),
        )):
            raise CatalogError(f"M04R-13 {version} query binding differs")
        _validate_proposal(case.get("forward_proposal"), query_id, contract["generation_id"])
        _validate_proposal(case.get("reverse_proposal"), query_id, contract["generation_id"])
        proposal_ignored = {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}
        if _without(case["forward_proposal"], proposal_ignored) != _without(
            case["reverse_proposal"], proposal_ignored,
        ):
            raise CatalogError(f"M04R-13 {version} proposal direction differs")
        if case["forward_proposal"]["input_digest"] != binding["packed_query_input_digest"] \
                or case["certificate"].get("input_digest") \
                != binding["certified_input_digest"]:
            raise CatalogError(f"M04R-13 {version} query input binding differs")
        _validate_certificate(
            case.get("certificate"), case.get("matches"), query_id,
            contract["generation_id"],
        )
        if case.get("rounds") != certificate.get("rounds"):
            raise CatalogError(f"M04R-13 {version} certificate rounds differ")
        if not all((
            case["forward_proposal"]["candidate_digest"]
            == diagnostic_case.get("forward_candidate_digest"),
            case["forward_proposal"]["result_digest"]
            == diagnostic_case.get("forward_result_digest"),
            case["reverse_proposal"]["candidate_digest"]
            == diagnostic_case.get("reverse_candidate_digest"),
            case["reverse_proposal"]["result_digest"]
            == diagnostic_case.get("reverse_result_digest"),
            certificate["threshold_closure_passes"]
            == diagnostic_case.get("threshold_closure_passes"),
        )):
            raise CatalogError(f"M04R-13 {version} diagnostic/case cross-link differs")
        proposal_resource_passed = proposal_resource_passed and all((
            metrics["forward_proposal_seconds"] <= 120.0,
            metrics["reverse_proposal_seconds"] <= 60.0,
            metrics["process_rss_mb"] <= 1_536.0,
        ))
        case_digests.append(case["result_digest"])
        certificate_digests.append(certificate["result_digest"])
    if seal.get("case_digests") != case_digests or not proposal_resource_passed:
        raise CatalogError(f"M04R-13 {version} producer aggregate differs")
    if not all((
        seal.get("preregistration_digest") == contract["preregistration_digest"],
        seal.get("resident_identity_digest") == resident["identity_digest"],
        seal.get("seal_digest")
        == _stable_hash(_without(seal, {"created_at", "seal_digest"})),
    )):
        raise CatalogError(f"M04R-13 {version} producer seal digest differs")
    _timestamp(seal.get("created_at"), "producer seal")

    claims: dict[str, Any] = {
        "proposal_resource_gate_passed": True,
        "exact_stage_slo_passed": None,
        "end_to_end_slo_passed": None,
        "authority_semantic_equal": None,
        "authority_trace_digest_equal": None,
        "finite_diagnostic_certificate_digest_equal": True,
        "development_only": True,
        "production_promotion_authorized": False,
    }
    key_digests: dict[str, Any] = {
        "preregistration_digest": contract["preregistration_digest"],
        "finite_threshold_diagnostic_result_digest": diagnostic_payload["result_digest"],
        "producer_seal_digest": seal["seal_digest"],
        "case_result_digests": case_digests,
        "certificate_result_digests": certificate_digests,
        "verifier_runtime_sha256": contract["git"]["files"].get(
            "experiments/m04r/verify_m04r13_threaded_certified_exposed.py"
        ),
    }
    if version == "v1":
        disposition = "superseded_pre_open"
        terminal_state = "truth_blind_producer_sealed_no_comparison_artifact"
        results_opened = False
    else:
        comparison = documents["COMPARISON.json"]
        comparison_seal = documents["COMPARISON_SEALED.json"]
        marker = documents["RESULTS_OPENED.json"]
        rows = comparison.get("rows")
        if not all((
            marker.get("preregistration_digest") == contract["preregistration_digest"],
            marker.get("result_digest")
            == _stable_hash(_without(marker, {"created_at", "result_digest"})),
        )):
            raise CatalogError("M04R-13 v2 truth marker digest differs")
        _timestamp(marker.get("created_at"), "truth marker")
        if not all((
            marker.get("status") == "authority_results_opened_after_producer_seal",
            marker.get("producer_seal_digest") == seal["seal_digest"],
            marker.get("production_promotion_authorized") is False,
            comparison.get("status") == "comparison_complete",
            comparison.get("development_only") is True,
            comparison.get("post_open") is True,
            comparison.get("production_promotion_authorized") is False,
            comparison.get("semantic_passed") is True,
            comparison.get("performance_passed") is True,
            comparison.get("passed") is True,
            type(rows) is list and len(rows) == 4,
            all(row.get("matches_equal") is True
                and row.get("accounting_equal") is True
                and row.get("stock_prefix_equal") is True
                and row.get("benchmark_prefix_equal") is True
                and row.get("semantic_passed") is True for row in rows),
            [row.get("query_episode_id") for row in rows] == list(QUERY_IDS),
            [row.get("candidate_case_digest") for row in rows] == case_digests,
            comparison.get("results_opened_digest") == marker["result_digest"],
            comparison.get("producer_seal_digest") == seal["seal_digest"],
            comparison.get("result_digest")
            == _stable_hash(_without(comparison, {"created_at", "result_digest"})),
            comparison_seal.get("status") == "terminal_comparison_sealed",
            comparison_seal.get("comparison_result_digest")
            == comparison.get("result_digest"),
            comparison_seal.get("passed") is True,
            comparison_seal.get("production_promotion_authorized") is False,
            comparison_seal.get("seal_digest")
            == _stable_hash(_without(comparison_seal, {"created_at", "seal_digest"})),
        )):
            raise CatalogError("M04R-13 v2 terminal comparison differs")
        _timestamp(comparison.get("created_at"), "comparison")
        _timestamp(comparison_seal.get("created_at"), "comparison seal")
        trace_equal = all(
            row.get("certificate_result_digest_equal_diagnostic") is True
            for row in rows
        )
        claims["authority_semantic_equal"] = True
        claims["authority_trace_digest_equal"] = trace_equal
        key_digests.update({
            "results_opened_digest": marker["result_digest"],
            "comparison_result_digest": comparison["result_digest"],
            "comparison_seal_digest": comparison_seal["seal_digest"],
        })
        disposition = "active_terminal_development_evidence"
        terminal_state = "post_open_comparison_sealed_and_independently_verified"
        results_opened = True
    return {
        "version": version, "disposition": disposition,
        "terminal_state": terminal_state, "results_opened": results_opened,
        "root": root["path"], "root_files_digest": root["files_digest"],
        "root_files": root["files"], "key_digests": key_digests,
        "claims": claims,
    }


def _validate_verifications(
    primary: Mapping[str, Any], replay: Mapping[str, Any],
    v2: Mapping[str, Any],
) -> dict[str, Any]:
    values = [
        primary["documents"]["verification.json"],
        replay["documents"]["verification.json"],
    ]
    for value in values:
        if not all((
            value.get("result_digest")
            == _stable_hash(_without(value, {"created_at", "result_digest"})),
            type(value.get("artifact_sha256")) is dict,
            value.get("artifact_sha256_digest")
            == _stable_hash(value.get("artifact_sha256")),
            value.get("preregistration_digest")
            == v2["key_digests"]["preregistration_digest"],
            value.get("producer_seal_digest")
            == v2["key_digests"]["producer_seal_digest"],
            value.get("results_opened_digest")
            == v2["key_digests"]["results_opened_digest"],
            value.get("verifier_sha256")
            == v2["key_digests"]["verifier_runtime_sha256"],
            all(value["artifact_sha256"].get(relative) == digest
                for relative, digest in v2["root_files"].items()),
        )):
            raise CatalogError("M04R-13 v2 verification digest differs")
        _timestamp(value.get("created_at"), "verification")
    if not all((
        all(value.get("schema_version")
            == "m04r13-threaded-certified-independent-verification-v1"
            for value in values),
        all(value.get("verification_passed") is True for value in values),
        all(value.get("development_only") is True for value in values),
        all(value.get("production_promotion_authorized") is False for value in values),
        _without(values[0], {"created_at"})
        == _without(values[1], {"created_at"}),
        values[0].get("comparison_result_digest")
        == v2["key_digests"]["comparison_result_digest"],
        values[0].get("comparison_seal_digest")
        == v2["key_digests"]["comparison_seal_digest"],
    )):
        raise CatalogError("M04R-13 v2 independent verification differs")
    return {
        "primary_root": primary["path"],
        "primary_root_files_digest": primary["files_digest"],
        "replay_root": replay["path"],
        "replay_root_files_digest": replay["files_digest"],
        "deterministic_replay_equal": True,
        "verification_result_digest": values[0]["result_digest"],
        "artifact_sha256_digest": values[0]["artifact_sha256_digest"],
        "verification_passed": True,
        "development_only": True,
        "production_promotion_authorized": False,
    }


def _deterministic_catalog(
    repository: Path, *,
    known_root_files: Mapping[str, Mapping[str, str]] = KNOWN_ROOT_FILES,
    runtime_manifest_loader: Callable[[Path], Mapping[str, Any]] = _git_runtime_manifest,
    stored_runtime_manifest: Mapping[str, Any] | None = None,
    stored_runtime_validator: Callable[
        [Path, Mapping[str, Any]], None
    ] = _validate_stored_runtime_manifest,
) -> dict[str, Any]:
    repository = repository.absolute()
    evidence = repository / EVIDENCE_RELATIVE
    if repository.is_symlink() or repository.resolve() != repository:
        raise CatalogError("repository path is aliased")
    if evidence.is_symlink() or not evidence.is_dir():
        raise CatalogError("M04R-13 evidence directory differs")
    if set(known_root_files) != set(ROOT_ORDER):
        raise CatalogError("frozen evidence root set differs")
    if stored_runtime_manifest is None:
        runtime_manifest = dict(runtime_manifest_loader(repository))
        _validate_runtime_manifest(runtime_manifest)
    else:
        runtime_manifest = dict(stored_runtime_manifest)
        stored_runtime_validator(repository, runtime_manifest)
    observed_roots = {
        path.name for path in evidence.iterdir() if path.is_dir() and not path.is_symlink()
    }
    if observed_roots != set(ROOT_ORDER) or any(
        not path.is_dir() or path.is_symlink() for path in evidence.iterdir()
    ):
        raise CatalogError("M04R-13 root inventory differs")
    roots = {
        name: _validate_root(repository, evidence / name, known_root_files[name])
        for name in ROOT_ORDER
    }
    v1 = _validate_run(
        "v1", roots["threaded-certified-exposed-v1"],
        roots["finite-threshold-diagnostic-v1"],
    )
    v2 = _validate_run(
        "v2", roots["threaded-certified-exposed-v2"],
        roots["finite-threshold-diagnostic-v2"],
    )
    verification = _validate_verifications(
        roots["threaded-certified-exposed-verification-v2"],
        roots["threaded-certified-exposed-verification-v2-replay"], v2,
    )
    root_inventory = {
        name: {
            "path": roots[name]["path"],
            "files": roots[name]["files"],
            "files_digest": roots[name]["files_digest"],
        } for name in ROOT_ORDER
    }
    deterministic = {
        "schema_version": SCHEMA_VERSION, "status": STATUS,
        "scope": "retrospective_exposed_development_evidence_only",
        "active_run": "v2", "superseded_runs": ["v1"],
        "root_order": list(ROOT_ORDER), "roots": root_inventory,
        "roots_digest": _stable_hash(root_inventory),
        "runtime_manifest": runtime_manifest,
        "runs": [v1, v2], "v2_independent_verification": verification,
        "claim_definitions": {
            "proposal_resource_gate_passed": (
                "frozen per-case forward proposal <=120s, reverse proposal <=60s, "
                "and process RSS <=1536 MiB"
            ),
            "exact_stage_slo_passed": (
                "null: M04R-13 recorded exact-stage timing but preregistered no "
                "exact-stage SLO"
            ),
            "end_to_end_slo_passed": (
                "null: M04R-13 recorded case wall time but preregistered no "
                "end-to-end SLO"
            ),
            "authority_trace_digest_equal": (
                "descriptive only; false is expected because progressive and older "
                "exhaustive authorities have different valid certificate traces"
            ),
            "development_only": "four previously exposed hard cases",
            "production_promotion_authorized": (
                "always false; M04R-13 cannot authorize production"
            ),
        },
        "development_only": True,
        "production_promotion_authorized": False,
    }
    for name in ROOT_ORDER:
        _resnapshot_root(evidence / name, known_root_files[name])
    return deterministic


def build_catalog(
    repository: Path, *,
    known_root_files: Mapping[str, Mapping[str, str]] = KNOWN_ROOT_FILES,
    runtime_manifest_loader: Callable[[Path], Mapping[str, Any]] = _git_runtime_manifest,
) -> dict[str, Any]:
    deterministic = _deterministic_catalog(
        repository, known_root_files=known_root_files,
        runtime_manifest_loader=runtime_manifest_loader,
    )
    return {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog_digest": _stable_hash(deterministic),
    }


def validate_catalog(
    payload: Mapping[str, Any], repository: Path, *,
    known_root_files: Mapping[str, Mapping[str, str]] = KNOWN_ROOT_FILES,
    runtime_manifest_loader: Callable[[Path], Mapping[str, Any]] = _git_runtime_manifest,
    stored_runtime_validator: Callable[
        [Path, Mapping[str, Any]], None
    ] = _validate_stored_runtime_manifest,
) -> None:
    stored_runtime_manifest = (
        payload.get("runtime_manifest") if type(payload) is dict else None
    )
    if type(stored_runtime_manifest) is not dict:
        raise CatalogError("catalog runtime manifest is absent")
    expected = _deterministic_catalog(
        repository, known_root_files=known_root_files,
        runtime_manifest_loader=runtime_manifest_loader,
        stored_runtime_manifest=stored_runtime_manifest,
        stored_runtime_validator=stored_runtime_validator,
    )
    expected_keys = set(expected) | {
        "created_at", "catalog_digest",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise CatalogError("catalog fields differ")
    _timestamp(payload["created_at"], "catalog")
    deterministic = _without(payload, {"created_at", "catalog_digest"})
    if deterministic != expected:
        raise CatalogError("catalog evidence reconstruction differs")
    if payload["catalog_digest"] != _stable_hash(deterministic):
        raise CatalogError("catalog digest differs")


def write_catalog(
    path: Path, payload: Mapping[str, Any], *,
    protected_roots: tuple[Path, ...] = (),
) -> None:
    if ".." in path.parts:
        raise CatalogError("catalog output path is aliased")
    absolute = path.absolute()
    resolved = absolute.resolve()
    for root in protected_roots:
        protected = root.resolve()
        if (resolved == protected or resolved.is_relative_to(protected)
                or protected.is_relative_to(resolved)):
            raise CatalogError("catalog output overlaps immutable evidence")
    ancestor = absolute.parent
    while ancestor != ancestor.parent:
        if ancestor.exists() and ancestor.is_symlink():
            raise CatalogError("catalog output ancestry contains a symlink")
        ancestor = ancestor.parent
    if path.exists() or path.is_symlink():
        raise CatalogError("catalog output must be absent")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(
                dict(payload), indent=2, sort_keys=True, allow_nan=False,
            ) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError as exc:
        raise CatalogError("catalog output already exists") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--repository", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--repository", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.absolute()
    output = repository / OUTPUT_RELATIVE
    if arguments.command == "build":
        payload = build_catalog(repository)
        # A final reconstruction immediately before publication closes the gap
        # between the initial evidence reads and the create-only write.
        validate_catalog(payload, repository)
        write_catalog(
            output, payload,
            protected_roots=(repository / EVIDENCE_RELATIVE,),
        )
    else:
        payload, _sha = _read_json(output)
        validate_catalog(payload, repository)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

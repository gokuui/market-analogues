"""Freeze and verify the outcome-blind reuse authorities for R1-B B0-01."""

from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import datetime, timezone
import errno
import fcntl
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from market_analogues.packed_bound_store import load_packed_generation
from market_analogues.types import stable_hash


SCHEMA = "m04r14-r1b-b001-authority-audit-v1"
PREREGISTRATION = Path("experiments/m04r/m04r14_r1b_b001_authority_audit_preregistered.json")
OUTPUT = Path("config/data/analogues/m04r14/r1b-b001-authority-audit-v1")
R1A_PREREG = Path("experiments/m04r/m04r14_r1a_exposure_audit_v2_preregistered.json")
R1A_RESULT = Path("config/data/analogues/m04r14/r1a-exposure-audit-v2/RESULT.json")
R1A_VERIFIED = Path(
    "config/data/analogues/m04r14/r1a-exposure-audit-v2-integrity-verification-v1/VERIFIED.json"
)
R1A_FIRST_VERIFIED = Path(
    "config/data/analogues/m04r14/r1a-exposure-audit-v2-verification/VERIFIED.json"
)
R1A_ROOT = R1A_RESULT.parent
SHADOW_REGISTRY = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-denominator-v1/query-registry.json"
)
SHADOW_SEAL = Path("config/data/analogues/m04r14/nasdaq-shadow-denominator-v1/SEALED.json")
SHADOW_DENOMINATOR_VERIFIED = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-denominator-v1-verification/VERIFIED.json"
)
SHADOW_SEMANTIC_VERIFIED = Path(
    "config/data/analogues/m04r14/nasdaq-shadow-interrupted-verification-v1/VERIFIED.json"
)
SHADOW_CASES = Path("config/data/analogues/m04r14/nasdaq-shadow-snapshot-v1/cases")
WF_ROOT = Path("config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1")
WF_REGISTRY = WF_ROOT / "walk-forward-query-registry.json"
WF_SEAL = WF_ROOT / "SEALED.json"
WF_VERIFIED = Path(
    "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1-verification/VERIFIED.json"
)
D1_ROOT = Path("config/data/analogues/m04r14/t14-10-wf03d-exclusion-repair-full-v2")
D1_RESULT = D1_ROOT / "RESULT.json"
D1_MANIFEST = D1_ROOT / "MANIFEST.json"
D1_VERIFIED = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-exclusion-repair-full-v2-verification/VERIFIED.json"
)
D2_ROOT = Path("config/data/analogues/m04r14/t14-10-wf03d-cross-store-manifest-v1")
D2_CONTRACT = D2_ROOT / "CONTRACT.json"
D2_RESULT = D2_ROOT / "RESULT.json"
D2_MANIFEST = D2_ROOT / "MANIFEST.json"
D2_LINKS = D2_ROOT / "raw_links.parquet"
D2_REQUESTS = D2_ROOT / "episode_requests.parquet"
D2_VERIFIED = Path(
    "config/data/analogues/m04r14/t14-10-wf03d-cross-store-manifest-v1-verification/VERIFIED.json"
)
PACK_ROOT = Path("config/data/analogues/poc/m04r/packed-bound-full/store")
PACK_RESULT = Path("config/data/analogues/poc/m04r/packed-bound-full/packed-bound-full.json")
GENERATION = "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
PROVENANCE = "83ccfa62ac7ffec03e48d0f8a5634c7b8f1b8b0dde1426be22cc343fe116f62d"
PACK_MANIFEST = PACK_ROOT / "generations" / GENERATION / "manifest.json"
PACK_ROWS = PACK_MANIFEST.parent / "bound-rows.bin"
PACK_OVERFLOW = PACK_MANIFEST.parent / "overflow-exact-fallback.bin"
SOURCE_ROOTS = (
    Path("config/data/analogues/m04r14/t14-10-wf03-composite-batch-v2/cases"),
    Path("config/data/analogues/m04r14/t14-10-wf03-combined-batch-v5/cases"),
    Path("config/data/analogues/m04r14/t14-10-wf03-baseline-batch-v2/cases"),
    D1_ROOT / "repairs",
)
RUNTIME = (
    "experiments/m04r/m04r14_r1b_b001_authority_audit.py",
    "tests/test_r1b_b001_authority_audit.py",
    "src/market_analogues/packed_bound_store.py",
    "src/market_analogues/types.py",
)
METHODS = ("composite", "price_only", "deterministic_random", "recent_return_volatility")
LINK_COLUMNS = (
    "query_id", "query_case_id", "query_symbol", "query_cutoff", "fold_id",
    "fold_role", "method", "rank", "matched_episode_id", "matched_symbol",
    "matched_cutoff", "quality_tier", "distance_hex", "latest_eligible_ns",
    "match_digest", "effective_matches_digest", "resolution_kind",
    "source_artifact_path", "source_artifact_sha256", "source_artifact_digest",
)
REQUEST_COLUMNS = ("episode_id", "dataset_id", "symbol", "cutoff", "quality_tier")
FORBIDDEN_COMPONENTS = {
    "t14-09-forward-outcomes-v1", "t14-09-evidence-cards-v1",
    "t14-10-wf03d-outcome-store-v1", "t14-10-wf03d-prediction-store-v2",
    "t14-10-wf04-nonfinal-evaluation-v2", "t14-10-wf04-final-evaluation-v1",
    "t14-11-stockbee", "t14-12-post-signal",
}
FORBIDDEN_PATH_TOKENS = (
    "outcome", "prediction", "evidence-card", "evidence_card", "stockbee",
    "forward-return", "forward_return", "wf04", "t14-11", "t14-12",
)
OBSOLETE_ROOTS = (
    "r1a-exposure-audit-v1", "t14-10-wf03d-exclusion-repair-full-v1",
    "t14-10-wf03-combined-batch-v1", "t14-10-wf03-combined-batch-v2",
    "t14-10-wf03-combined-batch-v3", "t14-10-wf03-combined-batch-v4",
)
CASE_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds",
    "exact_seconds", "final_exact_seconds", "frontier_attempt_measurements",
    "peak_rss_mb", "result_digest", "checkpoint_integrity_digest",
}
PACK_NONDETERMINISTIC = {
    "created_at", "build_seconds", "generation_seconds", "validation_seconds",
    "peak_rss_mb", "scan_peak_rss_mb", "validation_peak_rss_mb",
    "inherited_scan_ru_maxrss_mb", "result_digest",
}
FROZEN_AUTHORITIES: dict[str, dict[str, Any]] = {
    "r1a": {
        "preregistration_digest": "6a125b098aee5deca3f3d3c00a0ceaef4de99e22e93ee0184e1ed13a584af584",
        "result_digest": "b22ec5f4cc8a23a6f8e9bcede9795e0bc9717f370076a365abc5666a03da4e2a",
        "first_verification_digest": "e31c31b6fc0935763151a68fcd5d77c2260ba60434fd49d1bd1058570d34a4df",
        "full_integrity_digest": "36d80c00328f2b7cdc52cdb98f8ff6669fa292b89f563c4b44da8158cb6af3c1",
        "queries": 3270, "links": 65400, "null_replicates": 512,
    },
    "shadow": {
        "registry_digest": "e5a1024e021425cace30d2172fee426ff1e6490702ea4f30583ad056e1ca89f2",
        "case_manifest_digest": "bc446bde4a5e4796487fa57aee336b1bafcde15b71c7ade0fd11a373352b5ed6",
        "retrieval_identity_digest": "a144be53e930308bf21c866bf766a9a5be7012d029c04b3dfec7cc5de5abb3c1",
        "denominator_verification_digest": "ece1ee5534020c3fee36cf996908467f94c0fd1bf48f170f5e7cbbd76f68228c",
        "semantic_verification_digest": "a8bdd370d757208075cb648eb35a81b52e872995891f12ec527fa45e804ef701",
    },
    "walk_forward": {
        "registry_digest": "784e771be69fe77a8684af6b56815ba701c90164400a2f93e53fa835a5da82c4",
        "query_digest": "4a4d070f4269542ea6ba8a5dc18e2abc3336edfbb415d6bfe7a36b0373c45b80",
        "manifest_digest": "32ff643b12087636c777a0a49120c97b44ad0714ad264d2917e7a51f7283f118",
        "seal_digest": "c7e07d78b785e9695a34388a3ef43c48bd3839a9bafa890b39b65f0718434a16",
        "verification_digest": "ae83d45c2fd0a2e98c47394c725c7013b125a585cb76c9f5b75a74399d73f810",
    },
    "d1_v2": {
        "result_digest": "a5dd3f01951c9f4f783899db3a49313743556d82b94051bbf6cbe54aa158854a",
        "manifest_digest": "33dccb5152cbb17bb686502a8188c16b2bf61a8e2ec61ba4f813ab39f4e2770d",
        "effective_inventory_digest": "94a33dfe215e06557f12637c016f0f9a65256b68a960718f9d10ce8eca709f31",
        "verification_digest": "5dca3f52f3e3f3a68a2f423fe01bdc591776e40a0f0bc45eea027f48061c2eca",
    },
    "d2": {
        "preregistration_digest": "544acfe3f79b71e5f4d012aeb169147124791fb402f50c940e0d168e028a02db",
        "result_digest": "3b2bef733c2c9efbbf41f9b38c5345dfee6ec9cf9936c1f7daa24c6923aefe81",
        "manifest_digest": "633dc1663d6f739a2636c670f434e50326ea256f296dca1b3b141a5f2c75be1d",
        "verification_digest": "1e2c5c1f98e9a23b28bd95d8e079a9a6aa46743be92fdbc1e9d1cc607c2aabbe",
        "raw_link_digest": "a0e831270c2182697dd37712d4a360fbc72640b31976541f031540b0982db6d3",
        "request_digest": "8c098c24ec33dd490d216983c999e33f7675c8f69dadd49d0a3a32bcd3485d11",
    },
    "packed": {
        "generation_id": GENERATION, "provenance_digest": PROVENANCE,
        "result_digest": "43e7246ac20e35fe696fe816764e102e4f04472ba3eb3b36fa09615dee42662f",
        "content_digest": "27dfa03eea244c1c13d27dcae7970707c60505141a297502de36588b6928437e",
        "pack_contract_digest": "9af266e24c0b539243e70c250c3c88f512c6e913a4ef9b1ada050016f0175dbf",
        "quantized_bound_contract_digest": "692e5f40b79397245d65604268c314790347fd0db124ddf985e8367ad36690d1",
        "eligible_rows": 3786156, "rows": 3786121, "overflow_rows": 35,
        "rows_bytes": 9207846272, "overflow_bytes": 1120,
        "rows_sha256": "73dfbd620bc81b3c4e4b61678a47587e2074bad4024a87f285116a123fdac6fe",
        "overflow_sha256": "d9579dfed102ac56e88943682f8a8eed50491a5a1a14842fcf028728a17ef5fd",
        "manifest_sha256": "b11c50e626be16805fc05c8f6125b31b545b35c1ac93733470f6d1ed260c732f",
    },
}

_OPENED_PATHS: dict[str, dict[str, Any]] = {}


class B001AuthorityError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B001AuthorityError(message)


def _record_open(path: Path, *, size: int, digest: str) -> None:
    key = path.absolute().as_posix()
    value = {"bytes": int(size), "sha256": digest}
    prior = _OPENED_PATHS.get(key)
    _require(prior is None or prior == value, f"authority changed across reads: {path}")
    _OPENED_PATHS[key] = value


def _sha(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    result = digest.hexdigest()
    _record_open(path, size=path.stat().st_size, digest=result)
    return result


def _load(path: Path, expected: type = dict) -> Any:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, f"duplicate JSON key {key}: {path}")
            result[key] = value
        return result

    raw = path.read_bytes()
    _record_open(path, size=len(raw), digest=sha256(raw).hexdigest())
    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            B001AuthorityError(f"non-finite JSON {token}: {path}")
        ),
    )
    _require(isinstance(value, expected), f"JSON {expected.__name__} required: {path}")
    return value


def _opened_path_manifest(repository: Path, prereg: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected_sha = dict(prereg["allowed_file_sha256"])
    expected_sha.update(prereg["runtime_sha256"])
    for item in prereg["dynamic_read_manifests"]["shadow_cases"]:
        expected_sha[(SHADOW_CASES.parent / str(item["path"])).as_posix()] = item["sha256"]
    for item in prereg["dynamic_read_manifests"]["d1_source_artifacts"]:
        expected_sha[str(item["path"])] = item["sha256"]
    for item in prereg["dynamic_read_manifests"]["d1_upstream_source_artifacts"]:
        path = str(item["path"])
        prior = expected_sha.get(path)
        _require(prior is None or prior == item["sha256"],
                 f"conflicting opened source identity: {path}")
        expected_sha[path] = item["sha256"]
    source_union = _source_union_manifest(
        prereg["dynamic_read_manifests"]["d1_source_artifacts"],
        prereg["dynamic_read_manifests"]["d1_upstream_source_artifacts"],
        expected_count=int(prereg["d1_source_union_artifact_count"]),
    )
    _require(stable_hash(source_union) == prereg["d1_source_union_manifest_digest"],
             "opened D1 source union digest differs")
    prereg_key = PREREGISTRATION.as_posix()
    prereg_actual = _OPENED_PATHS.get((repository / PREREGISTRATION).absolute().as_posix())
    _require(prereg_actual is not None, "preregistration read was not recorded")
    expected_sha[prereg_key] = prereg_actual["sha256"]
    actual: list[dict[str, Any]] = []
    for absolute, identity in sorted(_OPENED_PATHS.items()):
        path = Path(absolute)
        _require(path.is_relative_to(repository), f"opened path escaped repository: {path}")
        relative = path.relative_to(repository).as_posix()
        actual.append({"path": relative, **identity})
    _require({item["path"] for item in actual} == set(expected_sha),
             "actual opened authority path closure differs")
    _require(all(item["sha256"] == expected_sha[item["path"]] for item in actual),
             "actual opened authority identity differs")
    return actual


def _git(repository: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *args), cwd=repository, capture_output=True,
        text=not binary, check=False,
    )
    _require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout if binary else result.stdout.strip()


def _clean_head(repository: Path) -> str:
    _require(not _git(repository, "status", "--porcelain", "--untracked-files=all"),
             "clean committed tree required")
    return str(_git(repository, "rev-parse", "HEAD"))


def _digest(value: Mapping[str, Any], omitted: set[str]) -> str:
    return stable_hash({key: item for key, item in value.items() if key not in omitted})


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
    for start in range(0, len(records), 16_384):
        payload = json.dumps(
            _plain(list(records[start:start + 16_384])), sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode()
        digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    return digest.hexdigest()


def _seal_paths(repository: Path, seal_path: Path) -> tuple[Path, ...]:
    seal = _load(repository / seal_path)
    root = seal_path.parent
    names: list[str] = []
    for item in seal.get("files", []):
        _require(type(item) is dict and type(item.get("path")) is str,
                 f"invalid sealed file declaration: {seal_path}")
        name = str(item["path"])
        relative = Path(name)
        _require(not relative.is_absolute() and len(relative.parts) == 1
                 and relative.name not in names,
                 f"unsafe sealed file declaration: {seal_path}:{name}")
        names.append(relative.name)
    return tuple(repository / root / name for name in names)


def _core_files(repository: Path) -> tuple[Path, ...]:
    fixed = tuple(repository / path for path in (
        R1A_PREREG, R1A_RESULT, R1A_ROOT / "NULL_REPLICATES.json",
        R1A_ROOT / "QUERY_DIAGNOSTICS.json", R1A_ROOT / "report.html",
        R1A_FIRST_VERIFIED, R1A_VERIFIED,
        SHADOW_REGISTRY, SHADOW_SEAL, SHADOW_DENOMINATOR_VERIFIED,
        SHADOW_SEMANTIC_VERIFIED, WF_REGISTRY, WF_SEAL, WF_VERIFIED,
        D1_RESULT, D1_MANIFEST, D1_VERIFIED, D2_CONTRACT, D2_RESULT,
        D2_MANIFEST, D2_LINKS, D2_REQUESTS, D2_VERIFIED, PACK_RESULT,
        PACK_MANIFEST, PACK_ROWS, PACK_OVERFLOW,
    ))
    values = (*fixed, *_seal_paths(repository, SHADOW_SEAL),
              *_seal_paths(repository, WF_SEAL))
    _require(all(path.is_file() and not path.is_symlink() for path in values),
             "fixed authority must contain regular files")
    _require(all(path.resolve().is_relative_to(repository) for path in values),
             "fixed authority escaped repository")
    unique = tuple(dict.fromkeys(path.absolute() for path in values))
    return unique


def _case_manifest(repository: Path) -> list[dict[str, Any]]:
    root = repository / SHADOW_CASES.parent
    paths = sorted((repository / SHADOW_CASES).glob("*.json"))
    _require(len(paths) == 3270, "shadow case inventory differs")
    return [{
        "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
        "sha256": _sha(path),
    } for path in paths]


def _validate_manifest(root: Path, manifest: Sequence[Mapping[str, Any]],
                       *, expected_count: int, label: str) -> None:
    _require(len(manifest) == expected_count, f"{label} manifest count differs")
    prior = ""
    seen: set[str] = set()
    for item in manifest:
        _require(set(item) == {"path", "bytes", "sha256"},
                 f"{label} manifest shape differs")
        name = str(item["path"])
        _require(name > prior and name not in seen, f"{label} manifest order differs")
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 f"unsafe {label} manifest path")
        path = root / relative
        _require(path.is_file() and not path.is_symlink()
                 and path.stat().st_size == item["bytes"]
                 and _sha(path) == item["sha256"], f"{label} manifest entry differs: {name}")
        prior = name; seen.add(name)


def _source_manifest(repository: Path, links: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    if links is None:
        links = pd.read_parquet(repository / D2_LINKS, columns=["source_artifact_path"])
    names = sorted(set(map(str, links["source_artifact_path"])))
    result = []
    for name in names:
        path = _authorized_dynamic_path(repository, name)
        result.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    _require(len(result) == 11848, "D1 source-artifact inventory differs")
    return result


def _upstream_source_manifest(repository: Path) -> list[dict[str, Any]]:
    registry = _load(repository / WF_REGISTRY)
    queries = registry.get("queries_data", [])
    identifiers = [str(row.get("episode_id")) for row in queries]
    _require(len(identifiers) == len(set(identifiers)) == 3936,
             "walk-forward upstream query inventory differs")
    paths = sorted(
        repository / root / f"{identifier}.json"
        for root in SOURCE_ROOTS[:3] for identifier in identifiers
    )
    result = [{
        "path": path.relative_to(repository).as_posix(),
        "bytes": path.stat().st_size, "sha256": _sha(path),
    } for path in paths]
    _require(len(result) == 11808, "D1 original upstream inventory differs")
    return result


def _source_union_manifest(
    effective: Sequence[Mapping[str, Any]], upstream: Sequence[Mapping[str, Any]],
    *, expected_count: int,
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for item in (*effective, *upstream):
        value = {"path": str(item["path"]), "bytes": int(item["bytes"]),
                 "sha256": str(item["sha256"])}
        prior = by_path.get(value["path"])
        _require(prior is None or prior == value,
                 f"conflicting D1 provenance identity: {value['path']}")
        by_path[value["path"]] = value
    result = [by_path[path] for path in sorted(by_path)]
    _require(len(result) == expected_count, "D1 provenance union count differs")
    return result


def _authorized_dynamic_path(repository: Path, value: str) -> Path:
    relative = Path(value)
    _require(not relative.is_absolute() and ".." not in relative.parts,
             f"unsafe dynamic path: {value}")
    lowered = value.lower()
    _require(not any(token in lowered for token in (
        *FORBIDDEN_COMPONENTS, *FORBIDDEN_PATH_TOKENS, *OBSOLETE_ROOTS,
    )),
             f"forbidden authority path: {value}")
    candidate = repository / relative
    _require(candidate.is_file() and not candidate.is_symlink(),
             f"source artifact absent or linked: {value}")
    target = candidate.resolve()
    roots = tuple((repository / root).resolve() for root in SOURCE_ROOTS)
    _require(any(target.is_relative_to(root) for root in roots),
             f"path is outside permitted D1 roots: {value}")
    _require(target.is_file() and not target.is_symlink(), f"source artifact absent: {value}")
    return target


def _claims() -> dict[str, Any]:
    return {
        "reuse_authority_only": True, "scientific_statistics_opened": False,
        "outcomes_or_labels_accessed": False, "prediction_paths_accessed": False,
        "evidence_card_paths_accessed": False, "stockbee_paths_accessed": False,
        "forward_return_paths_accessed": False, "adequacy_labels_authorized": False,
        "predictive_claim_authorized": False, "ranking_change_authorized": False,
        "production_promotion_authorized": False,
    }


def _contract(
    repository: Path, head: str, files: Sequence[Path], cases: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]], upstream_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_union = _source_union_manifest(sources, upstream_sources, expected_count=12020)
    return {
        "schema_version": "m04r14-r1b-b001-authority-preregistration-v1",
        "implementation_commit": head,
        "output": str(OUTPUT),
        "runtime_sha256": {relative: _sha(repository / relative) for relative in RUNTIME},
        "frozen_authorities": FROZEN_AUTHORITIES,
        "frozen_authorities_digest": stable_hash(FROZEN_AUTHORITIES),
        "allowed_file_sha256": {
            path.relative_to(repository).as_posix(): _sha(path) for path in files
        },
        "dynamic_read_roots": [str((repository / root).resolve()) for root in SOURCE_ROOTS],
        "shadow_case_manifest_digest": stable_hash(list(cases)),
        "shadow_case_count": len(cases),
        "d1_source_manifest_digest": stable_hash(list(sources)),
        "d1_source_artifact_count": len(sources),
        "d1_upstream_source_manifest_digest": stable_hash(list(upstream_sources)),
        "d1_upstream_source_artifact_count": len(upstream_sources),
        "d1_source_union_manifest_digest": stable_hash(source_union),
        "d1_source_union_artifact_count": len(source_union),
        "dynamic_read_manifests": {
            "shadow_case_root": SHADOW_CASES.parent.as_posix(),
            "shadow_cases": list(cases),
            "d1_source_root": ".",
            "d1_source_artifacts": list(sources),
            "d1_upstream_source_artifacts": list(upstream_sources),
        },
        "selected_authorities": {
            "r1a": "r1a-exposure-audit-v2/full-integrity-v1",
            "shadow": "nasdaq-shadow-denominator-v1/snapshot-v1",
            "walk_forward": "t14-10-walk-forward-query-registry-v1",
            "d1": "t14-10-wf03d-exclusion-repair-full-v2",
            "d2": "t14-10-wf03d-cross-store-manifest-v1",
            "packed_generation": GENERATION,
        },
        "obsolete_authority_roots_retained_but_forbidden": list(OBSOLETE_ROOTS),
        "inventory": {
            "shadow_queries": 3270, "shadow_links": 65400,
            "walk_forward_queries": 3936, "walk_forward_scored_queries": 3360,
            "walk_forward_method_lanes": 15744, "effective_links": 314880,
            "unique_effective_episodes": 274331, "packed_candidate_episodes": 3786156,
            "packed_symbols": 11584, "d1_original_upstream_cases": 11808,
            "d1_effective_source_artifacts": 11848,
            "d1_source_artifact_union": 12020,
        },
        "claims": _claims(),
    }


def preregister(repository: Path) -> dict[str, Any]:
    _OPENED_PATHS.clear()
    repository = repository.resolve(); head = _clean_head(repository)
    path = repository / PREREGISTRATION; output = repository / OUTPUT
    _require(not path.exists() and not path.is_symlink(), "preregistration already exists")
    _require(not output.exists() and not output.is_symlink(), "output already exists")
    for relative in RUNTIME:
        _require((repository / relative).is_file(), f"runtime absent: {relative}")
    files = _core_files(repository); cases = _case_manifest(repository)
    links = pd.read_parquet(repository / D2_LINKS, columns=["source_artifact_path"])
    sources = _source_manifest(repository, links)
    upstream_sources = _upstream_source_manifest(repository)
    state = _contract(repository, head, files, cases, sources, upstream_sources)
    payload = {**state, "preregistration_digest": stable_hash(state)}
    _atomic_file(path, payload); return payload


def _validate_h1(repository: Path, prereg: Mapping[str, Any]) -> str:
    head = _clean_head(repository); h0 = str(prereg.get("implementation_commit"))
    _require(str(_git(repository, "rev-parse", "HEAD^")) == h0, "H0/H1 lineage differs")
    changed = str(_git(
        repository, "diff-tree", "--no-commit-id", "--name-only", "-r", head,
    )).splitlines()
    _require(changed == [str(PREREGISTRATION)], "H1 is not sole-child preregistration")
    blob = _git(repository, "show", f"{head}:{PREREGISTRATION}", binary=True)
    _require(blob == (repository / PREREGISTRATION).read_bytes(),
             "committed preregistration bytes differ")
    _require(prereg.get("preregistration_digest")
             == _digest(prereg, {"preregistration_digest"}), "preregistration digest differs")
    for relative, expected in prereg.get("runtime_sha256", {}).items():
        _require(relative in RUNTIME and _sha(repository / relative) == expected,
                 f"runtime differs: {relative}")
        for commit in (h0, head):
            committed = _git(repository, "show", f"{commit}:{relative}", binary=True)
            _require(sha256(committed).hexdigest() == expected,
                     f"committed runtime differs: {relative}")
    _require(set(prereg.get("runtime_sha256", {})) == set(RUNTIME),
             "runtime closure differs")
    return head


def _validate_contract(repository: Path, prereg: Mapping[str, Any]) -> None:
    files = _core_files(repository); cases = _case_manifest(repository)
    links = pd.read_parquet(repository / D2_LINKS, columns=["source_artifact_path"])
    sources = _source_manifest(repository, links)
    upstream_sources = _upstream_source_manifest(repository)
    expected = _contract(
        repository, str(prereg["implementation_commit"]), files, cases, sources,
        upstream_sources,
    )
    actual = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    _require(actual == expected, "frozen authority contract differs")
    _require(not any(
        obsolete in value for value in prereg["selected_authorities"].values()
        for obsolete in OBSOLETE_ROOTS
    ), "obsolete authority selected")
    _require(prereg.get("frozen_authorities") == FROZEN_AUTHORITIES
             and prereg.get("frozen_authorities_digest") == stable_hash(FROZEN_AUTHORITIES),
             "frozen authority constant map differs")
    expected_paths = {path.relative_to(repository).as_posix() for path in files}
    _require(set(prereg["allowed_file_sha256"]) == expected_paths,
             "allowed authority path closure differs")
    manifests = prereg["dynamic_read_manifests"]
    _require(manifests["shadow_cases"] == cases
             and manifests["d1_source_artifacts"] == sources
             and manifests["d1_upstream_source_artifacts"] == upstream_sources,
             "dynamic authority manifest differs")
    _validate_manifest(repository / manifests["shadow_case_root"], cases,
                       expected_count=3270, label="shadow case")
    _validate_manifest(repository, sources, expected_count=11848, label="D1 source")
    _validate_manifest(repository, upstream_sources, expected_count=11808,
                       label="D1 upstream source")
    source_union = _source_union_manifest(sources, upstream_sources, expected_count=12020)
    _require(prereg["d1_upstream_source_manifest_digest"] == stable_hash(upstream_sources)
             and prereg["d1_upstream_source_artifact_count"] == 11808
             and prereg["d1_source_union_manifest_digest"] == stable_hash(source_union)
             and prereg["d1_source_union_artifact_count"] == 12020,
             "D1 upstream/effective provenance closure differs")


def _validate_static_no_leakage(repository: Path) -> None:
    tree = ast.parse((repository / RUNTIME[0]).read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    forbidden = ("outcome", "prediction", "evidence_card", "stockbee", "forward_return")
    _require(not any(any(token in module.lower() for token in forbidden) for module in modules),
             "forbidden scientific-data import detected")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) \
                or node.func.id != "Path" or not node.args:
            continue
        literal = node.args[0]
        if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
            lowered = literal.value.lower()
            _require(not any(token in lowered for token in FORBIDDEN_PATH_TOKENS),
                     "forbidden scientific-data path literal detected")


def _validate_r1a_and_shadow(repository: Path) -> dict[str, Any]:
    prereg = _load(repository / R1A_PREREG)
    result = _load(repository / R1A_RESULT)
    first_verified = _load(repository / R1A_FIRST_VERIFIED)
    verified = _load(repository / R1A_VERIFIED)
    registry = _load(repository / SHADOW_REGISTRY)
    seal = _load(repository / SHADOW_SEAL)
    denominator_verified = _load(repository / SHADOW_DENOMINATOR_VERIFIED)
    semantic_verified = _load(repository / SHADOW_SEMANTIC_VERIFIED)
    frozen_r1a = FROZEN_AUTHORITIES["r1a"]
    frozen_shadow = FROZEN_AUTHORITIES["shadow"]
    _require(prereg["preregistration_digest"]
             == _digest(prereg, {"preregistration_digest"})
             == frozen_r1a["preregistration_digest"]
             and prereg.get("schema_version")
                 == "m04r14-r1a-exposure-audit-preregistration-v2"
             and prereg["inputs"]["queries"] == frozen_r1a["queries"]
             and prereg["inputs"]["retrieved_links"] == frozen_r1a["links"]
             and prereg["inputs"]["packed_candidate_episodes"] == 3786156
             and prereg["inputs"]["packed_candidate_symbols"] == 11584
             and prereg["execution"]["null_replicates"] == frozen_r1a["null_replicates"]
             and prereg["execution"]["workers"] == 12
             and prereg["claims"] == {
                 "development_diagnostic_only": True,
                 "adequacy_labels_authorized": False,
                 "predictive_claim_authorized": False,
                 "production_promotion_authorized": False,
                 "real_forward_outcomes_accessed": False,
             }, "R1-A preregistration differs")
    _require(result["result_digest"] == _digest(result, {"result_digest", "elapsed_seconds"}),
             "R1-A result differs")
    _require(result["result_digest"] == frozen_r1a["result_digest"]
             and result.get("schema_version") == "m04r14-r1a-exposure-audit-v1"
             and result.get("passed") is True
             and result.get("preregistration_digest") == prereg["preregistration_digest"]
             and result.get("inventory") == {
                 "candidate_episodes": 3786156, "candidate_symbols": 11584,
                 "queries": frozen_r1a["queries"], "retrieved_links": frozen_r1a["links"],
                 "unique_latest_eligible_cutoffs": 106,
             }
             and result.get("real_forward_outcomes_accessed") is False
             and result.get("adequacy_labels_authorized") is False
             and result.get("predictive_claim_authorized") is False
             and result.get("production_promotion_authorized") is False,
             "R1-A result/count/claim binding differs")
    _require(first_verified["verification_digest"]
             == _digest(first_verified, {"verification_digest", "created_at"})
             == frozen_r1a["first_verification_digest"]
             and first_verified.get("passed") is True
             and all(first_verified.get("gates", {}).values())
             and first_verified.get("verified_result_digest") == result["result_digest"]
             and first_verified.get("verified_queries") == frozen_r1a["queries"]
             and first_verified.get("verified_links") == frozen_r1a["links"]
             and first_verified.get("verified_null_replicates")
                 == frozen_r1a["null_replicates"]
             and first_verified.get("real_forward_outcomes_accessed") is False
             and first_verified.get("predictive_claim_authorized") is False
             and first_verified.get("production_promotion_authorized") is False,
             "R1-A first verification differs")
    _require(verified["verification_digest"]
             == _digest(verified, {"verification_digest", "created_at"})
             == frozen_r1a["full_integrity_digest"]
             and verified.get("passed") is True and all(verified.get("gates", {}).values())
             and verified.get("verified_result_digest") == result["result_digest"]
             and verified.get("verified_preregistration_digest")
                 == prereg["preregistration_digest"]
             and verified.get("verified_first_verification_digest")
                 == first_verified["verification_digest"]
             and verified.get("verified_queries") == frozen_r1a["queries"]
             and verified.get("verified_null_replicates") == frozen_r1a["null_replicates"]
             and verified.get("real_forward_outcomes_accessed") is False,
             "R1-A full integrity differs")
    _require(verified.get("adequacy_labels_authorized") is False
             and verified.get("predictive_claim_authorized") is False
             and verified.get("production_promotion_authorized") is False,
             "R1-A full-integrity claim boundary differs")
    _require({path.name for path in (repository / R1A_ROOT).iterdir()} == {
        "NULL_REPLICATES.json", "QUERY_DIAGNOSTICS.json", "RESULT.json", "report.html",
    } and all(not path.is_symlink() for path in (repository / R1A_ROOT).iterdir()),
             "R1-A result directory closure differs")
    _require(_sha(repository / R1A_ROOT / "NULL_REPLICATES.json")
             == result["artifacts"]["null_replicates_sha256"]
             and _sha(repository / R1A_ROOT / "QUERY_DIAGNOSTICS.json")
             == result["artifacts"]["query_diagnostics_sha256"],
             "R1-A result artifacts differ")
    _require(registry["registry_digest"] == _digest(registry, {"registry_digest"})
             == frozen_shadow["registry_digest"]
             and registry.get("passed") is True and registry.get("scheduled_queries") == 3270
             and len(registry.get("cases_data", [])) == 3270
             and registry.get("real_forward_outcomes_accessed") is False,
             "shadow registry differs")
    seal_state = {key: value for key, value in seal.items()
                  if key not in {"seal_digest", "created_at"}}
    declared = seal.get("files", [])
    root = repository / SHADOW_SEAL.parent
    actual_manifest = [{
        "path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path),
    } for path in sorted(root.iterdir()) if path.name != "SEALED.json"]
    _require({path.name for path in root.iterdir()} == {
        "SEALED.json", "denominator.html", "denominator.parquet",
        "query-registry.json", "query-registry.parquet",
    } and all(not path.is_symlink() and path.is_file() for path in root.iterdir())
             and declared == actual_manifest
             and seal.get("manifest_digest") == stable_hash(actual_manifest)
             and seal.get("seal_digest") == stable_hash(seal_state)
             and seal.get("registry_digest") == registry["registry_digest"],
             "shadow denominator seal differs")
    _require(denominator_verified["result_digest"]
             == _digest(denominator_verified, {"result_digest", "created_at"})
             == frozen_shadow["denominator_verification_digest"]
             and denominator_verified.get("passed") is True
             and denominator_verified.get("registry_digest") == registry["registry_digest"]
             and denominator_verified.get("seal_digest") == seal["seal_digest"]
             and denominator_verified.get("scheduled_queries") == 3270
             and denominator_verified.get("real_forward_outcomes_accessed") is False,
             "shadow denominator verification differs")
    rows = {str(row["episode_id"]): row for row in registry["cases_data"]}
    _require(len(rows) == 3270, "shadow identities are not unique")
    manifest = []
    links = 0
    identity: list[dict[str, Any]] = []
    for path in sorted((repository / SHADOW_CASES).glob("*.json")):
        case = _load(path); query_id = str(case.get("query_episode_id"))
        row = rows.get(query_id)
        _require(row is not None and all((
            case.get("gate_passed") is True,
            case.get("real_forward_outcomes_accessed") is False,
            case.get("result_digest") == _digest(case, CASE_OMITTED),
            case.get("checkpoint_integrity_digest")
                == _digest(case, {"created_at", "checkpoint_integrity_digest"}),
            case.get("registry_case_id") == row["case_id"],
            case.get("query_symbol") == row["symbol"],
            case.get("query_cutoff") == row["cutoff"],
            case.get("latest_eligible_cutoff") == row["latest_eligible_cutoff"],
            case.get("registry_digest") == registry["registry_digest"],
            case.get("generation_id") == GENERATION,
            len(case.get("matches", [])) == 20,
        )), f"shadow case differs: {query_id}")
        distances = [float(match["total_distance"]) for match in case["matches"]]
        matched_ids = [str(match["episode_id"]) for match in case["matches"]]
        symbols: dict[str, int] = {}
        for rank, match in enumerate(case["matches"], 1):
            symbols[str(match["symbol"])] = symbols.get(str(match["symbol"]), 0) + 1
            _require(pd.Timestamp(match["cutoff"]) <= pd.Timestamp(case["latest_eligible_cutoff"]),
                     f"shadow causal gap differs: {query_id}")
            identity.append({"query_episode_id": query_id,
                             "episode_id": str(match["episode_id"]), "rank": rank})
        _require(all(math.isfinite(value) and value >= 0 for value in distances)
                 and list(zip(distances, matched_ids, strict=True))
                     == sorted(zip(distances, matched_ids, strict=True))
                 and len(set(matched_ids)) == 20 and max(symbols.values()) <= 3,
                 f"shadow match ordering differs: {query_id}")
        links += len(case["matches"])
        manifest.append({
            "path": path.relative_to(repository / SHADOW_CASES.parent).as_posix(),
            "bytes": path.stat().st_size, "sha256": _sha(path),
        })
    manifest_digest = stable_hash(manifest)
    _require(len(manifest) == 3270 and links == 65400
             and manifest_digest == result["inputs"]["case_manifest_digest"]
             and manifest_digest == prereg["inputs"]["case_manifest_digest"]
             and result["inputs"]["registry_digest"] == registry["registry_digest"]
             and result["inputs"]["retrieval_identity_digest"] == stable_hash(identity),
             "shadow case manifest differs")
    _require(manifest_digest == frozen_shadow["case_manifest_digest"]
             and stable_hash(identity) == frozen_shadow["retrieval_identity_digest"]
             and result["inputs"] == {
                 "case_manifest_digest": frozen_shadow["case_manifest_digest"],
                 "packed_generation_id": GENERATION,
                 "packed_provenance_digest": PROVENANCE,
                 "registry_digest": frozen_shadow["registry_digest"],
                 "retrieval_identity_digest": frozen_shadow["retrieval_identity_digest"],
                 "semantic_verification_digest": frozen_shadow[
                     "semantic_verification_digest"
                 ],
             }, "R1-A/shadow input cross-binding differs")
    _require(semantic_verified["result_digest"]
             == _digest(semantic_verified, {"result_digest", "created_at"})
             == frozen_shadow["semantic_verification_digest"]
             and semantic_verified.get("semantic_passed") is True
             and semantic_verified.get("overall_t14_08_passed") is False
             and semantic_verified.get("verified_cases") == 3270
             and semantic_verified.get("verified_matches") == 65400
             and semantic_verified.get("registry_digest") == registry["registry_digest"]
             and semantic_verified.get("case_manifest_digest") == manifest_digest
             and semantic_verified.get("real_forward_outcomes_accessed") is False
             and result["inputs"]["semantic_verification_digest"]
                 == semantic_verified["result_digest"]
             and prereg["inputs"]["semantic_verification_digest"]
                 == semantic_verified["result_digest"],
             "shadow semantic verification differs")
    return {
        "r1a_preregistration_digest": prereg["preregistration_digest"],
        "r1a_result_digest": result["result_digest"],
        "r1a_full_integrity_digest": verified["verification_digest"],
        "shadow_registry_digest": registry["registry_digest"],
        "shadow_denominator_verification_digest": denominator_verified["result_digest"],
        "shadow_semantic_verification_digest": semantic_verified["result_digest"],
        "shadow_case_manifest_digest": manifest_digest,
        "shadow_retrieval_identity_digest": stable_hash(identity),
    }


def _validate_wf_registry(repository: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    registry = _load(repository / WF_REGISTRY); seal = _load(repository / WF_SEAL)
    verified = _load(repository / WF_VERIFIED)
    frozen = FROZEN_AUTHORITIES["walk_forward"]
    _require(registry["registry_digest"] == _digest(registry, {"registry_digest"})
             == frozen["registry_digest"]
             and registry.get("passed") is True and registry.get("queries") == 3936
             and registry.get("scored_queries") == 3360
             and registry.get("historical_walk_forward_query_outcomes_opened") is False
             and registry.get("final_period_result_opened") is False,
             "walk-forward registry differs")
    queries = tuple(registry.get("queries_data", []))
    ids = [str(row["episode_id"]) for row in queries]
    case_ids = [str(row["case_id"]) for row in queries]
    _require(len(queries) == len(set(ids)) == len(set(case_ids)) == 3936
             and registry["query_digest"] == stable_hash(list(queries))
             == frozen["query_digest"],
             "walk-forward query identities differ")
    expected_files = {str(row["path"]): row for row in seal.get("files", [])}
    actual = sorted(path.name for path in (repository / WF_ROOT).iterdir() if path.name != "SEALED.json")
    _require(sorted(expected_files) == actual
             and seal["manifest_digest"] == stable_hash(seal["files"])
             == frozen["manifest_digest"],
             "walk-forward file manifest differs")
    _require(seal["seal_digest"] == _digest(seal, {"seal_digest", "created_at"})
             == frozen["seal_digest"],
             "walk-forward seal differs")
    for name, item in expected_files.items():
        path = repository / WF_ROOT / name
        _require(path.stat().st_size == item["bytes"] and _sha(path) == item["sha256"],
                 f"walk-forward file differs: {name}")
    _require(verified["result_digest"]
             == _digest(verified, {"result_digest", "elapsed_seconds"})
             == frozen["verification_digest"]
             and verified.get("passed") is True and verified.get("queries") == 3936
             and verified.get("registry_digest") == registry["registry_digest"]
             and verified.get("registry_sha256") == _sha(repository / WF_REGISTRY)
             and verified.get("registry_seal_sha256") == _sha(repository / WF_SEAL)
             and verified.get("scored_queries") == 3360
             and verified.get("manifest_closed") is True
             and verified.get("historical_walk_forward_query_outcomes_opened") is False
             and verified.get("final_period_result_opened") is False,
             "walk-forward verification differs")
    return {
        "wf_registry_digest": registry["registry_digest"],
        "wf_query_digest": registry["query_digest"],
        "wf_manifest_digest": seal["manifest_digest"],
        "wf_verification_digest": verified["result_digest"],
    }, queries


def _validate_packed(repository: Path) -> tuple[dict[str, Any], Any]:
    result = _load(repository / PACK_RESULT); manifest = _load(repository / PACK_MANIFEST)
    frozen = FROZEN_AUTHORITIES["packed"]
    _require(result["result_digest"] == _digest(result, PACK_NONDETERMINISTIC)
             == frozen["result_digest"]
             and result.get("gate_passed") is True
             and result.get("poc_passed") is True
             and result.get("real_forward_outcomes_accessed") is False
             and result.get("generation_id") == GENERATION
             and result.get("eligible_rows") == frozen["eligible_rows"]
             and result.get("rows") == frozen["rows"]
             and result.get("overflow_rows") == frozen["overflow_rows"]
             and result.get("pack_bytes")
                 == frozen["rows_bytes"] + frozen["overflow_bytes"]
             and result.get("pack_contract_digest") == frozen["pack_contract_digest"]
             and result.get("authority_row_accounting_passed") is True
             and len(result.get("full_authority_row_counts", [])) == 12
             and all(row.get("row_accounting_matches") is True
                     for row in result.get("full_authority_row_counts", [])),
             "packed build result differs")
    _require(manifest["manifest_digest"] == _digest(manifest, {"manifest_digest"})
             and manifest.get("manifest_digest") == frozen["generation_id"]
             and manifest.get("provenance_digest") == frozen["provenance_digest"]
             and manifest.get("real_forward_outcomes_accessed") is False
             and manifest.get("eligible_row_count") == frozen["eligible_rows"]
             and manifest.get("row_count") == frozen["rows"]
             and manifest.get("overflow_count") == frozen["overflow_rows"]
             and manifest.get("rows_bytes") == frozen["rows_bytes"]
             and manifest.get("overflow_bytes") == frozen["overflow_bytes"]
             and manifest.get("rows_sha256") == frozen["rows_sha256"]
             and manifest.get("overflow_sha256") == frozen["overflow_sha256"]
             and manifest.get("pack_contract_digest") == frozen["pack_contract_digest"]
             and manifest.get("quantized_bound_contract_digest")
                 == frozen["quantized_bound_contract_digest"]
             and _sha(repository / PACK_MANIFEST) == frozen["manifest_sha256"]
             and len(manifest.get("symbols", [])) == 11584,
             "packed manifest differs")
    _require(result["eligible_rows"] == manifest["eligible_row_count"]
             and result["rows"] == manifest["row_count"]
             and result["overflow_rows"] == manifest["overflow_count"]
             and result["pack_bytes"]
                 == manifest["rows_bytes"] + manifest["overflow_bytes"]
             and result["pack_contract_digest"] == manifest["pack_contract_digest"],
             "packed result/manifest linkage differs")
    _require({path.name for path in (repository / PACK_MANIFEST.parent).iterdir()} == {
        "manifest.json", "bound-rows.bin", "overflow-exact-fallback.bin",
    } and manifest["rows_file"] == PACK_ROWS.name
             and manifest["overflow_file"] == PACK_OVERFLOW.name,
             "packed generation file closure differs")
    loaded = load_packed_generation(
        repository / PACK_ROOT, GENERATION, expected_provenance_digest=PROVENANCE,
        verify_content=True, validate_records=True,
    )
    file_bindings = {
        "manifest": {"bytes": (repository / PACK_MANIFEST).stat().st_size,
                     "sha256": _sha(repository / PACK_MANIFEST)},
        "overflow": {"bytes": manifest["overflow_bytes"],
                     "sha256": manifest["overflow_sha256"]},
        "rows": {"bytes": manifest["rows_bytes"], "sha256": manifest["rows_sha256"]},
    }
    resident_content = {
        "schema_version": "m04r-resident-packed-store-content-v1",
        "generation_id": GENERATION, "manifest_digest": GENERATION,
        "provenance_digest": PROVENANCE,
        "pack_contract_digest": manifest["pack_contract_digest"],
        "quantized_bound_contract_digest": manifest["quantized_bound_contract_digest"],
        "physical_generation_bytes": sum(item["bytes"] for item in file_bindings.values()),
        "source_files": file_bindings, "mirror_files": file_bindings,
    }
    _require(stable_hash(resident_content) == frozen["content_digest"],
             "packed resident content reconstruction differs")
    return {
        "packed_result_digest": result["result_digest"],
        "packed_generation_id": GENERATION, "packed_provenance_digest": PROVENANCE,
        "packed_candidate_episodes": manifest["eligible_row_count"],
        "packed_content_digest": stable_hash(resident_content),
    }, loaded


def _source_matches(source: Mapping[str, Any], method: str, repaired: bool) -> list[dict[str, Any]]:
    if repaired:
        values = source.get("corrected_matches")
    elif method == "composite":
        values = source.get("retrieval", {}).get("matches")
    elif method == "price_only":
        values = source.get("matches")
    else:
        values = source.get(
            "random_neighbors" if method == "deterministic_random" else "rank_neighbors"
        )
    _require(isinstance(values, list) and len(values) == 20, "D1 source matches differ")
    return values


def _packed_identities(loaded: Any, identifiers: Sequence[str]) -> dict[str, tuple[str, str, str]]:
    ids = np.concatenate((loaded.rows["episode_id"], loaded.overflow["episode_id"]))
    cutoffs = np.concatenate((loaded.rows["cutoff_ns"], loaded.overflow["cutoff_ns"]))
    symbols = np.concatenate((loaded.rows["symbol_id"], loaded.overflow["symbol_id"]))
    tiers = np.concatenate((loaded.rows["quality_tier"], loaded.overflow["quality_tier"]))
    order = np.argsort(ids, kind="stable"); ordered = ids[order]
    _require(len(np.unique(ordered)) == len(ordered), "packed episode IDs duplicate")
    requested = np.asarray([np.void(bytes.fromhex(value)) for value in identifiers], dtype="V12")
    positions = np.searchsorted(ordered, requested)
    _require(np.all(positions < len(ordered)) and np.array_equal(ordered[positions], requested),
             "D2 episode absent from packed generation")
    selected = order[positions]; quality = {1: "A", 2: "B"}
    return {
        identifier: (
            loaded.symbols[int(symbols[index])],
            pd.Timestamp(int(cutoffs[index]), unit="ns").isoformat(),
            quality[int(tiers[index])],
        ) for identifier, index in zip(identifiers, selected, strict=True)
    }


def _validate_d1_d2(
    repository: Path, queries: Sequence[Mapping[str, Any]], loaded: Any,
) -> dict[str, Any]:
    frozen_d1 = FROZEN_AUTHORITIES["d1_v2"]
    frozen_d2 = FROZEN_AUTHORITIES["d2"]
    d1_result = _load(repository / D1_RESULT); d1_manifest = _load(repository / D1_MANIFEST)
    d1_verified = _load(repository / D1_VERIFIED)
    for value, field in ((d1_result, "result_digest"), (d1_manifest, "manifest_digest"),
                         (d1_verified, "verification_digest")):
        _require(value[field] == _digest(value, {field}), f"D1 {field} differs")
    _require(d1_result["result_digest"] == frozen_d1["result_digest"]
             and d1_manifest["manifest_digest"] == frozen_d1["manifest_digest"]
             and d1_manifest["effective_inventory_digest"]
                 == frozen_d1["effective_inventory_digest"]
             and d1_verified["verification_digest"] == frozen_d1["verification_digest"],
             "frozen D1-v2 authority differs")
    _require(d1_result.get("passed") is True and d1_result.get("query_count") == 3936
             and d1_result.get("effective_neighbour_links") == 314880
             and d1_result.get("methods_per_query") == 4
             and d1_result.get("repair_receipts") == 212
             and d1_result.get("manifest_digest") == d1_manifest["manifest_digest"]
             and d1_result.get("manifest_sha256") == _sha(repository / D1_MANIFEST)
             and d1_result.get("all_effective_matches_exclude_query_symbol") is True
             and d1_manifest.get("query_count") == 3936
             and d1_manifest.get("method_links") == 15744
             and d1_manifest.get("effective_neighbour_links") == 314880
             and d1_verified.get("passed") is True and all(d1_verified.get("gates", {}).values())
             and d1_verified.get("producer_result_digest") == d1_result["result_digest"]
             and d1_verified.get("producer_result_sha256") == _sha(repository / D1_RESULT)
             and d1_verified.get("manifest_digest") == d1_manifest["manifest_digest"]
             and d1_verified.get("query_count") == 3936
             and d1_verified.get("method_lanes") == 15744
             and d1_verified.get("effective_neighbour_links") == 314880
             and d1_verified.get("effective_inventory_digest")
                == d1_manifest["effective_inventory_digest"]
             and d1_verified.get("cross_store_manifest_construction_authorized") is True
             and d1_verified.get("outcomes_or_labels_used") is False
             and d1_verified.get("historical_walk_forward_query_outcomes_opened") is False
             and d1_verified.get("final_period_result_opened") is False
             and d1_verified.get("production_promotion_authorized") is False
             and d1_result.get("outcomes_or_labels_used") is False
             and d1_result.get("historical_walk_forward_query_outcomes_opened") is False
             and d1_result.get("final_period_result_opened") is False
             and d1_manifest.get("outcomes_or_labels_used") is False
             and d1_manifest.get("historical_walk_forward_query_outcomes_opened") is False
             and d1_manifest.get("final_period_result_opened") is False,
             "current D1 authority differs")
    contract = _load(repository / D2_CONTRACT); result = _load(repository / D2_RESULT)
    manifest = _load(repository / D2_MANIFEST); verified = _load(repository / D2_VERIFIED)
    _require(contract["preregistration_digest"]
             == _digest(contract, {"preregistration_digest"})
             == frozen_d2["preregistration_digest"], "D2 contract differs")
    expected_d2_inputs = {
        "repair_result_digest": d1_result["result_digest"],
        "repair_result_sha256": _sha(repository / D1_RESULT),
        "repair_manifest_digest": d1_manifest["manifest_digest"],
        "repair_manifest_sha256": _sha(repository / D1_MANIFEST),
        "repair_verification_digest": d1_verified["verification_digest"],
        "repair_verification_sha256": _sha(repository / D1_VERIFIED),
        "repair_effective_inventory_digest": d1_manifest["effective_inventory_digest"],
        "registry_sha256": _sha(repository / WF_REGISTRY),
        "resident_content_digest": stable_hash({
            "schema_version": "m04r-resident-packed-store-content-v1",
            "generation_id": loaded.manifest["manifest_digest"],
            "manifest_digest": loaded.manifest["manifest_digest"],
            "provenance_digest": loaded.manifest["provenance_digest"],
            "pack_contract_digest": loaded.manifest["pack_contract_digest"],
            "quantized_bound_contract_digest": loaded.manifest[
                "quantized_bound_contract_digest"
            ],
            "physical_generation_bytes": (
                (repository / PACK_MANIFEST).stat().st_size
                + loaded.manifest["rows_bytes"] + loaded.manifest["overflow_bytes"]
            ),
            "source_files": {
                "manifest": {"bytes": (repository / PACK_MANIFEST).stat().st_size,
                             "sha256": _sha(repository / PACK_MANIFEST)},
                "overflow": {"bytes": loaded.manifest["overflow_bytes"],
                             "sha256": loaded.manifest["overflow_sha256"]},
                "rows": {"bytes": loaded.manifest["rows_bytes"],
                         "sha256": loaded.manifest["rows_sha256"]},
            },
            "mirror_files": {
                "manifest": {"bytes": (repository / PACK_MANIFEST).stat().st_size,
                             "sha256": _sha(repository / PACK_MANIFEST)},
                "overflow": {"bytes": loaded.manifest["overflow_bytes"],
                             "sha256": loaded.manifest["overflow_sha256"]},
                "rows": {"bytes": loaded.manifest["rows_bytes"],
                         "sha256": loaded.manifest["rows_sha256"]},
            },
        }),
        "packed_generation_id": GENERATION,
        "packed_provenance_digest": PROVENANCE,
    }
    _require(contract.get("inputs") == expected_d2_inputs
             and contract.get("query_count") == 3936
             and contract.get("method_lane_count") == 15744
             and contract.get("link_count") == 314880
             and contract.get("unique_episode_count") == 274331
             and contract.get("methods") == list(METHODS)
             and contract.get("outcomes_or_labels_used") is False
             and contract.get("historical_walk_forward_query_outcomes_opened") is False
             and contract.get("final_period_result_opened") is False
             and contract.get("production_promotion_authorized") is False,
             "D2 frozen D1-v2/packed provenance differs")
    _require(manifest["manifest_digest"] == _digest(manifest, {"manifest_digest"})
             and result["result_digest"] == _digest(result, {"result_digest"})
             and verified["verification_digest"] == _digest(verified, {"verification_digest"}),
             "D2 terminal digest differs")
    _require(manifest["manifest_digest"] == frozen_d2["manifest_digest"]
             and result["result_digest"] == frozen_d2["result_digest"]
             and verified["verification_digest"] == frozen_d2["verification_digest"],
             "frozen D2 authority differs")
    _require(result.get("passed") is True and verified.get("passed") is True
             and all(verified.get("gates", {}).values())
             and verified.get("producer_result_digest") == result["result_digest"]
             and result.get("manifest_digest") == manifest["manifest_digest"]
             and result.get("manifest_sha256") == _sha(repository / D2_MANIFEST)
             and result.get("preregistration_digest") == contract["preregistration_digest"]
             and manifest.get("preregistration_digest") == contract["preregistration_digest"]
             and verified.get("preregistration_digest") == contract["preregistration_digest"]
             and verified.get("manifest_digest") == manifest["manifest_digest"]
             and result.get("query_count") == 3936
             and result.get("method_lane_count") == 15744
             and result.get("link_count") == 314880
             and result.get("unique_episode_count") == 274331
             and manifest.get("query_count") == 3936
             and manifest.get("method_lane_count") == 15744
             and manifest.get("link_count") == 314880
             and manifest.get("unique_episode_count") == 274331
             and verified.get("query_count") == 3936
             and verified.get("method_lane_count") == 15744
             and verified.get("link_count") == 314880
             and verified.get("unique_episode_count") == 274331
             and verified.get("source_artifact_count") == 11848
             and result.get("outcomes_or_labels_used") is False
             and result.get("historical_walk_forward_query_outcomes_opened") is False
             and result.get("final_period_result_opened") is False
             and verified.get("outcomes_or_labels_used") is False
             and verified.get("historical_walk_forward_query_outcomes_opened") is False
             and verified.get("final_period_result_opened") is False
             and verified.get("production_promotion_authorized") is False,
             "current D2 authority differs")
    _require({path.name for path in (repository / D2_ROOT).iterdir()} == {
        "CONTRACT.json", "MANIFEST.json", "RESULT.json", "raw_links.parquet",
        "episode_requests.parquet",
    }, "D2 directory closure differs")
    table_by = {row["path"]: row for row in manifest["tables"]}
    _require(manifest.get("file_inventory")
             == ["CONTRACT.json", "raw_links.parquet", "episode_requests.parquet"]
             and set(table_by) == {"raw_links.parquet", "episode_requests.parquet"},
             "D2 table manifest differs")
    for name, entry in table_by.items():
        path = repository / D2_ROOT / name
        _require(entry["bytes"] == path.stat().st_size and entry["sha256"] == _sha(path),
                 f"D2 table bytes differ: {name}")
    link_frame = pd.read_parquet(repository / D2_LINKS)
    request_frame = pd.read_parquet(repository / D2_REQUESTS)
    _require(tuple(link_frame.columns) == LINK_COLUMNS and tuple(request_frame.columns) == REQUEST_COLUMNS
             and len(link_frame) == 314880 and len(request_frame) == 274331,
             "D2 table schema/count differs")
    links = _plain(link_frame.to_dict("records")); requests = _plain(request_frame.to_dict("records"))
    link_digest = _semantic_digest(links); request_digest = _semantic_digest(requests)
    _require(link_digest == result["raw_link_semantic_digest"]
             == table_by["raw_links.parquet"]["semantic_digest"]
             == contract["raw_link_semantic_digest"]
             == verified["raw_link_semantic_digest"]
             == frozen_d2["raw_link_digest"]
             and request_digest == result["episode_request_semantic_digest"]
             == table_by["episode_requests.parquet"]["semantic_digest"]
             == contract["episode_request_semantic_digest"]
             == verified["episode_request_semantic_digest"]
             == frozen_d2["request_digest"],
             "D2 table semantic digest differs")
    entries = d1_manifest.get("queries", [])
    _require(len(entries) == len(queries) == 3936, "D1/WF query closure differs")
    identifiers = sorted({str(row["matched_episode_id"]) for row in links})
    identities = _packed_identities(loaded, identifiers)
    source_cache: dict[str, tuple[dict[str, Any], str]] = {}
    effective_states: list[dict[str, str]] = []
    position = 0
    for query, entry in zip(queries, entries, strict=True):
        query_id = str(query["episode_id"])
        _require((entry.get("query_id"), entry.get("case_id"), entry.get("symbol"), entry.get("cutoff"))
                 == (query_id, query["case_id"], query["symbol"], query["cutoff"]),
                 "D1 query provenance differs")
        _require([row.get("method") for row in entry.get("methods", [])] == list(METHODS),
                 "D1 method order differs")
        baseline_path = repository / SOURCE_ROOTS[2] / f"{query_id}.json"
        baseline_case = _load(baseline_path)
        _require(baseline_case.get("case_digest")
                 == _digest(baseline_case, {"case_digest"}),
                 "D1 baseline source seal differs")
        latest_eligible_ns = int(baseline_case["latest_eligible_ns"])
        upstream_paths = {
            "composite": repository / SOURCE_ROOTS[0] / f"{query_id}.json",
            "price_only": repository / SOURCE_ROOTS[1] / f"{query_id}.json",
            "baselines": baseline_path,
        }
        upstream_sha = {key: _sha(path) for key, path in upstream_paths.items()}
        for method_entry in entry["methods"]:
            method = str(method_entry["method"]); lane = links[position:position + 20]
            _require(len(lane) == 20 and [row["rank"] for row in lane] == list(range(1, 21))
                     and all(row["query_id"] == query_id and row["method"] == method for row in lane),
                     "D2 lane order differs")
            path_name = str(lane[0]["source_artifact_path"])
            _require(all(row["source_artifact_path"] == path_name for row in lane),
                     "D2 lane source path differs")
            path = _authorized_dynamic_path(repository, path_name)
            cached = source_cache.get(path_name)
            if cached is None:
                source = _load(path)
                digest_field = "receipt_digest" if "repair_receipt" in method_entry else "case_digest"
                artifact_digest = source.get(digest_field)
                _require(artifact_digest == _digest(source, {digest_field}),
                         f"D1 source seal differs: {path_name}")
                cached = (source, _sha(path)); source_cache[path_name] = cached
            source, source_sha = cached
            repaired = method_entry.get("kind") == "top21_drop_query_symbol"
            expected_path = (
                D1_ROOT / str(method_entry.get("repair_receipt"))
                if repaired else SOURCE_ROOTS[
                    0 if method == "composite" else 1 if method == "price_only" else 2
                ] / f"{query_id}.json"
            ).as_posix()
            _require(path_name == expected_path, "D1 source artifact route differs")
            _require(method_entry.get("source_case_sha256") == upstream_sha,
                     "D1 upstream source provenance differs")
            matches = _source_matches(source, method, repaired)
            _require(stable_hash(matches) == method_entry["effective_matches_digest"],
                     "D1 effective match digest differs")
            if repaired:
                _require(source_sha == method_entry["repair_receipt_sha256"]
                         and source["receipt_digest"] == method_entry["repair_receipt_digest"]
                         and source.get("query_id") == query_id
                         and source.get("case_id") == query["case_id"]
                         and source.get("symbol") == query["symbol"]
                         and source.get("cutoff") == query["cutoff"]
                         and source.get("method") == method
                         and source.get("source_case_sha256") == upstream_sha
                         and source.get("outcomes_or_labels_used") is False
                         and source.get("historical_walk_forward_query_outcomes_opened") is False
                         and source.get("final_period_result_opened") is False
                         and source.get("production_promotion_authorized") is False,
                         "D1 repair provenance differs")
            else:
                key = method if method in {"composite", "price_only"} else "baselines"
                _require(source_sha == method_entry["source_case_sha256"][key],
                         "D1 upstream provenance differs")
                _require(source.get("query_id") == query_id
                         and source.get("case_id") == query["case_id"]
                         and source.get("symbol") == query["symbol"]
                         and source.get("cutoff") == query["cutoff"]
                         and source.get("outcomes_or_labels_used") is False
                         and source.get("historical_walk_forward_query_outcomes_opened") is False
                         and source.get("final_period_result_opened") is False,
                         "D1 upstream source binding differs")
            symbols: set[str] = set()
            for row, match in zip(lane, matches, strict=True):
                episode_id = str(match["episode_id"]); identity = identities[episode_id]
                expected_distance = (
                    float(match["total_distance"]).hex() if method == "composite"
                    else match.get("distance_hex")
                )
                _require(match.get("symbol") == identity[0]
                         and (match.get("cutoff") is None
                              or pd.Timestamp(match["cutoff"]).isoformat() == identity[1]),
                         "D1 source match identity differs")
                _require(all((
                    row["query_case_id"] == query["case_id"],
                    row["query_symbol"] == query["symbol"], row["query_cutoff"] == query["cutoff"],
                    row["fold_id"] == query["fold_id"], row["fold_role"] == query["fold_role"],
                    row["matched_episode_id"] == episode_id,
                    (row["matched_symbol"], row["matched_cutoff"], row["quality_tier"]) == identity,
                    row["distance_hex"] == expected_distance,
                    row["match_digest"] == stable_hash(match),
                    row["effective_matches_digest"] == method_entry["effective_matches_digest"],
                    row["resolution_kind"] == method_entry["kind"],
                    row["source_artifact_sha256"] == source_sha,
                    row["source_artifact_digest"]
                        == source.get("receipt_digest", source.get("case_digest")),
                    row["matched_symbol"] != query["symbol"],
                    row["latest_eligible_ns"] == latest_eligible_ns,
                    pd.Timestamp(row["matched_cutoff"]).value <= int(row["latest_eligible_ns"]),
                )), "D2 effective link reconstruction differs")
                if row["distance_hex"] is not None:
                    distance = float.fromhex(row["distance_hex"])
                    _require(math.isfinite(distance) and distance >= 0, "D2 distance differs")
                symbols.add(row["matched_symbol"])
            _require(len(symbols) == 20, "D2 lane symbol diversity differs")
            effective_states.append({
                "query_id": query_id, "method": method,
                "matches_digest": stable_hash(matches),
            })
            position += 20
    _require(position == 314880 and len(source_cache) == 11848, "D2 coverage differs")
    _require(stable_hash(effective_states) == d1_manifest["effective_inventory_digest"],
             "D1 effective inventory digest differs")
    expected_requests = [{
        "episode_id": value, "dataset_id": "nasdaq", "symbol": identities[value][0],
        "cutoff": identities[value][1], "quality_tier": identities[value][2],
    } for value in identifiers]
    _require(requests == expected_requests, "D2 episode request deduplication differs")
    return {
        "d1_result_digest": d1_result["result_digest"],
        "d1_manifest_digest": d1_manifest["manifest_digest"],
        "d1_verification_digest": d1_verified["verification_digest"],
        "d1_effective_inventory_digest": d1_manifest["effective_inventory_digest"],
        "d2_result_digest": result["result_digest"],
        "d2_manifest_digest": manifest["manifest_digest"],
        "d2_verification_digest": verified["verification_digest"],
        "d2_raw_link_semantic_digest": link_digest,
        "d2_episode_request_semantic_digest": request_digest,
        "d1_source_artifacts": len(source_cache),
    }


def _atomic_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise B001AuthorityError(f"create-only path exists: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _report(payload: Mapping[str, Any]) -> str:
    inventory = payload["inventory"]
    gate_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in sorted(payload["gates"].items())
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>R1-B B0-01 reuse-authority audit</title></head><body>"
        "<h1>R1-B B0-01 reuse-authority producer: PASS, verification pending</h1>"
        "<p>This outcome-blind producer reconstructed the frozen reuse authorities. "
        "A separate independent verifier is still required before B0-01 is complete.</p>"
        f"<p>Result digest: <code>{escape(str(payload['result_digest']))}</code></p>"
        f"<p>Current queries/cases: {inventory['shadow_queries']:,}; current links: "
        f"{inventory['shadow_links']:,}; walk-forward identities: "
        f"{inventory['walk_forward_queries']:,}; effective D2 links: "
        f"{inventory['effective_links']:,}; packed candidates: "
        f"{inventory['packed_candidate_episodes']:,}.</p>"
        "<p>No outcome, prediction, evidence-card, Stockbee or forward-return store was "
        "opened. No scientific statistic or predictive claim was produced.</p>"
        f"<table><thead><tr><th>Gate</th><th>Status</th></tr></thead><tbody>{gate_rows}"
        "</tbody></table></body></html>\n"
    )


def _atomic_bytes(path: Path, content: bytes) -> None:
    _require(not path.exists() and not path.is_symlink(), f"create-only path exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content); handle.flush(); os.fsync(handle.fileno())


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "atomic no-replace rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise B001AuthorityError(f"create-only output exists: {destination}")
    raise B001AuthorityError(
        f"atomic no-replace publication failed: {os.strerror(error)}"
    )


def _publish(output: Path, payload: Mapping[str, Any]) -> None:
    _require(not output.exists() and not output.is_symlink(), "create-only output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = os.open(output.parent / f".{output.name}.publish.lock", os.O_RDWR | os.O_CREAT, 0o600)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _require(not output.exists() and not output.is_symlink(), "create-only output exists")
        temporary.mkdir(); _atomic_file(temporary / "RESULT.json", payload)
        _atomic_bytes(temporary / "report.html", _report(payload).encode())
        _require({path.name for path in temporary.iterdir()} == {"RESULT.json", "report.html"},
                 "publication file closure differs")
        descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        _rename_noreplace(temporary, output)
        parent = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(parent)
        finally: os.close(parent)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        fcntl.flock(lock, fcntl.LOCK_UN); os.close(lock)


def _result_state(
    prereg: Mapping[str, Any], h1: str, authority_digests: Mapping[str, Any],
    gates: Mapping[str, bool], opened_paths: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(gates and all(value is True for value in gates.values()),
             "producer gate did not pass")
    _require(len(opened_paths) == len({str(item["path"]) for item in opened_paths}),
             "opened path manifest contains duplicates")
    return {
        "schema_version": SCHEMA,
        "status": "producer_gate_pass_pending_independent_verification",
        "passed": True, "producer_gate_passed": True,
        "independent_verification_complete": False,
        "b001_complete": False, "later_stage_authorized": False,
        "preregistration_commit": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "frozen_authorities_digest": prereg["frozen_authorities_digest"],
        "authority_digests": dict(authority_digests),
        "inventory": prereg["inventory"],
        "allowed_path_contract_digest": stable_hash({
            "files": prereg["allowed_file_sha256"],
            "roots": prereg["dynamic_read_roots"],
            "shadow_cases": prereg["shadow_case_manifest_digest"],
            "d1_sources": prereg["d1_source_manifest_digest"],
            "d1_upstream_sources": prereg["d1_upstream_source_manifest_digest"],
            "d1_source_union": prereg["d1_source_union_manifest_digest"],
        }),
        "opened_path_count": len(opened_paths),
        "opened_path_manifest": list(opened_paths),
        "opened_path_manifest_digest": stable_hash(list(opened_paths)),
        "gates": dict(gates), **_claims(),
    }


def run(repository: Path) -> dict[str, Any]:
    _OPENED_PATHS.clear()
    repository = repository.resolve(); prereg = _load(repository / PREREGISTRATION)
    h1 = _validate_h1(repository, prereg); _validate_contract(repository, prereg)
    _validate_static_no_leakage(repository)
    r1a = _validate_r1a_and_shadow(repository)
    wf, queries = _validate_wf_registry(repository)
    packed, loaded = _validate_packed(repository)
    d1d2 = _validate_d1_d2(repository, queries, loaded)
    gates = {
        "clean_h0_sole_child_h1_and_runtime_bound": True,
        "frozen_authority_constant_map_bound": True,
        "allowed_file_and_dynamic_root_closure_reconstructed": True,
        "obsolete_authorities_rejected": True,
        "r1a_result_and_full_integrity_reconstructed": True,
        "all_3270_shadow_cases_and_65400_links_reconstructed": True,
        "all_3936_walk_forward_query_identities_and_manifest_reconstructed": True,
        "d1_effective_symbol_exclusion_provenance_reconstructed": True,
        "all_314880_d2_links_and_274331_requests_reconstructed": True,
        "packed_generation_content_records_and_result_reconstructed": True,
        "outcome_prediction_evidence_stockbee_forward_return_paths_excluded": True,
        "no_scientific_statistic_computed": True,
    }
    opened_paths = _opened_path_manifest(repository, prereg)
    gates["actual_opened_path_manifest_exact"] = True
    state = _result_state(
        prereg, h1, {**r1a, **wf, **packed, **d1d2}, gates, opened_paths,
    )
    payload = {
        **state, "result_digest": stable_hash(state),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _publish(repository / OUTPUT, payload); return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("preregister", "run"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = preregister(args.repository) if args.mode == "preregister" else run(args.repository)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

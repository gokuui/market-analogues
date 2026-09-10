"""Independent reconstruction of the frozen B0-01 reuse authority audit.

The frozen preregistration digest anchors the entire contract, including all
authority constants. This verifier does not import the producer, its tests,
packed-store reader, or any project hashing/validation implementation.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCHEMA = "m04r14-r1b-b001-authority-integrity-verification-v1"
BASE = Path("config/data/analogues/m04r14")
PREREG = Path("experiments/m04r/m04r14_r1b_b001_authority_audit_preregistered.json")
PREREG_DIGEST = "b3e7d6af05882a08aea4c13db674ffabb153cb228171e2118201fb464c0d7705"
PRODUCER = BASE / "r1b-b001-authority-audit-v1"
OUTPUT = BASE / "r1b-b001-authority-integrity-verification-v1"
RUNTIME = (
    "experiments/m04r/m04r14_r1b_b001_authority_verifier.py",
    "tests/test_r1b_b001_authority_verifier.py",
)
METHODS = ("composite", "price_only", "deterministic_random", "recent_return_volatility")
SOURCES = tuple(BASE / name / "cases" for name in (
    "t14-10-wf03-composite-batch-v2", "t14-10-wf03-combined-batch-v5",
    "t14-10-wf03-baseline-batch-v2",
))
D1 = BASE / "t14-10-wf03d-exclusion-repair-full-v2"
D2 = BASE / "t14-10-wf03d-cross-store-manifest-v1"
WF = BASE / "t14-10-walk-forward-query-registry-v1"
SHADOW = BASE / "nasdaq-shadow-denominator-v1"
CASES = BASE / "nasdaq-shadow-snapshot-v1/cases"
R1A = BASE / "r1a-exposure-audit-v2"
PACK = Path("config/data/analogues/poc/m04r/packed-bound-full")
CASE_OMITTED = {
    "created_at", "proposal_seconds", "amortized_proposal_seconds", "exact_seconds",
    "final_exact_seconds", "frontier_attempt_measurements", "peak_rss_mb",
    "result_digest", "checkpoint_integrity_digest",
}
LINK_COLUMNS = (
    "query_id", "query_case_id", "query_symbol", "query_cutoff", "fold_id", "fold_role",
    "method", "rank", "matched_episode_id", "matched_symbol", "matched_cutoff",
    "quality_tier", "distance_hex", "latest_eligible_ns", "match_digest",
    "effective_matches_digest", "resolution_kind", "source_artifact_path",
    "source_artifact_sha256", "source_artifact_digest",
)
REQUEST_COLUMNS = ("episode_id", "dataset_id", "symbol", "cutoff", "quality_tier")
GATES = (
    "clean_h0_sole_child_h1_and_runtime_bound", "frozen_authority_constant_map_bound",
    "allowed_file_and_dynamic_root_closure_reconstructed", "obsolete_authorities_rejected",
    "r1a_result_and_full_integrity_reconstructed",
    "all_3270_shadow_cases_and_65400_links_reconstructed",
    "all_3936_walk_forward_query_identities_and_manifest_reconstructed",
    "d1_effective_symbol_exclusion_provenance_reconstructed",
    "all_314880_d2_links_and_274331_requests_reconstructed",
    "packed_generation_content_records_and_result_reconstructed",
    "outcome_prediction_evidence_stockbee_forward_return_paths_excluded",
    "no_scientific_statistic_computed", "actual_opened_path_manifest_exact",
)
DENIED = (
    "scientific_statistics_opened", "outcomes_or_labels_accessed", "prediction_paths_accessed",
    "evidence_card_paths_accessed", "stockbee_paths_accessed", "forward_return_paths_accessed",
    "adequacy_labels_authorized", "predictive_claim_authorized", "ranking_change_authorized",
    "production_promotion_authorized",
)
PACK_DTYPE = np.dtype([
    ("episode_id", "V12"), ("cutoff_ns", "<i8"), ("symbol_id", "<u4"),
    ("quality_tier", "u1"), ("presence", "u1", (3,)), ("coarse", "<f2", (128,)),
    ("samples_48", "<f2", (19, 48)), ("stage", "<f2", (48,)),
    ("structural", "<f2", (9,)), ("error_radii", "<f4", (41,)), ("padding", "V46"),
])
OVERFLOW_DTYPE = np.dtype([
    ("episode_id", "V12"), ("cutoff_ns", "<i8"), ("symbol_id", "<u4"),
    ("quality_tier", "u1"), ("padding", "V7"),
])


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             default=str, allow_nan=False).encode()).hexdigest()


def without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def load(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    try:
        result = json.loads(path.read_bytes(), object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                VerificationError("nonfinite JSON")))
    except (OSError, ValueError) as error:
        raise VerificationError(f"invalid JSON: {path}") from error
    require(type(result) is dict, "JSON object required")
    return result


def file_sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    result = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    require(result.returncode == 0, f"git operation failed: {args[0]}")
    return result.stdout


def check_prereg(prereg: Mapping[str, Any]) -> None:
    require(prereg.get("preregistration_digest") == PREREG_DIGEST
            == digest(without(prereg, "preregistration_digest")),
            "frozen preregistration identity differs")
    require(digest(prereg["frozen_authorities"]) == prereg["frozen_authorities_digest"],
            "frozen authority constants differ")


def lineage(root: Path, prereg: Mapping[str, Any]) -> tuple[str, str]:
    require(not git(root, "status", "--porcelain", "--untracked-files=all").strip(),
            "verification requires a clean committed tree")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    h0 = prereg["implementation_commit"]
    candidates = []
    for commit in git(root, "log", "--format=%H", "--", str(PREREG)).decode().splitlines():
        if git(root, "rev-parse", f"{commit}^").decode().strip() == h0:
            candidates.append(commit)
    require(len(candidates) == 1, "H0/H1 lineage differs")
    h1 = candidates[0]
    require(git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", h1)
            .decode().splitlines() == [str(PREREG)], "H1 sole-file closure differs")
    require(git(root, "show", f"{h1}:{PREREG}") == (root / PREREG).read_bytes(),
            "committed preregistration differs")
    git(root, "merge-base", "--is-ancestor", h1, head)
    for path, expected in prereg["runtime_sha256"].items():
        require(file_sha(root / path) == expected, f"producer runtime changed: {path}")
        for commit in (h0, h1, head):
            require(sha256(git(root, "show", f"{commit}:{path}")).hexdigest() == expected,
                    f"committed producer runtime changed: {path}")
    for path in RUNTIME:
        require(git(root, "show", f"{head}:{path}") == (root / path).read_bytes(),
                f"verifier runtime is not committed: {path}")
    return head, h1


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and relative == path.as_posix(),
            "unsafe authority path")
    candidate = root / path
    require(candidate.resolve().is_relative_to(root.resolve())
            and not candidate.is_symlink() and candidate.is_file(), "authority path escaped or linked")
    require(not any(part.is_symlink() for part in candidate.parents if part != root.parent),
            "linked authority ancestor")
    return candidate


def opened_manifest(root: Path, prereg: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = dict(prereg["allowed_file_sha256"])
    expected.update(prereg["runtime_sha256"])
    manifests = prereg["dynamic_read_manifests"]
    union = {}
    for name, count, prefix, digest_key in (
        ("shadow_cases", 3270, CASES.parent, "shadow_case_manifest_digest"),
        ("d1_source_artifacts", 11848, Path("."), "d1_source_manifest_digest"),
        ("d1_upstream_source_artifacts", 11808, Path("."), "d1_upstream_source_manifest_digest"),
    ):
        rows = manifests[name]
        require(len(rows) == count and digest(rows) == prereg[digest_key], "dynamic manifest differs")
        require([r["path"] for r in rows] == sorted({r["path"] for r in rows}),
                "dynamic manifest order/uniqueness differs")
        for row in rows:
            relative = (prefix / row["path"]).as_posix()
            require(set(row) == {"path", "bytes", "sha256"}, "manifest shape differs")
            require(relative not in expected or expected[relative] == row["sha256"],
                    "conflicting source identity")
            expected[relative] = row["sha256"]
            require(safe_path(root, relative).stat().st_size == row["bytes"], "source bytes differ")
            if name != "shadow_cases":
                require(relative not in union or union[relative] == row, "source union conflict")
                union[relative] = row
    require(len(union) == 12020 and digest([union[k] for k in sorted(union)])
            == prereg["d1_source_union_manifest_digest"], "source union differs")
    require(sorted(path.relative_to(root / CASES.parent).as_posix()
                   for path in (root / CASES).glob("*.json"))
            == [r["path"] for r in manifests["shadow_cases"]], "shadow directory closure differs")
    expected[PREREG.as_posix()] = file_sha(root / PREREG)
    rows = []
    forbidden = ("outcome", "prediction", "evidence-card", "evidence_card", "stockbee",
                 "forward-return", "forward_return", "wf04", "t14-11", "t14-12")
    for relative in sorted(expected):
        require(not any(token in relative.lower() for token in forbidden), "forbidden opened authority")
        path = safe_path(root, relative)
        require(file_sha(path) == expected[relative], f"authority bytes differ: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": expected[relative]})
    require(len(rows) == 15332, "opened authority count differs")
    return rows


def sealed(root: Path, folder: Path, expected_names: set[str] | None = None) -> dict[str, Any]:
    seal = load(root / folder / "SEALED.json")
    actual = sorted(path.name for path in (root / folder).iterdir())
    names = [row["path"] for row in seal["files"]]
    require(len(names) == len(set(names)) and all(Path(n).name == n for n in names), "seal paths differ")
    require(actual == sorted(["SEALED.json", *names]), "sealed directory closure differs")
    if expected_names is not None:
        require(set(actual) == expected_names, "fixed sealed directory differs")
    rows = [{"path": name, "bytes": (root / folder / name).stat().st_size,
             "sha256": file_sha(root / folder / name)} for name in sorted(names)]
    require(rows == seal["files"] and digest(rows) == seal["manifest_digest"], "seal manifest differs")
    require(digest(without(seal, "created_at", "seal_digest")) == seal["seal_digest"], "seal digest differs")
    return seal


def anchored(root: Path, path: Path, field: str, expected: str, *omitted: str) -> dict[str, Any]:
    value = load(root / path)
    require(value.get(field) == expected == digest(without(value, field, *omitted)),
            f"frozen terminal identity differs: {path}")
    return value


def fields(value: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    require(all(key in value and type(value[key]) is type(item) and value[key] == item
                for key, item in expected.items()), f"{label} field binding differs")


def receipt_gates(value: Mapping[str, Any], label: str) -> None:
    gates = value.get("gates")
    require(isinstance(gates, dict) and bool(gates) and all(flag is True for flag in gates.values()),
            f"{label} receipt gates differ")


def shadow_authorities(root: Path, frozen: Mapping[str, Any]) -> dict[str, Any]:
    a, s = frozen["r1a"], frozen["shadow"]
    ap = anchored(root, Path("experiments/m04r/m04r14_r1a_exposure_audit_v2_preregistered.json"),
                  "preregistration_digest", a["preregistration_digest"])
    ar = anchored(root, R1A / "RESULT.json", "result_digest", a["result_digest"], "elapsed_seconds")
    first = anchored(root, BASE / "r1a-exposure-audit-v2-verification/VERIFIED.json",
                     "verification_digest", a["first_verification_digest"], "created_at")
    av = anchored(root, BASE / "r1a-exposure-audit-v2-integrity-verification-v1/VERIFIED.json",
                  "verification_digest", a["full_integrity_digest"], "created_at")
    registry = anchored(root, SHADOW / "query-registry.json", "registry_digest", s["registry_digest"])
    seal = sealed(root, SHADOW, {"SEALED.json", "denominator.html", "denominator.parquet",
                               "query-registry.json", "query-registry.parquet"})
    dv = anchored(root, BASE / "nasdaq-shadow-denominator-v1-verification/VERIFIED.json",
                  "result_digest", s["denominator_verification_digest"], "created_at")
    sv = anchored(root, BASE / "nasdaq-shadow-interrupted-verification-v1/VERIFIED.json",
                  "result_digest", s["semantic_verification_digest"], "created_at")
    denied = {key: False for key in ("adequacy_labels_authorized", "predictive_claim_authorized",
                                    "production_promotion_authorized", "real_forward_outcomes_accessed")}
    fields(ap["inputs"], {"queries": 3270, "retrieved_links": 65400,
                         "packed_candidate_episodes": 3786156, "packed_candidate_symbols": 11584}, "R1-A inputs")
    fields(ap["execution"], {"workers": 12, "null_replicates": 512}, "R1-A execution")
    fields(ap["claims"], {"development_diagnostic_only": True, **denied}, "R1-A prereg claims")
    fields(ar, {"passed": True, "preregistration_digest": ap["preregistration_digest"], **denied}, "R1-A result")
    require(ar["inventory"] == {"candidate_episodes": 3786156, "candidate_symbols": 11584,
            "queries": 3270, "retrieved_links": 65400, "unique_latest_eligible_cutoffs": 106}, "R1-A inventory differs")
    fields(first, {"passed": True, "verified_result_digest": ar["result_digest"],
                   "verified_queries": 3270, "verified_links": 65400, "verified_null_replicates": 512,
                   "real_forward_outcomes_accessed": False, "predictive_claim_authorized": False,
                   "production_promotion_authorized": False}, "R1-A first verifier")
    fields(av, {"passed": True, "verified_result_digest": ar["result_digest"],
                "verified_preregistration_digest": ap["preregistration_digest"],
                "verified_first_verification_digest": first["verification_digest"],
                "verified_queries": 3270, "verified_null_replicates": 512, **denied}, "R1-A full verifier")
    receipt_gates(first, "R1-A first"); receipt_gates(av, "R1-A full")
    fields(registry, {"passed": True, "scheduled_queries": 3270, "real_forward_outcomes_accessed": False}, "shadow registry")
    fields(dv, {"passed": True, "registry_digest": registry["registry_digest"], "seal_digest": seal["seal_digest"],
                "scheduled_queries": 3270, "real_forward_outcomes_accessed": False}, "shadow denominator verifier")
    fields(sv, {"semantic_passed": True, "overall_t14_08_passed": False, "verified_cases": 3270,
                "verified_matches": 65400, "registry_digest": registry["registry_digest"],
                "case_manifest_digest": s["case_manifest_digest"], "real_forward_outcomes_accessed": False}, "shadow semantic verifier")
    require(ar["passed"] is av["passed"] is first["passed"] is registry["passed"] is dv["passed"] is True,
            "R1-A/shadow terminal gate differs")
    require(sv["semantic_passed"] is True and sv["overall_t14_08_passed"] is False,
            "interrupted qualification incorrectly promoted")
    require(dv["seal_digest"] == seal["seal_digest"] and av["verified_result_digest"] == ar["result_digest"]
            and av["verified_preregistration_digest"] == ap["preregistration_digest"]
            and av["verified_first_verification_digest"] == first["verification_digest"],
            "R1-A/shadow cross-binding differs")
    require({p.name for p in (root / R1A).iterdir()}
            == {"RESULT.json", "NULL_REPLICATES.json", "QUERY_DIAGNOSTICS.json", "report.html"},
            "R1-A artifact closure differs")
    for name, key in (("NULL_REPLICATES.json", "null_replicates_sha256"),
                      ("QUERY_DIAGNOSTICS.json", "query_diagnostics_sha256")):
        require(file_sha(root / R1A / name) == ar["artifacts"][key], "R1-A sidecar differs")
    queries = {row["episode_id"]: row for row in registry["cases_data"]}
    require(len(queries) == len(registry["cases_data"]) == 3270, "shadow query uniqueness differs")
    identities, manifest, seen = [], [], set()
    for path in sorted((root / CASES).glob("*.json")):
        case = load(path)
        qid = case["query_episode_id"]
        require(qid in queries and qid not in seen, "shadow query identity differs")
        seen.add(qid)
        query = queries[qid]
        require(case["result_digest"] == digest(without(case, *CASE_OMITTED))
                and case["checkpoint_integrity_digest"]
                == digest(without(case, "created_at", "checkpoint_integrity_digest")), "shadow case seal differs")
        for left, right in (("registry_case_id", "case_id"), ("query_symbol", "symbol"),
                            ("query_cutoff", "cutoff"), ("latest_eligible_cutoff", "latest_eligible_cutoff")):
            require(case[left] == query[right], "shadow query binding differs")
        require(case["gate_passed"] is True and case["real_forward_outcomes_accessed"] is False
                and case["registry_digest"] == registry["registry_digest"]
                and case["generation_id"] == frozen["packed"]["generation_id"], "shadow authority differs")
        matches = case["matches"]
        require(len(matches) == len({m["episode_id"] for m in matches}) == 20, "shadow match count differs")
        order = [(float(m["total_distance"]), m["episode_id"]) for m in matches]
        require(order == sorted(order) and all(math.isfinite(d) and d >= 0 for d, _ in order),
                "shadow match order differs")
        require(max(sum(m["symbol"] == symbol for m in matches)
                    for symbol in {m["symbol"] for m in matches}) <= 3, "shadow symbol cap differs")
        for rank, match in enumerate(matches, 1):
            require(pd.Timestamp(match["cutoff"]) <= pd.Timestamp(case["latest_eligible_cutoff"]),
                    "shadow causal cutoff differs")
            identities.append({"query_episode_id": qid, "episode_id": match["episode_id"], "rank": rank})
        manifest.append({"path": path.relative_to(root / CASES.parent).as_posix(),
                         "bytes": path.stat().st_size, "sha256": file_sha(path)})
    require(seen == set(queries) and len(identities) == 65400
            and digest(manifest) == s["case_manifest_digest"]
            and digest(identities) == s["retrieval_identity_digest"], "shadow reconstruction differs")
    require(ar["inputs"] == {
        "case_manifest_digest": digest(manifest), "packed_generation_id": frozen["packed"]["generation_id"],
        "packed_provenance_digest": frozen["packed"]["provenance_digest"],
        "registry_digest": s["registry_digest"], "retrieval_identity_digest": digest(identities),
        "semantic_verification_digest": s["semantic_verification_digest"],
    }, "R1-A reconstructed inputs differ")
    return {
        "r1a_preregistration_digest": a["preregistration_digest"], "r1a_result_digest": a["result_digest"],
        "r1a_full_integrity_digest": a["full_integrity_digest"],
        **{f"shadow_{key}": s[key] for key in (
            "registry_digest", "denominator_verification_digest", "semantic_verification_digest",
            "case_manifest_digest", "retrieval_identity_digest")},
    }


def walk_forward(root: Path, frozen: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    f = frozen["walk_forward"]
    registry = anchored(root, WF / "walk-forward-query-registry.json", "registry_digest", f["registry_digest"])
    seal = sealed(root, WF)
    verified = anchored(root, BASE / "t14-10-walk-forward-query-registry-v1-verification/VERIFIED.json",
                        "result_digest", f["verification_digest"], "elapsed_seconds")
    expected = {"passed": True, "queries": 3936, "scored_queries": 3360,
                "historical_walk_forward_query_outcomes_opened": False, "final_period_result_opened": False}
    fields(registry, expected, "WF registry")
    fields(verified, {**expected, "registry_digest": registry["registry_digest"], "manifest_closed": True}, "WF verifier")
    queries = registry["queries_data"]
    require(len(queries) == len({q["episode_id"] for q in queries}) == len({q["case_id"] for q in queries})
            == 3936 and registry["scored_queries"] == 3360 and digest(queries) == f["query_digest"],
            "walk-forward identities differ")
    require(seal["seal_digest"] == f["seal_digest"] and seal["manifest_digest"] == f["manifest_digest"]
            and verified["registry_sha256"] == file_sha(root / WF / "walk-forward-query-registry.json")
            and verified["registry_seal_sha256"] == file_sha(root / WF / "SEALED.json")
            and verified["passed"] is registry["passed"] is True, "walk-forward authority differs")
    return {f"wf_{key}": f[key] for key in (
        "registry_digest", "query_digest", "manifest_digest", "verification_digest")}, queries


def check_records(records: np.ndarray, symbol_count: int, *, overflow: bool) -> None:
    require(np.all(records["symbol_id"] < symbol_count)
            and np.all(np.isin(records["quality_tier"], [1, 2])), "packed metadata differs")
    symbols, cutoffs = records["symbol_id"], records["cutoff_ns"]
    require(np.all((symbols[1:] > symbols[:-1]) |
                   ((symbols[1:] == symbols[:-1]) & (cutoffs[1:] > cutoffs[:-1]))),
            "packed order differs")
    if not overflow:
        for start in range(0, len(records), 8192):
            chunk = records[start:start + 8192]
            require(all(np.isfinite(chunk[field]).all() for field in (
                "coarse", "samples_48", "stage", "structural", "error_radii"))
                and np.all(chunk["error_radii"] >= 0), "packed numerical values differ")


def packed(root: Path, frozen: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    f = frozen["packed"]
    folder = PACK / "store/generations" / f["generation_id"]
    manifest = anchored(root, folder / "manifest.json", "manifest_digest", f["generation_id"])
    result = anchored(root, PACK / "packed-bound-full.json", "result_digest", f["result_digest"],
        "created_at", "build_seconds", "generation_seconds", "validation_seconds", "peak_rss_mb",
        "scan_peak_rss_mb", "validation_peak_rss_mb", "inherited_scan_ru_maxrss_mb")
    fields(result, {"gate_passed": True, "poc_passed": True, "generation_id": f["generation_id"],
                    "real_forward_outcomes_accessed": False, "eligible_rows": 3786156, "rows": 3786121,
                    "overflow_rows": 35, "pack_bytes": f["rows_bytes"] + f["overflow_bytes"],
                    "pack_contract_digest": f["pack_contract_digest"], "authority_row_accounting_passed": True}, "packed result")
    require(len(result["full_authority_row_counts"]) == 12
            and all(row["row_accounting_matches"] is True for row in result["full_authority_row_counts"]), "packed row accounting differs")
    fields(manifest, {"eligible_row_count": 3786156, "row_count": 3786121, "overflow_count": 35,
                      "rows_bytes": f["rows_bytes"], "overflow_bytes": f["overflow_bytes"],
                      "rows_sha256": f["rows_sha256"], "overflow_sha256": f["overflow_sha256"],
                      "pack_contract_digest": f["pack_contract_digest"],
                      "quantized_bound_contract_digest": f["quantized_bound_contract_digest"],
                      "real_forward_outcomes_accessed": False, "rows_file": "bound-rows.bin",
                      "overflow_file": "overflow-exact-fallback.bin"}, "packed manifest")
    require(digest(manifest["provenance"]) == f["provenance_digest"], "packed provenance content differs")
    require({p.name for p in (root / folder).iterdir()}
            == {"manifest.json", "bound-rows.bin", "overflow-exact-fallback.bin"}, "packed closure differs")
    require(manifest["row_bytes"] == PACK_DTYPE.itemsize == 2432
            and manifest["overflow_row_bytes"] == OVERFLOW_DTYPE.itemsize == 32
            and manifest["row_count"] == 3786121 and manifest["overflow_count"] == 35
            and len(manifest["symbols"]) == len(set(manifest["symbols"])) == 11584
            and manifest["provenance_digest"] == f["provenance_digest"], "packed contract differs")
    require(result["gate_passed"] is result["poc_passed"] is True
            and result["eligible_rows"] == 3786156, "packed terminal differs")
    files = {}
    for label, name in (("manifest", "manifest.json"), ("rows", "bound-rows.bin"),
                        ("overflow", "overflow-exact-fallback.bin")):
        files[label] = {"bytes": (root / folder / name).stat().st_size,
                        "sha256": file_sha(root / folder / name)}
        require(files[label]["sha256"] == f[f"{label}_sha256"], "packed bytes differ")
    content = {
        "schema_version": "m04r-resident-packed-store-content-v1",
        "generation_id": f["generation_id"], "manifest_digest": f["generation_id"],
        "provenance_digest": f["provenance_digest"], "pack_contract_digest": f["pack_contract_digest"],
        "quantized_bound_contract_digest": f["quantized_bound_contract_digest"],
        "physical_generation_bytes": sum(v["bytes"] for v in files.values()),
        "source_files": files, "mirror_files": files,
    }
    require(digest(content) == f["content_digest"], "packed content identity differs")
    main = np.memmap(root / folder / "bound-rows.bin", dtype=PACK_DTYPE, mode="r")
    extra = np.memmap(root / folder / "overflow-exact-fallback.bin", dtype=OVERFLOW_DTYPE, mode="r")
    check_records(main, 11584, overflow=False); check_records(extra, 11584, overflow=True)
    columns = {key: np.concatenate((main[key], extra[key])) for key in (
        "episode_id", "cutoff_ns", "symbol_id", "quality_tier")}
    order = np.argsort(columns["episode_id"], kind="stable")
    columns = {key: value[order] for key, value in columns.items()}
    require(np.all(columns["episode_id"][1:] != columns["episode_id"][:-1]), "packed duplicate episodes")
    columns["symbols"] = manifest["symbols"]
    return {
        "packed_result_digest": f["result_digest"], "packed_generation_id": f["generation_id"],
        "packed_provenance_digest": f["provenance_digest"], "packed_candidate_episodes": 3786156,
        "packed_content_digest": digest(content),
    }, columns


def packed_lookup(columns: Mapping[str, Any], identifiers: Sequence[str]) -> dict[str, tuple[str, str, str]]:
    requested = np.asarray([np.void(bytes.fromhex(identifier)) for identifier in identifiers], dtype="V12")
    positions = np.searchsorted(columns["episode_id"], requested)
    require(np.all(positions < len(columns["episode_id"])), "requested episode absent from pack")
    require(np.array_equal(columns["episode_id"][positions], requested), "requested episode absent from pack")
    return {identifier: (
        columns["symbols"][int(columns["symbol_id"][position])],
        pd.Timestamp(int(columns["cutoff_ns"][position]), unit="ns").isoformat(),
        {1: "A", 2: "B"}[int(columns["quality_tier"][position])],
    ) for identifier, position in zip(identifiers, positions, strict=True)}


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def table_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    result = sha256(f"canonical-json-record-chunks-v1\0{len(rows)}\0".encode())
    for begin in range(0, len(rows), 16384):
        encoded = json.dumps(plain(rows[begin:begin + 16384]), sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()
        result.update(len(encoded).to_bytes(8, "big")); result.update(encoded)
    return result.hexdigest()


def extract_matches(source: Mapping[str, Any], method: str, repaired: bool) -> list[dict[str, Any]]:
    require(method in METHODS, "unknown retrieval method")
    if repaired:
        matches = source.get("corrected_matches")
    else:
        route = {"composite": source.get("retrieval", {}).get("matches"),
                 "price_only": source.get("matches"), "deterministic_random": source.get("random_neighbors"),
                 "recent_return_volatility": source.get("rank_neighbors")}
        matches = route[method]
    require(isinstance(matches, list) and len(matches) == 20, "source match inventory differs")
    return matches


def link_record(query: Mapping[str, Any], method_entry: Mapping[str, Any], match: Mapping[str, Any],
                rank: int, identity: tuple[str, str, str], source_path: str, source_sha: str,
                source_digest: str, latest_ns: int) -> dict[str, Any]:
    method = method_entry["method"]
    distance = float(match["total_distance"]).hex() if method == "composite" else match.get("distance_hex")
    require(match["symbol"] == identity[0] and identity[0] != query["symbol"], "source symbol/exclusion differs")
    require(match.get("cutoff") is None or pd.Timestamp(match["cutoff"]).isoformat() == identity[1],
            "source cutoff differs")
    require(pd.Timestamp(identity[1]).value <= latest_ns, "source causal cutoff differs")
    if distance is not None:
        require(math.isfinite(float.fromhex(distance)) and float.fromhex(distance) >= 0, "source distance differs")
    return dict(zip(LINK_COLUMNS, (
        query["episode_id"], query["case_id"], query["symbol"], query["cutoff"], query["fold_id"],
        query["fold_role"], method, rank, match["episode_id"], *identity, distance, latest_ns,
        digest(match), method_entry["effective_matches_digest"], method_entry["kind"],
        source_path, source_sha, source_digest,
    ), strict=True))


def d1d2(root: Path, frozen: Mapping[str, Any], queries: Sequence[Mapping[str, Any]],
         columns: Mapping[str, Any]) -> dict[str, Any]:
    f1, f2 = frozen["d1_v2"], frozen["d2"]
    a = anchored(root, D1 / "RESULT.json", "result_digest", f1["result_digest"])
    am = anchored(root, D1 / "MANIFEST.json", "manifest_digest", f1["manifest_digest"])
    av = anchored(root, BASE / "t14-10-wf03d-exclusion-repair-full-v2-verification/VERIFIED.json",
                  "verification_digest", f1["verification_digest"])
    contract = anchored(root, D2 / "CONTRACT.json", "preregistration_digest", f2["preregistration_digest"])
    b = anchored(root, D2 / "RESULT.json", "result_digest", f2["result_digest"])
    bm = anchored(root, D2 / "MANIFEST.json", "manifest_digest", f2["manifest_digest"])
    bv = anchored(root, BASE / "t14-10-wf03d-cross-store-manifest-v1-verification/VERIFIED.json",
                  "verification_digest", f2["verification_digest"])
    denied = {key: False for key in ("outcomes_or_labels_used", "historical_walk_forward_query_outcomes_opened",
                                    "final_period_result_opened")}
    fields(a, {"passed": True, "query_count": 3936, "effective_neighbour_links": 314880, "methods_per_query": 4,
               "repair_receipts": 212, "manifest_digest": am["manifest_digest"],
               "all_effective_matches_exclude_query_symbol": True, **denied}, "D1 result")
    fields(am, {"query_count": 3936, "method_links": 15744, "effective_neighbour_links": 314880,
                "effective_inventory_digest": f1["effective_inventory_digest"], **denied}, "D1 manifest")
    fields(av, {"passed": True, "producer_result_digest": a["result_digest"], "manifest_digest": am["manifest_digest"],
                "query_count": 3936, "method_lanes": 15744, "effective_neighbour_links": 314880,
                "effective_inventory_digest": am["effective_inventory_digest"],
                "cross_store_manifest_construction_authorized": True, "production_promotion_authorized": False,
                **denied}, "D1 verifier")
    receipt_gates(av, "D1"); receipt_gates(bv, "D2")
    counts = {"query_count": 3936, "method_lane_count": 15744, "link_count": 314880, "unique_episode_count": 274331}
    fields(contract, {**counts, "methods": list(METHODS), **denied,
                      "production_promotion_authorized": False}, "D2 contract")
    fields(b, {**counts, "passed": True, "manifest_digest": bm["manifest_digest"],
               "preregistration_digest": contract["preregistration_digest"], **denied}, "D2 result")
    fields(bm, {**counts, "preregistration_digest": contract["preregistration_digest"],
                "file_inventory": ["CONTRACT.json", "raw_links.parquet", "episode_requests.parquet"]}, "D2 manifest")
    fields(bv, {**counts, "passed": True, "producer_result_digest": b["result_digest"],
                "preregistration_digest": contract["preregistration_digest"], "manifest_digest": bm["manifest_digest"],
                "source_artifact_count": 11848, "production_promotion_authorized": False, **denied}, "D2 verifier")
    require(a["passed"] is av["passed"] is b["passed"] is bv["passed"] is True,
            "D1/D2 terminal gate differs")
    require(a["manifest_sha256"] == file_sha(root / D1 / "MANIFEST.json")
            and av["producer_result_sha256"] == file_sha(root / D1 / "RESULT.json")
            and b["manifest_sha256"] == file_sha(root / D2 / "MANIFEST.json"), "D1/D2 file binding differs")
    require({p.name for p in (root / D2).iterdir()}
            == {"CONTRACT.json", "RESULT.json", "MANIFEST.json", "raw_links.parquet", "episode_requests.parquet"},
            "D2 directory closure differs")
    frames = [pd.read_parquet(root / D2 / name) for name in ("raw_links.parquet", "episode_requests.parquet")]
    require(tuple(frames[0].columns) == LINK_COLUMNS and tuple(frames[1].columns) == REQUEST_COLUMNS,
            "D2 table schema differs")
    links, requests = [plain(frame.to_dict("records")) for frame in frames]
    require(len(links) == 314880 and len(requests) == 274331, "D2 table counts differ")
    ld, rd = table_digest(links), table_digest(requests)
    require(ld == f2["raw_link_digest"] and rd == f2["request_digest"], "D2 semantic table differs")
    for value, label in ((contract, "D2 contract"), (b, "D2 result"), (bv, "D2 verifier")):
        fields(value, {"raw_link_semantic_digest": ld, "episode_request_semantic_digest": rd}, label)
    for name, semantic in (("raw_links.parquet", ld), ("episode_requests.parquet", rd)):
        entries = [entry for entry in bm["tables"] if entry["path"] == name]
        require(len(entries) == 1 and entries[0]["semantic_digest"] == semantic
                and entries[0]["sha256"] == file_sha(root / D2 / name)
                and entries[0]["bytes"] == (root / D2 / name).stat().st_size, "D2 table manifest differs")
    expected_inputs = {
        "repair_result_digest": a["result_digest"], "repair_result_sha256": file_sha(root / D1 / "RESULT.json"),
        "repair_manifest_digest": am["manifest_digest"], "repair_manifest_sha256": file_sha(root / D1 / "MANIFEST.json"),
        "repair_verification_digest": av["verification_digest"],
        "repair_verification_sha256": file_sha(root / BASE / "t14-10-wf03d-exclusion-repair-full-v2-verification/VERIFIED.json"),
        "repair_effective_inventory_digest": f1["effective_inventory_digest"],
        "registry_sha256": file_sha(root / WF / "walk-forward-query-registry.json"),
        "resident_content_digest": frozen["packed"]["content_digest"],
        "packed_generation_id": frozen["packed"]["generation_id"],
        "packed_provenance_digest": frozen["packed"]["provenance_digest"],
    }
    require(contract["inputs"] == expected_inputs, "D2 reconstructed authority inputs differ")
    ids = sorted({row["matched_episode_id"] for row in links})
    identities = packed_lookup(columns, ids)
    source_cache, effective, cursor = {}, [], 0
    require(len(am["queries"]) == len(queries) == 3936, "D1 query count differs")
    for query, entry in zip(queries, am["queries"], strict=True):
        require([entry[k] for k in ("query_id", "case_id", "symbol", "cutoff")]
                == [query[k] for k in ("episode_id", "case_id", "symbol", "cutoff")], "D1 query binding differs")
        qid = query["episode_id"]
        original = {key: folder / f"{qid}.json" for key, folder in zip(
            ("composite", "price_only", "baselines"), SOURCES, strict=True)}
        original_sha = {key: file_sha(root / path) for key, path in original.items()}
        baseline = load(root / original["baselines"])
        require(digest(without(baseline, "case_digest")) == baseline["case_digest"], "baseline seal differs")
        require([m["method"] for m in entry["methods"]] == list(METHODS), "D1 method order differs")
        for lane in entry["methods"]:
            method = lane["method"]
            repaired = lane["kind"] == "top21_drop_query_symbol"
            relative = (D1 / lane["repair_receipt"] if repaired else original[
                method if method in original else "baselines"]).as_posix()
            require(lane["source_case_sha256"] == original_sha, "D1 upstream identity differs")
            if relative not in source_cache:
                source = load(safe_path(root, relative))
                key = "receipt_digest" if repaired else "case_digest"
                require(source[key] == digest(without(source, key)), "source semantic seal differs")
                source_cache[relative] = (source, file_sha(root / relative), source[key])
            source, source_hash, source_digest = source_cache[relative]
            require([source[k] for k in ("query_id", "case_id", "symbol", "cutoff")]
                    == [query[k] for k in ("episode_id", "case_id", "symbol", "cutoff")], "source query differs")
            require(all(source[k] is False for k in (
                "outcomes_or_labels_used", "historical_walk_forward_query_outcomes_opened", "final_period_result_opened")),
                "source claim boundary differs")
            if repaired:
                require(source_hash == lane["repair_receipt_sha256"] and source_digest == lane["repair_receipt_digest"]
                        and source["source_case_sha256"] == original_sha and source["method"] == method,
                        "repair provenance differs")
                fields(source, {"production_promotion_authorized": False}, "repair claims")
            matches = extract_matches(source, method, repaired)
            require(digest(matches) == lane["effective_matches_digest"], "effective matches differ")
            rebuilt = [link_record(query, lane, match, rank, identities[match["episode_id"]],
                                   relative, source_hash, source_digest, int(baseline["latest_eligible_ns"]))
                       for rank, match in enumerate(matches, 1)]
            require(len({row["matched_symbol"] for row in rebuilt}) == 20, "lane symbol diversity differs")
            require(rebuilt == links[cursor:cursor + 20], "independent D2 link reconstruction differs")
            cursor += 20
            effective.append({"query_id": qid, "method": method, "matches_digest": digest(matches)})
    require(cursor == 314880 and len(source_cache) == 11848
            and digest(effective) == f1["effective_inventory_digest"], "effective inventory reconstruction differs")
    rebuilt_requests = [dict(zip(REQUEST_COLUMNS, (identifier, "nasdaq", *identities[identifier]), strict=True))
                        for identifier in ids]
    require(rebuilt_requests == requests, "request deduplication reconstruction differs")
    return {
        **{f"d1_{key}": f1[key] for key in ("result_digest", "manifest_digest", "verification_digest", "effective_inventory_digest")},
        **{f"d2_{key}": f2[key] for key in ("result_digest", "manifest_digest", "verification_digest")},
        "d2_raw_link_semantic_digest": ld, "d2_episode_request_semantic_digest": rd,
        "d1_source_artifacts": len(source_cache),
    }


def result_state(prereg: Mapping[str, Any], h1: str, authorities: Mapping[str, Any],
                 opened: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contract = {"files": prereg["allowed_file_sha256"], "roots": prereg["dynamic_read_roots"],
                "shadow_cases": prereg["shadow_case_manifest_digest"],
                "d1_sources": prereg["d1_source_manifest_digest"],
                "d1_upstream_sources": prereg["d1_upstream_source_manifest_digest"],
                "d1_source_union": prereg["d1_source_union_manifest_digest"]}
    return {
        "schema_version": "m04r14-r1b-b001-authority-audit-v1",
        "status": "producer_gate_pass_pending_independent_verification", "passed": True,
        "producer_gate_passed": True, "independent_verification_complete": False,
        "b001_complete": False, "later_stage_authorized": False, "preregistration_commit": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "frozen_authorities_digest": prereg["frozen_authorities_digest"],
        "authority_digests": dict(authorities), "inventory": prereg["inventory"],
        "allowed_path_contract_digest": digest(contract), "opened_path_count": len(opened),
        "opened_path_manifest": list(opened), "opened_path_manifest_digest": digest(list(opened)),
        "gates": dict.fromkeys(GATES, True), "reuse_authority_only": True, **dict.fromkeys(DENIED, False),
    }


def report(payload: Mapping[str, Any]) -> str:
    inv = payload["inventory"]
    rows = "".join(f"<tr><td>{escape(key)}</td><td>{'PASS' if value else 'FAIL'}</td></tr>"
                   for key, value in sorted(payload["gates"].items()))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<title>R1-B B0-01 reuse-authority audit</title></head><body>'
        '<h1>R1-B B0-01 reuse-authority producer: PASS, verification pending</h1>'
        '<p>This outcome-blind producer reconstructed the frozen reuse authorities. '
        'A separate independent verifier is still required before B0-01 is complete.</p>'
        f'<p>Result digest: <code>{escape(str(payload["result_digest"]))}</code></p>'
        f'<p>Current queries/cases: {inv["shadow_queries"]:,}; current links: '
        f'{inv["shadow_links"]:,}; walk-forward identities: {inv["walk_forward_queries"]:,}; effective D2 links: '
        f'{inv["effective_links"]:,}; packed candidates: {inv["packed_candidate_episodes"]:,}.</p>'
        '<p>No outcome, prediction, evidence-card, Stockbee or forward-return store was '
        'opened. No scientific statistic or predictive claim was produced.</p>'
        f'<table><thead><tr><th>Gate</th><th>Status</th></tr></thead><tbody>{rows}'
        '</tbody></table></body></html>\n'
    )


def compare_result(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    require(set(actual) == set(expected) | {"created_at", "result_digest"}, "producer result closure differs")
    require(without(actual, "created_at", "result_digest") == expected
            and actual["result_digest"] == digest(expected), "producer result reconstruction differs")
    try:
        timestamp = datetime.fromisoformat(actual["created_at"])
    except (TypeError, ValueError) as error:
        raise VerificationError("producer timestamp invalid") from error
    require(timestamp.tzinfo is not None, "producer timestamp lacks timezone")


def compare_report(path: Path, actual: Mapping[str, Any]) -> None:
    require(path.is_file() and not path.is_symlink(), "regular producer report required")
    require(path.read_bytes() == report(actual).encode(), "producer HTML report differs")


def static_boundary(root: Path, prereg: Mapping[str, Any]) -> None:
    forbidden = ("outcome", "prediction", "evidence_card", "stockbee", "forward_return")
    for relative in ("experiments/m04r/m04r14_r1b_b001_authority_audit.py", RUNTIME[0]):
        tree = ast.parse((root / relative).read_text())
        for node in ast.walk(tree):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            require(not any(token in module.lower() for module in modules for token in forbidden),
                    "forbidden scientific import")
            if relative == RUNTIME[0]:
                require(not any(module.startswith(("market_analogues", "experiments", "tests"))
                                for module in modules), "verifier implementation independence differs")


def publish(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "verification receipt already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    require(not temporary.exists(), "temporary verification receipt exists")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise VerificationError("verification receipt already exists") from error
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    require(not (root / OUTPUT).exists(), "create-only verifier output already exists")
    prereg = load(root / PREREG); check_prereg(prereg)
    head, h1 = lineage(root, prereg)
    static_boundary(root, prereg)
    opened = opened_manifest(root, prereg)
    frozen = prereg["frozen_authorities"]
    authorities = shadow_authorities(root, frozen)
    wf, queries = walk_forward(root, frozen); authorities.update(wf)
    pack, columns = packed(root, frozen); authorities.update(pack)
    authorities.update(d1d2(root, frozen, queries, columns))
    expected = result_state(prereg, h1, authorities, opened)
    require({p.name for p in (root / PRODUCER).iterdir()} == {"RESULT.json", "report.html"},
            "producer publication closure differs")
    actual = load(root / PRODUCER / "RESULT.json"); compare_result(actual, expected)
    compare_report(root / PRODUCER / "report.html", actual)
    require(opened_manifest(root, prereg) == opened, "authority changed during verification")
    require(lineage(root, prereg) == (head, h1), "runtime changed during verification")
    state = {
        "schema_version": SCHEMA, "passed": True, "status": "verified",
        "verified_result_digest": actual["result_digest"], "verified_result_sha256": file_sha(root / PRODUCER / "RESULT.json"),
        "verified_report_sha256": file_sha(root / PRODUCER / "report.html"),
        "verified_preregistration_digest": PREREG_DIGEST, "preregistration_commit": h1,
        "verifier_commit": head, "verifier_runtime_sha256": {p: file_sha(root / p) for p in RUNTIME},
        "inventory": prereg["inventory"], "authority_digests": authorities,
        "opened_path_manifest_digest": digest(opened), "opened_path_count": len(opened),
        "gates": {**dict.fromkeys(GATES, True), "independent_result_and_report_reconstruction": True,
                  "independent_packed_binary_decoding": True, "committed_verifier_runtime": True},
        "b001_complete": True, "independent_verification_complete": True,
        "joint_b005_b2_contract_freeze_may_proceed": True,
        "b005_scientific_execution_authorized": False, "b2_scientific_execution_authorized": False,
        "reuse_authority_only": True, **dict.fromkeys(DENIED, False),
    }
    payload = {**state, "verification_digest": digest(state), "created_at": datetime.now(timezone.utc).isoformat()}
    publish(root / OUTPUT / "VERIFIED.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = run(args.repository)
    print(json.dumps({"passed": result["passed"], "verification_digest": result["verification_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

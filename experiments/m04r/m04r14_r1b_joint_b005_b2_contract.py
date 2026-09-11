"""Joint H0/H1 boundary for B0-05 and B2; no scientific calculation here.

Preregistration reads verified identity/support metadata only.  Both producers and
the geometry builder must execute at the *same* sole-child H1.  Later independent
verifiers have separate committed runtimes and may not alter this contract.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from market_analogues.adapters import DirectorySource, source_from_spec
from market_analogues.config import load_config_bytes
from market_analogues.types import stable_hash


SCHEMA = "m04r14-r1b-joint-b005-b2-preregistration-v1"
PREREGISTRATION = Path("experiments/m04r/m04r14_r1b_joint_b005_b2_preregistered.json")
BASE = Path("config/data/analogues/m04r14")
B001_RESULT = BASE / "r1b-b001-authority-audit-v1/RESULT.json"
B001_VERIFIED = BASE / "r1b-b001-authority-integrity-verification-v1/VERIFIED.json"
R1A_PREREG = Path("experiments/m04r/m04r14_r1a_exposure_audit_v2_preregistered.json")
V3_PREREG = Path("experiments/m04r/m04r14_r1b_balanced_v3_preregistered.json")
V2_RESULT = BASE / "r1b-support-pilot-v2/RESULT.json"
V2_VERIFIED = BASE / "r1b-support-pilot-v2-integrity-verification-v1/VERIFIED.json"
V2_CELLS = BASE / "r1b-support-pilot-v2/QUERY_CELLS.json"
V2_COHORT = BASE / "r1b-support-pilot-v2/COHORT_SUPPORT.json"
PARTITION_RESULT = BASE / "r1b-balanced-partition-v3/RESULT.json"
PARTITIONS = BASE / "r1b-balanced-partition-v3/PARTITIONS.json"
TRANSFORM = BASE / "r1b-balanced-partition-v3/TRANSFORM.json"
SUPPORT_RESULT = BASE / "r1b-balanced-support-v3/RESULT.json"
SUPPORT = BASE / "r1b-balanced-support-v3/SUPPORT.json"
SUPPORT_VERIFIED = BASE / "r1b-balanced-support-v3-integrity-verification-v1/VERIFIED.json"
REGISTRY = BASE / "nasdaq-shadow-denominator-v1/query-registry.json"
DATASET_CONFIG = Path("config/datasets.example.yaml")
CASES = BASE / "nasdaq-shadow-snapshot-v1/cases"
AUTHORITY_FILES = (
    B001_RESULT, B001_VERIFIED, R1A_PREREG, V3_PREREG, V2_RESULT,
    V2_VERIFIED, V2_CELLS, V2_COHORT, PARTITION_RESULT, PARTITIONS,
    TRANSFORM, SUPPORT_RESULT, SUPPORT, SUPPORT_VERIFIED, REGISTRY,
)
OUTPUTS = {
    "b005": str(BASE / "r1b-b005-shared-priority-v1"),
    "geometry": str(BASE / "r1b-b2-geometry-v1"),
    "b2": str(BASE / "r1b-b2-localization-v1"),
}
REQUIRED_RUNTIME = (
    "experiments/m04r/m04r14_r1b_joint_b005_b2_contract.py",
    "tests/test_r1b_joint_b005_b2_contract.py",
    "src/market_analogues/adequacy_shared_priority.py",
    "src/market_analogues/adequacy_localization.py",
    "tests/test_adequacy_shared_priority.py",
    "tests/test_adequacy_localization.py",
    "experiments/m04r/m04r14_r1b_b005_shared_priority.py",
    "experiments/m04r/m04r14_r1b_b2_geometry.py",
    "experiments/m04r/m04r14_r1b_b2_localization.py",
    "tests/test_r1b_b005_shared_priority.py",
    "tests/test_r1b_b2_geometry.py",
    "tests/test_r1b_b2_localization.py",
    "pyproject.toml",
)
FUTURE_VERIFIERS = (
    "experiments/m04r/m04r14_r1b_b005_shared_priority_verifier.py",
    "experiments/m04r/m04r14_r1b_b2_localization_verifier.py",
)
FORBIDDEN = (
    "outcome", "prediction", "evidence-card", "evidence_card", "stockbee",
    "forward-return", "forward_return", "wf04", "t14-11", "t14-12",
    "b003", "b0-03", "b004", "b0-04", "r1b-b1", "r1b_b1",
    "local-geometry", "cutoff-stability", "perturbation",
)
ANCHORS = {
    str(B001_RESULT): ("result_digest", "8b22a333d8a36d11f5e852c5ae278b79dec6bff8bd3c8c4c9d3609604464d6b5"),
    str(B001_VERIFIED): ("verification_digest", "94d92ba5f94618a04836afeb4bd2e8dcfb570a2fcd11323352937436fbf9b64c"),
    str(R1A_PREREG): ("preregistration_digest", "6a125b098aee5deca3f3d3c00a0ceaef4de99e22e93ee0184e1ed13a584af584"),
    str(V3_PREREG): ("preregistration_digest", "ca8a54dea1db1cb17b8e57314cea41408d0e1491f2fddfa6e8dfb6a8ab0330fd"),
    str(V2_RESULT): ("result_digest", "e1d774d22c7ce0fb8b7b6f0f5dc6bfb6e84be77f69484a707b1189078e8dd42a"),
    str(V2_VERIFIED): ("verification_digest", "1d14eabe98c2630dd94d5a8fbead62791368477573bcfcc85a80d758c6f0ca03"),
    str(PARTITION_RESULT): ("result_digest", "e9e84eeda5d8558110bcde4ae8876c1ae3fff8ad55996656a6f934ca121433f1"),
    str(SUPPORT_RESULT): ("result_digest", "5a1b22f115e67bb06800a398ef81479472bd2d02563dcdea76096b4cc46ae3b9"),
    str(SUPPORT_VERIFIED): ("verification_digest", "f808343c450388521e47ff78303286a1730f46cb40c76c7b74004f1353539d96"),
    str(REGISTRY): ("registry_digest", "e5a1024e021425cace30d2172fee426ff1e6490702ea4f30583ad056e1ca89f2"),
}
LOWER_METRICS = {"episode_unique", "episode_effective_number", "symbol_unique", "symbol_effective_number"}
METRICS = (
    "episode_unique", "episode_max", "episode_top_1_percent_share", "episode_hhi",
    "episode_effective_number", "episode_gini", "symbol_unique", "symbol_max",
    "symbol_top_1_percent_share", "symbol_hhi", "symbol_effective_number",
    "query_any_repeated_symbol_fraction", "repeated_symbol_twice_per_query",
    "repeated_symbol_thrice_per_query", "mean_pairwise_query_episode_overlap",
)


class JointContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise JointContractError(message)


def _embargoed_path(value: str | Path) -> bool:
    """Match embargo labels without treating opaque hexadecimal IDs as labels."""
    lowered = Path(value).as_posix().lower()
    for token in FORBIDDEN:
        if token in {"b003", "b004"}:
            if re.search(rf"(?:^|[-_/]){token}(?:$|[-_/.])", lowered):
                return True
        elif token in lowered:
            return True
    return False


def safe_path(repository: Path, relative: str | Path, allowed: Sequence[str | Path]) -> Path:
    """Exact allowlist and lexical/resolved embargo, checked before opening."""
    name = Path(relative)
    require(not name.is_absolute() and ".." not in name.parts and name.as_posix() not in {"", "."},
            f"unsafe relative path: {relative}")
    require(not _embargoed_path(name), f"embargoed path: {relative}")
    require(name.as_posix() in {Path(value).as_posix() for value in allowed}, f"unapproved path: {relative}")
    root = repository.resolve(); target = root / name
    require(all(not parent.is_symlink() for parent in (target, *target.parents) if parent != root.parent),
            f"symlink path forbidden: {relative}")
    require(target.resolve().is_relative_to(root), f"path escapes repository: {relative}")
    return target


def file_sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, expected: type = dict) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        result = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                            parse_constant=lambda value: require(False, f"non-finite JSON: {value}"))
    except (ValueError, OSError) as error:
        raise JointContractError(f"unreadable JSON: {path}") from error
    require(isinstance(result, expected), f"expected JSON {expected.__name__}: {path}")
    return result


def atomic_json(path: Path, payload: Any) -> None:
    """Atomic, durable, create-only publication, including dangling-symlink refusal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise JointContractError(f"create-only publication exists: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(("git", *arguments), cwd=repository, capture_output=True, text=True)
    require(result.returncode == 0, f"git command failed: {' '.join(arguments)}")
    return result.stdout.strip()


def clean_head(repository: Path) -> str:
    require(not git(repository, "status", "--porcelain", "--untracked-files=all"), "clean tree required")
    return git(repository, "rev-parse", "HEAD")


def runtime_paths(repository: Path) -> tuple[str, ...]:
    # Bind every local package dependency, not merely the top-level import list.
    package = tuple(path.relative_to(repository).as_posix()
                    for path in sorted((repository / "src/market_analogues").rglob("*.py")))
    return tuple(sorted(set((*REQUIRED_RUNTIME, *package))))


def environment() -> dict[str, Any]:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "byteorder": sys.byteorder, "machine": platform.machine(), "platform": platform.platform(),
            "numeric_dtype": "IEEE754 float64", "blas_threads": 1}


def specification() -> dict[str, Any]:
    """Frozen formulas and decisions.  Any changed choice requires a new version."""
    return {
        "inventory": {"queries": 3270, "cohort_episodes": 369, "cohort_links": 2865,
                      "primary_episodes": 357, "primary_links": 2791, "dimensions": 141},
        "population": {"cohort": "verified recurrence >=5 from unchanged sealed top20",
                       "primary": "exact episodes with verified K8 support >=4096",
                       "unit": "episode; all aggregate statistics are equal-episode means",
                       "specificity_reference": "all causally eligible members of the full 369 cohort",
                       "unsupported": "disclose all 12 excluded episodes and 74 links; no replacement"},
        "geometry": {
            "chunk_rows": 64, "distance_shards": 52,
            "production_workers": 12,
            "parallelism": "12 deterministic threads grouped by symbol for reconstruction; 12 processes for exact distance row shards",
            "shard_layout": "52 atomically renamed rows-{start:05d}-{stop:05d}/ directories, each containing METADATA.json, query_pair.npy(rows,3270)<f8 and query_candidate.npy(rows,369)<f8; 51x64 rows plus final 6; every shard binds H1/runtime/ID/transformed-array digests",
            "columns": "coarse[0:96] + stage.reshape(12,4,C)[:,0:3].ravel(C) + structural[0:9]",
            "queries": "reconstruct causal float64 vectors and match sealed v3 raw/transformed digests",
            "candidates": "369 exact causal OHLCV reconstructions; never packed float16 vectors",
            "candidate_ecdf": "tie at 1-based a..b: m=a+b; between values with L smaller: m=2L+1",
            "transform": "z=clip((m-(N+1))/(N-1),-1,+1); exact ties; query constants +0.0",
            "constant_candidate": "same ECDF formula: equal query constant ->0; below ->-1; above ->+1",
            "distance": "sqrt(math.fsum((za[j]-zb[j])**2 for j=0..140)/141)",
            "weights": "none; no whitening", "freeze_before_statistics": True,
            "required_digests": ["query_ids", "candidate_ids", "raw_query_vectors", "raw_candidate_vectors",
                                 "transformed_queries", "transformed_candidates", "query_pair_distances",
                                 "query_candidate_distances", "specificity_ranks", "causal_eligibility"],
            "array_encoding": "shape as canonical JSON plus C-order little-endian float64 bytes with positive zero; causal eligibility is C-order |b1",
        },
        "b005": {
            "seed": 947221, "replicates": 512, "top_k": 20,
            "shard_replicates": 8, "shards": 64, "production_workers": 12,
            "metrics": {name: "lower_is_more_concentrated" if name in LOWER_METRICS
                        else "higher_is_more_concentrated" for name in METRICS},
            "risk_set": "unchanged R1-A exact causal packed A/B risk sets and query-symbol exclusion",
            "selector": "greedy priority order; cap3 per symbol; reject inclusive 252-session overlap; stop20",
            "intervals": "integer session coordinates [5*i,5*i+251] inclusive, never nanoseconds; i is frozen per-symbol ordinal after complete row accounting",
            "global_episode": "sort by (shared episode SHA256,episode_id)",
            "hierarchical": "sort by (shared symbol SHA256,symbol_id,shared episode SHA256,episode_id)",
            "priority_fields": ["domain_utf8", "seed_uint64be", "replicate_uint32be", "entity_utf8"],
            "domains": {"episode": "market-analogues/r1b/b0-05/h-episode/v1", "symbol": "market-analogues/r1b/b0-05/h-symbol/v1"},
            "encoding": "lp16(utf8(domain)) || uint64be(seed) || uint32be(replicate) || lp16(utf8(entity)); lp16 is uint16be byte length; SHA256 full 32 bytes",
            "contract_binding": "H1 freezes seed and domains; no contract digest in B0-05 hash preimage",
            "replicate_indices": "0..511", "pvalue": "(1+tail_count)/513; inclusive direction-specific ties",
            "production_plan": "H_episode once per candidate per replicate reused by both null families; H_symbol once per symbol per replicate; exactly one global order and one hierarchical order per replicate reused by all 3270 risk-set filters",
            "work_counters_per_replicate": {"episode_hashes": 3786156, "symbol_hashes": 11584,
                                           "global_orders": 1, "hierarchical_orders": 1,
                                           "per_query_full_universe_hashes": 0, "per_query_full_universe_sorts": 0,
                                           "query_risk_sets": 3270, "global_risk_set_filters": 3270,
                                           "hierarchical_risk_set_filters": 3270,
                                           "total_risk_set_filters": 6540},
            "work_counters_complete_512": {"episode_hashes": 3786156 * 512, "symbol_hashes": 11584 * 512,
                                          "global_orders": 512, "hierarchical_orders": 512,
                                          "per_query_full_universe_hashes": 0, "per_query_full_universe_sorts": 0,
                                          "query_risk_sets": 3270 * 512,
                                          "global_risk_set_filters": 3270 * 512,
                                          "hierarchical_risk_set_filters": 3270 * 512,
                                          "total_risk_set_filters": 6540 * 512},
            "reference_only": "core evaluate_query_selections is synthetic/reference only; not production full-universe engine",
            "optimization_gate": "synthetic exhaustive reference equality for both families, inclusive overlap/cap3, forced priority ties, risk-set filters and 1/12-worker identities; exactly one global and one hierarchical full-universe ordering per replicate proven by instrumented counters",
            "interpretation": "two descriptive dependence sensitivities; cannot select or rescue any B2 choice",
        },
        "b2": {
            "replicates": 4096, "replicate_indices": "0..4095", "primary_k": 8,
            "shard_replicates": 32, "shards": 128,
            "episode_id_format": "canonical EpisodeKey.id: exactly 24 lowercase hexadecimal characters",
            "n0": "verified session_21 cell tuple", "n1": "session_21 tuple crossed with sealed K8 label",
            "selection": "within each episode/cell choose observed count of smallest (SHA256,query_id) from Ue",
            "priority_fields": ["domain_utf8", "contract_digest_decoded_hex", "family_utf8", "scheme_utf8", "replicate_uint32be", "episode_id_utf8", "query_id_utf8"],
            "priority_domain": "R1B-B2-conditional-priority-v1", "priority_family": "N0-N1-common",
            "priority_schemes": {"primary": "episode", "shared": "shared-query"},
            "priority_contract_identifier": "SHA256 canonical compact sorted JSON of entire specification dictionary; stored as b2_priority_contract_digest outside that dictionary; excludes all preregistration data and self-reference",
            "encoding": "each field is uint32be byte length followed by field bytes; SHA256 full 32 bytes",
            "common_random_numbers": "omit null/design name; same priorities for N0/N1/K12/K16; cells alone differ",
            "shared_priority": "omit episode_id field entirely; query priority shared across all episodes",
            "cohesion": "mean delta_chart over every unordered selected query pair, then mean over episodes",
            "specificity": "query-wise midrank distance within eligible 369 cohort: (rank-0.5)/n; selected-link mean within episode, then episode mean",
            "pvalue": "(1+count(null<=observed))/4097; inclusive ties; p<=0.01 iff tail_count<=39",
            "cohesion_effect": "(mean_null-observed)/mean_null; zero null mean => undefined",
            "specificity_effect": "mean_null-observed",
            "episode_improvement": "fraction with observed episode statistic strictly below its replicate mean",
            "gates": {"n1_both_p_le": 0.01, "cohesion_effect_ge": 0.05,
                      "specificity_effect_ge": 0.05, "each_improved_episode_fraction_ge": 0.60,
                      "n0_same_effect_and_fraction_gates": True, "threshold_equality_passes": True},
            "multiplicity": "intersection-union: both N1 co-primary alternatives must pass; no split p-values",
            "splits": "int(episode_id,16)&1; both nonempty hash splits meet 0.05 cohesion and 0.05 specificity effect",
            "leave_one_out": "all primary-episode deletions keep both effects strictly positive using same replicate table",
            "shared_robustness": "4096 shared-query replicates; both N0/N1 effects strictly positive",
            "breadth": "exp(-sum(p*log(p)))/min(8,m_e); N0 descriptive only; fixed under N1",
            "secondary": "K12/K16 effects plus exact primary selected-link production total and raw market_context diagnostics; no p-values, gate, or rescue",
            "secondary_link_rows": "one row per frozen primary (episode_id,query_id) observed link, sorted by episode_id then query_id; copy finite nonnegative total_distance and component_distances.market_context from its H1-bound top20 case match; require exact 2791-row identity closure",
            "secondary_link_aggregation": "for each distance publish raw rows and rows digest, math.fsum/count selected-link mean, and equal-primary-episode mean of within-episode math.fsum/count means in primary-ID order",
            "zero_null_cohesion": "valid undefined effect gives not_established; never divide by zero",
        },
        "execution": {
            "outputs": OUTPUTS, "workers": 12, "blas_threads": 1,
            "serial_parallel": "synthetic full byte identity at 1/12 workers; real deterministic order independent of scheduling",
            "summation": "math.fsum in sorted episode/query/replicate order; never arrival order",
            "all_producers_require_same_h1": True,
            "verifiers": list(FUTURE_VERIFIERS), "verifier_commits": "after both producers; independently reconstruct fixed outputs",
            "publication": "fsync temporary files and directory; collision-safe atomic create-only publication; never overwrite",
            "resume": "only hash-matching complete shards bound to H1, contract, runtime, geometry, row IDs and ranges; completed shards are never overwritten; unpublished exact-name temporary staging from a killed process is discarded, while any published invalid shard is unresolved",
            "performance": "report wall time, CPU, RSS and 1/12 synthetic scaling; no unsupported fastest claim",
            "external_source_manifest": "before H1, derive exact query/cohort symbol union from verified identities; resolve canonical filename-directory source; hash required raw files plus benchmark/config without OHLCV decoding; reject missing/duplicate/symlink/changed files; rehash at every H1 boundary",
        },
        "failure_taxonomy": {
            "established_pending_independent_verification": "all scientific/effect/robustness gates pass; verifier still required",
            "not_established_pending_independent_verification": "valid complete experiment misses scientific/effect gate or has zero null cohesion",
            "unresolved": "authority, support, geometry, nonfinite, missing/duplicate replica, split/LOEO/shared robustness, H0/H1/runtime, publication or verifier failure",
            "verified_established": "independent verifier passes established producer",
            "verified_not_established": "independent verifier passes valid negative producer",
            "no_adaptation": "no dropping cohort, changed seed, alternative score, new threshold, or secondary rescue",
        },
        "embargo": {"forbidden_path_tokens": list(FORBIDDEN), "before_h1": "verified metadata/support identities and exact raw file-byte hashing only; no OHLCV decoding or real geometry/statistic materialization",
                    "through_b2": "no outcome/label/diagnostic paths; registry authority paths only",
                    "external_data": "pre-H1 byte hashing binds only required canonical OHLCV/benchmark files; decoding after H1 only, with exact file hashes and causal prefixes reverified",
                    "prior_exposure": "R1-A concentration and v2/v3 support observed; B0-03/B0-04/B1/B0-05/B2 results unopened"},
        "claims": {"predictive_claim_authorized": False, "production_promotion_authorized": False,
                   "ranking_change_authorized": False, "adequacy_labels_authorized": False,
                   "real_forward_outcomes_accessed": False,
                   "maximum_after_verified_pass": "R1-A excess recurrence is structurally localized beyond causal exposure, nuisance matching and coarse shared-query structure"},
    }


def _anchored(repository: Path, relative: Path) -> dict[str, Any]:
    value = load_json(safe_path(repository, relative, AUTHORITY_FILES))
    field, expected = ANCHORS[str(relative)]
    # These authorities exclude timestamps, and only those timestamps, from semantics.
    require(value.get(field) == expected and stable_hash({key: item for key, item in value.items()
            if key not in {field, "created_at"}}) == expected, f"authority digest differs: {relative}")
    return value


def _strict_external_path(path: Path, *, directory: bool = False) -> Path:
    """Check the unresolved spelling so a symlink cannot hide behind resolve()."""
    require(path.is_absolute() and ".." not in path.parts, f"absolute canonical source path required: {path}")
    require(not _embargoed_path(path), f"embargoed external source path: {path}")
    require(not any(part.is_symlink() for part in (path, *path.parents)), f"symlink external source path: {path}")
    require(path.is_dir() if directory else path.is_file(), f"missing external {'directory' if directory else 'file'}: {path}")
    return path


def _immutable_file_bytes(path: Path) -> tuple[dict[str, Any], tuple[int, int], bytes]:
    """Hash bytes only and refuse replacement/rewrite during the read.

    The descriptor check guards the whole hash operation.  Its inode identity is
    used to reject aliases in this snapshot, but is not part of the content
    contract: a later byte-identical file copy remains the same frozen content.
    """
    _strict_external_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"nonregular external source: {path}")
        digest = sha256(); blocks: list[bytes] = []
        while block := os.read(descriptor, 8 << 20):
            digest.update(block); blocks.append(block)
        after = os.fstat(descriptor)
        _strict_external_path(path)
        current = path.stat()
        signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        require(signature(before) == signature(after) == signature(current), f"source changed while hashing: {path}")
        content = b"".join(blocks)
        require(len(content) == before.st_size, f"source byte count differs: {path}")
        return ({"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()},
                (before.st_dev, before.st_ino), content)
    except OSError as error:
        raise JointContractError(f"external source read failed: {path}") from error
    finally:
        os.close(descriptor)


def _immutable_file_record(path: Path) -> tuple[dict[str, Any], tuple[int, int]]:
    record, inode, _ = _immutable_file_bytes(path)
    return record, inode


def _bound_config(path: Path, expected_sha256: str) -> tuple[dict[str, Any], tuple[int, int], dict[str, Any], Any]:
    """Authenticate once, then parse both raw and typed views from those bytes."""
    record, inode, content = _immutable_file_bytes(path)
    require(record["sha256"] == expected_sha256, "source configuration hash differs")
    raw = yaml.safe_load(content)
    require(isinstance(raw, dict), "source configuration root malformed")
    return record, inode, raw, load_config_bytes(content, path=path)


def external_source_manifest(
    repository: Path, query_rows: Sequence[Mapping[str, Any]], population: Mapping[str, Any],
    source_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind source bytes without decoding a single bar or constructing geometry.

    This version deliberately supports the frozen filename-directory NASDAQ
    source.  Long-table/column adapters require a separately frozen resolver;
    instantiating their current source object would parse data before H1.
    """
    root = repository.resolve(); config_path = root / DATASET_CONFIG
    _strict_external_path(config_path)
    require(source_lock.get("source_lock_digest") == stable_hash({key: value for key, value in source_lock.items()
            if key != "source_lock_digest"}), "source lock digest differs")
    require(source_lock.get("config_path") == str(config_path), "source configuration path differs")
    config_record, config_inode, raw_config, config = _bound_config(
        config_path, str(source_lock.get("config_sha256")),
    )
    # load_config resolves relative paths.  Inspect original spellings first so
    # that resolution cannot erase a configured symlink component.
    require(isinstance(raw_config, dict) and isinstance(raw_config.get("datasets"), dict)
            and isinstance(raw_config["datasets"].get("nasdaq"), dict), "NASDAQ source configuration malformed")
    raw_dataset = raw_config["datasets"]["nasdaq"]
    require(raw_dataset.get("adapter", "directory") == "directory"
            and raw_dataset.get("symbol_from", "filename") == "filename", "external manifest requires filename-directory source; other adapters need a frozen resolver")
    raw_benchmark = raw_dataset.get("benchmark")
    require(isinstance(raw_benchmark, dict) and isinstance(raw_benchmark.get("path"), str)
            and isinstance(raw_dataset.get("path"), str), "source or benchmark configured path missing")
    for spelling, is_directory in ((raw_dataset["path"], True), (raw_benchmark["path"], False)):
        original = Path(spelling).expanduser()
        original = original if original.is_absolute() else config_path.parent / original
        _strict_external_path(original, directory=is_directory)
    require("nasdaq" in config.datasets, "frozen NASDAQ dataset missing")
    spec = config.datasets["nasdaq"]
    require(spec.adapter == "directory" and spec.symbol_from == "filename", "external manifest requires filename-directory source; other adapters need a frozen resolver")
    _strict_external_path(spec.path, directory=True)
    require(spec.benchmark is not None, "frozen benchmark missing")
    benchmark_path = _strict_external_path(spec.benchmark.path)
    query_ids = [str(row["episode_id"]) for row in query_rows]
    require(len(query_ids) == len(set(query_ids)) and sorted(query_ids) == population["query_ids"], "external manifest query identities differ")
    query_symbols = sorted({str(row["symbol"]) for row in query_rows})
    cohort_ids = [str(row["episode_id"]) for row in population["episodes"]]
    require(len(cohort_ids) == len(set(cohort_ids)) and sorted(cohort_ids) == population["cohort_ids"], "external manifest cohort identities differ")
    cohort_symbols = sorted({str(row["symbol"]) for row in population["episodes"]})
    symbols = sorted(set(query_symbols) | set(cohort_symbols))
    require(bool(symbols) and all(value and "/" not in value and "\\" not in value and value not in {".", ".."} for value in symbols), "unsafe or empty required source symbol")
    # DirectorySource's constructor only resolves glob metadata.  Do not call
    # load(), load_benchmark(), or a long-table factory during preregistration.
    source = source_from_spec(spec)
    require(type(source) is DirectorySource, "canonical directory resolver differs")
    matches: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(spec.path.glob(spec.file_glob)):
        if path.resolve() != benchmark_path.resolve():
            matches[path.stem].append(path)
    require(all(len(paths) == 1 for paths in matches.values()), "duplicate symbol paths in canonical directory")
    require(set(symbols) <= set(matches), "required external source symbol missing")
    require({key.source_symbol for key in source.instruments()} == set(matches), "canonical source inventory differs")
    records = []; inodes = {config_inode}; paths_seen = {str(config_path)}
    for symbol in symbols:
        path = matches[symbol][0]
        require(source._files.get(symbol) == path, "canonical source file resolution differs")
        _strict_external_path(path)
        require(path.is_relative_to(spec.path), "source file escapes canonical directory")
        record, inode = _immutable_file_record(path)
        require(str(path) not in paths_seen and inode not in inodes, "duplicate external source file/inode")
        paths_seen.add(str(path)); inodes.add(inode)
        records.append({"dataset_id": spec.dataset_id, "symbol": symbol, **record})
    benchmark_record, benchmark_inode = _immutable_file_record(benchmark_path)
    require(str(benchmark_path) not in paths_seen and benchmark_inode not in inodes, "duplicate benchmark/source file")
    require(benchmark_record["sha256"] == source_lock.get("benchmark_sha256"), "benchmark source authority changed")
    config_after, _ = _immutable_file_record(config_path)
    require(config_after == config_record, "source configuration changed during manifest")
    state = {"schema_version": "m04r14-r1b-external-source-manifest-v1", "dataset_id": "nasdaq",
             "resolver": "source_from_spec -> DirectorySource; verified unique stem mapping", "config": config_record,
             "dataset_spec": json.loads(json.dumps(asdict(spec), default=str)),
             "source_lock_digest": source_lock["source_lock_digest"],
             "query_ids_digest": stable_hash(population["query_ids"]), "cohort_ids_digest": stable_hash(population["cohort_ids"]),
             "query_symbols": query_symbols, "cohort_symbols": cohort_symbols, "required_symbols": symbols,
             "stock_files": records, "benchmark": benchmark_record, "stock_files_count": len(records),
             "ohlcv_decoded": False, "geometry_materialized": False,
             "immutability": "exact content hashes rechecked at every H1 boundary; mutation during hashing refused"}
    return {**state, "manifest_digest": stable_hash(state)}


def derive_population(
    query_rows: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]], assignments: Sequence[Mapping[str, Any]],
    support: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconstruct identities/cell combinatorics only; no chart distances."""
    query_ids = sorted(str(row["episode_id"]) for row in query_rows)
    require(len(set(query_ids)) == len(query_ids), "duplicate query identity")
    by_case = {str(row["query_episode_id"]): row for row in cases}
    by_cell = {str(row["query_episode_id"]): row for row in cells}
    by_label = {str(row["query_episode_id"]): row for row in assignments}
    require(len(by_case) == len(cases) == len(by_cell) == len(cells) == len(by_label) == len(assignments)
            == len(query_ids) and set(by_case) == set(by_cell) == set(by_label) == set(query_ids), "query membership differs")
    cell_map = {q: list(by_cell[q]["base_cells"]["session_21"]) for q in query_ids}
    labels = {q: {str(k): by_label[q]["labels"][str(k)] for k in (8, 12, 16)} for q in query_ids}
    selected: dict[str, list[str]] = defaultdict(list)
    metadata: dict[str, tuple[str, int]] = {}
    for q in query_ids:
        seen = set()
        for match in by_case[q]["matches"]:
            e = str(match["episode_id"])
            require(e not in seen, "duplicate selected episode")
            seen.add(e); selected[e].append(q)
            meta = (str(match["symbol"]), int(pd.Timestamp(match["cutoff"]).value))
            require(e not in metadata or metadata[e] == meta, "episode metadata conflict")
            metadata[e] = meta
    cohort = sorted(e for e, queries in selected.items() if len(queries) >= 5)
    by_support = {str(row["episode_id"]): row for row in support}
    require(len(by_support) == len(support) and set(cohort) == set(by_support), "cohort/support IDs differ")
    episodes = []
    for e in cohort:
        symbol, cutoff = metadata[e]
        eligible = [q for q in query_ids if cutoff <= pd.Timestamp(by_case[q]["latest_eligible_cutoff"]).value
                    and (symbol != by_case[q]["query_symbol"] or cutoff < pd.Timestamp(by_case[q]["query_start"]).value)]
        observed = sorted(selected[e]); require(set(observed) <= set(eligible), "observed outside causal risk set")
        designs = {}
        for name, k in (("N0", None), ("N1", 8), ("K12", 12), ("K16", 16)):
            def cell(q: str) -> tuple[Any, ...]:
                return tuple(cell_map[q]) + (() if k is None else (labels[q][str(k)],))
            occupied = Counter(cell(q) for q in observed)
            available = Counter(cell(q) for q in eligible)
            support_count = min(4096, math.prod(math.comb(available[key], count) for key, count in occupied.items()))
            expected = by_support[e]["n0_support"] if k is None else by_support[e]["n1_support"][str(k)]
            require(support_count == expected, f"support combinatorics differ: {e}/{name}")
            designs[name] = {"support_capped": support_count, "occupied_cells": [
                {"cell": list(key), "observed": count, "eligible": available[key]}
                for key, count in sorted(occupied.items(), key=lambda item: json.dumps(item[0], separators=(",", ":")))]}
        require(len(eligible) == by_support[e]["eligible_queries"] and len(observed) == by_support[e]["observed_inbound_queries"], "support denominator differs")
        episodes.append({"episode_id": e, "symbol": symbol, "cutoff_ns": cutoff, "observed_query_ids": observed,
                         "eligible_query_ids": eligible, "observed_count": len(observed), "designs": designs,
                         "primary": designs["N1"]["support_capped"] >= 4096})
    primary = [row["episode_id"] for row in episodes if row["primary"]]
    result = {"query_ids": query_ids, "cohort_ids": cohort, "primary_ids": primary,
              "query_cells": [{"query_id": q, "n0": cell_map[q], "labels": labels[q]} for q in query_ids],
              "episodes": episodes, "episode_split": {e: int(e, 16) & 1 for e in primary}}
    return {**result, "population_digest": stable_hash(result)}


def authority_snapshot(repository: Path) -> dict[str, Any]:
    authority = {str(path): _anchored(repository, path) for path in AUTHORITY_FILES if str(path) in ANCHORS}
    b001 = authority[str(B001_VERIFIED)]; balanced = authority[str(SUPPORT_VERIFIED)]
    require(b001.get("passed") is True and b001.get("b001_complete") is True
            and b001.get("joint_b005_b2_contract_freeze_may_proceed") is True, "B0-01 verification absent")
    require(balanced.get("passed") is True and balanced.get("support_decision_verified") is True
            and balanced.get("primary_k8_support_gate_passed") is True, "balanced support verification absent")
    r1a = authority[str(R1A_PREREG)]
    require(r1a["metric_directions"] == specification()["b005"]["metrics"]
            and r1a["execution"]["seed"] == 947221 and r1a["execution"]["null_replicates"] == 512, "R1-A null contract differs")
    hashes = {str(path): file_sha(safe_path(repository, path, AUTHORITY_FILES)) for path in AUTHORITY_FILES}
    for source, field, target in (
        (V2_RESULT, "query_cells_sha256", V2_CELLS), (V2_RESULT, "cohort_support_sha256", V2_COHORT),
        (PARTITION_RESULT, "partitions_sha256", PARTITIONS), (PARTITION_RESULT, "transform_sha256", TRANSFORM),
        (SUPPORT_RESULT, "support_sha256", SUPPORT),
    ):
        require(authority[str(source)][field] == hashes[str(target)], f"sidecar hash differs: {target}")
    verified_manifest = authority[str(B001_RESULT)]["opened_path_manifest"]
    case_rows = [row for row in verified_manifest if Path(row["path"]).parent == CASES]
    require(len(case_rows) == 3270, "verified case manifest inventory differs")
    case_paths = tuple(row["path"] for row in case_rows)
    cases = []
    for row in case_rows:
        path = safe_path(repository, row["path"], case_paths)
        require(file_sha(path) == row["sha256"] and path.stat().st_size == row["bytes"], "verified case changed")
        cases.append(load_json(path))
    partitions = load_json(safe_path(repository, PARTITIONS, AUTHORITY_FILES))
    population = derive_population(authority[str(REGISTRY)]["cases_data"], cases,
                                   load_json(safe_path(repository, V2_CELLS, AUTHORITY_FILES), list),
                                   partitions["assignments"], load_json(safe_path(repository, SUPPORT, AUTHORITY_FILES), list))
    require(len(population["query_ids"]) == 3270 and len(population["cohort_ids"]) == 369
            and len(population["primary_ids"]) == 357
            and sum(row["observed_count"] for row in population["episodes"]) == 2865
            and sum(row["observed_count"] for row in population["episodes"] if row["primary"]) == 2791,
            "frozen population inventory differs")
    return {"authority_file_sha256": hashes, "authority_semantic_digests": {
        name: {field: expected} for name, (field, expected) in ANCHORS.items()},
        "reuse_authority_manifest_digest": authority[str(B001_RESULT)]["opened_path_manifest_digest"],
        "reuse_authority_manifest": verified_manifest, "case_manifest": case_rows,
        "source_lock": authority[str(REGISTRY)]["source_lock"],
        "external_source_manifest": external_source_manifest(repository, authority[str(REGISTRY)]["cases_data"],
                                                             population, authority[str(REGISTRY)]["source_lock"]),
        "sealed_query_transform": load_json(safe_path(repository, TRANSFORM, AUTHORITY_FILES)),
        "population": population}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(); h0 = clean_head(repository)
    require(not (repository / PREREGISTRATION).exists() and not (repository / PREREGISTRATION).is_symlink(), "preregistration exists")
    for relative in OUTPUTS.values():
        require(not (repository / relative).exists() and not (repository / relative).is_symlink(), "scientific output already exists")
    paths = runtime_paths(repository)
    hashes = {name: file_sha(repository / name) for name in paths}
    payload = {"schema_version": SCHEMA, "implementation_commit": h0,
               "specification": specification(), "b2_priority_contract_digest": stable_hash(specification()), "environment": environment(),
               "runtime_sha256": hashes, "authorities": authority_snapshot(repository)}
    require(clean_head(repository) == h0, "H0 changed during preregistration")
    payload["preregistration_digest"] = stable_hash(payload)
    return payload


def preregister(repository: Path) -> dict[str, Any]:
    payload = build_preregistration(repository)
    atomic_json(safe_path(repository, PREREGISTRATION, (PREREGISTRATION,)), payload)
    return payload


def validate_h1(repository: Path, payload: Mapping[str, Any] | None = None) -> str:
    repository = repository.resolve(); head = clean_head(repository)
    path = safe_path(repository, PREREGISTRATION, (PREREGISTRATION,))
    actual = load_json(path)
    require(payload is None or actual == payload, "provided preregistration differs from committed file")
    payload = actual
    require(set(payload) == {"schema_version", "implementation_commit", "specification", "b2_priority_contract_digest", "environment", "runtime_sha256", "authorities", "preregistration_digest"}, "contract field closure differs")
    require(payload["preregistration_digest"] == stable_hash({key: value for key, value in payload.items() if key != "preregistration_digest"}), "contract digest differs")
    require(payload["schema_version"] == SCHEMA and payload["specification"] == specification(), "frozen specification differs")
    require(payload["b2_priority_contract_digest"] == stable_hash(specification()), "B2 priority contract digest differs")
    require(payload["environment"] == environment(), "runtime environment differs")
    parents = git(repository, "rev-list", "--parents", "-n", "1", "HEAD").split()
    require(parents == [head, payload["implementation_commit"]], "H1 must be sole child of H0")
    require(git(repository, "diff-tree", "--no-commit-id", "--name-status", "-r", "HEAD") == f"A\t{PREREGISTRATION.as_posix()}", "H1 must add exactly one joint preregistration")
    blob = subprocess.run(("git", "show", f"HEAD:{PREREGISTRATION}"), cwd=repository, capture_output=True)
    require(blob.returncode == 0 and blob.stdout == path.read_bytes(), "committed preregistration bytes differ")
    require(set(payload["runtime_sha256"]) == set(runtime_paths(repository)), "runtime file closure differs")
    for name, expected in payload["runtime_sha256"].items():
        require(file_sha(repository / name) == expected, f"runtime changed: {name}")
    require(payload["authorities"] == authority_snapshot(repository), "frozen authority/population differs")
    return head


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "validate"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = preregister(args.repository) if args.mode == "preregister" else {"h1": validate_h1(args.repository)}
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen machine contract for the M04R-14 untouched candidate-first run."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


SCHEMA = "m04r14-untouched-candidate-contract-v1"
BINDING_SCHEMA = "m04r14-untouched-registry-binding-v1"
PREREGISTRATION_SCHEMA = "m04r14-untouched-candidate-preregistration-v1"
RESULT_SCHEMA = "m04r14-untouched-candidate-result-v1"
VERIFICATION_SCHEMA = "m04r14-untouched-candidate-preopen-verification-v1"

REGISTRY_RELATIVE = Path(
    "config/data/analogues/m04r14/nasdaq-untouched-registry-v1"
)
REGISTRY_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/nasdaq-untouched-registry-v1-verification"
)
BINDING_RELATIVE = Path(
    "experiments/m04r/m04r14_untouched_registry_binding.json"
)
PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_untouched_candidate_preregistered.json"
)
CANDIDATE_RELATIVE = Path(
    "config/data/analogues/m04r14/untouched-candidate-v1"
)
CANDIDATE_VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/untouched-candidate-v1-verification"
)
CONFIG_RELATIVE = Path("config/datasets.example.yaml")
SOURCE_FULL_RELATIVE = Path("config/data/analogues/poc/m04r/packed-bound-full")
GENERATION_ID = "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
PROVENANCE_DIGEST = "83ccfa62ac7ffec03e48d0f8a5634c7b8f1b8b0dde1426be22cc343fe116f62d"
RESIDENT_ROOT = Path("/dev/shm/market-analogues/m04r11-candidate-v2") / GENERATION_ID

# These are the T14-05 qualified grouped-service controls.  They deliberately
# replace the older 16K->32K registry execution defaults without changing the
# mathematical search contract or final certified result.
CONTROLS: dict[str, Any] = {
    "block_rows": 4096,
    "deferred_alignments": True,
    "exact_workers_per_process": 1,
    "initial_frontier_rows": 1000,
    "maximum_frontier_rows": 16384,
    "numba_threads_per_process": 1,
    "processes": 8,
    "requested_positions": True,
    "seed_rows": 512,
    "sorted_joined_iqr_merge": True,
    "vector_lower_bounds": True,
}

PERFORMANCE_LIMITS: dict[str, Any] = {
    "candidate_wall_seconds_max": 2400.0,
    "case_exact_seconds_max": 240.0,
    "case_exact_seconds_p95": 120.0,
    "p95_method": "nearest-rank-ceiling",
    "process_swap_kib_max": 0,
}

CLAIMS: dict[str, Any] = {
    "authority_accessed": False,
    "real_forward_outcomes_accessed": False,
    "candidate_first": True,
    "cases": 72,
    "symbols": 36,
    "top_k": 20,
    "production_promotion_authorized": False,
}

FORBIDDEN_PATH_TOKENS = (
    "authorities-sealed", "authority-root", "forward-outcome", "outcomes",
    "results-opened",
)

RUNTIME_FILES = (
    "experiments/m04r/m04r14_untouched_candidate_contract.py",
    "experiments/m04r/m04r14_untouched_candidate.py",
    "experiments/m04r/verify_m04r14_untouched_candidate.py",
    "experiments/m04r/m04r11_build_authorities.py",
    "experiments/m04r/m04r13_threaded_certified_exposed.py",
    "experiments/m04r/m04r14_throughput_poc.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


CONTRACT_STATE = {
    "schema_version": SCHEMA,
    "registry_relative": REGISTRY_RELATIVE.as_posix(),
    "registry_verification_relative": REGISTRY_VERIFICATION_RELATIVE.as_posix(),
    "binding_relative": BINDING_RELATIVE.as_posix(),
    "preregistration_relative": PREREGISTRATION_RELATIVE.as_posix(),
    "candidate_relative": CANDIDATE_RELATIVE.as_posix(),
    "candidate_verification_relative": CANDIDATE_VERIFICATION_RELATIVE.as_posix(),
    "config_relative": CONFIG_RELATIVE.as_posix(),
    "source_full_relative": SOURCE_FULL_RELATIVE.as_posix(),
    "resident_root": str(RESIDENT_ROOT),
    "generation_id": GENERATION_ID,
    "provenance_digest": PROVENANCE_DIGEST,
    "controls": CONTROLS,
    "performance_limits": PERFORMANCE_LIMITS,
    "claims": CLAIMS,
    "forbidden_path_tokens": list(FORBIDDEN_PATH_TOKENS),
    "lifecycle": "clean H0 -> registry-binding-only H1 -> preregistration-only H2",
    "failure_policy": "create-only terminal failure; no resume or replacement",
}
CONTRACT_DIGEST = digest(CONTRACT_STATE)

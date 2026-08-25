from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters import file_fingerprint
from .m04r_quantized_verification import verify_m04r_quantized_bound
from .types import stable_hash


BRANCH_BOUND_PREREQUISITE_SCHEMA = "m04r-branch-bound-prerequisite-v1"


def validated_branch_bound_evidence(root: Path) -> dict[str, Any]:
    """Validate and bind the proof-supporting evidence required by M04R-11 v4."""
    million_path = root / "million-pair-gate.json"
    authority_path = root / "authority-gate.json"
    verification_path = root / "verification" / "m04r-quantized-bound.json"
    if not all(path.is_file() for path in (
        million_path, authority_path, verification_path,
    )):
        raise ValueError("branch-aware bound prerequisite evidence is incomplete")
    million = json.loads(million_path.read_text())
    authority = json.loads(authority_path.read_text())
    verification = json.loads(verification_path.read_text())
    result = verify_m04r_quantized_bound(
        million_path, authority_path, branch_aware=True,
    )
    expected_verification = {
        "schema_version": result.metrics["schema_version"],
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "contract": result.contract,
        "result_digest": result.result_digest,
    }
    if not result.passed or verification != expected_verification:
        raise ValueError("branch-aware bound prerequisite verification differs")
    deterministic = {
        "schema_version": BRANCH_BOUND_PREREQUISITE_SCHEMA,
        "contract_digest": result.contract["digest"],
        "verification_result_digest": result.result_digest,
        "million_result_digest": million["result_digest"],
        "authority_result_digest": authority["result_digest"],
        "files": {
            "million_pair_gate_sha256": file_fingerprint(million_path),
            "authority_gate_sha256": file_fingerprint(authority_path),
            "verification_sha256": file_fingerprint(verification_path),
        },
        "real_forward_outcomes_accessed": False,
    }
    return {**deterministic, "digest": stable_hash(deterministic)}

"""Verify untouched candidate semantics under the amended grouped-batch policy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r13_threaded_certified_exposed as m13
from experiments.m04r import m04r14_untouched_candidate_contract as contract
from experiments.m04r import verify_m04r14_untouched_candidate as base


SCHEMA = "m04r14-untouched-batch-amendment-verification-v1"
AMENDMENT = Path("config/m04r14-untouched-batch-policy-amendment.json")
OUTPUT = Path("config/data/analogues/m04r14/untouched-candidate-v1-batch-verification")


class AmendmentVerificationError(RuntimeError):
    pass


def _introduced_commit(repository: Path, path: Path) -> str:
    value = base._git(repository, "log", "--diff-filter=A", "--format=%H", "--", path.as_posix())
    commits = str(value).splitlines()
    if len(commits) != 1: raise AmendmentVerificationError(f"introduction commit differs: {path}")
    return commits[0]


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    amendment, amendment_raw = base._read(repository / AMENDMENT)
    result, result_raw = base._read(repository / contract.CANDIDATE_RELATIVE / "RESULT.json")
    binding, binding_raw = base._read(repository / contract.BINDING_RELATIVE)
    prereg, prereg_raw = base._read(repository / contract.PREREGISTRATION_RELATIVE)
    required_amendment = {
        "schema_version": "m04r14-untouched-batch-policy-amendment-v1",
        "status": "post-run-user-authorized-policy-correction",
        "original_result_digest": result.get("result_digest"),
        "original_performance_passed": False, "original_failure_preserved": True,
        "candidate_rerun_authorized": False, "production_promotion_authorized": False,
    }
    if any(amendment.get(key) != value for key, value in required_amendment.items()):
        raise AmendmentVerificationError("batch amendment boundary differs")
    batch = amendment.get("batch_acceptance")
    if batch != {"required_cases": 72, "required_matches_per_case": 20,
            "semantic_pass_required": True, "candidate_wall_seconds_max": 2400.0,
            "authority_accessed_required": False,
            "real_forward_outcomes_accessed_required": False}:
        raise AmendmentVerificationError("batch acceptance policy differs")
    deterministic = {key: value for key, value in result.items() if key not in {"result_digest", "created_at"}}
    if result.get("result_digest") != contract.digest(deterministic) \
            or result.get("semantic_passed") is not True \
            or result.get("performance_passed") is not False \
            or float(result.get("wall_seconds", float("inf"))) > 2400.0 \
            or result.get("authority_accessed") is not False \
            or result.get("real_forward_outcomes_accessed") is not False:
        raise AmendmentVerificationError("original terminal candidate differs")
    h0 = str(binding["implementation_h0"]); h1 = str(prereg["registry_binding_h1"])
    h2 = _introduced_commit(repository, contract.PREREGISTRATION_RELATIVE)
    base._only_child(repository, h1, h0, contract.BINDING_RELATIVE)
    base._only_child(repository, h2, h1, contract.PREREGISTRATION_RELATIVE)
    for name, expected in binding["runtime_files"].items():
        h0_raw = base._git(repository, "show", f"{h0}:{name}", raw=True)
        h2_raw = base._git(repository, "show", f"{h2}:{name}", raw=True)
        if sha256(h0_raw).hexdigest() != expected or sha256(h2_raw).hexdigest() != expected:
            raise AmendmentVerificationError(f"frozen runtime blob differs: {name}")
    registry_root = repository / contract.REGISTRY_RELATIVE
    registry, registry_raw = base._read(registry_root / "query-registry.json")
    receipt, _ = base._read(repository / contract.REGISTRY_VERIFICATION_RELATIVE / "VERIFIED.json")
    if registry.get("registry_digest") != binding.get("registry_digest") \
            or receipt.get("registry_digest") != binding.get("registry_digest") \
            or sha256(registry_raw).hexdigest() != binding.get("registry_sha256"):
        raise AmendmentVerificationError("registry evidence differs")
    root = repository / contract.CANDIDATE_RELATIVE; paths = sorted((root / "cases").glob("*.json"))
    manifest = [{"path": path.relative_to(root).as_posix(), "sha256": base._sha(path),
                 "bytes": path.stat().st_size} for path in paths]
    if len(paths) != 72 or manifest != result.get("case_manifest") \
            or contract.digest(manifest) != result.get("case_manifest_digest"):
        raise AmendmentVerificationError("candidate manifest differs")
    cases = [m13.CaseInput(index, dict(row)) for index, row in enumerate(registry["cases_data"])]
    inputs = m13.Inputs(repository, repository / contract.CONFIG_RELATIVE, registry_root,
        repository / contract.SOURCE_FULL_RELATIVE / "store", contract.RESIDENT_ROOT, root,
        contract.GENERATION_ID, contract.PROVENANCE_DIGEST, 1024 ** 3,
        str(registry["registry_digest"]), tuple(cases), str(prereg["preregistration_digest"]))
    observed = {}
    for path in paths:
        row, _ = base._read(path); query_id = str(row.get("query_episode_id"))
        if query_id in observed: raise AmendmentVerificationError("duplicate candidate query")
        observed[query_id] = row
    verified_matches = 0
    for case in cases:
        row = observed.get(case.query_id)
        if row is None or row.get("registry_case_id") != case.case_id \
                or row.get("gate_passed") is not True or len(row.get("matches", [])) != 20 \
                or row.get("result_digest") != base._deterministic_case_digest(row) \
                or row.get("checkpoint_integrity_digest") != base._integrity_digest(row):
            raise AmendmentVerificationError(f"candidate case differs: {case.case_id}")
        source, episode, request, packed = m13._case_context(inputs, case)
        query = m13.query_binding(source, episode, request, packed, contract.PROVENANCE_DIGEST)
        m13.validate_certificate_and_matches({**row["certificate"], "elapsed_seconds": 0.0},
            row["matches"], case.query_id,
            expected_input_digest=query["certified_input_digest"])
        verified_matches += len(row["matches"])
    resident = m13.resident_full(repository / contract.SOURCE_FULL_RELATIVE / "store",
        contract.RESIDENT_ROOT, contract.GENERATION_ID, contract.PROVENANCE_DIGEST, 1024 ** 3)
    if resident["identity_digest"] != result.get("resident_identity_digest") \
            or resident["content_digest"] != result.get("source_content_digest"):
        raise AmendmentVerificationError("source/resident lease differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "policy_classification": "post-run-user-authorized-batch-amendment",
        "original_performance_failure_preserved": True,
        "candidate_result_digest": result["result_digest"],
        "candidate_result_sha256": sha256(result_raw).hexdigest(),
        "amendment_sha256": sha256(amendment_raw).hexdigest(),
        "binding_sha256": sha256(binding_raw).hexdigest(),
        "preregistration_sha256": sha256(prereg_raw).hexdigest(),
        "registry_digest": registry["registry_digest"], "verified_cases": 72,
        "verified_matches": verified_matches, "batch_wall_seconds": result["wall_seconds"],
        "batch_wall_limit_seconds": 2400.0, "authority_accessed": False,
        "real_forward_outcomes_accessed": False, "results_open_authorized": True,
        "production_promotion_authorized": False}
    return {**state, "result_digest": contract.digest(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise AmendmentVerificationError("verification root exists")
    path.mkdir(parents=False); descriptor = os.open(path / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({**value, "created_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True).encode() + b"\n"); handle.flush(); os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True); value = verify(repository)
    if not args.dry_run: _publish(repository / OUTPUT, value)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())

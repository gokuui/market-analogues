"""Independent verifier for the separate M04R-14 untouched authority."""
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


SCHEMA = "m04r14-untouched-authority-verification-v1"
AUTHORITY_ROOT = Path("config/data/analogues/m04r14/untouched-authority-v1")
RESULTS_OPEN_MARKER = Path(
    "config/data/analogues/m04r14/untouched-results-opened-v1/RESULTS_OPENED.json"
)
OUTPUT = Path("config/data/analogues/m04r14/untouched-authority-v1-verification")


class VerificationError(RuntimeError): pass


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); root = repository / AUTHORITY_ROOT
    result, result_raw = base._read(root / "AUTHORITY.json")
    deterministic = {key: value for key, value in result.items() if key not in {"result_digest", "created_at"}}
    if result.get("result_digest") != contract.digest(deterministic) \
            or result.get("status") != "sealed" or result.get("semantic_passed") is not True \
            or result.get("cases") != 72 or result.get("matches") != 1440 \
            or result.get("candidate_result_accessed") is not False \
            or result.get("real_forward_outcomes_accessed") is not False:
        raise VerificationError("authority aggregate differs")
    registry_root = repository / contract.REGISTRY_RELATIVE
    registry, _ = base._read(registry_root / "query-registry.json")
    marker, marker_raw = base._read(repository / RESULTS_OPEN_MARKER)
    if marker.get("marker_digest") != result.get("marker_digest") \
            or marker.get("authority_access_authorized") is not True \
            or marker.get("outcome_access_authorized") is not False \
            or registry.get("registry_digest") != result.get("registry_digest"):
        raise VerificationError("authority input binding differs")
    paths = sorted((root / "cases").glob("*.json"))
    manifest = [{"path": path.relative_to(root).as_posix(), "sha256": base._sha(path),
                 "bytes": path.stat().st_size} for path in paths]
    if len(paths) != 72 or manifest != result.get("case_manifest") \
            or contract.digest(manifest) != result.get("case_manifest_digest"):
        raise VerificationError("authority manifest differs")
    cases = [m13.CaseInput(index, dict(row)) for index, row in enumerate(registry["cases_data"])]
    inputs = m13.Inputs(repository, repository / contract.CONFIG_RELATIVE, registry_root,
        repository / contract.SOURCE_FULL_RELATIVE / "store", contract.RESIDENT_ROOT, root,
        contract.GENERATION_ID, contract.PROVENANCE_DIGEST, 1024 ** 3,
        str(registry["registry_digest"]), tuple(cases), "authority-verification")
    observed: dict[str, dict[str, Any]] = {}
    for path in paths:
        row, _ = base._read(path); query_id = str(row.get("query_episode_id"))
        if query_id in observed: raise VerificationError("duplicate authority query")
        observed[query_id] = row
    # M13's semantic validator is intentionally parameterized by module-level
    # execution controls.  Bind it to the authority's frozen 16K->32K schedule
    # while validating; the candidate verifier separately uses 1K->16K.
    original_controls = (m13.INITIAL_FRONTIER, m13.MAXIMUM_FRONTIER)
    m13.INITIAL_FRONTIER, m13.MAXIMUM_FRONTIER = 16_384, 32_768
    verified_matches = 0
    try:
        for case in cases:
            row = observed.get(case.query_id)
            if row is None or row.get("registry_case_id") != case.case_id \
                    or row.get("gate_passed") is not True or len(row.get("matches", [])) != 20 \
                    or row.get("result_digest") != base._deterministic_case_digest(row) \
                    or row.get("checkpoint_integrity_digest") != base._integrity_digest(row):
                raise VerificationError(f"authority case differs: {case.case_id}")
            source, episode, request, packed = m13._case_context(inputs, case)
            query = m13.query_binding(source, episode, request, packed, contract.PROVENANCE_DIGEST)
            m13.validate_certificate_and_matches({**row["certificate"], "elapsed_seconds": 0.0},
                row["matches"], case.query_id,
                expected_input_digest=query["certified_input_digest"])
            verified_matches += len(row["matches"])
    finally:
        m13.INITIAL_FRONTIER, m13.MAXIMUM_FRONTIER = original_controls
    resident = m13.resident_full(repository / contract.SOURCE_FULL_RELATIVE / "store",
        contract.RESIDENT_ROOT, contract.GENERATION_ID, contract.PROVENANCE_DIGEST, 1024 ** 3)
    if resident["identity_digest"] != result.get("resident_identity_digest") \
            or resident["content_digest"] != result.get("source_content_digest"):
        raise VerificationError("authority source/resident lease differs")
    state = {"schema_version": SCHEMA, "status": "verified", "passed": True,
        "authority_result_digest": result["result_digest"],
        "authority_result_sha256": sha256(result_raw).hexdigest(),
        "results_open_marker_sha256": sha256(marker_raw).hexdigest(),
        "registry_digest": registry["registry_digest"], "verified_cases": 72,
        "verified_matches": verified_matches, "candidate_result_accessed": False,
        "real_forward_outcomes_accessed": False, "comparison_authorized": True,
        "production_promotion_authorized": False}
    return {**state, "result_digest": contract.digest(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink(): raise VerificationError("authority verification root exists")
    path.mkdir(parents=False); descriptor = os.open(path / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({**value, "created_at": datetime.now(timezone.utc).isoformat()}, indent=2,
            sort_keys=True).encode() + b"\n"); handle.flush(); os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); repository = args.repository.resolve(strict=True); value = verify(repository)
    _publish(repository / OUTPUT, value); print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())

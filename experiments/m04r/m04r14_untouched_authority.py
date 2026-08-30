"""Build a separate exact authority for the opened M04R-14 untouched cohort."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from experiments.m04r import m04r11_build_authorities as engine
from experiments.m04r import m04r13_threaded_certified_exposed as m13
from experiments.m04r import m04r14_untouched_candidate as support
from experiments.m04r import m04r14_untouched_candidate_contract as contract


SCHEMA = "m04r14-untouched-authority-v1"
OUTPUT = Path("config/data/analogues/m04r14/untouched-authority-v1")
MARKER = Path("config/data/analogues/m04r14/untouched-results-opened-v1/RESULTS_OPENED.json")


class AuthorityError(RuntimeError): pass


def controls(registry: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(registry["search_contract"]["controls"])
    expected = {"block_rows": 4096, "deferred_alignments": True,
        "exact_workers_per_process": 1, "initial_frontier_rows": 16384,
        "maximum_frontier_rows": 32768, "numba_threads_per_process": 1,
        "processes": 8, "requested_positions": True, "seed_rows": 512,
        "sorted_joined_iqr_merge": True, "vector_lower_bounds": True}
    if value != expected: raise AuthorityError("registry authority controls differ")
    return value


def _worker(repository: str, output: str, rows: tuple[dict[str, Any], ...],
            registry_digest: str, marker_digest: str, execution: dict[str, Any]) -> dict[str, Any]:
    repo = Path(repository)
    return engine._worker(str(repo / contract.CONFIG_RELATIVE),
        str(repo / contract.SOURCE_FULL_RELATIVE), output, contract.GENERATION_ID,
        execution, rows, execution["controls"])


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    marker = support._read(repository / MARKER)
    if marker.get("status") != "results_opened" or marker.get("authority_access_authorized") is not True \
            or marker.get("outcome_access_authorized") is not False:
        raise AuthorityError("results-open marker differs")
    registry = support._read(repository / contract.REGISTRY_RELATIVE / "query-registry.json")
    binding = support._read(repository / contract.BINDING_RELATIVE)
    rows = [dict(row) for row in registry.get("cases_data", [])]
    if len(rows) != 72 or len({row["episode_id"] for row in rows}) != 72 \
            or registry.get("registry_digest") != binding.get("registry_digest"):
        raise AuthorityError("authority registry differs")
    selected_controls = controls(registry)
    root = repository / OUTPUT
    if root.exists() or root.is_symlink(): raise AuthorityError("authority root exists")
    root.mkdir(parents=True); (root / "cases").mkdir()
    started = perf_counter()
    try:
        resident_before = m13.resident_full(repository / contract.SOURCE_FULL_RELATIVE / "store",
            contract.RESIDENT_ROOT, contract.GENERATION_ID, contract.PROVENANCE_DIGEST, 1024 ** 3)
        run = {"schema_version": SCHEMA, "status": "running", "marker_digest": marker["marker_digest"],
            "registry_digest": registry["registry_digest"], "generation_id": contract.GENERATION_ID,
            "controls": selected_controls, "source_content_digest": resident_before["content_digest"],
            "resident_identity_digest": resident_before["identity_digest"],
            "candidate_result_accessed": False, "real_forward_outcomes_accessed": False,
            "created_at": datetime.now(timezone.utc).isoformat()}
        support._atomic(root / "RUN_STARTED.json", run)
        groups = support.balanced_groups(rows, 8)
        state = {"schema_version": SCHEMA, "registry_digest": registry["registry_digest"],
            "generation_id": contract.GENERATION_ID, "marker_digest": marker["marker_digest"],
            "controls": selected_controls, "authority_role": "separate-32k-frontier"}
        execution = {**state, "contract_digest": contract.digest(state)}
        results: list[dict[str, Any]] = []; context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=8, mp_context=context) as pool:
            futures = {pool.submit(_worker, str(repository), str(root), group,
                registry["registry_digest"], marker["marker_digest"], execution): index
                for index, group in enumerate(groups)}
            for future in as_completed(futures): results.append({"group_index": futures[future], **future.result()})
        paths = sorted((root / "cases").glob("*.json")); observed = [support._read(path) for path in paths]
        if len(observed) != 72 or len({row.get("query_episode_id") for row in observed}) != 72 \
                or not all(row.get("gate_passed") is True and len(row.get("matches", [])) == 20
                           and row.get("real_forward_outcomes_accessed") is False for row in observed):
            raise AuthorityError("authority case gate differs")
        resident_after = m13.resident_full(repository / contract.SOURCE_FULL_RELATIVE / "store",
            contract.RESIDENT_ROOT, contract.GENERATION_ID, contract.PROVENANCE_DIGEST, 1024 ** 3)
        if resident_after["identity_digest"] != resident_before["identity_digest"] \
                or resident_after["content_digest"] != resident_before["content_digest"]:
            raise AuthorityError("authority source/resident changed")
        manifest = [{"path": path.relative_to(root).as_posix(), "sha256": support._sha(path),
                     "bytes": path.stat().st_size} for path in paths]
        final = {"schema_version": SCHEMA, "status": "sealed", "marker_digest": marker["marker_digest"],
            "registry_digest": registry["registry_digest"], "generation_id": contract.GENERATION_ID,
            "controls": selected_controls, "cases": 72, "matches": 1440,
            "groups": sorted(results, key=lambda row: row["group_index"]),
            "case_manifest": manifest, "case_manifest_digest": contract.digest(manifest),
            "wall_seconds": perf_counter() - started, "semantic_passed": True,
            "source_content_digest": resident_after["content_digest"],
            "resident_identity_digest": resident_after["identity_digest"],
            "candidate_result_accessed": False, "real_forward_outcomes_accessed": False,
            "production_promotion_authorized": False}
        value = {**final, "result_digest": contract.digest(final),
            "created_at": datetime.now(timezone.utc).isoformat()}
        support._atomic(root / "AUTHORITY.json", value); return value
    except BaseException as exc:
        if not (root / "AUTHORITY.json").exists() and not (root / "FAILED.json").exists():
            support._atomic(root / "FAILED.json", {"schema_version": SCHEMA, "status": "failed",
                "error_type": type(exc).__name__, "message": str(exc), "resume_authorized": False,
                "created_at": datetime.now(timezone.utc).isoformat()})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv); value = execute(args.repository)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())

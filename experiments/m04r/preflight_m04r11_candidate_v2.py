"""Development-only CAKE preflight for the complete candidate-v2 worker."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from html import escape
import json
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any, Mapping
from uuid import uuid4

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import m04r11_candidate_matrix_v2 as producer  # noqa: E402
from m04r11_candidate_v2_contract import (  # noqa: E402
    FROZEN_GENERATION_ID, PERFORMANCE_ATTEMPT_GATES, SCAN_PROTOCOL,
    SEMANTIC_CASE_GATES, expected_preregistration_path, expected_roots,
    load_and_validate_preregistration, performance_attempt_digest,
    semantic_case_digest,
)
from market_analogues.config import load_config  # noqa: E402
from market_analogues.resident_store import (  # noqa: E402
    prepare_resident_mirror_observed,
)
from market_analogues.types import stable_hash  # noqa: E402


DEVELOPMENT_REGISTRY_RELATIVE_PATH = (
    "config/data/analogues/poc/m04r/batch-query-registry/query-registry.json"
)
CONFIG_RELATIVE_PATH = "config/datasets.example.yaml"
CAKE_CASE_ID = "nasdaq-CAKE-current-252"
CAKE_SYMBOL = "CAKE"
CAKE_CUTOFF = "2026-03-30T00:00:00"
CAKE_EPISODE_ID = "cb2bdccd1386790ef8a16bd0"
EXPECTED_CANDIDATE_DIGEST = (
    "c8f51e202205367bf3cfda6a62cd3bcbe0e29e32f5b809d7e88431a194692555"
)
EXPECTED_SCAN_RESULT_DIGEST = (
    "bb36c3b770446bc1a62aa3552ab2ca1ccbe1a8c04a90d4db01cedb6e8180b75b"
)
PREFLIGHT_SCHEMA = "candidate-resident-worker-cake-preflight-v1"
PREFLIGHT_ROLE = {
    "performance_role": "development_preflight_exposed",
    "recall_role": "development_only_not_holdout",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_development_case(
    *, repository_root: Path, registry_path: Path, case_id: str,
    frozen_query_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_path = (
        repository_root.resolve() / DEVELOPMENT_REGISTRY_RELATIVE_PATH
    )
    if registry_path.resolve() != expected_path:
        raise ValueError("preflight development registry path differs")
    if case_id != CAKE_CASE_ID:
        raise ValueError("preflight permits only exposed CAKE current")
    registry = _read_json(registry_path)
    cases = [
        dict(row) for row in registry.get("cases_data", [])
        if row.get("case_id") == CAKE_CASE_ID
    ]
    if len(cases) != 1:
        raise ValueError("development registry has no unique CAKE current case")
    case = cases[0]
    if not all((
        registry.get("schema_version") == "m04r-batch-query-registry-v1",
        registry.get("passed") is True,
        registry.get("failures") == [],
        registry.get("real_forward_outcomes_accessed") is False,
        case.get("symbol") == CAKE_SYMBOL,
        case.get("cutoff") == CAKE_CUTOFF,
        case.get("episode_id") == CAKE_EPISODE_ID,
        case.get("dataset_id") == "nasdaq",
        case.get("lookback") == 252,
        case.get("representation_version") == "dense-v1",
    )):
        raise ValueError("development CAKE identity or registry seal differs")
    if case["episode_id"] in frozen_query_ids:
        raise ValueError("preflight query belongs to the untouched v2 registry")
    return registry, case


def _assert_poc_output(output_root: Path, roots: Mapping[str, str]) -> None:
    output = output_root.resolve()
    protected = tuple(Path(value).resolve() for value in roots.values())
    if any(
        output == value or output.is_relative_to(value)
        or value.is_relative_to(output)
        for value in protected
    ):
        raise ValueError("preflight output overlaps a frozen/protected root")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("preflight output must be fresh and empty")


def _publish_bytes_create_only(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError(f"preflight artifact already exists: {path}") from exc
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _augment_case_with_causal_prefixes(
    config_path: Path, case: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = producer._expected_query_context(config_path, case)
    if not all((
        context["query_episode_id"] == CAKE_EPISODE_ID,
        context["query_symbol"] == CAKE_SYMBOL,
    )):
        raise ValueError("rebuilt development CAKE query differs")
    augmented = {
        **dict(case),
        "stock_prefix": context["query_stock_prefix"],
        "benchmark_prefix": context["query_benchmark_prefix"],
    }
    return augmented, context


def _validate_worker_result(
    result: Mapping[str, Any], *, contract: Mapping[str, Any],
    case: Mapping[str, Any], context: Mapping[str, Any],
    resident: Mapping[str, Any], physical_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    semantic = dict(result.get("semantic", {}))
    performance = dict(result.get("performance", {}))
    producer._strict_validate_case_evidence(
        semantic, performance, contract=contract, case=case,
        role=PREFLIGHT_ROLE, expected_query=context, resident=resident,
        physical_rows=physical_rows,
    )
    scans = list(semantic["scan_semantics"])
    candidate_digests = [str(row["candidate_digest"]) for row in scans]
    result_digests = [str(row["result_digest"]) for row in scans]
    ready_start = dict(performance["ready_start"])
    ready_end = dict(performance["ready_end"])
    start_lease = dict(ready_start.get("file_identity_lease", {}))
    end_lease = dict(ready_end.get("file_identity_lease", {}))
    if not all((
        SCAN_PROTOCOL["outer_threads"] == 8,
        SCAN_PROTOCOL["maximum_in_flight_blocks"] == 8,
        candidate_digests == [EXPECTED_CANDIDATE_DIGEST] * 3,
        result_digests == [EXPECTED_SCAN_RESULT_DIGEST] * 3,
        semantic.get("candidate_digest_reconstructed")
        == EXPECTED_CANDIDATE_DIGEST,
        semantic.get("gates") is not None,
        tuple(semantic["gates"]) == SEMANTIC_CASE_GATES,
        all(value is True for value in semantic["gates"].values()),
        semantic.get("passed") is True,
        performance.get("gates") is not None,
        tuple(performance["gates"]) == PERFORMANCE_ATTEMPT_GATES,
        all(value is True for value in performance["gates"].values()),
        performance.get("passed") is True,
        semantic.get("semantic_digest") == semantic_case_digest(semantic),
        performance.get("performance_digest")
        == performance_attempt_digest(performance),
        ready_start == ready_end == resident["resident_ready_observation"],
        ready_start.get("content_digest") == resident["resident_content_digest"],
        start_lease == end_lease,
        start_lease.get("lease_digest") is not None,
        semantic.get("real_forward_outcomes_accessed") is False,
        performance.get("real_forward_outcomes_accessed") is False,
    )):
        raise ValueError("CAKE worker preflight digest, gate or identity differs")
    return semantic, performance


def run_preflight(
    *, repository_root: Path, config_path: Path,
    development_registry_path: Path, case_id: str, source_full_root: Path,
    resident_root: Path, output_root: Path,
) -> dict[str, Any]:
    repository = repository_root.resolve()
    if config_path.resolve() != repository / CONFIG_RELATIVE_PATH:
        raise ValueError("preflight config path differs")
    config = load_config(config_path)
    roots = expected_roots(config.artifact_dir)
    if not all((
        source_full_root.resolve() == Path(roots["source_full_root"]),
        resident_root.resolve() == Path(roots["resident_full_root"]),
    )):
        raise ValueError("preflight source or hardened READY root differs")
    _assert_poc_output(output_root, roots)
    source_pack = producer._source_pack_binding(source_full_root)
    implementation = producer._implementation_manifest()
    environment = producer._environment_manifest()
    frozen_registry = _read_json(Path(roots["registry_root"]) / "query-registry.json")
    preregistration = load_and_validate_preregistration(
        frozen_registry, config.artifact_dir, repository,
        expected_source_pack=source_pack,
        expected_implementation_manifest=implementation,
        expected_environment_manifest=environment,
    )
    contract = dict(preregistration["producer_contract"])
    producer._git_preregistration_binding(
        repository, expected_preregistration_path(repository),
        implementation_files=contract["implementation_manifest"]["files"],
    )
    _, case = _load_development_case(
        repository_root=repository, registry_path=development_registry_path,
        case_id=case_id,
        frozen_query_ids=set(contract["ordered_query_ids"]),
    )
    case, context = _augment_case_with_causal_prefixes(config_path, case)
    ready, validation_observation = prepare_resident_mirror_observed(
        source_full_root / "store", resident_root, FROZEN_GENERATION_ID,
        expected_provenance_digest=source_pack["provenance_digest"],
        reserve_bytes=int(contract["resident_policy"]["reserve_bytes"]),
        validate_existing=True,
    )
    ready_path = resident_root / "READY.json"
    resident = producer._resident_binding(
        ready, contract_digest=contract["contract_digest"],
        ready_path=ready_path,
        validation_observation=validation_observation,
    )
    resident_failures = producer._validate_resident_binding(
        resident, contract, ready,
    )
    if resident_failures:
        raise ValueError(f"preflight resident binding differs:{resident_failures}")
    expected_ready = producer._ready_observation(ready_path)
    context_spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1, mp_context=context_spawn,
    ) as executor:
        result = executor.submit(
            producer._worker, str(config_path), str(resident_root / "store"),
            str(ready_path), case, PREFLIGHT_ROLE,
            dict(contract["route_quotas"]), source_pack["provenance_digest"],
            int(source_pack["physical_rows"]), resident["binding_digest"],
            resident["resident_content_digest"], contract["contract_digest"],
            dict(contract["implementation_manifest"]), expected_ready, resident,
        ).result()
    semantic, performance = _validate_worker_result(
        result, contract=contract, case=case, context=context,
        resident=resident, physical_rows=int(source_pack["physical_rows"]),
    )
    deterministic = {
        "schema_version": PREFLIGHT_SCHEMA,
        "scope": "already-exposed CAKE current development query only",
        "registry_case_id": CAKE_CASE_ID,
        "query_episode_id": CAKE_EPISODE_ID,
        "query_symbol": CAKE_SYMBOL,
        "query_cutoff": CAKE_CUTOFF,
        "preregistration_digest": preregistration["preregistration_digest"],
        "producer_contract_digest": contract["contract_digest"],
        "implementation_manifest_digest": implementation["digest"],
        "environment_manifest_digest": environment["digest"],
        "generation_id": FROZEN_GENERATION_ID,
        "resident_binding_digest": resident["binding_digest"],
        "resident_content_digest": resident["resident_content_digest"],
        "scan_protocol": dict(contract["scan_protocol"]),
        "expected_candidate_digest": EXPECTED_CANDIDATE_DIGEST,
        "expected_scan_result_digest": EXPECTED_SCAN_RESULT_DIGEST,
        "semantic": semantic,
        "performance": performance,
        "all_semantic_gates_passed": True,
        "all_performance_gates_passed": True,
        "content_and_identity_lease_stable": True,
        "passed": True,
        "untouched_registry_queries_executed": 0,
        "authority_files_opened": False,
        "real_forward_outcomes_accessed": False,
    }
    evidence = {
        **deterministic, "created_at": datetime.now(timezone.utc).isoformat(),
        "result_digest": stable_hash(deterministic),
    }
    json_path = output_root / "preflight.json"
    html_path = output_root / "preflight.html"
    _publish_bytes_create_only(
        json_path,
        (json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    html = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-11 candidate v2 CAKE preflight</title></head><body>"
        "<h1>PASS — exposed CAKE development preflight</h1><p>No untouched "
        "query or authority result was opened.</p><pre>"
        f"{escape(json.dumps(evidence, indent=2, sort_keys=True))}"
        "</pre></body></html>"
    )
    _publish_bytes_create_only(html_path, html.encode())
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-registry", type=Path, required=True)
    parser.add_argument("--case-id", default=CAKE_CASE_ID)
    parser.add_argument("--source-full-root", type=Path, required=True)
    parser.add_argument("--resident-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    evidence = run_preflight(
        repository_root=Path(__file__).resolve().parents[2],
        config_path=args.config,
        development_registry_path=args.development_registry,
        case_id=args.case_id, source_full_root=args.source_full_root,
        resident_root=args.resident_root, output_root=args.output_root,
    )
    print(json.dumps({
        "passed": evidence["passed"],
        "result_digest": evidence["result_digest"],
        "output_root": str(args.output_root.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

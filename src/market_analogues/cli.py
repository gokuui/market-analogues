from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from .adapters import source_from_spec
from .authority import (
    AuthorityError, authority_universe_digest, run_authority_case,
    validate_authority_artifact, write_authority_report,
)
from .benchmark import (
    available_methods, benchmark_synthetic, benchmark_ucr, results_frame,
    write_comparison_report,
)
from .config import AppConfig, load_config
from .data_ledger import (
    build_data_ledger, load_availability_declaration, write_data_ledger_artifacts,
)
from .episodes import build_episode, build_manifest
from .exact_storage_feasibility import (
    verify_exact_storage_feasibility, write_exact_storage_report,
)
from .external_examples import (
    KULLAMAGI_POSITIONS_URL, analyze_kullamagi_examples,
    download_kullamagi_positions, parse_kullamagi_positions,
    write_external_example_artifacts,
)
from .exhaustive import (
    build_exact_frontier, exhaustive_frontier_search, write_exhaustive_report,
)
from .exact_batch_verification import verify_exact_batch_kernel, write_exact_batch_report
from .gates import GateReport, require_passed
from .fusion_verification import verify_candidate_fusion, write_fusion_report
from .gate12_registry import (
    build_gate12_registry, validate_gate12_registry, write_gate12_registry,
)
from .index import CoarseIndex
from .matrix_runner import run_authority_matrix
from .multiresolution_verification import (
    verify_multiresolution_state, write_multiresolution_verification,
)
from .outcomes import compute_outcomes, summarize_match_outcomes
from .oracle import run_oracle_suite, write_oracle_artifacts
from .pruning_verification import verify_exact_safe_pruning, write_pruning_report
from .production_verification import (
    verify_production_search, write_production_search_report,
)
from .precision_verification import (
    verify_float16_oracle_precision, write_precision_report,
)
from .product_contract import (
    load_product_contract, validate_trial_ledger, write_contract_artifacts,
)
from .quality import audit_source
from .report import write_search_report
from .representation import represent
from .search import SearchCandidate, exact_search
from .scale_ladder import (
    collect_scale_history, run_scale_ladder, write_scale_ladder_report,
)
from .scan import streaming_search
from .structural_verification import (
    load_structural_verifier_spec, verify_latent_structures,
    write_structural_verification,
)
from .structural_verification_v2 import (
    load_structural_verifier_v2_spec, verify_latent_structures_v2,
    write_structural_verification_v2,
)
from .types import InstrumentKey, SearchQuery
from .universe import verify_universe, write_universe_report
from .verification import run_synthetic_verifier, write_verification_gate
from .view_search import persisted_exact_search
from .view_store import build_view_store
from .yahoo_examples import fetch_yahoo_examples, write_external_example_workbook


def _load(args: argparse.Namespace) -> tuple[AppConfig, object]:
    config = load_config(args.config)
    if args.dataset not in config.datasets:
        raise SystemExit(f"unknown dataset {args.dataset!r}; choose from {sorted(config.datasets)}")
    return config, source_from_spec(config.datasets[args.dataset])


def _gates(config: AppConfig) -> Path:
    return config.artifact_dir / "gates"


def cmd_audit(args: argparse.Namespace) -> int:
    config, source = _load(args)
    output = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    started = perf_counter()
    frame = audit_source(source, output)
    counts = frame.tier.value_counts().to_dict()
    GateReport(f"01_ingestion_audit_{args.dataset}", True, {
        "dataset": args.dataset, "instruments": len(frame),
        "tier_A": int(counts.get("A", 0)), "tier_B": int(counts.get("B", 0)),
        "quarantined": int(counts.get("QUARANTINED", 0)),
        "elapsed_seconds": perf_counter() - started,
    }).write(_gates(config))
    print(output)
    return 0


def cmd_build_episodes(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"01_ingestion_audit_{args.dataset}")
    quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    quality = pd.read_parquet(quality_path) if quality_path.exists() else None
    output = config.artifact_dir / "episodes" / f"{args.dataset}.parquet"
    frame = build_manifest(
        source, config.representation_version, quality,
        stride=config.candidate_stride_bars, lookbacks=config.lookbacks,
        instrument_limit=args.instrument_limit, output=output,
    )
    GateReport(f"02_episode_manifest_{args.dataset}", bool(len(frame)), {
        "dataset": args.dataset, "episodes": len(frame),
        "lookbacks": sorted(frame.lookback.unique().tolist()) if len(frame) else [],
    }, [] if len(frame) else ["manifest is empty"]).write(_gates(config))
    print(output)
    return 0 if len(frame) else 2


def cmd_verify(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = run_synthetic_verifier(args.seeds_per_family)
    path = write_verification_gate(report, _gates(config))
    print(json.dumps(report.to_dict(), indent=2))
    print(path)
    return 0 if report.passed else 2


def cmd_verify_case_memory_contract(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    contract = load_product_contract(args.contract)
    declared_ledger = Path(contract.payload["evaluation"]["trial_ledger"])
    ledger_path = Path(args.trial_ledger).resolve() if args.trial_ledger else (
        declared_ledger if declared_ledger.is_absolute()
        else (contract.source.parent / declared_ledger).resolve()
    )
    ledger = validate_trial_ledger(ledger_path, contract)
    machine_path = Path(args.machine_output) if args.machine_output else (
        config.artifact_dir / "contracts" / f"{contract.contract_id}.json"
    )
    html_path = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / f"m00-{contract.contract_id}.html"
    )
    write_contract_artifacts(contract, ledger, machine_path, html_path)
    metrics = {
        "schema_version": contract.payload["schema_version"],
        "contract_id": contract.contract_id,
        "contract_digest": contract.digest,
        "status": contract.payload["status"],
        "primary_mode": contract.payload["decision"]["primary_mode"],
        "entry_open_enabled": contract.payload["decision"]["modes"]["entry_open"]["enabled"],
        "intraday_enabled": contract.payload["decision"]["modes"]["intraday"]["enabled"],
        "outcomes_may_affect_similarity": contract.payload["outcomes"]["outcomes_may_affect_similarity"],
        "trial_count": len(ledger["trials"]),
        "machine_artifact": str(machine_path.resolve()),
        "html_artifact": str(html_path.resolve()),
    }
    GateReport(
        "m00_case_memory_contract", True, metrics,
        source_hashes={
            "contract": contract.digest,
            "trial_ledger": sha256(ledger_path.read_bytes()).hexdigest(),
        },
    ).write(_gates(config))
    print(json.dumps({"passed": True, **metrics}, indent=2))
    print(html_path)
    return 0


def cmd_build_data_ledger(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    require_passed(_gates(config), "m00_case_memory_contract")
    declaration = load_availability_declaration(args.availability)
    results = []
    source_hashes = {"availability_declaration": declaration.digest}
    for dataset_id in args.datasets:
        if dataset_id not in config.datasets:
            raise SystemExit(f"unknown dataset {dataset_id!r}; choose from {sorted(config.datasets)}")
        quality_path = config.artifact_dir / "quality" / f"{dataset_id}.parquet"
        if not quality_path.exists():
            raise SystemExit(f"missing quality audit: {quality_path}")
        quality = pd.read_parquet(quality_path)
        source = source_from_spec(config.datasets[dataset_id])
        results.append(build_data_ledger(
            config.datasets[dataset_id], source, quality, declaration,
            workers=args.workers,
        ))
        source_hashes[f"quality_{dataset_id}"] = sha256(quality_path.read_bytes()).hexdigest()
    output_dir = Path(args.output_dir) if args.output_dir else config.artifact_dir / "data-ledger"
    machine_path, html_path, parquet_paths = write_data_ledger_artifacts(
        results, declaration, output_dir,
    )
    passed = bool(results) and all(result.passed for result in results)
    metrics = {
        "schema_version": "point-in-time-data-ledger-v1",
        "availability_digest": declaration.digest,
        "universe_boundary": declaration.universe_boundary,
        "datasets": {
            str(result.metrics["dataset"]): {
                key: result.metrics[key] for key in (
                    "source_instruments", "accounted_instruments", "usable_instruments",
                    "quarantined_instruments", "fingerprints_verified",
                    "fingerprints_unverified_load_error", "fingerprint_mismatches",
                    "first_source_timestamp", "last_source_timestamp",
                    "freshness_age_calendar_days", "freshness_status",
                    "current_usable_instruments_within_7_days",
                    "current_coverage_fraction_of_usable",
                    "current_after_close_analysis_available", "elapsed_seconds",
                    "instrument_ledger_digest",
                )
            }
            for result in results
        },
        "machine_artifact": str(machine_path.resolve()),
        "html_artifact": str(html_path.resolve()),
        "instrument_artifacts": [str(path.resolve()) for path in parquet_paths],
    }
    failures = [
        f"{result.metrics['dataset']}:{failure}"
        for result in results for failure in result.failures
    ]
    GateReport(
        "m01_point_in_time_data_ledger", passed, metrics, failures,
        source_hashes=source_hashes,
    ).write(_gates(config))
    print(json.dumps({"passed": passed, **metrics, "failures": failures}, indent=2))
    print(html_path)
    return 0 if passed else 2


def cmd_verify_multiresolution_state(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    require_passed(_gates(config), "m01_point_in_time_data_ledger")
    real_inputs = []
    source_hashes: dict[str, str] = {}
    registry_root = Path(args.registry_root) if args.registry_root else (
        config.artifact_dir / "gate12"
    )
    for dataset_id in args.datasets:
        if dataset_id not in config.datasets:
            raise SystemExit(f"unknown dataset {dataset_id!r}; choose from {sorted(config.datasets)}")
        quality_path = config.artifact_dir / "quality" / f"{dataset_id}.parquet"
        registry_path = registry_root / dataset_id / "query-registry.parquet"
        for required in (quality_path, registry_path):
            if not required.exists():
                raise SystemExit(f"missing M02 input: {required}")
        quality = pd.read_parquet(quality_path)
        registry = pd.read_parquet(registry_path)
        source = source_from_spec(config.datasets[dataset_id])
        real_inputs.append((dataset_id, source, quality, registry))
        source_hashes[f"quality_{dataset_id}"] = sha256(quality_path.read_bytes()).hexdigest()
        source_hashes[f"registry_{dataset_id}"] = sha256(registry_path.read_bytes()).hexdigest()
    result = verify_multiresolution_state(
        real_inputs,
        maximum_total_seconds=args.maximum_total_seconds,
        maximum_case_seconds=args.maximum_case_seconds,
        maximum_rss_mb=args.maximum_rss_mb,
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        config.artifact_dir / "m02-multiresolution"
    )
    machine_path, html_path, field_path = write_multiresolution_verification(
        result, output_dir,
    )
    GateReport(
        "m02_multiresolution_chart_state", result.passed,
        {
            **result.metrics,
            "machine_artifact": str(machine_path.resolve()),
            "html_artifact": str(html_path.resolve()),
            "field_contract_artifact": str(field_path.resolve()),
        },
        list(result.failures), source_hashes=source_hashes,
    ).write(_gates(config))
    print(json.dumps({"passed": result.passed, **result.metrics, "failures": result.failures}, indent=2))
    print(html_path)
    return 0 if result.passed else 2


def cmd_verify_latent_structures(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    require_passed(_gates(config), "m02_multiresolution_chart_state")
    spec = load_structural_verifier_spec(args.verifier)
    result = verify_latent_structures(spec)
    output_dir = Path(args.output_dir) if args.output_dir else (
        config.artifact_dir / "m03-latent-structures"
    )
    machine_path, html_path, distance_path = write_structural_verification(
        result, spec, output_dir,
    )
    GateReport(
        "m03_latent_structural_retrieval", result.passed,
        {
            **result.metrics,
            "machine_artifact": str(machine_path.resolve()),
            "html_artifact": str(html_path.resolve()),
            "distance_contract_artifact": str(distance_path.resolve()),
        },
        list(result.failures), source_hashes={"verifier_config": spec.digest},
    ).write(_gates(config))
    print(json.dumps({"passed": result.passed, **result.metrics, "failures": result.failures}, indent=2))
    print(html_path)
    return 0 if result.passed else 2


def cmd_verify_latent_structures_v2(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    require_passed(_gates(config), "m02_multiresolution_chart_state")
    spec = load_structural_verifier_v2_spec(args.verifier)
    result = verify_latent_structures_v2(spec)
    output_dir = Path(args.output_dir) if args.output_dir else (
        config.artifact_dir / "m03b-latent-structures"
    )
    machine_path, html_path, distance_path = write_structural_verification_v2(
        result, spec, output_dir,
    )
    GateReport(
        "m03b_latent_structural_retrieval", result.passed,
        {
            **result.metrics,
            "machine_artifact": str(machine_path.resolve()),
            "html_artifact": str(html_path.resolve()),
            "distance_contract_artifact": str(distance_path.resolve()),
        },
        list(result.failures), source_hashes={"verifier_config": spec.digest},
    ).write(_gates(config))
    print(json.dumps({"passed": result.passed, **result.metrics, "failures": result.failures}, indent=2))
    print(html_path)
    return 0 if result.passed else 2


def cmd_compare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    methods = available_methods(True, include_slow=args.include_slow)
    results = benchmark_synthetic(args.seeds_per_family, methods)
    if not args.skip_ucr:
        if importlib.util.find_spec("tslearn") is None:
            raise SystemExit("UCR comparison requires the bench extra: pip install '.[bench]'")
        results.extend(benchmark_ucr(args.ucr_dataset, args.ucr_test_limit, methods))
    frame = results_frame(results)
    output = Path(args.output) if args.output else config.artifact_dir / "reports" / "method-comparison.html"
    csv_path = output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    write_comparison_report(frame, output)
    print(frame.to_string(index=False))
    print(output)
    return 0


def cmd_verify_universe(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"01_ingestion_audit_{args.dataset}")
    require_passed(_gates(config), "03_synthetic_retrieval")
    quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    if not quality_path.exists():
        raise SystemExit(f"missing quality audit: {quality_path}")
    quality = pd.read_parquet(quality_path)
    query = build_episode(
        source, InstrumentKey(args.dataset, args.symbol), args.cutoff,
        args.lookback, config.representation_version,
    )
    request = SearchQuery(
        query.key, (args.dataset,), ("A", "B"), args.top_k,
        minimum_history_gap_bars=args.minimum_history_gap,
    )
    result = verify_universe(
        query, source, request, quality, stride=args.stride,
        reference_pool=args.reference_pool,
        comparison_pools=tuple(args.comparison_pools),
        per_instrument=args.per_instrument, scan_backend=args.scan_backend,
        workers=args.workers, max_seconds=args.max_seconds,
        max_rss_mb=args.max_rss_mb, minimum_pool_recall=args.minimum_pool_recall,
        recall_pool=args.recall_pool,
        candidate_strategy=args.candidate_strategy,
    )
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / f"universe-verification-{args.dataset}.html"
    )
    write_universe_report(result, output)
    GateReport(
        f"08_universe_verification_{args.dataset}", result.passed,
        result.metrics, list(result.failures),
    ).write(_gates(config))
    print(json.dumps({"passed": result.passed, **result.metrics, "failures": result.failures}, indent=2))
    print(output)
    return 0 if result.passed else 2


def cmd_verify_oracle(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"01_ingestion_audit_{args.dataset}")
    require_passed(_gates(config), "03_synthetic_retrieval")
    quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    if not quality_path.exists():
        raise SystemExit(f"missing quality audit: {quality_path}")
    quality = pd.read_parquet(quality_path)
    result = run_oracle_suite(
        source, quality, tuple(args.symbols), lookbacks=tuple(args.lookbacks),
        cutoff_quantiles=tuple(args.cutoff_quantiles),
        per_stratum=args.instruments_per_stratum,
        windows_per_instrument=args.windows_per_instrument,
        top_k=args.top_k, coarse_pools=tuple(args.coarse_pools),
        workers=args.workers, minimum_candidates=args.minimum_candidates,
        representation_version=config.representation_version,
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        config.artifact_dir / "oracles" / args.dataset
    )
    report = write_oracle_artifacts(result, output_dir)
    gate_metrics = {
        **result.metrics,
        "case_metrics": {case.case_id: case.metrics for case in result.cases},
    }
    GateReport(
        f"09_exhaustive_oracle_{args.dataset}", result.passed,
        gate_metrics, list(result.failures),
    ).write(_gates(config))
    print(json.dumps({"passed": result.passed, **gate_metrics, "failures": result.failures}, indent=2))
    print(report)
    return 0 if result.passed else 2


def cmd_verify_fusion(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"09_exhaustive_oracle_{args.dataset}")
    oracle_directory = Path(args.oracle_dir) if args.oracle_dir else (
        config.artifact_dir / "oracles" / args.dataset
    )
    result = verify_candidate_fusion(
        source, oracle_directory,
        representation_version=config.representation_version,
        pool_sizes=tuple(args.pool_sizes), acceptance_pool=args.acceptance_pool,
        minimum_recall=args.minimum_recall,
        per_instrument_view=args.per_instrument_view,
        view_mode=args.view_mode,
    )
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / f"{args.view_mode}-fusion-{args.dataset}.html"
    )
    write_fusion_report(result, output)
    gate_prefix = "10_multiview_fusion" if args.view_mode == "cheap" else "11a_signature_fusion"
    GateReport(
        f"{gate_prefix}_{args.dataset}", result.passed,
        {**result.metrics, "case_metrics": {
            row.case_id: {
                "candidates": int(row.candidates),
                "local_union_candidates": int(row.local_union_candidates),
                "oracle_matches": int(row.oracle_matches),
                "pool_recall": row.pool_recall,
                "seconds": float(row.seconds),
            } for row in result.cases.itertuples(index=False)
        }},
        list(result.failures),
    ).write(_gates(config))
    print(json.dumps({"passed": result.passed, **result.metrics,
                      "failures": result.failures}, indent=2))
    print(output)
    return 0 if result.passed else 2


def cmd_verify_pruning(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"09_exhaustive_oracle_{args.dataset}")
    oracle_directory = Path(args.oracle_dir) if args.oracle_dir else (
        config.artifact_dir / "oracles" / args.dataset
    )
    result = verify_exact_safe_pruning(
        source, oracle_directory,
        representation_version=config.representation_version,
        tolerance=args.tolerance,
    )
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / f"exact-safe-pruning-{args.dataset}.html"
    )
    write_pruning_report(result, output)
    GateReport(
        f"11b_exact_safe_pruning_{args.dataset}", result.passed,
        {**result.metrics, "case_metrics": {
            row.case_id: {
                "eligible_candidates": int(row.eligible_candidates),
                "oracle_matches": int(row.oracle_matches),
                "default_exact_evaluated": int(row.default_exact_evaluated),
                "default_safely_pruned": int(row.default_safely_pruned),
                "default_seconds": float(row.default_seconds),
                "bound_exact_evaluated": int(row.bound_exact_evaluated),
                "bound_safely_pruned": int(row.bound_safely_pruned),
                "bound_seconds": float(row.bound_seconds),
                "maximum_distance_delta": float(row.maximum_distance_delta),
                "maximum_component_delta": float(row.maximum_component_delta),
            } for row in result.cases.itertuples(index=False)
        }},
        list(result.failures),
    ).write(_gates(config))
    print(json.dumps({
        "passed": result.passed, **result.metrics, "failures": result.failures,
    }, indent=2))
    print(output)
    return 0 if result.passed else 2


def cmd_verify_float16_precision(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"09_exhaustive_oracle_{args.dataset}")
    oracle_directory = Path(args.oracle_dir) if args.oracle_dir else (
        config.artifact_dir / "oracles" / args.dataset
    )
    result = verify_float16_oracle_precision(
        source, oracle_directory,
        representation_version=config.representation_version,
        minimum_recall=args.minimum_recall,
        minimum_ndcg=args.minimum_ndcg,
    )
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / f"float16-precision-{args.dataset}.html"
    )
    write_precision_report(result, output)
    GateReport(
        f"12q_float16_precision_{args.dataset}", result.passed,
        {**result.metrics, "case_metrics": result.cases.to_dict(orient="records")},
        list(result.failures),
    ).write(_gates(config))
    print(json.dumps({"passed": result.passed, **result.metrics,
                      "failures": result.failures}, indent=2))
    print(output)
    return 0 if result.passed else 2


def cmd_verify_production_search(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(
        _gates(config), f"11a_view_shards_{args.dataset}_{args.lookback}",
    )
    require_passed(_gates(config), f"11b_exact_safe_pruning_{args.dataset}")
    quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    if not quality_path.exists():
        raise SystemExit(f"missing quality audit: {quality_path}")
    quality = pd.read_parquet(quality_path)
    query = build_episode(
        source, InstrumentKey(args.dataset, args.symbol), args.cutoff,
        args.lookback, config.representation_version,
    )
    request = SearchQuery(
        query.key, (args.dataset,), ("A", "B"), args.top_k,
        minimum_history_gap_bars=args.minimum_history_gap,
    )
    view_root = Path(args.view_store_root) if args.view_store_root else (
        config.artifact_dir / "view-store"
    )
    result = verify_production_search(
        query, source, request, view_root, quality=quality,
        candidate_pool=args.candidate_pool,
        per_instrument_view=args.per_instrument_view,
        workers=args.workers, repeat=args.repeat,
        use_dtw_bound=args.lb_keogh,
        max_seconds=args.max_seconds, max_rss_mb=args.max_rss_mb,
        tolerance=args.tolerance,
    )
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports"
        / f"production-search-{args.dataset}-{args.symbol}-{args.lookback}.html"
    )
    write_production_search_report(result, output)
    GateReport(
        f"11c_production_search_{args.dataset}_{args.lookback}", result.passed,
        {**result.metrics, "run_metrics": {
            str(int(row.run)): {
                key: value for key, value in row._asdict().items() if key != "run"
            } for row in result.runs.itertuples(index=False)
        }},
        list(result.failures),
    ).write(_gates(config))
    print(json.dumps({
        "passed": result.passed, **result.metrics, "failures": result.failures,
    }, indent=2))
    print(output)
    return 0 if result.passed else 2


def cmd_build_gate12_registry(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"09_exhaustive_oracle_{args.dataset}")
    oracle_directory = Path(args.oracle_dir) if args.oracle_dir else (
        config.artifact_dir / "oracles" / args.dataset
    )
    quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    if not quality_path.exists():
        raise SystemExit(f"missing quality audit: {quality_path}")
    quality = pd.read_parquet(quality_path)
    result = build_gate12_registry(
        source, oracle_directory, seed=args.seed, lookback=args.lookback,
        historical_quantile=args.historical_quantile,
        minimum_rows=args.minimum_rows,
        minimum_future_sessions=args.minimum_future_sessions,
        representation_version=config.representation_version,
        quality=quality,
        maximum_staleness_days=args.maximum_staleness_days,
    )
    output_directory = Path(args.output_dir) if args.output_dir else (
        config.artifact_dir / "gate12" / args.dataset
    )
    yaml_path, parquet_path, html_path = write_gate12_registry(
        result, output_directory,
    )
    GateReport(
        f"12a_query_registry_{args.dataset}", result.passed,
        {**result.metrics, "yaml": str(yaml_path), "parquet": str(parquet_path)},
        list(result.failures),
    ).write(_gates(config))
    print(json.dumps({
        "passed": result.passed, **result.metrics, "failures": result.failures,
    }, indent=2))
    print(html_path)
    return 0 if result.passed else 2


def _validated_view_manifest_rows(
    config: AppConfig,
    dataset_id: str,
    lookbacks: set[int],
    benchmark_fingerprint: str | None,
) -> int:
    view_gate_names = [
        f"11a_view_shards_{dataset_id}_{lookback}" for lookback in sorted(lookbacks)
    ]
    for gate_name in view_gate_names:
        require_passed(_gates(config), gate_name)
    path = config.artifact_dir / "view-store" / dataset_id / "manifest.json"
    if not path.exists():
        raise SystemExit(f"missing view-store manifest: {path}")
    payload = json.loads(path.read_text())
    if payload.get("failures"):
        raise SystemExit(f"view-store manifest contains failures: {path}")
    if payload.get("dataset_id") != dataset_id:
        raise SystemExit(f"view-store manifest dataset mismatch: {path}")
    if payload.get("representation_version") != config.representation_version:
        raise SystemExit(f"view-store representation version is stale: {path}")
    if payload.get("benchmark_fingerprint") != benchmark_fingerprint:
        raise SystemExit(f"view-store benchmark fingerprint is stale: {path}")
    shards = payload.get("shards")
    if not isinstance(shards, list):
        raise SystemExit(f"view-store manifest has no shard records: {path}")
    digest_payload = json.dumps(shards, sort_keys=True, separators=(",", ":"))
    digest = sha256(digest_payload.encode()).hexdigest()
    if digest != payload.get("manifest_digest"):
        raise SystemExit(f"view-store manifest digest mismatch: {path}")
    for gate_name in view_gate_names:
        gate = json.loads((_gates(config) / f"{gate_name}.json").read_text())
        if gate.get("metrics", {}).get("manifest_digest") != digest:
            raise SystemExit(f"view-store manifest changed after gate {gate_name}")
    eligible = [record for record in shards if int(record["lookback"]) in lookbacks]
    if {int(record["lookback"]) for record in eligible} != lookbacks:
        raise SystemExit(
            f"view-store manifest does not cover lookbacks {sorted(lookbacks)}: {path}"
        )
    if any(record.get("dataset_id") != dataset_id for record in eligible):
        raise SystemExit(f"view-store shard dataset mismatch: {path}")
    return sum(int(record["rows"]) for record in eligible)


def cmd_verify_exact_storage(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    unknown = set(args.datasets).difference(config.datasets)
    if unknown:
        raise SystemExit(
            f"unknown datasets {sorted(unknown)}; choose from {sorted(config.datasets)}"
        )
    sources = {}
    registries = {}
    total_rows = 0
    for dataset_id in args.datasets:
        require_passed(_gates(config), f"12a_query_registry_{dataset_id}")
        source = source_from_spec(config.datasets[dataset_id])
        registry_directory = config.artifact_dir / "gate12" / dataset_id
        yaml_path = registry_directory / "query-registry.yaml"
        parquet_path = registry_directory / "query-registry.parquet"
        quality_path = config.artifact_dir / "quality" / f"{dataset_id}.parquet"
        for required in (yaml_path, parquet_path, quality_path):
            if not required.exists():
                raise SystemExit(f"missing required Gate 12 input: {required}")
        quality = pd.read_parquet(quality_path)
        registry_failures = validate_gate12_registry(source, yaml_path, quality)
        if registry_failures:
            raise SystemExit(
                f"Gate 12 registry validation failed for {dataset_id}: "
                + "; ".join(registry_failures)
            )
        yaml_registry = pd.DataFrame((yaml.safe_load(yaml_path.read_text()) or {})["cases_data"])
        parquet_registry = pd.read_parquet(parquet_path)
        try:
            pd.testing.assert_frame_equal(
                yaml_registry.reset_index(drop=True),
                parquet_registry.reset_index(drop=True),
                check_dtype=False,
            )
        except AssertionError as exc:
            raise SystemExit(
                f"Gate 12 YAML/Parquet registry mismatch for {dataset_id}: {exc}"
            ) from exc
        lookbacks = {int(value) for value in parquet_registry.lookback.unique()}
        total_rows += _validated_view_manifest_rows(
            config, dataset_id, lookbacks, source.benchmark_fingerprint(),
        )
        sources[dataset_id] = source
        registries[dataset_id] = parquet_registry
    result = verify_exact_storage_feasibility(
        sources, registries, total_universe_rows=total_rows,
        sample_windows_per_symbol=args.sample_windows_per_symbol,
        comparison_queries=args.comparison_queries,
        comparison_candidates=args.comparison_candidates,
        tolerance=args.tolerance,
        disk_path=config.artifact_dir,
        disk_reserve_bytes=int(args.disk_reserve_gb * 1024 ** 3),
        frontier_bytes_per_row=args.frontier_bytes_per_row,
        compression_safety_factor=args.compression_safety_factor,
    )
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / "exact-storage-feasibility.html"
    )
    write_exact_storage_report(result, output)
    GateReport(
        "12b_exact_storage_feasibility", result.passed,
        {**result.metrics, "layouts": result.layouts.to_dict(orient="records")},
        list(result.failures),
    ).write(_gates(config))
    print(json.dumps({
        "passed": result.passed, **result.metrics,
        "layouts": result.layouts.to_dict(orient="records"),
        "failures": result.failures,
    }, indent=2))
    print(output)
    return 0 if result.passed else 2


def cmd_verify_exact_batch(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    require_passed(_gates(config), "12b_exact_storage_feasibility")
    unknown = set(args.datasets).difference(config.datasets)
    if unknown:
        raise SystemExit(f"unknown datasets {sorted(unknown)}")
    sources = {}
    registries = {}
    for dataset_id in args.datasets:
        require_passed(_gates(config), f"12a_query_registry_{dataset_id}")
        source = source_from_spec(config.datasets[dataset_id])
        directory = config.artifact_dir / "gate12" / dataset_id
        yaml_path = directory / "query-registry.yaml"
        parquet_path = directory / "query-registry.parquet"
        quality_path = config.artifact_dir / "quality" / f"{dataset_id}.parquet"
        quality = pd.read_parquet(quality_path)
        failures = validate_gate12_registry(source, yaml_path, quality)
        if failures:
            raise SystemExit("; ".join(failures))
        yaml_registry = pd.DataFrame((yaml.safe_load(yaml_path.read_text()) or {})["cases_data"])
        parquet_registry = pd.read_parquet(parquet_path)
        try:
            pd.testing.assert_frame_equal(
                yaml_registry.reset_index(drop=True),
                parquet_registry.reset_index(drop=True), check_dtype=False,
            )
        except AssertionError as exc:
            raise SystemExit(f"Gate 12 registry copies disagree: {exc}") from exc
        sources[dataset_id] = source
        registries[dataset_id] = parquet_registry
    result = verify_exact_batch_kernel(
        sources, registries, windows_per_symbol=args.windows_per_symbol,
        stride=args.stride, batch_size=args.batch_size,
        tolerance=args.tolerance, minimum_speedup=args.minimum_speedup,
        maximum_rss_mb=args.maximum_rss_mb,
    )
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / "exact-batch-kernel.html"
    )
    write_exact_batch_report(result, output)
    GateReport(
        "12c_exact_batch_kernel", result.passed,
        {**result.metrics, "case_metrics": result.cases.to_dict(orient="records")},
        list(result.failures),
    ).write(_gates(config))
    print(json.dumps({"passed": result.passed, **result.metrics,
                      "failures": result.failures}, indent=2))
    print(output)
    return 0 if result.passed else 2


def cmd_verify_exhaustive_frontier(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), "12c_exact_batch_kernel")
    require_passed(_gates(config), f"12a_query_registry_{args.dataset}")
    quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    quality = pd.read_parquet(quality_path)
    registry_directory = config.artifact_dir / "gate12" / args.dataset
    registry_yaml = registry_directory / "query-registry.yaml"
    registry_failures = validate_gate12_registry(source, registry_yaml, quality)
    if registry_failures:
        raise SystemExit("; ".join(registry_failures))
    yaml_registry = pd.DataFrame(
        (yaml.safe_load(registry_yaml.read_text()) or {})["cases_data"]
    )
    registry = pd.read_parquet(registry_directory / "query-registry.parquet")
    try:
        pd.testing.assert_frame_equal(
            yaml_registry.reset_index(drop=True),
            registry.reset_index(drop=True), check_dtype=False,
        )
    except AssertionError as exc:
        raise SystemExit(f"Gate 12 registry copies disagree: {exc}") from exc
    query = build_episode(
        source, InstrumentKey(args.dataset, args.symbol), args.cutoff,
        args.lookback, config.representation_version,
    )
    if query.key.id not in set(registry.episode_id.astype(str)):
        raise SystemExit("exhaustive verification query is not in the frozen Gate 12 registry")
    request = SearchQuery(
        query.key, (args.dataset,), ("A", "B"), args.top_k,
        max_per_instrument=args.max_per_instrument,
        minimum_history_gap_bars=args.minimum_history_gap,
    )
    root = Path(args.frontier_root) if args.frontier_root else (
        config.artifact_dir / "gate12" / "frontiers"
    )
    build = build_exact_frontier(
        query, source, request, quality, root, stride=args.stride,
        batch_size=args.batch_size, instrument_limit=args.instrument_limit,
        rebuild_invalid=args.rebuild_invalid,
    )
    resume = build_exact_frontier(
        query, source, request, quality, root, stride=args.stride,
        batch_size=args.batch_size, instrument_limit=args.instrument_limit,
    )
    failures = list(build.failures) + list(resume.failures)
    result = exhaustive_frontier_search(
        query, source, request, root, quality=quality, tolerance=args.tolerance,
        frontier_batch_rows=args.frontier_batch_rows,
        representation_cache_shards=args.representation_cache_shards,
    )
    repeated = exhaustive_frontier_search(
        query, source, request, root, quality=quality, tolerance=args.tolerance,
        frontier_batch_rows=args.frontier_batch_rows,
        representation_cache_shards=args.representation_cache_shards,
    )
    ids = [match.episode_key.id for match in result.matches]
    repeated_ids = [match.episode_key.id for match in repeated.matches]
    totals = [match.total_distance for match in result.matches]
    repeated_totals = [match.total_distance for match in repeated.matches]
    if ids != repeated_ids:
        failures.append("repeated exhaustive result IDs differ")
    maximum_repeat_delta = max(
        (abs(left - right) for left, right in zip(totals, repeated_totals)),
        default=0.0,
    )
    if maximum_repeat_delta > args.tolerance:
        failures.append(f"repeated total delta {maximum_repeat_delta:.3e} exceeds tolerance")
    if len(ids) != args.top_k:
        failures.append(f"returned {len(ids)} matches; require {args.top_k}")
    certificate = result.certificate
    if certificate.exact_evaluated + certificate.safely_pruned != certificate.eligible_candidates:
        failures.append("certificate accounting does not reconcile")
    if not certificate.stopped_early:
        failures.append("bounded smoke did not establish an early stopping frontier")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if rss > args.maximum_rss_mb:
        failures.append(f"peak RSS {rss:.2f} MB exceeds {args.maximum_rss_mb:.2f} MB")
    metrics = {
        "dataset": args.dataset, "symbol": args.symbol,
        "cutoff": query.key.cutoff.isoformat(), "lookback": args.lookback,
        "instrument_limit": args.instrument_limit,
        "frontier_batch_rows": args.frontier_batch_rows,
        "representation_cache_shards": args.representation_cache_shards,
        "manifest_digest": build.manifest_digest,
        "instruments_considered": build.instruments_considered,
        "instruments_built": build.instruments_built,
        "instruments_reused": build.instruments_reused,
        "quality_skipped": build.quality_skipped,
        "resume_instruments_reused": resume.instruments_reused,
        "eligible_candidates": certificate.eligible_candidates,
        "exact_evaluated": certificate.exact_evaluated,
        "safely_pruned": certificate.safely_pruned,
        "prune_fraction": (
            certificate.safely_pruned / certificate.eligible_candidates
            if certificate.eligible_candidates else 0.0
        ),
        "matches": len(ids), "stopped_early": certificate.stopped_early,
        "stop_threshold": (
            certificate.stop_threshold if np.isfinite(certificate.stop_threshold) else None
        ),
        "next_lower_bound": certificate.next_lower_bound,
        "maximum_recomputed_bound_delta": certificate.maximum_recomputed_bound_delta,
        "maximum_repeat_delta": maximum_repeat_delta,
        "build_seconds": build.seconds, "resume_seconds": resume.seconds,
        "search_seconds": certificate.elapsed_seconds,
        "repeat_search_seconds": repeated.certificate.elapsed_seconds,
        "peak_rss_mb": rss,
    }
    passed = not failures
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports"
        / f"exhaustive-frontier-{args.dataset}-{args.symbol}-{args.lookback}.html"
    )
    write_exhaustive_report(
        build, resume, result, output, passed=passed,
        failures=failures, metrics=metrics,
    )
    scope = "smoke" if args.instrument_limit is not None else "full"
    GateReport(
        f"12d_exhaustive_frontier_{scope}_{args.dataset}", passed,
        metrics, failures,
    ).write(_gates(config))
    print(json.dumps({"passed": passed, **metrics, "failures": failures}, indent=2))
    print(output)
    return 0 if passed else 2


def cmd_verify_exhaustive_scale(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"12d_exhaustive_frontier_smoke_{args.dataset}")
    quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    quality = pd.read_parquet(quality_path)
    registry_directory = config.artifact_dir / "gate12" / args.dataset
    registry_yaml = registry_directory / "query-registry.yaml"
    registry_failures = validate_gate12_registry(source, registry_yaml, quality)
    if registry_failures:
        raise SystemExit("; ".join(registry_failures))
    yaml_registry = pd.DataFrame(
        (yaml.safe_load(registry_yaml.read_text()) or {})["cases_data"]
    )
    registry = pd.read_parquet(registry_directory / "query-registry.parquet")
    try:
        pd.testing.assert_frame_equal(
            yaml_registry.reset_index(drop=True), registry.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise SystemExit(f"Gate 12 registry copies disagree: {exc}") from exc
    query = build_episode(
        source, InstrumentKey(args.dataset, args.symbol), args.cutoff,
        args.lookback, config.representation_version,
    )
    if query.key.id not in set(registry.episode_id.astype(str)):
        raise SystemExit("scale-ladder query is not in the frozen Gate 12 registry")
    request = SearchQuery(
        query.key, (args.dataset,), ("A", "B"), args.top_k,
        max_per_instrument=args.max_per_instrument,
        minimum_history_gap_bars=args.minimum_history_gap,
    )
    root = Path(args.frontier_root) if args.frontier_root else (
        config.artifact_dir / "gate12" / "scale-ladder" / args.dataset
    )
    hours = args.maximum_projected_hours
    if hours is None:
        hours = 2.0 if args.dataset == "nse" else 6.0
    result = run_scale_ladder(
        query, source, request, quality, root,
        fractions=tuple(args.fractions), seed=args.seed, stride=args.stride,
        batch_size=args.batch_size,
        frontier_batch_rows=args.frontier_batch_rows,
        representation_cache_shards=args.representation_cache_shards,
        tolerance=args.tolerance, maximum_rss_mb=args.maximum_rss_mb,
        disk_reserve_bytes=int(args.disk_reserve_gb * 1024 ** 3),
        maximum_projected_hours=hours,
        rebuild_invalid=args.rebuild_invalid,
    )
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports"
        / f"exhaustive-scale-ladder-{args.dataset}-{args.symbol}.html"
    )
    write_scale_ladder_report(result, output)
    required = (
        {0.01, 0.1, 0.5, 1.0} if args.dataset == "nse" else {0.01, 0.1}
    )
    scope = "full" if required.issubset(set(args.fractions)) else "smoke"
    metrics = {
        "dataset": result.dataset_id,
        "query_episode_id": result.query_episode_id,
        "seed": result.seed,
        "total_eligible_instruments": result.total_eligible_instruments,
        "fractions": list(result.fractions),
        "rungs": [asdict(rung) for rung in result.rungs],
    }
    GateReport(
        f"12e_exhaustive_scale_{scope}_{args.dataset}", result.passed,
        metrics, list(result.failures),
    ).write(_gates(config))
    print(json.dumps({"passed": result.passed, **metrics,
                      "failures": result.failures}, indent=2, default=str))
    print(output)
    return 0 if result.passed else 2


def cmd_aggregate_exhaustive_scale(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"12a_query_registry_{args.dataset}")
    quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    quality = pd.read_parquet(quality_path)
    registry_directory = config.artifact_dir / "gate12" / args.dataset
    registry_yaml = registry_directory / "query-registry.yaml"
    failures = validate_gate12_registry(source, registry_yaml, quality)
    if failures:
        raise SystemExit("; ".join(failures))
    registry = pd.read_parquet(registry_directory / "query-registry.parquet")
    if args.query_episode_id not in set(registry.episode_id.astype(str)):
        raise SystemExit("aggregate query is not in the frozen Gate 12 registry")
    rungs = collect_scale_history(
        _gates(config), args.dataset, args.query_episode_id, args.seed,
    )
    required = (
        {0.01, 0.1, 0.5, 1.0} if args.dataset == "nse" else {0.01, 0.1}
    )
    missing = sorted(required.difference(rungs))
    passed = not missing
    metrics = {
        "dataset": args.dataset, "query_episode_id": args.query_episode_id,
        "seed": args.seed, "required_fractions": sorted(required),
        "observed_fractions": sorted(rungs),
        "rungs": [rungs[fraction] for fraction in sorted(rungs)],
    }
    aggregate_failures = [f"missing passing fraction {value:g}" for value in missing]
    scope = "full" if passed else "progress"
    GateReport(
        f"12e_exhaustive_scale_{scope}_{args.dataset}", passed,
        metrics, aggregate_failures,
    ).write(_gates(config))
    print(json.dumps({"passed": passed, **metrics,
                      "failures": aggregate_failures}, indent=2, default=str))
    return 0 if passed else 2


def _validated_gate12_registry(config, source, dataset_id: str):
    quality = pd.read_parquet(
        config.artifact_dir / "quality" / f"{dataset_id}.parquet"
    )
    directory = config.artifact_dir / "gate12" / dataset_id
    yaml_path = directory / "query-registry.yaml"
    failures = validate_gate12_registry(source, yaml_path, quality)
    if failures:
        raise SystemExit("; ".join(failures))
    payload = yaml.safe_load(yaml_path.read_text()) or {}
    yaml_registry = pd.DataFrame(payload["cases_data"])
    parquet_registry = pd.read_parquet(directory / "query-registry.parquet")
    try:
        pd.testing.assert_frame_equal(
            yaml_registry.reset_index(drop=True), parquet_registry.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise SystemExit(f"Gate 12 registry copies disagree: {exc}") from exc
    return quality, parquet_registry, str(payload["registry_digest"])


def _largest_seed_frontier(config, dataset_id: str, query_episode_id: str) -> Path | None:
    root = config.artifact_dir / "gate12" / "scale-ladder" / dataset_id
    candidates = []
    for path in root.glob("scale-*"):
        try:
            scale = int(path.name.removeprefix("scale-"))
        except ValueError:
            continue
        if (path / query_episode_id / "shards").exists():
            candidates.append((scale, path))
    return max(candidates)[1] if candidates else None


def cmd_build_gate12_authority(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"12e_exhaustive_scale_full_{args.dataset}")
    quality, registry, registry_digest = _validated_gate12_registry(
        config, source, args.dataset,
    )
    selected = registry[registry.case_id.astype(str) == args.case_id]
    if len(selected) != 1:
        choices = ", ".join(sorted(registry.case_id.astype(str)))
        raise SystemExit(f"unknown or duplicate case {args.case_id!r}; choose from {choices}")
    case = selected.iloc[0]
    query = build_episode(
        source, InstrumentKey(args.dataset, str(case.symbol)), str(case.cutoff),
        int(case.lookback), str(case.representation_version),
    )
    if query.key.id != str(case.episode_id):
        raise SystemExit("authority query does not match its frozen episode ID")
    request = SearchQuery(
        query.key, (args.dataset,), ("A", "B"), args.top_k,
        max_per_instrument=args.max_per_instrument,
        minimum_history_gap_bars=args.minimum_history_gap,
    )
    authority_root = Path(args.authority_root) if args.authority_root else (
        config.artifact_dir / "gate12" / "authorities" / args.dataset
    )
    frontier_root = authority_root / "frontiers"
    artifact_path = authority_root / "cases" / f"{query.key.id}.json"
    if args.seed_frontier_root:
        seed_root = Path(args.seed_frontier_root)
    elif args.no_auto_seed:
        seed_root = None
    else:
        seed_root = _largest_seed_frontier(config, args.dataset, query.key.id)
    try:
        artifact, seeded, build, resume = run_authority_case(
            query, source, request, quality, frontier_root, artifact_path,
            registry_digest=registry_digest, seed_frontier_root=seed_root,
            stride=args.stride, batch_size=args.batch_size,
            frontier_batch_rows=args.frontier_batch_rows,
            representation_cache_shards=args.representation_cache_shards,
            tolerance=args.tolerance, rebuild_invalid=args.rebuild_invalid,
        )
    except AuthorityError as exc:
        raise SystemExit(str(exc)) from exc
    report = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / f"authority-{args.dataset}-{query.key.id}.html"
    )
    write_authority_report(artifact_path, report)
    payload = json.loads(artifact_path.read_text())
    certificate = payload["certificate"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    failures = []
    if rss > args.maximum_rss_mb:
        failures.append(f"peak RSS {rss:.2f} MB exceeds {args.maximum_rss_mb:.2f} MB")
    metrics = {
        "case_id": args.case_id, "dataset": args.dataset,
        "query_episode_id": query.key.id, "artifact": str(artifact_path),
        "authority_digest": artifact.authority_digest,
        "result_digest": artifact.result_digest,
        "eligible_candidates": artifact.eligible_candidates,
        "exact_evaluated": artifact.exact_evaluated,
        "safely_pruned": artifact.safely_pruned,
        "stopped_early": bool(certificate["stopped_early"]),
        "search_seconds": certificate["elapsed_seconds"],
        "repeat_search_seconds": payload["repeated_certificate"]["elapsed_seconds"],
        "seed_frontier_root": str(seed_root) if seed_root else None,
        "seeded_shards": seeded, "instruments_built": build.instruments_built,
        "instruments_reused": build.instruments_reused,
        "resume_instruments_reused": resume.instruments_reused,
        "peak_rss_mb": rss,
    }
    passed = not failures
    GateReport(
        f"12f_authority_case_{args.dataset}_{query.key.id}", passed, metrics, failures,
    ).write(_gates(config))
    print(json.dumps({"passed": passed, **metrics, "failures": failures}, indent=2))
    print(report)
    return 0 if passed else 2


def cmd_aggregate_gate12_authorities(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"12e_exhaustive_scale_full_{args.dataset}")
    quality, registry, registry_digest = _validated_gate12_registry(
        config, source, args.dataset,
    )
    authority_root = Path(args.authority_root) if args.authority_root else (
        config.artifact_dir / "gate12" / "authorities" / args.dataset
    )
    failures = []
    records = []
    universe_request = SearchQuery(
        build_episode(
            source,
            InstrumentKey(args.dataset, str(registry.iloc[0].symbol)),
            str(registry.iloc[0].cutoff), int(registry.iloc[0].lookback),
            str(registry.iloc[0].representation_version),
        ).key,
        (args.dataset,), ("A", "B"), args.top_k,
        max_per_instrument=args.max_per_instrument,
        minimum_history_gap_bars=args.minimum_history_gap,
    )
    universe_digest = authority_universe_digest(source, universe_request, quality)
    for case in registry.sort_values("case_id").itertuples(index=False):
        query = build_episode(
            source, InstrumentKey(args.dataset, str(case.symbol)), str(case.cutoff),
            int(case.lookback), str(case.representation_version),
        )
        request = SearchQuery(
            query.key, (args.dataset,), ("A", "B"), args.top_k,
            max_per_instrument=args.max_per_instrument,
            minimum_history_gap_bars=args.minimum_history_gap,
        )
        path = authority_root / "cases" / f"{query.key.id}.json"
        try:
            artifact = validate_authority_artifact(
                path, query=query, request=request,
                source_fingerprint=source.fingerprint(query.key.instrument),
                benchmark_fingerprint=source.benchmark_fingerprint(),
                registry_digest=registry_digest,
                universe_source_digest=universe_digest,
            )
        except (AuthorityError, OSError) as exc:
            failures.append(f"{case.case_id}: {exc}")
            records.append({"case_id": str(case.case_id), "status": "MISSING/INVALID"})
        else:
            records.append({
                "case_id": str(case.case_id), "status": "PASS",
                "query_episode_id": query.key.id,
                "authority_digest": artifact.authority_digest,
                "eligible_candidates": artifact.eligible_candidates,
                "exact_evaluated": artifact.exact_evaluated,
                "safely_pruned": artifact.safely_pruned,
            })
    passed = not failures and len(records) == 12
    metrics = {
        "dataset": args.dataset, "registry_digest": registry_digest,
        "required_cases": 12, "completed_cases": sum(r["status"] == "PASS" for r in records),
        "cases": records,
    }
    scope = "full" if passed else "progress"
    GateReport(
        f"12f_authority_matrix_{scope}_{args.dataset}", passed, metrics, failures,
    ).write(_gates(config))
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / f"authority-matrix-{args.dataset}.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{record['case_id']}</td><td>{record['status']}</td>"
        f"<td>{record.get('eligible_candidates', '')}</td>"
        f"<td>{record.get('exact_evaluated', '')}</td></tr>" for record in records
    )
    output.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Authority matrix</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto}table{border-collapse:collapse;width:100%}th,td{padding:.5rem;border-bottom:1px solid #ddd}</style>"
        f"</head><body><h1>Gate 12 {args.dataset} authority matrix: {'PASS' if passed else 'IN PROGRESS'}</h1>"
        f"<p>{metrics['completed_cases']} of 12 frozen cases have valid, provenance-locked exact authorities.</p>"
        f"<table><thead><tr><th>Case</th><th>Status</th><th>Eligible</th><th>Exact</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<h2>Failures</h2><pre>{json.dumps(failures, indent=2)}</pre></body></html>"
    )
    print(json.dumps({"passed": passed, **metrics, "failures": failures}, indent=2))
    print(output)
    return 0 if passed else 2


def cmd_run_gate12_authority_matrix(args: argparse.Namespace) -> int:
    result = run_authority_matrix(
        Path(args.config), tuple(args.datasets),
        disk_reserve_bytes=int(args.disk_reserve_gb * 1024 ** 3),
        maximum_rss_mb=args.maximum_rss_mb,
    )
    print(json.dumps({
        "passed": result.passed,
        "completed_cases": result.completed_cases,
        "required_cases": result.required_cases,
        "failed_case": result.failed_case,
        "progress_path": str(result.progress_path),
    }, indent=2))
    return 0 if result.passed else 2


def cmd_analyze_kullamagi_examples(args: argparse.Namespace) -> int:
    config, source = _load(args)
    csv_bytes = download_kullamagi_positions(args.url)
    result = analyze_kullamagi_examples(
        source, csv_bytes, config.representation_version,
        source_url=args.url,
        lookback=args.lookback, top_k=args.top_k,
        minimum_history_gap_bars=args.minimum_history_gap,
        permutations=args.permutations, seed=args.seed,
    )
    directory = Path(args.output_dir) if args.output_dir else (
        config.artifact_dir / "external-examples" / "kullamagi-positions-2021"
    )
    paths = write_external_example_artifacts(result, csv_bytes, directory)
    print(json.dumps({
        "passed": result.passed, **result.metrics,
        "failures": result.failures, "artifacts": [str(path) for path in paths],
    }, indent=2, default=str))
    return 0 if result.passed else 2


def cmd_analyze_kullamagi_yfinance(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    csv_bytes = download_kullamagi_positions(args.url)
    records = parse_kullamagi_positions(csv_bytes.decode("utf-8-sig"))
    if not records:
        raise SystemExit("the published tracker contains no parsed trade rows")
    directory = Path(args.output_dir) if args.output_dir else (
        config.artifact_dir / "external-examples" / "kullamagi-yfinance-2021"
    )
    start = min(record.entry_date for record in records) - pd.Timedelta(days=800)
    end = max(record.entry_date for record in records) + pd.Timedelta(days=2)
    fetch = fetch_yahoo_examples(
        records, directory / "yahoo-cache", start=start, end=end,
        batch_size=args.batch_size, reuse_cache=not args.refresh,
    )
    result = analyze_kullamagi_examples(
        fetch.source, csv_bytes, config.representation_version,
        source_url=args.url, lookback=args.lookback, top_k=args.top_k,
        minimum_history_gap_bars=args.minimum_history_gap,
        permutations=args.permutations, seed=args.seed,
    )
    causal_mode = f"causal_{args.minimum_history_gap}_sessions"
    causal = result.metrics["modes"][causal_mode]
    result.metrics.update({
        "data_provider": "Yahoo Finance via yfinance",
        "target_setup_purity": args.target_purity,
        "causal_top1_target_reached": (
            causal["top1_setup_agreement"] >= args.target_purity
        ),
        "causal_top_k_target_reached": (
            causal["top_k_setup_purity"] >= args.target_purity
        ),
        "yahoo_manifest": str(fetch.manifest_path),
    })
    paths = list(write_external_example_artifacts(result, csv_bytes, directory))
    yahoo_coverage = directory / "yahoo-coverage.parquet"
    fetch.coverage.to_parquet(yahoo_coverage, index=False)
    workbook = write_external_example_workbook(
        result, fetch, directory / "kullamagi-pattern-analysis.xlsx",
        target_purity=args.target_purity,
    )
    paths.extend((fetch.manifest_path, yahoo_coverage, workbook))
    print(json.dumps({
        "passed": result.passed, **result.metrics,
        "failures": result.failures, "artifacts": [str(path) for path in paths],
    }, indent=2, default=str))
    return 0 if result.passed else 2


def cmd_build_view_store(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"01_ingestion_audit_{args.dataset}")
    require_passed(_gates(config), "03_synthetic_retrieval")
    quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    if not quality_path.exists():
        raise SystemExit(f"missing quality audit: {quality_path}")
    quality = pd.read_parquet(quality_path)
    output_root = Path(args.output_root) if args.output_root else (
        config.artifact_dir / "view-store"
    )
    result = build_view_store(
        source, quality, output_root, lookbacks=tuple(args.lookbacks),
        stride=args.stride, representation_version=config.representation_version,
        workers=args.workers, instrument_limit=args.instrument_limit,
        rebuild_invalid=args.rebuild_invalid,
        storage_dtype=args.storage_dtype,
    )
    metrics = {
        "dataset": result.dataset_id,
        "instruments_considered": result.instruments_considered,
        "shards_built": result.instruments_built,
        "shards_reused": result.instruments_reused,
        "quality_skipped": result.quality_skipped,
        "rows": result.rows,
        "storage_dtype": args.storage_dtype,
        "lookbacks": sorted(set(args.lookbacks)),
        "stride": args.stride,
        "manifest": str(result.manifest_path),
        "manifest_digest": result.manifest_digest,
        "seconds": result.seconds,
    }
    scope = "-".join(str(value) for value in sorted(set(args.lookbacks)))
    gate_name = f"11a_view_shards_{args.dataset}_{scope}"
    if args.instrument_limit is not None:
        gate_name = f"11a_view_shards_smoke_{args.dataset}_{scope}"
    GateReport(
        gate_name, result.passed,
        metrics, list(result.failures),
    ).write(_gates(config))
    print(json.dumps({"passed": result.passed, **metrics,
                      "failures": result.failures}, indent=2))
    return 0 if result.passed else 2


def cmd_build_index(args: argparse.Namespace) -> int:
    config, source = _load(args)
    require_passed(_gates(config), f"02_episode_manifest_{args.dataset}")
    require_passed(_gates(config), "03_synthetic_retrieval")
    manifest_path = config.artifact_dir / "episodes" / f"{args.dataset}.parquet"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: run build-episodes first ({manifest_path})")
    manifest = pd.read_parquet(manifest_path)
    if args.limit and len(manifest) > args.limit:
        # A bounded smoke/performance build should still span the full symbol
        # and time range rather than taking an early alphabetical prefix.
        positions = np.linspace(0, len(manifest) - 1, args.limit, dtype=int)
        manifest = manifest.iloc[positions].reset_index(drop=True)
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    kept_rows: list[dict] = []
    failures: list[str] = []
    for row in manifest.itertuples(index=False):
        try:
            episode = build_episode(
                source, InstrumentKey(row.dataset_id, row.symbol), row.cutoff,
                int(row.lookback), row.representation_version,
                row.quality_tier, tuple(filter(None, str(row.quality_issues).split(";"))),
            )
            ids.append(row.episode_id)
            vectors.append(represent(episode).coarse)
            kept_rows.append(row._asdict())
        except Exception as exc:
            failures.append(f"{row.episode_id}:{type(exc).__name__}:{exc}")
    index_path = config.artifact_dir / "indexes" / f"{args.dataset}.npz"
    metadata_path = config.artifact_dir / "indexes" / f"{args.dataset}.parquet"
    matrix = np.vstack(vectors) if vectors else np.empty((0, 128), dtype=np.float32)
    CoarseIndex(ids, matrix).save(index_path)
    pd.DataFrame(kept_rows).to_parquet(metadata_path, index=False)
    passed = bool(ids) and not failures
    GateReport(f"04_coarse_index_{args.dataset}", passed, {
        "dataset": args.dataset, "indexed": len(ids), "failed": len(failures),
        "dimensions": 128,
    }, failures[:100]).write(_gates(config))
    print(index_path)
    return 0 if passed else 2


def _candidate_from_row(source, row) -> SearchCandidate:
    episode = build_episode(
        source, InstrumentKey(row.dataset_id, row.symbol), row.cutoff,
        int(row.lookback), row.representation_version,
        row.quality_tier, tuple(filter(None, str(row.quality_issues).split(";"))),
    )
    return SearchCandidate.from_episode(episode)


def cmd_search(args: argparse.Namespace) -> int:
    config, source = _load(args)
    query = build_episode(
        source, InstrumentKey(args.dataset, args.symbol), args.cutoff,
        args.lookback, config.representation_version,
    )
    request = SearchQuery(query.key, (args.dataset,), ("A", "B"), args.top_k)
    if args.view_store:
        require_passed(
            _gates(config), f"11a_view_shards_{args.dataset}_{args.lookback}",
        )
        require_passed(_gates(config), f"11b_exact_safe_pruning_{args.dataset}")
        quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
        if not quality_path.exists():
            raise SystemExit(f"missing quality audit: {quality_path}")
        quality = pd.read_parquet(quality_path)
        view_root = Path(args.view_store_root) if args.view_store_root else (
            config.artifact_dir / "view-store"
        )
        persisted = persisted_exact_search(
            query, source, request, view_root,
            candidate_pool=args.candidate_pool,
            per_instrument_view=args.per_instrument_view,
            workers=args.workers,
            use_dtw_bound=args.lb_keogh,
            quality=quality,
        )
        matches = list(persisted.matches)
        candidate_report = persisted.candidate_search
        pruning_report = persisted.pruning
        search_provenance = {
            "search_backend": "persisted_signature_exact_safe",
            "view_store_root": str(view_root),
            "view_manifest_digest": candidate_report.manifest_digest,
            "view_shards_loaded": str(candidate_report.shards_loaded),
            "view_rows_considered": str(candidate_report.rows_considered),
            "view_local_candidates": str(candidate_report.local_candidates),
            "candidate_pool": str(args.candidate_pool),
            "candidate_hits": str(len(candidate_report.hits)),
            "fingerprints_validated": str(persisted.fingerprints_validated),
            "fingerprint_validation_seconds": f"{persisted.fingerprint_validation_seconds:.4f}",
            "candidate_generation_seconds": f"{candidate_report.elapsed_seconds:.4f}",
            "materialization_seconds": f"{persisted.materialization_seconds:.4f}",
            "exact_scoring_seconds": f"{persisted.exact_scoring_seconds:.4f}",
            "elapsed_seconds": f"{persisted.elapsed_seconds:.4f}",
            "pruning_mode": "symmetric_multivariate_lb_keogh" if args.lb_keogh else "non_dtw_partial_sum",
            "exact_candidates_eligible": str(pruning_report.eligible_candidates),
            "exact_candidates_evaluated": str(pruning_report.exact_evaluated),
            "exact_candidates_safely_pruned": str(pruning_report.safely_pruned),
            "dtw_bounds_evaluated": str(pruning_report.dtw_bounds_evaluated),
        }
    elif args.streaming:
        require_passed(_gates(config), f"01_ingestion_audit_{args.dataset}")
        require_passed(_gates(config), "03_synthetic_retrieval")
        quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
        quality = pd.read_parquet(quality_path) if quality_path.exists() else None
        scan = streaming_search(
            query, source, request, stride=args.stride,
            candidate_pool=args.candidate_pool, quality=quality,
            instrument_limit=args.instrument_limit,
            scan_backend=args.scan_backend,
            workers=args.workers,
            candidate_strategy=args.candidate_strategy,
        )
        if scan.failures:
            preview = "; ".join(scan.failures[:3])
            raise RuntimeError(f"streaming scan was incomplete ({len(scan.failures)} failures): {preview}")
        matches = scan.matches
        search_provenance = {
            "search_backend": "streaming_vectorized_scan",
            "scan_backend": args.scan_backend,
            "instruments_scanned": str(scan.instruments_scanned),
            "instruments_considered": str(scan.instruments_considered),
            "quality_skipped": str(scan.quality_skipped),
            "windows_scanned": str(scan.windows_scanned),
            "coarse_candidates": str(scan.coarse_candidates),
            "elapsed_seconds": f"{scan.elapsed_seconds:.4f}",
            "peak_rss_mb": f"{scan.peak_rss_mb:.1f}",
            "failures": str(len(scan.failures)),
        }
    else:
        require_passed(_gates(config), f"04_coarse_index_{args.dataset}")
        index_path = config.artifact_dir / "indexes" / f"{args.dataset}.npz"
        metadata_path = config.artifact_dir / "indexes" / f"{args.dataset}.parquet"
        if not index_path.exists() or not metadata_path.exists():
            raise SystemExit("missing index; run build-index first or pass --streaming")
        index = CoarseIndex.load(index_path)
        metadata = pd.read_parquet(metadata_path).set_index("episode_id", drop=False)
        coarse_hits = index.query(represent(query).coarse, args.candidate_pool)
        candidates = [_candidate_from_row(source, metadata.loc[episode_id]) for episode_id, _ in coarse_hits]
        matches = exact_search(query, candidates, request)
        search_provenance = {
            "search_backend": "persisted_coarse_index",
            "candidate_index": str(index_path),
        }
    outcome_frames = []
    for match in matches:
        full = source.load(match.episode_key.instrument)
        outcome_frames.append(compute_outcomes(full, match.episode_key.cutoff))
    summary = summarize_match_outcomes(outcome_frames)
    report_path = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / f"{args.dataset}-{args.symbol}-{query.key.cutoff.date()}.html"
    )
    write_search_report(query, matches, report_path, summary, {
        "representation_version": config.representation_version,
        "query_episode_id": query.key.id,
        "source_fingerprint": source.fingerprint(query.key.instrument),
        **search_provenance,
    })
    print(report_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-analogues")
    sub = parser.add_subparsers(required=True)
    for name in ("audit", "build-episodes", "build-index", "search"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--dataset", required=True)
    sub.choices["audit"].set_defaults(func=cmd_audit)
    sub.choices["build-episodes"].set_defaults(func=cmd_build_episodes)
    sub.choices["build-episodes"].add_argument("--instrument-limit", type=int)
    sub.choices["build-index"].add_argument("--limit", type=int)
    sub.choices["build-index"].set_defaults(func=cmd_build_index)
    search = sub.choices["search"]
    search.add_argument("--symbol", required=True)
    search.add_argument("--cutoff", required=True)
    search.add_argument("--lookback", type=int, default=252)
    search.add_argument("--top-k", type=int, default=20)
    search.add_argument("--candidate-pool", type=int, default=1000)
    search_backend = search.add_mutually_exclusive_group()
    search_backend.add_argument(
        "--streaming", action="store_true",
        help="scan source windows without a prebuilt index",
    )
    search_backend.add_argument(
        "--view-store", action="store_true",
        help="retrieve persisted signatures and rerank with exact-safe pruning",
    )
    search.add_argument("--view-store-root")
    search.add_argument("--per-instrument-view", type=int, default=5)
    search.add_argument(
        "--lb-keogh", action="store_true",
        help="strengthen exact-safe pruning with optional LB_Keogh",
    )
    search.add_argument("--instrument-limit", type=int)
    search.add_argument("--stride", type=int, default=5)
    search.add_argument("--scan-backend", choices=["auto", "vector", "mass"], default="auto")
    search.add_argument("--candidate-strategy", choices=["price", "multiview"], default="price")
    search.add_argument("--workers", type=int, default=1)
    search.add_argument("--output")
    search.set_defaults(func=cmd_search)
    verify = sub.add_parser("verify")
    verify.add_argument("--config", required=True)
    verify.add_argument("--seeds-per-family", type=int, default=5)
    verify.set_defaults(func=cmd_verify)
    contract = sub.add_parser("verify-case-memory-contract")
    contract.add_argument("--config", required=True)
    contract.add_argument("--contract", required=True)
    contract.add_argument("--trial-ledger")
    contract.add_argument("--machine-output")
    contract.add_argument("--output")
    contract.set_defaults(func=cmd_verify_case_memory_contract)
    ledger = sub.add_parser("build-data-ledger")
    ledger.add_argument("--config", required=True)
    ledger.add_argument("--availability", required=True)
    ledger.add_argument("--datasets", nargs="+", required=True)
    ledger.add_argument("--workers", type=int, default=4)
    ledger.add_argument("--output-dir")
    ledger.set_defaults(func=cmd_build_data_ledger)
    multiresolution = sub.add_parser("verify-multiresolution-state")
    multiresolution.add_argument("--config", required=True)
    multiresolution.add_argument("--datasets", nargs="+", required=True)
    multiresolution.add_argument("--registry-root")
    multiresolution.add_argument("--maximum-total-seconds", type=float, default=60.0)
    multiresolution.add_argument("--maximum-case-seconds", type=float, default=2.0)
    multiresolution.add_argument("--maximum-rss-mb", type=float, default=1024.0)
    multiresolution.add_argument("--output-dir")
    multiresolution.set_defaults(func=cmd_verify_multiresolution_state)
    latent = sub.add_parser("verify-latent-structures")
    latent.add_argument("--config", required=True)
    latent.add_argument("--verifier", required=True)
    latent.add_argument("--output-dir")
    latent.set_defaults(func=cmd_verify_latent_structures)
    latent_v2 = sub.add_parser("verify-latent-structures-v2")
    latent_v2.add_argument("--config", required=True)
    latent_v2.add_argument("--verifier", required=True)
    latent_v2.add_argument("--output-dir")
    latent_v2.set_defaults(func=cmd_verify_latent_structures_v2)
    compare = sub.add_parser("compare-methods")
    compare.add_argument("--config", required=True)
    compare.add_argument("--seeds-per-family", type=int, default=5)
    compare.add_argument("--ucr-dataset", default="GunPoint")
    compare.add_argument("--ucr-test-limit", type=int, default=40)
    compare.add_argument("--skip-ucr", action="store_true")
    compare.add_argument("--include-slow", action="store_true", help="include ShapeDTW all-pairs benchmark")
    compare.add_argument("--output")
    compare.set_defaults(func=cmd_compare)
    universe = sub.add_parser("verify-universe")
    universe.add_argument("--config", required=True)
    universe.add_argument("--dataset", required=True)
    universe.add_argument("--symbol", required=True)
    universe.add_argument("--cutoff", required=True)
    universe.add_argument("--lookback", type=int, default=252)
    universe.add_argument("--top-k", type=int, default=20)
    universe.add_argument("--minimum-history-gap", type=int, default=60)
    universe.add_argument("--stride", type=int, default=5)
    universe.add_argument("--reference-pool", type=int, default=1000)
    universe.add_argument("--comparison-pools", type=int, nargs="+", default=[50, 100, 200, 500])
    universe.add_argument("--recall-pool", type=int, default=200)
    universe.add_argument("--minimum-pool-recall", type=float, default=.90)
    universe.add_argument("--per-instrument", type=int, default=5)
    universe.add_argument("--scan-backend", choices=["auto", "vector", "mass"], default="vector")
    universe.add_argument("--candidate-strategy", choices=["price", "multiview"], default="price")
    universe.add_argument("--workers", type=int, default=1)
    universe.add_argument("--max-seconds", type=float, default=1800)
    universe.add_argument("--max-rss-mb", type=float, default=4096)
    universe.add_argument("--output")
    universe.set_defaults(func=cmd_verify_universe)
    oracle = sub.add_parser("verify-oracle")
    oracle.add_argument("--config", required=True)
    oracle.add_argument("--dataset", required=True)
    oracle.add_argument("--symbols", nargs="+", required=True)
    oracle.add_argument("--lookbacks", type=int, nargs="+", default=[63, 126, 252])
    oracle.add_argument("--cutoff-quantiles", type=float, nargs="+", default=[.75, 1.0])
    oracle.add_argument("--instruments-per-stratum", type=int, default=2)
    oracle.add_argument("--windows-per-instrument", type=int, default=16)
    oracle.add_argument("--top-k", type=int, default=20)
    oracle.add_argument("--coarse-pools", type=int, nargs="+", default=[25, 50, 100])
    oracle.add_argument("--minimum-candidates", type=int, default=100)
    oracle.add_argument("--workers", type=int, default=4)
    oracle.add_argument("--output-dir")
    oracle.set_defaults(func=cmd_verify_oracle)
    fusion = sub.add_parser("verify-fusion")
    fusion.add_argument("--config", required=True)
    fusion.add_argument("--dataset", required=True)
    fusion.add_argument("--oracle-dir")
    fusion.add_argument("--pool-sizes", type=int, nargs="+", default=[50, 100, 125])
    fusion.add_argument("--acceptance-pool", type=int, default=125)
    fusion.add_argument("--minimum-recall", type=float, default=.95)
    fusion.add_argument("--per-instrument-view", type=int, default=5)
    fusion.add_argument("--view-mode", choices=["cheap", "signature"], default="cheap")
    fusion.add_argument("--output")
    fusion.set_defaults(func=cmd_verify_fusion)
    pruning = sub.add_parser("verify-pruning")
    pruning.add_argument("--config", required=True)
    pruning.add_argument("--dataset", required=True)
    pruning.add_argument("--oracle-dir")
    pruning.add_argument("--tolerance", type=float, default=1e-12)
    pruning.add_argument("--output")
    pruning.set_defaults(func=cmd_verify_pruning)
    precision = sub.add_parser("verify-float16-precision")
    precision.add_argument("--config", required=True)
    precision.add_argument("--dataset", required=True)
    precision.add_argument("--oracle-dir")
    precision.add_argument("--minimum-recall", type=float, default=1.0)
    precision.add_argument("--minimum-ndcg", type=float, default=.999)
    precision.add_argument("--output")
    precision.set_defaults(func=cmd_verify_float16_precision)
    production = sub.add_parser("verify-production-search")
    production.add_argument("--config", required=True)
    production.add_argument("--dataset", required=True)
    production.add_argument("--symbol", required=True)
    production.add_argument("--cutoff", required=True)
    production.add_argument("--lookback", type=int, default=252)
    production.add_argument("--top-k", type=int, default=20)
    production.add_argument("--minimum-history-gap", type=int, default=60)
    production.add_argument("--candidate-pool", type=int, default=175)
    production.add_argument("--per-instrument-view", type=int, default=5)
    production.add_argument("--workers", type=int, default=4)
    production.add_argument("--repeat", type=int, default=2)
    production.add_argument("--lb-keogh", action="store_true")
    production.add_argument("--view-store-root")
    production.add_argument("--max-seconds", type=float, default=300.0)
    production.add_argument("--max-rss-mb", type=float, default=1024.0)
    production.add_argument("--tolerance", type=float, default=1e-12)
    production.add_argument("--output")
    production.set_defaults(func=cmd_verify_production_search)
    registry = sub.add_parser("build-gate12-registry")
    registry.add_argument("--config", required=True)
    registry.add_argument("--dataset", required=True)
    registry.add_argument("--oracle-dir")
    registry.add_argument("--seed", default="gate12-query-v1")
    registry.add_argument("--lookback", type=int, default=252)
    registry.add_argument("--historical-quantile", type=float, default=.70)
    registry.add_argument("--minimum-rows", type=int, default=1000)
    registry.add_argument("--minimum-future-sessions", type=int, default=60)
    registry.add_argument("--maximum-staleness-days", type=int, default=120)
    registry.add_argument("--output-dir")
    registry.set_defaults(func=cmd_build_gate12_registry)
    storage = sub.add_parser("verify-exact-storage")
    storage.add_argument("--config", required=True)
    storage.add_argument("--datasets", nargs="+", default=["nse", "nasdaq"])
    storage.add_argument("--sample-windows-per-symbol", type=int, default=4)
    storage.add_argument("--comparison-queries", type=int, default=4)
    storage.add_argument("--comparison-candidates", type=int, default=8)
    storage.add_argument("--tolerance", type=float, default=1e-12)
    storage.add_argument("--disk-reserve-gb", type=float, default=5.0)
    storage.add_argument("--frontier-bytes-per-row", type=int, default=64)
    storage.add_argument("--compression-safety-factor", type=float, default=1.25)
    storage.add_argument("--output")
    storage.set_defaults(func=cmd_verify_exact_storage)
    exact_batch = sub.add_parser("verify-exact-batch")
    exact_batch.add_argument("--config", required=True)
    exact_batch.add_argument("--datasets", nargs="+", default=["nse", "nasdaq"])
    exact_batch.add_argument("--windows-per-symbol", type=int, default=100)
    exact_batch.add_argument("--stride", type=int, default=5)
    exact_batch.add_argument("--batch-size", type=int, default=128)
    exact_batch.add_argument("--tolerance", type=float, default=1e-12)
    exact_batch.add_argument("--minimum-speedup", type=float, default=20.0)
    exact_batch.add_argument("--maximum-rss-mb", type=float, default=512.0)
    exact_batch.add_argument("--output")
    exact_batch.set_defaults(func=cmd_verify_exact_batch)
    exhaustive = sub.add_parser("verify-exhaustive-frontier")
    exhaustive.add_argument("--config", required=True)
    exhaustive.add_argument("--dataset", required=True)
    exhaustive.add_argument("--symbol", required=True)
    exhaustive.add_argument("--cutoff", required=True)
    exhaustive.add_argument("--lookback", type=int, default=252)
    exhaustive.add_argument("--top-k", type=int, default=20)
    exhaustive.add_argument("--max-per-instrument", type=int, default=3)
    exhaustive.add_argument("--minimum-history-gap", type=int, default=60)
    exhaustive.add_argument("--stride", type=int, default=5)
    exhaustive.add_argument("--batch-size", type=int, default=128)
    exhaustive.add_argument("--frontier-batch-rows", type=int, default=256)
    exhaustive.add_argument("--representation-cache-shards", type=int, default=32)
    exhaustive.add_argument("--instrument-limit", type=int)
    exhaustive.add_argument("--rebuild-invalid", action="store_true")
    exhaustive.add_argument("--tolerance", type=float, default=1e-12)
    exhaustive.add_argument("--maximum-rss-mb", type=float, default=1024.0)
    exhaustive.add_argument("--frontier-root")
    exhaustive.add_argument("--output")
    exhaustive.set_defaults(func=cmd_verify_exhaustive_frontier)
    scale = sub.add_parser("verify-exhaustive-scale")
    scale.add_argument("--config", required=True)
    scale.add_argument("--dataset", required=True)
    scale.add_argument("--symbol", required=True)
    scale.add_argument("--cutoff", required=True)
    scale.add_argument("--lookback", type=int, default=252)
    scale.add_argument("--fractions", type=float, nargs="+", required=True)
    scale.add_argument("--seed", default="gate12-scale-v1")
    scale.add_argument("--top-k", type=int, default=20)
    scale.add_argument("--max-per-instrument", type=int, default=3)
    scale.add_argument("--minimum-history-gap", type=int, default=60)
    scale.add_argument("--stride", type=int, default=5)
    scale.add_argument("--batch-size", type=int, default=128)
    scale.add_argument("--frontier-batch-rows", type=int, default=256)
    scale.add_argument("--representation-cache-shards", type=int, default=2)
    scale.add_argument("--rebuild-invalid", action="store_true")
    scale.add_argument("--tolerance", type=float, default=1e-12)
    scale.add_argument("--maximum-rss-mb", type=float, default=1024.0)
    scale.add_argument("--maximum-projected-hours", type=float)
    scale.add_argument("--disk-reserve-gb", type=float, default=5.0)
    scale.add_argument("--frontier-root")
    scale.add_argument("--output")
    scale.set_defaults(func=cmd_verify_exhaustive_scale)
    aggregate_scale = sub.add_parser("aggregate-exhaustive-scale")
    aggregate_scale.add_argument("--config", required=True)
    aggregate_scale.add_argument("--dataset", required=True)
    aggregate_scale.add_argument("--query-episode-id", required=True)
    aggregate_scale.add_argument("--seed", default="gate12-scale-v1")
    aggregate_scale.set_defaults(func=cmd_aggregate_exhaustive_scale)
    authority = sub.add_parser("build-gate12-authority")
    authority.add_argument("--config", required=True)
    authority.add_argument("--dataset", required=True)
    authority.add_argument("--case-id", required=True)
    authority.add_argument("--top-k", type=int, default=20)
    authority.add_argument("--max-per-instrument", type=int, default=3)
    authority.add_argument("--minimum-history-gap", type=int, default=60)
    authority.add_argument("--stride", type=int, default=5)
    authority.add_argument("--batch-size", type=int, default=128)
    authority.add_argument("--frontier-batch-rows", type=int, default=256)
    authority.add_argument("--representation-cache-shards", type=int, default=2)
    authority.add_argument("--tolerance", type=float, default=1e-12)
    authority.add_argument("--maximum-rss-mb", type=float, default=1024.0)
    authority.add_argument("--rebuild-invalid", action="store_true")
    authority.add_argument("--authority-root")
    authority.add_argument("--seed-frontier-root")
    authority.add_argument("--no-auto-seed", action="store_true")
    authority.add_argument("--output")
    authority.set_defaults(func=cmd_build_gate12_authority)
    aggregate_authority = sub.add_parser("aggregate-gate12-authorities")
    aggregate_authority.add_argument("--config", required=True)
    aggregate_authority.add_argument("--dataset", required=True)
    aggregate_authority.add_argument("--top-k", type=int, default=20)
    aggregate_authority.add_argument("--max-per-instrument", type=int, default=3)
    aggregate_authority.add_argument("--minimum-history-gap", type=int, default=60)
    aggregate_authority.add_argument("--authority-root")
    aggregate_authority.add_argument("--output")
    aggregate_authority.set_defaults(func=cmd_aggregate_gate12_authorities)
    matrix_runner = sub.add_parser("run-gate12-authority-matrix")
    matrix_runner.add_argument("--config", required=True)
    matrix_runner.add_argument("--datasets", nargs="+", default=["nse", "nasdaq"])
    matrix_runner.add_argument("--disk-reserve-gb", type=float, default=5.0)
    matrix_runner.add_argument("--maximum-rss-mb", type=float, default=1024.0)
    matrix_runner.set_defaults(func=cmd_run_gate12_authority_matrix)
    external = sub.add_parser("analyze-kullamagi-examples")
    external.add_argument("--config", required=True)
    external.add_argument("--dataset", default="nasdaq")
    external.add_argument("--url", default=KULLAMAGI_POSITIONS_URL)
    external.add_argument("--lookback", type=int, default=252)
    external.add_argument("--top-k", type=int, default=5)
    external.add_argument("--minimum-history-gap", type=int, default=60)
    external.add_argument("--permutations", type=int, default=1000)
    external.add_argument("--seed", type=int, default=20210819)
    external.add_argument("--output-dir")
    external.set_defaults(func=cmd_analyze_kullamagi_examples)
    yahoo_external = sub.add_parser("analyze-kullamagi-yfinance")
    yahoo_external.add_argument("--config", required=True)
    yahoo_external.add_argument("--url", default=KULLAMAGI_POSITIONS_URL)
    yahoo_external.add_argument("--lookback", type=int, default=252)
    yahoo_external.add_argument("--top-k", type=int, default=5)
    yahoo_external.add_argument("--minimum-history-gap", type=int, default=60)
    yahoo_external.add_argument("--permutations", type=int, default=1000)
    yahoo_external.add_argument("--seed", type=int, default=20210819)
    yahoo_external.add_argument("--target-purity", type=float, default=0.75)
    yahoo_external.add_argument("--batch-size", type=int, default=25)
    yahoo_external.add_argument("--refresh", action="store_true")
    yahoo_external.add_argument("--output-dir")
    yahoo_external.set_defaults(func=cmd_analyze_kullamagi_yfinance)
    view_store = sub.add_parser("build-view-store")
    view_store.add_argument("--config", required=True)
    view_store.add_argument("--dataset", required=True)
    view_store.add_argument("--lookbacks", type=int, nargs="+", default=[63, 126, 252])
    view_store.add_argument("--stride", type=int, default=5)
    view_store.add_argument("--workers", type=int, default=4)
    view_store.add_argument("--instrument-limit", type=int)
    view_store.add_argument("--rebuild-invalid", action="store_true")
    view_store.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    view_store.add_argument("--output-root")
    view_store.set_defaults(func=cmd_build_view_store)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

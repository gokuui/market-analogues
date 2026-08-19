from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .adapters import source_from_spec
from .benchmark import (
    available_methods, benchmark_synthetic, benchmark_ucr, results_frame,
    write_comparison_report,
)
from .config import AppConfig, load_config
from .episodes import build_episode, build_manifest
from .gates import GateReport, require_passed
from .fusion_verification import verify_candidate_fusion, write_fusion_report
from .index import CoarseIndex
from .outcomes import compute_outcomes, summarize_match_outcomes
from .oracle import run_oracle_suite, write_oracle_artifacts
from .quality import audit_source
from .report import write_search_report
from .representation import represent
from .search import SearchCandidate, exact_search
from .scan import streaming_search
from .types import InstrumentKey, SearchQuery
from .universe import verify_universe, write_universe_report
from .verification import run_synthetic_verifier, write_verification_gate


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
    )
    output = Path(args.output) if args.output else (
        config.artifact_dir / "reports" / f"multiview-fusion-{args.dataset}.html"
    )
    write_fusion_report(result, output)
    GateReport(
        f"10_multiview_fusion_{args.dataset}", result.passed,
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
    if args.streaming:
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
    search.add_argument("--streaming", action="store_true", help="scan source windows without a prebuilt index")
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
    fusion.add_argument("--output")
    fusion.set_defaults(func=cmd_verify_fusion)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

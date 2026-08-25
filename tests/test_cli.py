from pathlib import Path

import yaml

from market_analogues.benchmark import euclidean_path
from market_analogues.cli import build_parser, main
from market_analogues.gates import GateReport


def test_cli_exposes_gated_workflow() -> None:
    parser = build_parser()
    commands = ["audit", "build-episodes", "build-index", "verify", "verify-case-memory-contract", "build-data-ledger", "verify-multiresolution-state", "verify-latent-structures", "verify-latent-structures-v2", "verify-m04-candidate-case", "aggregate-m04-candidate-recall", "diagnose-m04r-incident", "verify-m04r-causal-prefixes", "verify-m04r-distance-v1", "verify-m04r-feature-kernel", "verify-m04r-quantized-bound", "verify-m04r-proposal-v2", "verify-m04r-quantized-ranks", "verify-m04r-packed-bound", "verify-m04r-full-pack", "verify-m04r-global-bound-proposal", "verify-m04r-certified-packed-search", "verify-m04r-certified-matrix", "build-m04r-batch-registry", "build-m04r-validation-registry", "search", "compare-methods", "verify-universe", "verify-oracle", "verify-fusion", "verify-pruning", "verify-float16-precision", "verify-production-search", "build-gate12-registry", "verify-exact-storage", "verify-exact-batch", "verify-exhaustive-frontier", "verify-exhaustive-scale", "aggregate-exhaustive-scale", "build-gate12-authority", "aggregate-gate12-authorities", "run-gate12-authority-matrix", "analyze-kullamagi-examples", "analyze-kullamagi-yfinance", "build-view-store"]
    for command in commands:
        argv = [command, "--config", "config.yaml"]
        if command in {"audit", "build-episodes", "build-index", "search", "verify-universe", "verify-oracle", "verify-fusion", "verify-pruning", "verify-float16-precision", "verify-production-search", "build-gate12-registry", "build-m04r-batch-registry", "build-m04r-validation-registry", "verify-exhaustive-frontier", "verify-exhaustive-scale", "aggregate-exhaustive-scale", "build-gate12-authority", "aggregate-gate12-authorities", "build-view-store"}:
            argv += ["--dataset", "test"]
        if command in {"search", "verify-universe", "verify-production-search", "verify-exhaustive-frontier", "verify-exhaustive-scale"}:
            argv += ["--symbol", "AAA", "--cutoff", "2020-01-01"]
        if command == "verify-exhaustive-scale":
            argv += ["--fractions", "0.01"]
        if command == "aggregate-exhaustive-scale":
            argv += ["--query-episode-id", "abc"]
        if command == "build-gate12-authority":
            argv += ["--case-id", "demo-case"]
        if command == "verify-oracle":
            argv += ["--symbols", "AAA"]
        if command == "verify-case-memory-contract":
            argv += ["--contract", "contract.yaml"]
        if command == "build-data-ledger":
            argv += ["--availability", "availability.yaml", "--datasets", "test"]
        if command == "verify-multiresolution-state":
            argv += ["--datasets", "test"]
        if command in {"verify-latent-structures", "verify-latent-structures-v2"}:
            argv += ["--verifier", "structural.yaml"]
        if command in {"verify-m04-candidate-case", "aggregate-m04-candidate-recall", "diagnose-m04r-incident", "verify-m04r-causal-prefixes", "verify-m04r-distance-v1", "verify-m04r-feature-kernel"}:
            argv += ["--contract", "m04.yaml"]
        if command == "verify-m04-candidate-case":
            argv += ["--episode-id", "abc"]
        args = parser.parse_args(argv)
        assert callable(args.func)


def test_cli_runs_gated_end_to_end_workflow(
    tmp_path: Path, directory_dataset: Path, bars, monkeypatch,
) -> None:
    artifacts = tmp_path / "artifacts"
    config = tmp_path / "datasets.yaml"
    config.write_text(yaml.safe_dump({
        "artifact_dir": str(artifacts),
        "representation_version": "dense-v1",
        "candidate_stride_bars": 20,
        "lookbacks": [126],
        "datasets": {
            "demo": {
                "adapter": "directory", "path": str(directory_dataset),
                "format": "parquet", "timestamp_column": "date",
            },
        },
    }))
    common = ["--config", str(config), "--dataset", "demo"]
    assert main(["audit", *common]) == 0
    assert main(["build-episodes", *common]) == 0
    assert main(["verify", "--config", str(config), "--seeds-per-family", "1"]) == 0
    assert main(["build-index", *common, "--limit", "20"]) == 0

    cutoff = str(bars.date.iloc[-1].date())
    indexed_report = tmp_path / "indexed.html"
    assert main([
        "search", *common, "--symbol", "AAA", "--cutoff", cutoff,
        "--lookback", "126", "--top-k", "3", "--output", str(indexed_report),
    ]) == 0
    assert indexed_report.exists()

    streaming_report = tmp_path / "streaming.html"
    assert main([
        "search", *common, "--symbol", "AAA", "--cutoff", cutoff,
        "--lookback", "126", "--top-k", "3", "--streaming",
        "--stride", "20", "--scan-backend", "vector",
        "--output", str(streaming_report),
    ]) == 0
    assert streaming_report.exists()

    assert main([
        "build-view-store", *common, "--lookbacks", "126", "--stride", "20",
        "--workers", "2",
    ]) == 0
    GateReport("11b_exact_safe_pruning_demo", True, {"test_fixture": True}).write(
        artifacts / "gates"
    )
    persisted_report = tmp_path / "persisted.html"
    assert main([
        "search", *common, "--symbol", "AAA", "--cutoff", cutoff,
        "--lookback", "126", "--top-k", "3", "--candidate-pool", "12",
        "--view-store", "--per-instrument-view", "3", "--workers", "2",
        "--output", str(persisted_report),
    ]) == 0
    persisted_text = persisted_report.read_text()
    assert "persisted_signature_exact_safe" in persisted_text
    assert "exact_candidates_safely_pruned" in persisted_text

    # Keep this command deterministic and independent of optional packages;
    # external methods are differentially tested in their own tests.
    monkeypatch.setattr(
        "market_analogues.cli.available_methods",
        lambda *_args, **_kwargs: {"euclidean_path": euclidean_path},
    )
    comparison = tmp_path / "comparison.html"
    assert main([
        "compare-methods", "--config", str(config), "--seeds-per-family", "1",
        "--skip-ucr", "--output", str(comparison),
    ]) == 0
    assert comparison.exists()

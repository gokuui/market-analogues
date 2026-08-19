from pathlib import Path

import yaml

from market_analogues.benchmark import euclidean_path
from market_analogues.cli import build_parser, main


def test_cli_exposes_gated_workflow() -> None:
    parser = build_parser()
    commands = ["audit", "build-episodes", "build-index", "verify", "search", "compare-methods", "verify-universe", "verify-oracle", "verify-fusion"]
    for command in commands:
        argv = [command, "--config", "config.yaml"]
        if command in {"audit", "build-episodes", "build-index", "search", "verify-universe", "verify-oracle", "verify-fusion"}:
            argv += ["--dataset", "test"]
        if command in {"search", "verify-universe"}:
            argv += ["--symbol", "AAA", "--cutoff", "2020-01-01"]
        if command == "verify-oracle":
            argv += ["--symbols", "AAA"]
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

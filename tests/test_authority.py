import json
from pathlib import Path

import pytest

from market_analogues.authority import (
    AuthorityError, authority_universe_digest, load_authority_artifact,
    run_authority_case, validate_authority_artifact, write_authority_artifact,
    write_authority_report,
)
from market_analogues.exhaustive import build_exact_frontier, exhaustive_frontier_search
from market_analogues.episodes import build_episode
from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.types import InstrumentKey, SearchQuery


def test_authority_round_trip_and_tamper_detection(
    directory_dataset: Path, bars, tmp_path: Path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    query = build_episode(
        source, InstrumentKey("test", "AAA"), bars.date.iloc[-1], 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 1,
        minimum_history_gap_bars=20, max_per_instrument=1,
    )
    quality = __import__("pandas").DataFrame({
        "symbol": ["AAA", "BBB"], "tier": ["A", "A"], "issues": ["", ""],
    })
    root = tmp_path / "frontier"
    build = build_exact_frontier(query, source, request, quality, root, stride=5)
    resume = build_exact_frontier(query, source, request, quality, root, stride=5)
    first = exhaustive_frontier_search(query, source, request, root, quality=quality)
    repeated = exhaustive_frontier_search(query, source, request, root, quality=quality)
    path = tmp_path / "authority.json"
    artifact = write_authority_artifact(
        query, request, build, resume, first, repeated, path,
        source_fingerprint=source.fingerprint(query.key.instrument),
        benchmark_fingerprint=source.benchmark_fingerprint(),
        registry_digest="locked",
    )
    assert artifact.eligible_candidates == build.eligible_rows
    assert artifact.eligible_candidates == (
        artifact.exact_evaluated + artifact.safely_pruned
    )
    assert validate_authority_artifact(
        path, query=query, request=request,
        source_fingerprint=source.fingerprint(query.key.instrument),
        benchmark_fingerprint=source.benchmark_fingerprint(),
        registry_digest="locked",
        universe_source_digest=authority_universe_digest(source, request, quality),
    ) == artifact
    report = write_authority_report(path, tmp_path / "authority.html")
    assert "blind exact authority: PASS" in report.read_text()
    with pytest.raises(AuthorityError, match="provenance is stale"):
        validate_authority_artifact(
            path, query=query, request=request,
            source_fingerprint=source.fingerprint(query.key.instrument),
            benchmark_fingerprint=source.benchmark_fingerprint(),
            registry_digest="changed",
            universe_source_digest=authority_universe_digest(source, request, quality),
        )
    with pytest.raises(AuthorityError, match="universe_source_digest"):
        validate_authority_artifact(
            path, query=query, request=request,
            source_fingerprint=source.fingerprint(query.key.instrument),
            benchmark_fingerprint=source.benchmark_fingerprint(),
            registry_digest="locked", universe_source_digest="changed",
        )
    payload = json.loads(path.read_text())
    payload["matches"][0]["total_distance"] += 0.01
    path.write_text(json.dumps(payload))
    with pytest.raises(AuthorityError, match="content digest mismatch"):
        load_authority_artifact(path)


def test_authority_runner_seeds_and_resumes_completed_checkpoint(
    directory_dataset: Path, bars, tmp_path: Path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    query = build_episode(
        source, InstrumentKey("test", "AAA"), bars.date.iloc[-1], 63, "dense-v1",
    )
    request = SearchQuery(
        query.key, ("test",), ("A", "B"), 1,
        minimum_history_gap_bars=20, max_per_instrument=1,
    )
    quality = __import__("pandas").DataFrame({
        "symbol": ["AAA", "BBB"], "tier": ["A", "A"], "issues": ["", ""],
    })
    seed = tmp_path / "seed"
    assert build_exact_frontier(query, source, request, quality, seed, stride=5).passed
    root = tmp_path / "authority-frontier"
    path = tmp_path / "case.json"
    first, seeded, build, resume = run_authority_case(
        query, source, request, quality, root, path,
        registry_digest="locked", seed_frontier_root=seed,
    )
    assert seeded == 2
    assert build.instruments_built == 0 and build.instruments_reused == 2
    assert resume.instruments_reused == 2
    second, seeded_again, checkpoint, _ = run_authority_case(
        query, source, request, quality, root, path,
        registry_digest="locked", seed_frontier_root=seed,
    )
    assert second.authority_digest == first.authority_digest
    assert seeded_again == 0 and checkpoint.manifest_digest == "checkpoint"

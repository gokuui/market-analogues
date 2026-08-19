from __future__ import annotations

import pandas as pd

from market_analogues.adapters import DirectorySource
from market_analogues.config import DatasetSpec
from market_analogues.episodes import build_episode
from market_analogues.fusion_verification import verify_candidate_fusion, write_fusion_report
from market_analogues.types import InstrumentKey


def test_fusion_verifier_enriches_oracle_and_writes_report(
    directory_dataset, tmp_path,
) -> None:
    source = DirectorySource(DatasetSpec(
        "test", "directory", directory_dataset, "parquet", timestamp_column="date",
    ))
    bars = source.load(InstrumentKey("test", "AAA"))
    query_cutoff = bars.timestamp.iloc[-1]
    case_id = "test-AAA-latest-126"
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    rows = []
    for position in (200, 240, 299):
        episode = build_episode(
            source, InstrumentKey("test", "BBB"), bars.timestamp.iloc[position],
            126, "dense-v1", "A",
        )
        rows.append({
            "episode_id": episode.key.id,
            "dataset_id": "test", "symbol": "BBB", "cutoff": episode.key.cutoff,
            "lookback": 126, "quality_tier": "A",
            "oracle_selected": position == 200,
        })
    pd.DataFrame(rows).to_parquet(oracle / f"{case_id}.parquet", index=False)
    pd.DataFrame([{
        "case_id": case_id, "query": "test:AAA", "cutoff": query_cutoff,
        "lookback": 126,
    }]).to_parquet(oracle / "oracle-summary.parquet", index=False)

    result = verify_candidate_fusion(
        source, oracle, representation_version="dense-v1",
        pool_sizes=(1, 3), acceptance_pool=3, minimum_recall=1.0,
        per_instrument_view=3,
    )
    assert result.passed
    enriched = pd.read_parquet(oracle / f"{case_id}.parquet")
    assert "candidate_view_stage" in enriched
    assert enriched.candidate_view_version.nunique() == 1
    report = write_fusion_report(result, tmp_path / "fusion.html")
    assert "Cheap multi-view candidate union" in report.read_text()

    signature = verify_candidate_fusion(
        source, oracle, representation_version="dense-v1",
        pool_sizes=(1, 3), acceptance_pool=3, minimum_recall=1.0,
        per_instrument_view=3, view_mode="signature",
    )
    assert signature.passed
    enriched = pd.read_parquet(oracle / f"{case_id}.parquet")
    assert "signature_view_stage" in enriched

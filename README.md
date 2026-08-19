# Market Analogues

A standalone, market-agnostic engine for finding structurally similar historical OHLCV episodes. It reads external datasets without modifying them, builds causal representations, verifies similarity with deterministic synthetic tests, and generates explainable HTML reports.

## Input contract

Required logical columns are `timestamp`, `open`, `high`, `low`, `close`, `volume`, plus a symbol supplied by a column or filename. CSV and Parquet are supported as one-file-per-symbol directories or long tables. Benchmarks are optional.

```bash
python -m pip install -e '.[dev,bench]'
market-analogues audit --config config/datasets.example.yaml --dataset nse
market-analogues build-episodes --config config/datasets.example.yaml --dataset nse
market-analogues verify --config config/datasets.example.yaml
market-analogues compare-methods --config config/datasets.example.yaml
market-analogues build-view-store --config config/datasets.example.yaml \
  --dataset nse --lookbacks 252 --stride 5
market-analogues search --config config/datasets.example.yaml --dataset nse \
  --symbol reliance --cutoff 2025-12-31 --streaming
```

Source datasets are always read-only. Derived manifests, vectors, indexes, outcomes, reports, and gate records are written beneath the configured artifact directory.

## Design

The representation is causal and multi-view: normalized price path and returns,
overnight/intraday movement, candle geometry, volatility and contraction, robust
volume/shocks, distances from highs and moving averages, benchmark regime,
relative strength, and causally confirmed directional-change events. A vectorized
correlation-plus-magnitude scan (optionally using STUMPY/MASS for long series) or
a persisted portable coarse index proposes candidates. An exact composite distance
then reranks them using rigid, stage-wise, structural, and bounded-DTW comparisons.
Candidates are separated from the query by at least 60 sessions, and overlapping
matches from one instrument are deduplicated.

Forward returns and excursions are computed only after retrieval and never enter
the ranking. Results are descriptive historical evidence, not forecasts.

- [Implementation and verification plan](docs/implementation-plan.html)
- [Research evidence map](docs/research-evidence.html)
- [Repository comparison and adopted ideas](docs/repository-comparison.html)
- [Complete-universe verification and next-step plan](docs/universe-verification-plan.html)
- [Current verification status and next implementation plan](docs/verification-status-and-next-plan.html)
- [Running development and decision log](docs/development-log.html)

## Verification

```bash
PYTHONPATH=src pytest
PYTHONPATH=src python3 -m market_analogues.cli verify \
  --config config/datasets.example.yaml
```

The verifier uses deterministic synthetic OHLCV families and metamorphic tests:
unit scaling must not alter similarity, exact clones must rank first, transformed
positives must beat reversed/context-contradictory negatives, future mutation must
have zero effect, and persisted coarse-index serialization must preserve neighbors.
Gate JSON keeps an immutable history beneath `artifact_dir/gates/history`.

The current regression record is 60 passing tests with 85% measured source
statement coverage. This establishes a verified retrieval baseline, not trading
profitability or production-scale completeness.

The stratified exact oracle and sampled multi-view recall gate pass at candidate
pool 175. Complete-universe coverage remains sound, but the richer on-demand
scanner is too slow and its shortlist still changes materially between pools 175
and 1,000. The universe gate therefore remains failed until persisted view shards,
exact-safe pruning, and independent full-universe validation are complete.

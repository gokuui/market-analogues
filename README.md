# Market Analogues

> **Status: closed, 2026-09-29. Failed as a predictive trading tool.** The retrieval engine works as engineered, but historical chart analogues did not predict outcomes beyond simple features. Pattern-encoding follow-ups did not produce a tradable gain once fills were realistic. All derived data artifacts have been deleted. **Read [RETROSPECTIVE.md](RETROSPECTIVE.md)** for why this started, what was built, and how it failed. The detailed record is L-326 in [docs/development-log.html](docs/development-log.html).

A standalone, market-agnostic engine for finding structurally similar historical OHLCV episodes. It reads external datasets without modifying them, builds causal representations, verifies similarity with deterministic synthetic tests, and generates explainable HTML reports.

## Input contract

Required logical columns are `timestamp`, `open`, `high`, `low`, `close`, `volume`, plus a symbol supplied by a column or filename. CSV and Parquet are supported as one-file-per-symbol directories or long tables. Benchmarks are optional.

```bash
python -m pip install -e '.[dev,bench]'
# Add `external-data` when running the Yahoo-labelled-example workbook.
python -m pip install -e '.[dev,external-data]'
market-analogues audit --config config/datasets.example.yaml --dataset nse
market-analogues build-episodes --config config/datasets.example.yaml --dataset nse
market-analogues verify --config config/datasets.example.yaml
market-analogues verify-case-memory-contract --config config/datasets.example.yaml \
  --contract config/case-memory-contract.yaml
market-analogues build-data-ledger --config config/datasets.example.yaml \
  --availability config/data-availability.yaml --datasets nse nasdaq --workers 8
market-analogues verify-multiresolution-state --config config/datasets.example.yaml \
  --datasets nse nasdaq
market-analogues verify-latent-structures --config config/datasets.example.yaml \
  --verifier config/structural-verifier.yaml
market-analogues verify-latent-structures-v2 --config config/datasets.example.yaml \
  --verifier config/structural-verifier-v2.yaml
market-analogues verify-m04-candidate-case --config config/datasets.example.yaml \
  --contract config/m04-candidate-recall-contract.yaml --episode-id 1736f00c337dfa0b6cf60e10
market-analogues aggregate-m04-candidate-recall --config config/datasets.example.yaml \
  --contract config/m04-candidate-recall-contract.yaml
market-analogues diagnose-m04r-incident --config config/datasets.example.yaml \
  --contract config/m04-candidate-recall-contract.yaml
market-analogues verify-m04r-causal-prefixes --config config/datasets.example.yaml \
  --contract config/m04-candidate-recall-contract.yaml --dataset nasdaq
market-analogues verify-m04r-distance-v1 --config config/datasets.example.yaml \
  --contract config/m04-candidate-recall-contract.yaml --dataset nasdaq
market-analogues verify-m04r-feature-kernel --config config/datasets.example.yaml \
  --contract config/m04-candidate-recall-contract.yaml --dataset nasdaq
market-analogues verify-m04r-quantized-bound --config config/datasets.example.yaml
market-analogues verify-m04r-proposal-v2 --config config/datasets.example.yaml
market-analogues verify-m04r-quantized-ranks --config config/datasets.example.yaml
market-analogues verify-m04r-packed-bound --config config/datasets.example.yaml
market-analogues verify-m04r-full-pack --config config/datasets.example.yaml
market-analogues verify-m04r-global-bound-proposal \
  --config config/datasets.example.yaml
market-analogues build-m04r-validation-registry \
  --config config/datasets.example.yaml --dataset nasdaq
python experiments/m04r/m04r11_build_authorities.py \
  --config config/datasets.example.yaml \
  --registry config/data/analogues/m04r10/nasdaq-untouched-authority-registry/query-registry.json \
  --full-root config/data/analogues/poc/m04r/packed-bound-full \
  --authority-root config/data/analogues/m04r11/authorities-sealed
market-analogues compare-methods --config config/datasets.example.yaml
market-analogues build-view-store --config config/datasets.example.yaml \
  --dataset nse --lookbacks 252 --stride 5
market-analogues verify-pruning --config config/datasets.example.yaml \
  --dataset nse
market-analogues build-gate12-registry --config config/datasets.example.yaml \
  --dataset nse
market-analogues verify-exact-storage --config config/datasets.example.yaml \
  --datasets nse nasdaq
market-analogues verify-exact-batch --config config/datasets.example.yaml \
  --datasets nse nasdaq
market-analogues verify-float16-precision --config config/datasets.example.yaml \
  --dataset nse
market-analogues verify-exhaustive-frontier --config config/datasets.example.yaml \
  --dataset nse --symbol homefirst --cutoff 2024-12-02 \
  --lookback 252 --top-k 20 --instrument-limit 8
market-analogues verify-exhaustive-scale --config config/datasets.example.yaml \
  --dataset nse --symbol homefirst --cutoff 2026-02-11 \
  --lookback 252 --fractions 0.01 0.1
market-analogues aggregate-exhaustive-scale --config config/datasets.example.yaml \
  --dataset nse --query-episode-id 99fe49f51285b398361b7b13
market-analogues build-gate12-authority --config config/datasets.example.yaml \
  --dataset nse --case-id nse-homefirst-current-252
market-analogues aggregate-gate12-authorities --config config/datasets.example.yaml \
  --dataset nse
market-analogues run-gate12-authority-matrix --config config/datasets.example.yaml \
  --datasets nse nasdaq
market-analogues analyze-kullamagi-examples --config config/datasets.example.yaml \
  --dataset nasdaq
market-analogues analyze-kullamagi-yfinance --config config/datasets.example.yaml \
  --target-purity 0.75
market-analogues search --config config/datasets.example.yaml --dataset nse \
  --symbol reliance --cutoff 2026-02-11 --lookback 252 \
  --candidate-pool 175 --view-store
.venv/bin/python experiments/m04r/distance_v1_bound_gate.py \
  --pairs 1000000 --reference-pairs 2000 \
  --output config/data/analogues/poc/m04r/distance-v1-bound-gate.json
.venv/bin/python experiments/m04r/quantized_bound_gate.py --pairs 1000000 \
  --output config/data/analogues/poc/m04r/quantized-bound-million-gate.json
.venv/bin/python experiments/m04r/quantized_bound_authority_gate.py \
  --config config/datasets.example.yaml --symbols 64 \
  --output config/data/analogues/poc/m04r/quantized-bound-authority-gate.json
.venv/bin/python experiments/m04r/proposal_v2_authority_gate.py \
  --config config/datasets.example.yaml --symbols 0 --workers 8 \
  --output config/data/analogues/poc/m04r/proposal-v2-full.json
.venv/bin/python experiments/m04r/hybrid_route_gate.py \
  --artifact-dir config/data/analogues \
  --compact-evidence config/data/analogues/poc/m04r/proposal-v2-full.json \
  --output config/data/analogues/poc/m04r/bound-assisted-route-gate.json
.venv/bin/python experiments/m04r/quantized_bound_rank_gate.py \
  --config config/datasets.example.yaml --workers 8 --worker-chunk 4 \
  --checkpoint-every 128 \
  --output config/data/analogues/poc/m04r/quantized-bound-rank-full.json
.venv/bin/market-analogues verify-m04r-quantized-ranks \
  --config config/datasets.example.yaml
.venv/bin/python experiments/m04r/packed_bound_1pct.py \
  --config config/datasets.example.yaml \
  --output-root config/data/analogues/poc/m04r/packed-bound-1pct --workers 8
.venv/bin/market-analogues verify-m04r-packed-bound \
  --config config/datasets.example.yaml
.venv/bin/python experiments/m04r/packed_bound_1pct.py \
  --config config/datasets.example.yaml \
  --output-root config/data/analogues/poc/m04r/packed-bound-full \
  --workers 8 --full-universe
.venv/bin/market-analogues verify-m04r-full-pack \
  --config config/datasets.example.yaml
.venv/bin/python experiments/m04r/global_bound_proposal_gate.py \
  --config config/datasets.example.yaml \
  --full-root config/data/analogues/poc/m04r/packed-bound-full \
  --output config/data/analogues/poc/m04r/global-bound-proposal-full.json \
  --block-rows 2048
.venv/bin/market-analogues verify-m04r-global-bound-proposal \
  --config config/datasets.example.yaml
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

The M02 state layer retains the complete 252-session derived channel frame and
adds explicit 252/126/63/21/10/5-session summaries and masked samples. Every one
of its 42 fields declares source inputs, earliest observation time and missingness;
missing benchmark context remains missing rather than becoming an observed zero.

- [Implementation and verification plan](docs/implementation-plan.html)
- [Research evidence map](docs/research-evidence.html)
- [Repository comparison and adopted ideas](docs/repository-comparison.html)
- [Complete-universe verification and next-step plan](docs/universe-verification-plan.html)
- [Current verification status and next implementation plan](docs/verification-status-and-next-plan.html)
- [End-to-end closure plan](docs/end-to-end-closure-plan.html)
- [Gate 12 exhaustive reference and independent validation plan](docs/gate12-exhaustive-reference-plan.html)
- [Case-based market memory: implementation and verification plan](docs/case-based-market-memory-plan.html)
- [M04R research audit and preflight POC evidence](docs/m04r-research-and-poc-report.html)
- [M04R certified NASDAQ retrieval remediation plan](docs/m04r-certified-nasdaq-retrieval-plan.html)
- [Frozen M00 case-memory contract](config/case-memory-contract.yaml)
- [Append-only case-memory trial ledger](config/case-memory-trials.yaml)
- [Passing M03b topology verifier](config/structural-verifier-v2.yaml)
- [M01 data-availability declaration](config/data-availability.yaml)
- [Precision policy: float16 versus native](docs/precision-policy.html)
- External labelled-example audit: generated at
  `config/data/analogues/external-examples/kullamagi-positions-2021/report.html`
- Yahoo-expanded tracker workbook: generated at
  `config/data/analogues/external-examples/kullamagi-yfinance-2021/kullamagi-pattern-analysis.xlsx`
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

The complete regression suite is the release authority; its current count is
recorded in the running development log rather than duplicated here. M03's first locked unseen
trial failed and remains in the ledger. The separately preregistered M03b holdout
passes all eight unnamed families at 100% top-1, 100% precision@5 and zero
critical-negative errors in 33.71 seconds. This establishes a verified retrieval baseline, not trading
profitability or production-scale completeness.

M01 accounts for all 14,185 configured source instruments with zero changed
fingerprints: 1,843 NSE and 11,584 NASDAQ instruments are usable, while 262 and
496 are explicitly quarantined. The available stock snapshots end on 11 February
2026 and 30 March 2026 respectively, so they support historical development but
not a current-date query. Point-in-time membership, delisting returns, identity
history, sector history and event history remain declared unavailable.

The stratified exact oracle and sampled multi-view recall gate pass at candidate
pool 175. All 24 independent Gate 12 full-universe authorities and both market
aggregates now pass, providing frozen exact reference rankings. Production recall
against those references has not yet been measured, so the persisted exact-safe
backend remains opt-in. This is completeness evidence, not outcome or profitability evidence.

## License

MIT; see [LICENSE](LICENSE). Nothing here is investment advice or a trading signal.

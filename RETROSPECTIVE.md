# Market Analogues: retrospective

**Status: closed on 2026-09-29. The project did not produce a working predictive or trading tool.**

This file is the plain-language account of why the project started, what was built, how it was tested, and why it failed. The detailed, append-only record is `docs/development-log.html` (the final entry is L-326). GitHub shows HTML files as source, so download that file to read it.

---

## 1. Why we started

The starting belief was that **stock charts repeat**. A stock moves through stages: a base, a breakout, an advance, consolidation, then another advance. So if you could find the past stocks whose charts developed most like today's, what happened to them next should tell you something about today's stock.

The plan (written on 2026-08-18, in `docs/plans/2026-08-18-historical-analogue-*`) said we should not hunt for named patterns such as VCPs, flags or cups. Instead it asked:

> Among episodes that were already observable at the query date, which stocks developed most similarly (price action, shocks, volume, relative strength, market context), and what happened after their cutoffs?

The target was swing trading on **NASDAQ** (about 12k stocks, including delisted ones) and **NSE India** (about 1.9k stocks), using daily OHLCV data.

## 2. What we built (2026-08-19 to 2026-09-11)

- **Representation.** A causal, multi-view view of the prior 252 trading sessions: normalised price path, returns, candle geometry, volatility and contraction, volume shocks, distance from highs and moving averages, relative strength, market regime, and directional-change events. It also includes multi-resolution summaries (252/126/63/21/10/5 sessions).
- **Retrieval.** Certified exact search across the whole universe, using a composite distance (rigid, structural and bounded-DTW), with deduplication and a 60-session separation rule.
- **Outcomes.** 5-, 20- and 60-session forward returns, maximum favourable and adverse excursions, barrier hits, and censoring. These are always computed after retrieval, so outcomes never influence which analogues are chosen.
- **On top of that:** evidence cards, "future path modes" (clustering what analogues did next), and a prospective listener for live out-of-sample testing.
- **Verification.** About 1,800 tests, synthetic metamorphic checks, independently rebuilt receipts, and preregistered gates. Roughly 500 commits in all.

**The engineering worked.** Retrieval returned exactly what it was designed to return.

## 3. How it failed

### 3.1 The analogues did not predict

- **Walk-forward test (WF-04, 2026-09-09).** On an untouched final block of 480 queries, the analogue forecast had a Brier skill of **−3.46%** versus a plain base rate. It was worse than predicting "the usual".
- **Decision checkpoint (2026-09-26).**
  - The chart retriever on its own scored **−2.4 skill points**.
  - A gradient-boosted model on **22 simple price/volume features** scored **+3.4**.
  - Adding the chart retriever to that model added **+0.15**, which is not significant. When combined with the simple model, the full retriever got **0% weight**.
- Confirming even the small positive blend with live forward testing would have needed about **99 months** of data.

### 3.2 Better encodings did not fix it

We then tried the idea that the encoding was the problem (`experiments/p0`, `p1`, `p2`):

- **A better target.** A stop-aware R-multiple ladder: stopped out, no move, or reached +1R / +2R / +3R / +5R before the stop.
- **Anchored and event features (E1):** trend clarity, gap size in σ, breakout volume, 4% bursts, new highs, stage, relative strength and breadth. These carried a small, real ranking signal on NASDAQ, but it was economically negligible (ΔRankIC +0.006).
- **A swing-leg "string" of the base (E2).** This is VCP grammar encoded as a sequence of ZigZag legs. **It added nothing.**
- **A trade-level test on existing strategy entries** passed, but it was not significantly better than the simple baseline. A one-time held-out check on 112 unseen trades did not confirm it.

### 3.3 The test bed itself was phantom

The strategies we used to test "does this improve real entries?" came from the sibling `loser` repo. When re-run with realistic NSE circuit-limit fills (bars locked at the limit can't be bought, and limit-down exits can't be sold), **every leaderboard strategy collapsed**:

| Strategy | Fills allowed on locked bars | Realistic fills |
|---|---|---|
| mom atr0.7 | 81% CAGR, 11% max DD | **−5% CAGR, 64% max DD** |
| mom BestV2 | 79% CAGR, 19% max DD | **−5% CAGR, 68% max DD** |
| Best other row | Calmar 3–9 | **Calmar ≤ 0.36; 7 of 10 lose money** |

A stricter lock rule gave the same answer. The earlier trade-level "passes" had been measured on those phantom trades.

### 3.4 Process failure: early "passes" that were bugs

Several first results looked encouraging and turned out to be wrong:

- A NaN-handling bug silently dropped **58% of NSE entries**, and they were the best ones.
- Stale-price rows with near-zero ATR produced realized R values of about **−7×10⁸**, which distorted the scoring.
- The survivorship audit used the wrong snapshot end date.
- Momentum trades that entered on locked bars averaged **+28%** and could not have been filled.

Each of these was caught only when the result was re-examined. **Trusting first results was the biggest process mistake.**

## 4. What we learned

1. **Charts that look alike did not lead to similar outcomes.** Unsupervised whole-chart matching failed here and also fails in the literature (for example, perceptually-important-points plus DTW failed on 18 of 18 equity indices). The usable information is local, anchored and event-driven, and it is small.
2. **Test against simple features first.** Any new representation has to beat a plain GBM on simple features before infrastructure is built around it. Weeks of verification rigour did not replace an early test of predictive value.
3. **Model circuit locks in NSE backtests** (`skip_circuit_locked` and `defer_circuit_lock_exit` in `loser`). Otherwise most of the apparent edge is phantom.
4. **Distrust a first positive result.** Check dropped rows, extreme values and fillability before believing it.
5. **Set stop rules in advance.** Pass/fail rules written down before each run are what made it possible to say "failed" clearly.

## 5. What was not tried

- A **setup-conditioned** version: analogues only at breakouts, gaps and bursts, plus earnings dates from SEC 8-K filings. It is written up in `docs/plans/2026-09-27-setup-analogue-plan-v3.html` and was never started. The research behind it suggests any gain would be modest, concentrated in earnings setups, and would come from event context rather than chart shape.

## 6. What's in this repo now

- `src/market_analogues/` — the retrieval engine: representation, distance, search, outcomes and reports.
- `experiments/` — every study script: `m04r` (the original gates), `decision`, `p0`, `p1`, `p2`. Each of `experiments/{decision,p0,p1,p2}` has a `*RESULTS*.md` file.
- `docs/` — plans and the development log (HTML).
- **No data.** All derived artifacts (about 38 GB) were deleted on 2026-09-29. The sealed digests in the log refer to files that no longer exist, and the price data was never part of this repo.

## 7. If you want to play with it

What works with no data:

```bash
python -m pip install -e '.[dev]'
# Synthetic verifier: deterministic synthetic OHLCV families and metamorphic checks
# (unit scaling, clones rank first, future mutation has no effect, index round-trip).
# First set artifact_dir in a copy of config/datasets.example.yaml to a writable folder.
PYTHONPATH=src python -m market_analogues.cli verify --config <your-copy>.yaml
```

To use your own data, follow the input contract in `README.md`:
- **Required columns:** `timestamp, open, high, low, close, volume`, plus a symbol.
- **Layout:** one file per symbol, or a long table, in CSV or Parquet.
- **Config:** point the `datasets` section of your config at the files. Then run `audit`, `build-episodes` and `search` (see `README.md`).

Caveats:
- Most `experiments/` scripts and many tests expect the deleted artifacts or the author's local data paths (`/home/vinay/...`), so they will not run as-is.
- **Nothing here is a trading signal.** The whole point of this retrospective is that it didn't work.

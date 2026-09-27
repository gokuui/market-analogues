# P2 results (2026-09-27): vault check and portfolio-level filter test

The inputs are in `experiments/p1/P1_RESULTS.md`, where the P1 gate passed at the trade level.
Raw outputs live in `config/data/analogues/p0/nse/` (`vault_*`, `daily_scores.parquet` and
`p2_portfolio/`), which are Git-ignored.

## Verdict

- **The P1 improvement does not survive.**
  - On the held-back vault, the E1+E2 filter gave nothing.
  - In full portfolio backtests, the E1+E2 filter was worse than the plain 22-feature baseline filter.
- **The momentum strategies used as the test bed have no realistic edge.** With circuit-lock realism
  on, both lose money with drawdowns of 64–68%. Their leaderboard numbers (about 80% CAGR) come from
  phantom fills on circuit-locked bars.
- **Recommendation: stop the predictive pattern track.** No encoding tested (the 22 features, E1
  anchored/event features, E2 swing legs) gives a robust, tradable improvement on these strategies.

## 1. Vault (one-time open; entries with signal date 2025-09-01 to 2026-02-10)

- A single frozen model per feature set was trained on outcomes completed before 2025-09-01.
- 10 of the entries had a locked entry bar and were excluded, leaving 112 fillable entries.
- "Skip the bottom third" uses ranks within the vault set.

| Model | IC | Kept − all | Kept mean | Skipped mean |
|---|---|---|---|---|
| Baseline, 22 features | +0.069 | **+0.70%** | +0.98% | −1.14% |
| Baseline + E1 + E2 | −0.005 | **−0.09%** | +0.19% | +0.47% |

There are only 112 trades over about 5 months, so the noise band is roughly ±1–2%. This is not proof
on its own, but it matches the P1 paired test, which found E1+E2 was not significantly better than the
baseline.

## 2. Portfolio backtests (loser engine, NSE `data/validated`, capital 1L, 3 positions, 1% risk, liquidity 5%)

**Filter rule (causal):**
- Every NSE stock-day is scored by the model fit on outcomes completed before that year
  (`experiments/p2/daily_scores.py`).
- A candidate signal is dropped when its within-date score percentile is below the one-third quantile
  of the strategy's own candidate-signal percentiles over the trailing 250 sessions.
- A dropped signal frees its slot for the next-ranked signal.

**Circuit realism:** `skip_circuit_locked` and `defer_circuit_lock_exit` both on, with lock threshold
range < 0.1%.

| Strategy | Filter | Circuit realism | CAGR (2015→2025-08) | Max DD | Calmar | Sharpe | Trades | Vault CAGR / DD |
|---|---|---|---|---|---|---|---|---|
| mom atr0.7 | none | off | 81.4% | 11.0% | 7.40 | 4.68 | 1003 | +5.3% / 14.1% |
| mom atr0.7 | none | **on** | **−5.3%** | **64.4%** | −0.08 | −0.35 | 843 | −7.5% / 15.3% |
| mom atr0.7 | baseline | on | +2.8% | 50.8% | 0.06 | 0.28 | 802 | −7.3% / 10.8% |
| mom atr0.7 | E1+E2 | on | −1.5% | 57.7% | −0.03 | −0.05 | 815 | −6.1% / 12.3% |
| mom BestV2 | none | off | 78.5% | 18.7% | 4.20 | 4.47 | 858 | +7.1% / 13.0% |
| mom BestV2 | none | **on** | **−4.9%** | **67.9%** | −0.07 | −0.32 | 728 | −11.2% / 17.6% |
| mom BestV2 | baseline | on | −0.5% | 53.3% | −0.01 | 0.03 | 682 | −1.2% / 10.3% |
| mom BestV2 | E1+E2 | on | −3.6% | 55.7% | −0.07 | −0.23 | 687 | −5.2% / 13.8% |

**Filter counts:** each filter dropped about 29k of about 91k candidate signals. About 1.1k signals had
no score and were allowed through.

The circuit-off runs reproduce the leaderboard figures (76%/15.5% and 71.9%/17.1% in `CLAUDE.md`), so
the harness matches the original engine. Turning realism on removes the whole edge.

**Reading the results:**
- The baseline filter improves CAGR by about 5–8 points and cuts drawdown by about 14 points, but it
  cannot make either strategy viable.
- E1+E2 is worse than the baseline filter in both strategies and in both segments.

## 3. Consequences

1. **Trade-level results from P0 and P1 are unreliable.** They used trades from circuit-off backtests.
   Even "fillable" entries could exit at limit-down bars that were not really sellable. The portfolio
   test with realism on supersedes them.
2. **The loser leaderboard's momentum rows are unrealistic** (S-2 "mom atr0.7 h60 p3" and BestV2). This
   extends `docs/CIRCUIT_LOCK_REALISM.md`, which found the same for the ML books. Neither momentum
   strategy is a live book. The live gen books were checked with circuit variants in `results/dhan_parity`.
3. **Pattern research:**
   - Anchored and event features carry a little cross-sectional information: NASDAQ ΔRankIC was
     +0.006 (t 4–5).
   - The swing-leg (VCP grammar) encoding adds nothing.
   - None of it produced a robust, tradable improvement.
   - Per the plan's stop rule, the predictive track stops here. E3 (chart CNN) and the S-track
     retrieval are not pursued.

## What remains useful

- **The label store and harness** (`experiments/p0`): R-ladder labels, the walk-forward evaluator, paired
  daily comparison and the entry go/no-go. They can be reused to test any future signal honestly.
- **The descriptive analogue viewer** (R2 modes): still available as a research and visual tool, not as
  a signal.

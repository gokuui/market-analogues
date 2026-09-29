# P1 results (2026-09-26)

The gate and the feature definitions were fixed beforehand in `experiments/p1/README.md`. Raw outputs
live in `config/data/analogues/p0/` (Git-ignored).

## Verdict

- **Primary gate: PASS.** This is the pooled fillable test on existing strategy entries, using the 22
  baseline features plus E1 and E2.
- The pass is modest. Adding E1+E2 to the baseline is **not significantly better** than the baseline
  in a paired test.
- The gain comes from **E1**, the anchored and event features. **E2**, the swing-leg "string", adds
  nothing by itself.

## Entry go/no-go (NSE, fillable entries, n = 2,077; skip the bottom-scored third per year)

| Feature set | Gate | IC | Kept − all (trade-weighted) | Kept − all (month mean) | 90% CI (month blocks) | Positive eras | Profit factor all → kept |
|---|---|---|---|---|---|---|---|
| Baseline (P0) | fail | 0.114 | +1.42% | +0.67% | [−0.15, +1.48] | 5/5 | 1.94 → 2.39 |
| **Baseline + E1 + E2 (primary)** | **pass** | 0.107 | +1.28% | +1.11% | **[+0.32, +1.97]** | 5/5 | 1.94 → 2.34 |
| Baseline + E1 | pass | 0.115 | +1.36% | +1.16% | [+0.21, +2.12] | 5/5 | 1.94 → 2.36 |
| Baseline + E2 | fail | 0.111 | +1.10% | +0.35% | [−0.40, +1.11] | 5/5 | 1.94 → 2.30 |

**Paired E1+E2 minus baseline on the same entries:** the month-mean improvement is +0.50%, with a 90% CI
of **[−0.27, +1.33]**, which is not significant. The two models make the same skip decision on 86% of
entries.

**Per strategy, primary run:**

| Strategy | n | Kept − all | 90% CI | Positive eras |
|---|---|---|---|---|
| mom BestV2 | 636 | +2.24% | [+1.07, +3.26] | 5/5 |
| mom atr0.7 | 787 | +1.54% | [+0.47, +2.83] | 5/5 |
| gen498 ref IS (2010–20) | 234 | +1.14% | [−0.40, +1.71] | 2/3 |
| **gen498 ref OOS (2021–25)** | 269 | **−0.26%** | [−4.23, +1.17] | 1/2 |
| VCP s19 / s20 (2023+) | 79 / 72 | +0.90 / +0.82% | [+0.35, +3.69] / [−0.30, +1.96] | — |

The filter helps the momentum strategies. It does **not** help gen498 in its recent out-of-sample
period, and gen498 is the live strategy.

## NASDAQ all-stock walk-forward (secondary): E1+E2 vs baseline, paired daily

The comparison covers 989 dates from 2006 to 2025-08.

| Metric | Δ mean | NW t |
|---|---|---|
| RankIC vs 5-day excess return | +0.0061 | +5.0 |
| RankIC vs 20-day excess return | +0.0064 | +4.1 (positive in 16/20 years) |
| RankIC vs realized R | +0.0095 | +8.6 |
| RPS improvement | +0.14% skill | +0.7 |
| Top-10 liquid net R | −0.013R | −0.5 |
| Previously-opened segment (2024-01 to 2025-08) | no improvement on any metric | |

The ranking gain is statistically clear but economically small. It does not reach the top-10 liquid
picks, and it does not show up in the most recent, previously-opened period.

## Bugs and changes during P1

These are logged as the README requires. No thresholds or definitions changed.

1. `true_range_atr` crashed on an empty price file. It now returns an empty array.
2. `entries_gonogo` now skips stocks with fewer than 300 bars before computing features. Those entries
   already had NaN for `ret_252` and were excluded, so the baseline numbers are unchanged.
3. A test assertion was wrong: leg depth is measured in current ATR, not in the ATR at the time the
   pivot formed. The test was corrected; the code was already right.

## Interpretation

- Anchored and event information adds a little over the 22 features: trend clarity, gaps in σ units,
  breakout volume, bursts, new highs, stage and relative strength, and breadth.
- Swing-leg or VCP-grammar encoding (E2) adds nothing measurable, alone or in combination.
- A filter built on this could plausibly improve the momentum strategies' entries. The evidence is
  trade-level, on data those strategies were built on, and on a survivors-only NSE universe.

## Proposed next steps (need the owner's decision)

1. **Portfolio-level test.** Add the E1+E2 filter to the loser engine for mom BestV2 and mom atr0.7,
   with circuit-lock realism **on**, and compare Calmar with and without it. In a portfolio, a skipped
   trade frees a slot for the next signal, so the trade-level gain may not carry over.
2. **Vault, one time only.** Score the 122 held-back entries (signal date on or after 2025-09-01) with
   the frozen E1+E2 model. This is a small sample, but it is the only data neither the model nor I have
   seen.
3. **Deprioritise** E3 (chart CNN) and the S-track retrieval. P1 found the useful information in simple
   anchored features, not in shape sequences.

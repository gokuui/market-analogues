# P1: do pattern features add what the 22 simple features could not?

This file was written on 2026-09-26, **before any P1 result was computed**. It follows
`experiments/p0/P0_RESULTS.md`, where the entry go/no-go failed marginally: the pooled fillable
kept-minus-all month-block 90% CI was [−0.15%, +1.48%].

## Candidate features (all causal, computed at the close of the signal session)

**E1: anchored and event features**
- Trend clarity: signed R² of the log-price fit over 63 and 252 sessions, and information
  discreteness `sign(r252) · (%down − %up)`.
- Close location in the 5- and 60-session high–low range.
- Gaps in σ units, `ln(open/prev close) / σ252`: today's gap, the max over the last 20 sessions, and
  days since the last gap above 2σ.
- Breakout volume: the max over the last 10 sessions of volume/SMA50(volume) on days that closed
  above the prior 50-session high.
- Bursts: days since the last +4% close, and the count of +4% closes in the last 20 sessions.
- New highs: the count of 252-session closing highs in the last 63 sessions.
- Stage and relative strength:
  - distance from the 150-session SMA and that SMA's 20-session slope, plus a stage code 1–4;
  - Mansfield RS against the market index and its 20-session change.
- Breadth (whole market): the share of stocks above their 50-session SMA, and its 20-session change.

**E2: the swing-leg "string" of the base**
- ZigZag at 1.5 × ATR20, run online, so a pivot counts only once it has been confirmed.
- Each of the last 3 down-legs and last 3 up-legs is described by:
  - depth in ATR units;
  - duration;
  - volume relative to SMA50.
- Plus: the contraction count (consecutive down-legs that shrink), the ratio of the last down-leg to
  the one before it, the distance from the last swing high (the pivot) in ATR units, the current
  unconfirmed leg (move and bars), and the number of pivots in the last 120 sessions.

## Gate (same rule as P0; decided before results)

**Primary.** Score the existing loser strategy entries with the baseline 22 features plus E1 plus E2.
Use the same models, refits, vault and fillable-only rule as `experiments/p0/entries_gonogo.py`.
The gate passes only if both hold:
- the pooled fillable kept-minus-all month-block bootstrap 90% lower bound is **above 0**;
- it is positive in **at least 4 of 5** primary eras.

If this fails, **stop the predictive track.**

**Secondary (reported, not gating).**
- The NASDAQ all-stock walk-forward (`experiments/p0/evaluate.py --extra`), compared with the P0
  baseline. For each metric, report the daily paired Δ with its Newey–West t: RPS skill, RankIC vs 5-
  and 20-day excess return, and top-10 liquid net R.
- The same comparison for E1 only and for E2 only, to see where any gain comes from. The primary gate
  uses only the combined E1+E2 run.

No thresholds, windows or feature definitions change after the first result. Bug fixes are allowed,
and each one is logged in `P1_RESULTS.md`.

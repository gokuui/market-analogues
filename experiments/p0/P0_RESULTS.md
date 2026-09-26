# P0 results (2026-09-26)

Plan: `docs/plans/2026-09-26-pattern-research-plan-v2.html`, including the review amendments.
Raw outputs live in `config/data/analogues/p0/`. That directory is Git-ignored, so the files exist only
on the data machine.

## Verdict

- **Go/no-go on existing strategy entries: FAIL, marginally.** The rule was fixed before the run.
- **NASDAQ all-stock baseline:** no probability skill. It shows small but real cross-sectional ranking
  skill on excess returns.

## Label stores

| Market | Rows | Symbols | C0 stop | C1 none | C2 +1R | C3 +2R | C4 +3R | C5 +5R |
|---|---|---|---|---|---|---|---|---|
| NASDAQ | 3.58M | 7,734 | 48.2% | 3.3% | 19.7% | 11.8% | 10.9% | 6.0% |
| NSE | 1.12M | 1,887 | 48.4% | 2.8% | 18.3% | 11.2% | 11.4% | 7.9% |

- The grid samples every 5th session.
- 4,904 NASDAQ symbols ended early. 1,292 of them are flagged as likely failures.
- **No NSE symbol ended early.** The NSE files are survivors only.

## Go/no-go: scoring loser strategy entries (NSE)

**Method**
- The baseline is 22 causal features feeding cumulative-ordinal HGB heads.
- It is refit per entry year on NSE rows whose outcomes completed before that year.
- Each entry's signal date is the session before its entry date.
- Entries on or after 2025-09-01 are held back in the vault; 122 were excluded.
- The decision uses fillable entries only, meaning the entry-day bar was not locked (high > low).

**Rule, fixed beforehand:** among the pooled fillable entries, skipping the bottom-scored third must
give a kept-minus-all trade return with a month-block bootstrap 90% lower bound above 0. It must also
be positive in at least 4 of 5 eras.

| | n | IC(score, trade return) | Kept − all (trade-weighted) | Kept − all (month mean) | 90% CI | Positive eras |
|---|---|---|---|---|---|---|
| **Pooled, fillable** | 2,077 | 0.114 | +1.43% | +0.67% | **[−0.15, +1.48]** | 5/5 |
| mom BestV2 | 636 | | +2.19% | | [+0.91, +5.55] | 4/5 |
| mom atr0.7 | 787 | | +1.12% | | [−0.13, +1.64] | 5/5 |
| gen498 ref, IS 2010–20 | 234 | | +1.37% | | [−0.96, +1.69] | 2/3 |
| gen498 ref, OOS 2021–25 | 269 | | +0.65% | | [−3.11, +0.44] | 2/2 |
| VCP s19 / s20 (2023+) | 79 / 72 | | +0.48 / −0.17% | | cross 0 | — |

The lower bound is below zero, so the gate **fails**. Only mom BestV2 passes on its own, and that
per-strategy result is post-hoc, so it cannot be used to rescue the pooled gate.

**Locked entries:** 599 entries had a locked entry bar (high == low). Their mean trade return was
**+28.4%**, against +3.9% for fillable entries. These are almost certainly phantom fills, as in
`docs/CIRCUIT_LOCK_REALISM.md`, and the loser momentum backtests include them.

**Control checks** (run on an earlier version of the score before the stale-price fix): the gain was
not a volatility tilt. The score correlated about −0.35 with ATR%. About 90% of the gain came from
nearness to the 52-week high alone.

## NASDAQ all-stock walk-forward baseline

The test years are 2006 to 2025-08. There are 989 dates and 2.70M rows. The censored and penalized
variants are nearly identical.

| Metric | Value | NW t |
|---|---|---|
| RPS skill vs training climatology | −0.47% | −1.4 |
| RPS skill vs same-date climatology (oracle reference) | −8.3% | |
| Daily RankIC: expected R vs 5-day excess return | +0.017 | +6.0 |
| Daily RankIC: expected R vs 20-day excess return | +0.023 | +5.2 |
| Daily RankIC: expected R vs realized R (stop + time exit) | −0.031 | −9.7 |
| Top-10 liquid picks minus all liquid, net R per trade | +0.07R | +2.1 |
| Previously-opened segment, 2024-01 to 2025-08: RPS skill | +0.98% | +2.2 |

**Reading the table**
- Probability calibration loses to constant base rates. The damage is concentrated in regime breaks:
  RPS skill was −5.7% in 2008 and −4.4% in 2011.
- Cross-sectional ranking of excess returns is real but small.
- The negative IC against realized R is a label-design finding, not noise.
  - The model prefers low-volatility stocks (corr −0.24 with ATR%).
  - For those stocks, a routine gap is many R, so C0 lumps a −1R stop together with −4R gap-throughs.
  - A future label should split C0 by gap severity, or predict realized R directly.

## Bugs found and fixed during P0

1. **Flat bars.** Flat (circuit) bars turned `close_location_20` and `range_contraction_10_63` into
   NaN for 10–63 sessions. That silently dropped 58% of NSE strategy entries, which averaged +14.9%
   against +0.7% for the rest. Fixed in `experiments/p0/features.py`; the decision-study function is
   left unchanged.
2. **Near-zero ATR.** Stale-price rows had near-zero ATR, so realized R reached about −7e8. That
   poisoned the per-class mean R used for expected R. Fixed by `clean_store`: R must be at least 0.2%
   of price, and realized R is clipped to [−10, 20].
3. **Invalid first PASS.** The first go/no-go PASS was produced by bugs 1 and 2. It is superseded by
   the table above.

## Suggested next step, to be decided by the owner

The gate failed, but only marginally, and only with the 22 simple features. P1 is the actual test of
the pattern hypothesis:
- anchored and event features (E1);
- the swing-leg "string" of the last base (E2).

P1 is cheap, about 2–3 days. The proposed condition, fixed now: run P1 with this same entry go/no-go
as the primary gate. If P1 still fails the pooled fillable gate, stop the predictive track.

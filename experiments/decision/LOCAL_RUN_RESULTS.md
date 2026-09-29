# Decision checkpoint — local run results (2026-09-26)

These are the results of running the Step 1–3 decision scripts on the owner's machine against the real
data and the saved WF-04 artifacts. They respond to the plan in `experiments/decision/README.md`.
The raw outputs are in `config/data/analogues/decision/`, which is Git-ignored and exists only on that
machine. The numbers below are copied from those JSON files.

## Summary

**Decision: stop the predictive track** (`stop_predictive_track_keep_descriptive_tool_if_used`).

The chart parts add skill over the no-chart mixture. They fail the stop rule against the plain feature
model: a gradient-boosted model on the 22 simple features beats the whole analogue system on its own.
Forward testing (M2) would need about 99 months, so it is not a near-term plan either.

## Runs performed

| Run | Command | Result |
|---|---|---|
| Sanity tests | `pytest tests/test_decision_chart_value.py tests/test_decision_survivorship.py` | 13/13 passed (14/14 after the fix below) |
| Step 1 fast | `chart_value --skip-feature-model` | Completed, `step1-fast.json` |
| Step 1 smoke | `chart_value --symbol-limit 500` | Timed out at 90 min in the GBM monthly fit; see note 1 |
| Step 1 full | `chart_value --workers 8` (defaults: stride 5, max rows 400k) | Completed in about 2.5 h, `step1-full.json` |
| Step 3 | `survivorship_audit --workers 8` | First run was wrong (bug, see below); rerun after the fix, `step3-survivorship.json` |

## Step 1 — held-fold Brier skill vs matched base rate (percentage points)

| Method | Pooled | validation_1 | validation_2 | validation_3 | final_untouched |
|---|---|---|---|---|---|
| matched_base_rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| composite_only | −2.41 | −2.46 | −0.69 | −4.60 | −2.02 |
| no_chart_mixture | +1.07 | +1.50 | +0.27 | +0.71 | +1.79 |
| with_chart_mixture | +1.91 | +2.21 | +1.37 | +1.34 | +2.68 |
| m1_frozen_weights_in_sample | +1.93 | +2.19 | +1.48 | +1.52 | +2.50 |
| logistic | +1.36 | +1.73 | +0.35 | +2.08 | +1.31 |
| gbm | +2.97 | +6.92 | +1.34 | +1.82 | +1.52 |
| **features_no_chart** | **+3.43** | +6.55 | +2.05 | +2.47 | +2.46 |
| features_with_chart | +3.59 | +6.35 | +2.14 | +2.83 | +2.84 |

Held rows: 1,678. Feature model: 12,080 symbols, 2,757,217 training rows. Training label shares:
adverse_first 62.7%, favorable_first 31.7%, no_touch 5.6%. One query had no features.

### Paired increments (skill points, month-block 90% interval, 74 months)

| Comparison | Mean | 90% interval | Positive folds | Passes stop rule |
|---|---|---|---|---|
| chart_increment_over_mixture | +0.84 | [+0.13, +1.57] | 4/4 | yes |
| **chart_increment_over_feature_model** | **+0.15** | **[−0.09, +0.40]** | 3/4 | **no** (below 0.5, interval crosses 0) |
| current_mixture_vs_features_no_chart | −1.53 | [−3.29, +0.26] | 1/4 | — |

Per-fold chart increment over the feature model: validation_1 −0.21, validation_2 +0.10,
validation_3 +0.35, final_untouched +0.34.

### Forward-selected weights (identical or near-identical across folds)

- features_no_chart: gbm 0.8, recent_return_volatility 0.2, matched_causal_history 0.0
- features_with_chart: **composite 0.0**, price_only 0.1, gbm 0.7–0.8, recent_return_volatility 0.1–0.2
- with_chart (no feature model): composite 0.2, price_only 0.3, matched_causal_history 0.1–0.2,
  recent_return_volatility 0.3–0.4

When the feature model is available, the full chart retriever gets zero weight. The +0.84 gain over
the plain mixture shows the chart parts were standing in for information that simple features carry
better.

## Step 2 — prospective power

Effect +1.91 skill points, monthly SD 7.60, about 22.7 queries per month. **99 months** are needed
for 80% power, far above the 18-month limit, so `m2_near_term_feasible = false`.

## Step 3 — survivorship (corrected)

```json
{
  "snapshot_end": "2026-03-30",
  "symbols": 12080,
  "ended_early_symbols": 8519,
  "median_yearly_attrition_percent": 6.94,
  "final_60_log_return_median": {"ended_early": 0.0142, "survivors": -0.0725},
  "ended_early_last_close_below_1_percent": 25.05,
  "reading": "attrition_present_check_whether_delisting_returns_are_recorded"
}
```

Yearly attrition runs 4–18% (median 6.9%), with peaks in 2001 (13.3%) and 2021–2025 (11.8–18.5%). So
the universe is **not** a survivor snapshot. Ended-early names have a *higher* median final-60-session
return than survivors. That points to many exits being acquisitions, and to failed names missing their
final delisting return. The remaining bias is missing delisting returns, not missing dead stocks.

## Bug fixed in `survivorship_audit.py`

`summarize()` took the snapshot end from the benchmark calendar, which runs to 2026-05-08, but the NASDAQ
stock files end on 2026-03-30. As a result, all 12,080 symbols were flagged `ended_early` and the
survivor medians came out `null`. The fix clips the benchmark sessions to the last date any symbol
reached before it computes the end and threshold.
`test_benchmark_running_past_stock_snapshot_does_not_mark_survivors_ended` was added as a regression
test. Before the fix the headline attrition was 6.94%, and it still is, because the by-year counts for
completed years were unaffected.

## Notes for the next iteration

1. `--symbol-limit` barely shrinks the smoke run. The universe is the query symbols plus N more, and the
   query symbols alone come to about 2,300. The slow part is also not features: it is the sequential
   monthly refit (one HistGradientBoosting fit on up to 400k rows per query month). A smoke mode that
   caps the number of refit months or `max_rows`, or parallel refits across months, would make it
   usable.
2. The full run's log ends at `fitting logistic monthly` with no summary block, even though
   `step1-full.json` was written correctly and the run used `python -u`. The cause has not been
   investigated; it may be an environment artifact of the detached `nohup` launch. `print_summary`
   reads only keys present in the JSON.
3. Suggested direction: keep market-analogues as a descriptive "charts like this, what happened next"
   tool. If a predictor is wanted, start from the 22-feature GBM (+3.4 points), not the retriever. It
   has not been tested as a trading rule, and a +3.4-point Brier skill is a modest edge.

# Decision checkpoint: is more predictive work worth it?

These are quick, exploratory kill tests, not preregistered gates. They reuse
already-consumed development evidence and source files, open no untouched outcome,
and authorize no claim. Their only job is to decide where the next month goes.

Use: personal trading decisions and a research tool.

## Why

The M1 grid search gave the full chart retriever 10% of the mixture weight, the
normalized price path 20%, and the non-chart components (matched base rate and a
three-number return/volatility neighbour) 70%. That suggests most of the skill
does not come from chart similarity. Nothing so far compares the chart retrievers
with a plain feature model built from the same causal inputs.

## Steps and stop rules (frozen in `chart_value.STOP_RULES`)

| Step | Script | Question | Stop rule |
|---|---|---|---|
| 1 | `chart_value.py` | Do `composite`/`price_only` add Brier skill over the best non-chart mixture, and over a plain feature model blended with the base rate? Every weight is selected on earlier folds only. | The chart increment must be ≥ 0.5 skill points, have a month-block 90% lower bound > 0, and be positive in ≥ 3/4 held folds, against **both** comparators. Otherwise stop the predictive track. |
| 2 | `chart_value.py` (same run) | How many prospective months would M2 need to confirm the development effect at 80% power, one-sided α = 0.05, with month-level dependence? | More than 18 months means M2 is not a near-term plan. |
| 3 | `survivorship_audit.py` | How much of the universe stops trading each year? | Median yearly attrition below 2% suggests a survivor snapshot, so outcome distributions are biased upward. Decide whether to buy delisting-aware data. |
| 4 | — | Decide | See the table below. |

| Result | Next move |
|---|---|
| Chart adds value and M2 is feasible | Continue, narrowed to event-conditioned setups (e.g. after a breakout or 4% day). |
| Chart adds value but M2 is slow | The signal may be real but is slow to confirm; decide whether to wait. |
| Chart adds nothing | Stop the predictive track. Keep a minimal "similar charts + outcome spread" viewer only if you would actually use it. |

## Run (on the machine that has the data and sealed artifacts)

```bash
.venv/bin/pip install -e '.[dev]'
# Fast part A only (minutes): mixture ablation plus power estimate.
PYTHONPATH=.:src .venv/bin/python -m experiments.decision.chart_value --skip-feature-model
# Full run: universe feature table (every 5th session per symbol) plus monthly refits.
PYTHONPATH=.:src .venv/bin/python -m experiments.decision.chart_value --workers 8
# Optional smoke run first: query symbols plus the first 500 universe symbols.
PYTHONPATH=.:src .venv/bin/python -m experiments.decision.chart_value --symbol-limit 500 \
  --output config/data/analogues/decision/step1-smoke.json
PYTHONPATH=.:src .venv/bin/python -m experiments.decision.survivorship_audit --workers 8
```

Outputs are written to `config/data/analogues/decision/` (Git-ignored) and a
summary is printed.

## Caveats

- Development folds were already used to choose the M1 weights. The ablation is
  still a fair kill test: if the chart retrievers cannot show value even here,
  they will not on a fresh holdout. A pass is not validation.
- The feature model uses the same barrier label as T14-09 (+2 ATR before −1 ATR
  within 20 sessions; same-bar double touches dropped; next 20 benchmark sessions
  required). `tests/test_decision_chart_value.py` checks that against
  `causal_outcomes`.
- The feature model trains on the whole configured universe, which carries the same
  survivorship bias as every other component.
- The power estimate uses the with-chart mixture versus the base rate, the more
  lenient of M2's two comparators, so it is a lower bound on the months needed.

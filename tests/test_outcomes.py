from __future__ import annotations

import pandas as pd

from market_analogues.outcomes import compute_outcomes, summarize_match_outcomes


def test_outcomes_are_forward_only_and_censored(bars: pd.DataFrame) -> None:
    canonical = bars.rename(columns={"date": "timestamp"})
    cutoff = canonical.timestamp.iloc[-11]
    result = compute_outcomes(canonical, cutoff, horizons=(5, 10, 20))
    assert not bool(result.iloc[0].censored)
    assert not bool(result.iloc[1].censored)
    assert bool(result.iloc[2].censored)
    expected = canonical.close.iloc[-1] / canonical.close.iloc[-11] - 1
    assert result.iloc[1].forward_return == pytest.approx(expected)


def test_outcome_summary_excludes_censored_rows(bars: pd.DataFrame) -> None:
    canonical = bars.rename(columns={"date": "timestamp"})
    one = compute_outcomes(canonical, canonical.timestamp.iloc[-30], horizons=(5,))
    two = compute_outcomes(canonical, canonical.timestamp.iloc[-3], horizons=(5,))
    summary = summarize_match_outcomes([one, two])
    assert summary.iloc[0].sample_size == 1
    assert summary.iloc[0].return_q25 == summary.iloc[0].return_q75


import pytest

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_analogues.post_signal_outcome_join import (
    PostSignalOutcomeJoinError, identity_projection, join_year_outcomes,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date = pd.Timestamp("2020-01-02"); identity = {"population": "broad", "signal_name": "up_close_4pct",
        "signal_id": "s1", "signal_date": date, "event_symbol": "E"}
    events = pd.DataFrame([{**identity, "control_count": 5, "full_match": True}])
    controls = pd.DataFrame([{**identity, "control_symbol": symbol, "match_rank": rank,
                              "selection_digest": f"d{rank}"} for rank, symbol in enumerate("ABCDF", 1)])
    rows = []
    for index, symbol in enumerate("EABCDF"):
        row = {"symbol": symbol, "signal_date": date}
        for horizon in (5, 20, 60):
            row.update({f"complete_{horizon}": not (symbol == "E" and horizon == 60),
                f"status_{horizon}": "complete" if not (symbol == "E" and horizon == 60) else "source_end_before_horizon",
                f"endpoint_close_return_{horizon}": index / 10,
                f"benchmark_relative_log_return_{horizon}": index / 20,
                f"endpoint_gain_25pct_{horizon}": float(index >= 3),
                f"maximum_favorable_excursion_{horizon}": index / 5,
                f"maximum_adverse_excursion_{horizon}": -index / 20})
        row["barrier_code_20"] = index % 4; rows.append(row)
    return events, controls, pd.DataFrame(rows)


def test_join_retains_identities_and_requires_all_six_complete() -> None:
    events, controls, panel = _inputs(); subjects, paired, coverage = join_year_outcomes(events, controls, panel)
    assert len(subjects) == 18 and len(paired) == 3
    assert paired.set_index("horizon_sessions").paired_complete.to_dict() == {5: True, 20: True, 60: False}
    assert np.isnan(paired.loc[paired.horizon_sessions.eq(60), "paired_endpoint_close_return_difference"]).all()
    assert coverage.paired_complete_rows.tolist() == [1, 1, 0]


def test_outcome_mutation_cannot_change_identity_projection() -> None:
    events, controls, panel = _inputs(); before, _, _ = join_year_outcomes(events, controls, panel)
    for column in panel.columns:
        if column.startswith(("endpoint_", "benchmark_relative_", "maximum_")): panel[column] = 999.
    after, _, _ = join_year_outcomes(events, controls, panel)
    pd.testing.assert_frame_equal(identity_projection(before), identity_projection(after), check_exact=True)


def test_join_rejects_missing_control_rank() -> None:
    events, controls, panel = _inputs()
    with pytest.raises(PostSignalOutcomeJoinError, match="control count"):
        join_year_outcomes(events, controls.iloc[:-1], panel)


def test_empty_year_retains_schema_and_zero_rows() -> None:
    events, controls, panel = _inputs()
    subjects, paired, coverage = join_year_outcomes(events.iloc[0:0], controls.iloc[0:0], panel)
    assert subjects.empty and paired.empty and coverage.empty
    assert "paired_complete" in paired and "paired_complete_fraction" in coverage

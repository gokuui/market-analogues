from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market_analogues.stockbee_study import clustered_winners, symbol_risk_rows
from experiments.m04r import m04r14_t14_11_stockbee_events as events
from experiments.m04r import verify_m04r14_t14_11_stockbee_events as verifier
from tests.test_stockbee_study import _bars


def test_independent_event_clustering_matches_kernel() -> None:
    risk = symbol_risk_rows(_bars(600), "SYN")
    expected = clustered_winners(risk)
    expected["event_id"] = [
        events.stable_hash(["contract", row.symbol, int(row.horizon_sessions), int(row.start_position)])[:24]
        for row in expected.itertuples(index=False)
    ]
    observed = verifier._reference_events(risk, "contract")
    pd.testing.assert_frame_equal(
        expected.sort_values(["start", "symbol", "horizon_sessions"]).reset_index(drop=True), observed,
    )


def test_prevalence_aggregate_counts_broad_and_investable() -> None:
    risk = symbol_risk_rows(_bars(600), "SYN")
    rows = pd.DataFrame(events._aggregate(risk, "unclustered_risk_set"))
    assert set(rows.population) == {"broad", "investable"}
    selected = rows.loc[
        (rows.population == "broad") & (rows.horizon_sessions == 21)
        & (rows.outcome_group == "winner") & (rows.exposure == "up_close_4pct")
        & (rows.window == "start_day")
    ].iloc[0]
    winners = risk.loc[(risk.horizon_sessions == 21) & risk.winner_25pct]
    assert selected.rows == len(winners)
    assert selected.exposed_rows == int(winners.up_close_4pct_start_day.sum())

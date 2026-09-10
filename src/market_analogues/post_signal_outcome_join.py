"""Immutable-identity outcome joining for the T14-12 post-signal study."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


HORIZONS = (5, 20, 60)
METRICS = (
    "endpoint_close_return", "benchmark_relative_log_return", "endpoint_gain_25pct",
    "maximum_favorable_excursion", "maximum_adverse_excursion",
)
IDENTITY_COLUMNS = (
    "population", "signal_name", "signal_id", "signal_date", "event_symbol",
)


class PostSignalOutcomeJoinError(ValueError): pass


def _panel_columns(horizons: Sequence[int]) -> list[str]:
    columns = ["symbol", "signal_date"]
    for horizon in horizons:
        columns.extend((f"complete_{horizon}", f"status_{horizon}"))
        columns.extend(f"{metric}_{horizon}" for metric in METRICS)
        if horizon == 20: columns.append("barrier_code_20")
    return columns


def validate_match_identities(events: pd.DataFrame, controls: pd.DataFrame) -> None:
    required_events = {*IDENTITY_COLUMNS, "control_count", "full_match"}
    required_controls = {*IDENTITY_COLUMNS, "control_symbol", "match_rank", "selection_digest"}
    if not required_events.issubset(events) or not required_controls.issubset(controls):
        raise PostSignalOutcomeJoinError("matched identity columns are incomplete")
    if events.duplicated(list(IDENTITY_COLUMNS)).any(): raise PostSignalOutcomeJoinError("duplicate event identity")
    if controls.duplicated([*IDENTITY_COLUMNS, "match_rank"]).any(): raise PostSignalOutcomeJoinError("duplicate control rank")
    counts = controls.groupby(list(IDENTITY_COLUMNS), sort=False).size()
    expected = events.set_index(list(IDENTITY_COLUMNS)).control_count.astype(int)
    if not counts.reindex(expected.index, fill_value=0).equals(expected):
        raise PostSignalOutcomeJoinError("control count differs from event identity")
    if not events.control_count.eq(5).all() or not events.full_match.all():
        raise PostSignalOutcomeJoinError("final outcome join requires five frozen controls per event")
    ranks = controls.groupby(list(IDENTITY_COLUMNS), sort=False).match_rank.agg(list)
    if any(sorted(int(value) for value in row) != [1, 2, 3, 4, 5] for row in ranks):
        raise PostSignalOutcomeJoinError("control ranks differ")


def join_year_outcomes(
    events: pd.DataFrame, controls: pd.DataFrame, panel: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join outcomes without filtering or changing any frozen event/control identity."""
    horizons = tuple(int(value) for value in horizons)
    if horizons != HORIZONS: raise PostSignalOutcomeJoinError("outcome horizons differ")
    validate_match_identities(events, controls)
    columns = _panel_columns(horizons)
    if not set(columns).issubset(panel): raise PostSignalOutcomeJoinError("panel outcome columns are incomplete")
    lookup = panel.loc[:, columns].copy()
    if lookup.duplicated(["signal_date", "symbol"]).any(): raise PostSignalOutcomeJoinError("duplicate panel outcome identity")
    event_subjects = events.loc[:, list(IDENTITY_COLUMNS)].copy()
    event_subjects["subject_role"] = "event"; event_subjects["subject_symbol"] = event_subjects.event_symbol
    event_subjects["match_rank"] = 0; event_subjects["selection_digest"] = ""
    control_subjects = controls.loc[:, [*IDENTITY_COLUMNS, "control_symbol", "match_rank", "selection_digest"]].copy()
    control_subjects["subject_role"] = "control"; control_subjects["subject_symbol"] = control_subjects.pop("control_symbol")
    subjects = pd.concat([event_subjects, control_subjects], ignore_index=True)
    joined = subjects.merge(lookup, left_on=["signal_date", "subject_symbol"],
                            right_on=["signal_date", "symbol"], how="left", validate="many_to_one").drop(columns="symbol")
    if len(joined) != len(events) + len(controls): raise PostSignalOutcomeJoinError("subject identity was lost during join")
    rows = []
    for horizon in horizons:
        selected = joined.loc[:, [*IDENTITY_COLUMNS, "subject_role", "subject_symbol", "match_rank", "selection_digest"]].copy()
        selected["horizon_sessions"] = horizon
        selected["complete"] = joined[f"complete_{horizon}"].fillna(False).astype(bool)
        selected["status"] = joined[f"status_{horizon}"].fillna("panel_identity_absent").astype(str)
        for metric in METRICS: selected[metric] = joined[f"{metric}_{horizon}"].astype(float)
        selected["barrier_code"] = joined["barrier_code_20"].astype("Int8") if horizon == 20 else pd.Series(pd.NA, index=joined.index, dtype="Int8")
        rows.append(selected)
    subject_outcomes = pd.concat(rows, ignore_index=True)
    paired_rows = []
    for horizon, current in subject_outcomes.groupby("horizon_sessions", sort=True):
        event = current.loc[current.subject_role.eq("event")].set_index(list(IDENTITY_COLUMNS))
        controls_by_event = current.loc[current.subject_role.eq("control")].groupby(list(IDENTITY_COLUMNS), sort=False)
        control_complete = controls_by_event.complete.sum().reindex(event.index, fill_value=0).astype(int)
        paired = event.reset_index().loc[:, list(IDENTITY_COLUMNS)].copy(); paired["horizon_sessions"] = int(horizon)
        paired["event_complete"] = event.complete.to_numpy(bool); paired["complete_control_count"] = control_complete.to_numpy(int)
        paired["paired_complete"] = paired.event_complete & paired.complete_control_count.eq(5)
        paired["event_status"] = event.status.to_numpy(str)
        for metric in METRICS:
            control_mean = controls_by_event[metric].mean().reindex(event.index)
            paired[f"event_{metric}"] = event[metric].to_numpy(float)
            paired[f"mean_control_{metric}"] = control_mean.to_numpy(float)
            paired[f"paired_{metric}_difference"] = paired[f"event_{metric}"] - paired[f"mean_control_{metric}"]
            paired.loc[~paired.paired_complete, [f"mean_control_{metric}", f"paired_{metric}_difference"]] = np.nan
        paired_rows.append(paired)
    paired_outcomes = pd.concat(paired_rows, ignore_index=True)
    coverage = paired_outcomes.groupby(["population", "signal_name", "horizon_sessions"], sort=True).agg(
        event_rows=("signal_id", "size"), event_complete_rows=("event_complete", "sum"),
        five_control_complete_rows=("complete_control_count", lambda values: int(np.sum(np.asarray(values) == 5))),
        paired_complete_rows=("paired_complete", "sum"),
    ).reset_index()
    coverage["paired_complete_fraction"] = coverage.paired_complete_rows / coverage.event_rows
    return subject_outcomes, paired_outcomes, coverage


def identity_projection(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the ordered immutable projection used by the mutation invariant."""
    columns = [*IDENTITY_COLUMNS, "subject_role", "subject_symbol", "match_rank", "selection_digest", "horizon_sessions"]
    if not set(columns).issubset(frame): raise PostSignalOutcomeJoinError("identity projection columns are incomplete")
    return frame.loc[:, columns].reset_index(drop=True)

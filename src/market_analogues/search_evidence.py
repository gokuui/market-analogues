"""Portable, post-retrieval evidence construction for user-facing searches.

This module deliberately receives an already frozen match list.  It has no search
or representation imports, so future outcomes cannot influence neighbour
selection.  A benchmark enables the stricter market-calendar and relative-return
contract; benchmark-free datasets still receive explicitly qualified stock-only
outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .causal_outcomes import (
    HORIZONS as CAUSAL_HORIZONS,
    compute_episode_outcomes,
    outcome_embargo,
)
from .outcomes import compute_outcomes
from .types import AnalogueMatch, stable_hash


DISPLAY_HORIZONS = (5, 20, 60)
MINIMUM_DESCRIPTIVE_SAMPLE = 10
SEARCH_EVIDENCE_CONTRACT = {
    "schema_version": "portable-search-evidence-v1",
    "retrieval_boundary": "matches_are_immutable_before_outcomes",
    "horizons_sessions": list(DISPLAY_HORIZONS),
    "strict_calendar_source": "configured_benchmark",
    "benchmark_absence_policy": "stock_calendar_only_with_explicit_warning",
    "outcome_embargo": "completion_timestamp_lte_query_cutoff",
    "minimum_descriptive_sample": MINIMUM_DESCRIPTIVE_SAMPLE,
    "claims": "descriptive_only_not_forecast_or_trading_signal",
}
SEARCH_EVIDENCE_CONTRACT_DIGEST = stable_hash(SEARCH_EVIDENCE_CONTRACT)


@dataclass(frozen=True)
class SearchEvidence:
    rows: pd.DataFrame
    summary: pd.DataFrame
    contract_digest: str
    retrieval_identity_digest: str
    outcome_digest: str
    benchmark_available: bool


def _finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _identity_projection(matches: Sequence[AnalogueMatch]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "episode_id": match.episode_key.id,
            "dataset": match.episode_key.instrument.dataset_id,
            "symbol": match.episode_key.instrument.source_symbol,
            "cutoff": pd.Timestamp(match.episode_key.cutoff).isoformat(),
            "lookback": int(match.episode_key.lookback),
            "representation_version": match.episode_key.representation_version,
            "total_distance_hex": float(match.total_distance).hex(),
            "component_distances_hex": {
                key: float(value).hex()
                for key, value in sorted(match.component_distances.items())
            },
        }
        for rank, match in enumerate(matches, 1)
    ]


def retrieval_identity_digest(matches: Sequence[AnalogueMatch]) -> str:
    """Digest only immutable retrieval output, never any future value."""
    return stable_hash(_identity_projection(matches))


def _base_row(rank: int, match: AnalogueMatch, horizon: int) -> dict[str, Any]:
    return {
        "match_rank": rank,
        "matched_dataset": match.episode_key.instrument.dataset_id,
        "matched_symbol": match.episode_key.instrument.source_symbol,
        "matched_episode_id": match.episode_key.id,
        "matched_cutoff": pd.Timestamp(match.episode_key.cutoff).isoformat(),
        "total_distance": float(match.total_distance),
        "horizon_sessions": int(horizon),
    }


def _strict_rows(
    source: OHLCVSource,
    matches: Sequence[AnalogueMatch],
    benchmark: pd.DataFrame,
    query_cutoff: pd.Timestamp,
    source_content_digest: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, match in enumerate(matches, 1):
        key = match.episode_key.instrument
        fingerprint = source.fingerprint(key)
        bundle = compute_episode_outcomes(
            source.load(key), benchmark,
            episode_id=match.episode_key.id,
            cutoff=match.episode_key.cutoff,
            source_fingerprint=fingerprint,
            contract_digest=SEARCH_EVIDENCE_CONTRACT_DIGEST,
            source_content_digest=source_content_digest,
            horizons=CAUSAL_HORIZONS,
        )
        indexed = bundle.outcomes.set_index("horizon_sessions")
        for horizon in DISPLAY_HORIZONS:
            outcome = indexed.loc[horizon]
            observable, reason = outcome_embargo(
                outcome.completion_timestamp, query_cutoff,
                complete=bool(outcome.complete),
            )
            rows.append({
                **_base_row(rank, match, horizon),
                "calendar_validation": "configured_benchmark",
                "complete": bool(outcome.complete),
                "outcome_status": str(outcome.status),
                "completion_timestamp": outcome.completion_timestamp,
                "observable_at_query": observable,
                "eligibility_reason": reason,
                "available_sessions": int(outcome.available_sessions),
                "close_return": _finite_or_none(outcome.close_return),
                "benchmark_relative_return": _finite_or_none(
                    outcome.benchmark_relative_return,
                ),
                "maximum_favorable_excursion": _finite_or_none(
                    outcome.maximum_favorable_excursion,
                ),
                "maximum_adverse_excursion": _finite_or_none(
                    outcome.maximum_adverse_excursion,
                ),
                "barrier_label": (
                    None if pd.isna(outcome.barrier_label)
                    else str(outcome.barrier_label)
                ),
            })
    return rows


def _stock_only_rows(
    source: OHLCVSource,
    matches: Sequence[AnalogueMatch],
    query_cutoff: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, match in enumerate(matches, 1):
        bars = source.load(match.episode_key.instrument).sort_values("timestamp").reset_index(drop=True)
        outcomes = compute_outcomes(
            bars, match.episode_key.cutoff, horizons=DISPLAY_HORIZONS,
        ).set_index("horizon_bars")
        origin_positions = bars.index[bars["timestamp"] <= pd.Timestamp(match.episode_key.cutoff)]
        origin = int(origin_positions[-1])
        for horizon in DISPLAY_HORIZONS:
            outcome = outcomes.loc[horizon]
            completion = None
            if not bool(outcome.censored):
                completion = pd.Timestamp(bars.iloc[origin + horizon]["timestamp"])
            observable = bool(
                completion is not None and completion <= query_cutoff
            )
            reason = (
                "eligible_stock_calendar_only" if observable
                else "incomplete_horizon" if bool(outcome.censored)
                else "outcome_not_yet_observable"
            )
            rows.append({
                **_base_row(rank, match, horizon),
                "calendar_validation": "unavailable_without_benchmark",
                "complete": not bool(outcome.censored),
                "outcome_status": (
                    "complete_stock_calendar_only" if not bool(outcome.censored)
                    else "source_end_before_horizon"
                ),
                "completion_timestamp": completion.isoformat() if completion is not None else None,
                "observable_at_query": observable,
                "eligibility_reason": reason,
                "available_sessions": int(outcome.available_bars),
                "close_return": _finite_or_none(outcome.forward_return),
                "benchmark_relative_return": None,
                "maximum_favorable_excursion": _finite_or_none(
                    outcome.max_favorable_excursion,
                ),
                "maximum_adverse_excursion": _finite_or_none(
                    outcome.max_adverse_excursion,
                ),
                "barrier_label": None,
            })
    return rows


def summarize_search_evidence(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for horizon in DISPLAY_HORIZONS:
        group = rows.loc[rows["horizon_sessions"] == horizon]
        eligible = group.loc[group["observable_at_query"] & group["complete"]]
        relative = eligible["benchmark_relative_return"].dropna().astype(float)
        returns = eligible["close_return"].dropna().astype(float)
        barriers = eligible["barrier_label"].dropna()
        summaries.append({
            "horizon_sessions": horizon,
            "retrieved_matches": int(len(group)),
            "complete_outcomes": int(group["complete"].sum()),
            "eligible_outcomes": int(len(eligible)),
            "not_yet_observable": int(
                (group["eligibility_reason"] == "outcome_not_yet_observable").sum()
            ),
            "median_return": float(returns.median()) if len(returns) else np.nan,
            "return_q25": float(returns.quantile(.25)) if len(returns) else np.nan,
            "return_q75": float(returns.quantile(.75)) if len(returns) else np.nan,
            "positive_rate": float((returns > 0).mean()) if len(returns) else np.nan,
            "gain_25pct_rate": float((returns >= .25).mean()) if len(returns) else np.nan,
            "benchmark_sample_size": int(len(relative)),
            "median_benchmark_relative_return": (
                float(relative.median()) if len(relative) else np.nan
            ),
            "median_mfe": (
                float(eligible["maximum_favorable_excursion"].dropna().median())
                if eligible["maximum_favorable_excursion"].notna().any() else np.nan
            ),
            "median_mae": (
                float(eligible["maximum_adverse_excursion"].dropna().median())
                if eligible["maximum_adverse_excursion"].notna().any() else np.nan
            ),
            "favorable_first_rate": (
                float((barriers == "favorable_first").mean()) if len(barriers) else np.nan
            ),
            "adverse_first_rate": (
                float((barriers == "adverse_first").mean()) if len(barriers) else np.nan
            ),
            "minimum_descriptive_sample_met": len(eligible) >= MINIMUM_DESCRIPTIVE_SAMPLE,
        })
    return pd.DataFrame(summaries)


def build_search_evidence(
    source: OHLCVSource,
    matches: Sequence[AnalogueMatch],
    *,
    query_cutoff: pd.Timestamp | str,
) -> SearchEvidence:
    """Join outcomes after retrieval and produce explicit coverage diagnostics."""
    cutoff = pd.Timestamp(query_cutoff)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    identity_digest = retrieval_identity_digest(matches)
    benchmark = source.load_benchmark()
    benchmark_fingerprint = source.benchmark_fingerprint()
    source_content_digest = stable_hash({
        "retrieval_identity_digest": identity_digest,
        "benchmark_fingerprint": benchmark_fingerprint,
        "source_fingerprints": sorted({
            str(match.episode_key.instrument): source.fingerprint(match.episode_key.instrument)
            for match in matches
        }.items()),
    })
    records = (
        _strict_rows(source, matches, benchmark, cutoff, source_content_digest)
        if benchmark is not None
        else _stock_only_rows(source, matches, cutoff)
    )
    rows = pd.DataFrame(records)
    if rows.empty:
        rows = pd.DataFrame(columns=[
            "match_rank", "matched_dataset", "matched_symbol", "matched_episode_id",
            "matched_cutoff", "total_distance", "horizon_sessions",
            "calendar_validation", "complete", "outcome_status",
            "completion_timestamp", "observable_at_query", "eligibility_reason",
            "available_sessions", "close_return", "benchmark_relative_return",
            "maximum_favorable_excursion", "maximum_adverse_excursion", "barrier_label",
        ])
    summary = summarize_search_evidence(rows) if len(rows) else pd.DataFrame()
    normalized = rows.astype(object).where(pd.notna(rows), None).to_dict("records")
    return SearchEvidence(
        rows=rows,
        summary=summary,
        contract_digest=SEARCH_EVIDENCE_CONTRACT_DIGEST,
        retrieval_identity_digest=identity_digest,
        outcome_digest=stable_hash(normalized),
        benchmark_available=benchmark is not None,
    )

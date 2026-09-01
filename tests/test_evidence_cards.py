from __future__ import annotations

from copy import deepcopy
import inspect
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market_analogues.evidence_cards import (
    EvidenceCardError,
    build_evidence_card,
    rank_weight,
)
from experiments.m04r import m04r14_t14_09_evidence_card_oracle as oracle


def _rows() -> list[dict]:
    result = []
    for rank in range(1, 21):
        symbol = "QUERY" if rank == 1 else ("DUP" if rank in (2, 3) else f"S{rank}")
        outcomes = {}
        eligibility = {}
        for horizon in (5, 10, 20, 40, 60, 126):
            complete = not (horizon == 20 and rank == 4)
            reason = "eligible"
            eligible = complete
            if horizon == 20 and rank == 5:
                eligible = False
                reason = "outcome_not_yet_observable"
            elif not complete:
                reason = "incomplete_horizon"
            label = (
                "censored" if not complete else
                "ambiguous_same_first_touch_bar" if rank == 6 else
                "favorable_first" if rank % 2 == 0 else "adverse_first"
            )
            outcomes[str(horizon)] = {
                "complete": complete, "status": "complete" if complete else "source_end_before_horizon",
                "close_return": rank / 100.0,
                "benchmark_relative_return": rank / 200.0,
                "maximum_favorable_excursion": rank / 50.0,
                "maximum_adverse_excursion": -rank / 100.0,
                "mfe_atr": float(rank), "mae_atr": -rank / 2.0,
                "barrier_label": label if horizon == 20 else None,
            }
            eligibility[str(horizon)] = {"eligible": eligible, "reason": reason}
        result.append({
            "query_case_id": "case", "query_episode_id": "query-episode",
            "query_symbol": "QUERY", "query_cutoff": "2026-03-30T00:00:00",
            "match_rank": rank, "matched_episode_id": f"episode-{rank}",
            "matched_symbol": symbol, "matched_cutoff": f"2020-01-{rank:02d}",
            "total_distance": rank / 10.0,
            "component_distances": {"price": rank / 20.0}, "quality_tier": "A",
            "eligibility_by_horizon": eligibility, "outcomes_by_horizon": outcomes,
            "future_path_episode_reference": f"episode-{rank}",
        })
    return result


def _card(rows=None):
    return build_evidence_card(
        rows or _rows(), contract_digest="contract",
        provenance={"store": "sealed", "registry": "frozen"},
    )


def test_card_preserves_raw_ranks_and_applies_dependence_controls() -> None:
    card = _card()
    assert [row["match_rank"] for row in card["raw_analogue_rows"]] == list(range(1, 21))
    assert card["raw_sample_counts"] == {
        "raw_links": 20, "raw_unique_episodes": 20,
        "raw_unique_symbols": 19, "same_symbol_links": 1,
    }
    primary = card["primary_summary"]
    assert primary["effective_rows_before_horizon_eligibility"] == 18
    assert primary["effective_match_ranks"] == [2, *range(4, 21)]
    assert primary["eligible_by_horizon"]["20"]["eligible_rows"] == 16
    assert 1 in card["same_symbol_panel_ranks"]


def test_card_statistics_and_locked_weights_are_hand_calculated() -> None:
    card = _card()
    eligible_ranks = [2, *range(6, 21)]
    summary = card["primary_summary"]["eligible_by_horizon"]["20"]
    measure = summary["measures"]["close_return"]
    assert measure["count"] == 16
    assert measure["unweighted"]["mean"] == pytest.approx(
        math.fsum(rank / 100 for rank in eligible_ranks) / 16
    )
    weights = [rank_weight(rank) for rank in eligible_ranks]
    assert summary["weighted_effective_sample_size"] == pytest.approx(
        math.fsum(weights) ** 2 / math.fsum(weight * weight for weight in weights)
    )
    assert measure["locked_weighted"]["mean"] == pytest.approx(
        math.fsum(rank / 100 * rank_weight(rank) for rank in eligible_ranks)
        / math.fsum(weights)
    )
    assert measure["unweighted"]["q25_linear"] == pytest.approx(.0875)


def test_ineligible_outcomes_are_redacted_but_censor_status_remains_visible() -> None:
    card = _card()
    censored = card["raw_analogue_rows"][3]["outcomes_by_horizon"]["20"]
    future = card["raw_analogue_rows"][4]["outcomes_by_horizon"]["20"]
    assert censored == {
        "withheld": True, "reason": "incomplete_horizon",
        "status": "source_end_before_horizon", "barrier_label": "censored",
    }
    assert future == {"withheld": True, "reason": "outcome_not_yet_observable"}


def test_card_is_input_order_invariant_and_always_abstains() -> None:
    rows = _rows()
    assert _card(rows)["card_digest"] == _card(list(reversed(rows)))["card_digest"]
    card = _card(rows)
    assert card["predictive_claim_status"] == "abstain"
    assert card["abstention_reasons"] == ["failed_calibration", "poor_data_quality"]


def test_independent_scalar_oracle_matches_complete_card_exactly() -> None:
    rows = _rows()
    expected = oracle.reference_card(
        rows, contract_digest="contract",
        provenance={"store": "sealed", "registry": "frozen"},
    )
    assert _card(rows) == expected
    source = inspect.getsource(oracle)
    assert "market_analogues.evidence_cards" not in source


def test_outcome_mutation_cannot_change_raw_rank_or_primary_selection() -> None:
    rows = _rows()
    before = _card(rows)
    changed = deepcopy(rows)
    changed[9]["outcomes_by_horizon"]["20"]["close_return"] = -9.0
    after = _card(changed)
    assert [row["match_rank"] for row in before["raw_analogue_rows"]] \
        == [row["match_rank"] for row in after["raw_analogue_rows"]]
    assert before["primary_summary"]["effective_rows_before_horizon_eligibility"] \
        == after["primary_summary"]["effective_rows_before_horizon_eligibility"]
    assert before["card_digest"] != after["card_digest"]


def test_invalid_rank_and_horizon_inventories_fail_closed() -> None:
    rows = _rows()
    rows[0]["match_rank"] = 2
    with pytest.raises(EvidenceCardError, match="ranks"):
        _card(rows)
    rows = _rows()
    rows[0]["outcomes_by_horizon"].pop("126")
    with pytest.raises(EvidenceCardError, match="horizon"):
        _card(rows)

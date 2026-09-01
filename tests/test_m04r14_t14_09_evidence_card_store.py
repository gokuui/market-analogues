from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_09_evidence_card_store as store
from experiments.m04r import m04r14_t14_09_evidence_card_synthetic_gate as synthetic
from experiments.m04r import verify_m04r14_t14_09_evidence_card_store as verifier
from market_analogues.evidence_cards import build_evidence_card


def _card():
    rows = synthetic._fixture("mixed")
    for row in rows:
        row.update({
            "match_digest": f"match-{row['match_rank']}",
            "candidate_case_result_digest": "case-result",
            "source_fingerprint": f"source-{row['match_rank']}",
        })
    return build_evidence_card(
        rows, contract_digest="contract",
        provenance={"source": "unit", "store": "sealed"},
    )


def test_store_and_verifier_paths_are_disjoint_and_create_only() -> None:
    assert store.OUTPUT != store.VERIFICATION
    assert store.PREREGISTRATION != store.OUTPUT
    assert store.QUERY_COUNT == 3270
    assert store.LINK_COUNT == 65400


def test_independent_full_card_verifier_has_no_production_aggregation_import() -> None:
    source = inspect.getsource(verifier)
    assert "market_analogues.evidence_cards" not in source
    assert "import m04r14_t14_09_evidence_card_store" not in source
    assert "reference_card" in source


def test_raw_and_summary_projection_preserve_card_bindings() -> None:
    card = _card()
    raw = store._raw_record(card, card["raw_analogue_rows"][0])
    summary = store._summary_record(card)
    assert raw["match_rank"] == 1
    assert raw["card_digest"] == card["card_digest"]
    assert summary["card_digest"] == card["card_digest"]
    assert summary["card_path"] == f"cards/{card['query_episode_id']}.json"
    assert summary["predictive_claim_status"] == "abstain"
    assert "failed_calibration" in summary["abstention_reasons_json"]


def test_searchable_html_escapes_query_content_and_keeps_safety_boundary() -> None:
    row = store._summary_record(_card())
    row["query_symbol"] = "<script>alert(1)</script>"
    value = store._html([row], "sealed")
    assert "<script>alert(1)</script>" not in value
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in value
    assert "Descriptive only." in value
    assert "not forecasts" in value


def test_chunked_store_digest_is_order_invariant_and_matches_verifier() -> None:
    frame = pd.DataFrame({
        "query_episode_id": ["b", "a"], "match_rank": [2, 1],
        "nullable": pd.Series([pd.NA, 3], dtype="Int64"),
    })
    observed = store._frame_digest(frame, ("query_episode_id", "match_rank"))
    assert store._frame_digest(
        frame.iloc[::-1].reset_index(drop=True),
        ("query_episode_id", "match_rank"),
    ) == observed
    assert verifier._frame_digest(
        frame.iloc[::-1].reset_index(drop=True),
        ("query_episode_id", "match_rank"),
    ) == observed


def test_independent_exact_equality_rejects_every_frozen_tamper_class() -> None:
    expected = _card()
    mutations = {}

    value = deepcopy(expected)
    value["raw_analogue_rows"][0]["match_rank"] = 2
    mutations["rank"] = value

    value = deepcopy(expected)
    value["raw_analogue_rows"][1]["matched_symbol"] = "CHANGED"
    mutations["symbol_deduplication"] = value

    value = deepcopy(expected)
    value["raw_analogue_rows"][0]["eligibility_by_horizon"]["20"]["eligible"] = False
    mutations["eligibility"] = value

    value = deepcopy(expected)
    value["raw_analogue_rows"][3]["outcomes_by_horizon"]["20"]["status"] = "changed"
    mutations["censor_state"] = value

    value = deepcopy(expected)
    value["raw_analogue_rows"][5]["outcomes_by_horizon"]["20"]["barrier_label"] = "favorable_first"
    mutations["ambiguity"] = value

    value = deepcopy(expected)
    value["primary_summary"]["primary_barrier_locked_weighted_mass"]["favorable_first"] += 0.1
    mutations["weight"] = value

    value = deepcopy(expected)
    measure = value["primary_summary"]["eligible_by_horizon"]["20"]["measures"]["close_return"]
    measure["unweighted"]["median"] += 0.1
    mutations["quantile"] = value

    value = deepcopy(expected)
    value["raw_analogue_rows"][1]["outcomes_by_horizon"]["20"]["close_return"] = -99.0
    mutations["counterexample"] = value

    value = deepcopy(expected)
    value["abstention_reasons"] = []
    mutations["abstention"] = value

    value = deepcopy(expected)
    value["provenance"]["store"] = "tampered"
    mutations["provenance"] = value

    for name, observed in mutations.items():
        with pytest.raises(verifier.EvidenceVerificationError, match="independent card differs"):
            verifier._require_card_equal(observed, expected)
        assert observed != expected, name

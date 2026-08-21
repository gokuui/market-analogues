from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .types import stable_hash


CONTRACT_SCHEMA_VERSION = "case-memory-contract-v1"
TRIAL_LEDGER_SCHEMA_VERSION = "case-memory-trial-ledger-v1"


class ProductContractError(ValueError):
    pass


@dataclass(frozen=True)
class ProductContract:
    source: Path
    payload: dict[str, Any]
    digest: str

    @property
    def contract_id(self) -> str:
        return str(self.payload["contract_id"])


@dataclass(frozen=True)
class OutcomeEligibility:
    eligible: bool
    reason: str
    candidate_cutoff: pd.Timestamp
    historical_query_timestamp: pd.Timestamp
    outcome_completion_timestamp: pd.Timestamp | None
    horizon_sessions: int
    available_sessions: int


TOP_LEVEL_KEYS = {
    "schema_version", "contract_id", "status", "frozen_before_outcome_evaluation",
    "purpose", "decision", "representation", "retrieval", "outcomes", "evidence",
    "evaluation", "reporting",
}


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductContractError(f"{location} must be a mapping")
    return value


def _require_keys(value: Mapping[str, Any], keys: set[str], location: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ProductContractError(f"{location} missing required keys: {missing}")


def _require_positive_ints(value: Any, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ProductContractError(f"{location} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ProductContractError(f"{location} must contain positive integers")
    if len(set(value)) != len(value):
        raise ProductContractError(f"{location} must not contain duplicates")
    return tuple(value)


def validate_contract(payload: Mapping[str, Any]) -> None:
    unknown = sorted(set(payload) - TOP_LEVEL_KEYS)
    missing = sorted(TOP_LEVEL_KEYS - set(payload))
    if unknown or missing:
        raise ProductContractError(
            f"top-level contract keys differ; missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise ProductContractError(f"unsupported contract schema: {payload['schema_version']!r}")
    if not isinstance(payload["contract_id"], str) or not payload["contract_id"].strip():
        raise ProductContractError("contract_id must be a non-empty string")
    if payload["status"] != "frozen" or payload["frozen_before_outcome_evaluation"] is not True:
        raise ProductContractError("contract must be frozen before outcome evaluation")

    purpose = _require_mapping(payload["purpose"], "purpose")
    _require_keys(purpose, {"question", "intended_user", "output_kind", "prediction_or_advice"}, "purpose")
    if purpose["output_kind"] != "historical_descriptive_evidence":
        raise ProductContractError("purpose.output_kind must remain historical_descriptive_evidence")
    if purpose["prediction_or_advice"] is not False:
        raise ProductContractError("prediction_or_advice must be false")

    decision = _require_mapping(payload["decision"], "decision")
    _require_keys(decision, {"primary_mode", "intended_direction", "holding_horizon_sessions", "modes"}, "decision")
    holding = _require_positive_ints(decision["holding_horizon_sessions"], "decision.holding_horizon_sessions")
    if len(holding) != 2 or holding[0] > holding[1]:
        raise ProductContractError("holding_horizon_sessions must be [minimum, maximum]")
    modes = _require_mapping(decision["modes"], "decision.modes")
    _require_keys(modes, {"after_close_daily", "entry_open", "intraday"}, "decision.modes")
    after_close = _require_mapping(modes["after_close_daily"], "decision.modes.after_close_daily")
    entry_open = _require_mapping(modes["entry_open"], "decision.modes.entry_open")
    intraday = _require_mapping(modes["intraday"], "decision.modes.intraday")
    if decision["primary_mode"] != "after_close_daily":
        raise ProductContractError("M00 primary mode must be after_close_daily")
    if after_close.get("enabled") is not True:
        raise ProductContractError("after_close_daily must be enabled")
    if entry_open.get("enabled") is not False or intraday.get("enabled") is not False:
        raise ProductContractError("entry_open and intraday must remain disabled until their gates pass")

    representation = _require_mapping(payload["representation"], "representation")
    _require_keys(representation, {"required_lookbacks_sessions", "required_views", "missing_context_policy", "future_data_policy"}, "representation")
    lookbacks = _require_positive_ints(representation["required_lookbacks_sessions"], "representation.required_lookbacks_sessions")
    if set(lookbacks) != {5, 10, 21, 63, 126, 252}:
        raise ProductContractError("required lookbacks must be exactly 5, 10, 21, 63, 126 and 252")
    if representation["future_data_policy"] != "forbidden":
        raise ProductContractError("future_data_policy must be forbidden")
    if not isinstance(representation["required_views"], list) or len(representation["required_views"]) < 6:
        raise ProductContractError("at least six representation views are required")

    retrieval = _require_mapping(payload["retrieval"], "retrieval")
    _require_keys(retrieval, {"displayed_neighbors", "minimum_history_gap_sessions", "primary_evidence_max_per_symbol", "same_symbol_memory", "event_cluster_contribution_cap", "market_policy", "similarity_views", "retrieval_must_be_frozen_before_outcomes", "minimum_full_universe_top20_recall"}, "retrieval")
    for key in ("displayed_neighbors", "minimum_history_gap_sessions", "primary_evidence_max_per_symbol", "event_cluster_contribution_cap"):
        value = retrieval[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ProductContractError(f"retrieval.{key} must be a positive integer")
    if retrieval["retrieval_must_be_frozen_before_outcomes"] is not True:
        raise ProductContractError("retrieval must be frozen before outcomes")
    recall = retrieval["minimum_full_universe_top20_recall"]
    if not isinstance(recall, (int, float)) or isinstance(recall, bool) or not 0 < recall <= 1:
        raise ProductContractError("minimum_full_universe_top20_recall must be in (0, 1]")

    outcomes = _require_mapping(payload["outcomes"], "outcomes")
    _require_keys(outcomes, {"primary", "descriptive_horizons_sessions", "required_measures", "incomplete_horizon_policy", "causal_embargo", "outcomes_may_affect_similarity"}, "outcomes")
    primary = _require_mapping(outcomes["primary"], "outcomes.primary")
    _require_keys(primary, {"kind", "horizon_sessions", "favorable_atr", "adverse_atr", "atr_lookback_sessions", "atr_method", "same_bar_both_touched"}, "outcomes.primary")
    if primary["kind"] != "first_touch_atr_barrier" or primary["same_bar_both_touched"] != "ambiguous_excluded":
        raise ProductContractError("primary outcome must use unambiguous first-touch ATR barriers")
    if any(not isinstance(primary[k], (int, float)) or isinstance(primary[k], bool) or primary[k] <= 0 for k in ("horizon_sessions", "favorable_atr", "adverse_atr", "atr_lookback_sessions")):
        raise ProductContractError("primary horizon and ATR parameters must be positive")
    horizons = _require_positive_ints(outcomes["descriptive_horizons_sessions"], "outcomes.descriptive_horizons_sessions")
    if int(primary["horizon_sessions"]) not in horizons:
        raise ProductContractError("primary horizon must be included in descriptive horizons")
    embargo = _require_mapping(outcomes["causal_embargo"], "outcomes.causal_embargo")
    if embargo.get("eligible_when") != "outcome_completion_timestamp_lte_historical_query_timestamp":
        raise ProductContractError("causal outcome embargo rule is missing or changed")
    if outcomes["outcomes_may_affect_similarity"] is not False:
        raise ProductContractError("outcomes may never affect similarity")
    if outcomes["incomplete_horizon_policy"] != "censored_excluded_from_denominator":
        raise ProductContractError("incomplete horizons must be censored and excluded from denominators")

    evidence = _require_mapping(payload["evidence"], "evidence")
    _require_keys(evidence, {"raw_cases_before_summary", "minimum_effective_sample_size", "report_raw_and_effective_sample_size", "report_unweighted_and_locked_weighted_results", "required_sensitivity_panels", "abstain_when", "novelty_threshold_policy"}, "evidence")
    if evidence["raw_cases_before_summary"] is not True:
        raise ProductContractError("raw cases must be shown before summary")
    if evidence["report_raw_and_effective_sample_size"] is not True:
        raise ProductContractError("raw and effective sample sizes must both be reported")
    if not isinstance(evidence["minimum_effective_sample_size"], int) or evidence["minimum_effective_sample_size"] < 2:
        raise ProductContractError("minimum_effective_sample_size must be an integer >= 2")
    required_abstentions = {"insufficient_effective_sample_size", "historically_novel_query", "unstable_neighborhood", "contradictory_context", "poor_data_quality", "failed_calibration"}
    if set(evidence["abstain_when"]) != required_abstentions:
        raise ProductContractError("abstention vocabulary must contain the six locked reasons")

    evaluation = _require_mapping(payload["evaluation"], "evaluation")
    _require_keys(evaluation, {"protocol", "split_policy", "primary_metrics", "required_baselines", "promotion_from_hit_rate_alone", "trial_ledger"}, "evaluation")
    if evaluation["protocol"] != "expanding_window_walk_forward":
        raise ProductContractError("evaluation must use expanding-window walk-forward")
    if evaluation["promotion_from_hit_rate_alone"] != "forbidden":
        raise ProductContractError("promotion from hit rate alone must be forbidden")

    reporting = _require_mapping(payload["reporting"], "reporting")
    _require_keys(reporting, {"formats", "show_counterexamples", "show_component_explanations", "show_provenance_hashes", "allowed_claim", "forbidden_claims"}, "reporting")
    if reporting["show_counterexamples"] is not True or reporting["show_provenance_hashes"] is not True:
        raise ProductContractError("counterexamples and provenance hashes must be shown")


def load_product_contract(path: str | Path) -> ProductContract:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text()) or {}
    mapping = _require_mapping(payload, "contract")
    validate_contract(mapping)
    canonical = json.loads(json.dumps(mapping, sort_keys=True))
    return ProductContract(source, canonical, stable_hash(canonical))


def validate_trial_ledger(path: str | Path, contract: ProductContract) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text()) or {}
    ledger = _require_mapping(payload, "trial ledger")
    _require_keys(ledger, {"schema_version", "contract_id", "contract_digest", "policy", "trials"}, "trial ledger")
    if ledger["schema_version"] != TRIAL_LEDGER_SCHEMA_VERSION:
        raise ProductContractError(f"unsupported trial ledger schema: {ledger['schema_version']!r}")
    if ledger["contract_id"] != contract.contract_id:
        raise ProductContractError("trial ledger contract_id does not match contract")
    if ledger["contract_digest"] != contract.digest:
        raise ProductContractError("trial ledger contract_digest does not match contract")
    policy = _require_mapping(ledger["policy"], "trial ledger policy")
    _require_keys(policy, {"append_only", "record_before_execution", "outcomes_cannot_rewrite_prior_trials"}, "trial ledger policy")
    if not all(policy[key] is True for key in ("append_only", "record_before_execution", "outcomes_cannot_rewrite_prior_trials")):
        raise ProductContractError("all trial ledger safeguards must be true")
    if not isinstance(ledger["trials"], list):
        raise ProductContractError("trial ledger trials must be a list")
    return json.loads(json.dumps(ledger, sort_keys=True, default=str))


def historical_outcome_eligibility(
    bars: pd.DataFrame,
    candidate_cutoff: pd.Timestamp | str,
    horizon_sessions: int,
    historical_query_timestamp: pd.Timestamp | str,
) -> OutcomeEligibility:
    if isinstance(horizon_sessions, bool) or not isinstance(horizon_sessions, int) or horizon_sessions <= 0:
        raise ValueError("horizon_sessions must be a positive integer")
    if "timestamp" not in bars.columns:
        raise ValueError("bars must contain timestamp")
    timestamps = pd.to_datetime(bars["timestamp"], errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("bars contain duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("bars must be sorted by timestamp")
    cutoff = pd.Timestamp(candidate_cutoff)
    query = pd.Timestamp(historical_query_timestamp)
    if cutoff >= query:
        return OutcomeEligibility(False, "analogue_not_earlier", cutoff, query, None, horizon_sessions, 0)
    eligible_positions = bars.index[timestamps <= cutoff]
    if not len(eligible_positions):
        raise ValueError("candidate cutoff precedes available bars")
    origin_position = int(bars.index.get_loc(eligible_positions[-1]))
    future_timestamps = timestamps.iloc[origin_position + 1:origin_position + 1 + horizon_sessions]
    available = len(future_timestamps)
    if available < horizon_sessions:
        return OutcomeEligibility(False, "incomplete_horizon", cutoff, query, None, horizon_sessions, available)
    completion = pd.Timestamp(future_timestamps.iloc[-1])
    if completion > query:
        return OutcomeEligibility(False, "outcome_not_yet_observable", cutoff, query, completion, horizon_sessions, available)
    return OutcomeEligibility(True, "eligible", cutoff, query, completion, horizon_sessions, available)


def write_contract_artifacts(
    contract: ProductContract,
    ledger: Mapping[str, Any],
    machine_path: Path,
    html_path: Path,
) -> tuple[Path, Path]:
    machine_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_digest": contract.digest,
        "contract": contract.payload,
        "trial_ledger": ledger,
    }
    machine_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    purpose = contract.payload["purpose"]
    decision = contract.payload["decision"]
    retrieval = contract.payload["retrieval"]
    outcomes = contract.payload["outcomes"]
    evidence = contract.payload["evidence"]
    reporting = contract.payload["reporting"]
    abstentions = "".join(f"<li><code>{escape(item)}</code></li>" for item in evidence["abstain_when"])
    forbidden = "".join(f"<li><code>{escape(item)}</code></li>" for item in reporting["forbidden_claims"])
    canonical_json = json.dumps(contract.payload, indent=2, sort_keys=True)
    html_path.write_text(f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>M00 Case Memory contract</title><style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;background:#f4f6f7;color:#17202a;line-height:1.55}}header,section{{background:white;border:1px solid #d5d8dc;border-radius:10px;padding:1.25rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.6rem;text-align:left;vertical-align:top;border-bottom:1px solid #ddd}}code{{overflow-wrap:anywhere}}.pass{{color:#117864}}.warning{{border-left:5px solid #9a6700}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f3;padding:1rem}}</style></head>
<body data-contract-digest=\"{contract.digest}\"><header><h1>M00 Case-Based Market Memory Contract: <span class=\"pass\">FROZEN</span></h1><p>{escape(purpose['question'])}</p><p>Contract <code>{escape(contract.contract_id)}</code> · SHA-256 <code>{contract.digest}</code></p></header>
<section><h2>Decision boundary</h2><table><tr><th>Primary mode</th><td>{escape(decision['primary_mode'])}</td></tr><tr><th>Information</th><td>{escape(decision['modes']['after_close_daily']['permitted_information'])}</td></tr><tr><th>Intended use</th><td>{escape(purpose['output_kind'])}</td></tr><tr><th>Prediction/advice</th><td>{str(purpose['prediction_or_advice']).lower()}</td></tr></table></section>
<section><h2>Retrieval boundary</h2><p>Show {retrieval['displayed_neighbors']} cases; primary evidence permits {retrieval['primary_evidence_max_per_symbol']} episode per symbol and {retrieval['event_cluster_contribution_cap']} contribution per event cluster. Outcomes attach only after retrieval is frozen.</p></section>
<section><h2>Primary outcome</h2><p>Within {outcomes['primary']['horizon_sessions']} observed sessions: first touch of +{outcomes['primary']['favorable_atr']} ATR versus −{outcomes['primary']['adverse_atr']} ATR. If both barriers occur in one daily bar, the case is ambiguous and excluded.</p><p><b>Embargo:</b> an outcome contributes at a historical query only when its completion timestamp is no later than that query timestamp. Incomplete horizons are censored.</p></section>
<section><h2>Mandatory abstention reasons</h2><ul>{abstentions}</ul></section>
<section class=\"warning\"><h2>Forbidden claims</h2><ul>{forbidden}</ul></section>
<section><h2>Canonical machine contract</h2><p>This block is rendered directly from the same validated object written to the JSON artifact.</p><pre id=\"canonical-contract\">{escape(canonical_json)}</pre></section></body></html>""")
    return machine_path, html_path

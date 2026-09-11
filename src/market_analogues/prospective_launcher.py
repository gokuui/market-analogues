"""Fail-closed one-shot launcher for prospective analogue batches."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Callable, Mapping


class ProspectiveLaunchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProspectiveLaunchError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def validate_seal(value: Mapping[str, Any], digest_field: str) -> None:
    require(type(value) is dict, "sealed document must be an object")
    require(type(value.get(digest_field)) is str, f"{digest_field} absent")
    state = {key: item for key, item in value.items()
             if key not in {digest_field, "created_at"}}
    require(value[digest_field] == stable(state), f"{digest_field} seal differs")


@dataclass(frozen=True)
class LaunchDecision:
    action: str
    reason: str
    availability_result_digest: str
    availability_verification_digest: str
    registry_callback_calls: int
    prediction_callback_calls: int
    post_freeze_outcomes_opened: bool = False


def decide_launch(
    contract: Mapping[str, Any], availability: Mapping[str, Any],
    verification: Mapping[str, Any], *,
    registry_callback: Callable[[], Any], prediction_callback: Callable[[], Any],
) -> LaunchDecision:
    """Make one decision and never call scientific work behind a blocked receipt."""
    validate_seal(availability, "result_digest")
    validate_seal(verification, "verification_digest")
    upstream = contract["upstream"]
    require(availability["contract_digest"] == upstream["availability_contract_digest"],
            "availability contract differs")
    require(verification["producer_result_digest"] == availability["result_digest"],
            "result and verification pair differs")
    require(verification["passed"] is True
            and verification["live_source_matched_at_verification"] is True,
            "independent live verification absent")
    flags = (
        bool(availability["readiness_passed"]),
        bool(availability["registry_creation_authorized"]),
        bool(verification["readiness_passed"]),
        bool(verification["registry_creation_authorized"]),
    )
    require(len(set(flags)) == 1, "readiness flags disagree")
    require(availability["source_values_opened"] is False
            and availability["post_freeze_outcomes_opened"] is False
            and verification["source_values_opened"] is False
            and verification["post_freeze_outcomes_opened"] is False,
            "preflight access boundary differs")
    if not flags[0]:
        require(availability["result_digest"] == upstream["blocked_result_digest"]
                and verification["verification_digest"]
                == upstream["blocked_verification_digest"],
                "blocked receipt is not the frozen verified pair")
        return LaunchDecision(
            action=contract["launcher"]["blocked_action"],
            reason="independently_verified_source_extension_required",
            availability_result_digest=availability["result_digest"],
            availability_verification_digest=verification["verification_digest"],
            registry_callback_calls=0, prediction_callback_calls=0,
        )
    # A future ready contract must replace the frozen blocked identities. This
    # prevents an old contract from being repurposed after new source data arrives.
    require(availability["result_digest"] != upstream["blocked_result_digest"]
            and verification["verification_digest"]
            != upstream["blocked_verification_digest"],
            "blocked identities cannot authorize a ready launch")
    registry_callback()
    prediction_callback()
    return LaunchDecision(
        action=contract["launcher"]["ready_action"],
        reason="independently_verified_source_is_ready",
        availability_result_digest=availability["result_digest"],
        availability_verification_digest=verification["verification_digest"],
        registry_callback_calls=1, prediction_callback_calls=1,
    )


def refresh_assessment(
    contract: Mapping[str, Any], *, credential_present: bool,
    free_bytes: int, canonical_stock_exists: bool,
    canonical_benchmark_exists: bool,
) -> dict[str, Any]:
    """Return metadata-only prerequisites; never expose a credential value."""
    refresh = contract["refresh"]
    require(type(credential_present) is bool and type(free_bytes) is int,
            "refresh observation types differ")
    checks = {
        "credential_present": credential_present,
        "capacity_passed": free_bytes >= int(refresh["minimum_free_bytes"]),
        "canonical_stock_source_present": canonical_stock_exists,
        "canonical_benchmark_present": canonical_benchmark_exists,
        "non_destructive_staging_required": True,
    }
    blockers = []
    if not checks["credential_present"]:
        blockers.append("required_provider_credential_absent")
    if not checks["capacity_passed"]:
        blockers.append("minimum_staging_capacity_absent")
    if not checks["canonical_stock_source_present"]:
        blockers.append("canonical_stock_source_absent")
    if not checks["canonical_benchmark_present"]:
        blockers.append("canonical_benchmark_absent")
    return {
        "provider": refresh["provider"],
        "credential_environment_variable": refresh["credential_environment_variable"],
        "credential_value_recorded": False,
        "observed_free_bytes": free_bytes,
        "minimum_free_bytes": int(refresh["minimum_free_bytes"]),
        "checks": checks,
        "ready_for_non_destructive_refresh": not blockers,
        "blocking_reasons": blockers,
        "canonical_inputs_modified": False,
        "fallback_provider_authorized": False,
    }


def decision_document(decision: LaunchDecision) -> dict[str, Any]:
    return {"schema_version": "prospective-real-source-launch-decision-v1",
            **asdict(decision)}

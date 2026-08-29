"""Machine-readable compatibility contract for the T14-03 exposed-60 run.

Coordination assumptions
------------------------
This module is a data contract, not a producer or evidence validator.  It may
be imported by the producer, preregistration tooling, and the independently
implemented verifier so that filenames, schemas, key sets, and fixed policy do
not drift.  The producer and verifier must still reconstruct evidence
independently and must not share their validation functions.

The 60 query IDs and the T14-02 bindings below come only from already-opened,
development evidence.  This module never accepts or reads authority or forward
outcome paths.  T14-03 is serial, truth-blind, development-only, and may not
authorize production promotion.  ``incomplete_tree`` describes durable prefix
reconstruction after failure; it never authorizes resuming that root.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


DESCRIPTOR_SCHEMA = "m04r14-all60-compatibility-contract-v1"
CANDIDATE_RELATIVE = "config/data/analogues/m04r14/all60-certified-development-v3"
VERIFIER_RELATIVE = (
    "config/data/analogues/m04r14/all60-certified-development-v3-verification"
)
PREREGISTRATION_RELATIVE = "experiments/m04r/m04r14_all60_preregistered_v3.json"

QUERY_IDS = (
    "d107eb49f0ae78d63a90bbce", "f51fc69c1920a76409acc8ff",
    "e20fa3265667c4d84c50aa92", "2429c38cd8e4f7d8ff1af984",
    "942bba10f93b0a36d4d21766", "224498b96b521ad0e84c703d",
    "e3ae797166ab7df0a224edf6",
    "1209f9a230cf2659dfe6c71c", "c4cc388da15e6d2a3619ccd1",
    "dbfd8bfae9e292f7a6945d5a", "068c024bfddc815a4262f8f3",
    "a51854b042fafc711133af00",
    "fa295f0f03b6666ac47bf011", "6b7f0710de564f35dfa71d3d",
    "5d4e4365b63ff184cbf29f9a", "c80df872a9795347564ee29a",
    "853c6d64815c9becb1878228", "6fd6c36d7d39a9465b08fc9b",
    "b84212b6562c9d51d009f43f", "f4bc8e7035177806722271c9",
    "b34beee408d49fe22b176d6e", "331714c210918787c48719a5",
    "1756dc64acfbe4616fa1e468", "05e0decc94668aa341e92eae",
    "fb31be3ab7cd34f020174fa5", "550a9bf7901e6b3a63b066f9",
    "2be08b4139a31a1ff45c3070", "997f016231b5f082f26b80a3",
    "10148b912c4a6ece5185f8f4", "f2af4d63144103d695102676",
    "79a661e61e4f319355476262", "a12a944767c0981848006a4c",
    "d36548afe4a31c02a1c828b5", "f9d5f80116223e7573d268fb",
    "5b10295ad8c367ea359d1c6a", "045010fe96406800141c4a39",
    "8969de452d79d52268bf05b9", "162c15ddf02b995547213a4a",
    "82fb0ed4212582ab1d7956c2", "64264271373980d5a7396b36",
    "68d9e68089e2fe4faeaa2660", "ba410435729adf180fb80b32",
    "bf6a4e286690a6c1670cfd6f", "3307023dbe2164d025e788da",
    "3618af07dedd52fb3bdb1ccd", "9d7365581643bd93e85beb67",
    "9d8507e422a0c5f7d5560e5d", "e497029808c5d57fbc801477",
    "99a0838725a09570b4a075ff", "58f35f93b18f5380251843ba",
    "a3002181e588bd9d95d8aae6", "9d56dcfa047f6aa21d3fc22d",
    "cbed786382137cff12507bc6", "61de02bac1422cbee2bddc4e",
    "67da5d1bccf2c9fb4826e89d", "9c6efffcbe5e7cbd5d464654",
    "9124908ee82121f1affd5981", "79c1f9d3a9dad9e29769dfd2",
    "678af00eb7c43f18e64de828", "2d5fd014631f9d4ce97ac34d",
)

SCHEMAS = {
    "contract": "m04r14-all60-preregistration-v1",
    "run": "m04r14-all60-run-started-v1",
    "source": "m04r14-all60-source-lock-v1",
    "resident": "m04r14-all60-resident-binding-v1",
    "event": "m04r14-all60-ledger-event-v1",
    "proposal": "m04r14-all60-proposal-v1",
    "exact": "m04r14-all60-exact-v1",
    "case_semantic": "m04r14-all60-case-semantic-v1",
    "case_measurement": "m04r14-all60-case-measurement-v1",
    "case_bundle": "m04r14-all60-case-bundle-v1",
    "semantics": "m04r14-all60-semantics-v1",
    "measurements": "m04r14-all60-measurements-v1",
    "complete": "m04r14-all60-complete-v1",
    "incomplete": "m04r14-all60-incomplete-v1",
    "verifier_receipt": "m04r14-all60-independent-verification-v1",
}


def _keys(*names: str) -> tuple[str, ...]:
    return tuple(sorted(names))


FIELD_KEYS = {
    "contract": _keys(
        "schema_version", "status", "runtime_binding", "roots", "query_ids",
        "execution_policy", "t14_02_binding", "claims", "preregistration_digest",
    ),
    "run": _keys(
        "schema_version", "status", "preregistration_digest", "query_ids",
        "execution_policy", "claims", "created_at", "result_digest",
    ),
    "source": _keys(
        "schema_version", "config_sha256", "registry_sha256", "registry_digest",
        "generation_id", "provenance_digest", "source_tree_digest",
        "query_binding_digests", "created_at", "result_digest",
    ),
    "resident": _keys(
        "schema_version", "ready_digest", "ready_file_sha256", "content_digest",
        "seal_digest", "lease", "store_root", "identity_digest", "created_at",
        "result_digest",
    ),
    "event": _keys(
        "schema_version", "sequence", "event", "ordinal", "query_id",
        "previous_event_digest", "case_bundle_sha256", "created_at", "event_digest",
    ),
    # The exact child authenticates this persisted wrapper by SHA before it
    # reads the sealed state.  Traversal timing remains in ``measurement``;
    # CASE.json later projects a separate timing-free semantic document.
    "proposal": _keys("state", "digest", "measurement", "created_at"),
    "exact": _keys(
        "schema_version", "ordinal", "query_id", "case_id", "workers",
        "proposal_sha256", "semantic", "measurement", "semantic_digest",
        "measurement_digest", "created_at", "result_digest",
    ),
    "case_semantic": _keys(
        "schema_version", "ordinal", "query_id", "case_id", "query_binding",
        "forward_proposal", "reverse_proposal", "proposal_parity",
        "certificate", "matches", "source_leases", "resident_leases",
        "semantic_digest",
    ),
    "case_measurement": _keys(
        "schema_version", "ordinal", "query_id", "case_id", "child_process",
        "forward_proposal_seconds", "reverse_proposal_seconds",
        "exact_stage_seconds", "end_to_end_seconds", "resources",
        "proposal_resource_gate_passed", "exact_stage_slo_passed",
        "end_to_end_slo_passed", "created_at", "measurement_digest",
    ),
    "case_bundle": _keys(
        "schema_version", "ordinal", "query_id", "case_id", "proposal_sha256",
        "exact_sha256", "semantic", "measurement", "semantic_digest",
        "measurement_digest", "created_at", "bundle_digest",
    ),
    "semantics": _keys(
        "schema_version", "status", "ordered_case_semantic_digests",
        "forward_reverse_equal", "all_certified", "all_finite",
        "semantic_passed", "claims", "created_at", "semantic_digest",
    ),
    "measurements": _keys(
        "schema_version", "status", "ordered_case_measurement_digests",
        "raw_case_metrics", "summary", "resource_gates", "slo_gates",
        "performance_passed", "created_at", "measurement_digest",
    ),
    "complete": _keys(
        "schema_version", "status", "preregistration_digest", "ledger_head_digest",
        "semantic_digest", "measurement_digest", "final_source_lease",
        "final_resident_lease", "leaf_manifest", "leaf_manifest_digest",
        "semantic_passed", "performance_passed", "development_only",
        "production_promotion_authorized", "created_at", "complete_digest",
    ),
    "incomplete": _keys(
        "schema_version", "status", "preregistration_digest", "failure_class",
        "failure_message", "completed_cases", "trailing_query_id", "trailing_stage",
        "ledger_head_digest", "partial_tree_manifest", "partial_tree_digest",
        "resume_authorized", "authority_open_authorized", "created_at",
        "incomplete_digest",
    ),
    "verifier_receipt": _keys(
        "schema_version", "status", "passed", "candidate_root",
        "candidate_complete_sha256", "candidate_complete_digest",
        "terminal_tree_digest", "semantic_digest", "measurement_digest",
        "verified_cases", "direct_raw_authority_accessed_by_verifier",
        "authority_derived_prerequisite_evidence_accessed_by_verifier",
        "forward_outcomes_accessed_by_verifier", "development_only",
        "production_promotion_authorized", "result_digest", "created_at",
    ),
}

CLAIMS = {
    "development_only": True,
    "cases_previously_exposed": True,
    "direct_raw_authority_accessed_by_this_run": False,
    "authority_derived_prerequisite_evidence_accessed_by_this_run": True,
    "forward_outcomes_accessed_by_this_run": False,
    "raw_authority_or_outcome_paths_accepted": False,
    "production_promotion_authorized": False,
}

T14_02_BINDING = {
    "candidate_root": "config/data/analogues/m04r14/exact-scheduler-poc-v2",
    "verification_root": (
        "config/data/analogues/m04r14/exact-scheduler-poc-v2-verification"
    ),
    "selected_exact_workers": 1,
    "candidate_complete_sha256": "e074be8b302764d9a3333635846819dfc81971f2fb0597f4f9cd50f620e54a54",
    "candidate_complete_digest": "b42781637ebb0861e4776038485bf3787e86cb94ceeacfbf35824a92b16635ee",
    "semantic_sha256": "08ca7aa7d0b1767d9902c385a4113000c72c13761102810e455d09705e7ca946",
    "semantic_digest": "d2e898148f34afc8cd0a080966f6cf4d537f280ca8c12e4dd0e771e131c374d5",
    "measurement_sha256": "8953b48f4e695dda57cbe0ad14911d2b42b3b22fcbe482ae5aa85501f77928c3",
    "measurement_digest": "da833b6ba0fde2cfdc50f8975df93624116041909f10390f19675f05a02eb2a6",
    "verification_sha256": "df39d90f43afe2ad9c9af1c96714eb34a202b2f97245f6d2ff7e306cf3240dc6",
    "verification_result_digest": "9de54c0e0b6737251d27bf73210e6119c80b66f2ff37624a1bcec74afd2e2ddd",
}

EXECUTION_POLICY = {
    "cases": 60,
    "exact_workers": T14_02_BINDING["selected_exact_workers"],
    "proposal_threads": 8,
    "case_concurrency": 1,
    "child_policy": "two-serial-fresh-spawned-children-per-case-registry-order",
    "proposal_child_cpus": 8,
    "exact_child_cpus": 1,
    "proposal_must_persist_before_exact_spawn": True,
    "run_hard_limit_seconds": 7200,
    "performance_limits": {
        "forward_proposal_seconds": 120.0,
        "reverse_proposal_seconds": 60.0,
        "proposal_process_rss_mb": 1536.0,
        "exact_stage_p95_seconds": 120.0,
        "exact_stage_max_seconds": 240.0,
        "end_to_end_p95_seconds": 180.0,
        "end_to_end_max_seconds": 300.0,
        "p95_method": "nearest-rank-ceiling",
        "p95_order_index_zero_based_for_60": 56,
    },
    "semantic_policy": "exact-timing-free",
    "measurement_policy": "separate-and-semantic-digest-bound",
    "forward_reverse_proposal_equality": "exact",
    "terminal_policy": "create-only-complete-last",
    "resume": False,
    "recovery": "validate-and-seal-gap-free-prefix-only-never-continue",
    "case_started_before_child_spawn": True,
    "case_completed_after_bundle_fsync": True,
}

TIMING_FIELDS = (
    "created_at", "elapsed_seconds", "end_to_end_seconds", "engine_seconds",
    "exact_stage_seconds", "forward_proposal_seconds", "measurement",
    "measurement_digest", "process", "resources", "reverse_proposal_seconds",
    "wall_seconds",
)

FOUNDATION_PATHS = (
    "CONTRACT.json", "RUN_STARTED.json", "SOURCE_LOCK.json", "RESIDENT.json",
)
AGGREGATE_PATHS = ("SEMANTICS.json", "MEASUREMENTS.json")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def stable_digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def validate_query_ids(query_ids: Sequence[str]) -> tuple[str, ...]:
    observed = tuple(query_ids)
    if observed != QUERY_IDS:
        raise ValueError("query IDs must equal the frozen registry order")
    if len(observed) != 60 or len(set(observed)) != 60:
        raise ValueError("query IDs must contain exactly 60 unique values")
    if any(type(value) is not str or len(value) != 24 for value in observed):
        raise ValueError("query ID encoding differs")
    return observed


def _event_path(sequence: int, event: str, query_id: str) -> str:
    return f"events/{sequence:03d}-{event}-{query_id}.json"


def _case_path(ordinal: int, query_id: str, leaf: str) -> str:
    return f"cases/{ordinal:03d}-{query_id}/{leaf}"


def successful_tree(query_ids: Sequence[str] = QUERY_IDS) -> tuple[str, ...]:
    ordered = validate_query_ids(query_ids)
    paths = list(FOUNDATION_PATHS)
    for ordinal, query_id in enumerate(ordered):
        paths.extend((
            _event_path(ordinal * 2, "CASE_STARTED", query_id),
            _case_path(ordinal, query_id, "PROPOSAL.json"),
            _case_path(ordinal, query_id, "EXACT-w1.json"),
            _case_path(ordinal, query_id, "CASE.json"),
            _event_path(ordinal * 2 + 1, "CASE_COMPLETED", query_id),
        ))
    paths.extend((*AGGREGATE_PATHS, "COMPLETE.json"))
    return tuple(sorted(paths))


def successful_directories(
    query_ids: Sequence[str] = QUERY_IDS,
) -> tuple[str, ...]:
    ordered = validate_query_ids(query_ids)
    return tuple(sorted(("cases", "events", *(
        f"cases/{ordinal:03d}-{query_id}"
        for ordinal, query_id in enumerate(ordered)
    ))))


def incomplete_directories(
    completed_cases: int,
    *,
    trailing_stage: str | None = None,
    query_ids: Sequence[str] = QUERY_IDS,
) -> tuple[str, ...]:
    ordered = validate_query_ids(query_ids)
    # Reuse the file-prefix validator for range/stage invariants.
    incomplete_tree(completed_cases, trailing_stage=trailing_stage,
                    query_ids=ordered)
    count = completed_cases + int(trailing_stage in {"proposal", "exact", "case"})
    return tuple(sorted(("cases", "events", *(
        f"cases/{ordinal:03d}-{query_id}"
        for ordinal, query_id in enumerate(ordered[:count])
    ))))


def incomplete_tree(
    completed_cases: int,
    *,
    trailing_stage: str | None = None,
    foundation_count: int = len(FOUNDATION_PATHS),
    aggregate_count: int = 0,
    query_ids: Sequence[str] = QUERY_IDS,
) -> tuple[str, ...]:
    """Return one valid terminal failure tree; never an execution resume plan.

    Foundation publication is an ordered prefix.  Case acquisition starts only
    after all four foundation files exist.  Aggregate publication starts only
    after all 60 cases complete and is also an ordered prefix.
    """
    ordered = validate_query_ids(query_ids)
    if type(completed_cases) is not int or not 0 <= completed_cases <= 60:
        raise ValueError("completed case prefix differs")
    if trailing_stage not in {None, "started", "proposal", "exact", "case"}:
        raise ValueError("trailing stage differs")
    if type(foundation_count) is not int or not 0 <= foundation_count <= 4:
        raise ValueError("foundation prefix differs")
    if type(aggregate_count) is not int or not 0 <= aggregate_count <= 2:
        raise ValueError("aggregate prefix differs")
    if foundation_count < 4 and (completed_cases or trailing_stage or aggregate_count):
        raise ValueError("case evidence precedes complete foundation")
    if trailing_stage is not None and completed_cases >= 60:
        raise ValueError("trailing case start has no next query")
    if aggregate_count and (completed_cases != 60 or trailing_stage is not None):
        raise ValueError("aggregate evidence precedes complete case prefix")
    paths = list(FOUNDATION_PATHS[:foundation_count])
    for ordinal, query_id in enumerate(ordered[:completed_cases]):
        paths.extend((
            _event_path(ordinal * 2, "CASE_STARTED", query_id),
            _case_path(ordinal, query_id, "PROPOSAL.json"),
            _case_path(ordinal, query_id, "EXACT-w1.json"),
            _case_path(ordinal, query_id, "CASE.json"),
            _event_path(ordinal * 2 + 1, "CASE_COMPLETED", query_id),
        ))
    if trailing_stage is not None:
        query_id = ordered[completed_cases]
        paths.append(_event_path(completed_cases * 2, "CASE_STARTED", query_id))
        stages = {
            "started": (),
            "proposal": ("PROPOSAL.json",),
            "exact": ("PROPOSAL.json", "EXACT-w1.json"),
            "case": ("PROPOSAL.json", "EXACT-w1.json", "CASE.json"),
        }
        paths.extend(
            _case_path(completed_cases, query_id, leaf)
            for leaf in stages[trailing_stage]
        )
    paths.extend(AGGREGATE_PATHS[:aggregate_count])
    paths.append("INCOMPLETE.json")
    return tuple(sorted(paths))


def semantic_projection(case_bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return the timing-free case identity and semantic payload."""
    required = {"ordinal", "query_id", "case_id", "semantic"}
    if not required <= set(case_bundle):
        raise ValueError("case bundle lacks semantic projection fields")
    semantic = case_bundle["semantic"]
    if type(semantic) is not dict or set(semantic) != set(FIELD_KEYS["case_semantic"]):
        raise ValueError("case semantic key set differs")
    def timing_keys(value: Any) -> set[str]:
        if type(value) is dict:
            return (set(value) & set(TIMING_FIELDS)) | set().union(
                *(timing_keys(item) for item in value.values()), set(),
            )
        if type(value) is list:
            return set().union(*(timing_keys(item) for item in value), set())
        return set()

    forbidden = timing_keys(semantic)
    if forbidden:
        raise ValueError(f"timing fields entered semantic payload: {sorted(forbidden)}")
    return {
        "ordinal": case_bundle["ordinal"],
        "query_id": case_bundle["query_id"],
        "case_id": case_bundle["case_id"],
        "semantic": dict(semantic),
    }


DESCRIPTOR = {
    "schema_version": DESCRIPTOR_SCHEMA,
    "paths": {
        "candidate": CANDIDATE_RELATIVE,
        "verifier": VERIFIER_RELATIVE,
        "preregistration": PREREGISTRATION_RELATIVE,
    },
    "schemas": SCHEMAS,
    "field_keys": FIELD_KEYS,
    "query_ids": QUERY_IDS,
    "claims": CLAIMS,
    "execution_policy": EXECUTION_POLICY,
    "timing_fields_excluded_from_semantics": TIMING_FIELDS,
    "t14_02_binding": T14_02_BINDING,
    "successful_tree": successful_tree(),
    "successful_directories": successful_directories(),
    "prefix_rules": {
        "foundation_order": FOUNDATION_PATHS,
        "event_order": ("CASE_STARTED", "CASE_COMPLETED"),
        "aggregate_order": AGGREGATE_PATHS,
        "maximum_completed_cases": 60,
        "maximum_open_started_cases": 1,
        "resume_authorized": False,
        "incomplete_is_terminal": True,
        "complete_and_incomplete_mutually_exclusive": True,
    },
}
DESCRIPTOR_DIGEST = stable_digest(DESCRIPTOR)


__all__ = (
    "AGGREGATE_PATHS", "CANDIDATE_RELATIVE", "CLAIMS", "DESCRIPTOR",
    "DESCRIPTOR_DIGEST", "DESCRIPTOR_SCHEMA", "EXECUTION_POLICY", "FIELD_KEYS",
    "FOUNDATION_PATHS", "PREREGISTRATION_RELATIVE", "QUERY_IDS", "SCHEMAS",
    "T14_02_BINDING", "TIMING_FIELDS", "VERIFIER_RELATIVE", "canonical_bytes",
    "incomplete_directories", "incomplete_tree", "semantic_projection",
    "stable_digest", "successful_directories", "successful_tree",
    "validate_query_ids",
)

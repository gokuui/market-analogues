"""Support-only second stage for the sealed R1-B v3 query partition."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from experiments.m04r import m04r14_r1a_exposure_audit as r1a
from experiments.m04r import m04r14_r1b_balanced_partition_v3 as partition
from market_analogues.adequacy_support import causally_eligible, matched_support
from market_analogues.types import stable_hash


SCHEMA = "m04r14-r1b-balanced-support-v3"
CAP = 4096
MIN_COVERAGE = 0.90


class BalancedSupportRunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BalancedSupportRunError(message)


def _summary(name: str, support: Mapping[str, int], cohort: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
    passing = {key for key, value in support.items() if value >= CAP}
    links = sum(len(value) for value in cohort.values())
    supported_links = sum(len(cohort[key]) for key in passing)
    result = {
        "design": name, "supported_episodes": len(passing), "cohort_episodes": len(cohort),
        "episode_coverage": len(passing) / len(cohort), "supported_links": supported_links,
        "cohort_links": links, "link_coverage": supported_links / links,
    }
    result["passes"] = result["episode_coverage"] >= MIN_COVERAGE \
        and result["link_coverage"] >= MIN_COVERAGE
    return result


def _primary_support_pass(designs: Sequence[Mapping[str, Any]]) -> bool:
    primary = [row for row in designs if row.get("role") == "primary"]
    _require(
        len(primary) == 1 and primary[0].get("global_structure_cells") == 8,
        "primary K8 support design differs",
    )
    return primary[0].get("passes") is True


def _design_summary(
    k: int, values: Mapping[str, int], cohort: Mapping[str, tuple[int, ...]],
    details: Mapping[str, Any], crossed_cells: int,
) -> dict[str, Any]:
    result = _summary(f"session_21_structure_{k}", values, cohort)
    result.update({
        "role": "primary" if k == 8 else "secondary_sensitivity",
        "global_structure_cells": k,
        "retained_structure_cells": details["retained_k"],
        "leaf_size_min": min(details["leaf_sizes"]),
        "leaf_size_max": max(details["leaf_sizes"]),
        "crossed_cells": crossed_cells,
    })
    return result


def _validate_partition_artifacts(
    result: Mapping[str, Any], transform: Mapping[str, Any], payload: Mapping[str, Any],
    prereg: Mapping[str, Any], h1: str, registry: Mapping[str, Any], query_ids: Sequence[str],
) -> None:
    expected_result = {
        "adequacy_labels_authorized", "b2_execution_authorized",
        "candidate_or_eligibility_inputs_accessed", "created_at", "partitions_sha256",
        "passed", "predictive_claim_authorized", "preregistration_commit",
        "preregistration_digest", "primary_k", "production_promotion_authorized",
        "queries", "r1b_statistics_opened", "real_forward_outcomes_accessed",
        "registry_digest", "result_digest", "schema_version", "secondary_k",
        "serial_parallel_partition_digest_identical",
        "serial_parallel_reconstruction_identical",
        "serial_parallel_transform_digest_identical", "status", "transform_sha256",
    }
    _require(set(result) == expected_result, "partition result field closure differs")
    _require(all((
        result["schema_version"] == partition.SCHEMA,
        result["status"] in {"partition_valid", "partition_invalid", "reproducibility_failed"},
        result["passed"] is (result["status"] == "partition_valid"),
        result["preregistration_commit"] == h1,
        result["preregistration_digest"] == prereg["preregistration_digest"],
        result["registry_digest"] == registry["registry_digest"],
        result["queries"] == len(query_ids) == 3270,
        result["primary_k"] == 8, result["secondary_k"] == [12, 16],
        result["candidate_or_eligibility_inputs_accessed"] is False,
        result["real_forward_outcomes_accessed"] is False,
        result["r1b_statistics_opened"] is False,
        result["b2_execution_authorized"] is False,
        result["adequacy_labels_authorized"] is False,
        result["predictive_claim_authorized"] is False,
        result["production_promotion_authorized"] is False,
        not result["serial_parallel_reconstruction_identical"] or (
            result["serial_parallel_transform_digest_identical"] is True
            and result["serial_parallel_partition_digest_identical"] is True
        ),
    )), "partition result semantics differ")
    _require(set(payload) == {"assignments", "partitions", "schema_version"}
             and payload["schema_version"] == "m04r14-r1b-balanced-partitions-v1",
             "partition payload closure differs")
    assignments = payload["assignments"]
    _require(len(assignments) == 3270
             and [row.get("query_episode_id") for row in assignments] == list(query_ids)
             and all(set(row) == {"labels", "query_episode_id"} for row in assignments),
             "partition assignment closure differs")
    _require(set(payload["partitions"]) == {"8", "12", "16"}, "partition K closure differs")
    valid_k = {
        int(key) for key, value in payload["partitions"].items()
        if value.get("status") == "partition_valid"
    }
    _require(all(set(row["labels"]) == {str(k) for k in valid_k} for row in assignments),
             "partition label-key closure differs")
    split_keys = {
        "axis_digest", "axis_mode", "boundary_margin_hex", "leading_eigenvalue_hex",
        "left_boundary_hex", "left_child_path", "left_rows", "left_target_leaves",
        "member_ids_digest", "path", "pivot_feature", "relative_eigengap_hex",
        "right_boundary_hex", "right_child_path", "right_rows", "right_target_leaves",
        "rows", "target_leaves",
    }
    for k in partition.K_VALUES:
        details = payload["partitions"][str(k)]
        _require(details.get("requested_k") == k, f"requested K{k} differs")
        if k not in valid_k:
            _require(set(details) == {"reason", "requested_k", "status"}
                     and details["status"] == "partition_invalid",
                     f"invalid partition closure K{k} differs")
            continue
        _require(set(details) == {
            "leaf_paths", "leaf_sizes", "membership_digest", "requested_k",
            "retained_k", "splits", "status",
        }, f"valid partition closure K{k} differs")
        labels = [int(row["labels"][str(k)]) for row in assignments]
        counts = Counter(labels)
        _require(details["retained_k"] == k and sorted(counts) == list(range(k)),
                 f"retained labels K{k} differ")
        _require(details["leaf_sizes"] == [counts[index] for index in range(k)]
                 and len(details["leaf_paths"]) == len(set(details["leaf_paths"])) == k,
                 f"leaf accounting K{k} differs")
        _require(set(details["leaf_sizes"]) <= {3270 // k, (3270 + k - 1) // k},
                 f"leaf balance K{k} differs")
        _require(details["membership_digest"] == stable_hash([
            [query_ids[index], label] for index, label in enumerate(labels)
        ]), f"membership digest K{k} differs")
        _require(len(details["splits"]) == k - 1
                 and all(set(row) == split_keys for row in details["splits"]),
                 f"split closure K{k} differs")
    normal_transform_keys = {
        "constant_columns", "integer_midrank_digest", "query_audits",
        "query_ids_digest", "raw_vector_digest", "schema_version", "shape",
        "transformed_digest",
    }
    allowed_transform_keys = normal_transform_keys | ({"reproducibility_comparison"}
        if result["status"] == "reproducibility_failed" else set())
    _require(set(transform) == allowed_transform_keys
             and transform["schema_version"] == "m04r14-r1b-balanced-transform-v1"
             and transform["shape"] == [3270, 141]
             and transform["query_ids_digest"] == partition.id_digest(query_ids)
             and len(transform["query_audits"]) == 3270
             and [row["query_episode_id"] for row in transform["query_audits"]] == list(query_ids)
             and all(set(row) == {
                 "benchmark_prefix_digest", "query_episode_id",
                 "query_representation_digest", "stock_prefix_digest",
             } for row in transform["query_audits"])
             and len(transform["constant_columns"]) == len(set(transform["constant_columns"]))
             and all(0 <= int(value) < 141 for value in transform["constant_columns"]),
             "transform artifact closure differs")
    primary_valid = payload["partitions"]["8"].get("status") == "partition_valid"
    expected_status = "partition_valid" if (
        result["serial_parallel_reconstruction_identical"] and primary_valid
    ) else ("reproducibility_failed" if not result["serial_parallel_reconstruction_identical"]
            else "partition_invalid")
    _require(result["status"] == expected_status, "partition status congruence differs")


def _html(result: Mapping[str, Any]) -> str:
    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}–{}</td><td>{:.2%}</td><td>{:.2%}</td><td>{}</td></tr>".format(
            escape(str(row["design"])), escape(str(row.get("role", ""))),
            row["retained_structure_cells"], row["leaf_size_min"], row["leaf_size_max"],
            row["episode_coverage"], row["link_coverage"],
            "PASS" if row["passes"] else "insufficient",
        ) for row in result.get("designs", [])
    )
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>R1-B v3 balanced support</title><style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.5rem;border-bottom:1px solid #ccc;text-align:left}}.note{{background:#fff5d8;padding:1rem}}</style></head><body><h1>R1-B v3 balanced structure support</h1><p><b>Status:</b> {escape(str(result['status']))}. Primary N1 is K=8; K12/K16 are secondary only.</p><table><thead><tr><th>Design</th><th>Role</th><th>Retained K</th><th>Leaf size range</th><th>Episode coverage</th><th>Link coverage</th><th>Gate</th></tr></thead><tbody>{rows}</tbody></table><p class='note'><b>Boundary:</b> support statistics only. No R1-B cohesion/specificity statistic, outcome, prediction, label, ranking change, production promotion or trading claim was opened or authorized.</p></body></html>"""


def _publish_failure(
    repository: Path, prereg: Mapping[str, Any], h1: str, status: str, reason: str,
    partition_result_digest: str,
) -> dict[str, Any]:
    _require(status in {"partition_invalid", "reproducibility_failed"}, "failure status differs")
    state = {
        "schema_version": SCHEMA, "status": status, "passed": False,
        "reason": reason, "preregistration_commit": h1,
        "preregistration_digest": prereg["preregistration_digest"],
        "partition_result_digest": partition_result_digest,
        "primary_k": 8, "b2_execution_authorized": False,
        "real_forward_outcomes_accessed": False, "r1b_statistics_opened": False,
        "adequacy_labels_authorized": False, "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
    }
    return _publish_result(repository, state, [])


def _publish_result(
    repository: Path, state: dict[str, Any], support_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output = repository / partition.SUPPORT_OUTPUT
    _require(not output.exists() and not output.is_symlink(), "support output exists")
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        temporary.mkdir(parents=True)
        partition._write(temporary / "SUPPORT.json", support_rows)
        state["support_sha256"] = partition._sha(temporary / "SUPPORT.json")
        deterministic = dict(state)
        result = {**state, "result_digest": stable_hash(deterministic),
                  "created_at": datetime.now(timezone.utc).isoformat()}
        partition._write(temporary / "RESULT.json", result)
        with (temporary / "report.html").open("w") as handle:
            handle.write(_html(result)); handle.flush(); os.fsync(handle.fileno())
        result["report_sha256"] = partition._sha(temporary / "report.html")
        # Bind the report hash by rewriting RESULT before publication.
        deterministic = {key: value for key, value in result.items() if key not in {"result_digest", "created_at"}}
        result["result_digest"] = stable_hash(deterministic)
        partition._write(temporary / "RESULT.json", result)
        partition._publish(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return result


def run(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(); prereg = partition._load(repository / partition.PREREGISTRATION)
    h1 = partition._validate_h1(repository, prereg)
    for path, expected in prereg["support_file_sha256"].items():
        _require(partition._sha(Path(path)) == expected, f"support authority differs: {path}")
    partition_root = repository / partition.OUTPUT
    _require(partition_root.is_dir(), "partition output absent")
    _require({path.name for path in partition_root.iterdir()} == {
        "PARTITIONS.json", "RESULT.json", "TRANSFORM.json",
    }, "partition output closure differs")
    partition_result = partition._load(partition_root / "RESULT.json")
    partitions = partition._load(partition_root / "PARTITIONS.json")
    transform = partition._load(partition_root / "TRANSFORM.json")
    _require(partition_result["result_digest"] == stable_hash({
        key: value for key, value in partition_result.items()
        if key not in {"result_digest", "created_at"}
    }), "partition result digest differs")
    _require(partition._sha(partition_root / "PARTITIONS.json") == partition_result["partitions_sha256"],
             "partition sidecar hash differs")
    _require(partition._sha(partition_root / "TRANSFORM.json") == partition_result["transform_sha256"],
             "transform sidecar hash differs")
    _require(partition_result["preregistration_commit"] == h1, "partition H1 differs")
    registry = partition._load(repository / partition.REGISTRY)
    registry_rows = {str(row["episode_id"]): row for row in registry["cases_data"]}
    query_ids = tuple(sorted(registry_rows))
    _validate_partition_artifacts(
        partition_result, transform, partitions, prereg, h1, registry, query_ids,
    )
    if partition_result.get("passed") is not True:
        status = str(partition_result.get("status"))
        reason = (
            json.dumps(transform.get("reproducibility_comparison", {}), sort_keys=True)
            if status == "reproducibility_failed"
            else str(partitions["partitions"]["8"].get("reason", "primary K8 invalid"))
        )
        return _publish_failure(
            repository, prereg, h1, status, reason, partition_result["result_digest"],
        )

    r1a_result = partition._load(repository / partition.R1A_RESULT)
    v2_result = partition._load(repository / partition.V2_RESULT)
    v2_verified = partition._load(repository / partition.V2_VERIFIED)
    query_cells = partition._load(repository / partition.V2_QUERY_CELLS, list)
    v2_cohort = partition._load(repository / partition.V2_COHORT, list)
    _require(v2_result["result_digest"] == partition._digest(v2_result, {"result_digest"}), "v2 result differs")
    _require(v2_verified["verification_digest"] == partition._digest(
        v2_verified, {"verification_digest", "created_at"}), "v2 receipt differs")
    _require(v2_verified.get("passed") is True and v2_verified["verified_result_digest"] == v2_result["result_digest"],
             "v2 integrity binding differs")
    _require(v2_verified.get("real_forward_outcomes_accessed") is False,
             "v2 outcome boundary differs")
    _require(r1a_result["result_digest"] == partition._digest(
        r1a_result, {"result_digest", "elapsed_seconds"}), "R1-A result digest differs")
    _require(v2_result["selected_base_design"] == "session_21", "v2 base design differs")
    _require(partition._sha(repository / partition.V2_QUERY_CELLS) == v2_result["query_cells_sha256"],
             "v2 query cells differ")
    _require(partition._sha(repository / partition.V2_COHORT) == v2_result["cohort_support_sha256"],
             "v2 cohort differs")

    query_index = {value: index for index, value in enumerate(query_ids)}
    _require(len(query_ids) == 3270, "query inventory differs")
    selected: defaultdict[str, list[str]] = defaultdict(list)
    recurrent_meta: dict[str, tuple[str, int]] = {}
    queries: dict[str, tuple[str, int, int]] = {}
    case_manifest = []
    for path in sorted((repository / partition.CASES).glob("*.json")):
        case = partition._load(path); query_id = str(case["query_episode_id"])
        registered = registry_rows.get(query_id)
        _require(registered is not None and all((
            case.get("gate_passed") is True, case.get("real_forward_outcomes_accessed") is False,
            case.get("result_digest") == r1a.shadow._deterministic_case_digest(case),
            case.get("checkpoint_integrity_digest") == r1a.shadow._integrity_digest(case),
            case.get("registry_case_id") == registered["case_id"],
            case.get("query_symbol") == registered["symbol"],
            len(case.get("matches", [])) == 20,
        )), f"case differs: {query_id}")
        queries[query_id] = (
            str(case["query_symbol"]),
            int(np.datetime64(case["query_start"], "ns").view(np.int64)),
            int(np.datetime64(case["latest_eligible_cutoff"], "ns").view(np.int64)),
        )
        for match in case["matches"]:
            episode_id = str(match["episode_id"]); selected[episode_id].append(query_id)
            recurrent_meta[episode_id] = (
                str(match["symbol"]), int(np.datetime64(match["cutoff"], "ns").view(np.int64)),
            )
        case_manifest.append({
            "path": path.relative_to(repository / partition.CASES.parent).as_posix(),
            "bytes": path.stat().st_size, "sha256": partition._sha(path),
        })
    _require(stable_hash(case_manifest) == r1a_result["inputs"]["case_manifest_digest"],
             "case manifest differs")
    cohort = {
        episode_id: tuple(sorted(query_index[value] for value in values))
        for episode_id, values in selected.items() if len(values) >= 5
    }
    _require(len(cohort) == 369 and sum(map(len, cohort.values())) == 2865, "cohort differs")
    eligible = {
        episode_id: tuple(index for index, query_id in enumerate(query_ids) if causally_eligible(
            candidate_symbol=recurrent_meta[episode_id][0], candidate_cutoff_ns=recurrent_meta[episode_id][1],
            query_symbol=queries[query_id][0], query_start_ns=queries[query_id][1],
            latest_eligible_ns=queries[query_id][2],
        )) for episode_id in cohort
    }
    _require(all(set(cohort[key]).issubset(eligible[key]) for key in cohort), "causal subset differs")

    cell_rows = {str(row["query_episode_id"]): row for row in query_cells}
    _require(len(cell_rows) == len(query_cells) == 3270 and set(cell_rows) == set(query_ids),
             "v2 query-cell identity differs")
    base_cells = tuple(tuple(cell_rows[value]["base_cells"]["session_21"]) for value in query_ids)
    assignment_rows = partitions["assignments"]
    _require([row["query_episode_id"] for row in assignment_rows] == list(query_ids),
             "partition query order differs")
    valid_k = tuple(
        k for k in partition.K_VALUES
        if partitions["partitions"][str(k)]["status"] == "partition_valid"
    )
    _require(8 in valid_k, "primary partition is invalid")
    _require(all(set(row["labels"]) == {str(k) for k in valid_k} for row in assignment_rows),
             "assignment label closure differs")
    labels = {k: tuple(int(row["labels"][str(k)]) for row in assignment_rows) for k in valid_k}
    for k in valid_k:
        details = partitions["partitions"][str(k)]
        _require(details["status"] == "partition_valid" and len(set(labels[k])) == k,
                 f"partition K{k} differs")
        _require(set(details["leaf_sizes"]) <= {3270 // k, (3270 + k - 1) // k},
                 f"partition balance K{k} differs")

    base_support = {
        episode_id: matched_support(eligible[episode_id], observed, base_cells, cap=CAP)
        for episode_id, observed in cohort.items()
    }
    v2_by_id = {str(row["episode_id"]): row for row in v2_cohort}
    _require({key: v2_by_id[key]["base_support"]["session_21"] for key in cohort} == base_support,
             "verified N0 support differs")
    support_by_k: dict[int, dict[str, int]] = {}
    designs = []
    for k in valid_k:
        details = partitions["partitions"][str(k)]
        crossed = tuple((*base_cells[index], labels[k][index]) for index in range(len(query_ids)))
        _require(len(set(crossed)) > len(set(base_cells)), f"crossed cells vacuous K{k}")
        _require({value[-1] for value in crossed} == set(range(k)),
                 f"not every structure label contributes crossed cells K{k}")
        support = {
            episode_id: matched_support(eligible[episode_id], observed, crossed, cap=CAP)
            for episode_id, observed in cohort.items()
        }
        support_by_k[k] = support
        designs.append(_design_summary(k, support, cohort, details, len(set(crossed))))
    primary_pass = _primary_support_pass(designs)
    status = "support_pass_pending_independent_verification" if primary_pass else "support_inadequate"
    support_rows = [{
        "episode_id": episode_id, "observed_inbound_queries": len(cohort[episode_id]),
        "eligible_queries": len(eligible[episode_id]), "n0_support": base_support[episode_id],
        "n1_support": {str(k): support_by_k[k][episode_id] for k in valid_k},
    } for episode_id in sorted(cohort)]
    state = {
        "schema_version": SCHEMA, "status": status, "passed": primary_pass,
        "preregistration_commit": h1, "preregistration_digest": prereg["preregistration_digest"],
        "partition_result_digest": partition_result["result_digest"],
        "v2_result_digest": v2_result["result_digest"],
        "v2_verification_digest": v2_verified["verification_digest"],
        "inventory": {"queries": 3270, "cohort_episodes": 369, "cohort_links": 2865},
        "designs": designs, "partition_status": {
            str(k): partitions["partitions"][str(k)]["status"] for k in partition.K_VALUES
        }, "primary_k": 8, "secondary_k": [12, 16],
        "n0_matching_design_support_verified": True,
        "primary_support_gate_passed": primary_pass,
        "passed_meaning": "producer support gate only; independent verification pending",
        "b2_execution_authorized": False, "real_forward_outcomes_accessed": False,
        "r1b_statistics_opened": False, "adequacy_labels_authorized": False,
        "predictive_claim_authorized": False, "production_promotion_authorized": False,
    }
    return _publish_result(repository, state, support_rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv); print(json.dumps(run(args.repository), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

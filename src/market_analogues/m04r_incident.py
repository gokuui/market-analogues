from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .adapters import OHLCVSource
from .authority import load_authority_artifact
from .episodes import build_episode
from .fusion import reciprocal_rank_fusion
from .m04_candidate_recall import (
    M04CandidateRecallSpec,
    load_completed_m04_case,
)
from .search import latest_eligible_cutoff
from .types import Episode, InstrumentKey, SearchQuery, stable_hash
from .view_search import _load_manifest, _load_record
from .view_signatures import episode_view_signature, signature_view_distances


M04R_INCIDENT_SCHEMA = "m04r-incident-attribution-v1"
_EXPECTED_DIAGNOSTIC_COUNTS = {
    "9158e964a522ca798ec7bd5f": {
        "in_candidate_pool": 18,
        "local_cap_loss": 0,
        "fusion_pool_loss": 2,
    },
    "a3b3396c25b9f0c163caf143": {
        "in_candidate_pool": 15,
        "local_cap_loss": 1,
        "fusion_pool_loss": 4,
    },
}


class M04RIncidentError(ValueError):
    pass


@dataclass(frozen=True)
class M04RIncidentResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    cases: tuple[dict[str, Any], ...]
    baseline_digest: str


def _case_result_digest(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("result_digest", None)
    return stable_hash(content)


def _authority_path(artifact_dir: Path, episode_id: str) -> Path:
    matches = sorted((artifact_dir / "gate12" / "authorities").glob(
        f"*/cases/{episode_id}.json"
    ))
    if len(matches) != 1:
        raise M04RIncidentError(
            f"expected one authority for {episode_id}, found {len(matches)}"
        )
    return matches[0]


def _registry_rows(path: Path, expected_digest: str) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text()) or {}
    if payload.get("registry_digest") != expected_digest:
        raise M04RIncidentError(f"query registry digest differs: {path}")
    rows = payload.get("cases_data") or []
    return {str(row["episode_id"]): dict(row) for row in rows}


def _current_input_state(
    source: OHLCVSource,
    row: Mapping[str, Any],
    benchmark_fingerprint: str | None,
) -> dict[str, Any]:
    instrument = InstrumentKey(str(row["dataset_id"]), str(row["symbol"]))
    current_source = source.fingerprint(instrument)
    expected_source = str(row["source_fingerprint"])
    expected_benchmark = row.get("benchmark_fingerprint")
    return {
        "expected_source_fingerprint": expected_source,
        "current_source_fingerprint": current_source,
        "source_fingerprint_matches": current_source == expected_source,
        "expected_benchmark_fingerprint": expected_benchmark,
        "current_benchmark_fingerprint": benchmark_fingerprint,
        "benchmark_fingerprint_matches": benchmark_fingerprint == expected_benchmark,
    }


def trace_view_store_targets(
    query: Episode,
    request: SearchQuery,
    root: Path,
    targets: tuple[Mapping[str, Any], ...],
    *,
    candidate_pool: int,
    per_instrument_view: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Reproduce the production local-cap and RRF route with target-level traces."""
    if candidate_pool < 1 or per_instrument_view < 1:
        raise ValueError("candidate limits must be positive")
    dataset_id = query.key.instrument.dataset_id
    manifest, manifest_digest = _load_manifest(root, dataset_id)
    if str(manifest.get("dataset_id")) != dataset_id:
        raise M04RIncidentError("view-store dataset differs")
    if str(manifest.get("representation_version")) != query.key.representation_version:
        raise M04RIncidentError("view-store representation version differs")
    if manifest.get("failures"):
        raise M04RIncidentError("view-store manifest records an incomplete build")
    records = [
        record for record in manifest["shards"]
        if int(record["lookback"]) == query.key.lookback
        and str(record["quality_tier"]) in request.quality_tiers
    ]
    records.sort(key=lambda value: (str(value["symbol"]), str(value["path"])))
    targets_by_symbol: dict[str, set[str]] = {}
    for target in targets:
        targets_by_symbol.setdefault(str(target["symbol"]), set()).add(
            str(target["episode_id"])
        )
    if sum(len(values) for values in targets_by_symbol.values()) != len(targets):
        raise M04RIncidentError("authority target episode IDs must be unique")
    traces = {
        str(target["episode_id"]): {
            "present_in_view_store": False,
            "eligible_in_view_store": False,
            "per_symbol_view_ranks": {},
            "view_distances": {},
            "local_admitted": False,
            "global_view_ranks_after_local_cap": {},
            "fusion_score": None,
            "global_fusion_rank": None,
            "candidate_pool_rank": None,
        }
        for target in targets
    }
    query_signature = episode_view_signature(query)
    latest_ns = int(latest_eligible_cutoff(
        query, request.minimum_history_gap_bars,
    ).value)
    query_start_ns = int(pd.Timestamp(query.bars.timestamp.iloc[0]).value)
    local_rows: list[dict[str, object]] = []
    rows_considered = 0
    view_names: tuple[str, ...] | None = None

    for record in records:
        shard = _load_record(root, record)
        all_ids = shard.episode_ids.astype(str)
        wanted = targets_by_symbol.get(str(record["symbol"]), set())
        if wanted:
            for episode_id in wanted.intersection(set(all_ids)):
                traces[episode_id]["present_in_view_store"] = True
        eligible = shard.cutoffs_ns <= latest_ns
        if str(record["symbol"]) == query.key.instrument.source_symbol:
            eligible &= shard.cutoffs_ns < query_start_ns
        positions = np.flatnonzero(eligible)
        rows_considered += len(positions)
        if not len(positions):
            continue
        distances = signature_view_distances(
            query_signature, shard.signatures[positions],
        )
        current_view_names = tuple(distances)
        if view_names is None:
            view_names = current_view_names
        elif current_view_names != view_names:
            raise M04RIncidentError("view names differ across shards")
        ids = shard.episode_ids[positions].astype(str)
        selected: set[int] = set()
        ranks_by_view: dict[str, np.ndarray] = {}
        for name, values in distances.items():
            order = np.lexsort((ids, values))
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(1, len(order) + 1)
            ranks_by_view[name] = ranks
            selected.update(int(index) for index in order[:per_instrument_view])
        if wanted:
            for episode_id in wanted:
                found = np.flatnonzero(ids == episode_id)
                if len(found) > 1:
                    raise M04RIncidentError(
                        f"duplicate eligible target {episode_id} in view store"
                    )
                if len(found) == 1:
                    local_index = int(found[0])
                    trace = traces[episode_id]
                    trace["eligible_in_view_store"] = True
                    trace["per_symbol_view_ranks"] = {
                        name: int(ranks_by_view[name][local_index])
                        for name in current_view_names
                    }
                    trace["view_distances"] = {
                        name: float(distances[name][local_index])
                        for name in current_view_names
                    }
                    trace["local_admitted"] = local_index in selected
        for local_index in sorted(selected):
            position = int(positions[local_index])
            local_rows.append({
                "episode_id": str(shard.episode_ids[position]),
                "dataset_id": str(record["dataset_id"]),
                "symbol": str(record["symbol"]),
                "cutoff_ns": int(shard.cutoffs_ns[position]),
                **{
                    name: float(values[local_index])
                    for name, values in distances.items()
                },
            })

    if view_names is None or not local_rows:
        raise M04RIncidentError("view store produced no locally admitted candidates")
    frame = pd.DataFrame(local_rows)
    if frame.episode_id.astype(str).duplicated().any():
        raise M04RIncidentError("local shard union contains duplicate episode IDs")
    fusion = reciprocal_rank_fusion(frame, view_names, pool_size=candidate_pool)
    ranking_by_id = {
        str(row.episode_id): row for row in fusion.rankings.itertuples(index=False)
    }
    selected_by_id = {
        str(row.episode_id): row for row in fusion.selected.itertuples(index=False)
    }
    for episode_id, trace in traces.items():
        ranked = ranking_by_id.get(episode_id)
        if ranked is not None:
            trace["global_view_ranks_after_local_cap"] = {
                name: int(getattr(ranked, f"rank_{name}")) for name in view_names
            }
            trace["fusion_score"] = float(ranked.fusion_score)
            trace["global_fusion_rank"] = int(ranked.fusion_rank)
        selected = selected_by_id.get(episode_id)
        if selected is not None:
            trace["candidate_pool_rank"] = int(selected.pool_rank)
    return traces, {
        "view_manifest_digest": manifest_digest,
        "view_names": list(view_names),
        "shards_loaded": len(records),
        "rows_considered": rows_considered,
        "local_candidates": len(frame),
        "candidate_pool_returned": len(fusion.selected),
    }


def _loss_reason(trace: Mapping[str, Any]) -> str:
    if not trace["present_in_view_store"]:
        return "missing_from_view_store"
    if not trace["eligible_in_view_store"]:
        return "ineligible_in_view_store"
    if not trace["local_admitted"]:
        return "local_cap_loss"
    if trace["candidate_pool_rank"] is None:
        return "fusion_pool_loss"
    return "in_candidate_pool"


def _reason_counts(neighbors: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in neighbors:
        reason = str(row["final_loss_reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def diagnose_m04r_incident(
    spec: M04CandidateRecallSpec,
    sources: Mapping[str, OHLCVSource],
    artifact_dir: Path,
) -> M04RIncidentResult:
    failures: list[str] = []
    cases: list[dict[str, Any]] = []
    retrieval = spec.payload["retrieval"]
    benchmark_fingerprints = {
        dataset_id: source.benchmark_fingerprint()
        for dataset_id, source in sources.items()
    }
    registry_cache: dict[str, dict[str, dict[str, Any]]] = {}

    for episode_id in spec.payload["holdout_episode_ids"]:
        authority_path = _authority_path(artifact_dir, episode_id)
        authority = load_authority_artifact(authority_path)
        authority_payload = json.loads(authority_path.read_text())
        if authority.authority_digest != spec.payload["authority_digests"][episode_id]:
            raise M04RIncidentError(f"authority digest differs for {episode_id}")
        query_meta = dict(authority_payload["query"])
        dataset_id = str(query_meta["dataset_id"])
        source = sources.get(dataset_id)
        if source is None:
            raise M04RIncidentError(f"no configured source for {dataset_id}")
        if dataset_id not in registry_cache:
            registry_cache[dataset_id] = _registry_rows(
                artifact_dir / "gate12" / dataset_id / "query-registry.yaml",
                str(spec.payload["registry_digests"][dataset_id]),
            )
        registry_row = registry_cache[dataset_id].get(episode_id)
        if registry_row is None:
            raise M04RIncidentError(f"registry omits {episode_id}")
        for name in (
            "dataset_id", "symbol", "cutoff", "lookback", "representation_version",
        ):
            if str(registry_row[name]) != str(query_meta[name]):
                raise M04RIncidentError(
                    f"registry and authority differ for {episode_id}/{name}"
                )
        input_state = _current_input_state(
            source, registry_row, benchmark_fingerprints[dataset_id],
        )
        exact_neighbors = [{
            "authority_rank": rank,
            "episode_id": str(match["episode_id"]),
            "symbol": str(match["symbol"]),
            "cutoff": str(match["cutoff"]),
            "authority_certified_eligible": True,
            "exact_total_distance": float(match["total_distance"]),
            "exact_component_distances": {
                str(name): float(value)
                for name, value in sorted(match["component_distances"].items())
            },
        } for rank, match in enumerate(authority_payload["matches"], 1)]
        if len(exact_neighbors) != int(retrieval["top_k"]):
            raise M04RIncidentError(
                f"authority {episode_id} contains {len(exact_neighbors)} matches"
            )
        case_path = artifact_dir / "m04-candidate-recall" / "cases" / f"{episode_id}.json"
        unchanged = (
            input_state["source_fingerprint_matches"]
            and input_state["benchmark_fingerprint_matches"]
        )
        base_case = {
            "query_episode_id": episode_id,
            "dataset_id": dataset_id,
            "symbol": str(query_meta["symbol"]),
            "cutoff": str(query_meta["cutoff"]),
            "authority_digest": authority.authority_digest,
            "authority_result_digest": authority.result_digest,
            "input_state": input_state,
        }
        if not unchanged:
            neighbors = [{
                **neighbor,
                "present_in_view_store": None,
                "eligible_in_view_store": None,
                "per_symbol_view_ranks": {},
                "view_distances": {},
                "local_admitted": None,
                "global_view_ranks_after_local_cap": {},
                "fusion_score": None,
                "global_fusion_rank": None,
                "candidate_pool_rank": None,
                "final_loss_reason": "blocked_input_version",
            } for neighbor in exact_neighbors]
            cases.append({
                **base_case,
                "status": "blocked_input_version",
                "prior_case_result_digest": None,
                "trace_metrics": None,
                "reason_counts": _reason_counts(neighbors),
                "authority_neighbors": neighbors,
            })
            continue
        completed = load_completed_m04_case(case_path, spec, episode_id)
        if completed is None:
            failures.append(f"unchanged case {episode_id} has no preserved M04 result")
            cases.append({
                **base_case,
                "status": "missing_preserved_case",
                "prior_case_result_digest": None,
                "trace_metrics": None,
                "reason_counts": {"missing_preserved_case": 20},
                "authority_neighbors": [{
                    **neighbor, "final_loss_reason": "missing_preserved_case",
                } for neighbor in exact_neighbors],
            })
            continue
        case_payload = json.loads(case_path.read_text())
        if case_payload.get("result_digest") != _case_result_digest(case_payload):
            raise M04RIncidentError(f"M04 case digest differs for {episode_id}")
        instrument = InstrumentKey(dataset_id, str(query_meta["symbol"]))
        query = build_episode(
            source, instrument, str(query_meta["cutoff"]),
            int(query_meta["lookback"]), str(query_meta["representation_version"]),
        )
        if query.key.id != episode_id:
            raise M04RIncidentError(f"reconstructed query differs for {episode_id}")
        request = SearchQuery(
            query.key, (dataset_id,), tuple(retrieval["quality_tiers"]),
            int(retrieval["top_k"]),
            minimum_history_gap_bars=int(retrieval["minimum_history_gap_sessions"]),
        )
        traces, trace_metrics = trace_view_store_targets(
            query, request, artifact_dir / "view-store",
            tuple(authority_payload["matches"]),
            candidate_pool=int(retrieval["candidate_pool"]),
            per_instrument_view=int(retrieval["per_instrument_view"]),
        )
        if trace_metrics["view_manifest_digest"] != spec.payload["view_manifest_digests"][dataset_id]:
            failures.append(f"view manifest differs for {episode_id}")
        prior_ranks = {
            str(row["episode_id"]): row.get("candidate_rank")
            for row in completed.authority_neighbors
        }
        exact_ids = {row["episode_id"] for row in exact_neighbors}
        if set(prior_ranks) != exact_ids:
            failures.append(f"preserved authority-neighbour IDs differ for {episode_id}")
        neighbors = []
        for neighbor in exact_neighbors:
            trace = traces[neighbor["episode_id"]]
            reason = _loss_reason(trace)
            if reason in {"missing_from_view_store", "ineligible_in_view_store"}:
                failures.append(
                    f"authority-certified eligible target has {reason}: "
                    f"{episode_id}/{neighbor['episode_id']}"
                )
            prior_rank = prior_ranks.get(neighbor["episode_id"])
            if trace["candidate_pool_rank"] != prior_rank:
                failures.append(
                    f"candidate-rank parity differs for {episode_id}/"
                    f"{neighbor['episode_id']}: {trace['candidate_pool_rank']} != {prior_rank}"
                )
            neighbors.append({
                **neighbor,
                **trace,
                "prior_candidate_pool_rank": prior_rank,
                "final_loss_reason": reason,
            })
        cases.append({
            **base_case,
            "status": "analyzed",
            "prior_case_result_digest": case_payload["result_digest"],
            "trace_metrics": trace_metrics,
            "reason_counts": _reason_counts(neighbors),
            "authority_neighbors": neighbors,
        })

    by_id = {str(case["query_episode_id"]): case for case in cases}
    for episode_id, expected in _EXPECTED_DIAGNOSTIC_COUNTS.items():
        observed = by_id.get(episode_id, {}).get("reason_counts", {})
        for reason, count in expected.items():
            if int(observed.get(reason, 0)) != count:
                failures.append(
                    f"{episode_id} {reason} count {observed.get(reason, 0)} != {count}"
                )
    analyzed = sum(case["status"] == "analyzed" for case in cases)
    blocked = sum(case["status"] == "blocked_input_version" for case in cases)
    neighbor_count = sum(len(case["authority_neighbors"]) for case in cases)
    if len(cases) != 23 or neighbor_count != 460:
        failures.append(
            f"incomplete accounting: cases={len(cases)}, neighbors={neighbor_count}"
        )
    if analyzed != 12 or blocked != 11:
        failures.append(f"unexpected case states: analyzed={analyzed}, blocked={blocked}")
    metrics = {
        "schema_version": M04R_INCIDENT_SCHEMA,
        "m04_contract_digest": spec.digest,
        "holdout_cases_accounted": len(cases),
        "authority_neighbors_accounted": neighbor_count,
        "analyzed_unchanged_cases": analyzed,
        "blocked_input_version_cases": blocked,
        "candidate_rank_parity_failures": sum(
            failure.startswith("candidate-rank parity differs") for failure in failures
        ),
        "outcome_inputs_opened": 0,
        "real_forward_outcomes_accessed": False,
        "input_roles": [
            "frozen_authority", "frozen_query_registry", "preserved_m04_case",
            "raw_query_ohlcv", "benchmark_ohlcv", "immutable_view_store",
        ],
    }
    deterministic = {
        "schema_version": M04R_INCIDENT_SCHEMA,
        "metrics": metrics,
        "failures": sorted(failures),
        "cases": cases,
    }
    baseline_digest = stable_hash(deterministic)
    return M04RIncidentResult(
        not failures, metrics, tuple(sorted(failures)), tuple(cases), baseline_digest,
    )


def write_m04r_incident(
    result: M04RIncidentResult,
    machine_path: Path,
    html_path: Path,
) -> tuple[Path, Path]:
    machine_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": M04R_INCIDENT_SCHEMA,
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "cases": list(result.cases),
        "baseline_digest": result.baseline_digest,
    }
    temporary = machine_path.with_suffix(machine_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(machine_path)
    summary_rows = "".join(
        f"<tr><td>{escape(str(case['dataset_id']))}</td>"
        f"<td>{escape(str(case['symbol']))}</td>"
        f"<td>{escape(str(case['cutoff']))}</td>"
        f"<td>{escape(str(case['status']))}</td>"
        f"<td><code>{escape(json.dumps(case['reason_counts'], sort_keys=True))}</code></td></tr>"
        for case in result.cases
    )
    detail_rows = "".join(
        f"<tr><td>{escape(str(case['symbol']))}</td>"
        f"<td>{row['authority_rank']}</td><td>{escape(str(row['symbol']))}</td>"
        f"<td>{escape(str(row['cutoff']))}</td>"
        f"<td>{row.get('global_fusion_rank') or '—'}</td>"
        f"<td>{row.get('candidate_pool_rank') or '—'}</td>"
        f"<td>{escape(str(row['final_loss_reason']))}</td></tr>"
        for case in result.cases for row in case["authority_neighbors"]
    )
    failure_items = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_temporary = html_path.with_suffix(html_path.suffix + ".tmp")
    html_temporary.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>M04R incident attribution</title><style>
body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;background:#f5f7f8;color:#17202a}}
header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.45rem;border-bottom:1px solid #ddd;text-align:left}}
.pass{{color:#117864}}.fail{{color:#b03a2e}}code{{font-size:.85em}}pre{{white-space:pre-wrap}}
</style></head><body><header><h1>M04R incident attribution: <span class="{status.lower()}">{status}</span></h1>
<p>This report explains the frozen M04 candidate losses. It does not repair or rerun the authority, and it opens no forward-outcome input.</p>
<p>Immutable baseline digest: <code>{result.baseline_digest}</code></p></header>
<section><h2>Verification metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section>
<section><h2>Case accounting</h2><table><thead><tr><th>Market</th><th>Query</th><th>Cutoff</th><th>Status</th><th>Final reasons</th></tr></thead><tbody>{summary_rows}</tbody></table></section>
<section><h2>All authority neighbours</h2><p>Exact component distances, per-symbol view ranks and global view ranks are retained in the machine JSON.</p>
<table><thead><tr><th>Query</th><th>Exact rank</th><th>Neighbour</th><th>Cutoff</th><th>Fusion rank</th><th>Pool rank</th><th>Reason</th></tr></thead><tbody>{detail_rows}</tbody></table></section>
<section><h2>Diagnostic failures</h2><ul>{failure_items}</ul></section></body></html>""")
    html_temporary.replace(html_path)
    return machine_path, html_path

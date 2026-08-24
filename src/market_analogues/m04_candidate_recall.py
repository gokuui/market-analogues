from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import resource
from time import perf_counter
from typing import Any, Mapping

import pandas as pd
import yaml

from .adapters import OHLCVSource
from .authority import load_authority_artifact
from .episodes import build_episode
from .search import latest_eligible_cutoff
from .types import InstrumentKey, SearchQuery, stable_hash
from .view_search import search_view_store


M04_CONTRACT_SCHEMA = "m04-candidate-recall-contract-v1"
M04_RESULT_SCHEMA = "m04-candidate-recall-result-v1"


class M04CandidateRecallError(ValueError):
    pass


@dataclass(frozen=True)
class M04CandidateRecallSpec:
    source: Path
    payload: dict[str, Any]
    digest: str


@dataclass(frozen=True)
class M04CaseResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    authority_neighbors: tuple[dict[str, Any], ...]


def load_m04_candidate_recall_spec(path: str | Path) -> M04CandidateRecallSpec:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text()) or {}
    if not isinstance(payload, Mapping):
        raise M04CandidateRecallError("M04 contract must be a mapping")
    required = {
        "schema_version", "contract_id", "development_case", "holdout_episode_ids",
        "authority_digests", "registry_digests", "view_manifest_digests",
        "retrieval", "acceptance", "scope",
    }
    if set(payload) != required:
        raise M04CandidateRecallError(
            f"M04 contract keys differ; missing={sorted(required - set(payload))}, "
            f"unknown={sorted(set(payload) - required)}"
        )
    if payload["schema_version"] != M04_CONTRACT_SCHEMA:
        raise M04CandidateRecallError("unsupported M04 contract schema")
    development = payload["development_case"]
    if not isinstance(development, Mapping) or set(development) != {
        "dataset_id", "episode_id", "observed_pool_recalls",
    }:
        raise M04CandidateRecallError("development case contract differs")
    holdout = payload["holdout_episode_ids"]
    authorities = payload["authority_digests"]
    if not isinstance(holdout, list) or len(holdout) != 23 or len(set(holdout)) != 23:
        raise M04CandidateRecallError("M04 requires exactly 23 unique holdout episodes")
    expected = set(holdout) | {str(development["episode_id"])}
    if not isinstance(authorities, Mapping) or set(authorities) != expected or len(expected) != 24:
        raise M04CandidateRecallError("authority digests must bind all 24 unique cases")
    if any(len(str(value)) != 64 for value in authorities.values()):
        raise M04CandidateRecallError("authority digests must be SHA-256 values")
    if set(payload["registry_digests"]) != {"nse", "nasdaq"}:
        raise M04CandidateRecallError("registry digests must bind NSE and NASDAQ")
    if set(payload["view_manifest_digests"]) != {"nse", "nasdaq"}:
        raise M04CandidateRecallError("view manifests must bind NSE and NASDAQ")
    retrieval = payload["retrieval"]
    if set(retrieval) != {
        "top_k", "candidate_pool", "per_instrument_view",
        "minimum_history_gap_sessions", "quality_tiers",
    }:
        raise M04CandidateRecallError("M04 retrieval contract differs")
    if int(retrieval["top_k"]) != 20 or int(retrieval["candidate_pool"]) < 20:
        raise M04CandidateRecallError("M04 requires top-20 and a sufficient candidate pool")
    if retrieval["quality_tiers"] != ["A", "B"]:
        raise M04CandidateRecallError("M04 quality tiers must be A and B")
    acceptance = payload["acceptance"]
    required_acceptance = {
        "minimum_authority_top20_recall_per_case", "maximum_case_seconds",
        "maximum_peak_rss_mb", "require_zero_temporal_violations",
        "require_zero_duplicate_candidate_ids",
    }
    if set(acceptance) != required_acceptance:
        raise M04CandidateRecallError("M04 acceptance contract differs")
    if float(acceptance["minimum_authority_top20_recall_per_case"]) < .95:
        raise M04CandidateRecallError("M04 per-case recall floor cannot be below 0.95")
    if acceptance["require_zero_temporal_violations"] is not True:
        raise M04CandidateRecallError("temporal violations must be forbidden")
    if acceptance["require_zero_duplicate_candidate_ids"] is not True:
        raise M04CandidateRecallError("candidate ID duplicates must be forbidden")
    scope = payload["scope"]
    if scope.get("candidate_inclusion_only") is not True:
        raise M04CandidateRecallError("M04 subgate must remain candidate-inclusion only")
    if scope.get("real_forward_outcomes_accessed") is not False:
        raise M04CandidateRecallError("M04 candidate recall cannot access outcomes")
    canonical = json.loads(json.dumps(payload, sort_keys=True))
    return M04CandidateRecallSpec(source, canonical, stable_hash(canonical))


def _authority_path(artifact_dir: Path, episode_id: str) -> Path:
    matches = list((artifact_dir / "gate12" / "authorities").glob(
        f"*/cases/{episode_id}.json",
    ))
    if len(matches) != 1:
        raise M04CandidateRecallError(
            f"expected one authority for {episode_id}, found {len(matches)}"
        )
    return matches[0]


def _result_digest(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("result_digest", None)
    return stable_hash(content)


def load_completed_m04_case(
    path: Path, spec: M04CandidateRecallSpec, episode_id: str,
) -> M04CaseResult | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise M04CandidateRecallError(f"invalid completed M04 case: {exc}") from exc
    if payload.get("result_digest") != _result_digest(payload):
        raise M04CandidateRecallError("completed M04 case digest mismatch")
    metrics = payload.get("metrics") or {}
    if metrics.get("contract_digest") != spec.digest:
        raise M04CandidateRecallError("completed M04 case uses a different contract")
    if metrics.get("query_episode_id") != episode_id:
        raise M04CandidateRecallError("completed M04 case episode differs")
    return M04CaseResult(
        bool(payload["passed"]), metrics, tuple(payload.get("failures") or ()),
        tuple(payload.get("authority_neighbors") or ()),
    )


def verify_m04_candidate_case(
    spec: M04CandidateRecallSpec,
    episode_id: str,
    source: OHLCVSource,
    artifact_dir: Path,
    *,
    output_path: Path | None = None,
) -> M04CaseResult:
    if episode_id not in spec.payload["holdout_episode_ids"]:
        raise M04CandidateRecallError("episode is not in the locked holdout")
    if output_path is not None:
        completed = load_completed_m04_case(output_path, spec, episode_id)
        if completed is not None:
            return completed
    started = perf_counter()
    failures: list[str] = []
    authority_path = _authority_path(artifact_dir, episode_id)
    authority = load_authority_artifact(authority_path)
    authority_payload = json.loads(authority_path.read_text())
    if authority.authority_digest != spec.payload["authority_digests"][episode_id]:
        raise M04CandidateRecallError("authority digest differs from M04 contract")
    query_meta = authority_payload["query"]
    dataset_id = str(query_meta["dataset_id"])
    if str(source.spec.dataset_id) != dataset_id:
        raise M04CandidateRecallError("source dataset differs from authority")
    registry_path = artifact_dir / "gate12" / dataset_id / "query-registry.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    if registry.get("registry_digest") != spec.payload["registry_digests"][dataset_id]:
        raise M04CandidateRecallError("query registry digest differs from M04 contract")
    registry_rows = {
        str(row["episode_id"]): row for row in registry.get("cases_data", [])
    }
    row = registry_rows.get(episode_id)
    if row is None:
        raise M04CandidateRecallError("authority episode is absent from registry")
    for key in ("dataset_id", "symbol", "cutoff", "lookback", "representation_version"):
        if str(row[key]) != str(query_meta[key]):
            raise M04CandidateRecallError(f"registry and authority differ for {key}")
    instrument = InstrumentKey(dataset_id, str(query_meta["symbol"]))
    if source.fingerprint(instrument) != row["source_fingerprint"]:
        raise M04CandidateRecallError("query source fingerprint changed")
    if source.benchmark_fingerprint() != row["benchmark_fingerprint"]:
        raise M04CandidateRecallError("benchmark fingerprint changed")
    view_manifest_path = artifact_dir / "view-store" / dataset_id / "manifest.json"
    view_manifest = json.loads(view_manifest_path.read_text())
    if view_manifest.get("manifest_digest") != spec.payload["view_manifest_digests"][dataset_id]:
        raise M04CandidateRecallError("view manifest digest differs from M04 contract")
    query = build_episode(
        source, instrument, str(query_meta["cutoff"]), int(query_meta["lookback"]),
        str(query_meta["representation_version"]),
    )
    if query.key.id != episode_id:
        raise M04CandidateRecallError("reconstructed query episode ID differs")
    retrieval = spec.payload["retrieval"]
    request = SearchQuery(
        query.key, (dataset_id,), tuple(retrieval["quality_tiers"]),
        int(retrieval["top_k"]),
        minimum_history_gap_bars=int(retrieval["minimum_history_gap_sessions"]),
    )
    search = search_view_store(
        query, request, artifact_dir / "view-store",
        candidate_pool=int(retrieval["candidate_pool"]),
        per_instrument_view=int(retrieval["per_instrument_view"]),
    )
    if search.manifest_digest != view_manifest["manifest_digest"]:
        failures.append("search used a different view manifest")
    ids = [hit.episode_id for hit in search.hits]
    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        failures.append(f"candidate pool contains {duplicate_ids} duplicate IDs")
    latest = latest_eligible_cutoff(query, request.minimum_history_gap_bars)
    query_start = pd.Timestamp(query.bars.timestamp.iloc[0])
    temporal_violations = 0
    for hit in search.hits:
        if hit.cutoff > latest:
            temporal_violations += 1
        if hit.instrument == query.key.instrument and hit.cutoff >= query_start:
            temporal_violations += 1
    if temporal_violations:
        failures.append(f"candidate pool contains {temporal_violations} temporal violations")
    ranks = {episode: rank for rank, episode in enumerate(ids, 1)}
    neighbors = tuple({
        "authority_rank": rank,
        "episode_id": match["episode_id"],
        "symbol": match["symbol"],
        "cutoff": match["cutoff"],
        "candidate_rank": ranks.get(match["episode_id"]),
        "retrieved": match["episode_id"] in ranks,
    } for rank, match in enumerate(authority_payload["matches"], 1))
    retrieved = sum(bool(value["retrieved"]) for value in neighbors)
    recall = retrieved / int(retrieval["top_k"])
    elapsed = perf_counter() - started
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    acceptance = spec.payload["acceptance"]
    if recall < float(acceptance["minimum_authority_top20_recall_per_case"]):
        failures.append(
            f"authority recall {recall:.3f} below "
            f"{acceptance['minimum_authority_top20_recall_per_case']:.3f}"
        )
    if elapsed > float(acceptance["maximum_case_seconds"]):
        failures.append(f"elapsed {elapsed:.2f}s exceeds {acceptance['maximum_case_seconds']:.2f}s")
    if rss_mb > float(acceptance["maximum_peak_rss_mb"]):
        failures.append(f"RSS {rss_mb:.1f}MB exceeds {acceptance['maximum_peak_rss_mb']:.1f}MB")
    metrics = {
        "schema_version": M04_RESULT_SCHEMA,
        "contract_id": spec.payload["contract_id"],
        "contract_digest": spec.digest,
        "query_episode_id": episode_id,
        "dataset_id": dataset_id,
        "symbol": query_meta["symbol"],
        "cutoff": query_meta["cutoff"],
        "authority_digest": authority.authority_digest,
        "authority_result_digest": authority.result_digest,
        "registry_digest": registry["registry_digest"],
        "view_manifest_digest": search.manifest_digest,
        "candidate_pool_requested": int(retrieval["candidate_pool"]),
        "candidate_pool_returned": len(ids),
        "authority_neighbors_retrieved": retrieved,
        "authority_top20_recall": recall,
        "duplicate_candidate_ids": duplicate_ids,
        "temporal_violations": temporal_violations,
        "rows_considered": search.rows_considered,
        "local_candidates": search.local_candidates,
        "shards_loaded": search.shards_loaded,
        "elapsed_seconds": elapsed,
        "peak_rss_mb": rss_mb,
        "real_forward_outcomes_accessed": False,
        "candidate_inclusion_only": True,
    }
    return M04CaseResult(not failures, metrics, tuple(failures), neighbors)


def write_m04_case(result: M04CaseResult, machine_path: Path, html_path: Path) -> tuple[Path, Path]:
    machine_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "authority_neighbors": list(result.authority_neighbors),
    }
    payload["result_digest"] = _result_digest(payload)
    temporary = machine_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(machine_path)
    rows = "".join(
        f"<tr><td>{row['authority_rank']}</td><td>{escape(str(row['symbol']))}</td>"
        f"<td>{escape(str(row['cutoff']))}</td><td>{row['candidate_rank'] or 'missing'}</td></tr>"
        for row in result.authority_neighbors
    )
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M04 candidate recall</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;background:#f5f7f8;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{white-space:pre-wrap}}</style></head><body><header><h1>M04 candidate inclusion: <span class={status.lower()}>{status}</span></h1><p>This subgate asks whether the frozen persisted multiview search includes at least 19 of each sealed exact authority's 20 neighbors. It does not claim exact reranking, topology ranking, diversity, novelty or outcome value.</p></header><section><h2>Metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section><section><h2>Authority neighbors</h2><table><thead><tr><th>Authority rank</th><th>Symbol</th><th>Cutoff</th><th>Candidate rank</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failures}</ul></section></body></html>""")
    return machine_path, html_path


def aggregate_m04_cases(
    spec: M04CandidateRecallSpec, case_dir: Path,
) -> tuple[bool, dict[str, Any], tuple[str, ...]]:
    rows = []
    failures = []
    for episode_id in spec.payload["holdout_episode_ids"]:
        path = case_dir / f"{episode_id}.json"
        result = load_completed_m04_case(path, spec, episode_id)
        if result is None:
            failures.append(f"missing case {episode_id}")
            continue
        rows.append(result.metrics)
        if not result.passed:
            failures.append(f"failed case {episode_id}")
    metrics = {
        "schema_version": "m04-candidate-recall-matrix-v1",
        "contract_id": spec.payload["contract_id"],
        "contract_digest": spec.digest,
        "required_cases": len(spec.payload["holdout_episode_ids"]),
        "completed_cases": len(rows),
        "passed_cases": sum(not any(f"failed case {row['query_episode_id']}" == f for f in failures) for row in rows),
        "minimum_recall": min((row["authority_top20_recall"] for row in rows), default=None),
        "mean_recall": sum(row["authority_top20_recall"] for row in rows) / len(rows) if rows else None,
        "maximum_elapsed_seconds": max((row["elapsed_seconds"] for row in rows), default=None),
        "maximum_peak_rss_mb": max((row["peak_rss_mb"] for row in rows), default=None),
        "total_rows_considered": sum(row["rows_considered"] for row in rows),
        "real_forward_outcomes_accessed": False,
        "candidate_inclusion_only": True,
        "cases": rows,
    }
    return len(rows) == len(spec.payload["holdout_episode_ids"]) and not failures, metrics, tuple(failures)


def write_m04_matrix(
    passed: bool,
    metrics: Mapping[str, Any],
    failures: tuple[str, ...],
    machine_path: Path,
    html_path: Path,
) -> tuple[Path, Path]:
    machine_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "passed": passed, "metrics": dict(metrics), "failures": list(failures),
    }
    payload["result_digest"] = _result_digest(payload)
    temporary = machine_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(machine_path)
    rows = "".join(
        f"<tr><td>{escape(str(row['dataset_id']))}</td><td>{escape(str(row['symbol']))}</td>"
        f"<td>{escape(str(row['cutoff']))}</td><td>{row['authority_top20_recall']:.1%}</td>"
        f"<td>{row['candidate_pool_returned']:,}</td><td>{row['elapsed_seconds']:.2f}</td>"
        f"<td>{row['peak_rss_mb']:.1f}</td></tr>"
        for row in metrics["cases"]
    )
    failure_items = "".join(f"<li>{escape(value)}</li>" for value in failures) or "<li>None</li>"
    status = "PASS" if passed else "FAIL"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M04 candidate-recall matrix</title><style>body{{font-family:system-ui,sans-serif;max-width:1300px;margin:2rem auto;background:#f5f7f8;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{white-space:pre-wrap}}</style></head><body><header><h1>M04 23-case candidate-recall holdout: <span class={status.lower()}>{status}</span></h1><p>All cases use sealed Gate 12 exact authorities and a pool size selected on one disclosed development query. This is candidate inclusion only; exact/topology reranking, diversity, novelty and outcomes remain unverified.</p></header><section><h2>Aggregate</h2><pre>{escape(json.dumps({key: value for key, value in metrics.items() if key != 'cases'}, indent=2, sort_keys=True))}</pre></section><section><h2>Cases</h2><table><thead><tr><th>Market</th><th>Query</th><th>Cutoff</th><th>Recall</th><th>Pool</th><th>Seconds</th><th>RSS MB</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failure_items}</ul></section></body></html>""")
    return machine_path, html_path

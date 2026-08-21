from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import resource
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .distance import representation_distance
from .latent_structures import LATENT_SPECS, generate_latent_structure
from .multiresolution import build_multiresolution_state
from .representation import represent
from .state_distance import multiresolution_state_distance, state_distance_contract
from .types import Episode, stable_hash


STRUCTURAL_VERIFIER_SCHEMA = "latent-structural-verifier-v1"


class StructuralVerifierError(ValueError):
    pass


@dataclass(frozen=True)
class StructuralVerifierSpec:
    source: Path
    payload: dict[str, Any]
    digest: str


@dataclass(frozen=True)
class StructuralVerification:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    queries: tuple[dict[str, Any], ...]


def load_structural_verifier_spec(path: str | Path) -> StructuralVerifierSpec:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text()) or {}
    if not isinstance(payload, Mapping):
        raise StructuralVerifierError("structural verifier must be a mapping")
    required = {
        "schema_version", "verifier_id", "distance_contract_digest", "latent_ids",
        "candidate_seeds", "validation_query_seeds", "positive_transformations",
        "critical_negative", "acceptance", "legacy_comparison",
    }
    if set(payload) != required:
        raise StructuralVerifierError(
            f"structural verifier keys differ; missing={sorted(required - set(payload))}, "
            f"unknown={sorted(set(payload) - required)}"
        )
    if payload["schema_version"] != STRUCTURAL_VERIFIER_SCHEMA:
        raise StructuralVerifierError(f"unsupported structural schema: {payload['schema_version']!r}")
    expected_ids = [spec.latent_id for spec in LATENT_SPECS]
    if payload["latent_ids"] != expected_ids:
        raise StructuralVerifierError("latent IDs do not match the independent generator contract")
    candidate_seeds = payload["candidate_seeds"]
    validation_seeds = payload["validation_query_seeds"]
    if not all(isinstance(values, list) and len(values) >= 5 for values in (candidate_seeds, validation_seeds)):
        raise StructuralVerifierError("candidate and validation seeds must each contain at least five values")
    if set(candidate_seeds) & set(validation_seeds):
        raise StructuralVerifierError("candidate and validation seeds must be disjoint")
    distance_digest = state_distance_contract()["digest"]
    if payload["distance_contract_digest"] != distance_digest:
        raise StructuralVerifierError("distance contract digest differs from frozen implementation")
    transforms = payload["positive_transformations"]
    if not isinstance(transforms, Mapping) or set(transforms) != {
        "tempo_by_query_index", "noise_scale_by_query_index", "price_scale", "volume_scale",
    }:
        raise StructuralVerifierError("positive transformation contract differs")
    if not transforms["tempo_by_query_index"] or not transforms["noise_scale_by_query_index"]:
        raise StructuralVerifierError("positive transformation schedules cannot be empty")
    negative = payload["critical_negative"]
    if not isinstance(negative, Mapping) or not all(value is True for value in negative.values()):
        raise StructuralVerifierError("all critical-negative transformations must be enabled")
    acceptance = payload["acceptance"]
    required_acceptance = {
        "top_k", "minimum_top1_per_family", "minimum_precision_at_k_per_family",
        "maximum_critical_negative_error_per_family", "maximum_total_seconds",
        "maximum_rss_mb",
    }
    if not isinstance(acceptance, Mapping) or set(acceptance) != required_acceptance:
        raise StructuralVerifierError("acceptance contract differs")
    if int(acceptance["top_k"]) != 5:
        raise StructuralVerifierError("M03 top_k must be 5")
    for key in ("minimum_top1_per_family", "minimum_precision_at_k_per_family"):
        if not .75 <= float(acceptance[key]) <= 1:
            raise StructuralVerifierError(f"{key} cannot be below 0.75")
    if not 0 <= float(acceptance["maximum_critical_negative_error_per_family"]) <= .02:
        raise StructuralVerifierError("critical-negative threshold cannot exceed 0.02")
    canonical = json.loads(json.dumps(payload, sort_keys=True))
    return StructuralVerifierSpec(source, canonical, stable_hash(canonical))


def _trim_episode(episode: Episode) -> Episode:
    cutoff = pd.Timestamp(episode.key.cutoff)
    bars = episode.bars[episode.bars.timestamp <= cutoff].tail(252).reset_index(drop=True)
    benchmark = episode.benchmark
    if benchmark is not None:
        benchmark = benchmark[benchmark.timestamp <= cutoff].copy()
    return Episode(episode.key, bars, benchmark, episode.quality_tier, episode.quality_issues)


def verify_latent_structures(spec: StructuralVerifierSpec) -> StructuralVerification:
    started = perf_counter()
    payload = spec.payload
    top_k = int(payload["acceptance"]["top_k"])
    candidates: list[dict[str, Any]] = []
    for latent_id in payload["latent_ids"]:
        for seed in payload["candidate_seeds"]:
            case = generate_latent_structure(latent_id, int(seed), role="candidate")
            episode = _trim_episode(case.episode)
            candidates.append({
                "latent_id": latent_id,
                "seed": int(seed),
                "state": build_multiresolution_state(episode),
                "legacy": represent(episode),
            })
    query_rows: list[dict[str, Any]] = []
    family_new: dict[str, list[tuple[float, float, float]]] = {
        latent_id: [] for latent_id in payload["latent_ids"]
    }
    family_legacy: dict[str, list[tuple[float, float, float]]] = {
        latent_id: [] for latent_id in payload["latent_ids"]
    }
    transforms = payload["positive_transformations"]
    for latent_id in payload["latent_ids"]:
        for query_index, seed in enumerate(payload["validation_query_seeds"]):
            tempo = float(transforms["tempo_by_query_index"][
                query_index % len(transforms["tempo_by_query_index"])
            ])
            noise_scale = float(transforms["noise_scale_by_query_index"][
                query_index % len(transforms["noise_scale_by_query_index"])
            ])
            case = generate_latent_structure(
                latent_id, int(seed), role="validation-query", tempo=tempo,
                noise_scale=noise_scale, price_scale=float(transforms["price_scale"]),
                volume_scale=float(transforms["volume_scale"]),
            )
            query_episode = _trim_episode(case.episode)
            query_state = build_multiresolution_state(query_episode)
            query_legacy = represent(query_episode)
            new_ranked = sorted(
                (
                    multiresolution_state_distance(query_state, candidate["state"])[0],
                    candidate["latent_id"], candidate["seed"],
                )
                for candidate in candidates
            )
            legacy_ranked = sorted(
                (
                    representation_distance(query_legacy, candidate["legacy"])[0],
                    candidate["latent_id"], candidate["seed"],
                )
                for candidate in candidates
            )
            negative = generate_latent_structure(
                latent_id, int(seed), role="critical-negative", tempo=tempo,
                noise_scale=noise_scale, price_scale=float(transforms["price_scale"]),
                volume_scale=float(transforms["volume_scale"]), critical_negative=True,
            )
            negative_episode = _trim_episode(negative.episode)
            negative_state_distance = multiresolution_state_distance(
                query_state, build_multiresolution_state(negative_episode),
            )[0]
            negative_legacy_distance = representation_distance(
                query_legacy, represent(negative_episode),
            )[0]
            best_new_positive = min(
                distance for distance, label, _ in new_ranked if label == latent_id
            )
            best_legacy_positive = min(
                distance for distance, label, _ in legacy_ranked if label == latent_id
            )
            new_top = new_ranked[:top_k]
            legacy_top = legacy_ranked[:top_k]
            new_top1 = float(new_top[0][1] == latent_id)
            new_precision = sum(label == latent_id for _, label, _ in new_top) / top_k
            new_negative_error = float(negative_state_distance <= best_new_positive)
            legacy_top1 = float(legacy_top[0][1] == latent_id)
            legacy_precision = sum(label == latent_id for _, label, _ in legacy_top) / top_k
            legacy_negative_error = float(negative_legacy_distance <= best_legacy_positive)
            family_new[latent_id].append((new_top1, new_precision, new_negative_error))
            family_legacy[latent_id].append((legacy_top1, legacy_precision, legacy_negative_error))
            query_rows.append({
                "query_id": f"{latent_id}-validation-{seed}",
                "latent_id": latent_id,
                "seed": int(seed),
                "tempo": tempo,
                "noise_scale": noise_scale,
                "new_top1_correct": bool(new_top1),
                "new_precision_at_5": new_precision,
                "new_critical_negative_error": bool(new_negative_error),
                "new_best_positive_distance": best_new_positive,
                "new_critical_negative_distance": negative_state_distance,
                "new_top5": [
                    {"latent_id": label, "seed": candidate_seed, "distance": distance}
                    for distance, label, candidate_seed in new_top
                ],
                "legacy_top1_correct": bool(legacy_top1),
                "legacy_precision_at_5": legacy_precision,
                "legacy_critical_negative_error": bool(legacy_negative_error),
                "legacy_top5": [
                    {"latent_id": label, "seed": candidate_seed, "distance": distance}
                    for distance, label, candidate_seed in legacy_top
                ],
            })
    per_family: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    acceptance = payload["acceptance"]
    for latent_id in payload["latent_ids"]:
        new = np.asarray(family_new[latent_id], dtype=float)
        legacy = np.asarray(family_legacy[latent_id], dtype=float)
        metrics = {
            "queries": int(len(new)),
            "new_top1": float(new[:, 0].mean()),
            "new_precision_at_5": float(new[:, 1].mean()),
            "new_critical_negative_error": float(new[:, 2].mean()),
            "legacy_top1": float(legacy[:, 0].mean()),
            "legacy_precision_at_5": float(legacy[:, 1].mean()),
            "legacy_critical_negative_error": float(legacy[:, 2].mean()),
        }
        per_family[latent_id] = metrics
        if metrics["new_top1"] < float(acceptance["minimum_top1_per_family"]):
            failures.append(f"{latent_id}:top1={metrics['new_top1']:.3f} below threshold")
        if metrics["new_precision_at_5"] < float(acceptance["minimum_precision_at_k_per_family"]):
            failures.append(
                f"{latent_id}:precision@5={metrics['new_precision_at_5']:.3f} below threshold"
            )
        if metrics["new_critical_negative_error"] > float(
            acceptance["maximum_critical_negative_error_per_family"]
        ):
            failures.append(
                f"{latent_id}:critical-negative={metrics['new_critical_negative_error']:.3f} above threshold"
            )
    elapsed = perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if elapsed > float(acceptance["maximum_total_seconds"]):
        failures.append(f"elapsed {elapsed:.2f}s exceeds {acceptance['maximum_total_seconds']:.2f}s")
    if peak_rss_mb > float(acceptance["maximum_rss_mb"]):
        failures.append(f"RSS {peak_rss_mb:.1f}MB exceeds {acceptance['maximum_rss_mb']:.1f}MB")
    all_new = np.asarray([value for values in family_new.values() for value in values])
    all_legacy = np.asarray([value for values in family_legacy.values() for value in values])
    metrics: dict[str, Any] = {
        "schema_version": "m03-latent-structural-result-v1",
        "verifier_id": payload["verifier_id"],
        "verifier_config_digest": spec.digest,
        "distance_contract_digest": state_distance_contract()["digest"],
        "latent_id_used_by_distance": False,
        "real_forward_outcomes_accessed": False,
        "families": len(payload["latent_ids"]),
        "candidates": len(candidates),
        "validation_queries": len(query_rows),
        "overall_new_top1": float(all_new[:, 0].mean()),
        "overall_new_precision_at_5": float(all_new[:, 1].mean()),
        "overall_new_critical_negative_error": float(all_new[:, 2].mean()),
        "overall_legacy_top1": float(all_legacy[:, 0].mean()),
        "overall_legacy_precision_at_5": float(all_legacy[:, 1].mean()),
        "overall_legacy_critical_negative_error": float(all_legacy[:, 2].mean()),
        "per_family": per_family,
        "acceptance": acceptance,
        "elapsed_seconds": elapsed,
        "peak_rss_mb": peak_rss_mb,
        "query_result_digest": stable_hash(query_rows),
    }
    return StructuralVerification(not failures, metrics, tuple(failures), tuple(query_rows))


def write_structural_verification(
    result: StructuralVerification,
    spec: StructuralVerifierSpec,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    distance_path = output_dir / "state-distance-contract.json"
    distance_path.write_text(json.dumps(state_distance_contract(), indent=2, sort_keys=True) + "\n")
    payload = {
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "queries": list(result.queries),
    }
    machine_path = output_dir / "m03-latent-structural-verification.json"
    machine_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    family_rows = "".join(
        f"<tr><td>{escape(latent_id)}</td><td>{values['new_top1']:.1%}</td>"
        f"<td>{values['new_precision_at_5']:.1%}</td><td>{values['new_critical_negative_error']:.1%}</td>"
        f"<td>{values['legacy_top1']:.1%}</td><td>{values['legacy_precision_at_5']:.1%}</td></tr>"
        for latent_id, values in result.metrics["per_family"].items()
    )
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html_path = output_dir / "m03-latent-structural-verification.html"
    html_path.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>M03 latent structural verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1250px;margin:2rem auto;padding:0 1rem;background:#f4f6f7;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.25rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{white-space:pre-wrap}}</style></head><body><header><h1>M03 unnamed latent-structure verifier: <span class={status.lower()}>{status}</span></h1><p>The generator's hidden latent ID supplies the oracle only after both distances are computed. Neither matcher receives names, labels or outcomes. Validation seeds were frozen before execution and are disjoint from the development smoke.</p><p>Verifier <code>{spec.digest}</code> · distance <code>{result.metrics['distance_contract_digest']}</code></p></header><section><h2>Per-family locked result</h2><table><thead><tr><th>Hidden family</th><th>New top-1</th><th>New precision@5</th><th>New critical error</th><th>Legacy top-1</th><th>Legacy precision@5</th></tr></thead><tbody>{family_rows}</tbody></table></section><section><h2>Metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section><section><h2>Failures</h2><ul>{failures}</ul></section></body></html>""")
    return machine_path, html_path, distance_path

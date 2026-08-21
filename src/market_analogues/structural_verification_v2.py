from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import resource
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import yaml

from .latent_structures import LATENT_SPECS, generate_latent_structure
from .multiresolution import build_multiresolution_state
from .state_distance import (
    multiresolution_state_distance,
    multiresolution_state_distance_v2,
    state_distance_contract,
    state_distance_v2_contract,
)
from .structural_verification import _trim_episode
from .types import stable_hash


STRUCTURAL_VERIFIER_V2_SCHEMA = "latent-structural-verifier-v2"


class StructuralVerifierV2Error(ValueError):
    pass


@dataclass(frozen=True)
class StructuralVerifierV2Spec:
    source: Path
    payload: dict[str, Any]
    digest: str


@dataclass(frozen=True)
class StructuralVerificationV2:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    queries: tuple[dict[str, Any], ...]


def load_structural_verifier_v2_spec(path: str | Path) -> StructuralVerifierV2Spec:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text()) or {}
    if not isinstance(payload, Mapping):
        raise StructuralVerifierV2Error("structural verifier v2 must be a mapping")
    required = {
        "schema_version", "verifier_id", "distance_contract_digest",
        "baseline_distance_contract_digest", "baseline_result_sha256", "latent_ids",
        "candidate_seeds", "prior_observed_query_seeds", "validation_query_seeds",
        "positive_transformations", "critical_negative", "acceptance",
    }
    if set(payload) != required:
        raise StructuralVerifierV2Error(
            f"structural verifier v2 keys differ; missing={sorted(required - set(payload))}, "
            f"unknown={sorted(set(payload) - required)}"
        )
    if payload["schema_version"] != STRUCTURAL_VERIFIER_V2_SCHEMA:
        raise StructuralVerifierV2Error(f"unsupported structural v2 schema: {payload['schema_version']!r}")
    expected_ids = [spec.latent_id for spec in LATENT_SPECS]
    if payload["latent_ids"] != expected_ids:
        raise StructuralVerifierV2Error("latent IDs do not match the independent generator contract")
    for key in ("candidate_seeds", "prior_observed_query_seeds", "validation_query_seeds"):
        values = payload[key]
        if not isinstance(values, list) or len(values) < 5 or not all(isinstance(v, int) for v in values):
            raise StructuralVerifierV2Error(f"{key} must contain at least five integer seeds")
        if len(set(values)) != len(values):
            raise StructuralVerifierV2Error(f"{key} must not contain duplicates")
    candidates = set(payload["candidate_seeds"])
    observed = set(payload["prior_observed_query_seeds"])
    validation = set(payload["validation_query_seeds"])
    if validation & (candidates | observed):
        raise StructuralVerifierV2Error("validation seeds must be untouched and disjoint from all prior seeds")
    if payload["distance_contract_digest"] != state_distance_v2_contract()["digest"]:
        raise StructuralVerifierV2Error("v2 distance contract digest differs from implementation")
    if payload["baseline_distance_contract_digest"] != state_distance_contract()["digest"]:
        raise StructuralVerifierV2Error("baseline distance contract digest differs from v1")
    result_sha = payload["baseline_result_sha256"]
    if not isinstance(result_sha, str) or len(result_sha) != 64:
        raise StructuralVerifierV2Error("baseline_result_sha256 must bind the sealed v1 result")
    transforms = payload["positive_transformations"]
    if not isinstance(transforms, Mapping) or set(transforms) != {
        "tempo_by_query_index", "noise_scale_by_query_index", "price_scale", "volume_scale",
    }:
        raise StructuralVerifierV2Error("positive transformation contract differs")
    if not transforms["tempo_by_query_index"] or not transforms["noise_scale_by_query_index"]:
        raise StructuralVerifierV2Error("positive transformation schedules cannot be empty")
    negative = payload["critical_negative"]
    if not isinstance(negative, Mapping) or not negative or not all(value is True for value in negative.values()):
        raise StructuralVerifierV2Error("all critical-negative transformations must be enabled")
    acceptance = payload["acceptance"]
    required_acceptance = {
        "top_k", "minimum_top1_per_family", "minimum_precision_at_k_per_family",
        "maximum_critical_negative_error_per_family", "maximum_total_seconds", "maximum_rss_mb",
    }
    if not isinstance(acceptance, Mapping) or set(acceptance) != required_acceptance:
        raise StructuralVerifierV2Error("acceptance contract differs")
    if int(acceptance["top_k"]) != 5:
        raise StructuralVerifierV2Error("M03b top_k must be 5")
    for key in ("minimum_top1_per_family", "minimum_precision_at_k_per_family"):
        if not .75 <= float(acceptance[key]) <= 1:
            raise StructuralVerifierV2Error(f"{key} cannot be below 0.75")
    if not 0 <= float(acceptance["maximum_critical_negative_error_per_family"]) <= .02:
        raise StructuralVerifierV2Error("critical-negative threshold cannot exceed 0.02")
    canonical = json.loads(json.dumps(payload, sort_keys=True))
    return StructuralVerifierV2Spec(source, canonical, stable_hash(canonical))


def verify_latent_structures_v2(spec: StructuralVerifierV2Spec) -> StructuralVerificationV2:
    started = perf_counter()
    payload = spec.payload
    top_k = int(payload["acceptance"]["top_k"])
    candidates: list[dict[str, Any]] = []
    for latent_id in payload["latent_ids"]:
        for seed in payload["candidate_seeds"]:
            episode = _trim_episode(generate_latent_structure(
                latent_id, int(seed), role="candidate",
            ).episode)
            candidates.append({
                "latent_id": latent_id,
                "seed": int(seed),
                "state": build_multiresolution_state(episode),
            })

    family_v2 = {latent_id: [] for latent_id in payload["latent_ids"]}
    family_v1 = {latent_id: [] for latent_id in payload["latent_ids"]}
    query_rows: list[dict[str, Any]] = []
    transforms = payload["positive_transformations"]
    for latent_id in payload["latent_ids"]:
        for query_index, seed in enumerate(payload["validation_query_seeds"]):
            tempo = float(transforms["tempo_by_query_index"][
                query_index % len(transforms["tempo_by_query_index"])
            ])
            noise_scale = float(transforms["noise_scale_by_query_index"][
                query_index % len(transforms["noise_scale_by_query_index"])
            ])
            generation = {
                "role": "validation-query", "tempo": tempo, "noise_scale": noise_scale,
                "price_scale": float(transforms["price_scale"]),
                "volume_scale": float(transforms["volume_scale"]),
            }
            query_state = build_multiresolution_state(_trim_episode(
                generate_latent_structure(latent_id, int(seed), **generation).episode,
            ))
            v2_ranked = sorted(
                (multiresolution_state_distance_v2(query_state, candidate["state"])[0],
                 candidate["latent_id"], candidate["seed"])
                for candidate in candidates
            )
            v1_ranked = sorted(
                (multiresolution_state_distance(query_state, candidate["state"])[0],
                 candidate["latent_id"], candidate["seed"])
                for candidate in candidates
            )
            negative_state = build_multiresolution_state(_trim_episode(
                generate_latent_structure(
                    latent_id, int(seed), critical_negative=True, **generation,
                ).episode,
            ))
            v2_negative = multiresolution_state_distance_v2(query_state, negative_state)[0]
            v1_negative = multiresolution_state_distance(query_state, negative_state)[0]
            best_v2_positive = min(distance for distance, label, _ in v2_ranked if label == latent_id)
            best_v1_positive = min(distance for distance, label, _ in v1_ranked if label == latent_id)
            v2_top = v2_ranked[:top_k]
            v1_top = v1_ranked[:top_k]
            v2_values = (
                float(v2_top[0][1] == latent_id),
                sum(label == latent_id for _, label, _ in v2_top) / top_k,
                float(v2_negative <= best_v2_positive),
            )
            v1_values = (
                float(v1_top[0][1] == latent_id),
                sum(label == latent_id for _, label, _ in v1_top) / top_k,
                float(v1_negative <= best_v1_positive),
            )
            family_v2[latent_id].append(v2_values)
            family_v1[latent_id].append(v1_values)
            query_rows.append({
                "query_id": f"{latent_id}-validation-{seed}",
                "latent_id": latent_id,
                "seed": int(seed),
                "tempo": tempo,
                "noise_scale": noise_scale,
                "v2_top1_correct": bool(v2_values[0]),
                "v2_precision_at_5": v2_values[1],
                "v2_critical_negative_error": bool(v2_values[2]),
                "v2_best_positive_distance": best_v2_positive,
                "v2_critical_negative_distance": v2_negative,
                "v2_top5": [
                    {"latent_id": label, "seed": candidate_seed, "distance": distance}
                    for distance, label, candidate_seed in v2_top
                ],
                "v1_top1_correct": bool(v1_values[0]),
                "v1_precision_at_5": v1_values[1],
                "v1_critical_negative_error": bool(v1_values[2]),
                "v1_top5": [
                    {"latent_id": label, "seed": candidate_seed, "distance": distance}
                    for distance, label, candidate_seed in v1_top
                ],
            })

    acceptance = payload["acceptance"]
    per_family: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for latent_id in payload["latent_ids"]:
        v2 = np.asarray(family_v2[latent_id], dtype=float)
        v1 = np.asarray(family_v1[latent_id], dtype=float)
        values = {
            "queries": int(len(v2)),
            "v2_top1": float(v2[:, 0].mean()),
            "v2_precision_at_5": float(v2[:, 1].mean()),
            "v2_critical_negative_error": float(v2[:, 2].mean()),
            "v1_top1": float(v1[:, 0].mean()),
            "v1_precision_at_5": float(v1[:, 1].mean()),
            "v1_critical_negative_error": float(v1[:, 2].mean()),
        }
        per_family[latent_id] = values
        if values["v2_top1"] < float(acceptance["minimum_top1_per_family"]):
            failures.append(f"{latent_id}:top1={values['v2_top1']:.3f} below threshold")
        if values["v2_precision_at_5"] < float(acceptance["minimum_precision_at_k_per_family"]):
            failures.append(f"{latent_id}:precision@5={values['v2_precision_at_5']:.3f} below threshold")
        if values["v2_critical_negative_error"] > float(acceptance["maximum_critical_negative_error_per_family"]):
            failures.append(f"{latent_id}:critical-negative={values['v2_critical_negative_error']:.3f} above threshold")

    elapsed = perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if elapsed > float(acceptance["maximum_total_seconds"]):
        failures.append(f"elapsed {elapsed:.2f}s exceeds {acceptance['maximum_total_seconds']:.2f}s")
    if peak_rss_mb > float(acceptance["maximum_rss_mb"]):
        failures.append(f"RSS {peak_rss_mb:.1f}MB exceeds {acceptance['maximum_rss_mb']:.1f}MB")
    all_v2 = np.asarray([item for values in family_v2.values() for item in values])
    all_v1 = np.asarray([item for values in family_v1.values() for item in values])
    metrics = {
        "schema_version": "m03b-latent-structural-result-v1",
        "verifier_id": payload["verifier_id"],
        "verifier_config_digest": spec.digest,
        "distance_contract_digest": state_distance_v2_contract()["digest"],
        "baseline_distance_contract_digest": state_distance_contract()["digest"],
        "baseline_result_sha256": payload["baseline_result_sha256"],
        "latent_id_used_by_distance": False,
        "real_forward_outcomes_accessed": False,
        "families": len(payload["latent_ids"]),
        "candidates": len(candidates),
        "validation_queries": len(query_rows),
        "overall_v2_top1": float(all_v2[:, 0].mean()),
        "overall_v2_precision_at_5": float(all_v2[:, 1].mean()),
        "overall_v2_critical_negative_error": float(all_v2[:, 2].mean()),
        "overall_v1_top1": float(all_v1[:, 0].mean()),
        "overall_v1_precision_at_5": float(all_v1[:, 1].mean()),
        "overall_v1_critical_negative_error": float(all_v1[:, 2].mean()),
        "per_family": per_family,
        "acceptance": acceptance,
        "elapsed_seconds": elapsed,
        "peak_rss_mb": peak_rss_mb,
        "query_result_digest": stable_hash(query_rows),
    }
    return StructuralVerificationV2(not failures, metrics, tuple(failures), tuple(query_rows))


def write_structural_verification_v2(
    result: StructuralVerificationV2,
    spec: StructuralVerifierV2Spec,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    distance_path = output_dir / "state-distance-v2-contract.json"
    distance_path.write_text(json.dumps(state_distance_v2_contract(), indent=2, sort_keys=True) + "\n")
    payload = {
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "queries": list(result.queries),
    }
    machine_path = output_dir / "m03b-latent-structural-verification.json"
    machine_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = "".join(
        f"<tr><td>{escape(latent_id)}</td><td>{values['v2_top1']:.1%}</td>"
        f"<td>{values['v2_precision_at_5']:.1%}</td><td>{values['v2_critical_negative_error']:.1%}</td>"
        f"<td>{values['v1_top1']:.1%}</td><td>{values['v1_precision_at_5']:.1%}</td></tr>"
        for latent_id, values in result.metrics["per_family"].items()
    )
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html_path = output_dir / "m03b-latent-structural-verification.html"
    html_path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M03b latent structural verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1250px;margin:2rem auto;padding:0 1rem;background:#f4f6f7;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.25rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;text-align:left;border-bottom:1px solid #ddd}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{white-space:pre-wrap}}</style></head><body><header><h1>M03b topology-aware unnamed verifier: <span class={status.lower()}>{status}</span></h1><p>The hidden latent ID is consulted only after both label-blind distances rank candidates. V2 is compared with frozen V1 on untouched seeds; neither distance sees outcomes or names.</p><p>Verifier <code>{spec.digest}</code> · v2 distance <code>{result.metrics['distance_contract_digest']}</code></p></header><section><h2>Per-family locked result</h2><table><thead><tr><th>Hidden family</th><th>V2 top-1</th><th>V2 precision@5</th><th>V2 critical error</th><th>V1 top-1</th><th>V1 precision@5</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section><section><h2>Failures</h2><ul>{failures}</ul></section></body></html>""")
    return machine_path, html_path, distance_path

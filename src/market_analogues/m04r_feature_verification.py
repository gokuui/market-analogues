from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .adapters import OHLCVSource
from .authority import load_authority_artifact
from .exact_aligned_features import (
    SAMPLES_48_NAMES, SAMPLES_64_NAMES, exact_feature_contract,
    exact_features_to_representation, representation_to_exact_features,
)
from .exact_batch import sliding_exact_representations
from .m04_candidate_recall import M04CandidateRecallSpec
from .representation import (
    Representation, coarse_vector, dense_channels, represent, resample_optional,
    stage_signature, structural_signature,
)
from .episodes import build_episode
from .types import InstrumentKey, stable_hash


M04R_FEATURE_VERIFIER_SCHEMA = "m04r-exact-feature-kernel-verification-v1"
NATIVE_PARITY_TOLERANCE = 1e-10


class M04RFeatureVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class M04RFeatureVerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    cases: tuple[dict[str, Any], ...]
    contract: dict[str, Any]
    result_digest: str


def _legacy_scalar_representation(episode) -> Representation:
    """Pre-M04R-03 pandas path, retained only as an audit reference."""
    channels = dense_channels(episode)
    return Representation(
        channels,
        coarse_vector(channels),
        {name: resample_optional(channels[name], 48) for name in SAMPLES_48_NAMES},
        {name: resample_optional(channels[name], 64) for name in SAMPLES_64_NAMES},
        stage_signature(channels),
        structural_signature(channels),
    )


def _representation_delta(left: Representation, right: Representation) -> tuple[float, int]:
    maximum = max(
        float(np.max(np.abs(left.coarse.astype(float) - right.coarse.astype(float)))),
        float(np.max(np.abs(left.stage - right.stage))),
        float(np.max(np.abs(left.structural - right.structural))),
    )
    mask_mismatches = 0
    for collection_name in ("samples_48", "samples_64"):
        left_collection = getattr(left, collection_name)
        right_collection = getattr(right, collection_name)
        if left_collection.keys() != right_collection.keys():
            mask_mismatches += len(set(left_collection) ^ set(right_collection))
        for name in left_collection.keys() & right_collection.keys():
            x, y = left_collection[name], right_collection[name]
            if (x is None) != (y is None):
                mask_mismatches += 1
            elif x is not None and y is not None:
                maximum = max(maximum, float(np.max(np.abs(x - y))))
    return maximum, mask_mismatches


def verify_m04r_feature_kernel(
    spec: M04CandidateRecallSpec,
    source: OHLCVSource,
    artifact_dir: Path,
    *,
    dataset_id: str = "nasdaq",
) -> M04RFeatureVerificationResult:
    if source.spec.dataset_id != dataset_id:
        raise M04RFeatureVerificationError("source dataset differs")
    registry_path = artifact_dir / "gate12" / dataset_id / "query-registry.yaml"
    registry = yaml.safe_load(registry_path.read_text()) or {}
    if registry.get("registry_digest") != spec.payload["registry_digests"][dataset_id]:
        raise M04RFeatureVerificationError("frozen registry digest differs")
    authority_root = artifact_dir / "gate12" / "authorities" / dataset_id / "cases"
    episode_ids = [
        episode_id for episode_id in spec.payload["holdout_episode_ids"]
        if (authority_root / f"{episode_id}.json").exists()
    ]
    if len(episode_ids) != 12:
        raise M04RFeatureVerificationError(
            f"expected 12 {dataset_id} authorities, found {len(episode_ids)}"
        )

    failures: list[str] = []
    cases: list[dict[str, Any]] = []
    representations_verified = 0
    maximum_legacy_delta = 0.0
    maximum_scalar_sliding_delta = 0.0
    legacy_mask_mismatches = scalar_sliding_mask_mismatches = 0
    round_trip_mismatches = nonfinite_rows = 0

    for episode_id in episode_ids:
        authority_path = authority_root / f"{episode_id}.json"
        authority = load_authority_artifact(authority_path)
        payload = json.loads(authority_path.read_text())
        if authority.authority_digest != spec.payload["authority_digests"][episode_id]:
            raise M04RFeatureVerificationError(f"authority digest differs for {episode_id}")
        query_meta = payload["query"]
        if source.fingerprint(
            InstrumentKey(dataset_id, str(query_meta["symbol"])),
        ) != payload["source_fingerprint"]:
            raise M04RFeatureVerificationError(f"query source changed for {episode_id}")
        if source.benchmark_fingerprint() != payload["benchmark_fingerprint"]:
            raise M04RFeatureVerificationError(f"benchmark changed for {episode_id}")
        metadata = [payload["query"], *payload["matches"]]
        rows: list[dict[str, Any]] = []
        for item in metadata:
            episode = build_episode(
                source, InstrumentKey(dataset_id, str(item["symbol"])),
                str(item["cutoff"]), int(item["lookback"]),
                str(item["representation_version"]),
            )
            scalar = represent(episode)
            legacy = _legacy_scalar_representation(episode)
            sliding_batch = sliding_exact_representations(
                episode.bars, episode.benchmark,
                lookback=len(episode.bars), stride=1,
            )
            if len(sliding_batch.representations) != 1:
                failures.append(f"sliding cardinality failed {episode.key.id}")
                continue
            sliding = sliding_batch.representations[0]
            legacy_delta, legacy_masks = _representation_delta(scalar, legacy)
            sliding_delta, sliding_masks = _representation_delta(scalar, sliding)
            features = representation_to_exact_features(scalar)
            rebuilt = exact_features_to_representation(features)
            round_trip_delta, round_trip_masks = _representation_delta(scalar, rebuilt)
            finite = np.isfinite(features.vector).all()
            maximum_legacy_delta = max(maximum_legacy_delta, legacy_delta)
            maximum_scalar_sliding_delta = max(
                maximum_scalar_sliding_delta, sliding_delta,
            )
            legacy_mask_mismatches += legacy_masks
            scalar_sliding_mask_mismatches += sliding_masks
            round_trip_mismatches += int(round_trip_delta != 0.0 or round_trip_masks != 0)
            nonfinite_rows += int(not finite)
            representations_verified += 1
            if legacy_delta > NATIVE_PARITY_TOLERANCE or legacy_masks:
                failures.append(f"legacy parity failed {episode.key.id}")
            if sliding_delta != 0.0 or sliding_masks:
                failures.append(f"scalar/sliding identity failed {episode.key.id}")
            if round_trip_delta != 0.0 or round_trip_masks:
                failures.append(f"feature round trip failed {episode.key.id}")
            if not finite:
                failures.append(f"non-finite feature row {episode.key.id}")
            rows.append({
                "episode_id": episode.key.id,
                "symbol": str(item["symbol"]),
                "cutoff": str(item["cutoff"]),
                "legacy_maximum_delta": legacy_delta,
                "legacy_mask_mismatches": legacy_masks,
                "scalar_sliding_maximum_delta": sliding_delta,
                "scalar_sliding_mask_mismatches": sliding_masks,
                "feature_digest": features.digest,
            })
        cases.append({
            "query_episode_id": episode_id,
            "authority_digest": authority.authority_digest,
            "representations": rows,
        })
    if representations_verified != 252:
        failures.append(
            f"verified {representations_verified} representations; require 252"
        )
    contract = exact_feature_contract()
    metrics = {
        "schema_version": M04R_FEATURE_VERIFIER_SCHEMA,
        "dataset_id": dataset_id,
        "feature_contract_digest": contract["digest"],
        "m04_contract_digest": spec.digest,
        "authorities_verified": len(cases),
        "representations_verified": representations_verified,
        "maximum_legacy_scalar_delta": maximum_legacy_delta,
        "native_parity_tolerance": NATIVE_PARITY_TOLERANCE,
        "legacy_mask_mismatches": legacy_mask_mismatches,
        "maximum_scalar_sliding_delta": maximum_scalar_sliding_delta,
        "scalar_sliding_mask_mismatches": scalar_sliding_mask_mismatches,
        "feature_round_trip_mismatches": round_trip_mismatches,
        "nonfinite_feature_rows": nonfinite_rows,
        "legacy_proxy_used_as_exact_feature": False,
        "real_forward_outcomes_accessed": False,
    }
    deterministic = {
        "schema_version": M04R_FEATURE_VERIFIER_SCHEMA,
        "metrics": metrics,
        "failures": sorted(failures),
        "cases": cases,
        "contract": contract,
    }
    return M04RFeatureVerificationResult(
        not failures, metrics, tuple(sorted(failures)), tuple(cases), contract,
        stable_hash(deterministic),
    )


def write_m04r_feature_verification(
    result: M04RFeatureVerificationResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    machine_path = output_dir / "m04r-exact-feature-kernel.json"
    contract_path = output_dir / "exact-feature-kernel-contract.json"
    html_path = output_dir / "m04r-exact-feature-kernel.html"
    payload = {
        "schema_version": M04R_FEATURE_VERIFIER_SCHEMA,
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "cases": list(result.cases),
        "contract": result.contract,
        "result_digest": result.result_digest,
    }
    for path, value in ((machine_path, payload), (contract_path, result.contract)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    rows = "".join(
        f"<tr><td><code>{escape(str(row['query_episode_id']))}</code></td>"
        f"<td>{len(row['representations'])}</td>"
        f"<td><code>{escape(str(row['authority_digest']))}</code></td></tr>"
        for row in result.cases
    )
    status = "PASS" if result.passed else "FAIL"
    failure_items = "".join(
        f"<li>{escape(value)}</li>" for value in result.failures
    ) or "<li>None</li>"
    temporary = html_path.with_suffix(html_path.suffix + ".tmp")
    temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M04R exact feature kernel audit</title><style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;background:#f5f7f8;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}code{{font-size:.8em;overflow-wrap:anywhere}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{white-space:pre-wrap}}</style></head><body><header><h1>M04R exact-aligned feature kernel: <span class="{status.lower()}">{status}</span></h1><p>Scalar queries, sliding candidates and lossless proposal-source rows share distance-v1 fields. Legacy proxy meanings remain explicitly separate. Outcomes and setup names are excluded.</p><p>Contract: <code>{result.contract['digest']}</code> · Result: <code>{result.result_digest}</code></p></header><section><h2>Metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section><section><h2>Authorities</h2><table><thead><tr><th>Query episode</th><th>Representations</th><th>Authority digest</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Contract</h2><pre>{escape(json.dumps(result.contract, indent=2, sort_keys=True))}</pre></section><section><h2>Failures</h2><ul>{failure_items}</ul></section></body></html>""")
    temporary.replace(html_path)
    return machine_path, html_path, contract_path

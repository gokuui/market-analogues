from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from market_analogues.latent_structures import LATENT_SPECS, generate_latent_structure
from market_analogues.multiresolution import build_multiresolution_state
from market_analogues.state_distance import (
    multiresolution_state_distance_v2,
    state_distance_contract,
    state_distance_v2_contract,
)
from market_analogues.types import Episode, EpisodeKey
import market_analogues.structural_verification_v2 as verification
from market_analogues.structural_verification_v2 import (
    StructuralVerifierV2Error,
    load_structural_verifier_v2_spec,
    verify_latent_structures_v2,
    write_structural_verification_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def test_official_v2_spec_is_frozen_and_uses_untouched_seeds() -> None:
    spec = load_structural_verifier_v2_spec(ROOT / "config" / "structural-verifier-v2.yaml")
    assert spec.digest == "06b17330d48889bb27b9257fd4776dbc4066cc7f007ec3a626822f7949fb7245"
    assert spec.payload["validation_query_seeds"] == list(range(600, 608))
    assert not set(spec.payload["validation_query_seeds"]) & set(spec.payload["prior_observed_query_seeds"])


def _spec_payload() -> dict:
    return {
        "schema_version": "latent-structural-verifier-v2",
        "verifier_id": "m03b-test",
        "distance_contract_digest": state_distance_v2_contract()["digest"],
        "baseline_distance_contract_digest": state_distance_contract()["digest"],
        "baseline_result_sha256": "9" * 64,
        "latent_ids": [spec.latent_id for spec in LATENT_SPECS[:2]],
        "candidate_seeds": [10, 11, 12, 13, 14],
        "prior_observed_query_seeds": [100, 101, 102, 103, 104],
        "validation_query_seeds": [700, 701, 702, 703, 704],
        "positive_transformations": {
            "tempo_by_query_index": [.9, 1.0, 1.1],
            "noise_scale_by_query_index": [.9, 1.0, 1.1],
            "price_scale": 3.7,
            "volume_scale": 11.0,
        },
        "critical_negative": {
            "reverse_tail_direction": True,
            "invert_tail_participation": True,
            "weaken_tail_market_context": True,
        },
        "acceptance": {
            "top_k": 5,
            "minimum_top1_per_family": .75,
            "minimum_precision_at_k_per_family": .75,
            "maximum_critical_negative_error_per_family": .02,
            "maximum_total_seconds": 60.0,
            # ru_maxrss includes pytest/plugin and long-lived harness high-water
            # marks. This is not resource qualification; official specs stay frozen.
            "maximum_rss_mb": 16384.0,
        },
    }


def _write_spec(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(payload or _spec_payload(), sort_keys=False))
    return path


def test_v2_distance_contract_is_bound_to_v1_and_symmetric() -> None:
    contract = state_distance_v2_contract()
    assert contract["digest"] == "4771d409e0a497d3f483e64a6a9f954da040826adf8f6d458cf166502bf465d3"
    assert contract["base_distance_digest"] == state_distance_contract()["digest"]
    left = build_multiresolution_state(generate_latent_structure("latent-04", 400).episode)
    right = build_multiresolution_state(generate_latent_structure("latent-04", 401).episode)
    assert multiresolution_state_distance_v2(left, right)[0] == pytest.approx(
        multiresolution_state_distance_v2(right, left)[0],
    )


def test_v2_distance_is_unit_invariant_and_future_causal() -> None:
    base_case = generate_latent_structure("latent-03", 55)
    scaled_case = generate_latent_structure(
        "latent-03", 55, price_scale=7.0, volume_scale=13.0,
    )
    base = build_multiresolution_state(base_case.episode)
    scaled = build_multiresolution_state(scaled_case.episode)
    assert multiresolution_state_distance_v2(base, scaled)[0] < 1e-6

    episode = base_case.episode
    cutoff = episode.bars["timestamp"].iloc[-21]
    key = EpisodeKey(episode.key.instrument, cutoff, 252, episode.key.representation_version)
    causal = Episode(key, episode.bars.copy(), episode.benchmark.copy())
    mutated_bars = episode.bars.copy()
    mutated_benchmark = episode.benchmark.copy()
    future = mutated_bars["timestamp"] > cutoff
    future_benchmark = mutated_benchmark["timestamp"] > cutoff
    mutated_bars.loc[future, ["open", "high", "low", "close"]] *= 100
    mutated_bars.loc[future, "volume"] *= 100
    mutated_benchmark.loc[future_benchmark, "close"] *= 100
    mutated = Episode(key, mutated_bars, mutated_benchmark)
    assert build_multiresolution_state(causal).state_digest == build_multiresolution_state(mutated).state_digest


def test_v2_spec_rejects_reused_validation_seed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(verification, "LATENT_SPECS", LATENT_SPECS[:2])
    payload = _spec_payload()
    payload["validation_query_seeds"][0] = 100
    with pytest.raises(StructuralVerifierV2Error, match="untouched and disjoint"):
        load_structural_verifier_v2_spec(_write_spec(tmp_path, payload))


def test_small_v2_unseen_gate_writes_auditable_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(verification, "LATENT_SPECS", LATENT_SPECS[:2])
    spec = load_structural_verifier_v2_spec(_write_spec(tmp_path))
    result = verify_latent_structures_v2(spec)
    assert result.passed, result.failures
    assert result.metrics["latent_id_used_by_distance"] is False
    assert result.metrics["real_forward_outcomes_accessed"] is False
    assert result.metrics["validation_queries"] == 10
    machine, html, distance = write_structural_verification_v2(result, spec, tmp_path / "out")
    assert json.loads(machine.read_text())["passed"] is True
    assert "topology-aware unnamed verifier" in html.read_text()
    assert json.loads(distance.read_text())["digest"] == state_distance_v2_contract()["digest"]

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from market_analogues.latent_structures import LATENT_SPECS, generate_latent_structure
from market_analogues.multiresolution import build_multiresolution_state
from market_analogues.state_distance import (
    multiresolution_state_distance,
    state_distance_contract,
)
import market_analogues.structural_verification as verification
from market_analogues.structural_verification import (
    StructuralVerifierError,
    load_structural_verifier_spec,
    verify_latent_structures,
    write_structural_verification,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_spec(tmp_path: Path) -> Path:
    payload = yaml.safe_load((ROOT / "config" / "structural-verifier.yaml").read_text())
    payload["latent_ids"] = [spec.latent_id for spec in LATENT_SPECS[:2]]
    payload["candidate_seeds"] = [10, 11, 12, 13, 14]
    payload["validation_query_seeds"] = [500, 501, 502, 503, 504]
    payload["acceptance"]["maximum_total_seconds"] = 60.0
    path = tmp_path / "structural.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_official_structural_spec_is_frozen_and_uses_unseen_seeds() -> None:
    spec = load_structural_verifier_spec(ROOT / "config" / "structural-verifier.yaml")
    assert spec.digest == "5a59236fb207266ecc54c995dc63c6d505c6a54fb9eda5e46ec1e9c1468046bf"
    assert spec.payload["validation_query_seeds"] == list(range(200, 208))
    assert not set(spec.payload["candidate_seeds"]) & set(spec.payload["validation_query_seeds"])
    assert spec.payload["distance_contract_digest"] == state_distance_contract()["digest"]


def test_structural_spec_rejects_distance_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load((ROOT / "config" / "structural-verifier.yaml").read_text())
    payload["distance_contract_digest"] = "0" * 64
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(StructuralVerifierError, match="distance contract digest"):
        load_structural_verifier_spec(path)


def test_masked_state_distance_is_symmetric_and_rejects_critical_negative() -> None:
    base = generate_latent_structure("latent-03", 55, role="query", price_scale=7, volume_scale=13)
    positive = generate_latent_structure("latent-03", 56, role="candidate")
    negative = generate_latent_structure("latent-03", 55, role="negative", price_scale=7, volume_scale=13, critical_negative=True)
    left = build_multiresolution_state(base.episode)
    right = build_multiresolution_state(positive.episode)
    bad = build_multiresolution_state(negative.episode)
    forward = multiresolution_state_distance(left, right)[0]
    reverse = multiresolution_state_distance(right, left)[0]
    assert forward == pytest.approx(reverse)
    assert forward < multiresolution_state_distance(left, bad)[0]


def test_small_unseen_structural_gate_and_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(verification, "LATENT_SPECS", LATENT_SPECS[:2])
    spec = load_structural_verifier_spec(_small_spec(tmp_path))
    result = verify_latent_structures(spec)
    assert result.passed
    assert result.metrics["latent_id_used_by_distance"] is False
    assert result.metrics["real_forward_outcomes_accessed"] is False
    assert result.metrics["validation_queries"] == 10
    assert all(
        values["new_top1"] >= .75
        and values["new_precision_at_5"] >= .75
        and values["new_critical_negative_error"] <= .02
        for values in result.metrics["per_family"].values()
    )
    machine, html, distance = write_structural_verification(result, spec, tmp_path / "out")
    assert json.loads(machine.read_text())["passed"] is True
    assert "unnamed latent-structure verifier" in html.read_text()
    assert json.loads(distance.read_text())["digest"] == state_distance_contract()["digest"]

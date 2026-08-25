from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from market_analogues.distance import (
    representation_distance, representation_distance_lower_bound,
)
from market_analogues.exact_aligned_features import SAMPLES_48_NAMES
from market_analogues.quantized_bound import (
    ERROR_VALUE_COUNT, FLOAT16_MAX, PACKED_ROW_BYTES, QuantizedBoundError,
    quantize_bound_row, quantized_batch_lower_bounds, quantized_bound_contract,
    quantized_representation_lower_bound,
)
from market_analogues.quantized_bound_reference import reference_quantized_lower_bound
from market_analogues.m04r_quantized_verification import (
    verify_m04r_quantized_bound, write_m04r_quantized_verification,
)
from market_analogues.representation import Representation, represent
from market_analogues.synthetic import FAMILIES, generate_case
from market_analogues.types import stable_hash


def _assert_safe(query: Representation, candidate: Representation) -> None:
    row = quantize_bound_row(candidate)
    production = quantized_representation_lower_bound(query, row)
    reference = reference_quantized_lower_bound(query, row)
    native_total, native_components, _ = representation_distance_lower_bound(
        query, candidate,
    )
    exact_total, exact_components, _ = representation_distance(query, candidate)
    assert production.total == pytest.approx(reference.total, abs=2e-12, rel=2e-12)
    assert production.rigid_price == pytest.approx(reference.rigid_price, abs=2e-12)
    for name in production.components:
        assert production.components[name] == pytest.approx(
            reference.components[name], abs=2e-12, rel=2e-12,
        )
        assert production.components[name] <= native_components[name] + 1e-12
        assert native_components[name] <= exact_components[name] + 1e-12
    assert production.total <= native_total + 1e-12
    assert native_total <= exact_total + 1e-12


def test_quantized_bound_contract_and_complete_row_projection_are_frozen() -> None:
    contract = quantized_bound_contract()
    assert contract["layout"]["packed_row_bytes"] == PACKED_ROW_BYTES == 2432
    assert contract["stored_fields"]["error_radii"][0] == ERROR_VALUE_COUNT == 41
    assert contract["outcomes_or_labels_used"] is False
    assert contract["digest"] == "692e5f40b79397245d65604268c314790347fd0db124ddf985e8367ad36690d1"


def test_quantized_bound_matches_independent_reference_and_is_safe_for_all_families() -> None:
    representations = [
        represent(generate_case(name, 301 + index).episode)
        for index, name in enumerate(sorted(FAMILIES))
    ]
    for index, query in enumerate(representations):
        _assert_safe(query, query)
        _assert_safe(query, representations[(index + 1) % len(representations)])
    query = representations[0]
    rows = [quantize_bound_row(candidate) for candidate in representations]
    batch = quantized_batch_lower_bounds(query, rows)
    for index, row in enumerate(rows):
        scalar = quantized_representation_lower_bound(query, row)
        assert batch.totals[index] == pytest.approx(scalar.total, abs=2e-12)
        for name in scalar.components:
            assert batch.components[name][index] == pytest.approx(
                scalar.components[name], abs=2e-12,
            )


def test_batch_iqr_uses_one_combined_percentile_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = represent(generate_case("steady_trend", 311).episode)
    candidates = [
        represent(generate_case(name, 312 + index).episode)
        for index, name in enumerate(sorted(FAMILIES))
    ]
    rows = [quantize_bound_row(candidate) for candidate in candidates]
    original = np.percentile
    calls: list[object] = []

    def counted(values: np.ndarray, q: object, **kwargs: object) -> np.ndarray:
        calls.append(q)
        return original(values, q, **kwargs)

    monkeypatch.setattr(np, "percentile", counted)
    combined = quantized_batch_lower_bounds(query, rows)
    assert len(calls) == sum(
        query.samples_48.get(name) is not None for name in SAMPLES_48_NAMES
    )
    assert all(tuple(value) == (25, 75) for value in calls)
    for index, row in enumerate(rows):
        scalar = quantized_representation_lower_bound(query, row)
        assert combined.totals[index] == pytest.approx(scalar.total, abs=2e-12)


def test_error_radii_are_outward_and_cover_every_stored_field() -> None:
    candidate = represent(generate_case("volatile_reversal", 321).episode)
    row = quantize_bound_row(candidate)
    radii = row.error_radii.astype(float)
    coarse_error = np.sqrt(np.mean(
        (candidate.coarse.astype(float) - row.coarse.astype(float)) ** 2,
    ))
    assert radii[0] >= coarse_error
    count = len(SAMPLES_48_NAMES)
    for index, name in enumerate(SAMPLES_48_NAMES):
        values = candidate.samples_48[name]
        if values is None:
            assert not row.presence[index]
            continue
        error = np.asarray(values) - row.samples_48[index].astype(float)
        assert radii[1 + index] >= np.sqrt(np.mean(error * error))
        assert radii[1 + count + index] >= np.max(np.abs(error))


def test_missingness_branches_and_constant_std_fallback_are_safe() -> None:
    query = represent(generate_case("steady_trend", 331).episode)
    candidate = represent(generate_case("rounded_base", 332).episode)
    query_48 = dict(query.samples_48)
    candidate_48 = dict(candidate.samples_48)
    query_48["benchmark_path"] = None
    candidate_48["benchmark_return"] = None
    query = replace(query, samples_48=query_48)
    candidate = replace(candidate, samples_48=candidate_48)
    _assert_safe(query, candidate)

    constant_48 = {
        name: None if value is None else np.full(48, 1e-9)
        for name, value in candidate.samples_48.items()
    }
    constant = replace(
        candidate, coarse=np.full(128, 1e-9, dtype=np.float32),
        samples_48=constant_48, stage=np.full(48, 1e-9),
        structural=np.full(9, 1e-9),
    )
    _assert_safe(query, constant)


def test_float16_rounding_boundaries_are_safe_and_overflow_fails_closed() -> None:
    query = represent(generate_case("steady_trend", 341).episode)
    candidate = represent(generate_case("rounded_base", 342).episode)
    boundary = np.asarray([
        0.0,
        float(np.nextafter(np.float16(0), np.float16(1))),
        float(np.nextafter(np.float16(1), np.float16(2))),
        -float(np.nextafter(np.float16(1), np.float16(2))),
        FLOAT16_MAX,
        -FLOAT16_MAX,
    ])
    coarse = np.resize(boundary, 128).astype(np.float64)
    changed = replace(candidate, coarse=coarse)
    _assert_safe(query, changed)

    overflow = replace(candidate, coarse=np.full(128, FLOAT16_MAX + 1.0))
    with pytest.raises(QuantizedBoundError, match="exceeds float16"):
        quantize_bound_row(overflow)
    nonfinite = replace(candidate, stage=np.full(48, np.nan))
    with pytest.raises(QuantizedBoundError, match="non-finite"):
        quantize_bound_row(nonfinite)


def test_evidence_verifier_is_digest_bound_and_writes_deterministically(
    tmp_path: Path,
) -> None:
    contract_digest = quantized_bound_contract()["digest"]
    million = {
        "schema_version": "m04r-quantized-bound-million-gate-v1",
        "contract_digest": contract_digest,
        "seed": 1,
        "pairs": 1_000_000,
        "tolerance": 1e-12,
        "violations": 0,
        "unscaled_violations": 0,
        "maximum_excess": 0.0,
        "maximum_unscaled_excess": 0.0,
        "maximum_ratio": .9,
        "positive": 900_000,
        "boundary_pairs": 15_625,
        "boundary_violations": 0,
        "boundary_unscaled_violations": 0,
        "outcomes_or_labels_used": False,
    }
    million["result_digest"] = stable_hash(million)
    cases = [{
        "passed": True, "pruning_retention": .995, "symbols": 1,
        "selected_symbols": ["A"], "source_scope_digest": "source-digest",
        "benchmark_fingerprint": "benchmark-digest",
    } for _ in range(12)]
    authority = {
        "schema_version": "m04r-quantized-bound-authority-gate-v1",
        "contract_digest": contract_digest,
        "authority_cases": cases,
        "case_count": 12,
        "total_rows": 368_000,
        "minimum_pruning_retention": .995,
        "maximum_total_excess": 0.0,
        "maximum_component_excess": 0.0,
        "overflow_rows": 0,
        "required_minimum_retention": .99,
        "tolerance": 1e-12,
        "packed_row_bytes": PACKED_ROW_BYTES,
        "projected_3_82m_gib": 8.65,
        "all_cases_passed": True,
        "real_forward_outcomes_accessed": False,
    }
    authority["result_digest"] = stable_hash(authority)
    million_path, authority_path = tmp_path / "million.json", tmp_path / "authority.json"
    million_path.write_text(json.dumps(million))
    authority_path.write_text(json.dumps(authority))
    result = verify_m04r_quantized_bound(million_path, authority_path)
    assert result.passed
    output = tmp_path / "output"
    paths = write_m04r_quantized_verification(result, output)
    first = [path.read_bytes() for path in paths]
    write_m04r_quantized_verification(result, output)
    assert [path.read_bytes() for path in paths] == first

    million["violations"] = 1
    million_path.write_text(json.dumps(million))
    failed = verify_m04r_quantized_bound(million_path, authority_path)
    assert not failed.passed
    assert "million-pair evidence digest differs" in failed.failures

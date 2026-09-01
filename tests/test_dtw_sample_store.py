from pathlib import Path

import numpy as np
import pytest

from market_analogues.dtw_interval_bound import (
    quantize_dtw_samples,
    quantized_dtw_lower_bound,
)
from market_analogues.dtw_sample_store import (
    DTW_SAMPLE_DTYPE,
    DtwSampleStoreError,
    dtw_sample_lower_bounds,
    load_dtw_sample_generation,
    make_dtw_sample_record,
    make_zero_dtw_sample_record,
    validate_dtw_sample_records,
    write_dtw_sample_generation_from_shards,
)
from market_analogues.representation import represent
from market_analogues.synthetic import generate_case


def _packed_manifest(rows: int, overflow: int) -> dict[str, object]:
    return {
        "manifest_digest": "a" * 64, "rows_sha256": "b" * 64,
        "overflow_sha256": "c" * 64, "row_count": rows,
        "overflow_count": overflow, "provenance_digest": "d" * 64,
    }


def test_store_round_trip_is_aligned_and_batch_equals_scalar(tmp_path: Path) -> None:
    query = represent(generate_case("trend_contraction_breakout", 130_000).episode)
    candidate = represent(generate_case("rounded_base", 130_001).episode)
    row = make_dtw_sample_record(candidate)
    zero = make_zero_dtw_sample_record()
    main = tmp_path / "main.bin"; overflow = tmp_path / "overflow.bin"
    row.tofile(main); zero.tofile(overflow)
    packed = _packed_manifest(1, 1)
    generation = write_dtw_sample_generation_from_shards(
        tmp_path / "store", [main], [overflow], packed_manifest=packed,
        provenance={"source": "synthetic"},
    )
    loaded = load_dtw_sample_generation(
        tmp_path / "store", generation, packed_manifest=packed,
    )
    expected = quantized_dtw_lower_bound(query, quantize_dtw_samples(candidate))
    assert dtw_sample_lower_bounds(query, loaded.rows)[0] == pytest.approx(expected, abs=1e-12)
    assert dtw_sample_lower_bounds(query, loaded.overflow)[0] == 0


def test_invalid_order_and_content_corruption_fail_closed(tmp_path: Path) -> None:
    representation = represent(generate_case("rounded_base", 130_002).episode)
    row = make_dtw_sample_record(representation)
    row["orders"][0, 0, 0] = row["orders"][0, 0, 1]
    with pytest.raises(DtwSampleStoreError, match="values"):
        validate_dtw_sample_records(row)
    valid = make_dtw_sample_record(representation)
    main = tmp_path / "main.bin"; overflow = tmp_path / "overflow.bin"
    valid.tofile(main); np.empty(0, dtype=DTW_SAMPLE_DTYPE).tofile(overflow)
    packed = _packed_manifest(1, 0)
    generation = write_dtw_sample_generation_from_shards(
        tmp_path / "store", [main], [overflow], packed_manifest=packed,
        provenance={},
    )
    path = tmp_path / "store" / "generations" / generation / "dtw-samples.bin"
    with path.open("r+b") as handle:
        handle.seek(0); handle.write(b"X")
    with pytest.raises(DtwSampleStoreError, match="file differs"):
        load_dtw_sample_generation(
            tmp_path / "store", generation, packed_manifest=packed,
        )

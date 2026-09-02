from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_analogues.baseline_neighbors import recent_return_volatility
from market_analogues.baseline_feature_store import (
    FEATURE_DTYPE,
    BaselineFeatureStoreError,
    features_for_packed_records,
    load_feature_generation,
    make_feature_record,
    validate_feature_records,
    write_feature_generation_from_shards,
)


def test_feature_generation_roundtrip_and_packed_binding(tmp_path: Path) -> None:
    main = tmp_path / "main.bin"; overflow = tmp_path / "overflow.bin"
    np.concatenate((make_feature_record(np.asarray([.1, .2, .3])),
                    make_feature_record(np.full(3, np.nan)))).tofile(main)
    make_feature_record(np.asarray([1., 2., 3.])).tofile(overflow)
    packed = {"manifest_digest": "packed", "provenance_digest": "prefix",
              "row_count": 2, "overflow_count": 1}
    root = tmp_path / "store"
    generation = write_feature_generation_from_shards(
        root, [main], [overflow], packed_manifest=packed,
        provenance={"purpose": "test"},
    )
    loaded = load_feature_generation(root, generation, packed_manifest=packed)
    np.testing.assert_equal(loaded.rows["values"],
                            [[.1, .2, .3], [np.nan, np.nan, np.nan]])
    with pytest.raises(BaselineFeatureStoreError, match="manifest differs"):
        load_feature_generation(root, generation,
                                packed_manifest={**packed, "manifest_digest": "changed"})


def test_feature_records_reject_partial_missing() -> None:
    rows = np.empty(1, dtype=FEATURE_DTYPE)
    rows["values"][0] = [1., np.nan, 2.]
    with pytest.raises(BaselineFeatureStoreError, match="finiteness"):
        validate_feature_records(rows)


def test_packed_feature_builder_preserves_order_and_last_duplicate() -> None:
    timestamps = pd.to_datetime([*range(70), 69], unit="D", origin="2020-01-01")
    close = np.exp(np.arange(71, dtype=float) * .01)
    frame = pd.DataFrame({"timestamp": timestamps, "close": close})
    packed_dtype = np.dtype([("cutoff_ns", "<i8")])
    records = np.asarray([
        (int(timestamps[69].value),),
        (int(timestamps[63].value),),
    ], dtype=packed_dtype)
    observed = features_for_packed_records(frame, records)["values"]
    expected = np.vstack((
        recent_return_volatility(close),
        recent_return_volatility(close[:64]),
    ))
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)

    records["cutoff_ns"][0] = int(pd.Timestamp("1990-01-01").value)
    with pytest.raises(BaselineFeatureStoreError, match="absent"):
        features_for_packed_records(frame, records)

    with pytest.raises(BaselineFeatureStoreError, match="not ordered"):
        features_for_packed_records(frame.iloc[::-1], records[1:])

    with pytest.raises(BaselineFeatureStoreError, match="inputs differ"):
        features_for_packed_records(frame, np.asarray([1], dtype=np.int64))

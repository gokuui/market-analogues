from pathlib import Path

import numpy as np
import pytest

from market_analogues.baseline_feature_store import (
    FEATURE_DTYPE,
    BaselineFeatureStoreError,
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

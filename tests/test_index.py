from __future__ import annotations

import numpy as np

from market_analogues.index import CoarseIndex, recall_at_k


def test_index_roundtrip_and_recall(tmp_path) -> None:
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(200, 128)).astype(np.float32)
    ids = [f"e-{i}" for i in range(len(vectors))]
    reference = CoarseIndex(ids, vectors)
    path = tmp_path / "index.npz"
    reference.save(path)
    loaded = CoarseIndex.load(path)
    assert loaded.query(vectors[17], 1)[0][0] == "e-17"
    assert recall_at_k(reference, loaded, vectors[:10], 50) == 1.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class CoarseIndex:
    """Portable exact coarse index; exact results are the ANN recall oracle."""

    ids: list[str]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or len(self.ids) != len(self.vectors):
            raise ValueError("ids and 2-D vectors must have equal length")
        self._model = NearestNeighbors(metric="euclidean", algorithm="auto")
        if len(self.vectors):
            self._model.fit(self.vectors)

    def query(self, vector: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
        if not len(self.vectors):
            return []
        count = min(max(1, k), len(self.vectors))
        distances, positions = self._model.kneighbors(
            np.asarray(vector, dtype=np.float32).reshape(1, -1), n_neighbors=count,
        )
        return [(self.ids[int(i)], float(d)) for i, d in zip(positions[0], distances[0])]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.asarray(self.ids), vectors=self.vectors)

    @classmethod
    def load(cls, path: Path) -> "CoarseIndex":
        data = np.load(path, allow_pickle=False)
        return cls(data["ids"].astype(str).tolist(), data["vectors"])


def recall_at_k(reference: CoarseIndex, proposed: CoarseIndex, queries: np.ndarray, k: int = 50) -> float:
    if len(queries) == 0:
        return 1.0
    scores: list[float] = []
    for query in queries:
        expected = {item[0] for item in reference.query(query, k)}
        actual = {item[0] for item in proposed.query(query, k)}
        scores.append(len(expected & actual) / max(len(expected), 1))
    return float(np.mean(scores))

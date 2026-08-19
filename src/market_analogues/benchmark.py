from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from html import escape
import math
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from .distance import bounded_dtw, representation_distance
from .representation import Representation, represent
from .synthetic import SyntheticCase, transform_case, verification_corpus
from .types import Episode, EpisodeKey, InstrumentKey


DistanceFunction = Callable[[Representation, Representation], float]


@dataclass(frozen=True)
class MethodResult:
    method: str
    suite: str
    metrics: dict[str, float | int | str]


def _resample(series: pd.Series, n: int = 64) -> np.ndarray:
    raw = series.astype(float).to_numpy()
    valid = np.isfinite(raw)
    if not valid.any():
        return np.zeros(n)
    idx = np.arange(len(raw))
    filled = np.interp(idx, idx[valid], raw[valid])
    return np.interp(np.linspace(0, len(raw) - 1, n), idx, filled)


def _path(rep: Representation) -> np.ndarray:
    return _resample(rep.channels["close_path"])


def _z(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    return (values - np.mean(values)) / std if std > 1e-12 else np.zeros_like(values)


def euclidean_path(a: Representation, b: Representation) -> float:
    x, y = _path(a), _path(b)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def correlation_path(a: Representation, b: Representation) -> float:
    x, y = _path(a), _path(b)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0 if np.allclose(x, y) else 2.0
    return float(1 - np.corrcoef(x, y)[0, 1])


def local_bounded_dtw(a: Representation, b: Representation) -> float:
    return bounded_dtw(_path(a), _path(b), .12)[0]


def local_composite(a: Representation, b: Representation) -> float:
    return representation_distance(a, b)[0]


def _tslearn_dtw(a: Representation, b: Representation) -> float:
    from tslearn.metrics import dtw
    return float(dtw(
        _path(a), _path(b), global_constraint="sakoe_chiba", sakoe_chiba_radius=8,
    ))


def _aeon_dtw(a: Representation, b: Representation) -> float:
    from aeon.distances import dtw_distance
    return float(dtw_distance(_path(a), _path(b), window=.12))


def _stumpy_mass(a: Representation, b: Representation) -> float:
    import stumpy
    result = np.asarray(stumpy.mass(_path(a), _path(b)), dtype=float)
    value = float(result[0]) if len(result) else float("inf")
    return value if np.isfinite(value) else float("inf")


def _shape_dtw(a: Representation, b: Representation) -> float:
    from shapedtw.shapedtw import shape_dtw
    from shapedtw.shapeDescriptors import CompoundDescriptor, PAADescriptor, SlopeDescriptor
    descriptor = CompoundDescriptor(
        [SlopeDescriptor(slope_window=3), PAADescriptor(piecewise_aggregation_window=3)],
        descriptors_weights=[3.0, 1.0],
    )
    result = shape_dtw(
        x=_path(a), y=_path(b), subsequence_width=8, shape_descriptor=descriptor,
    )
    return float(result.normalized_distance)


def available_methods(include_external: bool = True, include_slow: bool = False) -> dict[str, DistanceFunction]:
    methods: dict[str, DistanceFunction] = {
        "euclidean_path": euclidean_path,
        "correlation_path": correlation_path,
        "bounded_dtw_local": local_bounded_dtw,
        "composite_local": local_composite,
    }
    if include_external and importlib.util.find_spec("tslearn"):
        methods["dtw_tslearn"] = _tslearn_dtw
    if include_external and importlib.util.find_spec("aeon"):
        methods["dtw_aeon"] = _aeon_dtw
    if include_external and importlib.util.find_spec("stumpy"):
        methods["mass_stumpy"] = _stumpy_mass
    if include_external and include_slow and importlib.util.find_spec("shapedtw"):
        methods["shape_dtw"] = _shape_dtw
    return methods


def _ndcg(labels: list[int], relevant: int, k: int = 10) -> float:
    dcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(labels[:k]))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(relevant, k)))
    return dcg / ideal if ideal else 1.0


def benchmark_synthetic(
    seeds_per_family: int = 5,
    methods: dict[str, DistanceFunction] | None = None,
) -> list[MethodResult]:
    """Compare methods on the same deterministic financial morphology oracle."""
    methods = methods or available_methods()
    corpus = verification_corpus(seeds_per_family)
    candidate_reps = [(case, represent(case.episode)) for case in corpus]
    queries = corpus[::2]
    results: list[MethodResult] = []
    for method_name, distance in methods.items():
        started = perf_counter()
        rank_one = pair_hits = negative_errors = 0
        recalls: list[float] = []
        ndcgs: list[float] = []
        for original in queries:
            query = transform_case(
                original, name="benchmark-query", price_scale=5.7,
                volume_scale=19, time_shift_days=1000,
            )
            reverse = transform_case(original, name="benchmark-reverse", reverse_returns=True)
            query_rep, original_rep = represent(query.episode), represent(original.episode)
            positive_distance = distance(query_rep, original_rep)
            negative_distance = distance(query_rep, represent(reverse.episode))
            pair_hits += int(positive_distance < negative_distance)
            negative_errors += int(negative_distance <= positive_distance)
            ranked = sorted(
                ((distance(query_rep, rep), case) for case, rep in candidate_reps),
                key=lambda item: (item[0], item[1].episode.key.id),
            )[:10]
            rank_one += int(ranked[0][1].episode.key.instrument == original.episode.key.instrument)
            labels = [int(case.family == original.family) for _, case in ranked]
            recalls.append(sum(labels) / seeds_per_family)
            ndcgs.append(_ndcg(labels, seeds_per_family))
        n = len(queries)
        results.append(MethodResult(method_name, "financial_synthetic_v1", {
            "queries": n,
            "exact_clone_rank1": rank_one / n,
            "pair_ordering": pair_hits / n,
            "critical_negative_rate": negative_errors / n,
            "family_recall_at_10": float(np.mean(recalls)),
            "ndcg_at_10": float(np.mean(ndcgs)),
            "seconds": perf_counter() - started,
        }))
    return results


def _ucr_episode(values: np.ndarray, symbol: str, later: bool = False) -> Episode:
    values = np.asarray(values, dtype=float).ravel()
    scale = max(float(np.std(values)), 1e-8)
    close = 100 * np.exp(.12 * (values - values[0]) / scale)
    previous = np.r_[close[0], close[:-1]]
    movement = np.abs(np.diff(np.log(close), prepend=np.log(close[0])))
    high = np.maximum(previous, close) * (1 + .1 * movement + .001)
    low = np.minimum(previous, close) * (1 - .1 * movement - .001)
    start = "2030-01-01" if later else "2000-01-03"
    timestamps = pd.date_range(start, periods=len(close), freq="B")
    bars = pd.DataFrame({
        "timestamp": timestamps, "open": previous, "high": high, "low": low,
        "close": close, "volume": np.full(len(close), 1_000_000),
    })
    benchmark = pd.DataFrame({"timestamp": timestamps, "close": np.full(len(close), 1000.0)})
    key = EpisodeKey(InstrumentKey("ucr", symbol), timestamps[-1], len(close), "dense-v1")
    return Episode(key, bars, benchmark)


def benchmark_ucr(
    dataset: str = "GunPoint",
    test_limit: int = 40,
    methods: dict[str, DistanceFunction] | None = None,
) -> list[MethodResult]:
    """Independent 1-NN shape benchmark; requires the optional tslearn dataset loader."""
    from tslearn.datasets import UCR_UEA_datasets
    x_train, y_train, x_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    if x_train is None:
        raise RuntimeError(f"could not load UCR dataset {dataset}")
    methods = methods or available_methods()
    train = [represent(_ucr_episode(x, f"train-{i}")) for i, x in enumerate(x_train)]
    test_indices = np.linspace(0, len(x_test) - 1, min(test_limit, len(x_test)), dtype=int)
    test = [(represent(_ucr_episode(x_test[i], f"test-{i}", later=True)), y_test[i]) for i in test_indices]
    results: list[MethodResult] = []
    for method_name, distance in methods.items():
        started = perf_counter()
        correct = 0
        for query, label in test:
            nearest = min(
                range(len(train)), key=lambda position: distance(query, train[position]),
            )
            correct += int(y_train[nearest] == label)
        results.append(MethodResult(method_name, f"ucr_{dataset}", {
            "train_cases": len(train), "test_cases": len(test),
            "one_nn_accuracy": correct / len(test),
            "seconds": perf_counter() - started,
        }))
    return results


def results_frame(results: list[MethodResult]) -> pd.DataFrame:
    rows = [{"suite": result.suite, "method": result.method, **result.metrics} for result in results]
    return pd.DataFrame(rows)


def write_comparison_report(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = []
    for suite, group in frame.groupby("suite", sort=False):
        tables.append(f"<h2>{escape(str(suite))}</h2>{group.drop(columns=['suite']).to_html(index=False, border=0, float_format=lambda x: f'{x:.4f}')}")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analogue method comparison</title><style>body{{font-family:system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;background:#f7f8fa;color:#17202a}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:.65rem;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{background:#fff4dc;border-left:5px solid #b9770e;padding:1rem}}</style></head><body>
<h1>Historical analogue method comparison</h1><p class="note">The financial synthetic suite tests domain-specific metamorphic relations. UCR is an independent generic shape benchmark. Neither alone establishes trading usefulness; a method that wins one may lose the other.</p>{''.join(tables)}</body></html>"""
    path.write_text(html)
    return path

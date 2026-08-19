from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
import resource
from time import perf_counter

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .distance import representation_distance_lower_bound
from .exact_batch import (
    EXACT_BATCH_VERSION, batch_representation_lower_bounds,
    sliding_exact_representations,
)
from .representation import represent
from .types import Episode, EpisodeKey, InstrumentKey


@dataclass(frozen=True)
class ExactBatchVerification:
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    cases: pd.DataFrame


def _episode(
    bars: pd.DataFrame, benchmark: pd.DataFrame | None,
    key: InstrumentKey, position: int, lookback: int, version: str,
) -> Episode:
    window = bars.iloc[position - lookback + 1:position + 1].reset_index(drop=True)
    cutoff = pd.Timestamp(window.timestamp.iloc[-1])
    context = benchmark[benchmark.timestamp <= cutoff].copy() if benchmark is not None else None
    return Episode(EpisodeKey(key, cutoff, lookback, version), window, context)


def verify_exact_batch_kernel(
    sources: dict[str, OHLCVSource],
    registries: dict[str, pd.DataFrame],
    *,
    windows_per_symbol: int = 100,
    stride: int = 5,
    batch_size: int = 128,
    tolerance: float = 1e-12,
    minimum_speedup: float = 20.0,
    maximum_rss_mb: float = 512.0,
    timing_repeats: int = 3,
) -> ExactBatchVerification:
    if windows_per_symbol < 1 or stride < 1 or batch_size < 1 or timing_repeats < 1:
        raise ValueError("window, stride, batch and repeat controls must be positive")
    if tolerance < 0 or minimum_speedup <= 0 or maximum_rss_mb <= 0:
        raise ValueError("tolerance/resource controls are invalid")
    rows = []
    failures: list[str] = []
    for dataset_id in sorted(registries):
        source = sources[dataset_id]
        benchmark = source.load_benchmark()
        registry = registries[dataset_id]
        for symbol in sorted(registry.symbol.astype(str).unique()):
            record = registry[registry.symbol.astype(str) == symbol].iloc[0]
            lookback = int(record.lookback)
            version = str(record.representation_version)
            key = InstrumentKey(dataset_id, symbol)
            bars = source.load(key)
            count = min(windows_per_symbol, (len(bars) - lookback) // stride + 1)
            required = lookback + (count - 1) * stride
            frame = bars.tail(required).reset_index(drop=True)
            query = represent(_episode(
                frame, benchmark, key, len(frame) - 1, lookback, version,
            ))
            warm_length = min(len(frame), lookback + 9 * stride)
            sliding_exact_representations(
                frame.tail(warm_length).reset_index(drop=True), benchmark,
                lookback=lookback, stride=stride, batch_size=batch_size,
            )
            batch_timings = []
            for _ in range(timing_repeats):
                started = perf_counter()
                batch = sliding_exact_representations(
                    frame, benchmark, lookback=lookback, stride=stride,
                    batch_size=batch_size,
                )
                measured = batch_representation_lower_bounds(query, batch.representations)
                batch_timings.append(perf_counter() - started)
            batch_seconds = float(np.median(batch_timings))
            scalar_totals = []
            maximum_component_delta = 0.0
            started = perf_counter()
            for position in batch.positions:
                candidate = represent(_episode(
                    frame, benchmark, key, int(position), lookback, version,
                ))
                total, components, rigid = representation_distance_lower_bound(
                    query, candidate,
                )
                scalar_totals.append(total)
                row = len(scalar_totals) - 1
                maximum_component_delta = max(
                    maximum_component_delta,
                    abs(measured.rigid_price[row] - rigid),
                    *(abs(measured.components[name][row] - value)
                      for name, value in components.items()),
                )
            scalar_seconds = perf_counter() - started
            maximum_total_delta = float(np.max(np.abs(
                measured.totals - np.asarray(scalar_totals),
            )))
            speedup = scalar_seconds / max(batch_seconds, 1e-12)
            case_passed = (
                maximum_total_delta <= tolerance
                and maximum_component_delta <= tolerance
                and speedup >= minimum_speedup
            )
            if not case_passed:
                failures.append(
                    f"{dataset_id}:{symbol} parity/speed failed: "
                    f"total={maximum_total_delta:.3e}, component={maximum_component_delta:.3e}, "
                    f"speedup={speedup:.2f}x"
                )
            rows.append({
                "dataset": dataset_id, "symbol": symbol, "windows": len(batch.positions),
                "batch_seconds": batch_seconds, "scalar_seconds": scalar_seconds,
                "speedup": speedup, "windows_per_second": len(batch.positions) / batch_seconds,
                "maximum_total_delta": maximum_total_delta,
                "maximum_component_delta": maximum_component_delta,
                "passed": case_passed,
            })
    cases = pd.DataFrame(rows)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if rss > maximum_rss_mb:
        failures.append(f"peak RSS {rss:.2f} MB exceeds {maximum_rss_mb:.2f} MB")
    metrics = {
        "kernel_version": EXACT_BATCH_VERSION,
        "cases": len(cases),
        "windows": int(cases.windows.sum()) if len(cases) else 0,
        "minimum_speedup": float(cases.speedup.min()) if len(cases) else 0.0,
        "aggregate_speedup": (
            float(cases.scalar_seconds.sum() / cases.batch_seconds.sum())
            if len(cases) else 0.0
        ),
        "maximum_total_delta": float(cases.maximum_total_delta.max()) if len(cases) else 0.0,
        "maximum_component_delta": (
            float(cases.maximum_component_delta.max()) if len(cases) else 0.0
        ),
        "peak_rss_mb": rss,
        "required_minimum_speedup": minimum_speedup,
        "maximum_rss_mb": maximum_rss_mb,
        "batch_size": batch_size,
        "timing_repeats": timing_repeats,
        "stride": stride,
    }
    return ExactBatchVerification(bool(len(cases)) and not failures, metrics, tuple(failures), cases)


def write_exact_batch_report(result: ExactBatchVerification, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = result.cases.to_html(index=False, float_format=lambda value: f"{value:.6g}")
    status = "PASS" if result.passed else "FAIL"
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exact batch kernel verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1450px;margin:2rem auto;background:#f5f7f8;color:#17202a}}section,header{{background:white;padding:1.2rem;margin:1rem;border:1px solid #ddd;border-radius:10px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd}}</style></head><body><header><h1>T12-02 exact batch kernel: {status}</h1><p>Scalar differential, throughput and memory evidence across every frozen Gate 12 symbol.</p></header><section><pre>{escape(json.dumps(result.metrics, indent=2))}</pre></section><section>{table}</section><section><h2>Failures</h2><ul>{failures}</ul></section></body></html>""")

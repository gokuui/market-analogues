from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .episodes import build_episode
from .exact_storage_feasibility import quantize_representation
from .pruning_verification import _candidate_from_row
from .search import SearchCandidate, score_candidates, select_scored
from .types import InstrumentKey, SearchQuery


@dataclass(frozen=True)
class PrecisionVerification:
    passed: bool
    metrics: dict[str, object]
    failures: tuple[str, ...]
    cases: pd.DataFrame


def _graded_ndcg(expected: list[str], actual: list[str]) -> float:
    relevance = {
        episode_id: 1 / np.log2(rank + 1)
        for rank, episode_id in enumerate(expected, 1)
    }
    ideal = sum(
        relevance[episode_id] / np.log2(rank + 1)
        for rank, episode_id in enumerate(expected, 1)
    )
    received = sum(
        relevance.get(episode_id, 0.0) / np.log2(rank + 1)
        for rank, episode_id in enumerate(actual, 1)
    )
    return float(received / ideal) if ideal else 0.0


def verify_float16_oracle_precision(
    source: OHLCVSource,
    oracle_directory: Path,
    *,
    representation_version: str = "dense-v1",
    minimum_recall: float = 1.0,
    minimum_ndcg: float = .999,
) -> PrecisionVerification:
    """Measure practical candidate-side float16 impact against Gate 09 native rankings."""
    summary = pd.read_parquet(oracle_directory / "oracle-summary.parquet")
    benchmark = source.load_benchmark()
    dataset_id = source.instruments()[0].dataset_id
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    started = perf_counter()
    for case in summary.sort_values("case_id").itertuples(index=False):
        ranking = pd.read_parquet(oracle_directory / f"{case.case_id}.parquet")
        query_symbol = str(case.query).split(":", 1)[-1]
        query = build_episode(
            source, InstrumentKey(dataset_id, query_symbol), case.cutoff,
            int(case.lookback), representation_version,
        )
        candidates = [
            _candidate_from_row(source, benchmark, row, representation_version)
            for row in ranking.itertuples(index=False)
        ]
        quantized = [
            SearchCandidate(
                candidate.episode,
                quantize_representation(candidate.representation, "float16"),
            )
            for candidate in candidates
        ]
        finite = all(
            np.isfinite(candidate.representation.coarse).all()
            and np.isfinite(candidate.representation.stage).all()
            and np.isfinite(candidate.representation.structural).all()
            and all(
                values is None or np.isfinite(values).all()
                for collection in (
                    candidate.representation.samples_48,
                    candidate.representation.samples_64,
                )
                for values in collection.values()
            )
            for candidate in quantized
        )
        request = SearchQuery(
            query.key, (dataset_id,), ("A", "B"), int(case.top_k),
            minimum_history_gap_bars=60,
        )
        scored = score_candidates(query, quantized, request)
        selected = select_scored(scored, request)
        actual_ids = [match.episode_key.id for match in selected]
        expected = ranking.loc[ranking.oracle_selected].sort_values(
            ["oracle_rank", "episode_id"], kind="stable",
        )
        expected_ids = expected.episode_id.astype(str).tolist()
        recall = (
            len(set(actual_ids).intersection(expected_ids)) / len(expected_ids)
            if expected_ids else 0.0
        )
        ndcg = _graded_ndcg(expected_ids, actual_ids)
        native_by_id = ranking.set_index("episode_id")
        maximum_total_delta = 0.0
        maximum_component_delta = 0.0
        for item in scored:
            episode_id = item.match.episode_key.id
            native = native_by_id.loc[episode_id]
            maximum_total_delta = max(
                maximum_total_delta,
                abs(item.match.total_distance - float(native.exact_distance)),
            )
            for name, value in item.match.component_distances.items():
                maximum_component_delta = max(
                    maximum_component_delta,
                    abs(value - float(native[f"component_{name}"])),
                )
        passed = finite and recall >= minimum_recall and ndcg >= minimum_ndcg
        if not passed:
            failures.append(
                f"{case.case_id}:finite={finite}, recall={recall:.6f}, nDCG={ndcg:.6f}"
            )
        rows.append({
            "case_id": str(case.case_id), "candidates": len(ranking),
            "finite": finite, "top_k_recall": recall,
            "ordered_ids_equal": actual_ids == expected_ids,
            "graded_ndcg": ndcg,
            "maximum_total_delta": maximum_total_delta,
            "maximum_component_delta": maximum_component_delta,
            "passed": passed,
        })
    cases = pd.DataFrame(rows)
    metrics = {
        "dataset": dataset_id,
        "layout": "float16",
        "quantization_scope": "candidate representations; query remains native",
        "cases": len(cases),
        "cases_passed": int(cases.passed.sum()) if len(cases) else 0,
        "candidates": int(cases.candidates.sum()) if len(cases) else 0,
        "minimum_top_k_recall": float(cases.top_k_recall.min()) if len(cases) else 0.0,
        "minimum_graded_ndcg": float(cases.graded_ndcg.min()) if len(cases) else 0.0,
        "ordered_cases_equal": int(cases.ordered_ids_equal.sum()) if len(cases) else 0,
        "maximum_total_delta": float(cases.maximum_total_delta.max()) if len(cases) else 0.0,
        "maximum_component_delta": (
            float(cases.maximum_component_delta.max()) if len(cases) else 0.0
        ),
        "minimum_required_recall": minimum_recall,
        "minimum_required_ndcg": minimum_ndcg,
        "seconds": perf_counter() - started,
    }
    return PrecisionVerification(bool(len(cases)) and not failures, metrics, tuple(failures), cases)


def write_precision_report(result: PrecisionVerification, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if result.passed else "FAIL"
    failures = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    table = result.cases.to_html(index=False, float_format=lambda value: f"{value:.8g}")
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Float16 retrieval precision audit</title><style>body{{font-family:system-ui,sans-serif;max-width:1450px;margin:2rem auto;background:#f5f7f8;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd}}</style></head><body><header><h1>Float16 practical retrieval audit: {status}</h1><p>This is a behavioral approximation test, not a claim of native numeric equality or a proof that quantized values are safe native lower bounds.</p></header><section><pre>{escape(json.dumps(result.metrics, indent=2))}</pre></section><section>{table}</section><section><h2>Failures</h2><ul>{failures}</ul></section></body></html>""")
    return path

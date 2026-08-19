from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
import io
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .adapters import OHLCVSource
from .distance import (
    complete_representation_distance, representation_distance_lower_bound,
)
from .episodes import build_episode
from .representation import represent
from .search import latest_eligible_cutoff
from .types import Episode


KULLAMAGI_POSITIONS_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRUG7PzohHO7MapO-qy9hc5V6A2LJPNrQml0y504lAjJhWpBh9IpOwRMMZ6MgU4z7rvPpKcC9Zscpdo/"
    "pub?gid=0&single=true&output=csv"
)


@dataclass(frozen=True)
class LabelledExample:
    source_row: int
    entry_date: pd.Timestamp
    symbol: str
    side: str
    setup_raw: str
    setup: str
    result: str
    chart_url: str


@dataclass(frozen=True)
class PreparedExample:
    record: LabelledExample
    episode: Episode


@dataclass(frozen=True)
class ExternalExampleResult:
    metrics: dict[str, object]
    neighbours: pd.DataFrame
    coverage: pd.DataFrame
    passed: bool
    failures: tuple[str, ...]


def normalize_setup(value: str) -> str:
    value = value.strip()
    aliases = {
        "EP": "episodic_pivot",
        "Episodic Pivot": "episodic_pivot",
        "EP - Sector": "episodic_pivot",
        "Para": "parabolic",
        "Parabolic Short": "parabolic",
        "Para Long": "parabolic",
        "Bounce off MA": "moving_average_reaction",
        "Bounce of MA": "moving_average_reaction",
        "Short off MA": "moving_average_reaction",
        "Speculation": "speculation",
        "Market  Speculation": "speculation",
        "Market Play": "speculation",
    }
    return aliases.get(value, value.lower().replace(" ", "_"))


def parse_kullamagi_positions(text: str) -> tuple[LabelledExample, ...]:
    records: list[LabelledExample] = []
    for source_row, row in enumerate(csv.reader(io.StringIO(text)), 1):
        if len(row) < 17 or row[0].strip() == "Entry Date":
            continue
        try:
            entry_date = pd.Timestamp(row[0].strip())
        except (TypeError, ValueError):
            continue
        symbol = row[2].strip()
        if not symbol or not symbol[0].isalpha() or not symbol.replace(
            ".", "",
        ).replace("-", "").replace("/", "").isalnum():
            continue
        setup_raw = row[4].strip()
        records.append(LabelledExample(
            source_row, entry_date, symbol, row[3].strip(), setup_raw,
            normalize_setup(setup_raw), row[15].strip(), row[5].strip(),
        ))
    return tuple(records)


def download_kullamagi_positions(url: str = KULLAMAGI_POSITIONS_URL) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def _prepare_examples(
    source: OHLCVSource,
    records: tuple[LabelledExample, ...],
    representation_version: str,
    lookback: int,
) -> tuple[list[PreparedExample], pd.DataFrame]:
    keys_by_symbol = {key.source_symbol: key for key in source.instruments()}
    prepared: list[PreparedExample] = []
    coverage: list[dict[str, object]] = []
    seen: set[tuple[str, pd.Timestamp]] = set()
    for record in records:
        status = "usable"
        cutoff: pd.Timestamp | None = None
        if record.symbol not in keys_by_symbol:
            status = "missing_symbol"
        else:
            key = keys_by_symbol[record.symbol]
            try:
                bars = source.load(key)
                # The primary analysis ends before entry day. Daily close/volume for
                # an intraday trade was not known when the position was opened.
                prior = bars[bars.timestamp < record.entry_date]
                if len(prior) < lookback:
                    status = "insufficient_pre_entry_history"
                else:
                    cutoff = pd.Timestamp(prior.timestamp.iloc[-1])
                    identity = (record.symbol, cutoff)
                    if identity in seen:
                        status = "duplicate_symbol_cutoff"
                    else:
                        seen.add(identity)
                        episode = build_episode(
                            source, key, cutoff, lookback, representation_version,
                        )
                        prepared.append(PreparedExample(record, episode))
            except Exception as exc:
                status = f"load_error:{type(exc).__name__}"
        coverage.append({
            **asdict(record), "entry_date": record.entry_date.isoformat(),
            "cutoff": cutoff.isoformat() if cutoff is not None else None,
            "status": status,
        })
    return prepared, pd.DataFrame(coverage)


def _mode_metrics(
    mode: str,
    prepared: list[PreparedExample],
    neighbour_rows: list[dict[str, object]],
    candidate_sets: dict[int, tuple[int, ...]],
    top_k: int,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    rows = [row for row in neighbour_rows if row["mode"] == mode]
    by_query: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_query.setdefault(int(row["query_index"]), []).append(row)
    by_query = {
        index: sorted(values, key=lambda value: int(value["rank"]))
        for index, values in by_query.items() if len(values) == top_k
    }
    labels = [item.record.setup for item in prepared]
    sides = [item.record.side for item in prepared]
    top1 = np.asarray([
        labels[index] == str(values[0]["candidate_setup"])
        for index, values in by_query.items()
    ], dtype=float)
    purity = np.asarray([
        np.mean([labels[index] == str(value["candidate_setup"]) for value in values])
        for index, values in by_query.items()
    ], dtype=float)
    side = np.asarray([
        sides[index] == str(values[0]["candidate_side"])
        for index, values in by_query.items()
    ], dtype=float)
    expected = []
    for index in by_query:
        candidates = candidate_sets[index]
        expected.append(np.mean([labels[j] == labels[index] for j in candidates]))
    setup_counts = Counter(labels[index] for index in by_query)
    per_setup = {}
    for setup, count in sorted(setup_counts.items()):
        selected = [
            float(labels[index] == str(values[0]["candidate_setup"]))
            for index, values in by_query.items() if labels[index] == setup
        ]
        per_setup[setup] = {"queries": count, "top1_agreement": float(np.mean(selected))}
    eligible_setups = [value["top1_agreement"] for value in per_setup.values() if value["queries"] >= 3]
    observed_top1 = float(top1.mean()) if len(top1) else 0.0
    observed_purity = float(purity.mean()) if len(purity) else 0.0
    rng = np.random.default_rng(seed)
    randomized_top1 = []
    randomized_purity = []
    query_indices = tuple(by_query)
    for _ in range(permutations):
        random_candidates = {
            index: rng.choice(candidate_sets[index], size=top_k, replace=False)
            for index in query_indices
        }
        randomized_top1.append(np.mean([
            labels[index] == labels[int(random_candidates[index][0])]
            for index in query_indices
        ]))
        randomized_purity.append(np.mean([
            np.mean([
                labels[index] == labels[int(candidate)]
                for candidate in random_candidates[index]
            ])
            for index in query_indices
        ]))
    return {
        "queries": len(by_query),
        "top_k": top_k,
        "top1_setup_agreement": observed_top1,
        "top_k_setup_purity": observed_purity,
        "top1_side_agreement": float(side.mean()) if len(side) else 0.0,
        "candidate_frequency_expected_agreement": float(np.mean(expected)) if expected else 0.0,
        "majority_setup_accuracy": max(setup_counts.values(), default=0) / max(len(by_query), 1),
        "macro_top1_agreement_minimum_3_queries": (
            float(np.mean(eligible_setups)) if eligible_setups else 0.0
        ),
        "random_candidate_top1_p_value": (
            (1 + sum(value >= observed_top1 for value in randomized_top1))
            / (permutations + 1)
        ),
        "random_candidate_top_k_p_value": (
            (1 + sum(value >= observed_purity for value in randomized_purity))
            / (permutations + 1)
        ),
        "per_setup": per_setup,
    }


def analyze_kullamagi_examples(
    source: OHLCVSource,
    csv_bytes: bytes,
    representation_version: str,
    *,
    source_url: str = KULLAMAGI_POSITIONS_URL,
    lookback: int = 252,
    top_k: int = 5,
    minimum_history_gap_bars: int = 60,
    permutations: int = 1000,
    seed: int = 20210819,
) -> ExternalExampleResult:
    if (
        lookback < 126 or top_k < 1 or permutations < 1
        or minimum_history_gap_bars < 0
    ):
        raise ValueError("invalid labelled-example analysis controls")
    started = perf_counter()
    records = parse_kullamagi_positions(csv_bytes.decode("utf-8-sig"))
    analysis_input = [
        {
            "source_row": record.source_row,
            "entry_date": record.entry_date.isoformat(),
            "symbol": record.symbol,
            "side": record.side,
            "setup": record.setup,
        }
        for record in records
    ]
    prepared, coverage = _prepare_examples(
        source, records, representation_version, lookback,
    )
    failures = []
    load_errors = coverage.status.astype(str).str.startswith("load_error").sum()
    if load_errors:
        failures.append(f"{load_errors} source rows failed to load")
    if len(prepared) < 2 * top_k:
        failures.append(f"only {len(prepared)} usable unique examples")
    representations = [represent(item.episode) for item in prepared]
    bounds: dict[tuple[int, int], tuple[float, dict[str, float], float]] = {}
    for left in range(len(prepared)):
        for right in range(left + 1, len(prepared)):
            if prepared[left].record.symbol == prepared[right].record.symbol:
                continue
            bounds[left, right] = representation_distance_lower_bound(
                representations[left], representations[right],
            )
    exact: dict[tuple[int, int], tuple[float, dict[str, float]]] = {}

    def complete(left: int, right: int, bound):
        key = (min(left, right), max(left, right))
        if key not in exact:
            total, components, _ = complete_representation_distance(
                representations[key[0]], representations[key[1]], *bound,
            )
            exact[key] = (total, components)
        return exact[key]

    neighbour_rows: list[dict[str, object]] = []
    candidate_sets_by_mode: dict[str, dict[int, tuple[int, ...]]] = {}
    causal_mode = f"causal_{minimum_history_gap_bars}_sessions"
    for mode in ("retrospective", causal_mode):
        mode_sets: dict[int, tuple[int, ...]] = {}
        for query_index, query in enumerate(prepared):
            latest = latest_eligible_cutoff(
                query.episode, minimum_history_gap_bars,
            )
            candidates = tuple(
                index for index, candidate in enumerate(prepared)
                if index != query_index
                and candidate.record.symbol != query.record.symbol
                and (
                    mode == "retrospective"
                    or candidate.episode.key.cutoff <= latest
                )
            )
            if len(candidates) < top_k:
                continue
            mode_sets[query_index] = candidates
            ordered = []
            for candidate_index in candidates:
                key = (min(query_index, candidate_index), max(query_index, candidate_index))
                bound = bounds[key]
                ordered.append((bound[0], prepared[candidate_index].episode.key.id,
                                candidate_index, bound))
            ordered.sort()
            scored: list[tuple[float, str, int, dict[str, float]]] = []
            threshold = float("inf")
            for lower, episode_id, candidate_index, bound in ordered:
                if len(scored) >= top_k and lower > threshold:
                    break
                total, components = complete(query_index, candidate_index, bound)
                scored.append((total, episode_id, candidate_index, components))
                scored.sort(key=lambda value: (value[0], value[1]))
                scored = scored[:top_k]
                if len(scored) >= top_k:
                    threshold = scored[-1][0]
            for rank, (total, _, candidate_index, components) in enumerate(scored, 1):
                candidate = prepared[candidate_index]
                neighbour_rows.append({
                    "mode": mode, "query_index": query_index,
                    "candidate_index": candidate_index, "rank": rank,
                    "query_symbol": query.record.symbol,
                    "query_entry_date": query.record.entry_date.isoformat(),
                    "query_cutoff": query.episode.key.cutoff.isoformat(),
                    "query_setup": query.record.setup,
                    "query_side": query.record.side,
                    "candidate_symbol": candidate.record.symbol,
                    "candidate_entry_date": candidate.record.entry_date.isoformat(),
                    "candidate_cutoff": candidate.episode.key.cutoff.isoformat(),
                    "candidate_setup": candidate.record.setup,
                    "candidate_side": candidate.record.side,
                    "total_distance": total,
                    **{f"distance_{name}": value for name, value in components.items()},
                })
        candidate_sets_by_mode[mode] = mode_sets
    mode_metrics = {
        mode: _mode_metrics(
            mode, prepared, neighbour_rows, candidate_sets_by_mode[mode],
            top_k, permutations, seed,
        )
        for mode in candidate_sets_by_mode
    }
    metrics = {
        "source_url": source_url,
        "source_sha256": sha256(csv_bytes).hexdigest(),
        "analysis_input_sha256": sha256(json.dumps(
            analysis_input, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
        "source_rows": len(records),
        "source_unique_symbols": len({record.symbol for record in records}),
        "usable_unique_episodes": len(prepared),
        "usable_unique_symbols": len({item.record.symbol for item in prepared}),
        "coverage_status": coverage.status.value_counts().to_dict(),
        "lookback": lookback,
        "cutoff_policy": "previous_completed_session",
        "same_symbol_candidates_excluded": True,
        "minimum_history_gap_bars": minimum_history_gap_bars,
        "outcomes_used_in_similarity": False,
        "exact_pairs_completed": len(exact),
        "modes": mode_metrics,
        "elapsed_seconds": perf_counter() - started,
    }
    return ExternalExampleResult(
        metrics, pd.DataFrame(neighbour_rows), coverage,
        not failures, tuple(failures),
    )


def write_external_example_artifacts(
    result: ExternalExampleResult,
    csv_bytes: bytes,
    directory: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "source.csv"
    metrics_path = directory / "metrics.json"
    neighbours_path = directory / "neighbours.parquet"
    coverage_path = directory / "coverage.parquet"
    report_path = directory / "report.html"
    source_path.write_bytes(csv_bytes)
    metrics_path.write_text(json.dumps({
        "passed": result.passed, "failures": result.failures, **result.metrics,
    }, indent=2, sort_keys=True, default=str))
    result.neighbours.to_parquet(neighbours_path, index=False)
    result.coverage.to_parquet(coverage_path, index=False)
    modes = result.metrics["modes"]
    assert isinstance(modes, dict)
    mode_rows = "".join(
        "<tr>"
        f"<td>{escape(str(mode))}</td><td>{values['queries']}</td>"
        f"<td>{values['top1_setup_agreement']:.1%}</td>"
        f"<td>{values['majority_setup_accuracy']:.1%}</td>"
        f"<td>{values['top_k_setup_purity']:.1%}</td>"
        f"<td>{values['candidate_frequency_expected_agreement']:.1%}</td>"
        f"<td>{values['macro_top1_agreement_minimum_3_queries']:.1%}</td>"
        f"<td>{values['random_candidate_top1_p_value']:.4f}</td>"
        f"<td>{values['random_candidate_top_k_p_value']:.4f}</td>"
        "</tr>"
        for mode, values in modes.items()
    )
    coverage_rows = "".join(
        f"<tr><td>{escape(str(status))}</td><td>{count}</td></tr>"
        for status, count in sorted(result.metrics["coverage_status"].items())
    )
    causal_mode = f"causal_{result.metrics['minimum_history_gap_bars']}_sessions"
    examples = result.neighbours[
        (result.neighbours["mode"] == causal_mode)
        & (result.neighbours["rank"] == 1)
    ].head(25)
    example_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row.query_symbol))}</td>"
        f"<td>{escape(str(row.query_entry_date)[:10])}</td>"
        f"<td>{escape(str(row.query_setup))}</td>"
        f"<td>{escape(str(row.candidate_symbol))}</td>"
        f"<td>{escape(str(row.candidate_entry_date)[:10])}</td>"
        f"<td>{escape(str(row.candidate_setup))}</td>"
        f"<td>{float(row.total_distance):.4f}</td></tr>"
        for row in examples.itertuples(index=False)
    )
    report_path.write_text(f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Kullamagi labelled-example analysis</title><style>body{{font-family:system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem;background:#f5f7f8;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}.warn{{color:#9a6700}}code{{overflow-wrap:anywhere}}</style></head><body><header><h1>Kullamagi positions: exact chart-morphology audit</h1><p>This is a separate external labelled-example benchmark, not Gate 12 and not a profitability claim. Similarity uses only OHLCV and benchmark information available before the entry date; spreadsheet outcomes are never scored.</p></header><section><h2>Coverage</h2><p>The published sheet has {result.metrics['source_rows']} parsed trade rows. Our current NASDAQ-focused source supports {result.metrics['usable_unique_episodes']} unique pre-entry 252-session episodes across {result.metrics['usable_unique_symbols']} symbols.</p><table><thead><tr><th>Status</th><th>Rows</th></tr></thead><tbody>{coverage_rows}</tbody></table><p class=\"warn\">Missing NYSE/ETF/delisted symbols and recent IPOs make this a partial, selection-biased audit.</p></section><section><h2>Does the metric recover the sheet's setup labels?</h2><table><thead><tr><th>Mode</th><th>Queries</th><th>Top-1 setup</th><th>Majority baseline</th><th>Top-{result.metrics['modes']['retrospective']['top_k']} purity</th><th>Frequency baseline</th><th>Macro top-1</th><th>Random-candidate top-1 p</th><th>Random-candidate top-k p</th></tr></thead><tbody>{mode_rows}</tbody></table><p><b>Retrospective</b> permits examples from any date and asks only whether chart morphology clusters. It is inflated by same-date sector peers and is diagnostic only. <b>Causal</b> permits candidates at least {result.metrics['minimum_history_gap_bars']} observed query sessions earlier and is the primary result. Same-symbol candidates are excluded in both. Random-candidate tests sample from each query's own eligible set, preserving its date restrictions and available class mix.</p></section><section><h2>Primary causal nearest examples</h2><table><thead><tr><th>Query</th><th>Entry</th><th>Label</th><th>Nearest earlier stock</th><th>Entry</th><th>Label</th><th>Distance</th></tr></thead><tbody>{example_rows}</tbody></table></section><section><h2>Method boundaries</h2><ul><li>This appears to be a third-party tracker of Kullamagi positions, so its setup labels are external annotations rather than assumed ground truth.</li><li>Primary cutoff is the previous completed daily session. This avoids using an entry-day close or volume that was unknown at an intraday entry.</li><li>Setup labels are normalized only through declared aliases such as EP → episodic pivot.</li><li>Exact composite distance includes price development, candles/volatility, volume/shocks, market context, relative strength, structural turns and bounded DTW.</li><li>Class imbalance is reported explicitly; raw agreement alone cannot establish pattern recovery.</li><li>The benchmark tests whether labelled examples cluster among one another. It does not yet retrieve every matching window from the full historical universe.</li><li>Current symbol coverage is survivorship- and venue-limited, and contemporary adjusted OHLCV may not exactly reproduce the chart visible in 2021.</li><li>The spreadsheet itself states that entries/exits and sizing do not represent Kullamagi’s actual performance.</li></ul></section><section><h2>Integrity</h2><p>Source: <code>{escape(str(result.metrics['source_url']))}</code></p><p>Raw snapshot SHA-256: <code>{result.metrics['source_sha256']}</code></p><p>Analysis-input SHA-256: <code>{result.metrics['analysis_input_sha256']}</code></p><p>The sheet contains live price formulas, so the raw snapshot can change while the entry-date/symbol/side/setup analysis input remains unchanged.</p><p>Exact pairs completed: {result.metrics['exact_pairs_completed']}; elapsed: {result.metrics['elapsed_seconds']:.2f} seconds.</p></section></body></html>""")
    return source_path, metrics_path, neighbours_path, coverage_path, report_path

"""All-authority pruning and safety gate for the hardened float16 bound."""

from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.episodes import build_episode
from market_analogues.exact_batch import (
    batch_representation_lower_bounds, sliding_exact_representations,
)
from market_analogues.quantized_bound import (
    PACKED_ROW_BYTES, QuantizedBoundError, quantize_bound_row,
    branch_aware_quantized_batch_lower_bounds,
    branch_aware_quantized_bound_contract,
    quantized_batch_lower_bounds, quantized_bound_contract,
)
from market_analogues.representation import represent
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, stable_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--symbols", type=int, default=64)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--minimum-retention", type=float, default=.99)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch-aware", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source = source_from_spec(config.datasets["nasdaq"])
    root = config.artifact_dir / "gate12" / "authorities" / "nasdaq" / "cases"
    authority_paths = sorted(root.glob("*.json"))
    if len(authority_paths) != 12:
        raise ValueError(f"require 12 NASDAQ authorities; found {len(authority_paths)}")
    quality = pd.read_parquet(config.artifact_dir / "quality" / "nasdaq.parquet")
    tiers = dict(zip(quality.symbol.astype(str), quality.tier.astype(str)))
    universe = [
        key for key in source.instruments()
        if tiers.get(key.source_symbol, "A") in {"A", "B"}
    ]
    universe.sort(key=lambda key: sha256(
        f"m04r-quantized-authorities:{key.source_symbol}".encode(),
    ).hexdigest())
    benchmark = source.load_benchmark()
    benchmark_fingerprint = source.benchmark_fingerprint()
    fingerprint_cache: dict[str, str] = {}
    cases = []
    started = perf_counter()
    for authority_path in authority_paths:
        authority = json.loads(authority_path.read_text())
        metadata = authority["query"]
        query = build_episode(
            source, InstrumentKey("nasdaq", str(metadata["symbol"])),
            str(metadata["cutoff"]), int(metadata["lookback"]),
            str(metadata["representation_version"]),
        )
        if source.fingerprint(query.key.instrument) != authority["source_fingerprint"]:
            raise ValueError(f"query source changed for {authority['query_episode_id']}")
        if benchmark_fingerprint != authority["benchmark_fingerprint"]:
            raise ValueError(f"benchmark changed for {authority['query_episode_id']}")
        query_representation = represent(query)
        latest = latest_eligible_cutoff(query, 60)
        selected = list(universe[:args.symbols])
        selected_names = {key.source_symbol for key in selected}
        required = {str(row["symbol"]) for row in authority["matches"]}
        selected.extend(
            key for key in universe
            if key.source_symbol in required - selected_names
        )
        for key in selected:
            if key.source_symbol not in fingerprint_cache:
                fingerprint_cache[key.source_symbol] = source.fingerprint(key)
        selected_symbols = sorted(key.source_symbol for key in selected)
        source_scope_digest = stable_hash({
            symbol: fingerprint_cache[symbol] for symbol in selected_symbols
        })
        threshold = float(authority["certificate"]["stop_threshold"])
        rows = native_pruned = quantized_pruned = overflow_rows = 0
        maximum_total_excess = maximum_component_excess = 0.0
        for key in selected:
            frame = source.load(key)
            frame = frame[frame.timestamp <= latest].reset_index(drop=True)
            batch = sliding_exact_representations(
                frame, benchmark, lookback=int(metadata["lookback"]),
                stride=args.stride, batch_size=512,
            )
            if not batch.representations:
                continue
            packed = []
            for representation in batch.representations:
                try:
                    packed.append(quantize_bound_row(representation))
                except QuantizedBoundError:
                    overflow_rows += 1
            if len(packed) != len(batch.representations):
                continue
            native = batch_representation_lower_bounds(
                query_representation, batch.representations,
            )
            quantized = (
                branch_aware_quantized_batch_lower_bounds(
                    query_representation, packed,
                )
                if args.branch_aware else
                quantized_batch_lower_bounds(query_representation, packed)
            )
            maximum_total_excess = max(
                maximum_total_excess,
                max(float(np.max(quantized.totals - native.totals)), 0.0),
            )
            for name in quantized.components:
                maximum_component_excess = max(
                    maximum_component_excess,
                    max(float(np.max(
                        quantized.components[name] - native.components[name]
                    )), 0.0),
                )
            native_pruned += int(np.sum(native.totals >= threshold))
            quantized_pruned += int(np.sum(quantized.totals >= threshold))
            rows += len(packed)
        retention = quantized_pruned / native_pruned if native_pruned else 1.0
        cases.append({
            "query_episode_id": str(authority["query_episode_id"]),
            "symbol": str(metadata["symbol"]),
            "cutoff": str(metadata["cutoff"]),
            "authority_digest": str(authority["authority_digest"]),
            "symbols": len(selected),
            "selected_symbols": selected_symbols,
            "source_scope_digest": source_scope_digest,
            "benchmark_fingerprint": benchmark_fingerprint,
            "rows": rows,
            "threshold": threshold,
            "native_pruned": native_pruned,
            "quantized_pruned": quantized_pruned,
            "pruning_retention": retention,
            "maximum_total_excess": maximum_total_excess,
            "maximum_component_excess": maximum_component_excess,
            "overflow_rows": overflow_rows,
            "passed": (
                rows > 0 and overflow_rows == 0
                and maximum_total_excess <= args.tolerance
                and maximum_component_excess <= args.tolerance
                and retention >= args.minimum_retention
            ),
        })
    total_rows = sum(row["rows"] for row in cases)
    deterministic = {
        "schema_version": (
            "m04r-quantized-bound-authority-gate-v2"
            if args.branch_aware else "m04r-quantized-bound-authority-gate-v1"
        ),
        "contract_digest": (
            branch_aware_quantized_bound_contract()["digest"]
            if args.branch_aware else quantized_bound_contract()["digest"]
        ),
        "authority_cases": cases,
        "case_count": len(cases),
        "total_rows": total_rows,
        "minimum_pruning_retention": min(row["pruning_retention"] for row in cases),
        "maximum_total_excess": max(row["maximum_total_excess"] for row in cases),
        "maximum_component_excess": max(row["maximum_component_excess"] for row in cases),
        "overflow_rows": sum(row["overflow_rows"] for row in cases),
        "required_minimum_retention": args.minimum_retention,
        "tolerance": args.tolerance,
        "packed_row_bytes": PACKED_ROW_BYTES,
        "projected_3_82m_gib": PACKED_ROW_BYTES * 3_820_000 / 1024 ** 3,
        "all_cases_passed": all(row["passed"] for row in cases),
        "real_forward_outcomes_accessed": False,
    }
    payload = {
        **deterministic,
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "result_digest": stable_hash(deterministic),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    html_path = args.output.with_suffix(".html")
    table = "".join(
        f"<tr><td>{escape(row['symbol'])}</td><td>{escape(row['cutoff'])}</td>"
        f"<td>{row['rows']}</td><td>{row['pruning_retention']:.6%}</td>"
        f"<td>{row['maximum_total_excess']:.3g}</td><td>{row['overflow_rows']}</td>"
        f"<td>{'PASS' if row['passed'] else 'FAIL'}</td></tr>" for row in cases
    )
    html_path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>M04R quantized bound authority gate</title><style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}code{{overflow-wrap:anywhere}}</style></head><body><h1>M04R quantized bound authority gate: {'PASS' if deterministic['all_cases_passed'] else 'FAIL'}</h1><p>Contract <code>{deterministic['contract_digest']}</code>; result <code>{payload['result_digest']}</code>.</p><pre>{escape(json.dumps({key: value for key, value in payload.items() if key != 'authority_cases'}, indent=2, sort_keys=True))}</pre><table><thead><tr><th>Symbol</th><th>Cutoff</th><th>Rows</th><th>Retention</th><th>Max excess</th><th>Overflow</th><th>Status</th></tr></thead><tbody>{table}</tbody></table></body></html>""")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if deterministic["all_cases_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

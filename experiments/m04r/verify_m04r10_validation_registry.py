"""Independent M04R-10 registry/provenance/selection verifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.m04r_validation_registry import (
    _canonical_records, _selection_universe, build_contamination_ledger,
    default_search_contract, validate_m04r_validation_registry,
    validate_store_temporal_coverage,
)
from market_analogues.types import stable_hash


SCHEMA_VERSION = "m04r10-untouched-authority-registry-verification-v1"


def _coverage(cases: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        column: {
            str(key): int(value) for key, value in
            cases[column].value_counts().sort_index().items()
        }
        for column in (
            "quality_tier", "liquidity_stratum", "cutoff_role",
            "benchmark_regime", "era", "morphology_stratum", "data_context",
            "universe_size_band", "volatility_band", "drawdown_band",
        )
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    source = source_from_spec(config.datasets[args.dataset])
    registry_root = Path(args.registry_root)
    registry = json.loads((registry_root / "query-registry.json").read_text())
    quality_path = config.artifact_dir / "quality" / f"{args.dataset}.parquet"
    liquidity_path = config.artifact_dir / "oracles" / args.dataset / "liquidity-strata.parquet"
    quality = pd.read_parquet(quality_path)
    liquidity = pd.read_parquet(liquidity_path)
    failures = list(validate_m04r_validation_registry(source, registry_root))

    rebuilt_ledger = build_contamination_ledger(
        config.artifact_dir, Path(args.design_exclusions).resolve(),
    )
    ledger_equal = rebuilt_ledger == registry.get("contamination_ledger")
    if not ledger_equal:
        failures.append("independently rebuilt contamination ledger differs")
    rebuilt_universe, market_latest = _selection_universe(
        quality, liquidity,
        excluded=set(rebuilt_ledger["excluded_symbols"]),
        minimum_rows=int(registry["minimum_rows"]),
        maximum_staleness_days=int(registry["maximum_staleness_days"]),
    )
    stored_universe = pd.read_parquet(registry_root / "selection-universe.parquet")
    universe_equal = (
        stable_hash(_canonical_records(rebuilt_universe))
        == stable_hash(_canonical_records(stored_universe))
        == registry.get("universe_digest")
    )
    if not universe_equal:
        failures.append("independently rebuilt selection universe differs")
    current_search_contract = default_search_contract(config.artifact_dir)
    search_contract_equal = current_search_contract == registry.get("search_contract")
    if not search_contract_equal:
        failures.append("frozen search/store/distance contract differs")
    cases = pd.read_parquet(registry_root / "query-registry.parquet")
    cases_equal = _canonical_records(cases) == registry.get("cases_data")
    if not cases_equal:
        failures.append("Parquet and JSON case registries differ")
    coverage = _coverage(cases)
    coverage_equal = coverage == registry.get("coverage")
    if not coverage_equal:
        failures.append("independently recomputed coverage differs")
    store_coverage, store_coverage_failures = validate_store_temporal_coverage(
        source, cases, registry.get("search_contract") or {},
    )
    failures.extend(store_coverage_failures)
    gates = {
        "runtime_registry_validator_passed": not validate_m04r_validation_registry(
            source, registry_root,
        ),
        "contamination_ledger_reconstructed": ledger_equal,
        "selection_universe_reconstructed": universe_equal,
        "case_encodings_equal": cases_equal,
        "coverage_recomputed": coverage_equal,
        "search_store_distance_contract_reconstructed": search_contract_equal,
        "packed_store_covers_every_latest_eligible_cutoff": not store_coverage_failures,
        "sixty_cases_and_thirty_symbols": len(cases) == 60 and cases.symbol.nunique() == 30,
        "real_forward_outcomes_excluded": registry.get("real_forward_outcomes_accessed") is False,
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "registry_digest": registry.get("registry_digest"),
        "contamination_ledger_digest": rebuilt_ledger["ledger_digest"],
        "universe_digest": registry.get("universe_digest"),
        "transcript_digest": registry.get("transcript_digest"),
        "market_latest_timestamp_at_verification": market_latest.isoformat(),
        "cases": len(cases), "symbols": int(cases.symbol.nunique()),
        "coverage": coverage, "store_temporal_coverage": store_coverage,
        "gates": gates,
        "files_opened_by_verifier": [
            str(quality_path.resolve()), str(liquidity_path.resolve()),
            str((registry_root / "query-registry.json").resolve()),
            str((registry_root / "query-registry.parquet").resolve()),
            str((registry_root / "selection-universe.parquet").resolve()),
            str((registry_root / "selection-transcript.parquet").resolve()),
            "sealed contamination sources listed in the registry",
            "selected OHLCV and benchmark prefixes at or before frozen cutoffs",
        ],
        "real_forward_outcomes_accessed": False,
        "failures": list(dict.fromkeys(failures)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["passed"] = all(gates.values()) and not payload["failures"]
    payload["result_digest"] = stable_hash({
        key: value for key, value in payload.items()
        if key not in {"created_at", "result_digest"}
    })
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="nasdaq")
    parser.add_argument("--registry-root", required=True)
    parser.add_argument(
        "--design-exclusions", default="config/m04r10-design-exclusions.yaml",
    )
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = verify(args)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "m04r10-registry-verification.json"
    html_path = root / "m04r10-registry-verification.html"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = "PASS" if result["passed"] else "FAIL"
    css = "pass" if result["passed"] else "fail"
    html_path.write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>M04R-10 independent registry verification</title><style>"
        "body{font-family:system-ui;max-width:1200px;margin:2rem auto}"
        ".pass{color:#075}.fail{color:#a20}pre{white-space:pre-wrap}"
        "</style></head><body><h1>M04R-10 independent verification: "
        f"<span class=\"{css}\">{status}</span></h1><p>The verifier independently "
        "rebuilds contamination and eligible-universe snapshots, replays every "
        "selection prefix, re-hashes causal stock/benchmark inputs and binds the "
        "distance, proposal, store and performance contracts.</p><pre>"
        f"{escape(json.dumps(result, indent=2, sort_keys=True))}</pre></body></html>"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(html_path)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

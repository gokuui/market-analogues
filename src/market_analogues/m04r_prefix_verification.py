from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .adapters import OHLCVSource
from .authority import load_authority_artifact
from .causal_prefix import causal_prefix_digest, prefix_generation_digest
from .m04_candidate_recall import M04CandidateRecallSpec
from .types import InstrumentKey, stable_hash


M04R_PREFIX_VERIFIER_SCHEMA = "m04r-causal-prefix-verification-v1"


class M04RPrefixVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class M04RPrefixVerificationResult:
    passed: bool
    metrics: dict[str, Any]
    failures: tuple[str, ...]
    cases: tuple[dict[str, Any], ...]
    result_digest: str


def _future_mutation(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    changed = frame.copy()
    future = changed["timestamp"] > cutoff
    if future.any():
        changed.loc[future, "close"] = changed.loc[future, "close"] * 1.37
        changed.loc[future, "volume"] = changed.loc[future, "volume"] + 17
        return changed
    last = changed.iloc[-1].copy()
    last["timestamp"] = pd.Timestamp(last["timestamp"]) + pd.offsets.BDay()
    last["close"] = float(last["close"]) * 1.01
    return pd.concat([changed, last.to_frame().T], ignore_index=True)


def _historical_mutation(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    changed = frame.copy()
    positions = changed.index[changed["timestamp"] <= cutoff]
    if not len(positions):
        raise M04RPrefixVerificationError("cannot mutate an empty causal prefix")
    position = positions[-1]
    changed.loc[position, "close"] = float(changed.loc[position, "close"]) * 1.0001
    return changed


def verify_m04r_causal_prefixes(
    spec: M04CandidateRecallSpec,
    source: OHLCVSource,
    artifact_dir: Path,
    *,
    dataset_id: str = "nasdaq",
) -> M04RPrefixVerificationResult:
    if source.spec.dataset_id != dataset_id:
        raise M04RPrefixVerificationError("source dataset differs")
    registry_path = artifact_dir / "gate12" / dataset_id / "query-registry.yaml"
    registry = yaml.safe_load(registry_path.read_text()) or {}
    if registry.get("registry_digest") != spec.payload["registry_digests"][dataset_id]:
        raise M04RPrefixVerificationError("frozen registry digest differs")
    registry_rows = {
        str(row["episode_id"]): row for row in registry.get("cases_data", [])
    }
    episode_ids = []
    for episode_id in spec.payload["holdout_episode_ids"]:
        paths = list((artifact_dir / "gate12" / "authorities" / dataset_id / "cases").glob(
            f"{episode_id}.json"
        ))
        if paths:
            episode_ids.append(episode_id)
    if len(episode_ids) != 12:
        raise M04RPrefixVerificationError(
            f"expected 12 {dataset_id} authorities, found {len(episode_ids)}"
        )
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    benchmark = source.load_benchmark()
    physical_benchmark = source.benchmark_fingerprint()
    for episode_id in episode_ids:
        authority_path = (
            artifact_dir / "gate12" / "authorities" / dataset_id / "cases"
            / f"{episode_id}.json"
        )
        authority = load_authority_artifact(authority_path)
        payload = json.loads(authority_path.read_text())
        if authority.authority_digest != spec.payload["authority_digests"][episode_id]:
            raise M04RPrefixVerificationError(f"authority digest differs for {episode_id}")
        query = payload["query"]
        registry_row = registry_rows.get(episode_id)
        if registry_row is None:
            raise M04RPrefixVerificationError(f"registry omits {episode_id}")
        for name in ("dataset_id", "symbol", "cutoff", "lookback", "representation_version"):
            if str(registry_row[name]) != str(query[name]):
                raise M04RPrefixVerificationError(
                    f"registry and authority differ for {episode_id}/{name}"
                )
        instrument = InstrumentKey(dataset_id, str(query["symbol"]))
        cutoff = pd.Timestamp(query["cutoff"])
        bars = source.load(instrument)
        physical_source = source.fingerprint(instrument)
        stock = source.causal_prefix_fingerprint(instrument, cutoff)
        stock_repeat = source.causal_prefix_fingerprint(instrument, cutoff)
        stock_future = causal_prefix_digest(_future_mutation(bars, cutoff), cutoff)
        stock_revision = causal_prefix_digest(_historical_mutation(bars, cutoff), cutoff)
        benchmark_prefix = source.benchmark_causal_prefix_fingerprint(cutoff)
        benchmark_repeat = source.benchmark_causal_prefix_fingerprint(cutoff)
        benchmark_future = (
            causal_prefix_digest(_future_mutation(benchmark, cutoff), cutoff)
            if benchmark is not None else None
        )
        benchmark_revision = (
            causal_prefix_digest(_historical_mutation(benchmark, cutoff), cutoff)
            if benchmark is not None else None
        )
        checks = {
            "physical_source_matches_frozen": physical_source == registry_row["source_fingerprint"],
            "physical_benchmark_matches_frozen": physical_benchmark == registry_row["benchmark_fingerprint"],
            "stock_repeat_matches": stock_repeat.digest == stock.digest,
            "stock_future_append_or_mutation_ignored": stock_future.digest == stock.digest,
            "stock_historical_revision_detected": stock_revision.digest != stock.digest,
            "benchmark_repeat_matches": (
                benchmark_repeat is not None and benchmark_prefix is not None
                and benchmark_repeat.digest == benchmark_prefix.digest
            ),
            "benchmark_future_append_or_mutation_ignored": (
                benchmark_future is not None and benchmark_prefix is not None
                and benchmark_future.digest == benchmark_prefix.digest
            ),
            "benchmark_historical_revision_detected": (
                benchmark_revision is not None and benchmark_prefix is not None
                and benchmark_revision.digest != benchmark_prefix.digest
            ),
        }
        for name, passed in checks.items():
            if not passed:
                failures.append(f"{episode_id} failed {name}")
        generation_digest = prefix_generation_digest(
            dataset_id=dataset_id,
            query_cutoff=cutoff,
            representation_version=str(query["representation_version"]),
            stock_prefixes={instrument.source_symbol: stock},
            benchmark_prefix=benchmark_prefix,
        )
        rows.append({
            "query_episode_id": episode_id,
            "symbol": instrument.source_symbol,
            "cutoff": cutoff.isoformat(),
            "physical_source_fingerprint": physical_source,
            "physical_benchmark_fingerprint": physical_benchmark,
            "stock_prefix": asdict(stock),
            "benchmark_prefix": asdict(benchmark_prefix) if benchmark_prefix else None,
            "generation_digest": generation_digest,
            "checks": checks,
        })
    metrics = {
        "schema_version": M04R_PREFIX_VERIFIER_SCHEMA,
        "dataset_id": dataset_id,
        "m04_contract_digest": spec.digest,
        "authorities_verified": len(rows),
        "checks_executed": sum(len(row["checks"]) for row in rows),
        "failed_checks": len(failures),
        "false_append_invalidations": sum(
            not row["checks"][name] for row in rows for name in (
                "stock_future_append_or_mutation_ignored",
                "benchmark_future_append_or_mutation_ignored",
            )
        ),
        "accepted_historical_revisions": sum(
            not row["checks"][name] for row in rows for name in (
                "stock_historical_revision_detected",
                "benchmark_historical_revision_detected",
            )
        ),
        "real_forward_outcomes_accessed": False,
    }
    deterministic = {
        "schema_version": M04R_PREFIX_VERIFIER_SCHEMA,
        "metrics": metrics,
        "failures": sorted(failures),
        "cases": rows,
    }
    return M04RPrefixVerificationResult(
        not failures, metrics, tuple(sorted(failures)), tuple(rows),
        stable_hash(deterministic),
    )


def write_m04r_prefix_verification(
    result: M04RPrefixVerificationResult,
    machine_path: Path,
    html_path: Path,
) -> tuple[Path, Path]:
    payload = {
        "schema_version": M04R_PREFIX_VERIFIER_SCHEMA,
        "passed": result.passed,
        "metrics": result.metrics,
        "failures": list(result.failures),
        "cases": list(result.cases),
        "result_digest": result.result_digest,
    }
    machine_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = machine_path.with_suffix(machine_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(machine_path)
    rows = "".join(
        f"<tr><td>{escape(str(row['symbol']))}</td><td>{escape(str(row['cutoff']))}</td>"
        f"<td>{row['stock_prefix']['rows']}</td><td><code>{row['stock_prefix']['digest']}</code></td>"
        f"<td><code>{row['benchmark_prefix']['digest'] if row['benchmark_prefix'] else 'none'}</code></td></tr>"
        for row in result.cases
    )
    failure_items = "".join(f"<li>{escape(value)}</li>" for value in result.failures) or "<li>None</li>"
    status = "PASS" if result.passed else "FAIL"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_temporary = html_path.with_suffix(html_path.suffix + ".tmp")
    html_temporary.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M04R causal-prefix verification</title><style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;background:#f5f7f8;color:#17202a}}header,section{{background:white;border:1px solid #ddd;border-radius:10px;padding:1.2rem;margin:1rem}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}code{{font-size:.78em;overflow-wrap:anywhere}}.pass{{color:#117864}}.fail{{color:#b03a2e}}pre{{white-space:pre-wrap}}</style></head><body><header><h1>M04R causal-prefix provenance: <span class="{status.lower()}">{status}</span></h1><p>Physical file hashes remain in the machine artifact for byte-level audit. Experiment identity uses canonical, ordered OHLCV values only through each query cutoff, so future appends cannot invalidate historical evidence.</p><p>Result digest: <code>{result.result_digest}</code></p></header><section><h2>Metrics</h2><pre>{escape(json.dumps(result.metrics, indent=2, sort_keys=True))}</pre></section><section><h2>Frozen NASDAQ authorities</h2><table><thead><tr><th>Symbol</th><th>Cutoff</th><th>Stock rows</th><th>Stock prefix digest</th><th>Benchmark prefix digest</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>Failures</h2><ul>{failure_items}</ul></section></body></html>""")
    html_temporary.replace(html_path)
    return machine_path, html_path

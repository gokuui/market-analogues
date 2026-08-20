from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .adapters import OHLCVSource, file_fingerprint
from .external_examples import ExternalExampleResult, LabelledExample
from .types import InstrumentKey


DownloadFunction = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class YahooFetchResult:
    source: "YahooParquetSource"
    manifest: dict[str, object]
    manifest_path: Path
    coverage: pd.DataFrame


def yahoo_symbol(symbol: str) -> str:
    """Translate the common class-share spelling used by Yahoo Finance."""
    return symbol.strip().upper().replace(".", "-")


def _canonical_yahoo_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    output = frame.copy()
    if isinstance(output.columns, pd.MultiIndex):
        output.columns = output.columns.get_level_values(-1)
    output = output.rename(columns={
        "Date": "timestamp", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    if "timestamp" not in output.columns:
        output = output.reset_index()
        output = output.rename(columns={output.columns[0]: "timestamp"})
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if any(column not in output.columns for column in required):
        return pd.DataFrame()
    output = output[required].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce", utc=True)
    output["timestamp"] = output["timestamp"].dt.tz_localize(None).dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["timestamp", "open", "high", "low", "close"])
    output["volume"] = output["volume"].fillna(0.0)
    output = output.replace([np.inf, -np.inf], np.nan).dropna()
    output = output.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    output = output.reset_index(drop=True)
    output.attrs.update(symbol=symbol, dataset_id="yfinance", interval="1d")
    return output


def _extract_downloaded(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    for level in range(frame.columns.nlevels):
        values = frame.columns.get_level_values(level).astype(str)
        if ticker in values:
            return frame.xs(ticker, axis=1, level=level, drop_level=True)
    return pd.DataFrame()


class YahooParquetSource(OHLCVSource):
    def __init__(self, root: Path, symbol_files: dict[str, str], benchmark_file: str):
        self.root = root
        self._files = {
            symbol: root / relative for symbol, relative in symbol_files.items()
        }
        self._benchmark = root / benchmark_file

    def instruments(self) -> list[InstrumentKey]:
        return [InstrumentKey("yfinance", symbol) for symbol in sorted(self._files)]

    def load(self, key: InstrumentKey) -> pd.DataFrame:
        if key.dataset_id != "yfinance" or key.source_symbol not in self._files:
            raise KeyError(str(key))
        return self._load_symbol(key.source_symbol).copy()

    @lru_cache(maxsize=256)
    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        frame = pd.read_parquet(self._files[symbol])
        frame.attrs.update(
            symbol=symbol, dataset_id="yfinance", interval="1d",
        )
        return frame

    def fingerprint(self, key: InstrumentKey) -> str:
        return file_fingerprint(self._files[key.source_symbol])

    def load_benchmark(self) -> pd.DataFrame | None:
        frame = self._load_benchmark()
        return frame.copy() if frame is not None else None

    @lru_cache(maxsize=1)
    def _load_benchmark(self) -> pd.DataFrame | None:
        if not self._benchmark.exists():
            return None
        frame = pd.read_parquet(self._benchmark)
        frame.attrs.update(symbol="SPY", dataset_id="yfinance", interval="1d")
        return frame

    def benchmark_fingerprint(self) -> str | None:
        return file_fingerprint(self._benchmark) if self._benchmark.exists() else None


def fetch_yahoo_examples(
    records: Iterable[LabelledExample],
    root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch_size: int = 25,
    downloader: DownloadFunction | None = None,
    reuse_cache: bool = True,
) -> YahooFetchResult:
    if batch_size < 1 or pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("invalid Yahoo download controls")
    if downloader is None:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - exercised by CLI environment
            raise RuntimeError(
                "install market-analogues[external-data] for Yahoo downloads",
            ) from exc
        downloader = yf.download
        yfinance_version = yf.__version__
    else:
        yfinance_version = "injected-test-downloader"

    root.mkdir(parents=True, exist_ok=True)
    bars_root = root / "bars"
    bars_root.mkdir(exist_ok=True)
    existing_manifest_path = root / "manifest.json"
    existing_manifest = {}
    if reuse_cache and existing_manifest_path.exists():
        try:
            existing_manifest = json.loads(existing_manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing_manifest = {}
    cache_compatible = (
        existing_manifest.get("start") == pd.Timestamp(start).isoformat()
        and existing_manifest.get("end_exclusive") == pd.Timestamp(end).isoformat()
        and existing_manifest.get("auto_adjust") is True
        and existing_manifest.get("repair") is True
    )
    unique_symbols = sorted({record.symbol for record in records})
    queries = {symbol: yahoo_symbol(symbol) for symbol in unique_symbols}
    requested = sorted(set(queries.values()) | {"SPY"})
    cached: dict[str, pd.DataFrame] = {}
    for query in requested:
        path = bars_root / f"{query}.parquet"
        if reuse_cache and cache_compatible and path.exists():
            frame = pd.read_parquet(path)
            if not frame.empty:
                cached[query] = frame
    reused_queries = len(cached)

    missing = [query for query in requested if query not in cached]
    errors: dict[str, str] = {}
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset:offset + batch_size]
        try:
            downloaded = downloader(
                batch, start=pd.Timestamp(start).date().isoformat(),
                end=pd.Timestamp(end).date().isoformat(), interval="1d",
                group_by="ticker", auto_adjust=True, actions=False,
                repair=True, keepna=False, progress=False, threads=True,
                timeout=30, multi_level_index=True,
            )
        except Exception as exc:  # preserve failure and retry each ticker below
            downloaded = pd.DataFrame()
            for query in batch:
                errors[query] = f"batch:{type(exc).__name__}:{exc}"
        for query in batch:
            frame = _canonical_yahoo_frame(_extract_downloaded(downloaded, query), query)
            if not frame.empty:
                cached[query] = frame
                frame.to_parquet(bars_root / f"{query}.parquet", index=False)

    # Batch downloads can omit one ticker without raising. Retry omissions serially.
    for query in [value for value in missing if value not in cached]:
        try:
            downloaded = downloader(
                query, start=pd.Timestamp(start).date().isoformat(),
                end=pd.Timestamp(end).date().isoformat(), interval="1d",
                group_by="ticker", auto_adjust=True, actions=False,
                repair=True, keepna=False, progress=False, threads=False,
                timeout=30, multi_level_index=True,
            )
            frame = _canonical_yahoo_frame(_extract_downloaded(downloaded, query), query)
            if frame.empty:
                errors[query] = "no_rows"
                continue
            cached[query] = frame
            frame.to_parquet(bars_root / f"{query}.parquet", index=False)
            errors.pop(query, None)
        except Exception as exc:
            errors[query] = f"retry:{type(exc).__name__}:{exc}"
    if "SPY" not in cached:
        raise RuntimeError("Yahoo SPY benchmark download is unavailable")

    symbol_files: dict[str, str] = {}
    coverage_rows = []
    for symbol in unique_symbols:
        query = queries[symbol]
        frame = cached.get(query)
        status = "downloaded" if frame is not None and not frame.empty else "unavailable"
        if status == "downloaded":
            symbol_files[symbol] = str(Path("bars") / f"{query}.parquet")
        coverage_rows.append({
            "symbol": symbol, "yahoo_symbol": query, "status": status,
            "rows": 0 if frame is None else len(frame),
            "first_date": None if frame is None else frame.timestamp.min(),
            "last_date": None if frame is None else frame.timestamp.max(),
            "error": errors.get(query, ""),
        })
    benchmark_file = str(Path("bars") / "SPY.parquet")
    manifest = {
        "schema_version": "yfinance-labelled-examples-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "Yahoo Finance via yfinance",
        "yfinance_version": yfinance_version,
        "start": pd.Timestamp(start).isoformat(),
        "end_exclusive": pd.Timestamp(end).isoformat(),
        "auto_adjust": True,
        "repair": True,
        "cached_queries_reused": reused_queries,
        "requested_symbols": len(unique_symbols),
        "available_symbols": len(symbol_files),
        "unavailable_symbols": len(unique_symbols) - len(symbol_files),
        "symbol_files": symbol_files,
        "benchmark_file": benchmark_file,
        "errors": errors,
    }
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    manifest["manifest_content_sha256"] = sha256(manifest_payload.encode()).hexdigest()
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    source = YahooParquetSource(root, symbol_files, benchmark_file)
    return YahooFetchResult(source, manifest, manifest_path, pd.DataFrame(coverage_rows))


def write_external_example_workbook(
    result: ExternalExampleResult,
    fetch: YahooFetchResult,
    path: Path,
    *,
    target_purity: float = 0.75,
) -> Path:
    if not 0 < target_purity <= 1:
        raise ValueError("target purity must be in (0, 1]")
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised by CLI environment
        raise RuntimeError(
            "install market-analogues[external-data] for Excel output",
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    causal_mode = f"causal_{result.metrics['minimum_history_gap_bars']}_sessions"
    modes = result.metrics["modes"]
    causal = modes[causal_mode]
    summary = pd.DataFrame([
        ("Analysis completed", result.passed),
        ("Parsed tracker rows", result.metrics["source_rows"]),
        ("Yahoo symbols requested", fetch.manifest["requested_symbols"]),
        ("Yahoo symbols available", fetch.manifest["available_symbols"]),
        ("Usable unique episodes", result.metrics["usable_unique_episodes"]),
        ("Usable unique symbols", result.metrics["usable_unique_symbols"]),
        ("Primary causal queries", causal["queries"]),
        ("Causal top-1 setup agreement", causal["top1_setup_agreement"]),
        ("Causal top-k setup purity", causal["top_k_setup_purity"]),
        ("Random-candidate expectation", causal["candidate_frequency_expected_agreement"]),
        ("Random-candidate top-1 p-value", causal["random_candidate_top1_p_value"]),
        ("Random-candidate top-k p-value", causal["random_candidate_top_k_p_value"]),
        ("Predeclared purity target", target_purity),
        ("Top-k target reached", causal["top_k_setup_purity"] >= target_purity),
        ("Top-1 target reached", causal["top1_setup_agreement"] >= target_purity),
        ("Outcomes used in similarity", result.metrics["outcomes_used_in_similarity"]),
        ("Source URL", result.metrics["source_url"]),
        ("Analysis-input SHA-256", result.metrics["analysis_input_sha256"]),
    ], columns=["Metric", "Value"])

    setup_rows = []
    for mode, values in modes.items():
        for setup, setup_values in values["per_setup"].items():
            setup_rows.append({
                "mode": mode, "setup": setup, **setup_values,
                "target_purity": target_purity,
                "top_k_target_reached": setup_values["top_k_purity"] >= target_purity,
            })
    setup_frame = pd.DataFrame(setup_rows)
    neighbours = result.neighbours.copy()
    causal_neighbours = neighbours[neighbours["mode"] == causal_mode].copy()
    top1 = causal_neighbours[causal_neighbours["rank"] == 1].copy()
    query_summary_rows = []
    for (_, query_symbol, query_date), values in causal_neighbours.groupby(
        ["query_index", "query_symbol", "query_entry_date"], sort=False,
    ):
        values = values.sort_values("rank")
        query_summary_rows.append({
            "query_symbol": query_symbol,
            "query_entry_date": query_date,
            "query_setup": values.query_setup.iloc[0],
            "query_side": values.query_side.iloc[0],
            "same_setup_in_top_k": int(values.same_setup.sum()),
            "top_k": len(values),
            "top_k_purity": float(values.same_setup.mean()),
            "target_reached": float(values.same_setup.mean()) >= target_purity,
            "nearest_symbols": ", ".join(values.candidate_symbol.astype(str)),
            "nearest_setups": ", ".join(values.candidate_setup.astype(str)),
            "nearest_distances": ", ".join(f"{value:.4f}" for value in values.total_distance),
            "query_chart_url": values.query_chart_url.iloc[0],
        })
    query_summary = pd.DataFrame(query_summary_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        query_summary.to_excel(writer, sheet_name="Query Summary", index=False)
        top1.to_excel(writer, sheet_name="Top1 Matches", index=False)
        causal_neighbours.to_excel(writer, sheet_name="Causal TopK", index=False)
        setup_frame.to_excel(writer, sheet_name="Setup Performance", index=False)
        result.coverage.to_excel(writer, sheet_name="Trade Coverage", index=False)
        fetch.coverage.to_excel(writer, sheet_name="Yahoo Coverage", index=False)
        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [str(cell.value) for cell in column_cells[:200] if cell.value is not None]
                width = min(max([len(value) for value in values] + [10]) + 2, 60)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
        for worksheet_name in ("Query Summary", "Top1 Matches", "Causal TopK", "Trade Coverage"):
            worksheet = workbook[worksheet_name]
            headers = {cell.value: cell.column for cell in worksheet[1]}
            for header in ("query_chart_url", "candidate_chart_url", "chart_url"):
                column = headers.get(header)
                if column is None:
                    continue
                for row in range(2, worksheet.max_row + 1):
                    cell = worksheet.cell(row, column)
                    if isinstance(cell.value, str) and cell.value.startswith("http"):
                        cell.hyperlink = cell.value
                        cell.style = "Hyperlink"
    return path

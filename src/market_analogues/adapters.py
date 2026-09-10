from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Iterator

import pandas as pd

from .config import DatasetSpec
from .causal_prefix import CausalPrefixDigest, causal_prefix_digest
from .types import InstrumentKey

CANONICAL_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class SourceError(ValueError):
    pass


def file_fingerprint(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise SourceError(f"unsupported format: {fmt}")


def canonicalize(frame: pd.DataFrame, spec: DatasetSpec, *, symbol: str | None = None) -> pd.DataFrame:
    df = frame.rename(columns=spec.column_map).copy()
    if spec.timestamp_column != "timestamp" and spec.timestamp_column in df.columns:
        df = df.rename(columns={spec.timestamp_column: "timestamp"})
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise SourceError(f"missing required columns: {missing}")
    cols = CANONICAL_COLUMNS + [c for c in ("source", "confidence") if c in df.columns]
    df = df[cols]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    source_timestamp_reordered = not df["timestamp"].is_monotonic_increasing
    source_duplicate_timestamps = bool(df["timestamp"].duplicated().any())
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if symbol is not None:
        df.attrs["symbol"] = symbol
    df.attrs["dataset_id"] = spec.dataset_id
    df.attrs["interval"] = spec.interval
    df.attrs["source_timestamp_reordered"] = source_timestamp_reordered
    df.attrs["source_duplicate_timestamps"] = source_duplicate_timestamps
    return df


class OHLCVSource(ABC):
    @abstractmethod
    def instruments(self) -> list[InstrumentKey]: ...

    @abstractmethod
    def load(self, key: InstrumentKey) -> pd.DataFrame: ...

    def load_borrowed(self, key: InstrumentKey) -> pd.DataFrame:
        """Return a frame that a caller promises to treat as read-only."""
        return self.load(key)

    @abstractmethod
    def fingerprint(self, key: InstrumentKey) -> str: ...

    def load_benchmark(self) -> pd.DataFrame | None:
        return None

    def benchmark_fingerprint(self) -> str | None:
        benchmark = self.load_benchmark()
        if benchmark is None:
            return None
        digest = sha256()
        digest.update("\0".join(str(column) for column in benchmark.columns).encode())
        digest.update(pd.util.hash_pandas_object(benchmark, index=True).values.tobytes())
        return digest.hexdigest()

    def causal_prefix_fingerprint(
        self, key: InstrumentKey, cutoff: pd.Timestamp | str,
    ) -> CausalPrefixDigest:
        return causal_prefix_digest(self.load(key), cutoff)

    def benchmark_causal_prefix_fingerprint(
        self, cutoff: pd.Timestamp | str,
    ) -> CausalPrefixDigest | None:
        benchmark = self.load_benchmark()
        return causal_prefix_digest(benchmark, cutoff) if benchmark is not None else None


class CachedOHLCVSource(OHLCVSource):
    """Explicit reusable frame cache for multi-query retrieval batches."""

    def __init__(self, source: OHLCVSource, *, max_entries: int | None = None):
        if max_entries is not None and (
            type(max_entries) is not int or max_entries < 1
        ):
            raise SourceError("cache maximum entries must be positive or None")
        self.source = source
        self.max_entries = max_entries
        self._frames: OrderedDict[InstrumentKey, pd.DataFrame] = OrderedDict()
        self._benchmark: pd.DataFrame | None = None
        self._benchmark_loaded = False
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    def instruments(self) -> list[InstrumentKey]:
        return self.source.instruments()

    def _cached_frame(self, key: InstrumentKey) -> pd.DataFrame:
        with self._lock:
            cached = self._frames.get(key)
            if cached is not None:
                self._frames.move_to_end(key)
                self.hits += 1
                return cached.copy()
        loaded = self.source.load(key)
        with self._lock:
            cached = self._frames.get(key)
            if cached is None:
                self._frames[key] = loaded
                self.misses += 1
                if self.max_entries is not None:
                    while len(self._frames) > self.max_entries:
                        self._frames.popitem(last=False)
                cached = loaded
            else:
                self._frames.move_to_end(key)
                self.hits += 1
            return cached

    def load(self, key: InstrumentKey) -> pd.DataFrame:
        return self._cached_frame(key).copy()

    def load_borrowed(self, key: InstrumentKey) -> pd.DataFrame:
        return self._cached_frame(key)

    def preload(
        self, keys: list[InstrumentKey] | tuple[InstrumentKey, ...],
        *, workers: int = 1,
    ) -> dict[str, int | None]:
        if type(workers) is not int or workers < 1:
            raise SourceError("cache preload workers must be positive")
        ordered = tuple(dict.fromkeys(keys))
        if workers == 1:
            for key in ordered:
                self._cached_frame(key)
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="ohlcv-preload",
            ) as executor:
                tuple(executor.map(self._cached_frame, ordered))
        return self.cache_state()

    def fingerprint(self, key: InstrumentKey) -> str:
        return self.source.fingerprint(key)

    def load_benchmark(self) -> pd.DataFrame | None:
        with self._lock:
            if self._benchmark_loaded:
                return self._benchmark.copy() if self._benchmark is not None else None
        loaded = self.source.load_benchmark()
        with self._lock:
            if not self._benchmark_loaded:
                self._benchmark = loaded
                self._benchmark_loaded = True
            return self._benchmark.copy() if self._benchmark is not None else None

    def benchmark_fingerprint(self) -> str | None:
        return self.source.benchmark_fingerprint()

    def cache_state(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "entries": len(self._frames), "max_entries": self.max_entries,
                "hits": self.hits, "misses": self.misses,
            }


class DirectorySource(OHLCVSource):
    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        if not spec.path.is_dir():
            raise SourceError(f"dataset directory does not exist: {spec.path}")
        self._files = {
            p.stem: p for p in sorted(spec.path.glob(spec.file_glob))
            if not spec.benchmark or p.resolve() != spec.benchmark.path.resolve()
        }

    def instruments(self) -> list[InstrumentKey]:
        return [InstrumentKey(self.spec.dataset_id, s) for s in sorted(self._files)]

    def load(self, key: InstrumentKey) -> pd.DataFrame:
        if key.dataset_id != self.spec.dataset_id or key.source_symbol not in self._files:
            raise KeyError(str(key))
        return self._load_symbol(key.source_symbol).copy()

    @lru_cache(maxsize=32)
    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        p = self._files[symbol]
        return canonicalize(_read(p, self.spec.format), self.spec, symbol=symbol)

    def fingerprint(self, key: InstrumentKey) -> str:
        return file_fingerprint(self._files[key.source_symbol])

    def load_benchmark(self) -> pd.DataFrame | None:
        cached = self._load_benchmark()
        return cached.copy() if cached is not None else None

    def benchmark_fingerprint(self) -> str | None:
        if not self.spec.benchmark:
            return None
        return file_fingerprint(self.spec.benchmark.path)

    @lru_cache(maxsize=1)
    def _load_benchmark(self) -> pd.DataFrame | None:
        if not self.spec.benchmark:
            return None
        b = self.spec.benchmark
        fmt = b.format or b.path.suffix.lstrip(".").lower()
        temp_spec = DatasetSpec(
            dataset_id=self.spec.dataset_id, adapter="directory", path=b.path.parent,
            format=fmt, timestamp_column=b.timestamp_column or self.spec.timestamp_column,
            timezone=self.spec.timezone, interval=self.spec.interval,
            column_map=self.spec.column_map,
        )
        return canonicalize(_read(b.path, fmt), temp_spec, symbol="__benchmark__")


class LongTableSource(OHLCVSource):
    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        if not spec.path.is_file():
            raise SourceError(f"long-table file does not exist: {spec.path}")
        raw = _read(spec.path, spec.format)
        if spec.symbol_column not in raw.columns:
            raise SourceError(f"missing symbol column: {spec.symbol_column}")
        self._raw = raw
        self._symbols = sorted(raw[spec.symbol_column].dropna().astype(str).unique())

    def instruments(self) -> list[InstrumentKey]:
        return [InstrumentKey(self.spec.dataset_id, s) for s in self._symbols]

    def load(self, key: InstrumentKey) -> pd.DataFrame:
        if key.dataset_id != self.spec.dataset_id or key.source_symbol not in self._symbols:
            raise KeyError(str(key))
        rows = self._raw[self._raw[self.spec.symbol_column].astype(str) == key.source_symbol]
        return canonicalize(rows, self.spec, symbol=key.source_symbol)

    def fingerprint(self, key: InstrumentKey) -> str:
        return self._source_fingerprint()

    @lru_cache(maxsize=1)
    def _source_fingerprint(self) -> str:
        return file_fingerprint(self.spec.path)

    def load_benchmark(self) -> pd.DataFrame | None:
        cached = self._load_benchmark()
        return cached.copy() if cached is not None else None

    def benchmark_fingerprint(self) -> str | None:
        if not self.spec.benchmark:
            return None
        return file_fingerprint(self.spec.benchmark.path)

    @lru_cache(maxsize=1)
    def _load_benchmark(self) -> pd.DataFrame | None:
        if not self.spec.benchmark:
            return None
        benchmark = self.spec.benchmark
        fmt = benchmark.format or benchmark.path.suffix.lstrip(".").lower()
        benchmark_spec = DatasetSpec(
            dataset_id=self.spec.dataset_id,
            adapter="directory",
            path=benchmark.path.parent,
            format=fmt,
            timestamp_column=benchmark.timestamp_column or self.spec.timestamp_column,
            timezone=self.spec.timezone,
            interval=self.spec.interval,
            column_map=self.spec.column_map,
        )
        return canonicalize(
            _read(benchmark.path, fmt), benchmark_spec, symbol="__benchmark__",
        )


def source_from_spec(spec: DatasetSpec) -> OHLCVSource:
    if spec.adapter == "directory":
        return DirectorySource(spec)
    if spec.adapter == "long_table":
        return LongTableSource(spec)
    raise SourceError(f"unsupported adapter: {spec.adapter}")

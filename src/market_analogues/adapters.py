from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import DatasetSpec
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
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if symbol is not None:
        df.attrs["symbol"] = symbol
    df.attrs["dataset_id"] = spec.dataset_id
    df.attrs["interval"] = spec.interval
    return df


class OHLCVSource(ABC):
    @abstractmethod
    def instruments(self) -> list[InstrumentKey]: ...

    @abstractmethod
    def load(self, key: InstrumentKey) -> pd.DataFrame: ...

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
        if key.source_symbol not in self._symbols:
            raise KeyError(str(key))
        rows = self._raw[self._raw[self.spec.symbol_column].astype(str) == key.source_symbol]
        return canonicalize(rows, self.spec, symbol=key.source_symbol)

    def fingerprint(self, key: InstrumentKey) -> str:
        return file_fingerprint(self.spec.path)


def source_from_spec(spec: DatasetSpec) -> OHLCVSource:
    if spec.adapter == "directory":
        return DirectorySource(spec)
    if spec.adapter == "long_table":
        return LongTableSource(spec)
    raise SourceError(f"unsupported adapter: {spec.adapter}")

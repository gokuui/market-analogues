from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd


CAUSAL_PREFIX_SCHEMA = "canonical-ohlcv-prefix-v1"
PREFIX_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class CausalPrefixError(ValueError):
    pass


@dataclass(frozen=True)
class CausalPrefixDigest:
    schema_version: str
    requested_cutoff: str
    coverage_cutoff: str
    rows: int
    digest: str


def _naive_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def causal_prefix_digest(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp | str,
) -> CausalPrefixDigest:
    """Hash canonical ordered OHLCV values visible through ``cutoff``.

    The digest is independent of file encoding and dataframe numeric dtypes. It
    deliberately rejects duplicate or reordered source timestamps instead of
    silently canonicalizing ambiguous input history.
    """
    missing = sorted(set(PREFIX_COLUMNS).difference(frame.columns))
    if missing:
        raise CausalPrefixError(f"prefix frame is missing columns: {missing}")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise CausalPrefixError("prefix frame contains invalid timestamps")
    if bool(frame.attrs.get("source_timestamp_reordered", False)):
        raise CausalPrefixError("source timestamps were reordered during canonicalization")
    if bool(frame.attrs.get("source_duplicate_timestamps", False)):
        raise CausalPrefixError("source contains duplicate timestamps")
    normalized = pd.Series([_naive_utc(value) for value in timestamps])
    if normalized.duplicated().any():
        raise CausalPrefixError("prefix frame contains duplicate timestamps")
    if not normalized.is_monotonic_increasing:
        raise CausalPrefixError("prefix frame timestamps are not strictly ordered")
    requested = _naive_utc(cutoff)
    positions = np.flatnonzero(normalized.to_numpy() <= requested.to_datetime64())
    if not len(positions):
        raise CausalPrefixError(f"no OHLCV rows at or before cutoff {requested.isoformat()}")
    last = int(positions[-1])
    if not np.array_equal(positions, np.arange(last + 1)):
        raise CausalPrefixError("eligible prefix is not contiguous")
    prefix_timestamps = normalized.iloc[:last + 1]
    numeric = frame.iloc[:last + 1].loc[:, PREFIX_COLUMNS[1:]].apply(
        pd.to_numeric, errors="coerce",
    ).to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise CausalPrefixError("prefix frame contains infinite OHLCV values")
    numeric = np.ascontiguousarray(numeric, dtype="<f8")
    numeric[numeric == 0.0] = 0.0
    numeric[np.isnan(numeric)] = np.nan
    timestamp_ns = np.ascontiguousarray(
        prefix_timestamps.to_numpy(dtype="datetime64[ns]").astype("<i8"),
    )
    header = json.dumps({
        "schema_version": CAUSAL_PREFIX_SCHEMA,
        "columns": PREFIX_COLUMNS,
        "rows": len(prefix_timestamps),
    }, sort_keys=True, separators=(",", ":")).encode()
    digest = sha256()
    digest.update(header)
    digest.update(timestamp_ns.tobytes())
    digest.update(numeric.tobytes())
    return CausalPrefixDigest(
        CAUSAL_PREFIX_SCHEMA,
        requested.isoformat(),
        pd.Timestamp(prefix_timestamps.iloc[-1]).isoformat(),
        len(prefix_timestamps),
        digest.hexdigest(),
    )


def prefix_generation_digest(
    *,
    dataset_id: str,
    query_cutoff: pd.Timestamp | str,
    representation_version: str,
    stock_prefixes: Mapping[str, CausalPrefixDigest],
    benchmark_prefix: CausalPrefixDigest | None,
) -> str:
    payload: dict[str, Any] = {
        "schema_version": "causal-prefix-generation-v1",
        "dataset_id": dataset_id,
        "query_cutoff": _naive_utc(query_cutoff).isoformat(),
        "representation_version": representation_version,
        "stock_prefixes": {
            symbol: asdict(value) for symbol, value in sorted(stock_prefixes.items())
        },
        "benchmark_prefix": asdict(benchmark_prefix) if benchmark_prefix else None,
    }
    return sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()

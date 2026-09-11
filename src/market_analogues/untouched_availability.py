"""Outcome-blind source-availability calculations for prospective evaluation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date
import os
from pathlib import Path
import stat
from typing import Iterable, Sequence

import pandas as pd
import pyarrow.parquet as pq


class AvailabilityError(RuntimeError):
    """Raised when an input cannot be audited without opening outcome values."""


@dataclass(frozen=True)
class DateMetadata:
    path: str
    bytes: int
    rows: int
    row_groups: int
    first_date: str
    last_date: str
    schema_names: tuple[str, ...]


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_ctime_ns, value.st_mode,
    )


def parquet_date_metadata(path: Path, timestamp_column: str = "date") -> DateMetadata:
    """Read schema and date-column footer statistics, never OHLCV values."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise AvailabilityError(f"unsafe or missing Parquet file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AvailabilityError(f"regular Parquet file required: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            parquet = pq.ParquetFile(handle)
            metadata = parquet.metadata
            names = tuple(parquet.schema_arrow.names)
            if timestamp_column not in names:
                raise AvailabilityError(f"timestamp column absent: {path}")
            column_index = names.index(timestamp_column)
            minima: list[pd.Timestamp] = []
            maxima: list[pd.Timestamp] = []
            for group_index in range(metadata.num_row_groups):
                column = metadata.row_group(group_index).column(column_index)
                statistics = column.statistics
                if statistics is None or not statistics.has_min_max:
                    raise AvailabilityError(f"date footer statistics absent: {path}")
                minima.append(pd.Timestamp(statistics.min))
                maxima.append(pd.Timestamp(statistics.max))
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise AvailabilityError(f"Parquet file changed during metadata read: {path}")
        if metadata.num_rows <= 0 or not minima:
            raise AvailabilityError(f"nonempty Parquet file required: {path}")
        return DateMetadata(
            path=path.name,
            bytes=before.st_size,
            rows=metadata.num_rows,
            row_groups=metadata.num_row_groups,
            first_date=min(minima).date().isoformat(),
            last_date=max(maxima).date().isoformat(),
            schema_names=names,
        )
    except AvailabilityError:
        raise
    except Exception as error:
        raise AvailabilityError(f"invalid Parquet metadata: {path}") from error
    finally:
        os.close(descriptor)


def parquet_dates(path: Path, timestamp_column: str = "date") -> pd.DatetimeIndex:
    """Read exactly one timestamp column through a stable, no-follow descriptor."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise AvailabilityError(f"unsafe or missing Parquet file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AvailabilityError(f"regular Parquet file required: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            table = pq.read_table(handle, columns=[timestamp_column])
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise AvailabilityError(f"Parquet file changed during date read: {path}")
        values = pd.DatetimeIndex(table.column(timestamp_column).to_pandas())
        if values.empty or values.hasnans or values.has_duplicates:
            raise AvailabilityError(f"unique nonnull dates required: {path}")
        if not values.is_monotonic_increasing:
            raise AvailabilityError(f"increasing dates required: {path}")
        return values
    except AvailabilityError:
        raise
    except Exception as error:
        raise AvailabilityError(f"timestamp column absent or invalid: {path}") from error
    finally:
        os.close(descriptor)


def scan_date_metadata(
    paths: Iterable[Path], *, timestamp_column: str = "date", workers: int = 12,
) -> list[DateMetadata]:
    """Scan date-only Parquet metadata in deterministic path order."""
    ordered = sorted(paths, key=lambda path: path.name)
    if workers < 1:
        raise AvailabilityError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(
            lambda path: parquet_date_metadata(path, timestamp_column), ordered,
        ))
    if len({value.path for value in values}) != len(values):
        raise AvailabilityError("duplicate source filename")
    return values


def month_end_sessions(
    sessions: Sequence[pd.Timestamp], *, after: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Return last observed session per calendar month strictly after a timestamp."""
    values = pd.DatetimeIndex(sessions).sort_values().unique()
    values = values[values > after]
    if values.empty:
        return []
    frame = pd.Series(values, index=values.to_period("M"))
    return [pd.Timestamp(value) for value in frame.groupby(level=0).max()]


def completed_month_end_sessions(
    sessions: Sequence[pd.Timestamp], *, after: pd.Timestamp, as_of: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Return month-end sessions only for calendar months already completed."""
    return [
        value for value in month_end_sessions(sessions, after=after)
        if value.to_period("M").end_time < as_of.tz_localize(None)
    ]


def required_schedule(
    benchmark_sessions: Sequence[pd.Timestamp], *, consumed_boundary: pd.Timestamp,
    purge_sessions: int, evaluation_months: int,
) -> dict[str, object]:
    """Build the observable portion of a calendar-derived evaluation schedule."""
    if purge_sessions < 1 or evaluation_months < 1:
        raise AvailabilityError("positive purge and evaluation lengths required")
    sessions = pd.DatetimeIndex(benchmark_sessions).sort_values().unique()
    post = sessions[sessions > consumed_boundary]
    if len(post) <= purge_sessions:
        return {
            "purge_completed": False,
            "purge_completion_session": None,
            "observed_month_end_cutoffs": [],
        }
    purge_completion = pd.Timestamp(post[purge_sessions - 1])
    cutoffs = month_end_sessions(sessions, after=purge_completion)[:evaluation_months]
    return {
        "purge_completed": True,
        "purge_completion_session": purge_completion.date().isoformat(),
        "observed_month_end_cutoffs": [value.date().isoformat() for value in cutoffs],
    }


def maturity_date(
    benchmark_sessions: Sequence[pd.Timestamp], cutoff: pd.Timestamp, horizon: int,
) -> str | None:
    """Return the horizon-th subsequent benchmark session if it is observable."""
    if horizon < 1:
        raise AvailabilityError("positive maturity horizon required")
    sessions = pd.DatetimeIndex(benchmark_sessions).sort_values().unique()
    future = sessions[sessions > cutoff]
    return future[horizon - 1].date().isoformat() if len(future) >= horizon else None


def source_counts(
    metadata: Sequence[DateMetadata], cutoffs: Sequence[str], *, minimum_rows: int,
) -> dict[str, int]:
    """Count source files with enough history and a bar through each cutoff."""
    if minimum_rows < 1:
        raise AvailabilityError("minimum rows must be positive")
    return {
        cutoff: sum(
            value.rows >= minimum_rows and date.fromisoformat(value.last_date)
            >= date.fromisoformat(cutoff)
            for value in metadata
        )
        for cutoff in cutoffs
    }


def metadata_state(values: Sequence[DateMetadata]) -> list[dict[str, object]]:
    """Return stable JSON-compatible metadata rows."""
    return [asdict(value) for value in values]

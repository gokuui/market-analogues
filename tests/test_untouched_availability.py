from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_analogues.untouched_availability import (
    AvailabilityError,
    completed_month_end_sessions,
    maturity_date,
    month_end_sessions,
    parquet_date_metadata,
    parquet_dates,
    required_schedule,
    scan_date_metadata,
    source_counts,
)


def write_fixture(path: Path, dates: list[str]) -> None:
    pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": range(len(dates)),
        "high": range(len(dates)),
        "low": range(len(dates)),
        "close": range(len(dates)),
        "volume": range(len(dates)),
    }).to_parquet(path, index=False)


def test_footer_scan_is_deterministic_and_date_only(tmp_path: Path) -> None:
    write_fixture(tmp_path / "b.parquet", ["2025-01-02", "2025-02-03"])
    write_fixture(tmp_path / "a.parquet", ["2024-12-31", "2025-03-04"])
    values = scan_date_metadata(tmp_path.glob("*.parquet"), workers=2)
    assert [value.path for value in values] == ["a.parquet", "b.parquet"]
    assert values[0].first_date == "2024-12-31"
    assert values[0].last_date == "2025-03-04"
    assert values[0].rows == 2
    assert list(parquet_dates(tmp_path / "a.parquet")) == list(pd.to_datetime([
        "2024-12-31", "2025-03-04",
    ]))
    assert source_counts(values, ["2025-02-01", "2025-04-01"], minimum_rows=2) == {
        "2025-02-01": 2, "2025-04-01": 0,
    }


def test_footer_scan_refuses_symlink_and_missing_timestamp(tmp_path: Path) -> None:
    write_fixture(tmp_path / "real.parquet", ["2025-01-02"])
    (tmp_path / "link.parquet").symlink_to(tmp_path / "real.parquet")
    with pytest.raises(AvailabilityError, match="unsafe"):
        parquet_date_metadata(tmp_path / "link.parquet")
    with pytest.raises(AvailabilityError, match="unsafe"):
        parquet_dates(tmp_path / "link.parquet")
    pd.DataFrame({"close": [1.0]}).to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(AvailabilityError, match="timestamp column absent"):
        parquet_date_metadata(tmp_path / "bad.parquet")


def test_schedule_purges_then_uses_calendar_month_ends() -> None:
    sessions = pd.bdate_range("2025-08-25", "2026-06-30")
    schedule = required_schedule(
        sessions, consumed_boundary=pd.Timestamp("2025-08-29"),
        purge_sessions=60, evaluation_months=4,
    )
    assert schedule == {
        "purge_completed": True,
        "purge_completion_session": "2025-11-21",
        "observed_month_end_cutoffs": [
            "2025-11-28", "2025-12-31", "2026-01-30", "2026-02-27",
        ],
    }
    assert maturity_date(sessions, pd.Timestamp("2026-02-27"), 60) == "2026-05-22"
    assert month_end_sessions(sessions, after=pd.Timestamp("2026-05-31")) == [
        pd.Timestamp("2026-06-30"),
    ]
    assert completed_month_end_sessions(
        sessions, after=pd.Timestamp("2026-04-30"), as_of=pd.Timestamp("2026-06-15"),
    ) == [pd.Timestamp("2026-05-29")]


def test_schedule_and_maturity_refuse_insufficient_history() -> None:
    sessions = pd.bdate_range("2025-09-01", periods=20)
    assert required_schedule(
        sessions, consumed_boundary=pd.Timestamp("2025-08-29"),
        purge_sessions=60, evaluation_months=12,
    )["purge_completed"] is False
    assert maturity_date(sessions, sessions[-1], 60) is None
    with pytest.raises(AvailabilityError):
        required_schedule(sessions, consumed_boundary=pd.Timestamp("2025-08-29"),
                          purge_sessions=0, evaluation_months=12)


def test_malformed_parquet_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(AvailabilityError, match="invalid Parquet metadata"):
        parquet_date_metadata(path)
    with pytest.raises(AvailabilityError, match="timestamp column absent or invalid"):
        parquet_dates(path)

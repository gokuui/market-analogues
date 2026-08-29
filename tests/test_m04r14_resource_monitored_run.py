from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import m04r14_resource_monitored_run as monitored


def test_counter_delta_is_exact_and_rejects_reset() -> None:
    assert monitored._counter_delta({"oom": 2}, {"oom": 3, "oom_kill": 0}) == {
        "oom": 1, "oom_kill": 0,
    }
    with pytest.raises(monitored.ResourceMonitorError, match="decreased"):
        monitored._counter_delta({"oom": 2}, {"oom": 1})


def test_status_parser_reads_linux_kib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proc = tmp_path / "proc/7"
    proc.mkdir(parents=True)
    (proc / "status").write_text("VmRSS:\t10 kB\nVmHWM:\t12 kB\nVmSwap:\t0 kB\n")
    real_path = Path

    def redirected(value: str) -> Path:
        if value == "/proc/7/status":
            return proc / "status"
        return real_path(value)

    monkeypatch.setattr(monitored, "Path", redirected)
    assert monitored._status_kib(7) == {"VmRSS": 10, "VmHWM": 12, "VmSwap": 0}


def test_command_freezes_full_p8_shape(tmp_path: Path) -> None:
    command = monitored._command(tmp_path, tmp_path / "out")
    assert command[-4:] == ["--processes", "8", "--case-limit", "60"]

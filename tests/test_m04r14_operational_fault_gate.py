from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m04r import m04r14_operational_fault_gate as gate


def test_tree_manifest_is_sorted_and_hash_bound(tmp_path: Path) -> None:
    (tmp_path / "cases").mkdir()
    (tmp_path / "z.json").write_text("z")
    (tmp_path / "cases/a.json").write_text("a")
    rows = gate._tree_manifest(tmp_path)
    assert [row["path"] for row in rows] == ["cases/a.json", "z.json"]
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_tree_manifest_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("value")
    (tmp_path / "alias").symlink_to(target)
    with pytest.raises(gate.FaultGateError, match="symlink"):
        gate._tree_manifest(tmp_path)


def test_restart_command_is_one_process_one_hard_case(tmp_path: Path) -> None:
    command = gate._command(tmp_path, tmp_path / "out")
    assert command[-4:] == ["--processes", "1", "--case-limit", "1"]

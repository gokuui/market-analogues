from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_11_stockbee_risk_set as runner
from tests.test_stockbee_study import _bars


def _sha(path: Path) -> str:
    digest = sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


def test_streamed_shard_is_atomic_complete_and_idempotent(tmp_path: Path) -> None:
    cache = tmp_path / "cache"; cache.mkdir()
    records = []
    for symbol in ("SYN1", "SYN2"):
        path = tmp_path / f"{symbol}.parquet"
        frame = _bars().rename(columns={"timestamp": "date"})
        frame.to_parquet(path, index=False)
        records.append({
            "symbol": symbol, "source_path": str(path), "rows_through_lock": len(frame),
            "coverage_last_timestamp": frame.date.max().isoformat(),
            "source_hash_at_lock": _sha(path),
        })
    first = runner._write_shard(0, records, str(cache), "contract")
    second = runner._write_shard(0, records, str(cache), "contract")
    assert first["result_digest"] == second["result_digest"]
    assert first["symbol_count"] == 2 and first["risk_rows"] > 0
    root = cache / "shard-00"
    assert {path.name for path in root.iterdir()} == {
        "risk-set.parquet", "symbol-accounting.parquet", "SHARD_SEALED.json",
    }
    accounting = pd.read_parquet(root / "symbol-accounting.parquet")
    risk = pd.read_parquet(root / "risk-set.parquet")
    assert set(accounting.symbol) == {"SYN1", "SYN2"}
    assert risk.groupby("symbol").size().to_dict() == dict(zip(accounting.symbol, accounting.risk_rows))

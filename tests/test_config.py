from pathlib import Path

import pytest

from market_analogues.config import ConfigError, DatasetSpec, load_config


def test_load_config_resolves_relative_paths(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
artifact_dir: artifacts
datasets:
  demo:
    adapter: directory
    path: bars
    format: parquet
    timestamp_column: date
""")
    loaded = load_config(cfg)
    assert loaded.artifact_dir == (tmp_path / "artifacts").resolve()
    assert loaded.datasets["demo"].path == (tmp_path / "bars").resolve()
    assert loaded.lookbacks == (21, 63, 126, 252)


@pytest.mark.parametrize("field,value", [("adapter", "api"), ("format", "json"), ("symbol_from", "magic")])
def test_invalid_dataset_contract(field, value, tmp_path):
    args = dict(dataset_id="x", adapter="directory", path=tmp_path, format="parquet")
    args[field] = value
    with pytest.raises(ConfigError):
        DatasetSpec(**args)


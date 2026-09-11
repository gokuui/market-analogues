from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkSpec:
    path: Path
    timestamp_column: str | None = None
    format: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    adapter: str
    path: Path
    format: str
    file_glob: str = "*.parquet"
    symbol_from: str = "filename"
    symbol_column: str = "symbol"
    timestamp_column: str = "timestamp"
    interval: str = "1d"
    timezone: str = "UTC"
    column_map: dict[str, str] = field(default_factory=dict)
    benchmark: BenchmarkSpec | None = None
    quality_manifest: Path | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ConfigError("dataset_id cannot be empty")
        if self.adapter not in {"directory", "long_table"}:
            raise ConfigError(f"unsupported adapter: {self.adapter}")
        if self.format not in {"csv", "parquet"}:
            raise ConfigError(f"unsupported format: {self.format}")
        if self.symbol_from not in {"filename", "column"}:
            raise ConfigError("symbol_from must be filename or column")


@dataclass(frozen=True)
class AppConfig:
    artifact_dir: Path
    datasets: dict[str, DatasetSpec]
    representation_version: str = "dense-v1"
    candidate_stride_bars: int = 5
    lookbacks: tuple[int, ...] = (21, 63, 126, 252)

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ConfigError("at least one dataset is required")
        if self.candidate_stride_bars < 1:
            raise ConfigError("candidate_stride_bars must be positive")
        if not self.lookbacks or any(x < 2 for x in self.lookbacks):
            raise ConfigError("lookbacks must contain integers >= 2")


def _path(value: str | Path, base: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def _config_from_mapping(raw: dict[str, Any], config_path: Path) -> AppConfig:
    datasets: dict[str, DatasetSpec] = {}
    for dataset_id, item in (raw.get("datasets") or {}).items():
        benchmark = None
        if item.get("benchmark"):
            b = item["benchmark"]
            benchmark = BenchmarkSpec(
                path=_path(b["path"], config_path.parent),
                timestamp_column=b.get("timestamp_column"),
                format=b.get("format"),
            )
        datasets[dataset_id] = DatasetSpec(
            dataset_id=dataset_id,
            adapter=item.get("adapter", "directory"),
            path=_path(item["path"], config_path.parent),
            format=item.get("format", "parquet"),
            file_glob=item.get("file_glob", "*.parquet"),
            symbol_from=item.get("symbol_from", "filename"),
            symbol_column=item.get("symbol_column", "symbol"),
            timestamp_column=item.get("timestamp_column", "timestamp"),
            interval=item.get("interval", "1d"),
            timezone=item.get("timezone", "UTC"),
            column_map=item.get("column_map", {}),
            benchmark=benchmark,
            quality_manifest=_path(item["quality_manifest"], config_path.parent)
            if item.get("quality_manifest") else None,
        )
    return AppConfig(
        artifact_dir=_path(raw.get("artifact_dir", "data/analogues"), config_path.parent),
        datasets=datasets,
        representation_version=raw.get("representation_version", "dense-v1"),
        candidate_stride_bars=int(raw.get("candidate_stride_bars", 5)),
        lookbacks=tuple(int(x) for x in raw.get("lookbacks", [21, 63, 126, 252])),
    )


def load_config_bytes(content: bytes, *, path: str | Path) -> AppConfig:
    """Parse already-read configuration bytes using ``path`` as their base.

    This entry point lets integrity-sensitive callers hash and parse the same
    immutable byte buffer instead of reopening a pathname between those steps.
    """
    config_path = Path(path).resolve()
    raw: Any = yaml.safe_load(content) or {}
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    return _config_from_mapping(raw, config_path)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    return load_config_bytes(config_path.read_bytes(), path=config_path)

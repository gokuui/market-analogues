from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any


@dataclass
class GateReport:
    task: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    code_version: str = "0.1.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self), indent=2, sort_keys=True)
        history = directory / "history" / self.task
        history.mkdir(parents=True, exist_ok=True)
        stamp = self.created_at.replace(":", "-")
        immutable = history / f"{stamp}.json"
        if immutable.exists():
            raise FileExistsError(f"gate record already exists: {immutable}")
        immutable.write_text(payload)
        path = directory / f"{self.task}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload)
        tmp.replace(path)
        return path


def require_passed(directory: Path, previous_task: str) -> None:
    path = directory / f"{previous_task}.json"
    if not path.exists():
        raise RuntimeError(f"required gate is missing: {previous_task}")
    if not json.loads(path.read_text()).get("passed"):
        raise RuntimeError(f"required gate failed: {previous_task}")

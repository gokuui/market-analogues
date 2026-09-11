from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.m04r import verify_m04r15_r2_full_runner_engineering_gate as verifier


ROOT = Path(__file__).resolve().parents[1]


def _values() -> tuple[dict, dict]:
    result = json.loads((ROOT / verifier.RESULT).read_text())
    contract = json.loads((ROOT / verifier.CONTRACT).read_text())
    return result, contract


def test_summary_accepts_published_engineering_gate() -> None:
    verifier._validate_summary(*_values())


@pytest.mark.parametrize("section,key,value", [
    ("architecture", "configured_workers", 11),
    ("architecture", "serial_parallel_partition_bytes_identical", False),
    ("architecture", "completed_prefix_reused_without_rewrite", False),
    ("authentic_input", "path_bearing_episode_count", 56_324),
    ("authentic_input", "dictionary_image_bytes", 600 << 20),
    ("performance", "peak_rss_kib", 3 << 20),
    (None, "real_future_path_modes_computed", True),
    (None, "predictive_claim_authorized", True),
])
def test_summary_rejects_mutation(section: str | None, key: str, value: object) -> None:
    result, contract = _values()
    result = deepcopy(result)
    target = result if section is None else result[section]
    target[key] = value
    deterministic = {name: item for name, item in result.items()
                     if name not in {"created_at", "performance", "result_digest"}}
    result["result_digest"] = verifier._stable(deterministic)
    with pytest.raises(verifier.EngineeringVerificationError, match="summary differs"):
        verifier._validate_summary(result, contract)

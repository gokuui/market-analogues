from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_composite_nonfinite_diagnostic as subject


def test_nonfinite_paths_are_exact_and_finite_values_are_ignored() -> None:
    value = {
        "finite": [0.0, 1],
        "certificate": {"rounds": [{"threshold": float("inf")}],
                        "minimum": float("-inf")},
        "nan": float("nan"),
    }
    assert subject._nonfinite_paths(value) == [
        {"path": "$.certificate.rounds[0].threshold", "value": "inf"},
        {"path": "$.certificate.minimum", "value": "-inf"},
        {"path": "$.nan", "value": "nan"},
    ]

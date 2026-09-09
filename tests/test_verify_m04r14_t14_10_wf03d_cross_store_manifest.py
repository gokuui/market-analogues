from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import (  # noqa: E402
    m04r14_t14_10_wf03d_cross_store_manifest as producer,
)
from experiments.m04r import (  # noqa: E402
    verify_m04r14_t14_10_wf03d_cross_store_manifest as subject,
)


def test_verifier_semantic_digest_is_independently_equivalent() -> None:
    rows = [
        {"query_id": "a", "rank": 1, "distance_hex": "0x1.0p+0"},
        {"query_id": "a", "rank": 2, "distance_hex": None},
    ]
    assert subject._semantic_digest(rows) == producer._semantic_digest(rows)


def test_verifier_distance_encoding_covers_all_method_families() -> None:
    assert subject._distance_hex("composite", {"total_distance": 1.5}) == float(1.5).hex()
    assert subject._distance_hex("price_only", {"distance_hex": "0x1.8p+0"}) == "0x1.8p+0"
    assert subject._distance_hex("deterministic_random", {"distance_hex": None}) is None

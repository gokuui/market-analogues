from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import (  # noqa: E402
    m04r14_t14_10_wf03d_cross_store_manifest as subject,
)


METHODS = ("composite", "price_only")


def _inventory() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    links: list[dict[str, object]] = []
    requests: dict[str, dict[str, str]] = {}
    for query_index in range(2):
        for method_index, method in enumerate(METHODS):
            for rank in range(1, 3):
                episode_id = f"{query_index * 4 + method_index * 2 + rank:024x}"
                symbol = f"M{query_index}{method_index}{rank}"
                cutoff = f"2019-0{rank}-01T00:00:00"
                requests[episode_id] = {
                    "episode_id": episode_id, "dataset_id": "nasdaq",
                    "symbol": symbol, "cutoff": cutoff, "quality_tier": "A",
                }
                links.append({
                    "query_id": f"{100 + query_index:024x}",
                    "query_symbol": f"Q{query_index}", "method": method,
                    "rank": rank, "matched_episode_id": episode_id,
                    "matched_symbol": symbol, "matched_cutoff": cutoff,
                    "quality_tier": "A", "latest_eligible_ns": 2_000_000_000_000_000_000,
                })
    return links, [requests[key] for key in sorted(requests)]


def _validate(links: list[dict[str, object]], requests: list[dict[str, str]]) -> None:
    subject._validate_logical_inventory(
        links, requests, expected_queries=2, expected_methods=METHODS, top_k=2,
    )


def test_synthetic_cross_store_inventory_passes() -> None:
    _validate(*_inventory())


@pytest.mark.parametrize("mutation", [
    "duplicate_rank", "query_symbol", "future_cutoff", "missing_request",
    "conflicting_request",
])
def test_synthetic_cross_store_inventory_rejects_mutations(mutation: str) -> None:
    links, requests = _inventory()
    links, requests = deepcopy(links), deepcopy(requests)
    if mutation == "duplicate_rank":
        links[1]["rank"] = 1
    elif mutation == "query_symbol":
        links[0]["matched_symbol"] = links[0]["query_symbol"]
        requests[0]["symbol"] = str(links[0]["query_symbol"])
    elif mutation == "future_cutoff":
        links[0]["latest_eligible_ns"] = 1
    elif mutation == "missing_request":
        requests.pop()
    else:
        requests[0]["symbol"] = "CONFLICT"
    with pytest.raises(subject.CrossStoreManifestError):
        _validate(links, requests)


def test_semantic_digest_is_order_and_null_sensitive() -> None:
    links, _ = _inventory()
    first = subject._semantic_digest(links)
    assert first == subject._semantic_digest(deepcopy(links))
    assert first != subject._semantic_digest(list(reversed(links)))
    changed = deepcopy(links)
    changed[0]["distance_hex"] = None
    assert first != subject._semantic_digest(changed)

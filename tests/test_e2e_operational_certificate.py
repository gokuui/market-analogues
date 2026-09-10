import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_e2e_operational_certificate as operational
from market_analogues.types import stable_hash


def test_semantic_receipt_boundary() -> None:
    state = {"passed": True, "cases": 12}
    value = {
        **state, "result_digest": stable_hash(state),
        "created_at": "now", "elapsed_seconds": 2.0,
    }
    assert operational._semantic_receipt(
        value, omitted={"result_digest", "created_at", "elapsed_seconds"},
    )
    value["cases"] = 11
    assert not operational._semantic_receipt(
        value, omitted={"result_digest", "created_at", "elapsed_seconds"},
    )


def test_corrupt_real_authority_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    matches = [{"total_distance": 1.0}]
    payload = {
        "schema_version": "gate12-authority-v1", "matches": matches,
        "result_digest": stable_hash(matches), "repeated_digest": stable_hash(matches),
        "certificate": {}, "repeated_certificate": {},
        "frontier_manifest_digest": "x", "frontier_build": {},
        "frontier_resume": {}, "query_episode_id": "q",
    }
    payload["authority_digest"] = stable_hash(payload)
    source.write_text(json.dumps(payload))
    assert operational._corruption_refused(source)


def test_html_parser_rejects_missing_file(tmp_path: Path) -> None:
    assert not operational._parse_html([tmp_path / "missing.html"])

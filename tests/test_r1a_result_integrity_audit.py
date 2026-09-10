from experiments.m04r.audit_m04r14_r1a_exposure_result import _semantic_digest
from market_analogues.types import stable_hash


def test_semantic_digest_omits_only_declared_publication_fields() -> None:
    payload = {"a": 1, "created_at": "later", "digest": "self"}
    assert _semantic_digest(payload, {"created_at", "digest"}) == stable_hash({"a": 1})
    assert _semantic_digest(payload, {"digest"}) == stable_hash({"a": 1, "created_at": "later"})

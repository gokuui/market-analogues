import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_e2e_release_snapshot as release
from market_analogues.types import stable_hash


def test_html_contract(tmp_path: Path) -> None:
    good = tmp_path / "good.html"
    good.write_text("<!doctype html><html lang='en'><body>x</body></html>")
    assert release._parse_html(good)
    good.write_text("<html><body>x</body></html>")
    assert not release._parse_html(good)
    good.write_text("<!doctype html><html><body>x</body>")
    assert not release._parse_html(good)


def test_operational_semantic_digest_excludes_publication_fields() -> None:
    state = {"schema_version": "x", "passed": True, "failures": []}
    payload = {
        **state, "result_digest": stable_hash(state), "created_at": "later",
        "elapsed_seconds": 4.0,
    }
    assert release._semantic_digest(
        payload, {"result_digest", "created_at", "elapsed_seconds"},
    )
    payload["passed"] = False
    assert not release._semantic_digest(
        payload, {"result_digest", "created_at", "elapsed_seconds"},
    )


def test_portable_semantic_digest_excludes_publication_fields() -> None:
    state = {"schema_version": "x", "failures": []}
    payload = {
        **state, "passed": True, "result_digest": stable_hash(state),
        "generated_at": "later",
    }
    assert release._portable_digest(payload)
    payload["failures"] = ["tampered"]
    assert not release._portable_digest(payload)


def test_load_json_rejects_non_object(tmp_path: Path) -> None:
    value = tmp_path / "value.json"
    value.write_text(json.dumps([1, 2]))
    try:
        release._load_json(value)
    except release.ReleaseSnapshotError:
        pass
    else:
        raise AssertionError("non-object receipt accepted")


def test_load_json_rejects_duplicate_and_nonfinite_values(tmp_path: Path) -> None:
    value = tmp_path / "value.json"
    for body in ('{"x": 1, "x": 2}', '{"x": NaN}'):
        value.write_text(body)
        try:
            release._load_json(value)
        except release.ReleaseSnapshotError:
            pass
        else:
            raise AssertionError(f"invalid JSON accepted: {body}")


def test_status_links_must_be_local_and_present(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "target.html").write_text("ok")
    status = docs / "status.html"
    status.write_text("<!doctype html><html><a href='../target.html'>x</a></html>")
    assert release._local_links_exist(tmp_path, status)
    status.write_text("<!doctype html><html><a href='../../escape'>x</a></html>")
    assert not release._local_links_exist(tmp_path, status)

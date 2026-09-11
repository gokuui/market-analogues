import ast
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.m04r import m04r14_r1b_b2_localization_verifier as verifier
from market_analogues.types import stable_hash


def test_independent_stable_hash_matches_repository_wire_format():
    value = {"z": [1, 2], "a": {"x": True}}
    assert verifier.stable(value) == stable_hash(value)


def test_authenticated_snapshot_rejects_mutation_and_symlink(tmp_path):
    path = tmp_path / "value"; path.write_bytes(b"abc")
    digest = sha256(b"abc").hexdigest()
    assert verifier.bound(path, digest, 3) == b"abc"
    path.write_bytes(b"abd")
    with pytest.raises(verifier.B2VerificationError, match="bound hash"):
        verifier.bound(path, digest, 3)
    link = tmp_path / "link"; link.symlink_to(path)
    with pytest.raises(verifier.B2VerificationError, match="unsafe"):
        verifier.snapshot(link)


def test_json_duplicate_and_nonfinite_are_refused(tmp_path):
    with pytest.raises(verifier.B2VerificationError, match="duplicate"):
        verifier.decode(b'{"a":1,"a":2}', tmp_path / "x")
    with pytest.raises(verifier.B2VerificationError, match="nonfinite"):
        verifier.decode(b'{"a":NaN}', tmp_path / "x")


def test_empirical_transform_independent_ties_constants_and_candidates(monkeypatch):
    monkeypatch.setattr(verifier, "QUERIES", 3)
    monkeypatch.setattr(verifier, "COHORT", 2)
    queries = np.zeros((3, verifier.DIMENSIONS)); candidates = np.zeros((2, verifier.DIMENSIONS))
    queries[:, 0] = [1, 2, 2]; candidates[:, 0] = [0, 3]
    candidates[:, 1] = [-1, 1]
    q, c, constants = verifier.empirical(queries, candidates)
    np.testing.assert_array_equal(q[:, 0], [-1, .5, .5])
    np.testing.assert_array_equal(c[:, 0], [-1, 1])
    np.testing.assert_array_equal(c[:, 1], [-1, 1])
    assert constants == list(range(1, verifier.DIMENSIONS))


def test_specificity_uses_every_eligible_full_cohort_member():
    distances = np.asarray([[.3, .2, .1], [.1, 99, .2]], dtype=float)
    eligible = np.asarray([[True, True, True], [True, False, True]])
    value = verifier.specificity(distances, eligible)
    np.testing.assert_array_equal(value[0], [5/6, 1/2, 1/6])
    assert np.isnan(value[1, 1])
    np.testing.assert_array_equal(value[1, [0, 2]], [1/4, 3/4])


def test_entropy_breadth_canonical_clamp_and_bounds():
    labels = np.asarray([0, 1, 2, 3, 4], dtype=np.int8)
    assert verifier.breadth(range(5), labels) == 1.0
    for selected in ([0], [0, 1], [0, 0, 1, 2, 3]):
        value = verifier.breadth(selected, labels)
        assert 0 < value <= 1


def test_priority_wire_format_and_full_hash():
    contract = "ab" * 32; episode = "e" * 24; query = "q"
    parts = [verifier.PRIORITY_DOMAIN, bytes.fromhex(contract), verifier.PRIORITY_FAMILY,
             b"episode", (17).to_bytes(4, "big"), episode.encode(), query.encode()]
    expected = sha256(b"".join(len(part).to_bytes(4, "big") + part for part in parts)).digest()
    assert verifier.priority(contract, 17, episode, query, False) == expected
    shared = parts[:]; shared[3] = b"shared-query"; shared.pop(-2)
    assert verifier.priority(contract, 17, None, query, True) == sha256(
        b"".join(len(part).to_bytes(4, "big") + part for part in shared)).digest()


def test_conditional_selection_preserves_cells_and_identity_ties():
    groups = ((1, (0, 1)), (2, (2, 3, 4)))
    priorities = dict.fromkeys(range(5), b"x" * 32)
    qids = ("z", "a", "d", "b", "c")
    assert verifier.select(groups, priorities, qids) == (1, 3, 4)


def test_effect_uses_equal_episode_weight_thresholds_and_strict_counts():
    observed = np.asarray([[.4, .4], [.4, .4], [.6, .6], [.4, .4], [.4, .4]])
    null = np.full((5, 2), .5)
    result = verifier.effect(observed, null)
    assert result["required_improved_episodes"] == 3
    assert result["cohesion_improved_episodes"] == 4
    assert result["specificity_improved_episodes"] == 4
    assert result["practical_pass"] is True


def test_replay_chunk_independently_builds_all_six_tables(monkeypatch):
    monkeypatch.setattr(verifier, "PRIMARY", 1)
    qids = ("q0", "q1", "q2", "q3")
    qq = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]).astype(float)
    ranks = np.asarray([[.1], [.3], [.5], [.7]])
    groups = (((1, (0, 1)), (1, (2, 3))),) * 4
    plan = verifier.Plan("e" * 24, 0, (0, 2), groups, (0, 1, 2, 3))
    verifier._REPLAY = verifier.Replay(qids, (plan,), qq, ranks, np.asarray([0,1,0,1]), "ab"*32)
    start, arrays = verifier.replay_chunk((0, 2))
    assert start == 0 and set(arrays) == set(verifier.ARRAYS)
    assert all(arrays[name].shape == (2, 1, 2) for name in verifier.NULLS)
    assert arrays["episode_n0_breadth"].shape == (2, 1)
    for name in ("episode_n0", "episode_n1", "episode_k12", "episode_k16"):
        np.testing.assert_array_equal(arrays[name], arrays["episode_n0"])


def test_npy_authenticates_file_and_semantic_digest(tmp_path):
    value = np.arange(6, dtype="<f8").reshape(2, 3); stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
    path = tmp_path / "a.npy"; path.write_bytes(stream.getvalue())
    record = {"path":"a.npy","sha256":sha256(stream.getvalue()).hexdigest(),
              "semantic_digest":verifier.geometry_semantic(value),"shape":[2,3],"dtype":"<f8","order":"C"}
    np.testing.assert_array_equal(verifier.npy(path, record, (2,3), np.dtype("<f8")), value)
    record["semantic_digest"] = "0" * 64
    with pytest.raises(verifier.B2VerificationError, match="semantic"):
        verifier.npy(path, record, (2,3), np.dtype("<f8"))


def test_create_only_receipt(tmp_path):
    path = tmp_path / "receipt.json"; verifier.publish(path, {"passed": True})
    assert json.loads(path.read_text()) == {"passed": True}
    with pytest.raises((verifier.B2VerificationError, FileExistsError)):
        verifier.publish(path, {"passed": False})


def test_verifier_has_no_producer_or_outcome_imports():
    path = Path(verifier.__file__); tree = ast.parse(path.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): imports.append(node.module or "")
    assert not any("b2_localization" in name or "adequacy_localization" in name for name in imports)
    assert not any("outcome" in name or "prediction" in name for name in imports)

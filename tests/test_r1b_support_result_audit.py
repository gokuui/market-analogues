from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from experiments.m04r import audit_m04r14_r1b_support_result as audit
from market_analogues.adequacy_support import (
    deterministic_terciles,
    farthest_first_partition,
    matched_support,
)


def test_strict_json_loaders_reject_duplicates_nonfinite_and_wrong_shape(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}')
    with pytest.raises(audit.SupportIntegrityError, match="duplicate JSON key"):
        audit._load_object(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('[{"a": NaN}]')
    with pytest.raises(audit.SupportIntegrityError, match="non-finite JSON"):
        audit._load_list(nonfinite)

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"a": 1}))
    with pytest.raises(audit.SupportIntegrityError, match="JSON list required"):
        audit._load_list(wrong)


@pytest.mark.parametrize(
    ("eligible", "observed", "cells", "cap"),
    [
        ((0, 1, 2, 3), (0, 2), ("a", "a", "b", "b"), 4096),
        ((0, 1, 2, 3, 4), (1, 3, 4), (0, 0, 0, 1, 1), 3),
        ((0, 1), (), ("a", "b"), 4096),
    ],
)
def test_independent_matched_support_matches_canonical(
    eligible: tuple[int, ...], observed: tuple[int, ...],
    cells: tuple[object, ...], cap: int,
) -> None:
    assert audit._matched_support(eligible, observed, cells, cap=cap) == matched_support(
        eligible, observed, cells, cap=cap,
    )


def test_independent_matched_support_rejects_membership_mutations() -> None:
    with pytest.raises(audit.SupportIntegrityError, match="duplicates"):
        audit._matched_support((0, 0), (0,), ("a",), cap=10)
    with pytest.raises(audit.SupportIntegrityError, match="not eligible"):
        audit._matched_support((0,), (1,), ("a", "b"), cap=10)
    with pytest.raises(audit.SupportIntegrityError, match="outside cells"):
        audit._matched_support((2,), (), ("a",), cap=10)


def test_independent_terciles_match_canonical_with_ties_and_nan() -> None:
    values = np.asarray([2.0, np.nan, 1.0, 2.0, 4.0, 3.0])
    ids = ("f", "e", "d", "c", "b", "a")
    np.testing.assert_array_equal(
        audit._deterministic_terciles(values, ids),
        deterministic_terciles(values, ids),
    )


@pytest.mark.parametrize("minimum_size", [1, 3, 20])
def test_independent_farthest_partition_matches_canonical(minimum_size: int) -> None:
    rng = np.random.default_rng(713)
    vectors = rng.normal(size=(19, 7))
    vectors[4] = vectors[3]
    ids = tuple(f"q-{index:02d}" for index in range(len(vectors)))
    actual = audit._farthest_first_partition(
        vectors, ids, clusters=6, minimum_size=minimum_size,
    )
    expected = farthest_first_partition(
        vectors, ids, clusters=6, minimum_size=minimum_size,
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


def test_structure_nontrivial_requires_real_selected_partition() -> None:
    base = {"distinct_cells": 71}
    assert not audit._structure_nontrivial(
        {"retained_structure_cells": 1, "distinct_cells": 71}, base, [0, 0, 0],
    )
    assert not audit._structure_nontrivial(
        {"retained_structure_cells": 2, "distinct_cells": 71}, base, [0, 1, 1],
    )
    assert audit._structure_nontrivial(
        {"retained_structure_cells": 2, "distinct_cells": 72}, base, [0, 1, 1],
    )


def test_atomic_publish_is_create_only(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    (first / "receipt").write_text("one")
    destination = tmp_path / "published"
    audit._atomic_publish_directory(first, destination)
    assert (destination / "receipt").read_text() == "one"

    second = tmp_path / "second"
    second.mkdir()
    (second / "receipt").write_text("two")
    with pytest.raises(audit.SupportIntegrityError, match="create-only output exists"):
        audit._atomic_publish_directory(second, destination)
    assert (destination / "receipt").read_text() == "one"


def test_atomic_receipt_leaves_exact_one_file_closure(tmp_path: Path) -> None:
    audit._atomic_receipt(tmp_path / "VERIFIED.json", {"passed": True})
    assert [path.name for path in tmp_path.iterdir()] == ["VERIFIED.json"]


def _git(repository: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=repository, check=True, capture_output=True)


def test_self_provenance_requires_clean_committed_bytes(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "audit@example.invalid")
    _git(tmp_path, "config", "user.name", "Audit Test")
    runtime = tmp_path / "runtime.py"
    runtime.write_text("VALUE = 1\n")
    _git(tmp_path, "add", "runtime.py")
    _git(tmp_path, "commit", "-qm", "bind runtime")
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert audit._require_clean_committed_files(tmp_path, ("runtime.py",)) == head

    runtime.write_text("VALUE = 2\n")
    with pytest.raises(audit.SupportIntegrityError, match="clean committed tree"):
        audit._require_clean_committed_files(tmp_path, ("runtime.py",))

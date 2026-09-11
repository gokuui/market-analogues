from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from market_analogues.future_mode_store import (
    ArrowPathIndex,
    FutureModeStoreError,
    canonical_json,
    install_shared_path_index,
    partition_payload,
    partition_query_ids,
    publish_partition,
    recover_stale_partition_temporaries,
    shared_episode_digest,
    validate_partition,
)


def path_row(episode: str, step: int) -> dict[str, object]:
    return {
        "benchmark_relative_close_return": step / 200,
        "close_return": step / 100,
        "contract_digest": "contract",
        "cutoff": "2024-01-31",
        "episode_id": episode,
        "expected_session_match": True,
        "source_content_digest": "content",
        "source_fingerprint": f"fingerprint-{episode}",
        "step": step,
        "timestamp": f"2024-02-{step:02d}",
    }


def write_paths(path: Path, episodes: tuple[str, ...] = ("a", "b")) -> None:
    pd.DataFrame([
        path_row(episode, step) for episode in episodes for step in range(1, 4)
    ]).to_parquet(path, index=False)


def result(query_id: str, value: int = 1) -> dict[str, object]:
    return {"query_case_id": query_id, "value": value}


def test_arrow_path_index_has_exact_contiguous_lookup(tmp_path: Path) -> None:
    path = tmp_path / "paths.parquet"
    write_paths(path)
    index = ArrowPathIndex.load(path)
    assert index.row_count == 6
    assert index.episode_count == 2
    assert index.image_bytes < path.stat().st_size * 20
    assert [row["step"] for row in index.rows("a")] == [1, 2, 3]
    assert [row["episode_id"] for row in index.rows("b")] == ["b"] * 3
    assert index.rows("missing") == ()


def test_preloaded_index_is_identical_in_serial_and_forked_workers(tmp_path: Path) -> None:
    path = tmp_path / "paths.parquet"
    write_paths(path, ("a", "b", "c"))
    install_shared_path_index(ArrowPathIndex.load(path))
    episode_ids = ("a", "missing", "c", "b")
    serial = tuple(map(shared_episode_digest, episode_ids))
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        parallel = tuple(pool.map(shared_episode_digest, episode_ids))
    assert parallel == serial


def test_arrow_path_index_refuses_noncontiguous_episode_and_symlink(tmp_path: Path) -> None:
    path = tmp_path / "paths.parquet"
    pd.DataFrame([
        path_row("a", 1), path_row("b", 1), path_row("a", 2),
    ]).to_parquet(path, index=False)
    with pytest.raises(FutureModeStoreError, match="not physically contiguous"):
        ArrowPathIndex.load(path)
    target = tmp_path / "target.parquet"
    write_paths(target)
    link = tmp_path / "link.parquet"
    link.symlink_to(target)
    with pytest.raises(FutureModeStoreError, match="unsafe"):
        ArrowPathIndex.load(link)

    null_path = tmp_path / "null.parquet"
    pd.DataFrame([path_row("a", 1), path_row(None, 2)]).to_parquet(
        null_path, index=False,
    )
    with pytest.raises(FutureModeStoreError, match="contains nulls"):
        ArrowPathIndex.load(null_path)


def test_partitioning_and_payload_are_deterministic() -> None:
    assert partition_query_ids(["d", "a", "c", "b", "e"], 2) == (
        ("a", "b", "c"), ("d", "e"),
    )
    left = partition_payload(
        0, ("a", "b"), [result("b", 2), result("a", 1)],
        contract_digest="contract",
    )
    right = partition_payload(
        0, ("a", "b"), [result("a", 1), result("b", 2)],
        contract_digest="contract",
    )
    assert canonical_json(left) == canonical_json(right)
    with pytest.raises(FutureModeStoreError, match="nonempty and unique"):
        partition_query_ids(["a", "a"], 2)
    with pytest.raises(FutureModeStoreError, match="sorted, nonempty and unique"):
        partition_payload(
            0, ("b", "a"), [result("a"), result("b")],
            contract_digest="contract",
        )


def test_atomic_partition_create_and_identical_resume(tmp_path: Path) -> None:
    query_ids = ("a", "b")
    rows = [result("a"), result("b")]
    assert publish_partition(
        tmp_path, 0, query_ids, rows, contract_digest="contract",
    ) == "created"
    original = (tmp_path / "partitions/partition-0000/PARTITION.json").read_bytes()
    assert publish_partition(
        tmp_path, 0, query_ids, list(reversed(rows)), contract_digest="contract",
    ) == "reused"
    assert (tmp_path / "partitions/partition-0000/PARTITION.json").read_bytes() == original
    value = validate_partition(
        tmp_path / "partitions/partition-0000", 0, query_ids,
        contract_digest="contract",
    )
    assert value["results_digest"]


def test_partial_partition_prefix_resumes_without_rewrite(tmp_path: Path) -> None:
    groups = partition_query_ids(["d", "c", "b", "a"], 2)
    publish_partition(
        tmp_path, 0, groups[0], [result(value) for value in groups[0]],
        contract_digest="contract",
    )
    first = tmp_path / "partitions/partition-0000/PARTITION.json"
    original = first.read_bytes()
    original_mtime = first.stat().st_mtime_ns
    states = []
    for partition_id, query_ids in enumerate(groups):
        states.append(publish_partition(
            tmp_path, partition_id, query_ids,
            [result(value) for value in query_ids], contract_digest="contract",
        ))
    assert states == ["reused", "created"]
    assert first.read_bytes() == original
    assert first.stat().st_mtime_ns == original_mtime


def test_crash_left_temporary_is_moved_aside_before_resume(tmp_path: Path) -> None:
    stale = tmp_path / "partitions/.partition-0000.crashed"
    stale.mkdir(parents=True)
    (stale / "PARTITION.json").write_text('{"incomplete":true}')
    recovered = recover_stale_partition_temporaries(tmp_path)
    assert len(recovered) == 1
    assert not stale.exists()
    assert (tmp_path / "interrupted" / recovered[0] / "PARTITION.json").is_file()
    assert publish_partition(
        tmp_path, 0, ("a",), [result("a")], contract_digest="contract",
    ) == "created"


def test_conflicting_partition_is_quarantined_and_refused(tmp_path: Path) -> None:
    query_ids = ("a",)
    publish_partition(
        tmp_path, 0, query_ids, [result("a")], contract_digest="contract",
    )
    path = tmp_path / "partitions/partition-0000/PARTITION.json"
    value = json.loads(path.read_text())
    value["results"][0]["value"] = 99
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(FutureModeStoreError, match="quarantined"):
        publish_partition(
            tmp_path, 0, query_ids, [result("a")], contract_digest="contract",
        )
    assert not (tmp_path / "partitions/partition-0000").exists()
    conflicts = list((tmp_path / "conflicts").iterdir())
    assert len(conflicts) == 1
    assert (conflicts[0] / "PARTITION.json").is_file()


def test_malformed_result_partition_is_quarantined_not_crashed(tmp_path: Path) -> None:
    target = tmp_path / "partitions/partition-0000"
    target.mkdir(parents=True)
    (target / "PARTITION.json").write_text(
        '{"results":[1],"schema_version":"m04r15-r2-mode-partition-v1"}\n'
    )
    with pytest.raises(FutureModeStoreError, match="quarantined"):
        publish_partition(
            tmp_path, 0, ("a",), [result("a")], contract_digest="contract",
        )
    assert len(list((tmp_path / "conflicts").iterdir())) == 1

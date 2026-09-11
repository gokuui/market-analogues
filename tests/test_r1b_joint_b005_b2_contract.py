from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import pytest

from experiments.m04r import m04r14_r1b_joint_b005_b2_contract as joint
from market_analogues.types import stable_hash


def test_bound_config_parses_the_authenticated_buffer_not_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "datasets.yaml"
    authenticated = b"datasets:\n  nasdaq:\n    path: authentic\n    format: parquet\n"
    path.write_bytes(b"datasets:\n  nasdaq:\n    path: replacement\n    format: csv\n")
    record = {"path": str(path), "bytes": len(authenticated),
              "sha256": sha256(authenticated).hexdigest()}
    monkeypatch.setattr(joint, "_immutable_file_bytes",
                        lambda value: (record, (1, 2), authenticated))
    actual_record, inode, raw, config = joint._bound_config(path, record["sha256"])
    assert actual_record == record and inode == (1, 2)
    assert raw["datasets"]["nasdaq"]["path"] == "authentic"
    assert config.datasets["nasdaq"].path == (tmp_path / "authentic").resolve()


def _git(root: Path, *args: str) -> str:
    run = subprocess.run(("git", *args), cwd=root, text=True, capture_output=True, check=True)
    return run.stdout.strip()


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, snapshot=None) -> dict:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    (root / "runtime.py").write_text("# synthetic frozen runtime\n")
    _git(root, "add", "runtime.py"); _git(root, "commit", "-qm", "H0")
    monkeypatch.setattr(joint, "runtime_paths", lambda _root: ("runtime.py",))
    monkeypatch.setattr(joint, "authority_snapshot", snapshot or (lambda _root: {"synthetic": True}))
    payload = joint.preregister(root)
    _git(root, "add", str(joint.PREREGISTRATION)); _git(root, "commit", "-qm", "H1")
    return payload


def _amend(root: Path, payload: dict) -> None:
    payload.pop("preregistration_digest", None)
    payload["preregistration_digest"] = stable_hash(payload)
    (root / joint.PREREGISTRATION).write_text(json.dumps(payload))
    _git(root, "add", str(joint.PREREGISTRATION)); _git(root, "commit", "--amend", "--no-edit", "-q")


def test_same_h1_supports_both_producers_and_geometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _fixture(tmp_path, monkeypatch)
    h1 = joint.validate_h1(tmp_path, payload)
    # Ignored result publication does not change H1, so later producers need no new commit.
    for name in joint.OUTPUTS:
        assert joint.validate_h1(tmp_path) == h1
    assert payload["specification"]["execution"]["all_producers_require_same_h1"] is True
    assert len(joint.FUTURE_VERIFIERS) == 2


@pytest.mark.parametrize("change", ["threshold", "null_name", "pool", "embargo", "extra", "population", "environment"])
def test_resigning_cannot_authorize_contract_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    payload = _fixture(tmp_path, monkeypatch)
    if change == "threshold":
        payload["specification"]["b2"]["gates"]["n1_both_p_le"] = .10
    elif change == "null_name":
        payload["specification"]["b2"]["priority_fields"].append("null_name")
    elif change == "pool":
        payload["specification"]["population"]["specificity_reference"] = "only primary 357"
    elif change == "embargo":
        payload["specification"]["embargo"]["forbidden_path_tokens"] = []
    elif change == "extra":
        payload["unregistered_option"] = True
    elif change == "population":
        payload["authorities"] = {"synthetic": False}
    else:
        payload["environment"]["numeric_dtype"] = "float16"
    _amend(tmp_path, payload)
    with pytest.raises(joint.JointContractError):
        joint.validate_h1(tmp_path)


def test_dirty_tree_and_post_h1_commit_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fixture(tmp_path, monkeypatch)
    (tmp_path / "new.py").write_text("# changed\n")
    with pytest.raises(joint.JointContractError, match="clean tree"):
        joint.validate_h1(tmp_path)
    _git(tmp_path, "add", "new.py"); _git(tmp_path, "commit", "-qm", "later verifier")
    with pytest.raises(joint.JointContractError, match="sole child"):
        joint.validate_h1(tmp_path)


def test_h1_extra_file_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fixture(tmp_path, monkeypatch)
    (tmp_path / "extra.txt").write_text("unauthorized H1 content\n")
    _git(tmp_path, "add", "extra.txt"); _git(tmp_path, "commit", "--amend", "--no-edit", "-q")
    with pytest.raises(joint.JointContractError, match="exactly one"):
        joint.validate_h1(tmp_path)


def test_frozen_runtime_mutation_rejected_even_if_h1_resigned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _fixture(tmp_path, monkeypatch)
    payload["runtime_sha256"]["runtime.py"] = "0" * 64
    _amend(tmp_path, payload)
    with pytest.raises(joint.JointContractError, match="runtime changed"):
        joint.validate_h1(tmp_path)


@pytest.mark.parametrize("path", ["../escape.json", "/tmp/absolute.json", "data/outcomes.json", "data/b003/RESULT.json",
                                  "data/r1b-b1/RESULT.json", "data/stockbee.json", "data/forward_return.json"])
def test_path_embargo_precedes_open_even_if_allowlisted(tmp_path: Path, path: str) -> None:
    with pytest.raises(joint.JointContractError):
        joint.safe_path(tmp_path, path, [path])


def test_symlink_ancestor_and_unlisted_path_rejected(tmp_path: Path) -> None:
    (tmp_path / "target").mkdir(); (tmp_path / "alias").symlink_to(tmp_path / "target", target_is_directory=True)
    with pytest.raises(joint.JointContractError, match="symlink"):
        joint.safe_path(tmp_path, "alias/file.json", ["alias/file.json"])
    with pytest.raises(joint.JointContractError, match="unapproved"):
        joint.safe_path(tmp_path, "other.json", ["allowed.json"])


def test_opaque_hex_id_containing_b004_is_not_mistaken_for_embargo_label(tmp_path: Path) -> None:
    relative = "cases/29136f9df2f7b614a93b004b.json"
    (tmp_path / "cases").mkdir()
    (tmp_path / relative).write_text("{}")
    assert joint.safe_path(tmp_path, relative, [relative]) == tmp_path / relative
    assert joint._embargoed_path("data/r1b-b004/RESULT.json")


def test_atomic_create_only_preserves_first_result_and_rejects_dangling_symlink(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    joint.atomic_json(path, {"first": True}); original = path.read_bytes()
    with pytest.raises(joint.JointContractError, match="create-only"):
        joint.atomic_json(path, {"second": True})
    assert path.read_bytes() == original
    alias = tmp_path / "alias.json"; alias.symlink_to(tmp_path / "absent.json")
    with pytest.raises(joint.JointContractError, match="create-only"):
        joint.atomic_json(alias, {"second": True})
    assert not list(tmp_path.glob(".*.tmp-*"))


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '[]'])
def test_strict_json_rejects_ambiguous_authorities(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "bad.json"; path.write_text(raw)
    with pytest.raises(joint.JointContractError):
        joint.load_json(path)


def _population_fixture() -> tuple:
    ids = [f"{i:024x}" for i in range(8)]
    episode = "f" * 24
    registry = [{"episode_id": q} for q in ids]
    cases = [{"query_episode_id": q, "query_symbol": "Q", "query_start": "2020-01-01",
              "latest_eligible_cutoff": "2020-01-01", "matches": [
                  {"episode_id": episode, "symbol": "E", "cutoff": "2019-01-01"}
              ] if i < 5 else []} for i, q in enumerate(ids)]
    cells = [{"query_episode_id": q, "base_cells": {"session_21": [1, "A", "high", 1, True]}} for q in ids]
    assignments = [{"query_episode_id": q, "labels": {"8": 0, "12": 0, "16": 0}} for q in ids]
    support = [{"episode_id": episode, "eligible_queries": 8, "observed_inbound_queries": 5,
                "n0_support": 56, "n1_support": {"8": 56, "12": 56, "16": 56}}]
    return registry, cases, cells, assignments, support


def test_population_reconstructs_eligibility_cells_counts_and_support_without_geometry() -> None:
    result = joint.derive_population(*_population_fixture())
    row = result["episodes"][0]
    assert row["observed_count"] == 5 and len(row["eligible_query_ids"]) == 8
    assert row["designs"]["N1"]["support_capped"] == 56
    assert result["primary_ids"] == []
    assert result["population_digest"] == stable_hash({k: v for k, v in result.items() if k != "population_digest"})


@pytest.mark.parametrize("change", ["duplicate", "support", "causal", "cohort", "cell"])
def test_population_mutations_fail_closed(change: str) -> None:
    registry, cases, cells, assignments, support = deepcopy(_population_fixture())
    if change == "duplicate":
        cases[0]["matches"].append(cases[0]["matches"][0])
    elif change == "support":
        support[0]["n1_support"]["8"] = 55
    elif change == "causal":
        cases[0]["latest_eligible_cutoff"] = "2018-01-01"
    elif change == "cohort":
        support[0]["episode_id"] = "e" * 24
    else:
        assignments[0]["labels"]["8"] = 1
    with pytest.raises(joint.JointContractError):
        joint.derive_population(registry, cases, cells, assignments, support)


def test_contract_has_explicit_negative_and_integrity_states() -> None:
    spec = joint.specification()
    assert "zero null cohesion" in spec["failure_taxonomy"]["not_established_pending_independent_verification"]
    assert "verifier failure" in spec["failure_taxonomy"]["unresolved"]
    assert spec["b2"]["gates"]["threshold_equality_passes"] is True
    assert "24 lowercase hexadecimal" in spec["b2"]["episode_id_format"]
    assert (spec["b005"]["shard_replicates"], spec["b005"]["shards"]) == (8, 64)
    assert (spec["geometry"]["chunk_rows"], spec["geometry"]["distance_shards"]) == (64, 52)
    assert (spec["b2"]["shard_replicates"], spec["b2"]["shards"]) == (32, 128)
    assert "null" not in spec["b2"]["priority_fields"]
    assert len(spec["b005"]["metrics"]) == 15
    assert all(value is False for key, value in spec["claims"].items() if key != "maximum_after_verified_pass")


def test_missing_future_runtime_fails_before_authority_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(joint, "clean_head", lambda _root: "h0")
    monkeypatch.setattr(joint, "authority_snapshot", lambda _root: pytest.fail("authorities read before required runtime"))
    with pytest.raises(joint.JointContractError, match="regular file"):
        joint.build_preregistration(tmp_path)


def test_frozen_b005_encoding_matches_core() -> None:
    from market_analogues.adequacy_shared_priority import priority_digest
    spec = joint.specification()["b005"]
    for domain in spec["domains"].values():
        domain_bytes = domain.encode("utf-8"); entity = "A:test".encode("utf-8")
        encoded = (len(domain_bytes).to_bytes(2, "big") + domain_bytes
                   + spec["seed"].to_bytes(8, "big") + (511).to_bytes(4, "big")
                   + len(entity).to_bytes(2, "big") + entity)
        assert priority_digest(domain=domain, seed=spec["seed"], replicate=511, identifier="A:test") == sha256(encoded).digest()


def test_frozen_b2_encoding_matches_core_without_digest_self_reference() -> None:
    from market_analogues.adequacy_localization import conditional_priority
    full = joint.specification(); spec = full["b2"]
    digest = stable_hash(full)
    assert "b2_priority_contract_digest" not in full
    for shared in (False, True):
        parts = [spec["priority_domain"].encode(), bytes.fromhex(digest), spec["priority_family"].encode(),
                 spec["priority_schemes"]["shared" if shared else "primary"].encode(), (4095).to_bytes(4, "big")]
        if not shared:
            parts.append(b"episode")
        parts.append(b"query")
        encoded = b"".join(len(part).to_bytes(4, "big") + part for part in parts)
        assert conditional_priority(digest, 4095, "episode", "query", shared_query=shared) == sha256(encoded).digest()


def _external_fixture(root: Path) -> tuple[list, dict, dict]:
    stock_root = root / "stocks"; stock_root.mkdir()
    for name in ("Q", "E", "UNUSED"):
        (stock_root / f"{name}.parquet").write_bytes(f"synthetic {name} bytes; deliberately not a parquet".encode())
    benchmark = root / "benchmark.parquet"; benchmark.write_bytes(b"synthetic benchmark")
    config_path = root / joint.DATASET_CONFIG; config_path.parent.mkdir()
    config_path.write_text(json.dumps({"datasets": {"nasdaq": {
        "adapter": "directory", "path": str(stock_root), "format": "parquet",
        "file_glob": "*.parquet", "symbol_from": "filename", "benchmark": {"path": str(benchmark)},
    }}}))
    query_rows = [{"episode_id": "000001", "symbol": "Q"}]
    population = {"query_ids": ["000001"], "cohort_ids": ["eeeeee"],
                  "episodes": [{"episode_id": "eeeeee", "symbol": "E"}]}
    lock = {"config_path": str(config_path), "config_sha256": joint.file_sha(config_path),
            "benchmark_sha256": joint.file_sha(benchmark)}
    lock["source_lock_digest"] = stable_hash(lock)
    return query_rows, population, lock


def _resign_lock(root: Path, lock: dict) -> None:
    lock["config_sha256"] = joint.file_sha(root / joint.DATASET_CONFIG)
    lock.pop("source_lock_digest", None); lock["source_lock_digest"] = stable_hash(lock)


def test_external_manifest_exact_union_without_decoding_bars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_analogues import adapters
    fixture = _external_fixture(tmp_path)
    monkeypatch.setattr(adapters, "_read", lambda *_args: pytest.fail("source bars decoded during contract freeze"))
    manifest = joint.external_source_manifest(tmp_path, *fixture)
    assert manifest["required_symbols"] == ["E", "Q"]
    assert manifest["stock_files_count"] == 2
    assert [row["symbol"] for row in manifest["stock_files"]] == ["E", "Q"]
    assert all("UNUSED" not in row["path"] for row in manifest["stock_files"])
    assert manifest["ohlcv_decoded"] is manifest["geometry_materialized"] is False
    assert manifest["manifest_digest"] == stable_hash({key: value for key, value in manifest.items() if key != "manifest_digest"})
    assert manifest == joint.external_source_manifest(tmp_path, *fixture)


@pytest.mark.parametrize("change", ["stock", "benchmark", "config", "symbol"])
def test_external_content_mutations_cannot_preserve_h1_manifest(tmp_path: Path, change: str) -> None:
    query_rows, population, lock = _external_fixture(tmp_path)
    original = joint.external_source_manifest(tmp_path, query_rows, population, lock)
    if change == "stock":
        (tmp_path / "stocks/E.parquet").write_bytes(b"rewritten source")
        assert joint.external_source_manifest(tmp_path, query_rows, population, lock) != original
    elif change == "benchmark":
        (tmp_path / "benchmark.parquet").write_bytes(b"rewritten benchmark")
        with pytest.raises(joint.JointContractError, match="benchmark source authority changed"):
            joint.external_source_manifest(tmp_path, query_rows, population, lock)
    elif change == "config":
        config = tmp_path / joint.DATASET_CONFIG; config.write_text(config.read_text() + "\n")
        with pytest.raises(joint.JointContractError, match="configuration hash"):
            joint.external_source_manifest(tmp_path, query_rows, population, lock)
    else:
        population["episodes"][0]["symbol"] = "UNUSED"
        assert joint.external_source_manifest(tmp_path, query_rows, population, lock) != original


@pytest.mark.parametrize("change", ["missing", "symlink_file", "symlink_directory", "duplicate_inode", "duplicate_stem", "long_table", "column_source"])
def test_external_manifest_refuses_ambiguous_resolution(tmp_path: Path, change: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_analogues import adapters
    query_rows, population, lock = _external_fixture(tmp_path)
    config_path = tmp_path / joint.DATASET_CONFIG
    config = json.loads(config_path.read_text())
    if change == "missing":
        (tmp_path / "stocks/E.parquet").unlink()
    elif change == "symlink_file":
        (tmp_path / "stocks/E.parquet").unlink()
        (tmp_path / "stocks/E.parquet").symlink_to(tmp_path / "stocks/UNUSED.parquet")
    elif change == "symlink_directory":
        alias = tmp_path / "alias"; alias.symlink_to(tmp_path / "stocks", target_is_directory=True)
        # Relative config paths must be checked before load_config resolves them.
        config["datasets"]["nasdaq"]["path"] = str(alias)
    elif change == "duplicate_inode":
        (tmp_path / "stocks/E.parquet").unlink()
        os.link(tmp_path / "stocks/Q.parquet", tmp_path / "stocks/E.parquet")
    elif change == "duplicate_stem":
        nested = tmp_path / "stocks/nested"; nested.mkdir()
        (nested / "E.parquet").write_bytes(b"ambiguous source")
        config["datasets"]["nasdaq"]["file_glob"] = "**/*.parquet"
    elif change == "long_table":
        config["datasets"]["nasdaq"]["adapter"] = "long_table"
    else:
        config["datasets"]["nasdaq"]["symbol_from"] = "column"
    config_path.write_text(json.dumps(config)); _resign_lock(tmp_path, lock)
    monkeypatch.setattr(adapters, "_read", lambda *_args: pytest.fail("ambiguous source decoded"))
    with pytest.raises(joint.JointContractError):
        joint.external_source_manifest(tmp_path, query_rows, population, lock)


def test_external_manifest_refuses_source_change_during_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "source.parquet"; path.write_bytes(b"initial source")
    original_read = os.read
    modified = False
    def changing_read(descriptor: int, size: int) -> bytes:
        nonlocal modified
        block = original_read(descriptor, size)
        if not modified:
            modified = True
            path.write_bytes(b"changed and longer source")
        return block
    monkeypatch.setattr(joint.os, "read", changing_read)
    with pytest.raises(joint.JointContractError, match="changed while hashing"):
        joint._immutable_file_record(path)


def test_b005_work_counters_freeze_hash_reuse_and_both_shared_orders() -> None:
    spec = joint.specification()["b005"]
    per = spec["work_counters_per_replicate"]
    assert per["episode_hashes"] == 3786156 and per["symbol_hashes"] == 11584
    assert per["global_orders"] == per["hierarchical_orders"] == 1
    assert per["per_query_full_universe_hashes"] == per["per_query_full_universe_sorts"] == 0
    assert per["query_risk_sets"] == 3270
    assert per["global_risk_set_filters"] == 3270
    assert per["hierarchical_risk_set_filters"] == 3270
    assert per["total_risk_set_filters"] == 6540
    assert spec["work_counters_complete_512"] == {key: value * 512 for key, value in per.items()}
    assert "integer session coordinates" in spec["intervals"]


def test_h1_rejects_external_file_drift_with_clean_git_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    external = tmp_path / "external"; external.mkdir()
    repository = tmp_path / "repository"; repository.mkdir()
    metadata = _external_fixture(external)
    snapshot = lambda _root: {"external_source_manifest": joint.external_source_manifest(external, *metadata)}
    _fixture(repository, monkeypatch, snapshot=snapshot)
    joint.validate_h1(repository)
    (external / "stocks/Q.parquet").write_bytes(b"changed external data while repository stays clean")
    assert _git(repository, "status", "--porcelain") == ""
    with pytest.raises(joint.JointContractError, match="authority/population differs"):
        joint.validate_h1(repository)

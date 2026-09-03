from pathlib import Path
import fcntl
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.m04r import m04r14_t14_10_wf03_combined_batch as subject
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03b_dtw_component_ladder as ladder


def preregistration() -> dict:
    return {
        "preregistration_digest": "pre",
        "contract": {"digest": "contract", "schema_version": "staged"},
        "inputs": {
            "packed_content_digest": "1" * 64,
            "dtw_semantic_identity_digest": "2" * 64,
        },
        "inventory": {
            "queries": 2, "scored_queries": 2, "warmup_queries": 0,
            "months": 1,
        },
    }


def row() -> dict:
    return {
        "episode_id": "a" * 24, "case_id": "case-a", "symbol": "AAA",
        "cutoff": "2020-01-31T00:00:00", "fold_id": "development",
        "fold_role": "development", "scored": True,
    }


def resource_observation() -> dict:
    return {
        "effective_cpus": subject.MINIMUM_EFFECTIVE_CPUS,
        "memory_total_kib": subject.MINIMUM_TOTAL_MEMORY_KIB,
        "memory_available_kib": subject.MINIMUM_AVAILABLE_MEMORY_KIB,
        "process_rss_kib": 100, "process_peak_rss_kib": 100,
        "process_swap_kib": 0,
    }


def test_runtime_freeze_covers_direct_and_transitive_inputs() -> None:
    assert {
        "config/datasets.example.yaml",
        "experiments/m04r/m04r14_t14_10_wf03b_dtw_component_ladder.py",
        "pyproject.toml",
        "src/market_analogues/causal_prefix.py",
        "src/market_analogues/config.py",
        "src/market_analogues/context.py",
        "src/market_analogues/distance.py",
        "src/market_analogues/episodes.py",
        "src/market_analogues/quantized_bound.py",
        "src/market_analogues/structural.py",
        "src/market_analogues/types.py",
    }.issubset(subject.RUNTIME_FILES)
    assert set(subject._runtime_environment()["packages"]) \
        == set(subject.RUNTIME_PACKAGES)


@pytest.mark.parametrize(("field", "value", "message"), (
    ("effective_cpus", 7, "cpus="),
    ("memory_total_kib", 1, "total-memory-kib="),
    ("memory_available_kib", 1, "available-memory-kib="),
    ("process_swap_kib", 1, "process-swap-kib="),
))
def test_resource_preflight_fails_early_with_exact_observation(
    monkeypatch, field: str, value: int, message: str,
) -> None:
    observation = resource_observation()
    observation[field] = value
    monkeypatch.setattr(subject, "_resource_observation", lambda: observation)
    with pytest.raises(subject.CombinedBatchError, match=message):
        subject._require_resources("test-stage")


def valid_case() -> dict:
    certificate = {
        "schema_version": "staged",
        "query_episode_id": "a" * 24,
        "packed_generation_id": base.GENERATION_ID,
        "dtw_generation_id": ladder.DTW_GENERATION_ID,
        "contract_digest": "contract",
        "input_digest": "input",
        "eligible_candidates": 100,
        "seed_rows": 100,
        "rigid_bound_evaluated": 100,
        "rigid_bound_admitted": 80,
        "dtw_bound_evaluated": 75,
        "combined_bound_admitted": 30,
        "exact_evaluated": 100,
        "native_bound_pruned": 0,
        "maximum_bound_excess": 0.0,
        "seed_threshold": 1.0,
        "final_threshold": 0.19,
        "minimum_rigid_pruned": 1.1,
        "minimum_combined_pruned": 1.2,
    }
    value = {
        "schema_version": "m04r14-wf03-combined-batch-case-v5",
        "status": "complete", "query_id": "a" * 24, "case_id": "case-a",
        "symbol": "AAA", "cutoff": "2020-01-31T00:00:00",
        "fold_id": "development", "fold_role": "development", "scored": True,
        "preregistration_digest": "pre",
        "packed_generation_id": base.GENERATION_ID,
        "packed_content_digest": "1" * 64,
        "dtw_generation_id": ladder.DTW_GENERATION_ID,
        "dtw_semantic_identity_digest": "2" * 64,
        "attempt_id": "attempt-0001",
        "resident_attempt_lease_digest": "3" * 64,
        "dtw_attempt_identity_digest": "4" * 64,
        "contract_digest": "contract", "certificate": certificate,
        "matches": [{
            "episode_id": f"{number:024x}", "symbol": f"S{number:02d}",
            "cutoff": "2019-01-01T00:00:00",
            "distance_hex": float(number / 100).hex(), "quality_tier": "A",
        } for number in range(20)],
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "process_swap_kib": 0,
        "resource_observation": resource_observation(),
    }
    certificate["result_digest"] = subject._certificate_result_digest(
        certificate, value["matches"]
    )
    value["semantic_digest"] = base.stable_hash(
        subject._case_semantic_state(value)
    )
    return base._sealed(value, "case_digest")


def test_case_path_accepts_only_episode_ids(tmp_path: Path) -> None:
    assert subject._case_path(tmp_path, "b" * 24) == tmp_path / f"{'b' * 24}.json"
    with pytest.raises(subject.CombinedBatchError, match="query ID"):
        subject._case_path(tmp_path, "../unsafe")


def test_valid_case_closes_all_structural_and_semantic_gates() -> None:
    value = valid_case()
    assert subject._validate_case(value, row(), preregistration()) == value


@pytest.mark.parametrize("mutation", (
    lambda value: value["matches"].append(dict(value["matches"][0])),
    lambda value: value["matches"][1].update(symbol="S00"),
    lambda value: value["certificate"].update(final_threshold=1.1),
    lambda value: value["certificate"].update(minimum_rigid_pruned=1.0),
    lambda value: value.update(outcomes_or_labels_used=True),
    lambda value: value.update(preregistration_digest="changed"),
    lambda value: value.update(process_swap_kib=1),
))
def test_case_validation_rejects_semantic_drift(mutation) -> None:
    value = valid_case()
    value.pop("case_digest")
    mutation(value)
    value["semantic_digest"] = base.stable_hash(
        subject._case_semantic_state(value)
    )
    value = base._sealed(value, "case_digest")
    with pytest.raises(subject.CombinedBatchError, match="case differs"):
        subject._validate_case(value, row(), preregistration())


def test_existing_invalid_receipt_is_refused_not_replaced(tmp_path: Path) -> None:
    path = subject._case_path(tmp_path, row()["episode_id"])
    path.write_text("{}")
    with pytest.raises(Exception):
        subject._existing_case(path, row(), preregistration())


def test_dtw_file_identity_detects_metadata_change(tmp_path: Path, monkeypatch) -> None:
    generation = tmp_path / "generations" / ladder.DTW_GENERATION_ID
    generation.mkdir(parents=True)
    for name in ("manifest.json", "dtw-samples.bin", "dtw-overflow-samples.bin"):
        (generation / name).write_bytes(name.encode())
    monkeypatch.setattr(ladder, "DTW_ROOT_RELATIVE", Path("."))
    first = subject._dtw_physical_identity(tmp_path)
    (generation / "dtw-samples.bin").write_bytes(b"changed")
    second = subject._dtw_physical_identity(tmp_path)
    assert first["digest"] != second["digest"]


def test_case_semantics_survive_new_physical_attempt_identity() -> None:
    first = valid_case()
    second = dict(first)
    second.pop("case_digest")
    second["attempt_id"] = "attempt-0002"
    second["resident_attempt_lease_digest"] = "5" * 64
    second["dtw_attempt_identity_digest"] = "6" * 64
    second["semantic_digest"] = base.stable_hash(
        subject._case_semantic_state(second)
    )
    second = base._sealed(second, "case_digest")
    assert second["semantic_digest"] == first["semantic_digest"]
    assert subject._validate_case(second, row(), preregistration()) == second


def test_terminal_result_crash_window_is_validated_and_reconciled(
    tmp_path: Path,
) -> None:
    root = tmp_path / "batch"
    cases = root / "cases"
    cases.mkdir(parents=True)
    first = valid_case()
    second = dict(first)
    second.pop("case_digest")
    second.update({
        "query_id": "b" * 24, "case_id": "case-b", "symbol": "BBB",
    })
    second["certificate"] = dict(second["certificate"])
    second["certificate"]["query_episode_id"] = "b" * 24
    second["certificate"]["result_digest"] = subject._certificate_result_digest(
        second["certificate"], second["matches"],
    )
    second["semantic_digest"] = base.stable_hash(
        subject._case_semantic_state(second)
    )
    second = base._sealed(second, "case_digest")
    for value in (first, second):
        base._atomic(cases / f"{value['query_id']}.json", value)
    results = [first, second]
    case_manifest = [{
        "query_id": value["query_id"], "case_digest": value["case_digest"],
        "sha256": base._sha(cases / f"{value['query_id']}.json"),
    } for value in results]
    state = {
        "schema_version": "m04r14-t14-10-wf03-combined-batch-result-v5",
        "status": "complete", "passed": True, "queries": 2,
        "scored_queries": 2, "warmup_queries": 0, "months": 1,
        "preregistration_digest": "pre",
        "packed_generation_id": base.GENERATION_ID,
        "packed_content_digest": "1" * 64,
        "dtw_generation_id": ladder.DTW_GENERATION_ID,
        "case_manifest_digest": base.stable_hash(case_manifest),
        "case_semantic_digest": base.stable_hash([
            value["semantic_digest"] for value in results
        ]),
        "minimum_eligible_candidates": 100,
        "maximum_eligible_candidates": 100,
        "maximum_seed_rows": 100, "expanded_seed_queries": 0,
        "preload_seconds": 1.0,
        "post_preload_resource_observation": resource_observation(),
        "peak_rss_mb": 1.0, "source_cache_state": {},
        "prepared_symbols": 2,
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
        "independent_verification_authorized": True,
        "terminal_attempt_id": "attempt-0001", "attempts": 1,
    }
    result = base._sealed(state)
    attempts = {"attempt-0001": {
        "resident_attempt_lease_digest": "3" * 64,
        "dtw_attempt_identity_digest": "4" * 64,
    }}
    assert subject._validate_terminal_result(
        result, results, preregistration(), root, attempts,
    ) == result
    attempt = root / "attempts" / "attempt-0001"
    attempt.mkdir(parents=True)
    base._atomic(attempt / "RUN_STARTED.json", base._sealed({
        "schema_version": "m04r14-wf03-combined-batch-attempt-v5",
        "status": "running", "attempt_id": "attempt-0001",
        "preregistration_digest": "pre",
        "packed_content_digest": "1" * 64,
        "resident_mode": "restored-after-restart",
        "dtw_validation_mode": "full-content-and-record-validation",
        "resident_ready_digest": "5" * 64,
        "resident_attempt_lease_digest": "3" * 64,
        "resident_validation_observation_digest": "6" * 64,
        "dtw_semantic_identity_digest": "2" * 64,
        "dtw_attempt_identity_digest": "4" * 64,
        "receipts_reused_at_start": 0,
        "initial_resource_observation": resource_observation(),
        "after_resident_resource_observation": resource_observation(),
        "created_at": "2026-01-01T00:00:00+00:00",
    }, "attempt_digest"))
    subject._reconcile_terminal_publication(root, result, results, attempts)
    complete = base._read(attempt / "COMPLETE.json")
    assert complete["result_digest"] == result["result_digest"]
    assert complete["receipts_computed"] == 2
    attempts["attempt-0001"].update({
        "_terminal_name": "COMPLETE.json", "_terminal": complete,
    })
    subject._reconcile_terminal_publication(root, result, results, attempts)
    assert base._read(root / "PROGRESS.json")["status"] == "complete"
    history = subject._attempt_history(root, preregistration())
    assert history["attempt-0001"]["_terminal_name"] == "COMPLETE.json"


def test_attempt_history_ignores_only_empty_crash_window_directory(
    tmp_path: Path,
) -> None:
    attempts = tmp_path / "attempts"
    (attempts / "attempt-0001").mkdir(parents=True)
    assert subject._attempt_history(tmp_path, preregistration()) == {}
    (attempts / "attempt-0001" / "unexpected").write_text("partial")
    with pytest.raises(subject.CombinedBatchError, match="files differ"):
        subject._attempt_history(tmp_path, preregistration())


def test_resident_restore_uses_content_not_previous_physical_identity(
    tmp_path: Path, monkeypatch,
) -> None:
    resident = tmp_path / "resident"
    content_digest = "a" * 64
    modes = []

    def prepare(*args, validate_existing, **kwargs):
        modes.append(validate_existing)
        resident.mkdir(parents=True, exist_ok=True)
        (resident / "READY.json").write_text("{}")
        return {"content_digest": content_digest}, {"observation_digest": "b" * 64}

    lease_number = iter(("c" * 64, "d" * 64, "e" * 64))
    monkeypatch.setattr(base, "RESIDENT_ROOT", resident)
    monkeypatch.setattr(subject, "prepare_resident_mirror_observed", prepare)
    monkeypatch.setattr(subject, "observe_ready_strict", lambda path: {
        "content_digest": content_digest, "ready_digest": "e" * 64,
        "ready_file_sha256": "f" * 64,
    })
    monkeypatch.setattr(subject, "resident_file_identity_lease", lambda path: {
        "content_digest": content_digest, "lease_digest": next(lease_number),
    })
    first = subject._ensure_resident(tmp_path, content_digest)
    second = subject._ensure_resident(tmp_path, content_digest)
    assert modes == [False, True]
    assert first["lease"]["lease_digest"] != second["lease"]["lease_digest"]
    assert first["content_digest"] == second["content_digest"] == content_digest


def test_resident_resume_reuses_exact_previously_validated_lease(
    tmp_path: Path, monkeypatch,
) -> None:
    resident = tmp_path / "resident"
    resident.mkdir()
    (resident / "READY.json").write_text("{}")
    content, ready, lease = "a" * 64, "b" * 64, "c" * 64
    monkeypatch.setattr(base, "RESIDENT_ROOT", resident)
    monkeypatch.setattr(subject, "observe_ready_strict", lambda path: {
        "content_digest": content, "ready_digest": ready,
        "ready_file_sha256": "d" * 64,
    })
    monkeypatch.setattr(subject, "resident_file_identity_lease", lambda path: {
        "content_digest": content, "lease_digest": lease,
    })
    monkeypatch.setattr(subject, "prepare_resident_mirror_observed", lambda *a, **k:
                        pytest.fail("unchanged validated lease must not be rehashed"))
    result = subject._ensure_resident(tmp_path, content, {"attempt-0001": {
        "packed_content_digest": content,
        "resident_ready_digest": ready,
        "resident_attempt_lease_digest": lease,
    }})
    assert result["mode"] == "reused-fully-validated-unchanged-lease"
    assert result["validation_observation"]["trusted_attempt_id"] \
        == "attempt-0001"


def test_execute_resumes_sealed_cases_under_a_new_attempt_lease(
    tmp_path: Path, monkeypatch,
) -> None:
    rows = [row(), {**row(), "episode_id": "b" * 24, "case_id": "case-b",
                    "symbol": "BBB"}]
    pre = preregistration()
    output = Path("output-v5")
    leases = iter(("3" * 64, "5" * 64))
    current_lease = {"value": ""}
    startup_options = []
    fail_second = True

    class FakeSource:
        def __init__(self, raw, max_entries=None):
            pass

        def preload(self, instruments, workers):
            pass

        def cache_state(self):
            return {"entries": 2, "max_entries": None}

    def case_for(current, attempt_id, lease, dtw_identity):
        value = valid_case()
        value.pop("case_digest")
        value.update({
            "query_id": current["episode_id"], "case_id": current["case_id"],
            "symbol": current["symbol"], "attempt_id": attempt_id,
            "resident_attempt_lease_digest": lease,
            "dtw_attempt_identity_digest": dtw_identity,
        })
        value["certificate"]["query_episode_id"] = current["episode_id"]
        value["certificate"]["result_digest"] = subject._certificate_result_digest(
            value["certificate"], value["matches"]
        )
        value["semantic_digest"] = base.stable_hash(
            subject._case_semantic_state(value)
        )
        return base._sealed(value, "case_digest")

    def run_case(current, **kwargs):
        nonlocal fail_second
        path = subject._case_path(kwargs["cases_root"], current["episode_id"])
        existing = subject._existing_case(path, current, pre)
        if existing is not None:
            return existing
        if current["episode_id"] == "b" * 24 and fail_second:
            fail_second = False
            raise RuntimeError("simulated restart")
        value = case_for(
            current, kwargs["attempt_id"], kwargs["resident_lease_digest"],
            kwargs["dtw_identity_digest"],
        )
        base._atomic(path, value)
        return value

    def ensure_resident(repository, expected, trusted_attempts=None):
        lease = next(leases)
        current_lease["value"] = lease
        return {
            "store_root": str(tmp_path), "content_digest": expected,
            "mode": "validated-existing", "ready_digest": "6" * 64,
            "lease": {"lease_digest": lease, "content_digest": expected},
            "validation_observation": {"observation_digest": "7" * 64},
        }

    monkeypatch.setattr(subject, "OUTPUT_RELATIVE", output)
    monkeypatch.setattr(subject, "validate_preregistration", lambda repo, value: (
        {"queries_data": rows}, {item["episode_id"]: item for item in rows},
    ))
    monkeypatch.setattr(subject, "_ensure_resident", ensure_resident)
    monkeypatch.setattr(subject, "_dtw_physical_identity", lambda repo: {
        "digest": "4" * 64,
    })
    monkeypatch.setattr(subject, "resident_file_identity_lease", lambda path: {
        "lease_digest": current_lease["value"],
    })
    monkeypatch.setattr(subject, "load_packed_generation", lambda *a, **k: (
        startup_options.append(("packed", k["validate_records"])),
        SimpleNamespace(manifest={}),
    )[1])
    monkeypatch.setattr(subject, "load_dtw_sample_generation", lambda *a, **k:
                        startup_options.append(("dtw", k["verify_content"])))
    monkeypatch.setattr(subject, "load_config", lambda path:
                        SimpleNamespace(datasets={"nasdaq": object()}))
    monkeypatch.setattr(subject, "source_from_spec", lambda spec:
                        SimpleNamespace(instruments=lambda: ("AAA", "BBB")))
    monkeypatch.setattr(subject, "CachedOHLCVSource", FakeSource)
    monkeypatch.setattr(subject, "_run_case", run_case)
    monkeypatch.setattr(subject, "_require_resources", lambda stage:
                        resource_observation())
    monkeypatch.setattr(subject, "_resource_observation", resource_observation)

    with pytest.raises(RuntimeError, match="simulated restart"):
        subject.execute(tmp_path, pre)
    first_path = subject._case_path(tmp_path / output / "cases", "a" * 24)
    first_digest = base._read(first_path)["case_digest"]
    result = subject.execute(tmp_path, pre)
    first = base._read(first_path)
    second = base._read(subject._case_path(
        tmp_path / output / "cases", "b" * 24,
    ))
    assert result["queries"] == 2
    assert first["case_digest"] == first_digest
    assert first["attempt_id"] == "attempt-0001"
    assert second["attempt_id"] == "attempt-0002"
    assert first["resident_attempt_lease_digest"] \
        != second["resident_attempt_lease_digest"]
    assert startup_options == [
        ("packed", True), ("dtw", True),
        ("packed", True), ("dtw", False),
    ]


def test_execute_refuses_a_concurrent_producer_before_validation(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(subject, "OUTPUT_RELATIVE", Path("output-v5"))
    lock = tmp_path / "output-v5.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(subject.CombinedBatchError, match="already running"):
            subject.execute(tmp_path, preregistration())
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

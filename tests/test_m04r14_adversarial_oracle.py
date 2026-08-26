from __future__ import annotations

import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/m04r/m04r14_adversarial_oracle.py"
SPEC = importlib.util.spec_from_file_location("m04r14_adversarial_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def _runtime_history(root: Path) -> tuple[dict[str, object], Path, str]:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "oracle@example.invalid")
    _git(root, "config", "user.name", "Oracle Test")
    runtime = Path("runtime.py")
    content = b"RUNTIME = 'h0'\n"
    (root / runtime).write_bytes(content)
    _git(root, "add", str(runtime))
    _git(root, "commit", "-q", "-m", "runtime H0")
    h0 = _git(root, "rev-parse", "HEAD")
    state = {
        "mode": "production-clean-head", "git_head": h0,
        "files": {str(runtime): sha256(content).hexdigest()},
        "contracts": oracle._contract_binding(),
        "environment": oracle._environment_binding(),
    }
    return {"state": state, "digest": oracle.stable_hash(state)}, runtime, h0


@pytest.fixture(scope="module")
def payload(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("m04r14-oracle")
    return oracle.build_oracle_payload(root / "work", oracle.test_runtime_manifest())


def test_oracle_forces_all_adversarial_states(payload: dict[str, object]) -> None:
    assert payload["passed"] is True
    assert payload["truth_free"] is True
    assert payload["real_forward_outcomes_accessed"] is False
    cardinality = payload["cardinality_and_boundaries"]
    assert set(cardinality["eligible_result_digests"]) == {"0", "19", "20", "25"}
    assert cardinality["duplicate_id_rejected"] == 1
    assert len(cardinality["upper_boundary_episode_id"]) == 24
    certified_cardinality = payload["certified_cardinality"]["state"]["rows"]
    assert certified_cardinality["0"]["status"] == "underfill-rejected"
    assert certified_cardinality["19"]["status"] == "underfill-rejected"
    assert certified_cardinality["20"]["status"] == "certified"
    assert len(certified_cardinality["20"]["matches"]["state"]["rows"]) == 20
    exact = payload["certified_vs_exhaustive"]
    assert exact["eligible_rows"] > 20
    assert exact["retained_main"] > 0
    assert exact["retained_overflow"] == 1
    assert exact["top_k"] == 3
    assert exact["tied_at_k"] > exact["top_k"]
    assert len(exact["per_symbol"]) == 3
    assert set(exact["per_symbol"].values()) == {1}
    assert exact["overlap_rejected"] > 0
    assert exact["cap_rejected"] > 0
    assert exact["closure_passes"] > 0
    assert exact["native_bound_pruned"] > 0
    assert exact["mutation_rejected"] is True
    assert exact["source_mutation_rejected"] == 1
    assert exact["store_mutation_rejected"] == 1
    assert exact["strict_finite_json"] is True
    properties = payload["randomized_property_matrix"]
    assert properties["fixed_seeds"] == [14_101, 14_102, 14_103, 14_104]
    assert len(properties["cases"]) == 8
    assert {row["configuration"] for row in properties["cases"]} == {
        "unconstrained", "constrained",
    }
    assert all(row["eligible_rows"] >= 20 for row in properties["cases"])
    assert any(row["sparse_missing_volume"] for row in properties["cases"])
    source_digests = {row["source"]["digest"] for row in properties["cases"]}
    input_digests = {
        row["certificate"]["state"]["input_digest"]
        for row in properties["cases"]
    }
    assert len(source_digests) == 4
    assert len(input_digests) == 8
    assert any(
        row["source"]["state"]["volume_nan_rows"]["S3"] > 0
        for row in properties["cases"]
    )
    assert all(
        len(row["matches"]["state"]["rows"])
        == row["certificate"]["state"]["rounds"][-1]["selected_rows"]
        for row in properties["cases"]
        if not row["certificate"]["state"]["threshold_closure_passes"]
    )
    assert sum(row["repeated_frontier_rounds"] for row in properties["cases"]) > 0
    assert payload["numeric_comparison"]["absolute_tolerance_hex"] == (1e-6).hex()
    assert all(value > 0 for value in payload["branch_hits"].values())
    json.dumps(payload, allow_nan=False)


def test_certified_raw_and_constraint_underfill_fail_closed(
    payload: dict[str, object],
) -> None:
    assert payload["certified_vs_exhaustive"]["underfill_rejections"] == {
        "raw": 1, "constraint": 1,
    }
    assert payload["branch_hits"]["raw_underfill_rejected"] == 1
    assert payload["branch_hits"]["constraint_underfill_rejected"] == 1


def test_max_per_instrument_underfill_completion_is_forced(
    payload: dict[str, object],
) -> None:
    reopening = payload["max_per_instrument_underfill_completion"]
    assert reopening["completed_rows"] == 1
    assert float.fromhex(reopening["raw_second_bound_hex"]) == 99.0
    assert float.fromhex(reopening["resulting_threshold_hex"]) == 100.0
    assert payload["branch_hits"]["max_per_completion_from_underfill"] == 1


def test_run_is_create_only_and_payload_is_canonical(
    tmp_path: Path, payload: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "evidence" / "oracle.json"
    output.parent.mkdir()
    monkeypatch.setattr(
        oracle, "build_oracle_payload", lambda *_args, **_kwargs: payload,
    )
    returned = oracle.run(
        output, runtime_manifest=oracle.test_runtime_manifest(),
        enforce_production=False,
    )
    assert json.loads(output.read_text()) == returned
    assert returned["result_digest"] == oracle.stable_hash({
        key: value for key, value in returned.items()
        if key != "result_digest"
    })
    oracle.validate_payload(returned, require_production=False)
    with pytest.raises(FileExistsError):
        oracle.run(
            output, runtime_manifest=oracle.test_runtime_manifest(),
            enforce_production=False,
        )


def test_cli_validate_is_independent_and_strict(
    tmp_path: Path, payload: dict[str, object],
) -> None:
    absent = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    assert absent.returncode != 0
    output = tmp_path / "oracle.json"
    output.write_text(json.dumps(payload, sort_keys=True))
    subprocess.run(
        [sys.executable, str(SCRIPT), "validate", "--input", str(output),
         "--allow-test-manifest"], cwd=ROOT,
        check=True, text=True, capture_output=True,
    )
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":2}')
    rejected = subprocess.run(
        [sys.executable, str(SCRIPT), "validate", "--input", str(duplicate),
         "--allow-test-manifest"], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    assert rejected.returncode != 0
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}')
    with pytest.raises(oracle.OracleError, match="nonfinite"):
        oracle._load_strict_json(nonfinite)
    input_alias = tmp_path / "input-alias.json"
    input_alias.symlink_to(output)
    with pytest.raises(oracle.OracleError, match="symlink"):
        oracle._load_strict_json(input_alias)


def test_output_rejects_symlinked_and_lexically_aliased_ancestry(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(oracle.OracleError, match="symlink|alias"):
        oracle.run(
            alias / "oracle.json", runtime_manifest=oracle.test_runtime_manifest(),
            enforce_production=False,
        )
    lexical_alias = real / ".." / "real" / "oracle.json"
    with pytest.raises(oracle.OracleError, match="alias"):
        oracle.run(
            lexical_alias, runtime_manifest=oracle.test_runtime_manifest(),
            enforce_production=False,
        )
    assert not (real / "oracle.json").exists()


def test_production_output_and_manifest_are_not_caller_overridable(
    tmp_path: Path, payload: dict[str, object],
) -> None:
    assert oracle._production_output(ROOT) == ROOT / oracle.CANONICAL_OUTPUT
    with pytest.raises(oracle.OracleError, match="fixes output"):
        oracle.run(output=tmp_path / "forbidden.json", enforce_production=True)
    with pytest.raises(oracle.OracleError, match="production runtime manifest"):
        oracle.validate_payload(
            payload,
            require_production=True,
        )


def test_runtime_h0_survives_unrelated_clean_h1(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    manifest, runtime, _ = _runtime_history(repo)
    (repo / "unrelated.txt").write_text("later unrelated evidence\n")
    _git(repo, "add", "unrelated.txt")
    _git(repo, "commit", "-q", "-m", "unrelated H1")
    oracle._validate_production_manifest_against_repo(
        manifest, repo, runtime_files=(runtime,),
    )


def test_runtime_manifest_covers_every_tracked_market_analogue_module() -> None:
    expected = {
        "experiments/m04r/m04r14_adversarial_oracle.py",
        *_git(ROOT, "ls-files", "--", "src/market_analogues/*.py").splitlines(),
    }
    assert {str(path) for path in oracle.RUNTIME_FILES} == expected
    environment = oracle._environment_binding()
    assert environment["digest"] == oracle.stable_hash({
        key: value for key, value in environment.items() if key != "digest"
    })
    assert set(environment["packages"]) == {"numpy", "pandas", "numba", "pyarrow"}


def test_runtime_h0_rejects_indirect_dependency_h1(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "oracle@example.invalid")
    _git(repo, "config", "user.name", "Oracle Test")
    runtime = Path("runtime.py")
    dependency = Path("indirect.py")
    (repo / runtime).write_text("from indirect import VALUE\n")
    (repo / dependency).write_text("VALUE = 1\n")
    _git(repo, "add", str(runtime), str(dependency))
    _git(repo, "commit", "-q", "-m", "runtime H0")
    h0 = _git(repo, "rev-parse", "HEAD")
    state = {
        "mode": "production-clean-head", "git_head": h0,
        "files": {
            str(path): sha256((repo / path).read_bytes()).hexdigest()
            for path in (runtime, dependency)
        },
        "contracts": oracle._contract_binding(),
        "environment": oracle._environment_binding(),
    }
    manifest = {"state": state, "digest": oracle.stable_hash(state)}
    (repo / dependency).write_text("VALUE = 2\n")
    _git(repo, "add", str(dependency))
    _git(repo, "commit", "-q", "-m", "indirect dependency drift H1")
    with pytest.raises(oracle.OracleError, match="HEAD runtime blob drifted"):
        oracle._validate_production_manifest_against_repo(
            manifest, repo, runtime_files=(runtime, dependency),
        )


def test_runtime_h0_rejects_committed_and_worktree_blob_drift(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    manifest, runtime, _ = _runtime_history(repo)
    (repo / runtime).write_text("RUNTIME = 'dirty'\n")
    with pytest.raises(oracle.OracleError, match="globally clean"):
        oracle._validate_production_manifest_against_repo(
            manifest, repo, runtime_files=(runtime,),
        )
    _git(repo, "add", str(runtime))
    _git(repo, "commit", "-q", "-m", "runtime drift H1")
    with pytest.raises(oracle.OracleError, match="HEAD runtime blob drifted"):
        oracle._validate_production_manifest_against_repo(
            manifest, repo, runtime_files=(runtime,),
        )


def test_runtime_h0_rejects_nonancestor_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    manifest, runtime, _ = _runtime_history(repo)
    _git(repo, "checkout", "-q", "--orphan", "unrelated-history")
    _git(repo, "rm", "-q", "--cached", str(runtime))
    (repo / runtime).write_text("RUNTIME = 'h0'\n")
    _git(repo, "add", str(runtime))
    _git(repo, "commit", "-q", "-m", "unrelated root")
    with pytest.raises(oracle.OracleError, match="not an ancestor"):
        oracle._validate_production_manifest_against_repo(
            manifest, repo, runtime_files=(runtime,),
        )


@pytest.mark.parametrize("kind", oracle.MUTATION_KINDS)
def test_independent_validator_rejects_rehashed_mutation(
    payload: dict[str, object], kind: str,
) -> None:
    changed = oracle.rehashed_mutation(payload, kind)
    with pytest.raises(oracle.OracleError):
        oracle.validate_payload(changed, require_production=False)


def test_failure_precedes_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "oracle.json"
    monkeypatch.setattr(
        oracle, "build_oracle_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(oracle.OracleError("fault")),
    )
    with pytest.raises(oracle.OracleError, match="fault"):
        oracle.run(
            output, runtime_manifest=oracle.test_runtime_manifest(),
            enforce_production=False,
        )
    assert not output.exists()


@pytest.mark.parametrize("fault", ("exit", "hang"))
def test_child_crash_or_timeout_never_publishes(
    tmp_path: Path, fault: str,
) -> None:
    output = tmp_path / f"{fault}.json"
    program = """
import importlib.util
import os
from pathlib import Path
import sys
import time
spec = importlib.util.spec_from_file_location("oracle_child", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if sys.argv[3] == "exit":
    module.build_oracle_payload = lambda *_a, **_k: os._exit(17)
else:
    module.build_oracle_payload = lambda *_a, **_k: time.sleep(60)
module.run(
    Path(sys.argv[2]), runtime_manifest=module.test_runtime_manifest(),
    enforce_production=False,
)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(SCRIPT), str(output), fault],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if fault == "exit":
        assert process.wait(timeout=5) == 17
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.5)
        process.kill()
        process.wait(timeout=5)
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*"))


def test_double_build_is_deterministic(
    tmp_path: Path, payload: dict[str, object],
) -> None:
    second = oracle.build_oracle_payload(
        tmp_path / "second-work", oracle.test_runtime_manifest(),
    )
    assert second == payload


def test_numeric_policy_accepts_boundary_and_rejects_nextafter() -> None:
    key = oracle.EpisodeKey(
        oracle.InstrumentKey(oracle.DATASET, "BOUNDARY"),
        oracle.pd.Timestamp("2024-01-31"), 10, "dense-v1",
    )

    def match(distance: float) -> SimpleNamespace:
        return SimpleNamespace(
            episode_key=key, total_distance=distance,
            component_distances={"price": distance}, alignment=((0, 0),),
        )

    expected = match(1.0)
    oracle._assert_match_parity(
        [match(1.0 + oracle.NUMERIC_ATOL)], [expected],
    )
    above = float(np.nextafter(1.0 + oracle.NUMERIC_ATOL, np.inf))
    with pytest.raises(oracle.OracleError, match="distance differs"):
        oracle._assert_match_parity([match(above)], [expected])

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from types import SimpleNamespace

import pandas as pd
import pytest

import test_m04r13_threaded_certified_exposed as support
import test_m04r13_finite_threshold_diagnostic as diagnostic_support


REPOSITORY = Path(__file__).resolve().parents[1]
VERIFIER_PATH = (
    REPOSITORY / "experiments/m04r/verify_m04r13_threaded_certified_exposed.py"
)


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("m04r13_independent_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load()


def test_independent_verifier_runtime_manifest_matches_producer() -> None:
    assert verifier.REQUIRED_RUNTIME_FILES == frozenset(
        support.producer.RUNTIME_FILES
    )


def test_independent_certificate_rejects_zero_pruned_with_native_minimum(
    tmp_path: Path,
) -> None:
    root = tmp_path / "producer"
    support._producer_fixture(root)
    query_id = verifier.QUERY_IDS[0]
    case = support.producer._read_json(
        root / "cases" / f"00-{query_id}.json"
    )
    certificate = case["certificate"]
    certificate["minimum_native_pruned_bound"] = 20.0
    certificate["next_lower_bound"] = 20.0
    certificate["stopped_early"] = True
    certificate["rounds"][-1]["next_lower_bound"] = 20.0
    certificate["result_digest"] = support._rehash_certificate(
        certificate, case["matches"],
    )
    forward = verifier._validate_proposal(
        case["forward_proposal"], query_id,
        case["forward_proposal"]["input_digest"], "forward",
    )
    with pytest.raises(verifier.VerificationError):
        verifier._validate_certificate(
            certificate, case["matches"], query_id,
            certificate["input_digest"], forward,
            case["forward_proposal"]["eligible_rows"],
        )


def test_independent_diagnostic_binding_uses_one_exact_byte_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "producer"
    prereg, _authorities = support._producer_fixture(root)
    resident = support.producer._read_json(root / "RESIDENT.json")
    cases = [support.producer._read_json(
        root / "cases" / f"{ordinal:02d}-{query_id}.json"
    ) for ordinal, query_id in enumerate(verifier.QUERY_IDS)]
    summaries = [
        diagnostic_support.diagnostic.summarize_case(case) for case in cases
    ]
    for summary in summaries:
        summary["rounds"][0]["frontier_rows"] = verifier.INITIAL_FRONTIER
        summary["result_digest"] = verifier.stable_hash(verifier._without(
            summary, {"result_digest"},
        ))
    config = tmp_path / verifier.CONFIG_RELATIVE
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("synthetic: true\n")
    payload = diagnostic_support.diagnostic.diagnostic_payload(
        repository=tmp_path, implementation_git=prereg["git"],
        registry_digest=verifier.REGISTRY_DIGEST,
        registry_cases_digest=prereg["registry_cases_digest"],
        resident=resident, cases=summaries,
    )
    path = tmp_path / verifier.DIAGNOSTIC_RELATIVE
    _write(path, payload)
    prereg.update({
        "git": prereg["git"], "environment": payload["environment"],
        "execution": payload["execution"],
        "config_sha256": payload["config_sha256"],
        "registry_cases_digest": payload["registry_cases_digest"],
        "resident_content_digest": resident["content_digest"],
        "resident_ready_digest": resident["ready_digest"],
        "resident_identity_digest": resident["identity_digest"],
        "finite_threshold_diagnostic": {
            "path": str(path.resolve()), "sha256": _sha(path),
            "result_digest": payload["result_digest"],
        },
    })
    monkeypatch.setattr(
        verifier, "_query_bindings",
        lambda _prereg: {
            query_id: cases[ordinal]["query_binding"]
            for ordinal, query_id in enumerate(verifier.QUERY_IDS)
        },
    )
    real_read = verifier._read_json_sha
    reads = 0

    def one_read(value: Path) -> tuple[dict[str, Any], str]:
        nonlocal reads
        reads += 1
        return real_read(value)

    monkeypatch.setattr(verifier, "_read_json_sha", one_read)
    verifier._validate_finite_diagnostic(prereg, tmp_path)
    assert reads == 1


@pytest.mark.parametrize("protected_name", (
    "source", "resident", "registry", "config.yaml", "preregistered.json",
))
def test_verifier_output_cannot_mutate_input_ancestry(
    tmp_path: Path, protected_name: str,
) -> None:
    protected = tmp_path / protected_name
    if protected.suffix:
        protected.write_text("input")
        output = protected / "verification"
    else:
        protected.mkdir()
        output = protected / "verification"
    with pytest.raises(verifier.VerificationError, match="overlaps"):
        verifier._fresh_output(output, (protected,))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _universe(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for ordinal, query_id in enumerate(verifier.QUERY_IDS):
        case = support.producer._read_json(
            root / "cases" / f"{ordinal:02d}-{query_id}.json"
        )
        proposals = {
            row["episode_id"]: row
            for row in case["forward_proposal"]["candidates"]
        }
        output[query_id] = {}
        for match in case["matches"]:
            proposal = proposals[match["episode_id"]]
            output[query_id][match["episode_id"]] = {
                "symbol": proposal["symbol"],
                "cutoff_ns": proposal["cutoff_ns"],
                "quality_tier": proposal["quality_tier"],
                "overflow_fallback": proposal["overflow_fallback"],
                "lower_bound": float.fromhex(proposal["lower_bound_hex"]),
                "eligible": True,
            }
    return output


def _terminal_fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "evidence"
    prereg, authorities = support._producer_fixture(root)
    support.comparator.compare(
        root, tmp_path / "unused-authority", tmp_path / "unused-verification.json",
        truth_loader=lambda *_args: authorities,
        **support._truth_free_test_kwargs(root, prereg),
    )
    authority_root = tmp_path / "authority"
    bindings: dict[str, tuple[str, str]] = {}
    for query_id, payload in authorities.items():
        path = authority_root / "cases" / f"{query_id}.json"
        _write(path, payload)
        bindings[query_id] = (_sha(path), payload["result_digest"])
    authority_verification = tmp_path / "authority-verification.json"
    _write(authority_verification, {
        "schema_version": "m04r11-certified-authority-verification-v4",
        "result_digest": verifier.AUTHORITY_VERIFICATION_DIGEST,
        "passed": True, "authority_correctness_passed": True,
        "production_promotion_authorized": False,
        "real_forward_outcomes_accessed": False,
        "authority_matrix_digest": verifier.AUTHORITY_MATRIX_DIGEST,
        "authority_seal_digest": verifier.AUTHORITY_SEAL_DIGEST,
        "generation_id": verifier.GENERATION_ID,
    })
    universe = _universe(root)
    return {
        "repository": REPOSITORY, "producer_root": root,
        "authority_root": authority_root,
        "authority_verification": authority_verification,
        "output_root": tmp_path / "verification",
        "repository_prereg": prereg, "enforce_topology": False,
        "git_validator": lambda *_args: None,
        "diagnostic_git_validator": lambda *_args: None,
        "config_validator": lambda *_args: None,
        "prereg_validator": lambda *_args, **_kwargs: None,
        "environment_loader": lambda: prereg["environment"],
        "binding_loader": lambda _prereg: support._fixture_bindings(root),
        "universe_loader": lambda _prereg, _cases: universe,
        "proposal_reconstructor": lambda *_args: None,
        "closure_reconstructor": lambda *_args: None,
        "source_identity_loader": lambda _root: {"stable": True},
        "resident_observer": lambda _prereg: support.producer._read_json(
            root / "RESIDENT.json"
        ),
        "verification_sha256": _sha(authority_verification),
        "verification_digest": verifier.AUTHORITY_VERIFICATION_DIGEST,
        "authority_case_bindings": bindings,
    }


def test_independent_verifier_reconstructs_terminal_fixture(tmp_path: Path) -> None:
    kwargs = _terminal_fixture(tmp_path)
    payload = verifier.verify(**kwargs)
    assert payload["verification_passed"] is True
    assert payload["experiment_passed"] is True
    assert set(path.name for path in kwargs["output_root"].iterdir()) == {
        "verification.json", "verification.html",
    }


def test_verifier_rejects_copied_producer_tree(tmp_path: Path) -> None:
    kwargs = _terminal_fixture(tmp_path)
    copied = tmp_path / "copied-evidence"
    shutil.copytree(kwargs["producer_root"], copied)
    kwargs["producer_root"] = copied
    with pytest.raises(verifier.VerificationError, match="ownership root"):
        verifier.verify(**kwargs)
    assert not kwargs["output_root"].exists()


def test_verifier_rejects_universe_metadata_tampering(tmp_path: Path) -> None:
    kwargs = _terminal_fixture(tmp_path)
    original = kwargs["universe_loader"](None, None)
    query_id = verifier.QUERY_IDS[0]
    identifier = next(iter(original[query_id]))
    original[query_id][identifier]["symbol"] = "TAMPERED"
    kwargs["universe_loader"] = lambda *_args: original
    with pytest.raises(verifier.VerificationError, match="universe identity"):
        verifier.verify(**kwargs)
    assert not kwargs["output_root"].exists()


def test_authority_bytes_are_hash_parsed_in_one_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _terminal_fixture(tmp_path)
    protected = {
        str(kwargs["authority_verification"]),
        *(str(kwargs["authority_root"] / "cases" / f"{value}.json")
          for value in verifier.QUERY_IDS),
    }
    real_open = verifier.os.open
    opens = {value: 0 for value in protected}

    def counted(path: Any, *args: Any, **inner: Any) -> int:
        key = str(path)
        if key in opens:
            opens[key] += 1
        return real_open(path, *args, **inner)

    monkeypatch.setattr(verifier.os, "open", counted)
    verifier.verify(**kwargs)
    assert set(opens.values()) == {1}


def test_verifier_rejects_existing_output_without_mutation(tmp_path: Path) -> None:
    kwargs = _terminal_fixture(tmp_path)
    kwargs["output_root"].mkdir()
    sentinel = kwargs["output_root"] / "sentinel"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(verifier.VerificationError, match="fresh"):
        verifier.verify(**kwargs)
    assert sentinel.read_bytes() == b"preserve"


def test_verifier_rejects_source_generation_identity_change(tmp_path: Path) -> None:
    kwargs = _terminal_fixture(tmp_path)
    observations = iter(({"identity": 1}, {"identity": 2}))
    kwargs["source_identity_loader"] = lambda _root: next(observations)
    with pytest.raises(verifier.VerificationError, match="identity changed"):
        verifier.verify(**kwargs)
    assert not kwargs["output_root"].exists()


def test_verifier_has_no_producer_or_comparator_import() -> None:
    source = VERIFIER_PATH.read_text()
    assert "import m04r13_threaded_certified_exposed" not in source
    assert "import compare_m04r13_threaded_certified_exposed" not in source
    assert "from experiments.m04r.m04r13" not in source


def test_independent_verifier_accepts_canonical_naive_source_cutoffs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    support._producer_fixture(root)
    query_id = verifier.QUERY_IDS[0]
    case = support.producer._read_json(root / "cases" / f"00-{query_id}.json")
    for match in case["matches"]:
        match["cutoff"] = pd.Timestamp(match["cutoff"]).tz_localize(None).isoformat()
    verifier._validate_matches(case["matches"])


def _mock_query_context(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = [{
        "episode_id": query_id, "symbol": f"Q{ordinal}",
        "cutoff": "2020-01-01", "lookback": 252,
        "representation_version": "v1",
    } for ordinal, query_id in enumerate(verifier.QUERY_IDS)]
    monkeypatch.setattr(verifier, "_validate_registry", lambda _root: (registry, "digest"))
    monkeypatch.setattr(
        verifier, "load_config",
        lambda _path: SimpleNamespace(datasets={"nasdaq": object()}),
    )
    monkeypatch.setattr(verifier, "source_from_spec", lambda _spec: object())
    by_symbol = {
        row["symbol"]: row["episode_id"] for row in registry
    }

    def episode(_source: Any, instrument: Any, *_args: Any) -> Any:
        return SimpleNamespace(
            key=SimpleNamespace(
                id=by_symbol[instrument.source_symbol], instrument=instrument,
            ),
            bars=pd.DataFrame({
                "timestamp": [pd.Timestamp("2020-01-01", tz="UTC")],
            }),
        )

    monkeypatch.setattr(verifier, "build_episode", episode)
    monkeypatch.setattr(verifier, "represent", lambda _episode: object())
    monkeypatch.setattr(
        verifier, "latest_eligible_cutoff",
        lambda *_args: pd.Timestamp("2021-01-01", tz="UTC"),
    )


def test_arbitrary_closure_exclusion_digest_fails_before_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_query_context(monkeypatch)
    candidates = [
        {"episode_id": f"{index:024x}"}
        for index in range(verifier.MAXIMUM_FRONTIER)
    ]
    cases: list[dict[str, Any]] = [{
        "forward_proposal": {"candidates": candidates},
        "certificate": {"threshold_closure_passes": [{
            "excluded_prefix_digest": "a" * 64,
        }]},
    }]
    cases.extend({
        "forward_proposal": {"candidates": []},
        "certificate": {"threshold_closure_passes": []},
    } for _ in range(3))
    monkeypatch.setattr(
        verifier, "scan_packed_bound_threshold",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid exclusion digest must fail before scanning")
        ),
    )
    prereg = {
        "roots": {
            "registry_root": str(tmp_path / "registry"),
            "source_full_root": str(tmp_path / "source"),
        },
        "config_path": str(tmp_path / "config.yaml"),
    }
    with pytest.raises(verifier.VerificationError, match="exclusion digest"):
        verifier._reconstruct_closure_scans(prereg, cases)


def test_independent_proposal_replay_detects_candidate_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_query_context(monkeypatch)
    root = tmp_path / "evidence"
    prereg, _authorities = support._producer_fixture(root)
    cases = [support.producer._read_json(
        root / "cases" / f"{ordinal:02d}-{query_id}.json"
    ) for ordinal, query_id in enumerate(verifier.QUERY_IDS)]
    by_query = {
        case["query_episode_id"]: json.loads(json.dumps(case)) for case in cases
    }

    def scan(_root: Path, _generation: str, query: Any, **_kwargs: Any) -> Any:
        assert "threads" not in _kwargs
        value = by_query[query.episode_id]["forward_proposal"]
        rows = tuple(verifier.BoundProposal(
            row["episode_id"], row["symbol"], row["cutoff_ns"],
            row["quality_tier"], float.fromhex(row["lower_bound_hex"]),
            tuple(row["routes"]), row["overflow_fallback"],
        ) for row in value["candidates"])
        return SimpleNamespace(
            **{key: value[key] for key in (
                "schema_version", "generation_id", "query_episode_id",
                "rows_scanned", "eligible_rows", "eligible_main_rows",
                "eligible_overflow_rows", "route_counts", "route_quotas",
                "candidate_digest", "result_digest", "contract_digest",
                "input_digest",
            )},
            candidates=rows, block_rows=4_093, block_order="reverse",
            elapsed_seconds=1.0, peak_rss_mb=1.0,
        )

    monkeypatch.setattr(verifier, "scan_packed_bound_proposals", scan)
    verifier._reconstruct_proposals(prereg, cases)
    cases[0]["forward_proposal"]["candidates"][0]["symbol"] = "DRIFT"
    with pytest.raises(verifier.VerificationError, match="proposal replay"):
        verifier._reconstruct_proposals(prereg, cases)

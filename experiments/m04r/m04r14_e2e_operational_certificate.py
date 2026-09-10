"""Bind sealed operational evidence to the current portable release path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.m04r import verify_m04r14_performance_qualification as performance
from market_analogues.adapters import source_from_spec
from market_analogues.authority import AuthorityError, load_authority_artifact
from market_analogues.config import load_config
from market_analogues.gates import GateReport
from market_analogues.nse_e2e_verification import validate_nse_e2e_result_payload
from market_analogues.types import stable_hash


SCHEMA = "m04r14-e2e-operational-certificate-v1"
OUTPUT = Path("config/data/analogues/portability/e2e-operational-certificate-v1")
AUTHORITY = Path("config/data/analogues/portability/nse-current-authorities-v1")
AUTHORITY_VERIFICATION = Path(
    "config/data/analogues/portability/"
    "nse-current-authorities-v1-verification/VERIFIED.json"
)
NSE_E2E = Path(
    "config/data/analogues/portability/nse-real-e2e-verification-v5"
)
PORTABLE_E2E = Path(
    "config/data/analogues/portability/portable-e2e-verification-v1/RESULT.json"
)
PERFORMANCE_RESULT = Path(
    "config/data/analogues/m04r14/performance-qualification-v1/RESULT.json"
)
PERFORMANCE_VERIFICATION = Path(
    "config/data/analogues/m04r14/"
    "performance-qualification-v1-verification/VERIFIED.json"
)
FAULT = Path(
    "config/data/analogues/m04r14/performance-operational-fault-v1/RESULT.json"
)
RESOURCE = Path(
    "config/data/analogues/m04r14/performance-resource-monitor-v1/RESULT.json"
)


class OperationalCertificateError(RuntimeError):
    pass


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise OperationalCertificateError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise OperationalCertificateError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                OperationalCertificateError(f"non-finite JSON: {path}:{token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperationalCertificateError(f"invalid JSON: {path}") from exc
    if type(value) is not dict:
        raise OperationalCertificateError(f"JSON object required: {path}")
    return value, raw


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _semantic_receipt(
    value: Mapping[str, Any], *, omitted: set[str],
) -> bool:
    return value.get("result_digest") == stable_hash({
        key: item for key, item in value.items() if key not in omitted
    })


def _corruption_refused(path: Path) -> bool:
    """Exercise the current authority loader against a tampered real artifact."""
    with tempfile.TemporaryDirectory(prefix="market-analogues-corruption-") as name:
        target = Path(name) / "authority.json"
        shutil.copy2(path, target)
        payload = json.loads(target.read_text())
        payload["matches"][0]["total_distance"] += 1.0
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        try:
            load_authority_artifact(target)
        except AuthorityError:
            return True
    return False


def _parse_html(paths: Sequence[Path]) -> bool:
    try:
        for path in paths:
            parser = HTMLParser()
            parser.feed(path.read_text())
            parser.close()
    except Exception:
        return False
    return True


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def execute(repository: Path, config_path: Path) -> dict[str, Any]:
    started = perf_counter()
    repository = repository.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    output = repository / OUTPUT
    if output.exists() or output.is_symlink():
        raise OperationalCertificateError("operational certificate root exists")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository,
        capture_output=True, text=True, check=True,
    ).stdout
    if status:
        raise OperationalCertificateError("operational certificate requires clean Git")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    config = load_config(config_path)

    # Re-run the already committed independent T14-05 reconstruction.  This is
    # read-only and avoids inventing a second performance truth after results.
    performance_state = performance.verify(
        repository / "config/data/analogues/m04r14/performance-qualification-v1",
        repository=repository,
    )
    performance_receipt, performance_receipt_raw = _read(
        repository / PERFORMANCE_VERIFICATION
    )
    performance_result, performance_result_raw = _read(repository / PERFORMANCE_RESULT)
    receipt_without_time = {
        key: value for key, value in performance_receipt.items() if key != "created_at"
    }
    performance_bound = receipt_without_time == performance_state

    authority_verification, authority_verification_raw = _read(
        repository / AUTHORITY_VERIFICATION
    )
    authority_verification_valid = all((
        _semantic_receipt(
            authority_verification,
            omitted={"result_digest", "elapsed_seconds", "created_at"},
        ),
        authority_verification.get("passed") is True,
        authority_verification.get("verified_cases") == 12,
        authority_verification.get("verified_matches") == 240,
        authority_verification.get("all_case_gates_passed") is True,
        authority_verification.get("outcomes_accessed") is False,
    ))
    nse_result, nse_result_raw = _read(repository / NSE_E2E / "RESULT.json")
    nse_failures = validate_nse_e2e_result_payload(nse_result)
    nse_bound = all((
        not nse_failures, nse_result.get("passed") is True,
        nse_result.get("authority_verification_digest")
        == authority_verification.get("result_digest"),
        nse_result.get("cases") == 12, nse_result.get("exact_matches") == 240,
        nse_result.get("evidence_rows") == 720,
    ))
    portable_result, portable_raw = _read(repository / PORTABLE_E2E)
    portable_state = {
        key: value for key, value in portable_result.items()
        if key not in {"passed", "result_digest", "failures"}
    }
    portable_state["failures"] = list(portable_result.get("failures") or [])
    portable_bound = all((
        portable_result.get("passed") is True,
        portable_result.get("result_digest") == stable_hash(portable_state),
        portable_result.get("retrieval_semantics_equal") is True,
        portable_result.get("future_mutation_retrieval_invariant") is True,
    ))

    fault, fault_raw = _read(repository / FAULT)
    resource, resource_raw = _read(repository / RESOURCE)
    cases: list[dict[str, Any]] = []
    repeated_equal = True
    resume_equal = True
    strict_stopping = True
    case_paths = sorted((repository / AUTHORITY / "cases").glob("*.json"))
    for path in case_paths:
        value, raw = _read(path)
        certificate = dict(value.get("certificate") or {})
        repeated = dict(value.get("repeated_certificate") or {})
        build = dict(value.get("frontier_build") or {})
        resume = dict(value.get("frontier_resume") or {})
        repeated_case = all((
            value.get("result_digest") == value.get("repeated_digest"),
            certificate.get("manifest_digest") == repeated.get("manifest_digest"),
            all(
                certificate.get(key) == repeated.get(key)
                for key in certificate if key != "elapsed_seconds"
            ),
        ))
        resume_case = all((
            build.get("passed") is True, resume.get("passed") is True,
            build.get("manifest_digest") == resume.get("manifest_digest"),
            int(resume.get("instruments_built", -1)) == 0,
            int(resume.get("instruments_reused", -1))
            == int(build.get("instruments_built", -2)),
        ))
        stop_case = all((
            certificate.get("stopped_early") is True,
            int(certificate.get("exact_evaluated", -1))
            + int(certificate.get("safely_pruned", -1))
            == int(certificate.get("eligible_candidates", -2)),
            float(certificate.get("next_lower_bound", -1))
            > float(certificate.get("stop_threshold", 0)),
        ))
        repeated_equal &= repeated_case
        resume_equal &= resume_case
        strict_stopping &= stop_case
        cases.append({
            "episode_id": path.stem, "sha256": _sha(raw),
            "result_digest": value.get("result_digest"),
            "manifest_digest": certificate.get("manifest_digest"),
            "repeated_equal": repeated_case, "checkpoint_resume_equal": resume_case,
            "strict_stopping": stop_case,
        })

    report_paths = sorted((repository / NSE_E2E).rglob("*.html"))
    no_temporary_files = not any(
        path.suffix == ".tmp" or path.name.endswith(".tmp")
        for root in (
            repository / AUTHORITY, repository / AUTHORITY_VERIFICATION,
            repository / NSE_E2E,
        )
        for path in root.rglob("*")
    )
    source = source_from_spec(config.datasets["nse"])
    benchmark = source.load_benchmark()
    current_benchmark_end = (
        pd.Timestamp(benchmark.timestamp.max()).isoformat()
        if benchmark is not None and len(benchmark) else None
    )
    prefix_cutoff = pd.Timestamp(nse_result["prefix_lock_cutoff"])
    append_isolation_observed = bool(
        current_benchmark_end is not None
        and pd.Timestamp(current_benchmark_end) > prefix_cutoff
        and nse_result.get("source_lock_unchanged") is True
    )
    effective_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else (os.cpu_count() or 1)
    producer_result, _ = _read(repository / AUTHORITY / "RESULT.json")
    progress, _ = _read(repository / AUTHORITY / "PROGRESS.json")
    producer_started = datetime.fromisoformat(
        json.loads((repository / (
            "experiments/m04r/m04r14_e2e_nse_authority_matrix_preregistered.json"
        )).read_text())["created_at"]
    )
    producer_finished = datetime.fromisoformat(str(progress["updated_at"]))
    resource_observation = {
        "effective_cpus": effective_cpus,
        "fresh_authority_workers": int(producer_result["workers"]),
        "fresh_authority_elapsed_from_preregistration_seconds": (
            producer_finished - producer_started
        ).total_seconds(),
        "fresh_authority_tree_bytes": _tree_bytes(repository / AUTHORITY),
        "fresh_verifier_elapsed_seconds": float(
            authority_verification["elapsed_seconds"]
        ),
        "fresh_run_peak_rss_recorded": False,
        "applicable_sealed_core_resource_tree_peak_rss_kib": int(
            resource["maximum_tree_rss_kib"]
        ),
        "applicable_sealed_core_resource_tree_swap_kib": int(
            resource["maximum_tree_swap_kib"]
        ),
        "applicable_sealed_core_oom_delta": int(
            resource["memory_events_delta"]["oom"]
        ),
        "resource_interpretation": (
            "Fresh NSE RSS was observed live but not durably recorded, so no exact "
            "fresh-run RSS claim is made; the independently sealed core monitor applies."
        ),
    }
    corruption_refused = bool(case_paths) and _corruption_refused(case_paths[0])
    gates = {
        "sealed_performance_reverified": performance_bound,
        "sealed_performance_all_gates": (
            performance_result.get("passed") is True
            and all((performance_result.get("gates") or {}).values())
        ),
        "forced_sigkill_failed_closed": all((
            fault.get("passed") is True, fault.get("forced_signal") == "SIGKILL",
            fault.get("same_root_retry_rejected") is True,
            (fault.get("restart_verification") or {}).get("passed") is True,
        )),
        "concurrency_semantics_equal": all(
            row.get("all_semantics_equal") is True
            for row in performance_result.get("concurrency") or []
        ) and len(performance_result.get("concurrency") or []) == 4,
        "resource_monitor_passed": (
            resource.get("passed") is True
            and all((resource.get("gates") or {}).values())
        ),
        "fresh_authority_independently_verified": authority_verification_valid,
        "fresh_nse_e2e_reconstructible": nse_bound,
        "portable_cross_adapter_bound": portable_bound,
        "current_case_inventory": len(cases) == 12,
        "strict_stopping_all_cases": strict_stopping,
        "repeated_search_all_cases": repeated_equal,
        "checkpoint_resume_all_cases": resume_equal,
        "corrupt_current_artifact_refused": corruption_refused,
        "atomic_trees_no_temporary_files": no_temporary_files,
        "all_current_reports_parse": len(report_paths) == 13 and _parse_html(report_paths),
        "future_benchmark_append_isolated": append_isolation_observed,
    }
    state = {
        "schema_version": SCHEMA, "status": "complete",
        "passed": all(gates.values()), "git_head": git_head,
        "performance_result_sha256": _sha(performance_result_raw),
        "performance_verification_sha256": _sha(performance_receipt_raw),
        "performance_verification_digest": performance_receipt["result_digest"],
        "fault_sha256": _sha(fault_raw), "resource_sha256": _sha(resource_raw),
        "authority_verification_sha256": _sha(authority_verification_raw),
        "authority_verification_digest": authority_verification["result_digest"],
        "nse_e2e_sha256": _sha(nse_result_raw),
        "nse_e2e_result_digest": nse_result["result_digest"],
        "portable_e2e_sha256": _sha(portable_raw),
        "portable_e2e_result_digest": portable_result["result_digest"],
        "cases": cases, "case_manifest_digest": stable_hash(cases),
        "nse_result_validation_failures": list(nse_failures),
        "current_benchmark_end": current_benchmark_end,
        "prefix_lock_cutoff": prefix_cutoff.isoformat(),
        "resource_observation": resource_observation,
        "gates": gates, "failures": [name for name, passed in gates.items() if not passed],
        "production_authorized": False,
    }
    state["result_digest"] = stable_hash(state)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        **state, "elapsed_seconds": perf_counter() - started,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target = output / "RESULT.json"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in gates.items()
    )
    (output / "report.html").write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>E2E operational certificate</title><style>body{font-family:system-ui;"
        "max-width:1100px;margin:2rem auto;padding:0 1rem}table{border-collapse:"
        "collapse;width:100%}th,td{padding:.55rem;border-bottom:1px solid #ddd}"
        ".pass{color:#176b37}</style></head><body><h1>E2E-04 operational and "
        f"performance certificate</h1><p class=\"pass\"><b>{'PASS' if state['passed'] else 'FAIL'}</b>"
        "</p><p>This certificate binds the independently sealed restart, concurrency "
        "and resource qualification to the current portable and real-NSE release "
        f"path.</p><table><tbody>{rows}</tbody></table><h2>Resources</h2><pre>"
        f"{escape(json.dumps(resource_observation, indent=2))}</pre><p>Result digest: "
        f"<code>{state['result_digest']}</code>.</p><p><b>Boundary:</b> performance "
        "and operational correctness are certified; predictive or trading value is "
        "not.</p></body></html>"
    )
    GateReport(
        "e2e_04_operational_performance", state["passed"],
        {
            "schema_version": SCHEMA, "result_digest": state["result_digest"],
            "machine_artifact": str(target.resolve()),
            "html_artifact": str((output / "report.html").resolve()),
            "gates": gates, "resource_observation": resource_observation,
        },
        state["failures"],
    ).write(config.artifact_dir / "gates")
    if not state["passed"]:
        raise OperationalCertificateError(
            "operational certificate failed: " + ", ".join(state["failures"])
        )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = execute(args.repository, args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

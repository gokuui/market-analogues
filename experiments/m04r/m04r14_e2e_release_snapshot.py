"""Publish the final E2E release snapshot from a clean, tested repository."""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Any, Sequence

from market_analogues.gates import GateReport
from market_analogues.nse_e2e_verification import validate_nse_e2e_result_payload
from market_analogues.types import stable_hash


SCHEMA = "m04r14-e2e-release-snapshot-v1"
OUTPUT = Path("config/data/analogues/portability/e2e-release-snapshot-v1")
STATUS_DOCUMENT = Path("docs/release-status.html")
REQUIRED_GATES = (
    "e2e_02_portable_cross_adapter",
    "e2e_03_nse_real",
    "e2e_04_operational_performance",
)
EXPECTED_EVIDENCE = {
    "e2e_02_portable_cross_adapter": (
        "portable-e2e-verification-v1",
        "config/data/analogues/portability/portable-e2e-verification-v1/RESULT.json",
    ),
    "e2e_03_nse_real": (
        "nse-real-e2e-verification-v2",
        "config/data/analogues/portability/nse-real-e2e-verification-v5/RESULT.json",
    ),
    "e2e_04_operational_performance": (
        "m04r14-e2e-operational-certificate-v1",
        "config/data/analogues/portability/e2e-operational-certificate-v2/RESULT.json",
    ),
}


class ReleaseSnapshotError(RuntimeError):
    """The release cannot be truthfully sealed."""


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.doctype = False
        self.html_depth = 0
        self.html_opened = 0
        self.html_closed = 0
        self.links: list[str] = []

    def handle_decl(self, decl: str) -> None:
        self.doctype |= decl.strip().lower() == "doctype html"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "html":
            self.html_depth += 1
            self.html_opened += 1
        if tag.lower() == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "html":
            self.html_depth -= 1
            self.html_closed += 1


def _run(repository: Path, *command: str, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=repository, text=True, capture_output=True,
        timeout=timeout, check=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_html(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        parser = _DocumentParser()
        parser.feed(path.read_text(encoding="utf-8"))
        parser.close()
    except (OSError, UnicodeError, ValueError):
        return False
    return all((
        parser.doctype, parser.html_opened == 1, parser.html_closed == 1,
        parser.html_depth == 0,
    ))


def _semantic_digest(payload: dict[str, Any], omitted: set[str]) -> bool:
    state = {key: value for key, value in payload.items() if key not in omitted}
    return stable_hash(state) == payload.get("result_digest")


def _portable_digest(payload: dict[str, Any]) -> bool:
    state = {
        key: value for key, value in payload.items()
        if key not in {"passed", "result_digest", "failures", "generated_at"}
    }
    state["failures"] = list(payload.get("failures") or [])
    return stable_hash(state) == payload.get("result_digest")


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ReleaseSnapshotError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ReleaseSnapshotError(f"non-finite JSON value {item}: {path}")
        ),
    )
    if not isinstance(value, dict):
        raise ReleaseSnapshotError(f"expected JSON object: {path}")
    return value, raw


def _gate_evidence(repository: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for task in REQUIRED_GATES:
        gate_path = repository / "config/data/analogues/gates" / f"{task}.json"
        try:
            gate, gate_raw = _load_json(gate_path)
            machine = Path(str(gate["metrics"]["machine_artifact"])).resolve()
            report = Path(str(gate["metrics"]["html_artifact"])).resolve()
            machine.relative_to(repository)
            report.relative_to(repository)
            expected_schema, expected_machine = EXPECTED_EVIDENCE[task]
            if any(path.is_symlink() for path in (gate_path, machine, report)):
                raise ReleaseSnapshotError(f"release evidence may not be symlinked: {task}")
            result, result_raw = _load_json(machine)
            expected_digest = str(gate["metrics"]["result_digest"])
            history = repository / "config/data/analogues/gates/history" / task
            history_match = history.is_dir() and any(
                item.is_file() and item.read_bytes() == gate_raw for item in history.glob("*.json")
            )
            valid = gate.get("task") == task
            valid &= bool(gate.get("passed")) and not gate.get("failures")
            valid &= gate.get("metrics", {}).get("schema_version") == expected_schema
            valid &= str(machine.relative_to(repository)) == expected_machine
            valid &= report == machine.with_name("report.html")
            valid &= history_match
            valid &= result.get("result_digest") == expected_digest
            valid &= result.get("schema_version") == expected_schema
            valid &= result.get("passed") is True and not result.get("failures")
            valid &= result.get("production_authorized", False) is False
            valid &= _parse_html(report)
            if task == REQUIRED_GATES[0]:
                valid &= _portable_digest(result)
            elif task == REQUIRED_GATES[1]:
                valid &= not validate_nse_e2e_result_payload(result)
            else:
                valid &= _semantic_digest(
                    result, {"result_digest", "created_at", "elapsed_seconds"},
                )
            rows.append({
                "task": task, "passed": bool(valid),
                "gate_sha256": hashlib.sha256(gate_raw).hexdigest(),
                "machine_artifact": str(machine.relative_to(repository)),
                "machine_sha256": hashlib.sha256(result_raw).hexdigest(),
                "html_artifact": str(report.relative_to(repository)),
                "html_sha256": _sha256(report), "result_digest": expected_digest,
                "gate_history_matched": history_match,
                "authority_verification_digest": result.get("authority_verification_digest"),
            })
            if not valid:
                failures.append(f"{task}_invalid")
        except (KeyError, OSError, ValueError, json.JSONDecodeError, ReleaseSnapshotError) as exc:
            failures.append(f"{task}_unreadable:{type(exc).__name__}")
    authority_digests = {
        row["authority_verification_digest"] for row in rows
        if row["task"] in {"e2e_03_nse_real", "e2e_04_operational_performance"}
    }
    if len(authority_digests) != 1 or None in authority_digests:
        failures.append("nse_operational_authority_binding_mismatch")
    return rows, failures


def _local_links_exist(repository: Path, status_path: Path) -> bool:
    parser = _DocumentParser()
    try:
        parser.feed(status_path.read_text(encoding="utf-8")); parser.close()
    except (OSError, UnicodeError, ValueError):
        return False
    for href in parser.links:
        target = href.split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        candidate = (status_path.parent / target).resolve()
        try:
            candidate.relative_to(repository)
        except ValueError:
            return False
        if not candidate.exists():
            return False
    return True


def _tracked_html(repository: Path) -> tuple[list[dict[str, Any]], bool]:
    listed = _run(repository, "git", "ls-files", "-z", "*.html")
    if listed.returncode:
        return [], False
    paths = sorted(item for item in listed.stdout.split("\0") if item)
    rows = [
        {"path": item, "sha256": _sha256(repository / item),
         "parsed": _parse_html(repository / item)}
        for item in paths
    ]
    return rows, bool(rows) and all(row["parsed"] for row in rows)


def _run_regression(repository: Path) -> dict[str, Any]:
    started = perf_counter()
    result = _run(
        repository, str(repository / ".venv/bin/python"), "-m", "pytest", "-q",
        timeout=3600,
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    counts = [int(value) for value in re.findall(r"(\d+) passed", combined)]
    return {
        "passed": result.returncode == 0 and bool(counts),
        "returncode": result.returncode, "tests_passed": counts[-1] if counts else 0,
        "elapsed_seconds": perf_counter() - started,
        "output_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        "output_tail": combined[-4000:],
        "command": ".venv/bin/python -m pytest -q",
        "python": sys.version.split()[0],
        "market_analogues_version": version("market-analogues"),
    }


def execute(repository: Path, config_path: Path, output: Path = OUTPUT) -> dict[str, Any]:
    repository = repository.resolve()
    output = (repository / output).resolve() if not output.is_absolute() else output.resolve()
    output.relative_to(repository)
    if output.exists():
        raise ReleaseSnapshotError(f"create-only output already exists: {output}")
    dirty = _run(repository, "git", "status", "--porcelain", "--untracked-files=no")
    if dirty.returncode or dirty.stdout.strip():
        raise ReleaseSnapshotError("tracked repository must be clean before release verification")
    head = _run(repository, "git", "rev-parse", "HEAD")
    tree = _run(repository, "git", "rev-parse", "HEAD^{tree}")
    fsck = _run(repository, "git", "fsck", "--full", "--no-progress", "--no-dangling")
    diff_check = _run(repository, "git", "diff", "--check", "HEAD")
    tracked_index = _run(repository, "git", "ls-files", "-s", "-z")
    if head.returncode or tree.returncode:
        raise ReleaseSnapshotError("cannot resolve release Git identity")

    evidence, evidence_failures = _gate_evidence(repository)
    html_rows, html_passed = _tracked_html(repository)
    status_path = repository / STATUS_DOCUMENT
    status_text = status_path.read_text(encoding="utf-8") if status_path.is_file() else ""
    status_complete = all((
        _parse_html(status_path), "Predictive or trading claim" in status_text,
        "NOT AUTHORIZED" in status_text,
        "252-session NSE" in status_text, "local frozen evidence" in status_text,
        all(row["result_digest"] in status_text for row in evidence),
        _local_links_exist(repository, status_path),
    ))
    regression = _run_regression(repository)
    post_test_clean = _run(repository, "git", "status", "--porcelain", "--untracked-files=no")
    gates = {
        "required_evidence_reconstructed": not evidence_failures and len(evidence) == len(REQUIRED_GATES),
        "full_regression_passed": bool(regression["passed"]),
        "tracked_html_parsed": html_passed,
        "status_index_complete": status_complete,
        "git_object_database_valid": fsck.returncode == 0,
        "git_diff_check_passed": diff_check.returncode == 0,
        "tracked_index_readable": tracked_index.returncode == 0 and bool(tracked_index.stdout),
        "repository_clean_before_and_after": post_test_clean.returncode == 0 and not post_test_clean.stdout.strip(),
        "release_boundary_non_predictive": "NOT AUTHORIZED" in status_text,
    }
    failures = evidence_failures + [name for name, passed in gates.items() if not passed]
    state: dict[str, Any] = {
        "schema_version": SCHEMA, "status": "complete", "passed": not failures,
        "git_head": head.stdout.strip(), "git_tree": tree.stdout.strip(),
        "tracked_index_sha256": hashlib.sha256(tracked_index.stdout.encode()).hexdigest(),
        "config_path": str(config_path), "evidence": evidence,
        "evidence_manifest_digest": stable_hash(evidence),
        "tracked_html": html_rows, "tracked_html_count": len(html_rows),
        "tracked_html_manifest_digest": stable_hash(html_rows),
        "regression": regression, "status_document": str(STATUS_DOCUMENT),
        "status_document_sha256": _sha256(status_path) if status_path.is_file() else None,
        "gates": gates, "failures": failures, "production_authorized": False,
        "predictive_or_trading_claim_authorized": False,
    }
    state["result_digest"] = stable_hash(state)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    temporary_result = temporary / "RESULT.json"
    descriptor = os.open(temporary_result, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, output)
    result_path = output / "RESULT.json"
    written, _ = _load_json(result_path)
    if not _semantic_digest(written, {"result_digest"}):
        raise ReleaseSnapshotError("written release receipt cannot be reconstructed")
    GateReport(
        "e2e_05_release_snapshot", state["passed"], {
            "schema_version": SCHEMA, "result_digest": state["result_digest"],
            "machine_artifact": str(result_path), "git_head": state["git_head"],
            "tests_passed": regression["tests_passed"],
            "tracked_html_count": len(html_rows), "gates": gates,
        }, failures,
    ).write(repository / "config/data/analogues/gates")
    if failures:
        raise ReleaseSnapshotError("release snapshot failed: " + ", ".join(failures))
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.repository, args.config, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

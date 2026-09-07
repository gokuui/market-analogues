"""Replay the frozen WF-03 baseline verifier in its historical Git environment.

The original producer correctly refuses to run when its frozen runtime files no
longer match the current checkout.  That protects production, but it also means
that historical evidence must be audited with the code that actually produced
and verified it.  This read-only helper creates a temporary shared Git clone at
the sealed verifier commit, hard-links only the required ignored artifacts,
runs the old verifier in dry-run mode, and compares every semantic field with
the retained verification receipt.  Elapsed time and its enclosing seal are
the only expected replay differences.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


PREREGISTRATION_RELATIVE = Path(
    "experiments/m04r/m04r14_t14_10_wf03_baseline_batch_v2_preregistered.json"
)
VERIFICATION_RELATIVE = Path(
    "config/data/analogues/m04r14/"
    "t14-10-wf03-baseline-batch-v2-verification/VERIFIED.json"
)
VERIFIER_RELATIVE = Path(
    "experiments/m04r/verify_m04r14_t14_10_wf03_baseline_batch.py"
)
REQUIRED_IGNORED_ROOTS = (
    Path("config/data/analogues/m04r14/t14-10-wf03-baseline-store-full-v1"),
    Path(
        "config/data/analogues/m04r14/"
        "t14-10-wf03-baseline-store-full-v1-verification"
    ),
    Path("config/data/analogues/m04r14/t14-10-wf03-baseline-batch-v2"),
)
REPLAY_SCHEMA = "m04r14-t14-10-wf03-baseline-historical-replay-v1"
VOLATILE_FIELDS = frozenset({"elapsed_seconds", "verification_digest"})


class HistoricalReplayError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def _digest(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalReplayError(f"cannot read strict JSON: {path}") from exc
    if type(value) is not dict:
        raise HistoricalReplayError(f"JSON root differs: {path}")
    return value


def _validate_seal(value: Mapping[str, Any], field: str) -> None:
    observed = value.get(field)
    payload = {key: item for key, item in value.items() if key != field}
    if type(observed) is not str or observed != _digest(payload):
        raise HistoricalReplayError(f"{field} differs")


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repository, text=True,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise HistoricalReplayError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _require_regular_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise HistoricalReplayError(f"required artifact root differs: {root}")
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise HistoricalReplayError(f"artifact tree contains unsafe entry: {path}")


def semantic_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic verifier receipt fields compared on replay."""
    return {key: item for key, item in value.items() if key not in VOLATILE_FIELDS}


def _link_artifacts(repository: Path, checkout: Path) -> None:
    for relative in REQUIRED_IGNORED_ROOTS:
        source = repository / relative
        destination = checkout / relative
        _require_regular_tree(source)
        if destination.exists() or destination.is_symlink():
            raise HistoricalReplayError(f"historical artifact destination exists: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, copy_function=os.link)


def replay(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain"):
        raise HistoricalReplayError("historical replay requires a clean checkout")

    preregistration = _read(repository / PREREGISTRATION_RELATIVE)
    verification = _read(repository / VERIFICATION_RELATIVE)
    _validate_seal(preregistration, "preregistration_digest")
    _validate_seal(verification, "verification_digest")

    implementation_commit = preregistration.get("implementation_commit")
    verifier_commit = verification.get("verifier_commit")
    if type(implementation_commit) is not str or type(verifier_commit) is not str:
        raise HistoricalReplayError("historical Git binding differs")
    _git(repository, "merge-base", "--is-ancestor", implementation_commit, verifier_commit)
    _git(repository, "merge-base", "--is-ancestor", verifier_commit, "HEAD")

    for relative, expected in preregistration.get("runtime_files", {}).items():
        blob = subprocess.run(
            ["git", "show", f"{implementation_commit}:{relative}"],
            cwd=repository, capture_output=True, check=False,
        )
        if blob.returncode or sha256(blob.stdout).hexdigest() != expected:
            raise HistoricalReplayError(f"frozen producer runtime differs: {relative}")
    verifier_blob = subprocess.run(
        ["git", "show", f"{verifier_commit}:{VERIFIER_RELATIVE.as_posix()}"],
        cwd=repository, capture_output=True, check=False,
    )
    if verifier_blob.returncode or sha256(verifier_blob.stdout).hexdigest() \
            != verification.get("verifier_runtime_sha256"):
        raise HistoricalReplayError("frozen verifier runtime differs")

    with tempfile.TemporaryDirectory(prefix="wf03-baseline-replay-") as temporary:
        checkout = Path(temporary) / "checkout"
        clone = subprocess.run(
            ["git", "clone", "--quiet", "--shared", "--no-checkout",
             str(repository), str(checkout)],
            text=True, capture_output=True, check=False,
        )
        if clone.returncode:
            raise HistoricalReplayError(f"temporary clone failed: {clone.stderr.strip()}")
        _git(checkout, "checkout", "--quiet", "--detach", verifier_commit)
        _link_artifacts(repository, checkout)
        if _git(checkout, "status", "--porcelain"):
            raise HistoricalReplayError("historical checkout is not clean after artifact link")

        environment = dict(os.environ)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join((
            str(checkout / "src"), str(checkout),
        ))
        process = subprocess.run(
            [sys.executable, str(checkout / VERIFIER_RELATIVE),
             "--repository", str(checkout), "--dry-run"],
            cwd=checkout, env=environment, text=True,
            capture_output=True, check=False,
        )
        if process.returncode:
            raise HistoricalReplayError(
                "frozen verifier failed: " + (process.stderr.strip() or process.stdout.strip())
            )
        try:
            observed = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise HistoricalReplayError("frozen verifier output is not JSON") from exc
        if type(observed) is not dict:
            raise HistoricalReplayError("frozen verifier output root differs")
        _validate_seal(observed, "verification_digest")

    if semantic_receipt(observed) != semantic_receipt(verification):
        raise HistoricalReplayError("historical verifier semantic receipt differs")
    state = {
        "schema_version": REPLAY_SCHEMA,
        "status": "complete",
        "passed": True,
        "current_commit": _git(repository, "rev-parse", "HEAD"),
        "implementation_commit": implementation_commit,
        "verifier_commit": verifier_commit,
        "preregistration_digest": preregistration["preregistration_digest"],
        "retained_verification_digest": verification["verification_digest"],
        "replay_verification_digest": observed["verification_digest"],
        "semantic_receipt_equal": True,
        "volatile_fields_excluded": sorted(VOLATILE_FIELDS),
        "queries_verified": observed["queries_verified"],
        "neighbor_references_verified": observed["neighbor_references_verified"],
        "oracle_queries": observed["oracle_queries"],
        "outcomes_or_labels_used": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
    }
    return {**state, "replay_digest": _digest(state)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = replay(args.repository)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

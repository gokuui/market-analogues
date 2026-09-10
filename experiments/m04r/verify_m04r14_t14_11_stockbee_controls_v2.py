"""Preregister and run the semantics-identical, bucket-cached Stockbee verifier V2."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from market_analogues.types import stable_hash

from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_11_stockbee_controls as target
from experiments.m04r import verify_m04r14_t14_11_stockbee_controls as v1


SCHEMA = "m04r14-t14-11-stockbee-controls-verification-v2"
PREREGISTRATION_RELATIVE = Path("experiments/m04r/verify_m04r14_t14_11_stockbee_controls_v2_preregistered.json")
OUTPUT_RELATIVE = Path("config/data/analogues/m04r14/t14-11-stockbee-controls-v1-verification-v2")
INNER_RELATIVE = Path("config/data/analogues/m04r14/t14-11-stockbee-controls-v1-verification-v2-inner")
RUNTIME_FILES = (
    "experiments/m04r/verify_m04r14_t14_11_stockbee_controls_v2.py",
    "experiments/m04r/verify_m04r14_t14_11_stockbee_controls.py",
    "experiments/m04r/m04r14_t14_11_stockbee_controls.py",
    "config/m04r14-t14-11-stockbee-contract.json", "pyproject.toml",
)


class V2VerificationError(RuntimeError): pass


def _now() -> str: return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=repository, capture_output=True, text=not raw, check=False)
    if result.returncode:
        error = result.stderr if not raw else result.stderr.decode(errors="replace")
        raise V2VerificationError(error.strip())
    return result.stdout if raw else result.stdout.strip()


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value); result[key] = stable_hash(result); result["created_at"] = _now(); return result


def _valid(value: Mapping[str, Any], key: str) -> bool:
    return value.get(key) == stable_hash({name: item for name, item in value.items() if name not in {key, "created_at"}})


def _store_binding(repository: Path) -> dict[str, Any]:
    path = repository / target.OUTPUT_RELATIVE / "SEALED.json"; seal = base._read(path)
    if not target._valid(seal, timing=True) or seal.get("file_manifest") != target._manifest(path.parent):
        raise V2VerificationError("producer store is not valid")
    return {"store_result_digest": seal["result_digest"], "store_sha256": _sha(path),
            "matched_control_rows": seal["matched_control_rows"], "event_match_rows": seal["event_match_rows"]}


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES): raise V2VerificationError("runtime file is uncommitted")
    return {name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest() for name in RUNTIME_FILES}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise V2VerificationError("clean worktree required")
    if (repository / OUTPUT_RELATIVE).exists() or (repository / INNER_RELATIVE).exists():
        raise V2VerificationError("V2 verification output already exists")
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    return _seal({
        "schema_version": SCHEMA, "status": "frozen_before_v2_verification",
        "implementation_h0": h0, "runtime_files": _runtime_manifest(repository, h0),
        "producer": _store_binding(repository),
        "v1_interruption": "no_receipt;manual_interrupt_after_redundant_per_event_bucket_rebuild_was_confirmed",
        "semantic_change_from_v1": False,
        "mechanical_change": "cache_independently_rebuilt_candidate_buckets_once_per_names_deciles_nonwinner_identity",
        "all_control_rows_still_checked": True, "all_inference_rows_still_reconstructed": True,
        "production_promotion_authorized": False,
    }, "preregistration_digest")


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
    found = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0: continue
        for child in values[1:]:
            parents = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child)).splitlines()
            if parents == [child, h0] and changed == [PREREGISTRATION_RELATIVE.as_posix()] \
                    and _git(repository, "show", f"{child}:{PREREGISTRATION_RELATIVE}", raw=True) == raw:
                found.append(child)
    if len(set(found)) != 1: raise V2VerificationError("expected exact V2 preregistration-only child")
    return found[0]


def validate_preregistration(repository: Path) -> tuple[dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise V2VerificationError("clean worktree required")
    path = repository / PREREGISTRATION_RELATIVE; raw = path.read_bytes(); prereg = base._read(path)
    if not _valid(prereg, "preregistration_digest") or prereg.get("producer") != _store_binding(repository):
        raise V2VerificationError("V2 preregistration or producer binding differs")
    h1 = _sole_child(repository, raw, str(prereg["implementation_h0"]))
    if subprocess.run(["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository).returncode:
        raise V2VerificationError("HEAD does not descend from V2 preregistration")
    for name, expected in prereg["runtime_files"].items():
        if _sha(repository / name) != expected: raise V2VerificationError(f"runtime drifted: {name}")
    return prereg, h1


def cached_selector():
    """Return a V1-signature selector that caches only group-invariant candidate buckets."""
    state: dict[str, Any] = {}

    def select(names, deciles, nonwinner, event_symbol, event_deciles, contract, event_id):
        if state.get("names") is not names or state.get("deciles") is not deciles \
                or state.get("nonwinner") is not nonwinner:
            buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for index in np.flatnonzero(nonwinner):
                buckets[tuple(int(value) for value in deciles[index])].append(int(index))
            state.clear(); state.update(names=names, deciles=deciles, nonwinner=nonwinner, buckets=buckets,
                                        all_indices=list(np.flatnonzero(nonwinner)))
        key = tuple(int(value) for value in event_deciles); exact = state["buckets"].get(key, [])
        if len(exact) >= 5: pool, tier = exact, "exact_all_four_deciles"
        else:
            pool = []
            for candidate in product(*[range(max(1, value - 1), min(10, value + 1) + 1) for value in key]):
                pool.extend(state["buckets"].get(tuple(candidate), []))
            if len(pool) >= 5: tier = "within_one_bucket_all_four"
            else: pool, tier = state["all_indices"], "same_date_horizon_unmatched"
        pool = [index for index in pool if str(names[index]) != event_symbol]
        pool.sort(key=lambda index: (v1._digest(contract, event_id, str(names[index])), str(names[index])))
        return pool[:5], tier

    return select


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True); prereg, h1 = validate_preregistration(repository)
    output = repository / OUTPUT_RELATIVE
    if output.exists():
        receipt = base._read(output / "VERIFIED.json")
        if not _valid(receipt, "verification_digest"): raise V2VerificationError("existing V2 receipt differs")
        return receipt
    started = perf_counter(); inner_root = repository / INNER_RELATIVE
    if inner_root.exists():
        inner = base._read(inner_root / "VERIFIED.json")
    else:
        original_selector, original_path = v1._select, target.VERIFICATION_RELATIVE
        try:
            v1._select = cached_selector(); target.VERIFICATION_RELATIVE = INNER_RELATIVE
            inner = v1.execute(repository)
        finally:
            v1._select, target.VERIFICATION_RELATIVE = original_selector, original_path
    inner_state = {name: value for name, value in inner.items() if name != "verification_digest"}
    if inner.get("verification_digest") != stable_hash(inner_state) or inner.get("passed") is not True \
            or inner.get("store_result_digest") != prereg["producer"]["store_result_digest"]:
        raise V2VerificationError("inner exhaustive receipt differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "preregistration_h1": h1, "preregistration_digest": prereg["preregistration_digest"],
        "store_result_digest": prereg["producer"]["store_result_digest"],
        "inner_verification_digest": inner["verification_digest"],
        "inner_verification_sha256": _sha(inner_root / "VERIFIED.json"),
        "verified_matched_control_rows": inner["verified_matched_control_rows"],
        "verified_event_match_rows": inner["verified_event_match_rows"],
        "verified_inference_rows": inner["verified_inference_rows"],
        "gates": inner["gates"], "semantic_change_from_v1": False,
        "production_promotion_authorized": False,
        "elapsed_seconds": perf_counter() - started,
    }
    receipt = _seal(state, "verification_digest")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        smoke._atomic_json(temporary / "VERIFIED.json", receipt); os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preregister", "run"):
        child = sub.add_parser(command); child.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        value = build_preregistration(args.repository); smoke._atomic_json(args.repository / PREREGISTRATION_RELATIVE, value)
    else: value = execute(args.repository)
    print(json.dumps(value, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())

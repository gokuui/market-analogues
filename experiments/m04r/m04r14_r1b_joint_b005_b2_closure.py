"""Fail-closed joint lock for the independently verified R1-B B0-05/B2 stage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4


SCHEMA = "m04r14-r1b-joint-b005-b2-closure-v2"
H1 = "0476e694c2d13b5d2e75b642b81597a03a5c6f76"
VERIFIER_COMMIT = "ace0b3b3bcf58cc4f1e233d744610d69ac79d67d"
PREREG_DIGEST = "734e65768f04cc9c8108e2dae480258f5fc48094c1db80517fc0f5bbe62af191"
OUTPUT = Path("config/data/analogues/m04r14/r1b-joint-b005-b2-closure-v2/LOCKED.json")
SUPERSEDED_V1_DIGEST = "2bbf07f386223af6a85d43261f212314c58c75232e0df291681fc2a5db197e81"
RUNTIME = (
    "experiments/m04r/m04r14_r1b_joint_b005_b2_closure.py",
    "tests/test_r1b_joint_b005_b2_closure.py",
)
EVIDENCE = {
    "preregistration": {
        "path": "experiments/m04r/m04r14_r1b_joint_b005_b2_preregistered.json",
        "sha256": "f16c2a9c12289ee2c2791ce39a48b0f27585a5f890bf7b62ce1c665a8fb7a74c",
    },
    "b005_result": {
        "path": "config/data/analogues/m04r14/r1b-b005-shared-priority-v1/RESULT.json",
        "sha256": "3380b1dd79e91b259d195b9f3060d83d59eebe88936d560a7ec6634c67c550a7",
        "digest": "c4bfe4f39d06e87d8e0ecc07c52395efd84cfda27c4c7d06879b16a8c325c8d7",
    },
    "b005_verification": {
        "path": "config/data/analogues/m04r14/r1b-b005-shared-priority-integrity-verification-v1/VERIFIED.json",
        "sha256": "e05efd733ef4909c629a07d61768ef634c8494a488331ebf277756273c9b23c4",
        "digest": "9ecea9308c5db7feeec941ab67fe9a3dd22593cd88bec33c807503108ba6918e",
    },
    "geometry_result": {
        "path": "config/data/analogues/m04r14/r1b-b2-geometry-v1/RESULT.json",
        "sha256": "16f1365cb8b534ee83cd5436225697af82e70c90f4360c5d07eaa60be8808301",
        "digest": "061f503a2f0fb8a2ba549aefbe337fe1c7b972a2c3909db8cf84eae6f90131e3",
    },
    "b2_result": {
        "path": "config/data/analogues/m04r14/r1b-b2-localization-v1/RESULT.json",
        "sha256": "d0a79f471d8d511d42cdd73fec5fc0bd7c80c47c28e695a28d13060a7fb4aa5a",
        "digest": "ea9275965210c51337de9f13c35116f1b92c41e2f0fa755aa6b84574f16a480c",
    },
    "b2_verification": {
        "path": "config/data/analogues/m04r14/r1b-b2-localization-integrity-verification-v1/VERIFIED.json",
        "sha256": "5b597fe7365567624848ef5bd8e3c22ea775e52f063ad63f8e235305b889f226",
        "digest": "3833a1472f808e82443dd54837a3d70d26e00729317eaf1dd8afa7caa0aefbf4",
    },
}


class ClosureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosureError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ClosureError(f"unsafe evidence: {path}") from error
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"regular evidence required: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def decode(content: bytes, path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {path}/{key}")
            result[key] = value
        return result
    try:
        value = json.loads(content, object_pairs_hook=pairs,
                           parse_constant=lambda token: require(False, f"nonfinite JSON: {path}/{token}"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ClosureError(f"invalid JSON: {path}") from error
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def load_bound(root: Path, record: Mapping[str, str]) -> tuple[dict[str, Any], bytes]:
    relative = Path(record["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe evidence path")
    content = snapshot(root / relative)
    require(sha256(content).hexdigest() == record["sha256"], f"evidence hash differs: {relative}")
    return decode(content, root / relative), content


def git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(("git", *args), cwd=root, capture_output=True,
                            text=not binary, check=False)
    require(result.returncode == 0, f"git {' '.join(args)} failed")
    return result.stdout if binary else result.stdout.strip()


def verify_receipt(value: Mapping[str, Any], *, b005: bool) -> None:
    excluded = {"verification_digest", "created_at"}
    if b005:
        excluded.add("performance")
    require(value["verification_digest"] == stable({key: item for key, item in value.items()
                                                     if key not in excluded}),
            "verification self-digest differs")
    require(value["passed"] is True and all(value["gates"].values()), "verification gate differs")
    require(value["verifier_commit"] == VERIFIER_COMMIT, "verifier commit differs")
    claims = value["claims"]
    for key in ("predictive_claim_authorized", "production_promotion_authorized",
                "ranking_change_authorized"):
        require(claims[key] is False, f"forbidden claim enabled: {key}")


def validate(root: Path) -> dict[str, Any]:
    prereg, _ = load_bound(root, EVIDENCE["preregistration"])
    require(prereg["preregistration_digest"] == PREREG_DIGEST
            == stable({key: value for key, value in prereg.items() if key != "preregistration_digest"}),
            "preregistration digest differs")
    b005, _ = load_bound(root, EVIDENCE["b005_result"])
    b005v, _ = load_bound(root, EVIDENCE["b005_verification"])
    geometry, _ = load_bound(root, EVIDENCE["geometry_result"])
    b2, _ = load_bound(root, EVIDENCE["b2_result"])
    b2v, _ = load_bound(root, EVIDENCE["b2_verification"])

    require(b005["result_digest"] == EVIDENCE["b005_result"]["digest"]
            and b005v["verification_digest"] == EVIDENCE["b005_verification"]["digest"]
            and b005v["producer_result_digest"] == b005["result_digest"], "B0-05 chain differs")
    require(b005["status"] == "diagnostic_complete_pending_independent_verification"
            and b005v["status"] == "verified_diagnostic_complete", "B0-05 status differs")
    verify_receipt(b005v, b005=True)
    require(b005["claims"]["real_forward_outcomes_accessed"] is False
            and b005v["claims"]["real_forward_outcomes_accessed"] is False,
            "B0-05 outcome boundary differs")

    require(geometry["geometry_digest"] == EVIDENCE["geometry_result"]["digest"]
            and b2["result_digest"] == EVIDENCE["b2_result"]["digest"]
            and b2v["verification_digest"] == EVIDENCE["b2_verification"]["digest"]
            and b2v["verified_result_digest"] == b2["result_digest"]
            and b2v["verified_geometry_digest"] == geometry["geometry_digest"], "B2 chain differs")
    require(geometry["status"] == "geometry_complete_pending_independent_verification"
            and b2["scientific"]["status"] == "established_pending_independent_verification"
            and b2["scientific"]["decision"]["status"] == "structurally_localized"
            and b2v["status"] == "verified_structurally_localized", "B2 status differs")
    verify_receipt(b2v, b005=False)
    require(geometry["claims"]["real_forward_outcomes_accessed"] is False
            and b2["scientific"]["claims"]["outcomes_opened"] is False
            and b2v["claims"]["outcomes_opened"] is False, "B2 outcome boundary differs")

    return {
        "preregistration_digest": PREREG_DIGEST,
        "h1_commit": H1,
        "b005": {"result_digest": b005["result_digest"],
                  "verification_digest": b005v["verification_digest"],
                  "finding": "shared-priority concentration mostly not exceptional"},
        "b2": {"geometry_digest": geometry["geometry_digest"],
               "result_digest": b2["result_digest"],
               "verification_digest": b2v["verification_digest"],
               "finding": "structurally localized under the frozen conditional nulls"},
    }


def publish(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), "create-only lock already exists")
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    require(not str(git(root, "status", "--porcelain", "--untracked-files=all")),
            "joint closure requires a clean committed tree")
    head = str(git(root, "rev-parse", "HEAD"))
    git(root, "merge-base", "--is-ancestor", H1, head)
    git(root, "merge-base", "--is-ancestor", VERIFIER_COMMIT, head)
    runtime = {relative: snapshot(root / relative) for relative in RUNTIME}
    for relative, content in runtime.items():
        require(git(root, "show", f"{head}:{relative}", binary=True) == content,
                f"closure runtime not committed: {relative}")
    evidence = validate(root)
    state = {
        "schema_version": SCHEMA,
        "status": "joint_b005_b2_verified_and_locked",
        "passed": True,
        "closure_commit": head,
        "runtime_sha256": {relative: sha256(content).hexdigest()
                           for relative, content in runtime.items()},
        "supersedes_closure_digest": SUPERSEDED_V1_DIGEST,
        "evidence": evidence,
        "scope": {
            "locked": ["B0-01", "B0-02", "B0-05", "B2"],
            "deferred_nonblocking_research": ["B0-03", "B0-04", "B0-06-full", "B1"],
        },
        "claims": {
            "structural_localization_verified": True,
            "b005_sensitivity_verified": True,
            "adequacy_labels_authorized": False,
            "predictive_claim_authorized": False,
            "ranking_change_authorized": False,
            "production_promotion_authorized": False,
            "trading_claim_authorized": False,
            "real_forward_outcomes_accessed": False,
        },
    }
    deterministic = {**state, "closure_digest": stable(state)}
    receipt = {**deterministic, "created_at": datetime.now(timezone.utc).isoformat()}
    publish(root / OUTPUT, receipt)
    require(validate(root) == evidence, "evidence changed during closure")
    require({relative: snapshot(root / relative) for relative in RUNTIME} == runtime,
            "closure runtime changed during publication")
    validate_locked(root)
    return receipt


def validate_locked(root: Path) -> dict[str, Any]:
    """Validate the sealed lock from its historical Git runtime without republishing it."""
    root = root.resolve(); path = root / OUTPUT
    receipt = decode(snapshot(path), path)
    required = {
        "schema_version", "status", "passed", "closure_commit", "runtime_sha256",
        "supersedes_closure_digest", "evidence", "scope", "claims", "closure_digest", "created_at",
    }
    require(set(receipt) == required, "lock field closure differs")
    require(receipt["schema_version"] == SCHEMA
            and receipt["status"] == "joint_b005_b2_verified_and_locked"
            and receipt["passed"] is True
            and receipt["supersedes_closure_digest"] == SUPERSEDED_V1_DIGEST,
            "lock envelope differs")
    deterministic = {key: value for key, value in receipt.items()
                     if key not in {"closure_digest", "created_at"}}
    require(receipt["closure_digest"] == stable(deterministic), "lock self-digest differs")
    commit = receipt["closure_commit"]
    git(root, "merge-base", "--is-ancestor", H1, commit)
    git(root, "merge-base", "--is-ancestor", VERIFIER_COMMIT, commit)
    for relative in RUNTIME:
        blob = git(root, "show", f"{commit}:{relative}", binary=True)
        require(sha256(blob).hexdigest() == receipt["runtime_sha256"][relative],
                f"historical closure runtime differs: {relative}")
    require(receipt["evidence"] == validate(root), "locked evidence differs")
    require(receipt["claims"] == {
        "structural_localization_verified": True,
        "b005_sensitivity_verified": True,
        "adequacy_labels_authorized": False,
        "predictive_claim_authorized": False,
        "ranking_change_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
        "real_forward_outcomes_accessed": False,
    }, "locked claim boundary differs")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "validate"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    try:
        args = parser.parse_args(argv)
        result = run(args.repository) if args.action == "run" else validate_locked(args.repository)
    except ClosureError as error:
        print(f"joint closure refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({"passed": result["passed"], "closure_digest": result["closure_digest"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

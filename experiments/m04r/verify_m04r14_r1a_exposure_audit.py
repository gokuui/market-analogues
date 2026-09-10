"""Reconstruct the R1-A audit and its deterministic null simulation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

from experiments.m04r import m04r14_r1a_exposure_audit as audit
from market_analogues.types import stable_hash


SCHEMA = "m04r14-r1a-exposure-audit-verification-v1"
OUTPUT = Path("config/data/analogues/m04r14/r1a-exposure-audit-v2-verification")


class VerificationError(RuntimeError):
    pass


class _HTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(); self.doctype = False; self.depth = 0; self.opened = 0
    def handle_decl(self, decl: str) -> None:
        self.doctype |= decl.lower().strip() == "doctype html"
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "html": self.depth += 1; self.opened += 1
    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "html": self.depth -= 1


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def _html_valid(path: Path) -> bool:
    parser = _HTML(); parser.feed(path.read_text(encoding="utf-8")); parser.close()
    return parser.doctype and parser.opened == 1 and parser.depth == 0


def execute(repository: Path, *, workers: int | None = None) -> dict[str, Any]:
    repository = repository.resolve()
    output = repository / OUTPUT
    if output.exists() or output.is_symlink():
        raise VerificationError(f"create-only output exists: {output}")
    prereg = audit._load(repository / audit.PREREGISTRATION)
    root = repository / audit.OUTPUT
    result = audit._load(root / "RESULT.json")
    deterministic = {
        key: value for key, value in result.items()
        if key not in {"result_digest", "elapsed_seconds"}
    }
    if result.get("result_digest") != stable_hash(deterministic):
        raise VerificationError("result semantic digest differs")
    if not _html_valid(root / "report.html"):
        raise VerificationError("report HTML is invalid")
    if result.get("artifacts") != {
        "null_replicates_sha256": _sha(root / "NULL_REPLICATES.json"),
        "query_diagnostics_sha256": _sha(root / "QUERY_DIAGNOSTICS.json"),
    }:
        raise VerificationError("result artifact hashes differ")
    generation = str(prereg["inputs"]["packed_generation_id"])
    store = audit.PACKED_RESIDENT
    if not (store / "generations" / generation / "manifest.json").is_file():
        store = repository / audit.PACKED_DURABLE
    metadata, manifest = audit._extract_metadata(
        store, generation,
        expected_provenance_digest=str(prereg["inputs"]["packed_provenance_digest"]),
    )
    registry = audit._load(repository / audit.REGISTRY)
    queries, query_rows, retrieval_rows, episode_counts, episode_meta, symbol_counts, actual, case_manifest = audit._actual(
        repository, metadata, registry,
    )
    located = audit._locate_observed(metadata, episode_meta)
    pools = {latest: audit._pool(metadata, latest) for latest in sorted({q.latest_ns for q in queries})}
    for query in queries:
        audit._query_mapping(metadata, *pools[query.latest_ns], query)
    replicate_count = int(prereg["execution"]["null_replicates"])
    worker_count = max(1, min(workers or int(prereg["execution"]["workers"]), replicate_count))
    observed_indices = sorted(located.values())
    audit._WORK.clear(); audit._WORK.update({
        "metadata": metadata, "queries": queries, "pools": pools,
        "observed_positions": {value: index for index, value in enumerate(observed_indices)},
        "seed": int(prereg["execution"]["seed"]),
    })
    chunks = tuple(tuple(range(first, replicate_count, worker_count)) for first in range(worker_count))
    if worker_count == 1:
        parts = [audit._simulate(chunks[0])]
    else:
        with get_context("fork").Pool(worker_count) as pool:
            parts = pool.map(audit._simulate, chunks)
    null_rows = sorted(
        (row for part in parts for row in part["metrics"]), key=lambda row: row["replicate"],
    )
    stored_null = json.loads((root / "NULL_REPLICATES.json").read_bytes())
    stored_queries = json.loads((root / "QUERY_DIAGNOSTICS.json").read_bytes())
    # Identity is reconstructed directly from sealed cases because query diagnostics
    # intentionally contain no neighbour list.
    identity = []
    for path in sorted((repository / audit.CASES).glob("*.json")):
        case = audit._load(path)
        identity.extend({
            "query_episode_id": case["query_episode_id"],
            "episode_id": item["episode_id"], "rank": rank,
        } for rank, item in enumerate(case["matches"], 1))
    gates = {
        "result_digest_reconstructed": True,
        "html_valid": True,
        "artifact_hashes_reconstructed": True,
        "case_manifest_reconstructed": (
            result["inputs"]["case_manifest_digest"] == stable_hash(case_manifest)
        ),
        "retrieval_identity_reconstructed": (
            result["inputs"]["retrieval_identity_digest"] == stable_hash(identity)
        ),
        "actual_metrics_reconstructed": result["actual"] == actual,
        "query_diagnostics_reconstructed": stored_queries == query_rows,
        "all_null_replicates_reconstructed": stored_null == null_rows,
        "null_digest_reconstructed": result["null"]["metrics_digest"] == stable_hash(null_rows),
        "packed_generation_reconstructed": manifest["manifest_digest"] == generation,
        "outcomes_excluded": result.get("real_forward_outcomes_accessed") is False,
    }
    if not all(gates.values()):
        raise VerificationError(f"verification gates failed: {[k for k,v in gates.items() if not v]}")
    state = {
        "schema_version": SCHEMA, "passed": True, "status": "deterministically_replayed",
        "production_promotion_authorized": False, "predictive_claim_authorized": False,
        "real_forward_outcomes_accessed": False,
        "verified_result_digest": result["result_digest"],
        "verified_null_replicates": replicate_count,
        "verified_queries": len(queries), "verified_links": sum(episode_counts.values()),
        "gates": gates,
    }
    payload = {**state, "verification_digest": stable_hash(state),
               "created_at": datetime.now(timezone.utc).isoformat()}
    temp = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        temp.mkdir(parents=True)
        (temp / "VERIFIED.json").write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True); raise
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.repository, workers=args.workers), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

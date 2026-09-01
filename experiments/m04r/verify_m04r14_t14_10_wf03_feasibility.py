"""Independent structural and semantic verifier for WF-03 feasibility evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.causal_prefix import causal_prefix_digest
from market_analogues.episodes import build_episode
from market_analogues.packed_bound_search import (
    BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
    BoundProposal,
    PackedBoundQuery,
    _packed_query_input_digest,
    bound_proposal_candidate_digest,
    packed_bound_search_contract,
)
from market_analogues.representation import represent, representation_input_digest
from market_analogues.search import latest_eligible_cutoff
from market_analogues.types import InstrumentKey, SearchQuery, stable_hash


GENERATION_ID = "9fc6ae0ec4451133d8006897162f3443803fd30a3c44b78e80a476fbb18bb483"
PROVENANCE_DIGEST = "83ccfa62ac7ffec03e48d0f8a5634c7b8f1b8b0dde1426be22cc343fe116f62d"
REGISTRY_DIGEST = "784e771be69fe77a8684af6b56815ba701c90164400a2f93e53fa835a5da82c4"
PREREGISTRATION = Path("experiments/m04r/m04r14_t14_10_wf03_feasibility_preregistered.json")
REGISTRY = Path("config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1/walk-forward-query-registry.json")
CONFIG = Path("config/datasets.example.yaml")
CANDIDATE = Path("config/data/analogues/m04r14/t14-10-wf03-feasibility-v1")
VERIFICATION = Path("config/data/analogues/m04r14/t14-10-wf03-feasibility-v1-verification")
PROBES = (
    ("early", "a69def453340e01048a52284"),
    ("middle", "4ebfb91d87892612d998e71e"),
    ("late", "b923b64e7f3b36fe2a8d0643"),
)
PROPOSAL_QUOTA = 16_385
TOP_K = 20
GAP = 60
SCHEMA = "m04r14-t14-10-wf03-feasibility-verification-v1"


class VerificationError(RuntimeError):
    pass


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        if key in value:
            raise VerificationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"regular JSON file required: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError(f"regular JSON file required: {path}")
        raw = bytearray()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size,
                             item.st_mtime_ns, item.st_ctime_ns, item.st_mode)
    if identity(before) != identity(after):
        raise VerificationError(f"file identity changed: {path}")
    try:
        result = json.loads(bytes(raw), object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerificationError(f"nonfinite token: {token}")))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}") from exc
    if type(result) is not dict:
        raise VerificationError(f"JSON object required: {path}")
    return result


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _seal(value: Mapping[str, Any], field: str) -> None:
    if value.get(field) != stable_hash({key: item for key, item in value.items() if key != field}):
        raise VerificationError(f"{field} differs")


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise VerificationError(f"create-only target exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".wf03-verify-", dir=path.parent)
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def _load_m13(repository: Path) -> Any:
    path = repository / "experiments/m04r/m04r13_threaded_certified_exposed.py"
    spec = importlib.util.spec_from_file_location("wf03_independent_m13", path)
    if spec is None or spec.loader is None:
        raise VerificationError("certified validator unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _query_context(repository: Path, row: Mapping[str, Any]) -> tuple[Any, Any, SearchQuery, PackedBoundQuery]:
    source = source_from_spec(load_config(repository / CONFIG).datasets["nasdaq"])
    episode = build_episode(source, InstrumentKey("nasdaq", row["symbol"]),
                            row["cutoff"], row["lookback"], row["representation_version"])
    request = SearchQuery(episode.key, ("nasdaq",), ("A", "B"), TOP_K,
                          False, True, 1, GAP)
    packed = PackedBoundQuery(
        episode.key.id, episode.key.instrument.source_symbol,
        int(pd.Timestamp(episode.bars.timestamp.iloc[0]).value),
        int(latest_eligible_cutoff(episode, GAP).value), represent(episode),
        request.quality_tiers,
    )
    if episode.key.id != row["episode_id"]:
        raise VerificationError("query reconstruction differs")
    return source, episode, request, packed


def _query_binding(source: Any, episode: Any, request: SearchQuery,
                   packed: PackedBoundQuery) -> dict[str, Any]:
    benchmark = source.load_benchmark()
    state = {
        "query_stock_prefix": asdict(causal_prefix_digest(
            source.load(episode.key.instrument), episode.key.cutoff,
        )),
        "query_benchmark_prefix": asdict(causal_prefix_digest(
            benchmark, episode.key.cutoff,
        )) if benchmark is not None else None,
        "request": {
            "search_datasets": list(request.search_datasets),
            "quality_tiers": list(request.quality_tiers), "top_k": request.top_k,
            "cross_dataset": request.cross_dataset,
            "deduplicate_overlaps": request.deduplicate_overlaps,
            "max_per_instrument": request.max_per_instrument,
            "minimum_history_gap_bars": request.minimum_history_gap_bars,
        },
        "packed_provenance_digest": PROVENANCE_DIGEST,
        "query_representation_digest": representation_input_digest(packed.representation),
    }
    return {**state, "packed_query_input_digest": _packed_query_input_digest(packed),
            "certified_input_digest": stable_hash(state)}


def _proposal(value: Mapping[str, Any], packed: PackedBoundQuery) -> tuple[BoundProposal, ...]:
    required = {
        "schema_version", "generation_id", "query_episode_id", "candidates",
        "rows_scanned", "eligible_rows", "eligible_main_rows", "eligible_overflow_rows",
        "route_counts", "route_quotas", "block_rows", "block_order", "elapsed_seconds",
        "peak_rss_mb", "candidate_digest", "result_digest", "contract_digest", "input_digest",
    }
    if type(value) is not dict or set(value) != required or type(value["candidates"]) is not list:
        raise VerificationError("proposal fields differ")
    try:
        candidates = tuple(BoundProposal(
            row["episode_id"], row["symbol"], row["cutoff_ns"], row["quality_tier"],
            float.fromhex(row["lower_bound_hex"]), tuple(row["routes"]),
            row["overflow_fallback"],
        ) for row in value["candidates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError("proposal candidate encoding differs") from exc
    deterministic = {
        "schema_version": BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        "contract_digest": packed_bound_search_contract(branch_aware=True)["digest"],
        "generation_id": GENERATION_ID, "query_episode_id": packed.episode_id,
        "rows_scanned": value["rows_scanned"], "eligible_rows": value["eligible_rows"],
        "eligible_main_rows": value["eligible_main_rows"],
        "eligible_overflow_rows": value["eligible_overflow_rows"],
        "route_counts": value["route_counts"], "route_quotas": value["route_quotas"],
        "candidate_digest": value["candidate_digest"],
        "real_forward_outcomes_accessed": False,
        "input_digest": _packed_query_input_digest(packed),
    }
    if not all((
        value["schema_version"] == BRANCH_AWARE_SEARCH_SCHEMA_VERSION,
        value["generation_id"] == GENERATION_ID,
        value["query_episode_id"] == packed.episode_id,
        value["contract_digest"] == deterministic["contract_digest"],
        value["input_digest"] == deterministic["input_digest"],
        value["route_quotas"] == {"composite": PROPOSAL_QUOTA},
        value["route_counts"] == {"composite": len(candidates)},
        value["eligible_rows"] == value["eligible_main_rows"] + value["eligible_overflow_rows"],
        len(candidates) == min(PROPOSAL_QUOTA, value["eligible_rows"]),
        list(candidates) == sorted(candidates, key=lambda item: (item.lower_bound, item.episode_id)),
        all(math.isfinite(item.lower_bound) and item.lower_bound >= 0
            and item.routes == ("composite",) for item in candidates),
        value["candidate_digest"] == bound_proposal_candidate_digest(candidates),
        value["result_digest"] == stable_hash(deterministic),
    )):
        raise VerificationError("proposal reconstruction differs")
    return candidates


def _semantic_proposal(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items()
            if key not in {"block_rows", "block_order", "elapsed_seconds", "peak_rss_mb"}}


def _closed_inventory(root: Path) -> None:
    expected_root = {"COMPLETE.json", "CONTRACT.json", "RUN_STARTED.json", "cases"}
    if {path.name for path in root.iterdir()} != expected_root:
        raise VerificationError("root inventory is not closed")
    expected_case_names = {
        f"{index:03d}-{label}-{query_id}" for index, (label, query_id) in enumerate(PROBES)
    }
    cases = root / "cases"
    if {path.name for path in cases.iterdir()} != expected_case_names:
        raise VerificationError("case inventory is not closed")
    for name in expected_case_names:
        case = cases / name
        if {path.name for path in case.iterdir()} != {"COMPLETE.json", "attempts"}:
            raise VerificationError("case leaf inventory is not closed")
        attempts = case / "attempts"
        if {path.name for path in attempts.iterdir()} != {"attempt-0001"}:
            raise VerificationError("unexpected retry/failure attempt")
        attempt = attempts / "attempt-0001"
        if {path.name for path in attempt.iterdir()} != {
            "RUN_STARTED.json", "PROPOSAL_FORWARD.json", "PROPOSAL_REVERSE.json",
            "EXACT.json", "COMPLETE.json",
        }:
            raise VerificationError("attempt inventory is not closed")


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    root = repository / CANDIDATE
    _closed_inventory(root)
    prereg = _read(repository / PREREGISTRATION); _seal(prereg, "preregistration_digest")
    if _read(root / "CONTRACT.json") != prereg \
            or prereg.get("inputs", {}).get("registry_digest") != REGISTRY_DIGEST:
        raise VerificationError("contract/preregistration differs")
    registry = _read(repository / REGISTRY)
    if registry.get("registry_digest") != REGISTRY_DIGEST \
            or registry["registry_digest"] != stable_hash({
                key: item for key, item in registry.items() if key != "registry_digest"
            }) or registry.get("historical_walk_forward_query_outcomes_opened") is not False:
        raise VerificationError("registry identity/future boundary differs")
    by_id = {row["episode_id"]: row for row in registry["queries_data"]}
    started = _read(root / "RUN_STARTED.json"); _seal(started, "result_digest")
    if started.get("preregistration_digest") != prereg["preregistration_digest"]:
        raise VerificationError("run-started contract differs")
    m13 = _load_m13(repository)
    summaries: list[dict[str, Any]] = []
    case_manifest: list[dict[str, Any]] = []
    durations: list[float] = []
    for ordinal, (label, query_id) in enumerate(PROBES):
        case_root = root / "cases" / f"{ordinal:03d}-{label}-{query_id}"
        attempt = case_root / "attempts/attempt-0001"
        leaf_names = ("RUN_STARTED.json", "PROPOSAL_FORWARD.json",
                      "PROPOSAL_REVERSE.json", "EXACT.json")
        leaves = {name: _read(attempt / name) for name in leaf_names}
        for value in leaves.values():
            _seal(value, "result_digest")
        row = by_id[query_id]
        source, episode, request, packed = _query_context(repository, row)
        binding = _query_binding(source, episode, request, packed)
        forward = leaves["PROPOSAL_FORWARD.json"]["proposal"]
        reverse = leaves["PROPOSAL_REVERSE.json"]["proposal"]
        forward_candidates = _proposal(forward, packed)
        reverse_candidates = _proposal(reverse, packed)
        if forward_candidates != reverse_candidates \
                or _semantic_proposal(forward) != _semantic_proposal(reverse):
            raise VerificationError("forward/reverse proposal parity differs")
        exact = leaves["EXACT.json"]
        if exact.get("query_binding") != binding:
            raise VerificationError("independently reconstructed causal binding differs")
        certificate, matches = exact.get("certificate"), exact.get("matches")
        m13.validate_certificate_and_matches(
            certificate, matches, query_id,
            expected_input_digest=binding["certified_input_digest"],
        )
        distances = [item["total_distance"] for item in matches]
        latest = pd.Timestamp(packed.latest_eligible_ns)
        if not all((
            len(matches) == TOP_K, len({item["symbol"] for item in matches}) == TOP_K,
            distances == sorted(distances),
            certificate["query_episode_id"] == query_id,
            certificate["generation_id"] == GENERATION_ID,
            certificate["eligible_candidates"] == forward["eligible_rows"],
            certificate["exact_evaluated"] + certificate["safely_pruned"]
                == certificate["eligible_candidates"],
            certificate["stop_threshold"] == distances[-1],
            certificate["next_lower_bound"] is not None
                and certificate["next_lower_bound"] > certificate["stop_threshold"],
            all(pd.Timestamp(item["cutoff"]) <= latest for item in matches),
        )):
            raise VerificationError("certified distinct-symbol/closure invariant differs")
        complete = _read(attempt / "COMPLETE.json"); _seal(complete, "complete_digest")
        expected_leaf_manifest = [{"path": name, "sha256": _sha(attempt / name)}
                                  for name in leaf_names]
        if complete.get("leaf_manifest") != expected_leaf_manifest \
                or complete.get("leaf_manifest_digest") != stable_hash(expected_leaf_manifest) \
                or complete.get("query_id") != query_id \
                or complete.get("proposal_parity") is not True \
                or complete.get("certified") is not True \
                or complete.get("distinct_match_symbols") != TOP_K:
            raise VerificationError("attempt terminal differs")
        case = _read(case_root / "COMPLETE.json"); _seal(case, "complete_digest")
        if case.get("attempt_relative") != "attempts/attempt-0001" \
                or case.get("attempt_complete_sha256") != _sha(attempt / "COMPLETE.json") \
                or case.get("attempt_complete_digest") != complete["complete_digest"]:
            raise VerificationError("case terminal binding differs")
        duration = complete["elapsed_seconds"]
        if type(duration) not in {int, float} or isinstance(duration, bool) \
                or not math.isfinite(duration) or duration < 0 \
                or complete.get("resources_after", {}).get("swap_kib") != 0:
            raise VerificationError("case resource evidence differs")
        durations.append(float(duration))
        case_manifest.append({
            "path": f"cases/{ordinal:03d}-{label}-{query_id}/COMPLETE.json",
            "sha256": _sha(case_root / "COMPLETE.json"),
            "complete_digest": case["complete_digest"],
        })
        summaries.append({
            "label": label, "query_id": query_id,
            "eligible_candidates": certificate["eligible_candidates"],
            "exact_evaluated": certificate["exact_evaluated"],
            "safely_pruned": certificate["safely_pruned"],
            "forward_seconds": forward["elapsed_seconds"],
            "reverse_seconds": reverse["elapsed_seconds"],
            "exact_seconds": exact["exact_wall_seconds"],
            "case_seconds": duration,
            "peak_rss_mb": complete["resources_after"]["max_rss_mb"],
            "certificate_result_digest": certificate["result_digest"],
            "match_digest": stable_hash(matches),
        })
    complete = _read(root / "COMPLETE.json"); _seal(complete, "complete_digest")
    estimate = sum(durations) / len(durations) * 3_936 / 3_600
    if not all((
        complete.get("status") == "complete", complete.get("cases") == 3,
        complete.get("all_certified") is True, complete.get("proposal_parity") is True,
        complete.get("case_manifest") == case_manifest,
        complete.get("case_manifest_digest") == stable_hash(case_manifest),
        complete.get("observed_case_seconds") == durations,
        complete.get("estimated_serial_3936_hours") == estimate,
        complete.get("historical_walk_forward_query_outcomes_opened") is False,
        complete.get("final_period_result_opened") is False,
        complete.get("production_promotion_authorized") is False,
    )):
        raise VerificationError("root aggregate differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "candidate_complete_digest": complete["complete_digest"],
        "candidate_complete_sha256": _sha(root / "COMPLETE.json"),
        "preregistration_digest": prereg["preregistration_digest"],
        "case_summaries": summaries, "case_summaries_digest": stable_hash(summaries),
        "forward_reverse_exact": True, "all_certified": True,
        "all_distinct_symbol_constraints_passed": True,
        "all_temporal_eligibility_checks_passed": True,
        "closed_inventory": True, "process_swap_kib_max": 0,
        "estimated_serial_3936_hours": estimate,
        "serial_scale_viable": False,
        "historical_walk_forward_query_outcomes_opened": False,
        "final_period_result_opened": False,
        "production_promotion_authorized": False,
    }
    return {**state, "verification_digest": stable_hash(state),
            "created_at": datetime.now(timezone.utc).isoformat()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.repository)
    if not args.dry_run:
        output = args.repository.resolve() / VERIFICATION / "VERIFIED.json"
        _atomic(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

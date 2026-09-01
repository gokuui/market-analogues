"""Preregister and build the complete T14-09 descriptive evidence-card store."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from experiments.m04r import m04r14_shadow_run as retrieval
from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from market_analogues.evidence_cards import HORIZONS, build_evidence_card
from market_analogues.types import stable_hash


SCHEMA = "m04r14-t14-09-evidence-card-store-v1"
PREREG_SCHEMA = "m04r14-t14-09-evidence-card-store-preregistration-v1"
PREREGISTRATION = Path(
    "experiments/m04r/m04r14_t14_09_evidence_card_store_preregistered.json"
)
CONTRACT = Path("config/m04r14-t14-09-evidence-card-contract.json")
CONTRACT_VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-evidence-card-contract-verification-v1/VERIFIED.json"
)
SYNTHETIC_GATE = Path(
    "config/data/analogues/m04r14/t14-09-evidence-card-synthetic-gate-v1/RESULT.json"
)
OUTCOME_STORE = Path("config/data/analogues/m04r14/t14-09-full-outcome-store-v1")
OUTCOME_VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-full-outcome-store-v1-verification/VERIFIED.json"
)
OUTPUT = Path("config/data/analogues/m04r14/t14-09-evidence-card-store-v1")
VERIFICATION = Path(
    "config/data/analogues/m04r14/t14-09-evidence-card-store-v1-verification"
)
RUNTIME_FILES = (
    "experiments/m04r/m04r14_t14_09_evidence_card_store.py",
    "experiments/m04r/verify_m04r14_t14_09_evidence_card_store.py",
    "experiments/m04r/m04r14_t14_09_evidence_card_oracle.py",
    "src/market_analogues/evidence_cards.py",
    "src/market_analogues/types.py",
)
QUERY_COUNT = 3270
LINK_COUNT = 65400
SEMANTIC_DIGEST_SCHEMA = "canonical-json-record-chunks-v1"
SEMANTIC_CHUNK_ROWS = 1024


class EvidenceStoreError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=True,
    )
    return result.stdout if raw else result.stdout.strip()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        return smoke._read(path)
    except Exception as exc:
        raise EvidenceStoreError(str(exc)) from exc


def _sha(path: Path) -> str:
    try:
        return smoke._sha(path)
    except Exception as exc:
        raise EvidenceStoreError(str(exc)) from exc


def _receipt_valid(value: Mapping[str, Any], *, timing: bool = False) -> bool:
    omitted = {"result_digest", "created_at"}
    if timing:
        omitted |= {"elapsed_seconds", "partition_elapsed_seconds"}
    return value.get("result_digest") == stable_hash({
        key: item for key, item in value.items() if key not in omitted
    })


def _runtime_manifest(repository: Path, head: str) -> dict[str, str]:
    tracked = set(str(_git(repository, "ls-tree", "-r", "--name-only", head)).splitlines())
    if any(name not in tracked for name in RUNTIME_FILES):
        raise EvidenceStoreError("runtime manifest contains an uncommitted file")
    return {
        name: sha256(_git(repository, "show", f"{head}:{name}", raw=True)).hexdigest()
        for name in RUNTIME_FILES
    }


def _validate_prerequisites(repository: Path) -> dict[str, Any]:
    contract, contract_raw = _read(repository / CONTRACT)
    contract_verified, contract_verified_raw = _read(repository / CONTRACT_VERIFICATION)
    synthetic, synthetic_raw = _read(repository / SYNTHETIC_GATE)
    outcome_seal, outcome_seal_raw = _read(repository / OUTCOME_STORE / "SEALED.json")
    outcome_verified, outcome_verified_raw = _read(repository / OUTCOME_VERIFICATION)
    if not all((
        contract.get("contract_digest") == contract_verified.get("contract_digest"),
        contract_verified.get("passed") is True,
        contract_verified.get("query_level_outcome_aggregation_opened") is False,
        _receipt_valid(contract_verified),
        synthetic.get("passed") is True,
        synthetic.get("contract_digest") == contract.get("contract_digest"),
        synthetic.get("query_level_real_outcome_aggregation_opened") is False,
        _receipt_valid(synthetic, timing=True),
        outcome_seal.get("passed") is True,
        outcome_seal.get("query_count") == QUERY_COUNT,
        outcome_seal.get("query_links") == LINK_COUNT,
        _receipt_valid(outcome_seal, timing=True),
        outcome_verified.get("passed") is True,
        outcome_verified.get("evidence_cards_authorized") is True,
        outcome_verified.get("store_result_digest") == outcome_seal.get("result_digest"),
        outcome_verified.get("query_count") == QUERY_COUNT,
        outcome_verified.get("query_links") == LINK_COUNT,
        _receipt_valid(outcome_verified),
        outcome_verified.get("production_promotion_authorized") is False,
    )):
        raise EvidenceStoreError("verified evidence-card prerequisites differ")
    return {
        "contract": contract,
        "contract_sha256": sha256(contract_raw).hexdigest(),
        "contract_verification": contract_verified,
        "contract_verification_sha256": sha256(contract_verified_raw).hexdigest(),
        "synthetic": synthetic,
        "synthetic_sha256": sha256(synthetic_raw).hexdigest(),
        "outcome_seal": outcome_seal,
        "outcome_seal_sha256": sha256(outcome_seal_raw).hexdigest(),
        "outcome_verification": outcome_verified,
        "outcome_verification_sha256": sha256(outcome_verified_raw).hexdigest(),
    }


def _source_file_bindings(repository: Path) -> dict[str, str]:
    names = (
        "query-match-links.parquet", "episode-outcomes.parquet",
        "future-paths.parquet", "COVERAGE.json", "SEALED.json",
    )
    return {name: _sha(repository / OUTCOME_STORE / name) for name in names}


def build_preregistration(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise EvidenceStoreError("globally clean Git worktree required")
    if (repository / OUTPUT).exists() or (repository / VERIFICATION).exists():
        raise EvidenceStoreError("evidence-card output/verification must be absent")
    prerequisites = _validate_prerequisites(repository)
    h0 = str(_git(repository, "rev-parse", "HEAD"))
    state = {
        "schema_version": PREREG_SCHEMA,
        "status": "frozen_before_complete_real_query_aggregation",
        "implementation_h0": h0,
        "runtime_files": _runtime_manifest(repository, h0),
        "contract_digest": prerequisites["contract"]["contract_digest"],
        "contract_sha256": prerequisites["contract_sha256"],
        "contract_verification_result_digest": prerequisites["contract_verification"]["result_digest"],
        "contract_verification_sha256": prerequisites["contract_verification_sha256"],
        "synthetic_gate_result_digest": prerequisites["synthetic"]["result_digest"],
        "synthetic_gate_sha256": prerequisites["synthetic_sha256"],
        "outcome_store_result_digest": prerequisites["outcome_seal"]["result_digest"],
        "outcome_store_sha256": prerequisites["outcome_seal_sha256"],
        "outcome_verification_result_digest": prerequisites["outcome_verification"]["result_digest"],
        "outcome_verification_sha256": prerequisites["outcome_verification_sha256"],
        "source_file_sha256": _source_file_bindings(repository),
        "query_count": QUERY_COUNT,
        "raw_neighbors_per_query": 20,
        "query_link_count": LINK_COUNT,
        "card_count": QUERY_COUNT,
        "summary_count": QUERY_COUNT,
        "output": str((repository / OUTPUT).resolve()),
        "verification": str((repository / VERIFICATION).resolve()),
        "query_level_real_outcome_aggregation_opened": False,
        "production_promotion_authorized": False,
    }
    return {**state, "preregistration_digest": stable_hash(state)}


def _sole_child(repository: Path, prereg_raw: bytes, h0: str) -> str:
    accepted: list[str] = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h0:
            continue
        for child in values[1:]:
            lineage = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(
                repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
            )).splitlines()
            if lineage != [child, h0] or changed != [PREREGISTRATION.as_posix()]:
                continue
            if _git(repository, "show", f"{child}:{PREREGISTRATION}", raw=True) == prereg_raw:
                accepted.append(child)
    if len(set(accepted)) != 1:
        raise EvidenceStoreError("expected one exact evidence preregistration-only child")
    return accepted[0]


def _validate_preregistration(repository: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise EvidenceStoreError("globally clean Git worktree required")
    prereg, raw = _read(repository / PREREGISTRATION)
    state = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("schema_version") != PREREG_SCHEMA \
            or prereg.get("preregistration_digest") != stable_hash(state):
        raise EvidenceStoreError("evidence preregistration seal differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_child(repository, raw, h0)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", h1, "HEAD"], cwd=repository,
    ).returncode:
        raise EvidenceStoreError("HEAD does not descend from evidence preregistration")
    for name, expected in prereg.get("runtime_files", {}).items():
        if any(sha256(value).hexdigest() != expected for value in (
            _git(repository, "show", f"{h0}:{name}", raw=True),
            _git(repository, "show", f"{h1}:{name}", raw=True),
            (repository / name).read_bytes(),
        )):
            raise EvidenceStoreError(f"runtime source drifted: {name}")
    prerequisites = _validate_prerequisites(repository)
    checks = {
        "contract_digest": prerequisites["contract"]["contract_digest"],
        "contract_sha256": prerequisites["contract_sha256"],
        "contract_verification_result_digest": prerequisites["contract_verification"]["result_digest"],
        "contract_verification_sha256": prerequisites["contract_verification_sha256"],
        "synthetic_gate_result_digest": prerequisites["synthetic"]["result_digest"],
        "synthetic_gate_sha256": prerequisites["synthetic_sha256"],
        "outcome_store_result_digest": prerequisites["outcome_seal"]["result_digest"],
        "outcome_store_sha256": prerequisites["outcome_seal_sha256"],
        "outcome_verification_result_digest": prerequisites["outcome_verification"]["result_digest"],
        "outcome_verification_sha256": prerequisites["outcome_verification_sha256"],
        "source_file_sha256": _source_file_bindings(repository),
        "query_count": QUERY_COUNT,
        "raw_neighbors_per_query": 20,
        "query_link_count": LINK_COUNT,
        "card_count": QUERY_COUNT,
        "summary_count": QUERY_COUNT,
        "output": str((repository / OUTPUT).resolve()),
        "verification": str((repository / VERIFICATION).resolve()),
        "query_level_real_outcome_aggregation_opened": False,
        "production_promotion_authorized": False,
    }
    if any(prereg.get(key) != value for key, value in checks.items()):
        raise EvidenceStoreError("evidence preregistered inputs differ")
    return prereg, prerequisites, h1


def _load_case_matches(
    repository: Path, query_id: str,
) -> dict[int, dict[str, Any]]:
    path = repository / retrieval.OUTPUT / "cases" / f"{query_id}.json"
    case, _ = _read(path)
    matches = case.get("matches")
    if case.get("query_episode_id") != query_id \
            or type(matches) is not list or len(matches) != 20:
        raise EvidenceStoreError(f"retrieval case inventory differs: {query_id}")
    return {rank: dict(match) for rank, match in enumerate(matches, 1)}


def _card_inputs(
    repository: Path,
) -> Iterator[tuple[list[dict[str, Any]], dict[str, str]]]:
    root = repository / OUTCOME_STORE
    links = pd.read_parquet(root / "query-match-links.parquet")
    outcomes = pd.read_parquet(root / "episode-outcomes.parquet")
    if len(links) != LINK_COUNT or len(outcomes) != 338268:
        raise EvidenceStoreError("sealed outcome frame row count differs")
    if links[["query_episode_id", "match_rank"]].duplicated().any() \
            or outcomes[["episode_id", "horizon_sessions"]].duplicated().any():
        raise EvidenceStoreError("sealed outcome identity is not unique")
    outcome_map = {
        (str(row["episode_id"]), int(row["horizon_sessions"])): smoke._plain(row)
        for row in outcomes.to_dict("records")
    }
    query_count = 0
    for query_id, frame in links.groupby("query_episode_id", sort=True):
        query_count += 1
        ordered = frame.sort_values("match_rank", kind="stable")
        if list(ordered.match_rank.astype(int)) != list(range(1, 21)):
            raise EvidenceStoreError(f"query ranks differ: {query_id}")
        rows: list[dict[str, Any]] = []
        case_by_rank = _load_case_matches(repository, str(query_id))
        for raw in ordered.to_dict("records"):
            link = smoke._plain(raw)
            rank = int(link["match_rank"])
            match = case_by_rank[rank]
            if not all((
                stable_hash(match) == link["match_digest"],
                str(match["episode_id"]) == link["matched_episode_id"],
                str(match["symbol"]) == link["matched_symbol"],
                str(match["cutoff"]) == link["matched_cutoff"],
                float(match["total_distance"]) == link["total_distance"],
            )):
                raise EvidenceStoreError(f"retrieval/outcome link differs: {query_id}:{rank}")
            eligibility = json.loads(str(link["outcome_eligibility_json"]))
            by_horizon = {
                str(horizon): outcome_map[(link["matched_episode_id"], horizon)]
                for horizon in HORIZONS
            }
            rows.append({
                **{key: link[key] for key in (
                    "query_case_id", "query_episode_id", "query_symbol", "query_cutoff",
                    "match_rank", "matched_episode_id", "matched_symbol", "matched_cutoff",
                    "total_distance", "match_digest", "candidate_case_result_digest",
                    "source_fingerprint",
                )},
                "component_distances": smoke._plain(match["component_distances"]),
                "quality_tier": str(match["quality_tier"]),
                "eligibility_by_horizon": eligibility,
                "outcomes_by_horizon": by_horizon,
                "primary_barrier_label": by_horizon["20"].get("barrier_label"),
                "future_path_episode_reference": link["matched_episode_id"],
            })
        provenance = {
            "contract_digest": "",  # replaced after prerequisite validation
            "outcome_store_result_digest": "",
            "outcome_verification_result_digest": "",
            "retrieval_case_result_digest": str(rows[0]["candidate_case_result_digest"]),
        }
        if len({row["candidate_case_result_digest"] for row in rows}) != 1:
            raise EvidenceStoreError(f"query case digest differs across links: {query_id}")
        yield rows, provenance
    if query_count != QUERY_COUNT:
        raise EvidenceStoreError("query group count differs")


def _raw_record(card: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    horizon20 = row["outcomes_by_horizon"]["20"]
    return {
        "query_case_id": card["query_case_id"],
        "query_episode_id": card["query_episode_id"],
        "query_symbol": card["query_symbol"],
        "query_cutoff": card["query_cutoff"],
        "match_rank": int(row["match_rank"]),
        "matched_episode_id": row["matched_episode_id"],
        "matched_symbol": row["matched_symbol"],
        "matched_cutoff": row["matched_cutoff"],
        "total_distance": row["total_distance"],
        "component_distances_json": json.dumps(row["component_distances"], sort_keys=True, separators=(",", ":")),
        "quality_tier": row["quality_tier"],
        "eligibility_by_horizon_json": json.dumps(row["eligibility_by_horizon"], sort_keys=True, separators=(",", ":")),
        "outcomes_by_horizon_json": json.dumps(row["outcomes_by_horizon"], sort_keys=True, separators=(",", ":"), allow_nan=False),
        "primary_barrier_label": horizon20.get("barrier_label"),
        "future_path_episode_reference": row["future_path_episode_reference"],
        "match_digest": row["match_digest"],
        "candidate_case_result_digest": row["candidate_case_result_digest"],
        "source_fingerprint": row["source_fingerprint"],
        "card_digest": card["card_digest"],
    }


def _summary_record(card: Mapping[str, Any]) -> dict[str, Any]:
    primary = card["primary_summary"]
    horizons = primary["eligible_by_horizon"]
    close20 = horizons["20"]["measures"]["close_return"]
    relative20 = horizons["20"]["measures"]["benchmark_relative_return"]
    unweighted = close20["unweighted"] or {}
    relative_unweighted = relative20["unweighted"] or {}
    raw = card["raw_sample_counts"]
    return {
        "query_case_id": card["query_case_id"],
        "query_episode_id": card["query_episode_id"],
        "query_symbol": card["query_symbol"],
        "query_cutoff": card["query_cutoff"],
        "card_digest": card["card_digest"],
        "card_path": f"cards/{card['query_episode_id']}.json",
        "raw_links": int(raw["raw_links"]),
        "raw_unique_episodes": int(raw["raw_unique_episodes"]),
        "raw_unique_symbols": int(raw["raw_unique_symbols"]),
        "same_symbol_links": int(raw["same_symbol_links"]),
        "primary_effective_rows": int(primary["effective_rows_before_horizon_eligibility"]),
        **{f"eligible_h{horizon}": int(horizons[str(horizon)]["eligible_rows"]) for horizon in HORIZONS},
        **{f"effective_sample_h{horizon}": float(horizons[str(horizon)]["weighted_effective_sample_size"]) for horizon in HORIZONS},
        "h20_close_count": int(close20["count"]),
        "h20_close_mean": unweighted.get("mean"),
        "h20_close_median": unweighted.get("median"),
        "h20_close_q25": unweighted.get("q25_linear"),
        "h20_close_q75": unweighted.get("q75_linear"),
        "h20_relative_count": int(relative20["count"]),
        "h20_relative_mean": relative_unweighted.get("mean"),
        "h20_relative_median": relative_unweighted.get("median"),
        "primary_barrier_counts_json": json.dumps(primary["primary_barrier_unweighted_counts"], sort_keys=True, separators=(",", ":")),
        "predictive_claim_status": card["predictive_claim_status"],
        "abstention_reasons_json": json.dumps(card["abstention_reasons"], separators=(",", ":")),
        "primary_summary_json": json.dumps(primary, sort_keys=True, separators=(",", ":"), allow_nan=False),
        "neighbor_sensitivity_json": json.dumps(card["neighbor_sensitivity"], sort_keys=True, separators=(",", ":"), allow_nan=False),
        "provenance_json": json.dumps(card["provenance"], sort_keys=True, separators=(",", ":")),
    }


def _html(summaries: Sequence[Mapping[str, Any]], seal_note: str) -> str:
    rows = []
    for row in summaries:
        median = row["h20_close_median"]
        relative = row["h20_relative_median"]
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['query_symbol']))}</td>"
            f"<td>{escape(str(row['query_cutoff']))}</td>"
            f"<td>{row['primary_effective_rows']}</td><td>{row['eligible_h20']}</td>"
            f"<td>{'' if median is None else f'{100 * median:.2f}%'}</td>"
            f"<td>{'' if relative is None else f'{100 * relative:.2f}%'}</td>"
            f"<td><a href=\"{escape(str(row['card_path']), quote=True)}\">JSON evidence</a></td>"
            "</tr>"
        )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>T14-09 evidence cards</title>
<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#18212b}input{width:100%;padding:.7rem;margin:.7rem 0 1rem;box-sizing:border-box}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:.45rem;border-bottom:1px solid #ddd;text-align:left}th{position:sticky;top:0;background:white}.warn{background:#fff4dc;border-left:5px solid #b9770e;padding:.8rem}</style></head><body>
<h1>Historical analogue evidence cards</h1><p class="warn"><b>Descriptive only.</b> These are not forecasts, calibrated probabilities or recommendations. Every card abstains pending walk-forward calibration.</p>
<p>Search the 3,270 sealed NASDAQ query cases. Medians summarize causally eligible, one-best-per-symbol historical analogues at the 20-session horizon.</p>
<input id="q" type="search" placeholder="Filter by symbol, cutoff or value" aria-label="Filter evidence cards">
<table><thead><tr><th>Query</th><th>Cutoff</th><th>Independent symbols</th><th>Eligible at 20</th><th>Median return</th><th>Median vs benchmark</th><th>Raw evidence</th></tr></thead><tbody>
""" + "\n".join(rows) + f"""</tbody></table><p>{escape(seal_note)}</p>
<script>const q=document.getElementById('q');q.addEventListener('input',()=>{{const s=q.value.toLowerCase();for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(s)}});</script></body></html>\n"""


def _manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names]


def _frame_digest(frame: pd.DataFrame, order: Sequence[str]) -> str:
    ordered = frame.sort_values(list(order), kind="stable").reset_index(drop=True)
    digest = sha256()
    digest.update(f"{SEMANTIC_DIGEST_SCHEMA}\0{len(ordered)}\0".encode())
    for start in range(0, len(ordered), SEMANTIC_CHUNK_ROWS):
        records = smoke._plain(
            ordered.iloc[start:start + SEMANTIC_CHUNK_ROWS].to_dict("records")
        )
        payload = json.dumps(
            records, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, prerequisites, h1 = _validate_preregistration(repository)
    final = repository / OUTPUT
    if final.exists() or final.is_symlink():
        raise EvidenceStoreError("complete evidence-card store is create-only")
    temporary = Path(tempfile.mkdtemp(prefix=".t14-09-evidence-store-", dir=final.parent))
    started = perf_counter()
    try:
        smoke._atomic_json(temporary / "RUN_STARTED.json", {
            "schema_version": SCHEMA, "status": "running",
            "preregistration_h1": h1,
            "preregistration_digest": prereg["preregistration_digest"],
            "query_level_real_outcome_aggregation_opened": True,
            "created_at": _now(),
        })
        (temporary / "cards").mkdir()
        raw_records: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        card_digests: list[dict[str, str]] = []
        for rows, provenance in _card_inputs(repository):
            provenance.update({
                "contract_digest": prerequisites["contract"]["contract_digest"],
                "outcome_store_result_digest": prerequisites["outcome_seal"]["result_digest"],
                "outcome_verification_result_digest": prerequisites["outcome_verification"]["result_digest"],
            })
            card = build_evidence_card(
                rows, contract_digest=prerequisites["contract"]["contract_digest"],
                provenance=provenance,
            )
            smoke._atomic_json(temporary / "cards" / f"{card['query_episode_id']}.json", card)
            raw_records.extend(_raw_record(card, row) for row in card["raw_analogue_rows"])
            summaries.append(_summary_record(card))
            card_digests.append({
                "query_episode_id": card["query_episode_id"],
                "card_digest": card["card_digest"],
            })
        raw_frame = pd.DataFrame(raw_records).sort_values(
            ["query_episode_id", "match_rank"], kind="stable",
        ).reset_index(drop=True)
        raw_frame["match_rank"] = raw_frame["match_rank"].astype("int64")
        summary_frame = pd.DataFrame(summaries).sort_values(
            ["query_episode_id"], kind="stable",
        ).reset_index(drop=True)
        if len(raw_frame) != LINK_COUNT or len(summary_frame) != QUERY_COUNT \
                or len(card_digests) != QUERY_COUNT:
            raise EvidenceStoreError("produced evidence coverage differs")
        smoke._atomic_parquet(temporary / "analogue-evidence.parquet", raw_frame)
        smoke._atomic_parquet(temporary / "query-evidence.parquet", summary_frame)
        html_text = _html(
            summary_frame.to_dict("records"),
            "See SEALED.json and each card's provenance for immutable source bindings.",
        )
        (temporary / "index.html").write_text(html_text)
        coverage_state = {
            "schema_version": SCHEMA, "status": "complete",
            "query_count": len(summary_frame), "card_count": len(card_digests),
            "raw_analogue_rows": len(raw_frame),
            "unique_matched_episodes": int(raw_frame.matched_episode_id.nunique()),
            "predictive_claim_status_counts": {
                str(key): int(value) for key, value in summary_frame.predictive_claim_status.value_counts().sort_index().items()
            },
            "primary_h20_eligible_minimum": int(summary_frame.eligible_h20.min()),
            "primary_h20_eligible_maximum": int(summary_frame.eligible_h20.max()),
            "cards_with_insufficient_effective_sample": int(summary_frame.abstention_reasons_json.str.contains("insufficient_effective_sample_size", regex=False).sum()),
            "query_level_real_outcome_aggregation_opened": True,
            "production_promotion_authorized": False,
        }
        coverage = {**coverage_state, "result_digest": stable_hash(coverage_state)}
        smoke._atomic_json(temporary / "COVERAGE.json", coverage)
        names = [
            "RUN_STARTED.json", "analogue-evidence.parquet", "query-evidence.parquet",
            "index.html", "COVERAGE.json",
        ] + [f"cards/{row['query_episode_id']}.json" for row in card_digests]
        manifest = _manifest(temporary, names)
        state = {
            "schema_version": SCHEMA, "status": "sealed", "passed": True,
            "preregistration_h1": h1,
            "preregistration_digest": prereg["preregistration_digest"],
            "contract_digest": prerequisites["contract"]["contract_digest"],
            "outcome_store_result_digest": prerequisites["outcome_seal"]["result_digest"],
            "outcome_verification_result_digest": prerequisites["outcome_verification"]["result_digest"],
            "query_count": QUERY_COUNT, "card_count": QUERY_COUNT,
            "raw_analogue_rows": LINK_COUNT,
            "card_digest_inventory_digest": stable_hash(card_digests),
            "semantic_digest_schema": SEMANTIC_DIGEST_SCHEMA,
            "raw_evidence_digest": _frame_digest(raw_frame, ("query_episode_id", "match_rank")),
            "query_evidence_digest": _frame_digest(summary_frame, ("query_episode_id",)),
            "coverage_result_digest": coverage["result_digest"],
            "file_count_excluding_seal": len(manifest),
            "file_manifest": manifest,
            "elapsed_seconds": perf_counter() - started,
            "query_level_real_outcome_aggregation_opened": True,
            "predictive_claims_emitted": False,
            "production_promotion_authorized": False,
        }
        deterministic = {key: value for key, value in state.items() if key != "elapsed_seconds"}
        seal = {**state, "result_digest": stable_hash(deterministic), "created_at": _now()}
        smoke._atomic_json(temporary / "SEALED.json", seal)
        os.replace(temporary, final)
        return seal
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-preregistration", action="store_true")
    parser.add_argument("--publish-preregistration", action="store_true")
    args = parser.parse_args(argv)
    if args.publish_preregistration and not args.build_preregistration:
        parser.error("--publish-preregistration requires --build-preregistration")
    repository = args.repository.resolve(strict=True)
    value = build_preregistration(repository) if args.build_preregistration else execute(repository)
    if args.publish_preregistration:
        smoke._atomic_json(repository / PREREGISTRATION, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

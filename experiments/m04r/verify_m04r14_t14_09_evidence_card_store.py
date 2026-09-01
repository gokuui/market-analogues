"""Independently reconstruct and verify the complete T14-09 evidence-card store."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.m04r import m04r14_shadow_run as retrieval
from experiments.m04r import m04r14_t14_09_outcome_smoke as smoke
from experiments.m04r.m04r14_t14_09_evidence_card_oracle import (
    HORIZONS,
    reference_card,
)
from market_analogues.types import stable_hash


SCHEMA = "m04r14-t14-09-evidence-card-store-verification-v1"
STORE_SCHEMA = "m04r14-t14-09-evidence-card-store-v1"
PREREG_SCHEMA = "m04r14-t14-09-evidence-card-store-preregistration-v1"
PREREGISTRATION = Path(
    "experiments/m04r/m04r14_t14_09_evidence_card_store_preregistered.json"
)
AMENDMENT = Path(
    "experiments/m04r/m04r14_t14_09_evidence_card_verification_amendment.json"
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
QUERY_COUNT = 3270
LINK_COUNT = 65400
SEMANTIC_DIGEST_SCHEMA = "canonical-json-record-chunks-v1"
SEMANTIC_CHUNK_ROWS = 1024


class EvidenceVerificationError(RuntimeError):
    pass


def _git(repository: Path, *args: str, raw: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, capture_output=True,
        text=not raw, check=True,
    )
    return result.stdout if raw else result.stdout.strip()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceVerificationError(f"regular JSON file required: {path}")
    raw = path.read_bytes()

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise EvidenceVerificationError(f"duplicate JSON key: {path}:{key}")
            result[key] = value
        return result

    value = json.loads(
        raw, object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            EvidenceVerificationError(f"non-finite JSON: {path}:{item}")
        ),
    )
    if type(value) is not dict:
        raise EvidenceVerificationError(f"JSON object required: {path}")
    return value, raw


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EvidenceVerificationError(f"regular file required: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _valid_receipt(value: Mapping[str, Any], *, timing: bool = False) -> bool:
    omitted = {"result_digest", "created_at"}
    if timing:
        omitted |= {"elapsed_seconds", "partition_elapsed_seconds"}
    return value.get("result_digest") == stable_hash({
        key: item for key, item in value.items() if key not in omitted
    })


def _sole_child(repository: Path, raw: bytes, h0: str) -> str:
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
            if lineage == [child, h0] and changed == [PREREGISTRATION.as_posix()] \
                    and _git(repository, "show", f"{child}:{PREREGISTRATION}", raw=True) == raw:
                accepted.append(child)
    if len(set(accepted)) != 1:
        raise EvidenceVerificationError("evidence preregistration lifecycle differs")
    return accepted[0]


def _sole_amendment_child(repository: Path, raw: bytes, h2: str) -> str:
    accepted: list[str] = []
    for line in str(_git(repository, "rev-list", "--all", "--children")).splitlines():
        values = line.split()
        if not values or values[0] != h2:
            continue
        for child in values[1:]:
            lineage = str(_git(repository, "rev-list", "--parents", "-n", "1", child)).split()
            changed = str(_git(
                repository, "diff-tree", "--no-commit-id", "--name-only", "-r", child,
            )).splitlines()
            if lineage == [child, h2] and changed == [AMENDMENT.as_posix()] \
                    and _git(repository, "show", f"{child}:{AMENDMENT}", raw=True) == raw:
                accepted.append(child)
    if len(set(accepted)) != 1:
        raise EvidenceVerificationError("verification amendment lifecycle differs")
    return accepted[0]


def _validate_amendment(
    repository: Path, prereg: Mapping[str, Any], h1: str,
) -> tuple[dict[str, Any], str]:
    amendment, raw = _read(repository / AMENDMENT)
    state = {key: value for key, value in amendment.items() if key != "amendment_digest"}
    if amendment.get("schema_version") != "m04r14-t14-09-evidence-verification-amendment-v1" \
            or amendment.get("amendment_digest") != stable_hash(state):
        raise EvidenceVerificationError("verification amendment seal differs")
    h2 = str(amendment.get("implementation_h2"))
    lineage = str(_git(repository, "rev-list", "--parents", "-n", "1", h2)).split()
    changed = str(_git(
        repository, "diff-tree", "--no-commit-id", "--name-only", "-r", h2,
    )).splitlines()
    if lineage != [h2, h1] or changed != [
        "experiments/m04r/m04r14_t14_09_evidence_card_oracle.py",
        "experiments/m04r/verify_m04r14_t14_09_evidence_card_store.py",
        "tests/test_m04r14_t14_09_evidence_card_store.py",
    ]:
        raise EvidenceVerificationError("verification correction commit scope differs")
    h3 = _sole_amendment_child(repository, raw, h2)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", h3, "HEAD"], cwd=repository,
    ).returncode:
        raise EvidenceVerificationError("HEAD does not descend from verification amendment")
    output_seal, output_raw = _read(repository / OUTPUT / "SEALED.json")
    expected = {
        "status": "frozen_after_first_verification_refusal_before_second_verification",
        "original_preregistration_h1": h1,
        "original_preregistration_digest": prereg["preregistration_digest"],
        "store_result_digest": output_seal.get("result_digest"),
        "store_result_sha256": sha256(output_raw).hexdigest(),
        "first_failure_query_episode_id": "21d70343dabd86cfab0ef272",
        "first_failure_field_class": "weighted_effective_sample_size",
        "producer_value": 14.620572929755886,
        "original_oracle_value": 14.620572929755884,
        "contract_formula": "(sum(weights) ** 2) / sum(weights ** 2)",
        "original_oracle_formula": "(sum(weights) * sum(weights)) / sum(weights ** 2)",
        "correction": "spell_the_independent_oracle_numerator_exactly_as_the_frozen_contract",
        "first_verification_receipt_published": False,
        "store_rebuilt_or_modified": False,
        "query_level_real_outcome_aggregation_opened": True,
        "production_promotion_authorized": False,
    }
    if any(amendment.get(key) != value for key, value in expected.items()):
        raise EvidenceVerificationError("verification amendment evidence differs")
    corrected = amendment.get("corrected_runtime_files")
    required = {
        "experiments/m04r/m04r14_t14_09_evidence_card_oracle.py",
        "experiments/m04r/verify_m04r14_t14_09_evidence_card_store.py",
    }
    if type(corrected) is not dict or set(corrected) != required:
        raise EvidenceVerificationError("corrected runtime inventory differs")
    for name, expected_hash in corrected.items():
        for revision in (h2, h3):
            if sha256(_git(repository, "show", f"{revision}:{name}", raw=True)).hexdigest() != expected_hash:
                raise EvidenceVerificationError(f"corrected runtime differs at {revision}: {name}")
        if _sha(repository / name) != expected_hash:
            raise EvidenceVerificationError(f"working corrected runtime differs: {name}")
    return amendment, h3


def _prerequisites(repository: Path) -> dict[str, Any]:
    contract, contract_raw = _read(repository / CONTRACT)
    contract_verified, contract_verified_raw = _read(repository / CONTRACT_VERIFICATION)
    synthetic, synthetic_raw = _read(repository / SYNTHETIC_GATE)
    outcome_seal, outcome_seal_raw = _read(repository / OUTCOME_STORE / "SEALED.json")
    outcome_verified, outcome_verified_raw = _read(repository / OUTCOME_VERIFICATION)
    if not all((
        contract_verified.get("passed") is True,
        contract_verified.get("contract_digest") == contract.get("contract_digest"),
        contract_verified.get("query_level_outcome_aggregation_opened") is False,
        _valid_receipt(contract_verified),
        synthetic.get("passed") is True,
        synthetic.get("contract_digest") == contract.get("contract_digest"),
        synthetic.get("query_level_real_outcome_aggregation_opened") is False,
        _valid_receipt(synthetic, timing=True),
        outcome_seal.get("passed") is True,
        outcome_seal.get("query_count") == QUERY_COUNT,
        outcome_seal.get("query_links") == LINK_COUNT,
        _valid_receipt(outcome_seal, timing=True),
        outcome_verified.get("passed") is True,
        outcome_verified.get("evidence_cards_authorized") is True,
        outcome_verified.get("store_result_digest") == outcome_seal.get("result_digest"),
        _valid_receipt(outcome_verified),
    )):
        raise EvidenceVerificationError("evidence-card prerequisite differs")
    return {
        "contract": contract,
        "contract_sha256": sha256(contract_raw).hexdigest(),
        "contract_verified": contract_verified,
        "contract_verified_sha256": sha256(contract_verified_raw).hexdigest(),
        "synthetic": synthetic,
        "synthetic_sha256": sha256(synthetic_raw).hexdigest(),
        "outcome_seal": outcome_seal,
        "outcome_seal_sha256": sha256(outcome_seal_raw).hexdigest(),
        "outcome_verified": outcome_verified,
        "outcome_verified_sha256": sha256(outcome_verified_raw).hexdigest(),
    }


def _source_hashes(repository: Path) -> dict[str, str]:
    return {name: _sha(repository / OUTCOME_STORE / name) for name in (
        "query-match-links.parquet", "episode-outcomes.parquet",
        "future-paths.parquet", "COVERAGE.json", "SEALED.json",
    )}


def _validate_preregistration(repository: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    prereg, raw = _read(repository / PREREGISTRATION)
    state = {key: value for key, value in prereg.items() if key != "preregistration_digest"}
    if prereg.get("schema_version") != PREREG_SCHEMA \
            or prereg.get("preregistration_digest") != stable_hash(state):
        raise EvidenceVerificationError("evidence preregistration differs")
    h0 = str(prereg.get("implementation_h0"))
    h1 = _sole_child(repository, raw, h0)
    corrected_names = {
        "experiments/m04r/m04r14_t14_09_evidence_card_oracle.py",
        "experiments/m04r/verify_m04r14_t14_09_evidence_card_store.py",
    }
    for name, expected in prereg.get("runtime_files", {}).items():
        for revision in (h0, h1):
            if sha256(_git(repository, "show", f"{revision}:{name}", raw=True)).hexdigest() != expected:
                raise EvidenceVerificationError(f"frozen runtime differs at {revision}: {name}")
        if name not in corrected_names and _sha(repository / name) != expected:
            raise EvidenceVerificationError(f"working runtime differs: {name}")
    amendment, _ = _validate_amendment(repository, prereg, h1)
    prerequisite = _prerequisites(repository)
    expected = {
        "contract_digest": prerequisite["contract"]["contract_digest"],
        "contract_sha256": prerequisite["contract_sha256"],
        "contract_verification_result_digest": prerequisite["contract_verified"]["result_digest"],
        "contract_verification_sha256": prerequisite["contract_verified_sha256"],
        "synthetic_gate_result_digest": prerequisite["synthetic"]["result_digest"],
        "synthetic_gate_sha256": prerequisite["synthetic_sha256"],
        "outcome_store_result_digest": prerequisite["outcome_seal"]["result_digest"],
        "outcome_store_sha256": prerequisite["outcome_seal_sha256"],
        "outcome_verification_result_digest": prerequisite["outcome_verified"]["result_digest"],
        "outcome_verification_sha256": prerequisite["outcome_verified_sha256"],
        "source_file_sha256": _source_hashes(repository),
        "query_count": QUERY_COUNT, "raw_neighbors_per_query": 20,
        "query_link_count": LINK_COUNT, "card_count": QUERY_COUNT,
        "summary_count": QUERY_COUNT,
        "output": str((repository / OUTPUT).resolve()),
        "verification": str((repository / VERIFICATION).resolve()),
        "query_level_real_outcome_aggregation_opened": False,
        "production_promotion_authorized": False,
    }
    if any(prereg.get(key) != value for key, value in expected.items()):
        raise EvidenceVerificationError("preregistered evidence input differs")
    return {**prereg, "verification_amendment_digest": amendment["amendment_digest"]}, prerequisite, h1


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def _case_matches(
    repository: Path, query_id: str,
) -> dict[int, dict[str, Any]]:
    case, _ = _read(repository / retrieval.OUTPUT / "cases" / f"{query_id}.json")
    matches = case.get("matches")
    if case.get("query_episode_id") != query_id \
            or type(matches) is not list or len(matches) != 20:
        raise EvidenceVerificationError(f"independent retrieval inventory differs: {query_id}")
    return {index: dict(match) for index, match in enumerate(matches, 1)}


def _reference_groups(
    repository: Path, prerequisite: Mapping[str, Any],
) -> list[dict[str, Any]]:
    root = repository / OUTCOME_STORE
    links = pd.read_parquet(root / "query-match-links.parquet")
    outcomes = pd.read_parquet(root / "episode-outcomes.parquet")
    if len(links) != LINK_COUNT or len(outcomes) != 338268 \
            or links[["query_episode_id", "match_rank"]].duplicated().any() \
            or outcomes[["episode_id", "horizon_sessions"]].duplicated().any():
        raise EvidenceVerificationError("independent source frame inventory differs")
    outcome_by_key = {
        (str(row["episode_id"]), int(row["horizon_sessions"])): _plain(row)
        for row in outcomes.to_dict("records")
    }
    cards = []
    for query_id, group in links.groupby("query_episode_id", sort=True):
        ordered = group.sort_values("match_rank", kind="stable")
        if list(ordered.match_rank.astype(int)) != list(range(1, 21)):
            raise EvidenceVerificationError(f"independent ranks differ: {query_id}")
        rows = []
        case_matches = _case_matches(repository, str(query_id))
        for link_raw in ordered.to_dict("records"):
            link = _plain(link_raw)
            rank = int(link["match_rank"])
            match = case_matches[rank]
            if stable_hash(match) != link["match_digest"] \
                    or str(match["episode_id"]) != link["matched_episode_id"] \
                    or str(match["symbol"]) != link["matched_symbol"] \
                    or str(match["cutoff"]) != link["matched_cutoff"] \
                    or float(match["total_distance"]) != link["total_distance"]:
                raise EvidenceVerificationError(f"independent match join differs: {query_id}:{rank}")
            horizon_outcomes = {
                str(horizon): outcome_by_key[(link["matched_episode_id"], horizon)]
                for horizon in HORIZONS
            }
            rows.append({
                **{key: link[key] for key in (
                    "query_case_id", "query_episode_id", "query_symbol", "query_cutoff",
                    "match_rank", "matched_episode_id", "matched_symbol", "matched_cutoff",
                    "total_distance", "match_digest", "candidate_case_result_digest",
                    "source_fingerprint",
                )},
                "component_distances": _plain(match["component_distances"]),
                "quality_tier": str(match["quality_tier"]),
                "eligibility_by_horizon": json.loads(str(link["outcome_eligibility_json"])),
                "outcomes_by_horizon": horizon_outcomes,
                "primary_barrier_label": horizon_outcomes["20"].get("barrier_label"),
                "future_path_episode_reference": link["matched_episode_id"],
            })
        if len({row["candidate_case_result_digest"] for row in rows}) != 1:
            raise EvidenceVerificationError("independent case digest differs within query")
        cards.append(reference_card(
            rows,
            contract_digest=prerequisite["contract"]["contract_digest"],
            provenance={
                "contract_digest": prerequisite["contract"]["contract_digest"],
                "outcome_store_result_digest": prerequisite["outcome_seal"]["result_digest"],
                "outcome_verification_result_digest": prerequisite["outcome_verified"]["result_digest"],
                "retrieval_case_result_digest": str(rows[0]["candidate_case_result_digest"]),
            },
        ))
    if len(cards) != QUERY_COUNT:
        raise EvidenceVerificationError("independent card count differs")
    return cards


def _visible_raw(card: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_case_id": card["query_case_id"], "query_episode_id": card["query_episode_id"],
        "query_symbol": card["query_symbol"], "query_cutoff": card["query_cutoff"],
        "match_rank": int(row["match_rank"]), "matched_episode_id": row["matched_episode_id"],
        "matched_symbol": row["matched_symbol"], "matched_cutoff": row["matched_cutoff"],
        "total_distance": row["total_distance"],
        "component_distances_json": json.dumps(row["component_distances"], sort_keys=True, separators=(",", ":")),
        "quality_tier": row["quality_tier"],
        "eligibility_by_horizon_json": json.dumps(row["eligibility_by_horizon"], sort_keys=True, separators=(",", ":")),
        "outcomes_by_horizon_json": json.dumps(row["outcomes_by_horizon"], sort_keys=True, separators=(",", ":"), allow_nan=False),
        "primary_barrier_label": row["outcomes_by_horizon"]["20"].get("barrier_label"),
        "future_path_episode_reference": row["future_path_episode_reference"],
        "match_digest": row["match_digest"],
        "candidate_case_result_digest": row["candidate_case_result_digest"],
        "source_fingerprint": row["source_fingerprint"], "card_digest": card["card_digest"],
    }


def _visible_summary(card: Mapping[str, Any]) -> dict[str, Any]:
    primary = card["primary_summary"]
    horizons = primary["eligible_by_horizon"]
    close = horizons["20"]["measures"]["close_return"]
    relative = horizons["20"]["measures"]["benchmark_relative_return"]
    close_unweighted = close["unweighted"] or {}
    relative_unweighted = relative["unweighted"] or {}
    counts = card["raw_sample_counts"]
    return {
        "query_case_id": card["query_case_id"], "query_episode_id": card["query_episode_id"],
        "query_symbol": card["query_symbol"], "query_cutoff": card["query_cutoff"],
        "card_digest": card["card_digest"], "card_path": f"cards/{card['query_episode_id']}.json",
        "raw_links": int(counts["raw_links"]), "raw_unique_episodes": int(counts["raw_unique_episodes"]),
        "raw_unique_symbols": int(counts["raw_unique_symbols"]), "same_symbol_links": int(counts["same_symbol_links"]),
        "primary_effective_rows": int(primary["effective_rows_before_horizon_eligibility"]),
        **{f"eligible_h{horizon}": int(horizons[str(horizon)]["eligible_rows"]) for horizon in HORIZONS},
        **{f"effective_sample_h{horizon}": float(horizons[str(horizon)]["weighted_effective_sample_size"]) for horizon in HORIZONS},
        "h20_close_count": int(close["count"]), "h20_close_mean": close_unweighted.get("mean"),
        "h20_close_median": close_unweighted.get("median"), "h20_close_q25": close_unweighted.get("q25_linear"),
        "h20_close_q75": close_unweighted.get("q75_linear"), "h20_relative_count": int(relative["count"]),
        "h20_relative_mean": relative_unweighted.get("mean"), "h20_relative_median": relative_unweighted.get("median"),
        "primary_barrier_counts_json": json.dumps(primary["primary_barrier_unweighted_counts"], sort_keys=True, separators=(",", ":")),
        "predictive_claim_status": card["predictive_claim_status"],
        "abstention_reasons_json": json.dumps(card["abstention_reasons"], separators=(",", ":")),
        "primary_summary_json": json.dumps(primary, sort_keys=True, separators=(",", ":"), allow_nan=False),
        "neighbor_sensitivity_json": json.dumps(card["neighbor_sensitivity"], sort_keys=True, separators=(",", ":"), allow_nan=False),
        "provenance_json": json.dumps(card["provenance"], sort_keys=True, separators=(",", ":")),
    }


def _records(frame: pd.DataFrame, order: Sequence[str]) -> list[dict[str, Any]]:
    return _plain(frame.sort_values(list(order), kind="stable").reset_index(drop=True).to_dict("records"))


def _manifest(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    return [{"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names]


def _frame_digest(frame: pd.DataFrame, order: Sequence[str]) -> str:
    ordered = frame.sort_values(list(order), kind="stable").reset_index(drop=True)
    digest = sha256()
    digest.update(f"{SEMANTIC_DIGEST_SCHEMA}\0{len(ordered)}\0".encode())
    for start in range(0, len(ordered), SEMANTIC_CHUNK_ROWS):
        records = _plain(ordered.iloc[start:start + SEMANTIC_CHUNK_ROWS].to_dict("records"))
        payload = json.dumps(
            records, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _require_card_equal(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if observed != expected:
        raise EvidenceVerificationError(
            f"independent card differs: {expected.get('query_episode_id', 'unknown')}"
        )


def verify(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    prereg, prerequisite, h1 = _validate_preregistration(repository)
    root = repository / OUTPUT
    if root.is_symlink() or not root.is_dir():
        raise EvidenceVerificationError("regular evidence-card output directory required")
    seal, seal_raw = _read(root / "SEALED.json")
    deterministic = {
        key: value for key, value in seal.items()
        if key not in {"elapsed_seconds", "result_digest", "created_at"}
    }
    if not all((
        seal.get("schema_version") == STORE_SCHEMA, seal.get("passed") is True,
        seal.get("result_digest") == stable_hash(deterministic),
        seal.get("preregistration_h1") == h1,
        seal.get("preregistration_digest") == prereg["preregistration_digest"],
        seal.get("contract_digest") == prerequisite["contract"]["contract_digest"],
        seal.get("outcome_store_result_digest") == prerequisite["outcome_seal"]["result_digest"],
        seal.get("outcome_verification_result_digest") == prerequisite["outcome_verified"]["result_digest"],
        seal.get("query_count") == QUERY_COUNT, seal.get("card_count") == QUERY_COUNT,
        seal.get("raw_analogue_rows") == LINK_COUNT,
        seal.get("query_level_real_outcome_aggregation_opened") is True,
        seal.get("predictive_claims_emitted") is False,
        seal.get("production_promotion_authorized") is False,
    )):
        raise EvidenceVerificationError("evidence-card store seal differs")
    cards_root = root / "cards"
    if cards_root.is_symlink() or not cards_root.is_dir():
        raise EvidenceVerificationError("regular card directory required")
    top_names = {path.name for path in root.iterdir()}
    if top_names != {
        "RUN_STARTED.json", "analogue-evidence.parquet", "query-evidence.parquet",
        "cards", "index.html", "COVERAGE.json", "SEALED.json",
    } or any(path.is_symlink() for path in root.iterdir()):
        raise EvidenceVerificationError("evidence output top-level closure differs")
    observed_card_names = sorted(path.name for path in cards_root.iterdir())
    if len(observed_card_names) != QUERY_COUNT or any(
        path.is_symlink() or not path.is_file() for path in cards_root.iterdir()
    ):
        raise EvidenceVerificationError("card file closure differs")
    expected_names = [
        "RUN_STARTED.json", "analogue-evidence.parquet", "query-evidence.parquet",
        "index.html", "COVERAGE.json",
    ] + [f"cards/{name}" for name in observed_card_names]
    if seal.get("file_count_excluding_seal") != len(expected_names) \
            or seal.get("file_manifest") != _manifest(root, expected_names):
        raise EvidenceVerificationError("evidence file manifest differs")

    expected_cards = _reference_groups(repository, prerequisite)
    expected_ids = [str(card["query_episode_id"]) for card in expected_cards]
    if observed_card_names != sorted(f"{query_id}.json" for query_id in expected_ids):
        raise EvidenceVerificationError("card filenames differ from independent queries")
    card_digests = []
    for expected in expected_cards:
        observed, _ = _read(cards_root / f"{expected['query_episode_id']}.json")
        _require_card_equal(observed, expected)
        card_digests.append({
            "query_episode_id": expected["query_episode_id"],
            "card_digest": expected["card_digest"],
        })
    raw_expected = [
        _visible_raw(card, row)
        for card in expected_cards for row in card["raw_analogue_rows"]
    ]
    summary_expected = [_visible_summary(card) for card in expected_cards]
    observed_raw = pd.read_parquet(root / "analogue-evidence.parquet")
    observed_summary = pd.read_parquet(root / "query-evidence.parquet")
    if _records(observed_raw, ("query_episode_id", "match_rank")) != raw_expected:
        raise EvidenceVerificationError("independent raw evidence differs")
    if _records(observed_summary, ("query_episode_id",)) != summary_expected:
        raise EvidenceVerificationError("independent query evidence differs")
    if seal.get("semantic_digest_schema") != SEMANTIC_DIGEST_SCHEMA \
            or seal.get("card_digest_inventory_digest") != stable_hash(card_digests) \
            or seal.get("raw_evidence_digest") != _frame_digest(
                observed_raw, ("query_episode_id", "match_rank"),
            ) or seal.get("query_evidence_digest") != _frame_digest(
                observed_summary, ("query_episode_id",),
            ):
        raise EvidenceVerificationError("evidence semantic digest differs")

    coverage, _ = _read(root / "COVERAGE.json")
    coverage_state = {key: value for key, value in coverage.items() if key != "result_digest"}
    expected_coverage = {
        "schema_version": STORE_SCHEMA, "status": "complete",
        "query_count": QUERY_COUNT, "card_count": QUERY_COUNT,
        "raw_analogue_rows": LINK_COUNT,
        "unique_matched_episodes": len({row["matched_episode_id"] for row in raw_expected}),
        "predictive_claim_status_counts": {"abstain": QUERY_COUNT},
        "primary_h20_eligible_minimum": min(row["eligible_h20"] for row in summary_expected),
        "primary_h20_eligible_maximum": max(row["eligible_h20"] for row in summary_expected),
        "cards_with_insufficient_effective_sample": sum(
            "insufficient_effective_sample_size" in row["abstention_reasons_json"]
            for row in summary_expected
        ),
        "query_level_real_outcome_aggregation_opened": True,
        "production_promotion_authorized": False,
    }
    if coverage_state != expected_coverage \
            or coverage.get("result_digest") != stable_hash(coverage_state) \
            or seal.get("coverage_result_digest") != coverage.get("result_digest"):
        raise EvidenceVerificationError("evidence coverage differs")
    html = (root / "index.html").read_text()
    if not all((
        "Descriptive only." in html, "not forecasts" in html,
        "pending walk-forward calibration" in html,
        html.count("JSON evidence") == QUERY_COUNT,
    )):
        raise EvidenceVerificationError("searchable HTML safety/content differs")
    state = {
        "schema_version": SCHEMA, "status": "verified", "passed": True,
        "store_result_digest": seal["result_digest"],
        "store_result_sha256": sha256(seal_raw).hexdigest(),
        "contract_digest": prerequisite["contract"]["contract_digest"],
        "verification_amendment_digest": prereg["verification_amendment_digest"],
        "query_count": QUERY_COUNT, "verified_cards": QUERY_COUNT,
        "verified_raw_analogue_rows": LINK_COUNT,
        "exact_independent_card_equality": True,
        "exact_independent_raw_equality": True,
        "exact_independent_summary_equality": True,
        "strict_file_manifest_closure": True,
        "query_level_real_outcome_aggregation_opened": True,
        "predictive_claims_emitted": False,
        "walk_forward_authorized": True,
        "production_promotion_authorized": False,
    }
    return {**state, "result_digest": stable_hash(state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise EvidenceVerificationError("evidence verification root exists")
    path.mkdir(parents=False)
    descriptor = os.open(path / "VERIFIED.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps({
            **value, "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = args.repository.resolve(strict=True)
    result = verify(repository)
    if not args.dry_run:
        _publish(repository / VERIFICATION, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

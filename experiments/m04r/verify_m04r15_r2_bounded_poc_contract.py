"""Verify the selection-blind R2-03 bounded-POC contract."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq


SCHEMA = "m04r15-r2-bounded-poc-contract-verification-v1"
CONTRACT = Path("config/m04r15-r2-bounded-poc-contract-v1.json")
LINKS = Path("config/data/analogues/m04r14/t14-09-full-outcome-store-v1/query-match-links.parquet")
R202_RESULT = Path("config/data/analogues/m04r15/r2-stability-synthetic-gate-v1/RESULT.json")
R202_VERIFIED = Path("config/data/analogues/m04r15/r2-stability-synthetic-gate-v1-verification/VERIFIED.json")
OUTPUT = Path("config/data/analogues/m04r15/r2-bounded-poc-contract-verification-v1")
TOP_KEYS = {"schema_version", "contract_id", "status", "upstream", "selection",
            "computation", "verification", "claim_boundary", "contract_digest"}


class PocContractError(RuntimeError): pass


def _require(value: bool, message: str) -> None:
    if not value: raise PocContractError(message)


def _stable(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"regular JSON required: {path}")
    raw = path.read_bytes(); value = json.loads(raw); _require(type(value) is dict, "JSON object required")
    return value, raw


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def validate(repository: Path, contract_path: Path | None = None) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    contract, raw = _read((contract_path or repository / CONTRACT).resolve(strict=True))
    _require(set(contract) == TOP_KEYS, "POC contract keys differ")
    state = {key: value for key, value in contract.items() if key != "contract_digest"}
    _require(all((contract.get("schema_version") == "m04r15-r2-bounded-poc-contract-v1",
                  contract.get("contract_id") == "r2-selection-blind-32-query-consumed-data-poc-v1",
                  contract.get("status") == "frozen_before_bounded_real_future_path_access",
                  contract.get("contract_digest") == _stable(state))), "POC contract identity differs")
    upstream = contract["upstream"]; producer, _ = _read(repository / R202_RESULT)
    verified, _ = _read(repository / R202_VERIFIED)
    _require(all((upstream == {
        "r2_contract_digest": "6bec009810afce7c58508fca28179579f5904382376dc9c5bce74aa74f41e08c",
        "r202_producer_result_digest": producer.get("result_digest"),
        "r202_producer_sha256": _sha(repository / R202_RESULT),
        "r202_verification_result_digest": verified.get("result_digest"),
        "r202_verification_sha256": _sha(repository / R202_VERIFIED),
    }, producer.get("passed") is True, verified.get("passed") is True,
        verified.get("bounded_consumed_data_poc_authorized") is True,
        verified.get("real_future_path_store_opened") is False)), "POC upstream differs")
    selection = contract["selection"]
    _require(set(selection) == {"source_column", "method", "sample_size", "selection_digest",
                                "query_case_ids", "future_outcomes_or_path_values_used"},
             "POC selection keys differ")
    ids = sorted(set(pq.read_table(repository / LINKS, columns=["query_case_id"])
                     .column(0).to_pylist()))
    selected = sorted(ids, key=lambda query: (
        sha256((upstream["r2_contract_digest"] + "\0" + query).encode()).hexdigest(), query,
    ))[:32]
    _require(all((selection.get("source_column") == "query_case_id_from_verified_query_match_links",
                  selection.get("method") == "first_32_by_ascending_sha256_of_utf8_r2_contract_digest_nul_query_case_id_then_query_id",
                  selection.get("sample_size") == 32,
                  selection.get("query_case_ids") == selected,
                  selection.get("selection_digest") == _stable(selected),
                  selection.get("future_outcomes_or_path_values_used") is False)), "POC selection differs")
    _require(contract["computation"] == {
        "views": ["absolute_close_return", "benchmark_relative_close_return"],
        "horizon_sessions": 60, "bootstrap_replicates": 256,
        "threshold_changes_authorized": False, "full_query_build_authorized": False,
        "selected_episode_filter_only": True, "record_wall_cpu_peak_rss_and_input_rows": True,
    }, "POC computation differs")
    _require(contract["verification"] == {
        "exact_selection_reconstruction": True, "all_20_raw_links_each_query": True,
        "outcome_mutation_must_not_change_primary_cohort_digest": True,
        "repeat_identity": True, "independent_reconstruction_required": True,
    }, "POC verification differs")
    _require(contract["claim_boundary"] == {
        "descriptive_poc_only": True, "predictive_claim_authorized": False,
        "production_promotion_authorized": False, "trading_claim_authorized": False,
    }, "POC claim boundary differs")
    result_state = {"schema_version": SCHEMA, "status": "verified_before_bounded_real_future_path_access",
                    "passed": True, "contract_digest": contract["contract_digest"],
                    "contract_sha256": sha256(raw).hexdigest(), "query_population": len(ids),
                    "selected_query_count": len(selected), "selection_digest": _stable(selected),
                    "columns_opened": ["query_case_id"], "future_path_store_opened": False,
                    "bounded_consumed_data_poc_authorized": True,
                    "full_query_build_authorized": False, "predictive_claim_authorized": False,
                    "production_promotion_authorized": False, "trading_claim_authorized": False}
    return {**result_state, "result_digest": _stable(result_state)}


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), "POC contract verification exists")
    path.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=".r2-poc-contract-", dir=path.parent))
    try:
        target=temporary/"VERIFIED.json"; descriptor=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        with os.fdopen(descriptor,"wb") as handle:
            handle.write((json.dumps({**value,"created_at":datetime.now(timezone.utc).isoformat()},indent=2,sort_keys=True)+"\n").encode());handle.flush();os.fsync(handle.fileno())
        os.rename(temporary,path)
    except Exception:
        try:
            if (temporary/"VERIFIED.json").exists():(temporary/"VERIFIED.json").unlink()
            temporary.rmdir()
        except OSError:pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--repository",type=Path,required=True);parser.add_argument("--contract",type=Path);parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv);value=validate(args.repository,args.contract)
    if not args.dry_run:_publish(args.repository.resolve(strict=True)/OUTPUT,value)
    print(json.dumps(value,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())

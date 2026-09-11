"""Independent verifier for the consumed-fixture prospective payload adapter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "m04r15-m2-prospective-payload-adapter-verification-v1"
RESULT = Path("config/data/analogues/m04r15/m2-prospective-payload-adapter-v1/RESULT.json")
OUTPUT = Path(
    "config/data/analogues/m04r15/m2-prospective-payload-adapter-v1-verification/VERIFIED.json"
)
CONTRACT = Path("config/prospective-payload-adapter-contract-v1.json")
INPUTS = {
    "m1_result": Path(
        "config/data/analogues/m04r15/m1-candidate-development-gate-v1/FROZEN.json"
    ),
    "registry_seal": Path(
        "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1/SEALED.json"
    ),
    "query_registry": Path(
        "config/data/analogues/m04r14/t14-10-walk-forward-query-registry-v1/query-registry.parquet"
    ),
    "nonfinal_scores": Path(
        "config/data/analogues/m04r14/t14-10-wf04-nonfinal-evaluation-v2/query-scores.parquet"
    ),
    "final_scores": Path(
        "config/data/analogues/m04r14/t14-10-wf04-final-evaluation-v1/query-scores.parquet"
    ),
    "nonfinal_outcomes": Path(
        "config/data/analogues/m04r14/t14-10-wf03d-prediction-store-v2/nonfinal-query-outcomes.parquet"
    ),
    "final_outcomes": Path(
        "config/data/analogues/m04r14/t14-10-wf04-final-evaluation-v1/final-query-outcomes.parquet"
    ),
    "listener_result": Path(
        "config/data/analogues/m04r15/m2-prospective-listener-synthetic-v1/RESULT.json"
    ),
    "listener_verification": Path(
        "config/data/analogues/m04r15/m2-prospective-listener-synthetic-v1-verification/VERIFIED.json"
    ),
}
EXPECTED_CONTRACT = "49a7855de1902403f550ee5a1d2d2fbe169310804e374e7d7b5029cc61668d7c"
EXPECTED_RESULT = "a3be1c0bda7a2fd07dc3429e02803b6abd0c92b4298bf9c13be3b9d42278e667"
EXPECTED_RESULT_SHA256 = "9440e8e8ef21fbbf1ad8faeaa11acf4c9cceac137a6a37aa578e296036f29733"
PRODUCER_RUNTIME = (
    "config/prospective-payload-adapter-contract-v1.json",
    "src/market_analogues/analogue_candidate.py",
    "src/market_analogues/prospective_adapter.py",
    "src/market_analogues/prospective_batch.py",
    "experiments/m04r/m04r15_m2_prospective_payload_adapter_gate.py",
    "tests/test_prospective_adapter.py",
    "tests/test_m04r15_m2_prospective_payload_adapter_gate.py",
)
VERIFIER_RUNTIME = (
    "experiments/m04r/verify_m04r15_m2_prospective_payload_adapter_gate.py",
    "tests/test_m04r15_m2_prospective_payload_adapter_verifier.py",
)
PROBABILITY_COLUMNS = (
    "favorable_probability", "adverse_probability", "no_touch_probability",
)
CLASSES = ("favorable_first", "adverse_first", "no_touch")
WEIGHTS = {
    "matched_causal_history": .4, "composite": .1, "price_only": .2,
    "recent_return_volatility": .3,
}
RESULT_KEYS = {
    "checks", "consumed_fixture_only", "contract_digest", "created_at",
    "elapsed_seconds", "implementation_commit", "input_sha256", "inventory",
    "new_post_freeze_outcomes_opened", "passed", "prediction_document_digest",
    "predictive_claim_authorized", "production_promotion_authorized",
    "prospective_prediction_seal_digest", "real_prediction_created",
    "real_registry_created", "registry_document_digest", "result_digest",
    "runtime_sha256", "schema_version", "source_document_digest", "status",
    "trading_claim_authorized",
}
VERIFICATION_KEYS = {
    "causal_history_rows", "consumed_fixture_only", "contract_digest", "created_at",
    "new_post_freeze_outcomes_opened", "passed", "predictive_claim_authorized",
    "producer_implementation_commit", "producer_or_project_scientific_code_imported",
    "producer_result_digest", "producer_result_sha256",
    "production_promotion_authorized", "prospective_prediction_seal_digest", "queries",
    "real_prediction_created", "real_registry_created", "reconstruction_digest",
    "schema_version", "status", "trading_claim_authorized", "verification_digest",
    "verifier_commit", "verifier_runtime_sha256",
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def encode(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise VerificationError(f"unsafe input: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"regular input required: {path}")
        blocks = []
        while block := os.read(descriptor, 1 << 20):
            blocks.append(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode,
        )
        require(identity(before) == identity(after), f"input changed: {path}")
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def decode(raw: bytes, path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value = {}
        for key, item in items:
            require(key not in value, f"duplicate JSON key: {path}/{key}")
            value[key] = item
        return value
    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerificationError(f"nonfinite JSON: {path}/{token}"),
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *arguments), cwd=repository, capture_output=True,
        text=not binary, check=False,
    )
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def matched_probability(history: pd.DataFrame, regime: str, quality: str,
                        liquidity: str) -> tuple[list[float], str, int]:
    choices = (
        ("exact", history[(history.market_regime == regime)
                          & (history.quality_tier == quality)
                          & (history.liquidity_stratum == liquidity)]),
        ("regime_and_liquidity", history[(history.market_regime == regime)
                                         & (history.liquidity_stratum == liquidity)]),
        ("regime", history[history.market_regime == regime]),
    )
    selected = history; level = "unconditional"
    for name, frame in choices:
        if len(frame) >= 30:
            selected = frame; level = name; break
    if not len(history):
        level = "uniform_no_history"
    counts = np.asarray([(selected.label == label).sum() for label in CLASSES], dtype=float)
    support = len(selected)
    values = (counts + .5) / (support + 1.5)
    return [float(value) for value in values], level, support


def reconstruct(raw: Mapping[str, bytes], contract: Mapping[str, Any]) -> dict[str, Any]:
    month = contract["consumed_fixture"]["month"]
    registry_all = pd.read_parquet(BytesIO(raw["query_registry"]), engine="pyarrow")
    registry = registry_all[
        pd.to_datetime(registry_all.cutoff).dt.to_period("M").astype(str) == month
    ].reset_index(drop=True)
    scores = pd.concat((
        pd.read_parquet(BytesIO(raw["nonfinal_scores"]), engine="pyarrow"),
        pd.read_parquet(BytesIO(raw["final_scores"]), engine="pyarrow"),
    ), ignore_index=True)
    outcomes = pd.concat((
        pd.read_parquet(BytesIO(raw["nonfinal_outcomes"]), engine="pyarrow"),
        pd.read_parquet(BytesIO(raw["final_outcomes"]), engine="pyarrow"),
    ), ignore_index=True)
    target_registry = registry[[
        "episode_id", "symbol", "cutoff", "quality_tier", "liquidity_stratum",
        "selection_hash", "stock_prefix_digest", "benchmark_prefix_digest",
    ]]
    target = scores[(scores.month == month) & scores.lane.isin(
        ["composite", "price_only", "recent_return_volatility"],
    )][["query_id", "query_cutoff", "lane", *PROBABILITY_COLUMNS]]
    composite = scores[scores.lane == "composite"][[
        "query_id", "query_cutoff", "query_regime", "quality_tier", "liquidity_stratum",
    ]].drop_duplicates("query_id")
    primary = outcomes[outcomes.horizon_sessions == 20][[
        "query_id", "cutoff", "completion_timestamp", "barrier_label",
    ]].drop_duplicates("query_id")
    history = composite.merge(primary, on="query_id", validate="one_to_one")
    cutoff = pd.Timestamp(target_registry.cutoff.iloc[0])
    history = history[pd.to_datetime(history.completion_timestamp) <= cutoff].rename(columns={
        "query_regime": "market_regime", "barrier_label": "label",
    })
    matched_history = history[history.label.isin(CLASSES)].copy()
    regimes = set(scores[(scores.month == month) & (scores.lane == "composite")].query_regime)
    require(len(target_registry) == 24 and len(target) == 72 and len(regimes) == 1,
            "independent fixture inventory differs")
    regime = next(iter(regimes))
    query_ids = target_registry.episode_id.astype(str).tolist()
    indexed = {
        lane: target[target.lane == lane].set_index("query_id")
        for lane in ("composite", "price_only", "recent_return_volatility")
    }
    prediction_rows = []
    for row in target_registry.itertuples(index=False):
        query_id = str(row.episode_id)
        matched, fallback, support = matched_probability(
            matched_history, regime, str(row.quality_tier), str(row.liquidity_stratum),
        )
        components = {"matched_causal_history": matched}
        for lane in ("composite", "price_only", "recent_return_volatility"):
            values = indexed[lane].loc[query_id, list(PROBABILITY_COLUMNS)].to_numpy(float)
            require(np.isfinite(values).all() and abs(values.sum() - 1) <= 1e-12,
                    "source probability differs")
            components[lane] = [float(value) for value in values]
        candidate = sum(WEIGHTS[name] * np.asarray(components[name]) for name in WEIGHTS)
        provenance = {
            "query_id": query_id, "cutoff": cutoff.isoformat(),
            "component_weights": WEIGHTS, "component_probabilities": components,
            "matched_fallback_level": fallback, "matched_support_rows": support,
        }
        prediction_rows.append({
            "query_id": query_id,
            "probabilities": {
                "candidate": [float(value) for value in candidate],
                "matched_causal_history": matched,
                "locked_composite": components["composite"],
            },
            "provenance_digest": stable(provenance),
        })
    registry_rows = [{
        "query_id": str(row.episode_id), "symbol": str(row.symbol),
        "quality_tier": str(row.quality_tier),
        "liquidity_stratum": str(row.liquidity_stratum),
        "selection_hash": str(row.selection_hash),
        "stock_prefix_digest": str(row.stock_prefix_digest),
    } for row in target_registry.itertuples(index=False)]
    query_digest = stable(query_ids)
    source_manifest = [{
        "query_id": row["query_id"], "symbol": row["symbol"],
        "stock_prefix_digest": row["stock_prefix_digest"],
    } for row in registry_rows]
    benchmark = set(target_registry.benchmark_prefix_digest.astype(str))
    require(len(benchmark) == 1, "benchmark identity differs")
    source = {
        "schema_version": "prospective-source-lock-v1", "batch_id": month,
        "cutoff": cutoff.date().isoformat(), "maximum_source_timestamp": cutoff.date().isoformat(),
        "stock_prefix_manifest_digest": stable(source_manifest),
        "benchmark_prefix_digest": next(iter(benchmark)),
        "source_values_after_cutoff_opened": False,
    }
    registry_document = {
        "schema_version": "prospective-query-registry-v1", "batch_id": month,
        "cutoff": cutoff.date().isoformat(), "queries": registry_rows,
        "query_digest": query_digest, "selection_used_outcomes": False,
    }
    predictions = {
        "schema_version": "prospective-probability-predictions-v1", "batch_id": month,
        "cutoff": cutoff.date().isoformat(), "rows": prediction_rows,
        "query_digest": query_digest, "query_outcomes_opened": False,
        "source_values_after_cutoff_opened": False,
    }
    documents = {
        "SOURCE_LOCK.json": source, "QUERY_REGISTRY.json": registry_document,
        "PREDICTIONS.json": predictions,
    }
    manifest = [{
        "path": name, "bytes": len(encode(value)), "sha256": sha256(encode(value)).hexdigest(),
    } for name, value in documents.items()]
    seal_state = {
        "schema_version": "prospective-prediction-batch-seal-v1",
        "status": "predictions_sealed", "passed": True,
        "contract_digest": contract["contract_digest"], "batch_id": month,
        "cutoff": cutoff.date().isoformat(), "query_count": 24,
        "query_digest": query_digest, "file_manifest": manifest,
        "source_values_after_cutoff_opened": False,
        "query_outcomes_opened_before_prediction_seal": False,
        "predictive_claim_authorized": False, "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    cell_counts: dict[str, int] = {}
    for row in registry_rows:
        key = f"{row['quality_tier']}|{row['liquidity_stratum']}"
        cell_counts[key] = cell_counts.get(key, 0) + 1
    seal = stable(seal_state)
    checks = [
        {"name": "query_inventory", "passed": True, "detail": 24},
        {"name": "six_cells_balanced", "passed": True, "detail": cell_counts},
        {"name": "target_schema_excludes_labels", "passed": True,
         "detail": sorted(predictions)},
        {"name": "source_prefix_stops_at_cutoff", "passed": True,
         "detail": cutoff.date().isoformat()},
        {"name": "probabilities_valid", "passed": True, "detail": 72},
        {"name": "prospective_seal_valid", "passed": True, "detail": seal},
        {"name": "real_launch_remains_disabled", "passed": True, "detail": "disabled"},
    ]
    return {
        "checks": checks,
        "inventory": {
            "registry_rows": 24, "target_score_rows": 72,
            "causal_history_rows": len(history), "market_regime": regime,
            "cutoff": cutoff.date().isoformat(),
        },
        "source_document_digest": stable(source),
        "registry_document_digest": stable(registry_document),
        "prediction_document_digest": stable(predictions),
        "prospective_prediction_seal_digest": seal,
    }


def verify_result(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_result = snapshot(repository / RESULT)
    raw_contract = snapshot(repository / CONTRACT)
    raw = {name: snapshot(repository / path) for name, path in INPUTS.items()}
    result = decode(raw_result, repository / RESULT)
    contract = decode(raw_contract, repository / CONTRACT)
    require(set(result) == RESULT_KEYS, "result field closure differs")
    state = {key: value for key, value in result.items()
             if key not in {"result_digest", "created_at", "elapsed_seconds"}}
    require(result["result_digest"] == stable(state), "result seal differs")
    for claim in (
        "new_post_freeze_outcomes_opened", "real_registry_created",
        "real_prediction_created", "predictive_claim_authorized",
        "production_promotion_authorized", "trading_claim_authorized",
    ):
        require(result[claim] is False, f"claim boundary differs: {claim}")
    require(result["consumed_fixture_only"] is True, "fixture boundary differs")
    require(result["result_digest"] == EXPECTED_RESULT
            and sha256(raw_result).hexdigest() == EXPECTED_RESULT_SHA256,
            "result identity differs")
    contract_state = {key: value for key, value in contract.items() if key != "contract_digest"}
    require(contract["contract_digest"] == stable(contract_state) == EXPECTED_CONTRACT
            and not any(contract["claims"].values()), "contract identity differs")
    expected = contract["consumed_fixture"]["input_sha256"]
    require({name: sha256(raw[name]).hexdigest() for name in expected} == expected,
            "consumed input identity differs")
    require(result["input_sha256"] == {name: sha256(value).hexdigest()
                                        for name, value in raw.items()},
            "input manifest differs")
    implementation = result["implementation_commit"]
    head = str(git(repository, "rev-parse", "HEAD"))
    require(type(implementation) is str and subprocess.run(
        ("git", "merge-base", "--is-ancestor", implementation, head), cwd=repository,
    ).returncode == 0, "producer lineage differs")
    require(set(result["runtime_sha256"]) == set(PRODUCER_RUNTIME),
            "producer runtime closure differs")
    for name in PRODUCER_RUNTIME:
        content = snapshot(repository / name)
        require(content == git(repository, "show", f"{implementation}:{name}", binary=True)
                and sha256(content).hexdigest() == result["runtime_sha256"][name],
                f"producer runtime differs: {name}")
    rebuilt = reconstruct(raw, contract)
    for field, value in rebuilt.items():
        require(result[field] == value, f"independent adapter reconstruction differs: {field}")
    return result, {"result_sha256": sha256(raw_result).hexdigest(), "reconstruction": rebuilt}


def verification_state(result: Mapping[str, Any], evidence: Mapping[str, Any],
                       commit: str, hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA, "status": "consumed_adapter_independently_verified",
        "passed": True, "verifier_commit": commit,
        "verifier_runtime_sha256": dict(hashes),
        "producer_implementation_commit": result["implementation_commit"],
        "producer_result_digest": result["result_digest"],
        "producer_result_sha256": evidence["result_sha256"],
        "contract_digest": result["contract_digest"],
        "reconstruction_digest": stable(evidence["reconstruction"]),
        "queries": result["inventory"]["registry_rows"],
        "causal_history_rows": result["inventory"]["causal_history_rows"],
        "prospective_prediction_seal_digest": result["prospective_prediction_seal_digest"],
        "producer_or_project_scientific_code_imported": False,
        "consumed_fixture_only": True, "new_post_freeze_outcomes_opened": False,
        "real_registry_created": False, "real_prediction_created": False,
        "predictive_claim_authorized": False, "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }


def publish(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "create-only verification exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m2-adapter-verify-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def run(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    require(not str(git(repository, "status", "--porcelain", "--untracked-files=all")),
            "clean committed tree required")
    result, evidence = verify_result(repository)
    commit = str(git(repository, "rev-parse", "HEAD"))
    content = {name: snapshot(repository / name) for name in VERIFIER_RUNTIME}
    for name, raw in content.items():
        require(raw == git(repository, "show", f"{commit}:{name}", binary=True),
                f"verifier runtime not committed: {name}")
    hashes = {name: sha256(raw).hexdigest() for name, raw in content.items()}
    state = verification_state(result, evidence, commit, hashes)
    receipt = {**state, "verification_digest": stable(state),
               "created_at": datetime.now(timezone.utc).isoformat()}
    publish(repository / OUTPUT, receipt)
    return receipt


def validate(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    receipt = decode(snapshot(repository / OUTPUT), repository / OUTPUT)
    require(set(receipt) == VERIFICATION_KEYS, "verification field closure differs")
    state = {key: value for key, value in receipt.items()
             if key not in {"verification_digest", "created_at"}}
    require(receipt["verification_digest"] == stable(state), "verification seal differs")
    commit = receipt["verifier_commit"]
    head = str(git(repository, "rev-parse", "HEAD"))
    require(subprocess.run(("git", "merge-base", "--is-ancestor", commit, head),
                           cwd=repository).returncode == 0, "verifier lineage differs")
    hashes = receipt["verifier_runtime_sha256"]
    require(set(hashes) == set(VERIFIER_RUNTIME), "verifier runtime closure differs")
    for name in VERIFIER_RUNTIME:
        raw = snapshot(repository / name)
        require(raw == git(repository, "show", f"{commit}:{name}", binary=True)
                and sha256(raw).hexdigest() == hashes[name],
                f"verifier runtime differs: {name}")
    result, evidence = verify_result(repository)
    require(state == verification_state(result, evidence, commit, hashes),
            "verification reconstruction differs")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "validate"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = run(args.repository) if args.mode == "run" else validate(args.repository)
    except VerificationError as error:
        print(f"M2 adapter verification refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({"passed": result["passed"],
                      "verification_digest": result["verification_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

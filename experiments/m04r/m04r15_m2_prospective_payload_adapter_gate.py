"""Verify the prospective payload adapter on one authenticated consumed month."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from market_analogues.prospective_adapter import WEIGHTS, build_prospective_documents
from market_analogues.prospective_batch import seal_prediction_batch, stable


SCHEMA = "m04r15-m2-prospective-payload-adapter-gate-v1"
CONTRACT = Path("config/prospective-payload-adapter-contract-v1.json")
OUTPUT = Path("config/data/analogues/m04r15/m2-prospective-payload-adapter-v1/RESULT.json")
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
RUNTIME = (
    "config/prospective-payload-adapter-contract-v1.json",
    "src/market_analogues/analogue_candidate.py",
    "src/market_analogues/prospective_adapter.py",
    "src/market_analogues/prospective_batch.py",
    "experiments/m04r/m04r15_m2_prospective_payload_adapter_gate.py",
    "tests/test_prospective_adapter.py",
    "tests/test_m04r15_m2_prospective_payload_adapter_gate.py",
)
PROBABILITY_COLUMNS = (
    "favorable_probability", "adverse_probability", "no_touch_probability",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise GateError(f"unsafe input: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"regular input required: {path}")
        blocks: list[bytes] = []
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
        value: dict[str, Any] = {}
        for key, item in items:
            require(key not in value, f"duplicate JSON key: {path}/{key}")
            value[key] = item
        return value
    try:
        value = json.loads(
            raw, object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                GateError(f"nonfinite JSON: {path}/{token}"),
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ("git", *arguments), cwd=repository, capture_output=True,
        text=not binary, check=False,
    )
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def build_consumed_fixture(raw: Mapping[str, bytes], month: str) -> tuple[
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any],
]:
    registry_all = pd.read_parquet(BytesIO(raw["query_registry"]), engine="pyarrow")
    registry = registry_all[
        pd.to_datetime(registry_all.cutoff).dt.to_period("M").astype(str) == month
    ][[
        "episode_id", "symbol", "cutoff", "quality_tier", "liquidity_stratum",
        "selection_hash", "stock_prefix_digest", "benchmark_prefix_digest",
    ]].reset_index(drop=True)
    scores = pd.concat((
        pd.read_parquet(BytesIO(raw["nonfinal_scores"]), engine="pyarrow"),
        pd.read_parquet(BytesIO(raw["final_scores"]), engine="pyarrow"),
    ), ignore_index=True)
    outcomes = pd.concat((
        pd.read_parquet(BytesIO(raw["nonfinal_outcomes"]), engine="pyarrow"),
        pd.read_parquet(BytesIO(raw["final_outcomes"]), engine="pyarrow"),
    ), ignore_index=True)
    require(len(registry) == 24, "fixture registry inventory differs")
    target = scores[(scores.month == month) & scores.lane.isin(
        ["composite", "price_only", "recent_return_volatility"],
    )][["query_id", "query_cutoff", "lane", *PROBABILITY_COLUMNS]].copy()
    composite = scores[scores.lane == "composite"][[
        "query_id", "query_cutoff", "query_regime", "quality_tier", "liquidity_stratum",
    ]].drop_duplicates("query_id")
    primary = outcomes[outcomes.horizon_sessions == 20][[
        "query_id", "cutoff", "completion_timestamp", "barrier_label",
    ]].drop_duplicates("query_id")
    history = composite.merge(primary, on="query_id", validate="one_to_one")
    cutoff = pd.Timestamp(registry.cutoff.iloc[0])
    history = history[pd.to_datetime(history.completion_timestamp) <= cutoff][[
        "query_id", "cutoff", "completion_timestamp", "barrier_label", "query_regime",
        "quality_tier", "liquidity_stratum",
    ]].rename(columns={
        "cutoff": "origin_cutoff", "barrier_label": "label",
        "query_regime": "market_regime",
    }).reset_index(drop=True)
    regimes = set(scores[(scores.month == month) & (scores.lane == "composite")].query_regime)
    require(len(regimes) == 1, "fixture market regime differs")
    documents = build_prospective_documents(
        registry=registry, lane_scores=target, prior_history=history,
        market_regime=next(iter(regimes)),
    )
    return documents, {
        "registry_rows": len(registry), "target_score_rows": len(target),
        "causal_history_rows": len(history), "market_regime": next(iter(regimes)),
        "cutoff": cutoff.date().isoformat(),
    }


def gate_state(raw: Mapping[str, bytes], contract: Mapping[str, Any]) -> dict[str, Any]:
    documents, inventory = build_consumed_fixture(
        raw, contract["consumed_fixture"]["month"],
    )
    source, registry, predictions = documents
    checks = []
    add = lambda name, passed, detail: checks.append({
        "name": name, "passed": bool(passed), "detail": detail,
    })
    add("query_inventory", len(registry["queries"]) == 24, len(registry["queries"]))
    cell_counts: dict[str, int] = {}
    for row in registry["queries"]:
        key = f"{row['quality_tier']}|{row['liquidity_stratum']}"
        cell_counts[key] = cell_counts.get(key, 0) + 1
    add("six_cells_balanced", set(cell_counts.values()) == {4} and len(cell_counts) == 6,
        cell_counts)
    add("target_schema_excludes_labels", predictions["query_outcomes_opened"] is False,
        sorted(predictions))
    add("source_prefix_stops_at_cutoff",
        source["maximum_source_timestamp"] == source["cutoff"], source["cutoff"])
    probabilities_valid = all(
        abs(sum(values) - 1) <= 1e-12
        for row in predictions["rows"] for values in row["probabilities"].values()
    )
    add("probabilities_valid", probabilities_valid, len(predictions["rows"]) * 3)
    with tempfile.TemporaryDirectory(prefix="m2-adapter-gate-") as name:
        root = Path(name)
        seal = seal_prediction_batch(
            root, contract_digest=contract["contract_digest"], source_lock=source,
            registry=registry, predictions=predictions, created_at="consumed-fixture",
        )
        add("prospective_seal_valid", seal["query_count"] == 24, seal["result_digest"])
    add("real_launch_remains_disabled",
        contract["real_launch"]["currently_authorized"] is False, "disabled")
    require(all(row["passed"] for row in checks), "adapter check failed")
    return {
        "checks": checks, "inventory": inventory,
        "source_document_digest": stable(source),
        "registry_document_digest": stable(registry),
        "prediction_document_digest": stable(predictions),
        "prospective_prediction_seal_digest": next(
            row["detail"] for row in checks if row["name"] == "prospective_seal_valid"
        ),
    }


def publish(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "create-only result exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m2-adapter-", dir=path.parent)
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
    raw = {name: snapshot(repository / path) for name, path in INPUTS.items()}
    raw_contract = snapshot(repository / CONTRACT)
    contract = decode(raw_contract, repository / CONTRACT)
    state_contract = {key: value for key, value in contract.items() if key != "contract_digest"}
    require(contract["contract_digest"] == stable(state_contract), "contract seal differs")
    expected = contract["consumed_fixture"]["input_sha256"]
    require({name: sha256(raw[name]).hexdigest() for name in expected} == expected,
            "consumed input identity differs")
    registry_seal = decode(raw["registry_seal"], repository / INPUTS["registry_seal"])
    listener = decode(raw["listener_result"], repository / INPUTS["listener_result"])
    listener_verification = decode(
        raw["listener_verification"], repository / INPUTS["listener_verification"],
    )
    require(registry_seal["seal_digest"] == contract["consumed_fixture"]["registry_seal_digest"]
            and registry_seal["registry_digest"]
            == contract["consumed_fixture"]["registry_digest"], "registry seal differs")
    require(listener["result_digest"] == contract["upstream"]["listener_result_digest"]
            and listener_verification["verification_digest"]
            == contract["upstream"]["listener_verification_digest"],
            "listener identity differs")
    m1_result = decode(raw["m1_result"], repository / INPUTS["m1_result"])
    require(m1_result["result_digest"] == contract["upstream"]["m1_result_digest"]
            and m1_result["candidate_contract_digest"]
            == contract["upstream"]["candidate_contract_digest"]
            and m1_result["selected_weights"] == WEIGHTS
            == contract["adapter"]["candidate_weights"], "M1 candidate binding differs")
    require(listener["real_prediction_created"] is False
            and listener_verification["post_freeze_outcomes_opened"] is False,
            "listener boundary differs")
    commit = str(git(repository, "rev-parse", "HEAD"))
    runtime = {name: snapshot(repository / name) for name in RUNTIME}
    for name, content in runtime.items():
        require(content == git(repository, "show", f"{commit}:{name}", binary=True),
                f"runtime not committed: {name}")
    started = perf_counter()
    evidence = gate_state(raw, contract)
    elapsed = perf_counter() - started
    state = {
        "schema_version": SCHEMA, "status": "consumed_fixture_adapter_verified",
        "passed": True, "implementation_commit": commit,
        "runtime_sha256": {name: sha256(value).hexdigest() for name, value in runtime.items()},
        "contract_digest": contract["contract_digest"],
        "input_sha256": {name: sha256(value).hexdigest() for name, value in raw.items()},
        **evidence,
        "consumed_fixture_only": True,
        "new_post_freeze_outcomes_opened": False,
        "real_registry_created": False, "real_prediction_created": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False, "trading_claim_authorized": False,
    }
    result = {
        **state, "result_digest": stable(state), "elapsed_seconds": elapsed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    publish(repository / OUTPUT, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = run(args.repository)
    except GateError as error:
        print(f"M2 adapter gate refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({
        "passed": result["passed"], "checks": len(result["checks"]),
        "result_digest": result["result_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

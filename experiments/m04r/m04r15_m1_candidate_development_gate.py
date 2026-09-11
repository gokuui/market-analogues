"""Freeze M1 candidate from verified, already-consumed WF-04 development evidence."""
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
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from market_analogues.analogue_candidate import (
    COMPONENTS,
    MIXTURE_WEIGHTS,
    candidate_probabilities,
    matched_causal_probabilities_batch,
)


SCHEMA = "m04r15-m1-candidate-development-gate-v1"
CONTRACT = Path("config/analogue-candidate-mixture-v1.json")
OUTPUT = Path("config/data/analogues/m04r15/m1-candidate-development-gate-v1/FROZEN.json")
RUNTIME = (
    "config/analogue-candidate-mixture-v1.json",
    "config/analogue-improvement-metric-contract-v1.json",
    "src/market_analogues/analogue_candidate.py",
    "experiments/m04r/m04r15_m1_candidate_development_gate.py",
    "tests/test_analogue_candidate.py",
    "tests/test_m04r15_m1_candidate_development_gate.py",
)
ROOT = Path("config/data/analogues/m04r14")
INPUTS = {
    "prediction_seal": ROOT / "t14-10-wf03d-prediction-store-v2/SEALED.json",
    "prediction_verification": ROOT / "t14-10-wf03d-prediction-store-v2-verification/VERIFIED.json",
    "nonfinal_seal": ROOT / "t14-10-wf04-nonfinal-evaluation-v2/SEALED.json",
    "nonfinal_verification": ROOT / "t14-10-wf04-nonfinal-evaluation-v2-verification/VERIFIED.json",
    "nonfinal_scores": ROOT / "t14-10-wf04-nonfinal-evaluation-v2/query-scores.parquet",
    "nonfinal_outcomes": ROOT / "t14-10-wf03d-prediction-store-v2/nonfinal-query-outcomes.parquet",
    "final_seal": ROOT / "t14-10-wf04-final-evaluation-v1/SEALED.json",
    "final_verification": ROOT / "t14-10-wf04-final-evaluation-v1-verification/VERIFIED.json",
    "final_scores": ROOT / "t14-10-wf04-final-evaluation-v1/query-scores.parquet",
    "final_outcomes": ROOT / "t14-10-wf04-final-evaluation-v1/final-query-outcomes.parquet",
    "r1b_lock": ROOT / "r1b-joint-b005-b2-closure-v3/LOCKED.json",
    "m0_result": Path("config/data/analogues/m04r15/m0-analogue-improvement-synthetic-v1/RESULT.json"),
    "m0_verification": Path("config/data/analogues/m04r15/m0-analogue-improvement-synthetic-v1-verification/VERIFIED.json"),
}
EXPECTED = {
    "prediction_seal": ("result_digest", "25f9261322ee597d883c0011e3fdf4fc2fed84903aa59e0cf2d45ab6cb09ca84"),
    "prediction_verification": ("verification_digest", "a541894505c0ded88640e95aabe07ccb629f887a842f420a633310e86d3ba13a"),
    "nonfinal_seal": ("result_digest", "410936a92ea90ea93884de6c4f3714cd1ce2b0c49d86ed35a4bd304d7fe38b12"),
    "nonfinal_verification": ("verification_digest", "dd1ce15005c62eada8b93441f81f4ed9534d4f91686cfc9cf21b339fc471ea95"),
    "final_seal": ("result_digest", "ef2c6643d95c9544a5a8d773ffeec8d19f23b8fdb599be6446e13726b1d0eda2"),
    "final_verification": ("verification_digest", "e91cd1c40428462d97fa63d0b2e7701d5b65189b8f6ca3f4a606948dfcc5023e"),
    "r1b_lock": ("closure_digest", "89a625cd31b1bcaf06be1ee09e246b9e06574fc49700998408b88477a86fe7a4"),
    "m0_result": ("result_digest", "011e7df31f4b13c880c2b45deca59eb09f0f618e1079a85dcffed749e964fffa"),
    "m0_verification": ("verification_digest", "e1bf47be33fe9c7ebbd84b8f5c971952ca0b1d1a8d48a82ad3c414f7bb5b467c"),
}
FOLDS = ("development", "validation_1", "validation_2", "validation_3", "final_untouched")
LANES = ("composite", "price_only", "recent_return_volatility")
PROBABILITY_COLUMNS = (
    "favorable_probability", "adverse_probability", "no_touch_probability",
)


class CandidateGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateGateError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise CandidateGateError(f"unsafe input: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"regular input required: {path}")
        chunks = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        after = os.fstat(descriptor)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size,
                                 item.st_mtime_ns, item.st_ctime_ns, item.st_mode)
        require(identity(before) == identity(after), f"input changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def decode_json(content: bytes, path: Path) -> dict[str, Any]:
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
        raise CandidateGateError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(("git", *arguments), cwd=repository, capture_output=True,
                            text=not binary, check=False)
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def validate_inputs(repository: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    content = {name: snapshot(repository / path) for name, path in INPUTS.items()}
    decoded = {name: decode_json(content[name], repository / INPUTS[name])
               for name in EXPECTED}
    for name, (field, expected) in EXPECTED.items():
        require(decoded[name].get(field) == expected, f"upstream identity differs: {name}")
        require(decoded[name].get("passed") is True, f"upstream did not pass: {name}")
    for name in ("prediction_verification", "nonfinal_verification", "final_verification"):
        value = decoded[name]
        require(value["verification_digest"] == stable({k: v for k, v in value.items()
                                                        if k != "verification_digest"}),
                f"upstream verification seal differs: {name}")
        require(value.get("production_promotion_authorized") is False,
                f"upstream claim boundary differs: {name}")
    for name in ("prediction_seal", "nonfinal_seal", "final_seal"):
        value = decoded[name]
        state = {k: v for k, v in value.items()
                 if k not in {"result_digest", "created_at", "elapsed_seconds"}}
        require(value["result_digest"] == stable(state), f"upstream result seal differs: {name}")
        require(value.get("production_promotion_authorized") is False,
                f"upstream claim boundary differs: {name}")
    require(decoded["r1b_lock"].get("claims", {}).get("predictive_claim_authorized") is False,
            "R1-B claim boundary differs")
    require(decoded["m0_result"].get("real_forward_outcomes_opened") is False
            and decoded["m0_verification"].get("real_forward_outcomes_opened") is False,
            "M0 outcome boundary differs")
    require(decoded["r1b_lock"]["closure_digest"] == stable({
        key: value for key, value in decoded["r1b_lock"].items()
        if key not in {"closure_digest", "created_at"}
    }), "R1-B closure seal differs")
    require(decoded["m0_result"]["result_digest"] == stable({
        key: value for key, value in decoded["m0_result"].items()
        if key not in {"result_digest", "created_at"}
    }), "M0 result seal differs")
    require(decoded["m0_verification"]["verification_digest"] == stable({
        key: value for key, value in decoded["m0_verification"].items()
        if key not in {"verification_digest", "created_at"}
    }), "M0 verification seal differs")

    manifest_paths = (
        ("prediction_seal", "nonfinal-query-outcomes.parquet", "nonfinal_outcomes"),
        ("nonfinal_seal", "query-scores.parquet", "nonfinal_scores"),
        ("final_seal", "query-scores.parquet", "final_scores"),
        ("final_seal", "final-query-outcomes.parquet", "final_outcomes"),
    )
    for seal_name, filename, content_name in manifest_paths:
        manifest = {row["path"]: row for row in decoded[seal_name]["file_manifest"]}
        require(filename in manifest, f"upstream file absent from manifest: {filename}")
        raw = content[content_name]
        require(len(raw) == manifest[filename]["bytes"]
                and sha256(raw).hexdigest() == manifest[filename]["sha256"],
                f"upstream file identity differs: {filename}")
    return content, decoded


def load_development(input_content: Mapping[str, bytes]) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    nonfinal = pd.read_parquet(BytesIO(input_content["nonfinal_scores"]), engine="pyarrow")
    final = pd.read_parquet(BytesIO(input_content["final_scores"]), engine="pyarrow")
    scores = pd.concat((nonfinal, final), ignore_index=True)
    require(len(nonfinal) == 24_192 and len(final) == 3_360 and len(scores) == 27_552,
            "development score row inventory differs")
    require(not scores.duplicated(["query_id", "lane"]).any()
            and set(scores.lane) == {
                "composite", "composite_unweighted", "deterministic_random",
                "price_only", "recent_return_volatility", "regime_only_frequency",
                "unconditional_market_frequency",
            }, "development score lane closure differs")
    columns = [
        "query_id", "query_cutoff", "fold_id", "query_regime", "quality_tier",
        "liquidity_stratum", "route_status", "multiclass_evaluable",
        "purged_evaluation_included",
    ]
    queries = scores.loc[scores.lane == "composite", columns].drop_duplicates("query_id")
    require(len(queries) == 3_936 and set(queries.fold_id) == {*FOLDS, "warmup"},
            "development query inventory differs")
    nonfinal_outcomes = pd.read_parquet(BytesIO(input_content["nonfinal_outcomes"]), engine="pyarrow")
    final_outcomes = pd.read_parquet(BytesIO(input_content["final_outcomes"]), engine="pyarrow")
    require(len(nonfinal_outcomes) == 20_736 and len(final_outcomes) == 2_880,
            "development outcome row inventory differs")
    outcomes = pd.concat((nonfinal_outcomes, final_outcomes), ignore_index=True)
    require(not outcomes.duplicated(["query_id", "horizon_sessions"]).any()
            and set(outcomes.horizon_sessions) == {5, 10, 20, 40, 60, 126},
            "development outcome horizon closure differs")
    outcomes = outcomes.loc[outcomes.horizon_sessions == 20, [
        "query_id", "cutoff", "completion_timestamp", "barrier_label",
    ]].drop_duplicates("query_id")
    require(len(outcomes) == 3_936, "development outcome inventory differs")
    queries = queries.merge(outcomes, on="query_id", how="left", validate="one_to_one")
    require(np.array_equal(queries.route_status.fillna("").to_numpy(),
                           queries.barrier_label.fillna("").to_numpy()),
            "score/outcome label projection differs")
    queries = queries.sort_values(["query_cutoff", "query_id"], kind="stable").reset_index(drop=True)
    records = [{
        "query_id": str(row.query_id), "origin_cutoff": row.cutoff,
        "completion_timestamp": row.completion_timestamp, "label": row.barrier_label,
        "market_regime": str(row.query_regime),
        "prefix_quality_class": str(row.quality_tier),
        "trailing_liquidity_cell": str(row.liquidity_stratum),
    } for row in queries.itertuples(index=False)]
    forecast_queries = [{
        "query_id": str(row.query_id), "query_cutoff": row.query_cutoff,
        "market_regime": str(row.query_regime),
        "prefix_quality_class": str(row.quality_tier),
        "trailing_liquidity_cell": str(row.liquidity_stratum),
    } for row in queries.itertuples(index=False)]
    matched = matched_causal_probabilities_batch(records, forecast_queries)
    matrices: dict[str, np.ndarray] = {
        "matched_causal_history": np.asarray([
            matched[str(query_id)].probabilities for query_id in queries.query_id
        ], dtype=np.float64),
    }
    for lane in LANES:
        selected = scores.loc[scores.lane == lane, ["query_id", *PROBABILITY_COLUMNS]] \
            .drop_duplicates("query_id").set_index("query_id")
        require(len(selected) == 3_936 and set(selected.index) == set(queries.query_id),
                f"development lane inventory differs: {lane}")
        matrices[lane] = selected.loc[queries.query_id, list(PROBABILITY_COLUMNS)].to_numpy(
            dtype=np.float64,
        )
    candidate = np.asarray([
        candidate_probabilities({name: matrices[name][index] for name in COMPONENTS})
        for index in range(len(queries))
    ], dtype=np.float64)
    matrices["candidate"] = candidate
    classes = {"favorable_first": 0, "adverse_first": 1, "no_touch": 2}
    truth = np.zeros((len(queries), 3), dtype=np.float64)
    for index, label in enumerate(queries.route_status):
        if label in classes:
            truth[index, classes[str(label)]] = 1.0
    queries["matched_fallback_level"] = [matched[str(value)].fallback_level
                                          for value in queries.query_id]
    queries["matched_support_rows"] = [matched[str(value)].support_rows
                                       for value in queries.query_id]
    return queries, matrices, truth


def brier(matrix: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.square(matrix - truth).sum(axis=1)


def metric_rows(queries: pd.DataFrame, matrices: Mapping[str, np.ndarray],
                truth: np.ndarray, candidate: np.ndarray) -> list[dict[str, Any]]:
    eligible = (queries.multiclass_evaluable.to_numpy(dtype=np.bool_)
                & queries.purged_evaluation_included.to_numpy(dtype=np.bool_))
    fold_values = queries.fold_id.astype(str).to_numpy()
    candidate_loss = brier(candidate, truth)
    matched_loss = brier(matrices["matched_causal_history"], truth)
    locked_loss = brier(matrices["composite"], truth)
    observed = truth.argmax(axis=1)
    positions = np.arange(len(truth))
    candidate_log = -np.log(candidate[positions, observed])
    matched_log = -np.log(matrices["matched_causal_history"][positions, observed])
    locked_log = -np.log(matrices["composite"][positions, observed])
    rows = []
    for fold in (*FOLDS, "pooled"):
        selected = eligible if fold == "pooled" else eligible & (fold_values == fold)
        require(selected.any(), f"empty development fold: {fold}")
        model = float(candidate_loss[selected].mean())
        matched = float(matched_loss[selected].mean())
        locked = float(locked_loss[selected].mean())
        rows.append({
            "fold": fold, "rows": int(selected.sum()), "candidate_brier": model,
            "matched_causal_brier": matched, "locked_retriever_brier": locked,
            "skill_vs_matched": float(1.0 - model / matched),
            "skill_vs_locked": float(1.0 - model / locked),
            "candidate_log_loss": float(candidate_log[selected].mean()),
            "matched_causal_log_loss": float(matched_log[selected].mean()),
            "locked_retriever_log_loss": float(locked_log[selected].mean()),
            "log_loss_difference_vs_matched": float(
                candidate_log[selected].mean() - matched_log[selected].mean()),
            "log_loss_difference_vs_locked": float(
                candidate_log[selected].mean() - locked_log[selected].mean()),
        })
    return rows


def grid(repository_data: tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries, matrices, truth = repository_data
    candidates = []
    for matched_tenths in range(11):
        for composite_tenths in range(11 - matched_tenths):
            for price_tenths in range(11 - matched_tenths - composite_tenths):
                recent_tenths = 10 - matched_tenths - composite_tenths - price_tenths
                weights = {
                    "matched_causal_history": matched_tenths / 10,
                    "composite": composite_tenths / 10,
                    "price_only": price_tenths / 10,
                    "recent_return_volatility": recent_tenths / 10,
                }
                candidate = sum(weights[name] * matrices[name] for name in COMPONENTS)
                rows = metric_rows(queries, matrices, truth, candidate)
                folds = rows[:-1]
                objective = min(min(row["skill_vs_matched"], row["skill_vs_locked"])
                                for row in folds)
                candidates.append({
                    "weights": weights, "minimum_fold_two_comparator_skill": objective,
                    "pooled_skill_vs_matched": rows[-1]["skill_vs_matched"],
                    "metrics": rows,
                })
    candidates.sort(key=lambda row: (
        -row["minimum_fold_two_comparator_skill"], -row["pooled_skill_vs_matched"],
        tuple(row["weights"][name] for name in COMPONENTS),
    ))
    require(len(candidates) == 286, "simplex grid inventory differs")
    leave_one_out = []
    eligible = (queries.multiclass_evaluable.to_numpy(dtype=np.bool_)
                & queries.purged_evaluation_included.to_numpy(dtype=np.bool_))
    fold_values = queries.fold_id.astype(str).to_numpy()
    matched_loss = brier(matrices["matched_causal_history"], truth)
    locked_loss = brier(matrices["composite"], truth)
    for held in FOLDS:
        training = eligible & (fold_values != held)
        held_mask = eligible & (fold_values == held)
        choices = []
        for row in candidates:
            candidate = sum(row["weights"][name] * matrices[name] for name in COMPONENTS)
            loss = brier(candidate, truth)
            choices.append((float(1.0 - loss[training].mean() / matched_loss[training].mean()), row, loss))
        selected_score, selected, loss = max(choices, key=lambda item: (
            item[0], -sum(item[1]["weights"][name] * index
                          for index, name in enumerate(COMPONENTS)),
        ))
        leave_one_out.append({
            "held_fold": held, "selected_weights": selected["weights"],
            "training_skill_vs_matched": selected_score,
            "held_rows": int(held_mask.sum()),
            "held_skill_vs_matched": float(1.0 - loss[held_mask].mean() / matched_loss[held_mask].mean()),
            "held_skill_vs_locked": float(1.0 - loss[held_mask].mean() / locked_loss[held_mask].mean()),
        })
    return candidates, leave_one_out


def publish(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), "create-only M1 result exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def execute(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    require(not str(git(repository, "status", "--porcelain", "--untracked-files=all")),
            "clean committed tree required")
    head = str(git(repository, "rev-parse", "HEAD"))
    runtime = {name: snapshot(repository / name) for name in RUNTIME}
    for name, content in runtime.items():
        require(git(repository, "show", f"{head}:{name}", binary=True) == content,
                f"runtime not committed: {name}")
    contract = decode_json(runtime[str(CONTRACT)], repository / CONTRACT)
    require(contract["contract_digest"] == stable({k: v for k, v in contract.items()
                                                   if k != "contract_digest"}),
            "candidate contract digest differs")
    require(contract.get("status") ==
            "frozen_on_consumed_development_evidence_before_new_untouched_outcomes"
            and contract.get("candidate", {}).get("component_weights") == dict(MIXTURE_WEIGHTS)
            and contract.get("development_selection", {}).get("selected_weights")
            == [MIXTURE_WEIGHTS[name] for name in COMPONENTS]
            and not any(contract.get("claims", {}).values()),
            "candidate contract semantic closure differs")
    input_content, _ = validate_inputs(repository)
    data = load_development(input_content)
    queries, matrices, truth = data
    candidates, leave_one_out = grid(data)
    selected = candidates[0]
    require(selected["weights"] == dict(MIXTURE_WEIGHTS), "frozen candidate is not grid optimum")
    require(selected["minimum_fold_two_comparator_skill"] > 0
            and all(row["held_skill_vs_matched"] > 0 and row["held_skill_vs_locked"] > 0
                    for row in leave_one_out),
            "development robustness check failed")
    require(all(row["log_loss_difference_vs_matched"] <= 0
                and row["log_loss_difference_vs_locked"] <= 0
                for row in selected["metrics"]),
            "development log-loss guard failed")
    fallback = {str(key): int(value) for key, value in
                queries.matched_fallback_level.value_counts().sort_index().items()}
    state = {
        "schema_version": SCHEMA, "status": "candidate_frozen_on_consumed_development",
        "passed": True, "implementation_commit": head,
        "runtime_sha256": {name: sha256(content).hexdigest() for name, content in runtime.items()},
        "candidate_contract_digest": contract["contract_digest"],
        "m0_contract_digest": contract["m0_metric_contract_digest"],
        "input_sha256": {name: sha256(content).hexdigest() for name, content in input_content.items()},
        "upstream_digests": {name: {field: digest}
                             for name, (field, digest) in EXPECTED.items()},
        "inventory": {
            "registered_queries": len(queries),
            "purged_primary_evaluable_queries": int(sum(row["rows"] for row in selected["metrics"][:-1])),
            "folds": list(FOLDS), "simplex_candidates": len(candidates),
            "matched_fallback_counts_all_registered": fallback,
        },
        "selected_weights": selected["weights"],
        "selection_objective": "maximum_minimum_fold_two_comparator_Brier_skill",
        "minimum_fold_two_comparator_skill": selected["minimum_fold_two_comparator_skill"],
        "fold_metrics": selected["metrics"],
        "leave_one_fold_out": leave_one_out,
        "positive_each_fold_against_both_development_comparators": True,
        "new_untouched_outcomes_opened": False,
        "development_outcomes_opened": True,
        "development_result_is_predictive_validation": False,
        "predictive_claim_authorized": False,
        "production_promotion_authorized": False,
        "trading_claim_authorized": False,
    }
    result = {**state, "result_digest": stable(state),
              "created_at": datetime.now(timezone.utc).isoformat()}
    publish(repository / OUTPUT, result)
    require({name: snapshot(repository / name) for name in RUNTIME} == runtime,
            "runtime changed during M1 publication")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    try:
        result = execute(parser.parse_args(argv).repository)
    except CandidateGateError as error:
        print(f"M1 candidate gate refused: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps({"passed": result["passed"], "result_digest": result["result_digest"],
                      "minimum_fold_two_comparator_skill": result["minimum_fold_two_comparator_skill"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

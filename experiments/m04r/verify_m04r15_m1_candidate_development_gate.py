"""Independent verifier for the M1 consumed-development candidate freeze."""
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


SCHEMA = "m04r15-m1-candidate-development-verification-v1"
EXPECTED_CANDIDATE_CONTRACT_DIGEST = (
    "75c5b8cb4557e0ed6bc65081a33bc9463bf1075b0e9ea3ebfbc71f8a819463a3"
)
EXPECTED_M0_CONTRACT_DIGEST = (
    "c2350617df4d69cf5064828d6adc4161b3abc65e1f9a55dfbc14979553c9ccf3"
)
RESULT = Path("config/data/analogues/m04r15/m1-candidate-development-gate-v1/FROZEN.json")
OUTPUT = Path("config/data/analogues/m04r15/m1-candidate-development-gate-v1-verification/VERIFIED.json")
CONTRACT = Path("config/analogue-candidate-mixture-v1.json")
PRODUCER_RUNTIME = (
    "config/analogue-candidate-mixture-v1.json",
    "config/analogue-improvement-metric-contract-v1.json",
    "src/market_analogues/analogue_candidate.py",
    "experiments/m04r/m04r15_m1_candidate_development_gate.py",
    "tests/test_analogue_candidate.py",
    "tests/test_m04r15_m1_candidate_development_gate.py",
)
VERIFIER_RUNTIME = (
    "experiments/m04r/verify_m04r15_m1_candidate_development_gate.py",
    "tests/test_m04r15_m1_candidate_development_verifier.py",
)
INPUTS = {
    "prediction_seal": Path("config/data/analogues/m04r14/t14-10-wf03d-prediction-store-v2/SEALED.json"),
    "prediction_verification": Path("config/data/analogues/m04r14/t14-10-wf03d-prediction-store-v2-verification/VERIFIED.json"),
    "nonfinal_seal": Path("config/data/analogues/m04r14/t14-10-wf04-nonfinal-evaluation-v2/SEALED.json"),
    "nonfinal_verification": Path("config/data/analogues/m04r14/t14-10-wf04-nonfinal-evaluation-v2-verification/VERIFIED.json"),
    "nonfinal_scores": Path("config/data/analogues/m04r14/t14-10-wf04-nonfinal-evaluation-v2/query-scores.parquet"),
    "nonfinal_outcomes": Path("config/data/analogues/m04r14/t14-10-wf03d-prediction-store-v2/nonfinal-query-outcomes.parquet"),
    "final_seal": Path("config/data/analogues/m04r14/t14-10-wf04-final-evaluation-v1/SEALED.json"),
    "final_verification": Path("config/data/analogues/m04r14/t14-10-wf04-final-evaluation-v1-verification/VERIFIED.json"),
    "final_scores": Path("config/data/analogues/m04r14/t14-10-wf04-final-evaluation-v1/query-scores.parquet"),
    "final_outcomes": Path("config/data/analogues/m04r14/t14-10-wf04-final-evaluation-v1/final-query-outcomes.parquet"),
    "r1b_lock": Path("config/data/analogues/m04r14/r1b-joint-b005-b2-closure-v3/LOCKED.json"),
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
CLASSES = ("favorable_first", "adverse_first", "no_touch")
COMPONENTS = ("matched_causal_history", "composite", "price_only", "recent_return_volatility")
WEIGHTS = {"matched_causal_history": .4, "composite": .1, "price_only": .2,
           "recent_return_volatility": .3}
FOLDS = ("development", "validation_1", "validation_2", "validation_3", "final_untouched")
PROBABILITY_COLUMNS = ("favorable_probability", "adverse_probability", "no_touch_probability")
SCORE_LANES = {
    "composite",
    "composite_unweighted",
    "deterministic_random",
    "price_only",
    "recent_return_volatility",
    "regime_only_frequency",
    "unconditional_market_frequency",
}
RESULT_KEYS = {
    "candidate_contract_digest", "created_at", "development_outcomes_opened",
    "development_result_is_predictive_validation", "fold_metrics",
    "implementation_commit", "input_sha256", "inventory", "leave_one_fold_out",
    "m0_contract_digest", "minimum_fold_two_comparator_skill",
    "new_untouched_outcomes_opened", "passed",
    "positive_each_fold_against_both_development_comparators",
    "predictive_claim_authorized", "production_promotion_authorized",
    "result_digest", "runtime_sha256", "schema_version", "selected_weights",
    "selection_objective", "status", "trading_claim_authorized", "upstream_digests",
}
VERIFICATION_KEYS = {
    "candidate_contract_digest", "created_at", "minimum_fold_two_comparator_skill",
    "new_untouched_outcomes_opened", "passed", "predictive_claim_authorized",
    "producer_implementation_commit", "producer_or_candidate_scientific_code_imported",
    "producer_result_digest", "producer_result_sha256",
    "production_promotion_authorized", "purged_primary_evaluable_queries",
    "reconstruction_digest", "registered_queries", "schema_version",
    "simplex_candidates", "status", "trading_claim_authorized",
    "verification_digest", "verifier_commit", "verifier_runtime_sha256",
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def stable(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def snapshot(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise VerificationError(f"unsafe file: {path}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"regular file required: {path}")
        blocks = []
        while block := os.read(descriptor, 1 << 20):
            blocks.append(block)
        after = os.fstat(descriptor)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size,
                                 item.st_mtime_ns, item.st_ctime_ns, item.st_mode)
        require(identity(before) == identity(after), f"file changed while read: {path}")
        return b"".join(blocks)
    finally:
        os.close(descriptor)


def decode(content: bytes, path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {path}/{key}")
            result[key] = value
        return result
    def reject(token: str) -> Any:
        raise VerificationError(f"nonfinite JSON: {path}/{token}")
    try:
        value = json.loads(content, object_pairs_hook=pairs, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {path}") from error
    require(type(value) is dict, f"JSON object required: {path}")
    return value


def git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(("git", *arguments), cwd=repository, capture_output=True,
                            text=not binary, check=False)
    require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout if binary else result.stdout.strip()


def inputs(repository: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    raw = {name: snapshot(repository / path) for name, path in INPUTS.items()}
    values = {name: decode(raw[name], repository / INPUTS[name]) for name in EXPECTED}
    for name, (field, expected) in EXPECTED.items():
        require(values[name].get(field) == expected and values[name].get("passed") is True,
                f"upstream identity/status differs: {name}")
    for name in ("prediction_verification", "nonfinal_verification", "final_verification"):
        value = values[name]
        require(value["verification_digest"] == stable({k: v for k, v in value.items()
                                                        if k != "verification_digest"})
                and value["production_promotion_authorized"] is False,
                f"upstream verification differs: {name}")
    for name in ("prediction_seal", "nonfinal_seal", "final_seal"):
        value = values[name]
        require(value["result_digest"] == stable({k: v for k, v in value.items()
                                                  if k not in {"result_digest", "created_at", "elapsed_seconds"}}),
                f"upstream seal differs: {name}")
    require(values["r1b_lock"]["closure_digest"] == stable({
        k: v for k, v in values["r1b_lock"].items() if k not in {"closure_digest", "created_at"}
    }) and values["r1b_lock"]["claims"]["predictive_claim_authorized"] is False,
            "R1-B lock differs")
    require(values["m0_result"]["result_digest"] == stable({
        k: v for k, v in values["m0_result"].items() if k not in {"result_digest", "created_at"}
    }), "M0 result differs")
    require(values["m0_verification"]["verification_digest"] == stable({
        k: v for k, v in values["m0_verification"].items()
        if k not in {"verification_digest", "created_at"}
    }) and values["m0_verification"]["predictive_claim_authorized"] is False,
            "M0 verification differs")
    manifests = (
        ("prediction_seal", "nonfinal-query-outcomes.parquet", "nonfinal_outcomes"),
        ("nonfinal_seal", "query-scores.parquet", "nonfinal_scores"),
        ("final_seal", "query-scores.parquet", "final_scores"),
        ("final_seal", "final-query-outcomes.parquet", "final_outcomes"),
    )
    for seal_name, filename, raw_name in manifests:
        manifest = {row["path"]: row for row in values[seal_name]["file_manifest"]}
        require(filename in manifest and len(raw[raw_name]) == manifest[filename]["bytes"]
                and sha256(raw[raw_name]).hexdigest() == manifest[filename]["sha256"],
                f"manifested input differs: {filename}")
    return raw, values


def causal_base(queries: pd.DataFrame) -> tuple[np.ndarray, list[str], list[int]]:
    events = []
    for row in queries.itertuples(index=False):
        if row.barrier_label not in CLASSES or pd.isna(row.completion_timestamp):
            continue
        origin = pd.Timestamp(row.cutoff); completion = pd.Timestamp(row.completion_timestamp)
        require(completion > origin, "outcome completion did not follow origin")
        events.append((completion, str(row.query_id), str(row.barrier_label),
                       (str(row.query_regime), str(row.quality_tier), str(row.liquidity_stratum))))
    events.sort(key=lambda item: (item[0], item[1]))
    ordered = sorted(((pd.Timestamp(row.query_cutoff), str(row.query_id),
                       (str(row.query_regime), str(row.quality_tier), str(row.liquidity_stratum)))
                      for row in queries.itertuples(index=False)), key=lambda item: (item[0], item[1]))
    counts = np.zeros(3, dtype=np.int64); exact = {}; regliq = {}; regime = {}
    position = {name: index for index, name in enumerate(CLASSES)}
    cursor = 0; result = {}; levels = {}; supports = {}
    for cutoff, query_id, cell in ordered:
        while cursor < len(events) and events[cursor][0] <= cutoff:
            _, _, label, event_cell = events[cursor]; p = position[label]
            counts[p] += 1
            exact.setdefault(event_cell, np.zeros(3, dtype=np.int64))[p] += 1
            regliq.setdefault((event_cell[0], event_cell[2]), np.zeros(3, dtype=np.int64))[p] += 1
            regime.setdefault(event_cell[0], np.zeros(3, dtype=np.int64))[p] += 1
            cursor += 1
        choices = (("exact", exact.get(cell)), ("regime_and_liquidity", regliq.get((cell[0], cell[2]))),
                   ("regime", regime.get(cell[0])))
        selected = counts; level = "unconditional"
        for name, value in choices:
            if value is not None and int(value.sum()) >= 30:
                selected = value; level = name; break
        support = int(selected.sum())
        if int(counts.sum()) == 0: level = "uniform_no_history"
        result[query_id] = (selected.astype(float) + .5) / (support + 1.5)
        levels[query_id] = level; supports[query_id] = support
    return (np.asarray([result[str(value)] for value in queries.query_id]),
            [levels[str(value)] for value in queries.query_id],
            [supports[str(value)] for value in queries.query_id])


def data(raw: Mapping[str, bytes]) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    nonfinal = pd.read_parquet(BytesIO(raw["nonfinal_scores"]), engine="pyarrow")
    final = pd.read_parquet(BytesIO(raw["final_scores"]), engine="pyarrow")
    require((len(nonfinal), len(final)) == (24_192, 3_360), "score inventory differs")
    scores = pd.concat((nonfinal, final), ignore_index=True)
    require(not scores.duplicated(["query_id", "lane"]).any()
            and len(scores) == 27_552 and set(scores.lane) == SCORE_LANES,
            "score key closure differs")
    columns = ["query_id", "query_cutoff", "fold_id", "query_regime", "quality_tier",
               "liquidity_stratum", "route_status", "multiclass_evaluable",
               "purged_evaluation_included"]
    queries = scores[scores.lane == "composite"][columns].copy()
    nf_out = pd.read_parquet(BytesIO(raw["nonfinal_outcomes"]), engine="pyarrow")
    f_out = pd.read_parquet(BytesIO(raw["final_outcomes"]), engine="pyarrow")
    require((len(nf_out), len(f_out)) == (20_736, 2_880), "outcome inventory differs")
    outcomes = pd.concat((nf_out, f_out), ignore_index=True)
    require(not outcomes.duplicated(["query_id", "horizon_sessions"]).any()
            and set(outcomes.horizon_sessions) == {5, 10, 20, 40, 60, 126},
            "outcome horizon closure differs")
    outcomes = outcomes[outcomes.horizon_sessions == 20][
        ["query_id", "cutoff", "completion_timestamp", "barrier_label"]]
    queries = queries.merge(outcomes, on="query_id", validate="one_to_one") \
        .sort_values(["query_cutoff", "query_id"], kind="stable").reset_index(drop=True)
    require(len(queries) == 3_936 and set(queries.fold_id) == {*FOLDS, "warmup"}
            and np.array_equal(queries.route_status.fillna("").to_numpy(),
                               queries.barrier_label.fillna("").to_numpy()),
            "query/label closure differs")
    matched, levels, supports = causal_base(queries)
    matrices = {"matched_causal_history": matched}
    for lane in COMPONENTS[1:]:
        selected = scores[scores.lane == lane].set_index("query_id")
        require(len(selected) == 3_936 and set(selected.index) == set(queries.query_id),
                f"lane closure differs: {lane}")
        matrices[lane] = selected.loc[queries.query_id, list(PROBABILITY_COLUMNS)].to_numpy(float)
    matrices["candidate"] = sum(WEIGHTS[name] * matrices[name] for name in COMPONENTS)
    truth = np.zeros((len(queries), 3)); positions = {name: i for i, name in enumerate(CLASSES)}
    for index, label in enumerate(queries.route_status):
        if label in positions: truth[index, positions[str(label)]] = 1
    queries["matched_fallback_level"] = levels; queries["matched_support_rows"] = supports
    return queries, matrices, truth


def metrics(queries: pd.DataFrame, matrices: Mapping[str, np.ndarray], truth: np.ndarray,
            candidate: np.ndarray) -> list[dict[str, Any]]:
    eligible = queries.multiclass_evaluable.to_numpy(bool) & queries.purged_evaluation_included.to_numpy(bool)
    folds = queries.fold_id.astype(str).to_numpy(); observed = truth.argmax(1); indexes = np.arange(len(truth))
    loss = lambda values: np.square(values - truth).sum(1)
    cb, mb, lb = loss(candidate), loss(matrices["matched_causal_history"]), loss(matrices["composite"])
    cl = -np.log(candidate[indexes, observed]); ml = -np.log(matrices["matched_causal_history"][indexes, observed]); ll = -np.log(matrices["composite"][indexes, observed])
    rows = []
    for fold in (*FOLDS, "pooled"):
        use = eligible if fold == "pooled" else eligible & (folds == fold)
        c, m, l = map(lambda x: float(x[use].mean()), (cb, mb, lb))
        c_log, m_log, l_log = map(lambda x: float(x[use].mean()), (cl, ml, ll))
        rows.append({"fold": fold, "rows": int(use.sum()), "candidate_brier": c,
                     "matched_causal_brier": m, "locked_retriever_brier": l,
                     "skill_vs_matched": float(1-c/m), "skill_vs_locked": float(1-c/l),
                     "candidate_log_loss": c_log, "matched_causal_log_loss": m_log,
                     "locked_retriever_log_loss": l_log,
                     "log_loss_difference_vs_matched": float(c_log-m_log),
                     "log_loss_difference_vs_locked": float(c_log-l_log)})
    return rows


def reconstruct(raw: Mapping[str, bytes]) -> dict[str, Any]:
    queries, matrices, truth = data(raw); candidates = []
    for wm in range(11):
        for wc in range(11-wm):
            for wp in range(11-wm-wc):
                wr=10-wm-wc-wp; weights=dict(zip(COMPONENTS,(wm/10,wc/10,wp/10,wr/10)))
                rows=metrics(queries,matrices,truth,sum(weights[n]*matrices[n] for n in COMPONENTS))
                candidates.append({"weights":weights,"minimum_fold_two_comparator_skill":min(min(r["skill_vs_matched"],r["skill_vs_locked"]) for r in rows[:-1]),"pooled_skill_vs_matched":rows[-1]["skill_vs_matched"],"metrics":rows})
    candidates.sort(key=lambda row:(-row["minimum_fold_two_comparator_skill"],-row["pooled_skill_vs_matched"],tuple(row["weights"][n] for n in COMPONENTS)))
    require(len(candidates)==286 and candidates[0]["weights"]==WEIGHTS,"independent grid optimum differs")
    eligible=queries.multiclass_evaluable.to_numpy(bool)&queries.purged_evaluation_included.to_numpy(bool);folds=queries.fold_id.astype(str).to_numpy();loss=lambda p:np.square(p-truth).sum(1);mb=loss(matrices["matched_causal_history"]);lb=loss(matrices["composite"]);lofo=[]
    for held in FOLDS:
        train=eligible&(folds!=held);test=eligible&(folds==held);choices=[]
        for row in candidates:
            values=sum(row["weights"][n]*matrices[n] for n in COMPONENTS);l=loss(values);score=float(1-l[train].mean()/mb[train].mean());choices.append((score,row,l))
        score,chosen,l=max(choices,key=lambda item:(item[0],-sum(item[1]["weights"][n]*i for i,n in enumerate(COMPONENTS))))
        lofo.append({"held_fold":held,"selected_weights":chosen["weights"],"training_skill_vs_matched":score,"held_rows":int(test.sum()),"held_skill_vs_matched":float(1-l[test].mean()/mb[test].mean()),"held_skill_vs_locked":float(1-l[test].mean()/lb[test].mean())})
    selected=candidates[0]
    return {"inventory":{"registered_queries":len(queries),"purged_primary_evaluable_queries":sum(r["rows"] for r in selected["metrics"][:-1]),"folds":list(FOLDS),"simplex_candidates":len(candidates),"matched_fallback_counts_all_registered":{str(k):int(v) for k,v in queries.matched_fallback_level.value_counts().sort_index().items()}},"selected":selected,"leave_one_fold_out":lofo}


def verify_producer(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_raw=snapshot(repository/RESULT);result=decode(result_raw,repository/RESULT)
    require(set(result) == RESULT_KEYS, "producer result field closure differs")
    state={k:v for k,v in result.items() if k not in {"result_digest","created_at"}}
    require(result.get("result_digest")==stable(state) and result.get("passed") is True
            and result.get("status")=="candidate_frozen_on_consumed_development","producer result differs")
    for claim in ("new_untouched_outcomes_opened","development_result_is_predictive_validation","predictive_claim_authorized","production_promotion_authorized","trading_claim_authorized"):
        require(result.get(claim) is False,f"claim boundary differs: {claim}")
    implementation=result.get("implementation_commit");head=str(git(repository,"rev-parse","HEAD"))
    require(type(implementation)is str and subprocess.run(("git","merge-base","--is-ancestor",implementation,head),cwd=repository).returncode==0,"producer lineage differs")
    require(set(result["runtime_sha256"])==set(PRODUCER_RUNTIME),"producer runtime closure differs")
    for name in PRODUCER_RUNTIME:
        content=snapshot(repository/name);require(content==git(repository,"show",f"{implementation}:{name}",binary=True) and sha256(content).hexdigest()==result["runtime_sha256"][name],f"producer runtime drift: {name}")
    contract=decode(snapshot(repository/CONTRACT),repository/CONTRACT);digest=contract.pop("contract_digest")
    require(digest == stable(contract) == result["candidate_contract_digest"]
            == EXPECTED_CANDIDATE_CONTRACT_DIGEST
            and result["m0_contract_digest"] == EXPECTED_M0_CONTRACT_DIGEST
            and contract["m0_metric_contract_digest"] == EXPECTED_M0_CONTRACT_DIGEST
            and contract["status"] ==
            "frozen_on_consumed_development_evidence_before_new_untouched_outcomes"
            and contract["candidate"]["component_weights"] == WEIGHTS
            and contract["secondary_continuous_distribution"]
            ["unbounded_quadratic_materialization"] is False
            and not any(contract["claims"].values()), "candidate contract differs")
    raw,_=inputs(repository);require(result["input_sha256"]=={n:sha256(v).hexdigest() for n,v in raw.items()},"producer input manifest differs")
    rebuilt=reconstruct(raw);selected=rebuilt["selected"]
    require(result["inventory"]==rebuilt["inventory"] and result["selected_weights"]==selected["weights"] and result["minimum_fold_two_comparator_skill"]==selected["minimum_fold_two_comparator_skill"] and result["fold_metrics"]==selected["metrics"] and result["leave_one_fold_out"]==rebuilt["leave_one_fold_out"],"producer scientific reconstruction differs")
    require(result["upstream_digests"]=={n:{f:d} for n,(f,d) in EXPECTED.items()},"upstream digest manifest differs")
    return result,{"result_sha256":sha256(result_raw).hexdigest(),"reconstruction":rebuilt}


def verification_state(result:Mapping[str,Any],evidence:Mapping[str,Any],commit:str,hashes:Mapping[str,str])->dict[str,Any]:
    return {"schema_version":SCHEMA,"status":"independently_verified","passed":True,"verifier_commit":commit,"verifier_runtime_sha256":dict(hashes),"producer_implementation_commit":result["implementation_commit"],"producer_result_digest":result["result_digest"],"producer_result_sha256":evidence["result_sha256"],"candidate_contract_digest":result["candidate_contract_digest"],"reconstruction_digest":stable(evidence["reconstruction"]),"registered_queries":3936,"purged_primary_evaluable_queries":2595,"simplex_candidates":286,"minimum_fold_two_comparator_skill":result["minimum_fold_two_comparator_skill"],"producer_or_candidate_scientific_code_imported":False,"new_untouched_outcomes_opened":False,"predictive_claim_authorized":False,"production_promotion_authorized":False,"trading_claim_authorized":False}


def publish(path:Path,value:Mapping[str,Any])->None:
    require(not path.exists() and not path.is_symlink(),"create-only verification exists");path.parent.mkdir(parents=True,exist_ok=True);descriptor,name=tempfile.mkstemp(prefix=".m1-verify-",dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(descriptor,"wb") as handle:handle.write((json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode());handle.flush();os.fsync(handle.fileno())
        os.link(temporary,path);directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(directory);os.close(directory)
    finally:temporary.unlink(missing_ok=True)


def run(repository:Path)->dict[str,Any]:
    repository=repository.resolve(strict=True);require(not str(git(repository,"status","--porcelain","--untracked-files=all")),"clean committed tree required");result,evidence=verify_producer(repository);commit=str(git(repository,"rev-parse","HEAD"));content={n:snapshot(repository/n) for n in VERIFIER_RUNTIME}
    for name,raw in content.items():require(raw==git(repository,"show",f"{commit}:{name}",binary=True),f"verifier runtime not committed: {name}")
    hashes={n:sha256(v).hexdigest() for n,v in content.items()};state=verification_state(result,evidence,commit,hashes);receipt={**state,"verification_digest":stable(state),"created_at":datetime.now(timezone.utc).isoformat()};publish(repository/OUTPUT,receipt);return receipt


def validate(repository:Path)->dict[str,Any]:
    repository=repository.resolve(strict=True);receipt=decode(snapshot(repository/OUTPUT),repository/OUTPUT);require(set(receipt)==VERIFICATION_KEYS,"verification receipt field closure differs");state={k:v for k,v in receipt.items() if k not in {"verification_digest","created_at"}};require(receipt.get("verification_digest")==stable(state),"verification digest differs");commit=receipt.get("verifier_commit");head=str(git(repository,"rev-parse","HEAD"));require(type(commit)is str and subprocess.run(("git","merge-base","--is-ancestor",commit,head),cwd=repository).returncode==0,"verifier lineage differs");hashes=receipt.get("verifier_runtime_sha256");require(type(hashes)is dict and set(hashes)==set(VERIFIER_RUNTIME),"verifier runtime closure differs")
    for name in VERIFIER_RUNTIME:
        content=snapshot(repository/name);require(content==git(repository,"show",f"{commit}:{name}",binary=True) and sha256(content).hexdigest()==hashes[name],f"verifier runtime drift: {name}")
    result,evidence=verify_producer(repository);require(state==verification_state(result,evidence,commit,hashes),"verification reconstruction differs");return receipt


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=("run","validate"));parser.add_argument("--repository",type=Path,default=Path.cwd());args=parser.parse_args(argv)
    try:result=run(args.repository) if args.mode=="run" else validate(args.repository)
    except VerificationError as error:print(f"M1 independent verification refused: {error}",file=os.sys.stderr);return 2
    print(json.dumps({"passed":result["passed"],"verification_digest":result["verification_digest"]},sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())

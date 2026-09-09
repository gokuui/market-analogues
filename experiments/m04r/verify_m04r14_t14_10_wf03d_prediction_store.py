"""Independently verify causal WF-03D predictions and access ordering."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_analogues.adapters import source_from_spec
from market_analogues.config import load_config
from market_analogues.types import InstrumentKey, stable_hash

from experiments.m04r import m04r14_t14_10_wf03_feasibility as base
from experiments.m04r import m04r14_t14_10_wf03d_prediction_store as producer
from experiments.m04r import verify_m04r14_t14_09_full_outcome_store as oracle_tools
from experiments.m04r.m04r14_t14_09_outcome_oracle import (
    prepare_reference_series,
    reference_prepared_episode,
)


SCHEMA = "m04r14-t14-10-wf03d-prediction-verification-v2"


class WalkForwardPredictionVerificationError(RuntimeError):
    pass


def _close(left: float, right: float) -> bool:
    return (math.isnan(left) and math.isnan(right)) or left == right


def _weights(ranks: Sequence[int], weighted: bool) -> np.ndarray:
    return np.asarray([
        2.0 ** (-(int(rank) - 1) / 10.0) if weighted else 1.0
        for rank in ranks
    ], dtype=np.float64)


def _probabilities(labels: Sequence[str], ranks: Sequence[int], weighted: bool) -> tuple[np.ndarray, float]:
    weights = _weights(ranks, weighted)
    mass = np.zeros(3, dtype=np.float64)
    positions = {name: index for index, name in enumerate(producer.PRIMARY_CLASSES)}
    for label, weight in zip(labels, weights):
        mass[positions[label]] += weight
    probability = (mass + .5) / (float(mass.sum()) + 1.5)
    ess = float(weights.sum() ** 2 / np.square(weights).sum()) if len(weights) else 0.0
    return probability, ess


def _binary_probabilities(labels: Sequence[str], ranks: Sequence[int], weighted: bool) -> tuple[np.ndarray,float]:
    weights=_weights(ranks,weighted); mass=np.zeros(2,dtype=np.float64); positions={"favorable_first":0,"adverse_first":1}
    for label,weight in zip(labels,weights): mass[positions[label]]+=weight
    probability=(mass+.5)/(float(mass.sum())+1.0)
    ess=float(weights.sum()**2/np.square(weights).sum()) if len(weights) else 0.0
    return probability,ess


def _quantiles(values: Sequence[float], ranks: Sequence[int], weighted: bool) -> tuple[np.ndarray, float]:
    if not values:
        return np.repeat(np.nan, len(producer.QUANTILES)), 0.0
    observations = np.asarray(values, dtype=np.float64)
    weights = _weights(ranks, weighted)
    order = np.argsort(observations, kind="stable")
    cumulative = np.cumsum(weights[order])
    positions = np.searchsorted(
        cumulative, np.asarray(producer.QUANTILES) * cumulative[-1], side="left",
    )
    forecast = observations[order][np.clip(positions, 0, len(order) - 1)]
    return forecast, float(weights.sum() ** 2 / np.square(weights).sum())


def _median(values: Sequence[float], ranks: Sequence[int], weighted: bool) -> tuple[float, float]:
    if not values:
        return np.nan, 0.0
    observations = np.asarray(values, dtype=np.float64)
    weights = _weights(ranks, weighted)
    order = np.argsort(observations, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = int(np.searchsorted(cumulative, .5 * cumulative[-1], side="left"))
    return float(observations[order][position]), float(weights.sum() ** 2 / np.square(weights).sum())


def _path_summary(matrix: np.ndarray, ranks: Sequence[int], weighted: bool) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    mass=_weights(ranks,weighted)[:,None]; valid=np.isfinite(matrix)
    order=np.argsort(np.where(valid,matrix,np.inf),axis=0,kind="stable")
    values=np.take_along_axis(matrix,order,axis=0)
    weights=np.take_along_axis(np.broadcast_to(mass,matrix.shape),order,axis=0)*np.take_along_axis(valid,order,axis=0)
    cumulative=np.cumsum(weights,axis=0); total=cumulative[-1]
    positions=np.argmax(cumulative>=.5*total,axis=0); median=values[positions,np.arange(matrix.shape[1])]
    count=valid.sum(axis=0).astype(np.int16); square=np.square(weights).sum(axis=0)
    ess=np.divide(np.square(total),square,out=np.zeros_like(total),where=square>0); median[count==0]=np.nan
    return median,count,ess


def _month_order_valid(root: Path, final: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    prediction = base._read(root / "PREDICTIONS_SEALED.json")
    if not producer._valid_seal(prediction, timing=True) \
            or prediction.get("file_manifest") != producer._month_file_manifest(root, producer.PREDICTION_FILES):
        raise WalkForwardPredictionVerificationError("monthly prediction seal differs")
    if final:
        forbidden = {"query-outcomes.parquet", "query-paths.parquet", "MONTH_CLOSED.json"}
        if any((root / name).exists() for name in forbidden):
            raise WalkForwardPredictionVerificationError("final query outcome opened")
        return prediction, None
    closed = base._read(root / "MONTH_CLOSED.json")
    if not producer._valid_seal(closed, timing=True) \
            or closed.get("prediction_seal_digest") != prediction.get("result_digest") \
            or closed.get("predictions_sealed_before_query_outcome_access") is not True \
            or pd.Timestamp(prediction["created_at"]) > pd.Timestamp(closed["created_at"]):
        raise WalkForwardPredictionVerificationError("monthly prediction/outcome order differs")
    return prediction, closed


def _independent_primary(
    observed: pd.DataFrame, month_registry: pd.DataFrame, links: pd.DataFrame,
    outcomes20: Mapping[str, str],
    eligibility20: Mapping[tuple[str, str, int], tuple[bool, str]],
    prior_distances: Sequence[float],
) -> None:
    by_query = {key: value for key, value in observed.groupby("query_id", sort=False)}
    for query in month_registry.itertuples(index=False):
        actual = by_query.get(query.query_id)
        if actual is None or len(actual) != len(producer.NEIGHBOR_LANES) * len(producer.PREFIXES):
            raise WalkForwardPredictionVerificationError("primary prediction inventory differs")
        source = links.loc[links.query_id == query.query_id]
        nearest = float.fromhex(str(source.loc[
            (source.method == "composite") & (source["rank"] == 1), "distance_hex",
        ].iloc[0]))
        prefix_favorable = []
        expected_by_lane: dict[tuple[str, int], tuple[np.ndarray, float, list[str | None]]] = {}
        for lane in producer.NEIGHBOR_LANES:
            method, weighted = producer._lane_source(lane)
            lane_rows = source.loc[source.method == method].sort_values("rank")
            for prefix in producer.PREFIXES:
                chosen = lane_rows.iloc[:prefix]
                routes: list[str | None] = []
                labels: list[str] = []; ranks: list[int] = []
                for match in chosen.itertuples(index=False):
                    allowed, reason = eligibility20[(query.query_id, method, int(match.rank))]
                    route = outcomes20[str(match.matched_episode_id)] if allowed else (
                        "censored" if reason == "incomplete_horizon" else None
                    )
                    routes.append(route)
                    if route in producer.PRIMARY_CLASSES:
                        labels.append(str(route)); ranks.append(int(match.rank))
                probability, ess = _probabilities(labels, ranks, weighted)
                expected_by_lane[(lane, prefix)] = probability, ess, routes
                if lane == "composite":
                    prefix_favorable.append(float(probability[0]))
        threshold = float(np.quantile(prior_distances, .95, method="linear")) if len(prior_distances) >= 250 else np.nan
        instability = max(prefix_favorable) - min(prefix_favorable)
        reasons=[]
        composite_routes=expected_by_lane[("composite",20)][2]
        if sum(value in producer.PRIMARY_CLASSES for value in composite_routes)<10: reasons.append("insufficient_effective_sample_size")
        if len(prior_distances)>=250 and nearest>threshold: reasons.append("historically_novel_query")
        if instability>.2: reasons.append("unstable_neighborhood")
        reasons.extend(("poor_data_quality","failed_calibration")); expected_reasons="|".join(reasons)
        for row in actual.itertuples(index=False):
            probability, ess, routes = expected_by_lane[(row.lane,int(row.prefix))]
            labels=[route for route in routes if route in producer.PRIMARY_CLASSES]
            directional_labels=[]; directional_ranks=[]
            for rank,route in enumerate(routes,1):
                if route in ("favorable_first","adverse_first"):
                    directional_labels.append(str(route)); directional_ranks.append(rank)
            directional,directional_ess=_binary_probabilities(
                directional_labels,directional_ranks,row.lane!="composite_unweighted",
            )
            checks=(
                _close(float(row.favorable_probability),float(probability[0])),
                _close(float(row.adverse_probability),float(probability[1])),
                _close(float(row.no_touch_probability),float(probability[2])),
                _close(float(row.directional_favorable_probability),float(directional[0])),
                _close(float(row.directional_adverse_probability),float(directional[1])),
                int(row.directional_eligible_rows)==len(directional_labels),
                _close(float(row.directional_effective_rows),directional_ess),
                int(row.eligible_rows)==len(labels), _close(float(row.effective_rows),ess),
                int(row.favorable_rows)==labels.count("favorable_first"),
                int(row.adverse_rows)==labels.count("adverse_first"),
                int(row.no_touch_rows)==labels.count("no_touch"),
                int(row.ambiguous_rows)==routes.count("ambiguous_same_first_touch_bar"),
                int(row.censored_rows)==routes.count("censored"),
                int(row.unavailable_rows)==routes.count(None),
                _close(float(row.nearest_composite_distance),nearest),
                int(row.novelty_reference_rows)==len(prior_distances),
                _close(float(row.novelty_threshold),threshold),
                _close(float(row.neighborhood_instability),instability),
                str(row.abstention_reasons)==expected_reasons,
                bool(row.forced_score_lane), not bool(row.selective_lane),
            )
            if not all(checks):
                raise WalkForwardPredictionVerificationError(f"primary formula differs: {query.query_id}:{row.lane}:{row.prefix}")


def _independent_baselines(
    observed: pd.DataFrame, month_registry: pd.DataFrame,
    prior: pd.DataFrame, regime: str,
) -> None:
    for query in month_registry.itertuples(index=False):
        available=prior.loc[(pd.to_datetime(prior.completion_timestamp)<=pd.Timestamp(query.cutoff))&prior.barrier_label.isin(producer.PRIMARY_CLASSES)] if len(prior) else prior
        labels=available.barrier_label.astype(str).tolist(); regimes=available.regime.astype(str).tolist() if len(available) else []
        unconditional,_=_probabilities(labels,list(range(1,len(labels)+1)),False)
        selected=[label for label,value in zip(labels,regimes) if value==regime]; fallback=len(selected)<50
        target=labels if fallback else selected; regime_probability,_=_probabilities(target,list(range(1,len(target)+1)),False)
        unconditional_directional_labels=[x for x in labels if x in ("favorable_first","adverse_first")]
        target_directional_labels=[x for x in target if x in ("favorable_first","adverse_first")]
        unconditional_directional,unconditional_directional_ess=_binary_probabilities(
            unconditional_directional_labels,list(range(1,len(unconditional_directional_labels)+1)),False,
        )
        regime_directional,regime_directional_ess=_binary_probabilities(
            target_directional_labels,list(range(1,len(target_directional_labels)+1)),False,
        )
        rows=observed.loc[observed.query_id==query.query_id]
        if len(rows)!=2: raise WalkForwardPredictionVerificationError("baseline inventory differs")
        for row in rows.itertuples(index=False):
            expected=unconditional if row.lane=="unconditional_market_frequency" else regime_probability
            expected_fallback=False if row.lane=="unconditional_market_frequency" else fallback
            expected_directional=unconditional_directional if row.lane=="unconditional_market_frequency" else regime_directional
            expected_directional_labels=unconditional_directional_labels if row.lane=="unconditional_market_frequency" else target_directional_labels
            expected_directional_ess=unconditional_directional_ess if row.lane=="unconditional_market_frequency" else regime_directional_ess
            if not all((
                int(row.prior_eligible_rows)==len(labels), int(row.prior_same_regime_rows)==sum(x==regime for x in regimes),
                str(row.query_regime)==regime, bool(row.fallback_to_unconditional)==expected_fallback,
                _close(float(row.favorable_probability),float(expected[0])),
                _close(float(row.adverse_probability),float(expected[1])),
                _close(float(row.no_touch_probability),float(expected[2])), bool(row.forced_score_lane),
                _close(float(row.directional_favorable_probability),float(expected_directional[0])),
                _close(float(row.directional_adverse_probability),float(expected_directional[1])),
                int(row.directional_eligible_rows)==len(expected_directional_labels),
                _close(float(row.directional_effective_rows),expected_directional_ess),
            )): raise WalkForwardPredictionVerificationError(f"baseline formula differs: {query.query_id}:{row.lane}")


def _independent_continuous(
    observed: pd.DataFrame, month_registry: pd.DataFrame, links: pd.DataFrame,
    outcome_map: Mapping[tuple[str,int], Mapping[str,Any]],
    eligibility: Mapping[tuple[str,str,int,int],bool],
) -> None:
    for query in month_registry.itertuples(index=False):
        source=links.loc[links.query_id==query.query_id]
        actual=observed.loc[observed.query_id==query.query_id]
        if len(actual)!=len(producer.NEIGHBOR_LANES)*6*len(producer.MEASURES):
            raise WalkForwardPredictionVerificationError("continuous inventory differs")
        for row in actual.itertuples(index=False):
            method,weighted=producer._lane_source(row.lane); lane=source.loc[source.method==method].sort_values("rank")
            values=[]; ranks=[]
            for match in lane.itertuples(index=False):
                if eligibility[(query.query_id,method,int(match.rank),int(row.horizon_sessions))]:
                    value=outcome_map[(str(match.matched_episode_id),int(row.horizon_sessions))][row.measure]
                    if pd.notna(value): values.append(float(value)); ranks.append(int(match.rank))
            forecast,ess=_quantiles(values,ranks,weighted)
            actual_values=[float(getattr(row,f"q{int(q*100):02d}")) for q in producer.QUANTILES]
            if not all((_close(x,y) for x,y in zip(actual_values,forecast))) \
                    or int(row.eligible_rows)!=len(values) or not _close(float(row.effective_rows),ess):
                raise WalkForwardPredictionVerificationError(f"continuous formula differs: {query.query_id}:{row.lane}:{row.horizon_sessions}:{row.measure}")


def _independent_paths(
    observed: pd.DataFrame, month_registry: pd.DataFrame,
    links: pd.DataFrame, paths: producer._PathIndex,
) -> None:
    for query in month_registry.itertuples(index=False):
        source=links.loc[links.query_id==query.query_id]; cutoff=np.datetime64(pd.Timestamp(query.cutoff),"ns")
        for lane in producer.NEIGHBOR_LANES:
            method,weighted=producer._lane_source(lane); selected=source.loc[source.method==method].sort_values("rank")
            matrices={name:np.full((20,126),np.nan) for name in producer.PATH_MEASURES}; ranks=selected["rank"].astype(int).tolist()
            for index,match in enumerate(selected.itertuples(index=False)):
                bounds=paths.slices.get(str(match.matched_episode_id))
                if bounds is None: continue
                start,stop=bounds; valid=paths.expected[start:stop]&(paths.timestamp[start:stop]<=cutoff)
                steps=paths.step[start:stop][valid].astype(int); within=(steps>=1)&(steps<=126)
                for name in producer.PATH_MEASURES: matrices[name][index,steps[within]-1]=paths.values[name][start:stop][valid][within]
            actual=observed.loc[(observed.query_id==query.query_id)&(observed.lane==lane)].sort_values("step")
            if len(actual)!=126: raise WalkForwardPredictionVerificationError("path inventory differs")
            for name in producer.PATH_MEASURES:
                median,count,ess=_path_summary(matrices[name],ranks,weighted)
                if not np.array_equal(actual[f"{name}_median"].to_numpy(dtype=float),median,equal_nan=True) \
                        or not np.array_equal(actual[f"{name}_rows"].to_numpy(dtype=np.int16),count) \
                        or not np.array_equal(actual[f"{name}_effective_rows"].to_numpy(dtype=float),ess,equal_nan=True):
                    raise WalkForwardPredictionVerificationError(f"path formula differs: {query.query_id}:{lane}:{name}")


def _verify_nonfinal_query_outcomes(
    repository: Path, registry: pd.DataFrame, regimes: pd.DataFrame, output: Path,
) -> tuple[int, int]:
    selected=registry.loc[pd.to_datetime(registry.cutoff)<producer.FINAL_START]
    final_ids=set(registry.loc[pd.to_datetime(registry.cutoff)>=producer.FINAL_START,"query_id"].astype(str))
    observed_outcomes=pd.read_parquet(output/"nonfinal-query-outcomes.parquet")
    observed_paths=pd.read_parquet(output/"nonfinal-query-paths.parquet")
    if set(observed_outcomes.query_id.astype(str))&final_ids or set(observed_paths.query_id.astype(str))&final_ids:
        raise WalkForwardPredictionVerificationError("final query outcome present in aggregate")
    if set(observed_outcomes.query_id.astype(str))!=set(selected.query_id.astype(str)):
        raise WalkForwardPredictionVerificationError("nonfinal query outcome inventory differs")
    source=source_from_spec(load_config(repository/base.CONFIG_RELATIVE).datasets["nasdaq"])
    benchmark=source.load_benchmark()
    if benchmark is None: raise WalkForwardPredictionVerificationError("oracle benchmark unavailable")
    reference_benchmark=prepare_reference_series(benchmark)
    contract=base._read(repository/producer.outcome_store.CONTRACT)
    source_content_digest=base._resident()["content_digest"]
    fingerprints=producer._source_fingerprints(repository)
    expected_outcomes=[]; expected_paths=[]
    regime_map={str(row.month):str(row.regime) for row in regimes.itertuples(index=False)}
    for symbol,queries in selected.groupby("symbol",sort=True):
        key=InstrumentKey("nasdaq",str(symbol)); stock=source.load(key); fingerprint=source.fingerprint(key)
        if fingerprints.get(str(symbol))!=fingerprint:
            raise WalkForwardPredictionVerificationError(f"oracle source differs: {symbol}")
        ordered_stock=stock.sort_values("timestamp",kind="stable").reset_index(drop=True)
        positions={stamp:index for index,stamp in enumerate(pd.to_datetime(ordered_stock.timestamp))}
        for query in queries.itertuples(index=False):
            position=positions.get(pd.Timestamp(query.cutoff))
            if position is None:
                raise WalkForwardPredictionVerificationError(f"oracle cutoff absent: {query.case_id}")
            local=ordered_stock.iloc[
                max(0,position-20):min(len(ordered_stock),position+127)
            ].reset_index(drop=True)
            reference_stock=prepare_reference_series(local)
            outcomes,paths=reference_prepared_episode(
                reference_stock,reference_benchmark,episode_id=str(query.query_id),
                cutoff=pd.Timestamp(query.cutoff),source_fingerprint=fingerprint,
                contract_digest=contract["contract_digest"],source_content_digest=source_content_digest,
            )
            for row in outcomes:
                expected_outcomes.append({**row,"query_id":str(query.query_id),"query_regime":regime_map[str(query.month)]})
            for row in paths: expected_paths.append({**row,"query_id":str(query.query_id)})
    expected_outcomes.sort(key=lambda row:(row["query_id"],row["horizon_sessions"]))
    expected_paths.sort(key=lambda row:(row["query_id"],row["step"]))
    if oracle_tools._records(observed_outcomes,("query_id","horizon_sessions"))!=expected_outcomes \
            or oracle_tools._records(observed_paths,("query_id","step"))!=expected_paths:
        raise WalkForwardPredictionVerificationError("independent nonfinal query outcome oracle differs")
    return len(observed_outcomes),len(observed_paths)


def verify(repository: Path) -> dict[str, Any]:
    started=perf_counter(); repository=repository.resolve(strict=True)
    prereg,registry,regimes,h1=producer.validate_preregistration(repository)
    cache=repository/producer.CACHE_RELATIVE; output=repository/producer.OUTPUT_RELATIVE
    seal=base._read(output/"SEALED.json")
    if not producer._valid_seal(seal,timing=True) or seal.get("passed") is not True:
        raise WalkForwardPredictionVerificationError("prediction store seal differs")
    links,outcomes,eligibility_frame=producer._load_prediction_inputs(repository)
    outcomes20={str(row.episode_id):str(row.barrier_label) for row in outcomes.loc[outcomes.horizon_sessions==20].itertuples(index=False)}
    outcome_map={(str(row.episode_id),int(row.horizon_sessions)):row._asdict() for row in outcomes.itertuples(index=False)}
    eligibility={(str(row.query_id),str(row.method),int(row.rank),int(row.horizon_sessions)):bool(row.eligible) for row in eligibility_frame.itertuples(index=False)}
    eligibility20={
        (str(row.query_id),str(row.method),int(row.rank)):
            (bool(row.eligible),str(row.reason))
        for row in eligibility_frame.loc[eligibility_frame.horizon_sessions==20].itertuples(index=False)
    }
    path_index=producer._PathIndex(output.parent/"t14-10-wf03d-outcome-store-v1"/"future-paths.parquet")
    prior_distances=[]; prior=producer._prior_query_outcomes(cache,[]); verified_months=[]; month_semantics={}
    regime_map={str(row.month):producer._plain(row._asdict()) for row in regimes.itertuples(index=False)}
    for index,month in enumerate(prereg["month_order"]):
        month_registry=registry.loc[registry.month==month]; root=cache/f"month-{month}"; final=pd.Timestamp(f"{month}-01")>=producer.FINAL_START
        prediction,closed=_month_order_valid(root,final)
        primary=pd.read_parquet(root/producer.PREDICTION_FILES[0]); baselines=pd.read_parquet(root/producer.PREDICTION_FILES[1]); continuous=pd.read_parquet(root/producer.PREDICTION_FILES[2]); path_rows=pd.read_parquet(root/producer.PREDICTION_FILES[3])
        _independent_primary(primary,month_registry,links,outcomes20,eligibility20,prior_distances)
        _independent_baselines(baselines,month_registry,prior,regime_map[month]["regime"])
        _independent_continuous(continuous,month_registry,links,outcome_map,eligibility)
        _independent_paths(path_rows,month_registry,links,path_index)
        if not final:
            opened=pd.read_parquet(root/"query-outcomes.parquet"); add=opened.loc[opened.horizon_sessions==20,["query_id","completion_timestamp","barrier_label","query_regime"]].rename(columns={"query_regime":"regime"})
            prior=pd.concat([prior,add],ignore_index=True)
        if bool(month_registry.scored.iloc[0]):
            prior_distances.extend(primary.loc[(primary.lane=="composite")&(primary.prefix==20),"nearest_composite_distance"].tolist())
        verified_months.append(prediction["result_digest"])
        month_semantics[month]=prediction["semantic_digests"]
        print(f"[wf03d-prediction-verifier] month={month} complete={index+1}/{len(prereg['month_order'])}",flush=True)
    if verified_months!=seal.get("month_prediction_result_digests"):
        raise WalkForwardPredictionVerificationError("aggregate month digest order differs")
    names=("raw-predictions.parquet","baseline-predictions.parquet","continuous-predictions.parquet","path-predictions.parquet","nonfinal-query-outcomes.parquet","nonfinal-query-paths.parquet")
    if seal.get("file_manifest")!=producer._month_file_manifest(output,names):
        raise WalkForwardPredictionVerificationError("aggregate file manifest differs")
    aggregate_specs=(
        ("raw-predictions.parquet","neighbor",("query_id","lane","prefix")),
        ("baseline-predictions.parquet","baseline",("query_id","lane")),
        ("continuous-predictions.parquet","continuous",("query_id","lane","horizon_sessions","measure")),
        ("path-predictions.parquet","path",("query_id","lane","step")),
    )
    for name,key,order in aggregate_specs:
        aggregate=pd.read_parquet(output/name)
        if set(aggregate.month.astype(str))!=set(prereg["month_order"]):
            raise WalkForwardPredictionVerificationError(f"aggregate month coverage differs: {name}")
        for month,frame in aggregate.groupby("month",sort=False):
            if producer._frame_digest(frame,order)!=month_semantics[str(month)][key]:
                raise WalkForwardPredictionVerificationError(f"aggregate/month semantic mismatch: {name}:{month}")
    query_outcome_rows,query_path_rows=_verify_nonfinal_query_outcomes(
        repository,registry,regimes,output,
    )
    state={
        "schema_version":SCHEMA,"status":"verified","passed":True,
        "preregistration_digest":prereg["preregistration_digest"],"store_result_digest":seal["result_digest"],
        "store_result_sha256":producer._sha(output/"SEALED.json"),"verified_months":len(verified_months),
        "verified_queries":producer.EXPECTED_QUERIES,"verified_neighbor_prediction_rows":seal["neighbor_prediction_rows"],
        "verified_baseline_prediction_rows":seal["baseline_prediction_rows"],"verified_continuous_prediction_rows":seal["continuous_prediction_rows"],
        "verified_path_prediction_rows":seal["path_prediction_rows"],"registry_episode_overlap_count":prereg["registry_episode_overlap_count"],
        "independently_recomputed_nonfinal_query_outcome_rows":query_outcome_rows,
        "independently_recomputed_nonfinal_query_path_rows":query_path_rows,
        "gates":{"all_neighbor_formulas_reconstructed":True,"all_baselines_reconstructed_from_closed_prior_receipts_only":True,"all_continuous_quantiles_reconstructed":True,"all_pointwise_paths_reconstructed":True,"all_nonfinal_query_outcomes_match_independent_oracle":True,"all_month_prediction_before_outcome_orders_valid":True,"all_final_query_outcomes_unopened":True,"all_physical_seals_valid":True},
        "historical_query_outcomes_accessed":True,"final_period_result_opened":False,"evaluation_authorized":True,
        "production_promotion_authorized":False,"elapsed_seconds":perf_counter()-started,"created_at":datetime.now(timezone.utc).isoformat(),
    }
    return {**state,"verification_digest":stable_hash(state)}


def publish(repository: Path) -> Path:
    value=verify(repository); root=repository.resolve(strict=True)/producer.VERIFICATION_RELATIVE; root.mkdir(parents=True,exist_ok=True); path=root/"VERIFIED.json"
    if path.exists(): raise WalkForwardPredictionVerificationError("prediction verification already exists")
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"w") as handle: json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    return path


def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository",required=True,type=Path); args=parser.parse_args(argv)
    print(publish(args.repository)); return 0


if __name__=="__main__": raise SystemExit(main())

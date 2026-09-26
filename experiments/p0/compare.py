"""Paired daily comparison of a candidate run against a reference run.

Both runs are scored on the same (symbol, date) rows; each metric is differenced
date by date and tested with a Newey-West t on the daily differences.

Usage:
    PYTHONPATH=.:src .venv/bin/python -m experiments.p0.compare --dataset nasdaq \
        --candidate e1e2 --reference baseline
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from experiments.p0.evaluate import PREVIOUSLY_OPENED, ROOT, daily_metrics, load_store, newey_west_t

METRICS = ("rps_model", "ic_realized_r20", "ic_exret5", "ic_exret20", "top_net_r", "top_minus_all")


def paired(dataset: str, candidate: str, reference: str) -> dict[str, object]:
    store, _ = load_store(dataset)
    cols = ["symbol", "date", "y20", "realized_r20", "exret5", "exret20", "r_pct", "liquid"]
    runs = {}
    for name in (candidate, reference):
        pred = pd.read_parquet(ROOT / dataset / "runs" / name / "predictions.parquet")
        runs[name] = pred
    keys = runs[candidate][["symbol", "date"]].merge(runs[reference][["symbol", "date"]])
    daily = {}
    for name, pred in runs.items():
        frame = keys.merge(pred, on=["symbol", "date"]).merge(store[cols], on=["symbol", "date"])
        frame = frame.loc[frame["y20"] >= 0]
        daily[name] = daily_metrics(frame, "y20", "realized_r20", dataset).set_index("date")
    a, b = daily[candidate], daily[reference]
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    def block(mask) -> dict[str, float]:
        out = {"dates": int(mask.sum())}
        for m in METRICS:
            if m not in a:
                continue
            # lower RPS is better: report reference minus candidate so positive = better
            diff = (b[m] - a[m]) if m == "rps_model" else (a[m] - b[m])
            diff = diff[mask]
            key = "rps_improvement" if m == "rps_model" else f"delta_{m}"
            out[f"{key}_mean"] = float(diff.mean())
            out[f"{key}_t"] = newey_west_t(diff.to_numpy())
        rps_ref = b.loc[mask, "rps_model"].sum()
        out["rps_skill_vs_reference_pct"] = float(100 * (1 - a.loc[mask, "rps_model"].sum() / rps_ref))
        return out

    everything = pd.Series(True, index=common)
    result = {"candidate": candidate, "reference": reference, "all": block(everything)}
    if dataset in PREVIOUSLY_OPENED:
        lo, hi = PREVIOUSLY_OPENED[dataset]
        result["before_previously_opened"] = block(pd.Series(common < lo, index=common))
        result["previously_opened"] = block(pd.Series((common >= lo) & (common <= hi), index=common))
    years = pd.Series(common.year, index=common)
    result["positive_years_ic_exret20"] = int(sum(
        (a.loc[years == y, "ic_exret20"] - b.loc[years == y, "ic_exret20"]).mean() > 0
        for y in years.unique()))
    result["years"] = int(years.nunique())
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", default="baseline")
    args = parser.parse_args(argv)
    result = paired(args.dataset, args.candidate, args.reference)
    out = ROOT / args.dataset / "runs" / args.candidate / f"compare_vs_{args.reference}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

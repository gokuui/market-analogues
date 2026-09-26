"""P0b go/no-go: does the baseline forecaster sort the trader's own strategy entries?

For each existing strategy trade, the signal date is the NSE session before the
entry date. The baseline R-ladder model (22 causal features, cumulative heads) is
refit once per entry year on NSE label-store rows whose outcome completed before
that year, then scores the entry. Entries on or after the vault start are not
scored.

Question: would skipping the bottom-scored third of entries have improved the
trades the strategy actually took (its own exits and pnl)?

Stop rule, fixed before results (pooled over the listed strategies):
  kept-minus-all mean trade return > 0 with month-block bootstrap 90% lower
  bound > 0, and kept-minus-all > 0 in at least 4 of the 5 eras.
The decision uses only entries whose entry-day bar was not locked (high == low),
because a locked bar cannot be bought at the open.
Caveat: these strategies were developed on the same NSE data, and NSE files are
survivors only, so a pass is optimistic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from experiments.decision.chart_value import FEATURES, benchmark_features, session_key
from experiments.p0.features import REQUIRED, p0_stock_features
from experiments.p0.evaluate import ROOT, VAULT_START, clean_store, fit_predict

LOSER = Path("/home/vinay/code/loser")
STRATEGIES = {
    "mom_atr07_p3": LOSER / "results/audit_mom_atr07_p3/trades.csv",
    "mom_bestv2_p3": LOSER / "results/audit_mom_bestv2_p3/trades.csv",
    "gen498_ref_is": LOSER / "docs/gen498_reference/ar_gen498_IS.csv",
    "gen498_ref_oos": LOSER / "docs/gen498_reference/ar_gen498_OOS.csv",
    "vcp_s19": LOSER / "results/audit_s19/trades.csv",
    "vcp_s20": LOSER / "results/audit_s20/trades.csv",
}
ERAS = ((2010, 2014), (2015, 2016), (2017, 2018), (2019, 2020), (2021, 2022), (2023, 2025))
PRIMARY_ERAS = ERAS[1:]
SKIP_FRACTION = 1 / 3
STOP_RULE = {"min_positive_eras": 4, "bootstrap_lower_90_above_zero": True}
OUTPUT = ROOT / "nse" / "gonogo"


def load_entries() -> pd.DataFrame:
    frames = []
    for name, path in STRATEGIES.items():
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        t = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
        t["strategy"] = name
        t["ret"] = t["pnl_pct"].astype(float) / 100.0
        frames.append(t[["strategy", "symbol", "entry_date", "exit_date", "ret"]])
    return pd.concat(frames, ignore_index=True)


def entry_features(entries: pd.DataFrame, config: str) -> pd.DataFrame:
    from market_analogues.adapters import SourceError, source_from_spec
    from market_analogues.config import load_config
    from market_analogues.types import InstrumentKey
    source = source_from_spec(load_config(config).datasets["nse"])
    bench = source.load_benchmark()
    sessions = session_key(bench["timestamp"])
    bfeat = benchmark_features(bench)
    rows = []
    for symbol, group in entries.groupby("symbol"):
        try:
            stock = source.load(InstrumentKey("nse", symbol))
        except (SourceError, FileNotFoundError, KeyError) as error:
            print(f"skip {symbol}: {error}", file=sys.stderr)
            continue
        stock = stock.dropna(subset=["open", "high", "low", "close", "volume"])
        stock = stock.drop_duplicates("timestamp", keep=False).reset_index(drop=True)
        feats = p0_stock_features(stock, bfeat)
        for index, row in group.iterrows():
            pos = sessions.searchsorted(row["entry_date"].normalize()) - 1
            if pos < 0:
                continue
            signal = sessions[pos]
            if signal in feats.index:
                values = feats.loc[signal]
                if values[list(REQUIRED)].notna().all():
                    bar = stock.loc[session_key(stock["timestamp"]) == row["entry_date"].normalize()]
                    locked = bool(len(bar) and (bar["high"] == bar["low"]).iloc[0])
                    rows.append({"row": index, "signal_date": signal,
                                 "entry_bar_locked": locked, **values.to_dict()})
    return pd.DataFrame(rows).set_index("row")


def month_block_bootstrap(diff_by_month: pd.Series, draws: int = 4000, seed: int = 7):
    rng = np.random.default_rng(seed)
    values = diff_by_month.to_numpy(float)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def evaluate_entries(scored: pd.DataFrame) -> dict[str, object]:
    scored = scored.copy()
    scored["year"] = scored["signal_date"].dt.year
    # skip decision uses the within-year rank so score drift across refits cancels
    scored["pct"] = scored.groupby("year")["er"].rank(pct=True)
    scored["kept"] = scored["pct"] > SKIP_FRACTION
    scored["month"] = scored["signal_date"].dt.to_period("M")
    all_mean = scored["ret"].mean()
    kept_mean = scored.loc[scored["kept"], "ret"].mean()
    # per-month kept-minus-all, weighting months by their trade count
    by_month = scored.groupby("month").apply(
        lambda m: (m.loc[m["kept"], "ret"].mean() - m["ret"].mean()) if m["kept"].any() else 0.0)
    lo, hi = month_block_bootstrap(by_month)
    eras = {}
    for a, b in ERAS:
        e = scored.loc[scored["year"].between(a, b)]
        if len(e) >= 20:
            eras[f"{a}-{b}"] = {"n": int(len(e)),
                                "kept_minus_all_pct": round(100 * (e.loc[e["kept"], "ret"].mean()
                                                                    - e["ret"].mean()), 3),
                                "ic": round(float(e["er"].rank().corr(e["ret"].rank())), 3)}
    primary = [v["kept_minus_all_pct"] for k, v in eras.items()
               if tuple(map(int, k.split("-"))) in PRIMARY_ERAS]
    wins = scored["ret"] > 0
    def pf(s):
        g, l = s[s > 0].sum(), -s[s < 0].sum()
        return float(g / l) if l > 0 else float("inf")
    return {
        "n": int(len(scored)), "ic_spearman": float(scored["er"].rank().corr(scored["ret"].rank())),
        "mean_ret_all_pct": 100 * all_mean, "mean_ret_kept_pct": 100 * kept_mean,
        "mean_ret_skipped_pct": 100 * scored.loc[~scored["kept"], "ret"].mean(),
        "kept_minus_all_pct": 100 * (kept_mean - all_mean),
        # the bootstrap resamples the month-level series, so its point estimate is the
        # equal-weight month mean, not the trade-weighted number above
        "kept_minus_all_month_mean_pct": 100 * float(by_month.mean()),
        "bootstrap_90_month_pct": [100 * lo, 100 * hi],
        "win_rate_all": float(wins.mean()), "win_rate_kept": float(wins[scored["kept"]].mean()),
        "profit_factor_all": pf(scored["ret"]), "profit_factor_kept": pf(scored.loc[scored["kept"], "ret"]),
        "eras": eras, "positive_primary_eras": int(sum(v > 0 for v in primary)),
        "primary_eras_counted": len(primary),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/datasets.example.yaml")
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    entries = load_entries()
    feats = entry_features(entries, str(Path(args.config).resolve()))
    entries = entries.join(feats, how="inner")
    vault = entries["signal_date"] >= VAULT_START
    print(f"entries with features: {len(entries)}; vault excluded: {int(vault.sum())}",
          file=sys.stderr)
    entries = entries.loc[~vault]
    store = pd.read_parquet(ROOT / "nse" / "store.parquet")
    store = clean_store(store.loc[(store["date"] < VAULT_START) & (store["y20"] >= 0)])
    rng = np.random.default_rng(7)
    scored = []
    for year, group in entries.groupby(entries["signal_date"].dt.year):
        train = store.loc[store["completion"] < pd.Timestamp(year=year, month=1, day=1)]
        if len(train) < 50_000:
            print(f"{year}: too little training data ({len(train)})", file=sys.stderr)
            continue
        if len(train) > args.max_rows:
            train = train.iloc[np.sort(rng.choice(len(train), args.max_rows, replace=False))]
        probs, er = fit_predict(train, group, list(FEATURES), "y20")
        group = group.assign(er=er, p_ge2=probs[:, 3:].sum(axis=1), p_stop=probs[:, 0])
        scored.append(group)
        print(f"{year}: scored {len(group)} entries (train {len(train):,})", file=sys.stderr)
    scored = pd.concat(scored)
    result = {"stop_rule": STOP_RULE, "skip_fraction": SKIP_FRACTION,
              "vault_start": str(VAULT_START.date()),
              "caveats": ["strategies developed on the same NSE data",
                          "NSE files contain survivors only"],
              "pooled": evaluate_entries(scored),
              "by_strategy": {name: evaluate_entries(g) for name, g in
                              scored.loc[~scored["entry_bar_locked"].astype(bool)].groupby("strategy")
                              if len(g) >= 60}}
    locked = scored["entry_bar_locked"].astype(bool)
    result["locked_entries"] = {"n": int(locked.sum()),
                                "mean_ret_pct": float(100 * scored.loc[locked, "ret"].mean()),
                                "mean_ret_unlocked_pct": float(100 * scored.loc[~locked, "ret"].mean())}
    result["pooled_fillable"] = evaluate_entries(scored.loc[~locked])
    # the decision uses fillable entries only: a locked entry bar cannot be bought
    pooled = result["pooled_fillable"]
    result["decision"] = {
        "basis": "pooled_fillable",
        "pass": bool(pooled["bootstrap_90_month_pct"][0] > 0
                     and pooled["positive_primary_eras"] >= STOP_RULE["min_positive_eras"]),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUTPUT / "scored_entries.parquet")
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reference implementation of the primary CAL-TRANSFER fixed-test matched-history analysis.

The manuscript's frozen reported values are preserved separately in results/manuscript/.
This implementation follows the final paper's equations and fixed-test design.

Key equations:
  G(x)       = exp(mean_j log E_j(x))
  delta_k    = median_i [log E_t(x_i) - log G(x_i)]
  Ehat_t(x)  = G(x) * exp(delta_k)

Local residual sensitivity extension:
  h(x)       = median_{q in N_K(x)} d_z(x,z_q) + eps
  w_q(x)     = exp(-d_z(x,z_q)/h(x))
  R(x)       = sum_q w_q r_q / sum_q w_q
  alpha_k    = min(0.8, k/10)
  r(x)       = (1-alpha_k) delta_k + alpha_k R(x)
  Ehat_local = G(x) * exp(r(x))
"""
from __future__ import annotations
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, wilcoxon

GPU_ORDER = ["Quadro RTX 6000", "RTX 3060", "RTX 3090", "Titan Xp"]
BUDGETS = (0, 1, 2, 4, 8, 16, 32)

def mape(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean(np.abs(p-y)/y))

def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    den = np.sum((y-y.mean())**2)
    return float(1 - np.sum((y-p)**2)/den) if den > 0 else np.nan

def tau(y, p):
    return float(kendalltau(y, p, nan_policy="omit").statistic)

def geometric_prior(frame, target_gpu):
    src = [g for g in GPU_ORDER if g != target_gpu]
    arr = frame[src].to_numpy(float)
    return np.exp(np.mean(np.log(arr), axis=1))

def global_scale(prior, y_pilot, prior_pilot):
    if len(y_pilot) == 0:
        return 0.0
    return float(np.median(np.log(y_pilot) - np.log(prior_pilot)))

def standardize_source_vectors(frame, target_gpu, fit_index):
    src = [g for g in GPU_ORDER if g != target_gpu]
    Z = np.log(frame[src].to_numpy(float))
    pos = {idx:i for i,idx in enumerate(frame.index)}
    fit_rows = np.array([pos[i] for i in fit_index], dtype=int)
    mu = Z[fit_rows].mean(axis=0)
    sd = Z[fit_rows].std(axis=0)
    sd[sd == 0] = 1.0
    return pd.DataFrame((Z-mu)/sd, index=frame.index, columns=src)

def coverage_order(Z_pool):
    X = Z_pool.to_numpy(float)
    n = len(X)
    centroid = X.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(X-centroid, axis=1)))
    sel = [first]
    remaining = set(range(n)) - {first}
    while remaining:
        best, best_score = None, -np.inf
        for i in sorted(remaining):
            d = min(np.linalg.norm(X[i]-X[j]) for j in sel)
            if d > best_score:
                best, best_score = i, d
        sel.append(best)
        remaining.remove(best)
    return list(Z_pool.index[np.array(sel, dtype=int)])

def quantile_order(prior_pool):
    ranks = pd.Series(prior_pool, index=prior_pool.index).rank(method="average", pct=True)
    first = (ranks-0.5).abs().idxmin()
    sel = [first]
    remaining = [i for i in ranks.index if i != first]
    while remaining:
        best, best_score = None, -np.inf
        for i in remaining:
            d = min(abs(float(ranks.loc[i]) - float(ranks.loc[j])) for j in sel)
            if d > best_score:
                best, best_score = i, d
        sel.append(best)
        remaining.remove(best)
    return sel

def local_predict(test_ids, pilot_ids, prior, target_energy, Z, delta, K=5, cap=0.8):
    if len(pilot_ids) < 3:
        return prior.loc[test_ids].to_numpy(float) * np.exp(delta)
    residual = np.log(target_energy.loc[pilot_ids]) - np.log(prior.loc[pilot_ids])
    alpha = min(cap, len(pilot_ids)/10.0)
    out = []
    for x in test_ids:
        zx = Z.loc[x].to_numpy(float)
        d = np.array([np.linalg.norm(zx - Z.loc[q].to_numpy(float)) for q in pilot_ids], float)
        order = np.argsort(d)[:min(K, len(d))]
        nn_d = d[order]
        h = float(np.median(nn_d) + 1e-12)
        w = np.exp(-nn_d / h)
        rr = residual.iloc[order].to_numpy(float)
        R = float(np.sum(w*rr)/np.sum(w))
        rx = (1-alpha)*delta + alpha*R
        out.append(float(prior.loc[x])*math.exp(rx))
    return np.array(out, float)

def holm(pvals):
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m-rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj

def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    pos = r[d > 0].sum()
    neg = r[d < 0].sum()
    return float((pos-neg)/(pos+neg))

def bootstrap_mean_ci(d, rng, n_boot=5000):
    d = np.asarray(d, float)
    n = len(d)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = rng.choice(d, size=n, replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/ec_nas/cal_transfer_complete_case_88_pivot.csv")
    ap.add_argument("--outer-splits", type=int, default=200)
    ap.add_argument("--random-draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results/cal_transfer_rerun")
    args = ap.parse_args()

    frame = pd.read_csv(args.input, index_col=0)
    missing = [g for g in GPU_ORDER if g not in frame.columns]
    if missing:
        raise ValueError(f"Missing GPU columns: {missing}")
    if len(frame) != 88:
        raise ValueError(f"Expected 88 complete-case architectures, got {len(frame)}")
    if not np.all(np.isfinite(frame[GPU_ORDER].to_numpy(float))) or (frame[GPU_ORDER] <= 0).any().any():
        raise ValueError("Energy table must be finite and strictly positive.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_ids = np.array(frame.index)
    detailed = []

    for split in range(args.outer_splits):
        test_ids = list(rng.choice(all_ids, size=24, replace=False))
        test_set = set(test_ids)
        pool_ids = [i for i in all_ids if i not in test_set]
        assert len(pool_ids) == 64

        for target in GPU_ORDER:
            y = frame[target]
            prior_vals = geometric_prior(frame, target)
            prior = pd.Series(prior_vals, index=frame.index)
            Z = standardize_source_vectors(frame, target, pool_ids)
            cov_order = coverage_order(Z.loc[pool_ids])
            q_order = quantile_order(prior.loc[pool_ids])

            # k=0
            p0 = prior.loc[test_ids].to_numpy(float)
            detailed.append(dict(split=split,target=target,k=0,policy="random",draw=0,
                                 mape=mape(y.loc[test_ids],p0),r2=r2(y.loc[test_ids],p0),
                                 tau=tau(y.loc[test_ids],p0),pilot_wh=0.0))
            detailed.append(dict(split=split,target=target,k=0,policy="coverage",draw=0,
                                 mape=mape(y.loc[test_ids],p0),r2=r2(y.loc[test_ids],p0),
                                 tau=tau(y.loc[test_ids],p0),pilot_wh=0.0))
            detailed.append(dict(split=split,target=target,k=0,policy="quantile",draw=0,
                                 mape=mape(y.loc[test_ids],p0),r2=r2(y.loc[test_ids],p0),
                                 tau=tau(y.loc[test_ids],p0),pilot_wh=0.0))

            for k in BUDGETS[1:]:
                # expected random performance
                for draw in range(args.random_draws):
                    pilots = list(rng.choice(pool_ids, size=k, replace=False))
                    delta = global_scale(prior.loc[pilots], y.loc[pilots], prior.loc[pilots])
                    pred = prior.loc[test_ids].to_numpy(float) * np.exp(delta)
                    detailed.append(dict(split=split,target=target,k=k,policy="random",draw=draw,
                                         mape=mape(y.loc[test_ids],pred),r2=r2(y.loc[test_ids],pred),
                                         tau=tau(y.loc[test_ids],pred),
                                         pilot_wh=float(y.loc[pilots].sum()/3600.0)))

                for policy, order in (("coverage", cov_order), ("quantile", q_order)):
                    pilots = order[:k]
                    delta = global_scale(prior.loc[pilots], y.loc[pilots], prior.loc[pilots])
                    pred = prior.loc[test_ids].to_numpy(float) * np.exp(delta)
                    detailed.append(dict(split=split,target=target,k=k,policy=policy,draw=0,
                                         mape=mape(y.loc[test_ids],pred),r2=r2(y.loc[test_ids],pred),
                                         tau=tau(y.loc[test_ids],pred),
                                         pilot_wh=float(y.loc[pilots].sum()/3600.0)))
                    if policy == "coverage":
                        lp = local_predict(test_ids, pilots, prior, y, Z, delta)
                        detailed.append(dict(split=split,target=target,k=k,policy="coverage_local",draw=0,
                                             mape=mape(y.loc[test_ids],lp),r2=r2(y.loc[test_ids],lp),
                                             tau=tau(y.loc[test_ids],lp),
                                             pilot_wh=float(y.loc[pilots].sum()/3600.0)))

    d = pd.DataFrame(detailed)
    d.to_csv(outdir/"matched_history_detailed.csv", index=False)

    # Collapse random draws inside each split/target first, as required for expected-random comparison.
    x = d.groupby(["split","target","k","policy"], as_index=False).agg(
        mape=("mape","mean"), r2=("r2","mean"), tau=("tau","mean"), pilot_wh=("pilot_wh","mean")
    )
    summary = x.groupby(["k","policy"], as_index=False).agg(
        mean_mape=("mape","mean"), mean_r2=("r2","mean"),
        mean_tau=("tau","mean"), mean_pilot_wh=("pilot_wh","mean")
    )
    summary.to_csv(outdir/"matched_history_summary.csv", index=False)

    # Paired selector inference: GPU-average within each outer split.
    paired = x[x.policy.isin(["coverage","random"])].groupby(
        ["split","k","policy"], as_index=False
    ).mape.mean()
    wide = paired.pivot_table(index=["split","k"], columns="policy", values="mape").reset_index()
    stats = []
    raw_p = []
    for k in BUDGETS[1:]:
        z = wide[wide.k.eq(k)]
        diff = (z["coverage"]-z["random"]).to_numpy(float)
        w = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        ci = bootstrap_mean_ci(diff, np.random.default_rng(args.seed+1000+k))
        raw_p.append(float(w.pvalue))
        stats.append(dict(k=k,coverage_mape=float(z.coverage.mean()),random_mape=float(z.random.mean()),
                          difference=float(diff.mean()),ci_low=ci[0],ci_high=ci[1],
                          p_raw=float(w.pvalue),rank_biserial=rank_biserial(diff)))
    adj = holm(raw_p)
    for row,p in zip(stats,adj):
        row["p_holm"] = float(p)
    pd.DataFrame(stats).to_csv(outdir/"coverage_vs_expected_random.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote detailed rerun outputs to {outdir}")

if __name__ == "__main__":
    main()

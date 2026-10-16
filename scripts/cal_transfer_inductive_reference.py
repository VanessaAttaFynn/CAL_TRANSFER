#!/usr/bin/env python3
"""Auxiliary CAL-TRANSFER inductive robustness stress-test reference implementation.

This analysis intentionally compounds:
  (1) source-side architecture-to-energy prediction error, and
  (2) source-to-target hardware calibration error.

It must therefore be interpreted as a compound robustness check, not an error decomposition.

The public repository snapshot did not preserve the manuscript's historical 50-split seed manifest;
use frozen manuscript Table 7 for the reported values and this script for transparent future reruns.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor

GPU_ORDER = ["Quadro RTX 6000", "RTX 3060", "RTX 3090", "Titan Xp"]
BUDGETS = (0,1,2,4,8,16)

def mape(y,p): return float(np.mean(np.abs(np.asarray(p)-np.asarray(y))/np.asarray(y)))
def r2(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    den=np.sum((y-y.mean())**2)
    return float(1-np.sum((y-p)**2)/den) if den>0 else np.nan
def tau(y,p): return float(kendalltau(y,p,nan_policy="omit").statistic)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hardware", default="data/processed/ec_nas/cal_transfer_complete_case_88.csv")
    ap.add_argument("--architecture", default="data/processed/ec_nas/ec_nas_features.csv")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/cal_transfer_rerun/inductive_reference.csv")
    args=ap.parse_args()

    hw=pd.read_csv(args.hardware)
    arch=pd.read_csv(args.architecture)

    # Keep one architecture-level descriptor row.
    keep=["architecture_hash","params","depth","n_edges","n_conv3x3","n_conv1x1","n_maxpool"]
    missing=[c for c in keep if c not in arch.columns]
    if missing:
        raise ValueError(f"Architecture descriptor table is missing {missing}")
    arch=arch[keep].drop_duplicates("architecture_hash").copy()
    arch["log_params"]=np.log(arch["params"].astype(float))
    feat=["log_params","depth","n_edges","n_conv3x3","n_conv1x1","n_maxpool"]

    wide=hw.pivot(index="architecture_hash",columns="gpu_type",values="target")[GPU_ORDER]
    ids=wide.dropna().index.intersection(arch.architecture_hash)
    if len(ids)!=88:
        raise ValueError(f"Expected 88 architecture IDs with descriptors and four-GPU energy, found {len(ids)}")
    arch=arch.set_index("architecture_hash").loc[ids]
    wide=wide.loc[ids]

    rng=np.random.default_rng(args.seed)
    rows=[]
    for rep in range(args.repeats):
        perm=rng.permutation(ids)
        source_train=list(perm[:44])
        cal=list(perm[44:68])
        test=list(perm[68:88])

        for target in GPU_ORDER:
            sources=[g for g in GPU_ORDER if g!=target]
            pred_source={}
            for source in sources:
                rf=RandomForestRegressor(
                    n_estimators=100,min_samples_leaf=2,max_features=1.0,
                    random_state=args.seed+rep
                )
                rf.fit(arch.loc[source_train,feat], np.log(wide.loc[source_train,source]))
                pred_source[source]=pd.Series(
                    np.exp(rf.predict(arch.loc[cal+test,feat])), index=cal+test
                )

            # Geometric source prior from predicted source-side energy histories.
            pred_matrix=np.column_stack([pred_source[g].loc[cal+test].to_numpy(float) for g in sources])
            prior=pd.Series(np.exp(np.mean(np.log(pred_matrix),axis=1)), index=cal+test)
            y=wide[target]

            for k in BUDGETS:
                if k==0:
                    delta=0.0; pilots=[]
                else:
                    pilots=list(rng.choice(cal,size=k,replace=False))
                    delta=float(np.median(np.log(y.loc[pilots])-np.log(prior.loc[pilots])))
                p=prior.loc[test].to_numpy(float)*np.exp(delta)
                rows.append(dict(
                    repeat=rep,target=target,k=k,mape=mape(y.loc[test],p),r2=r2(y.loc[test],p),
                    tau=tau(y.loc[test],p),
                    pilot_wh=float(y.loc[pilots].sum()/3600.0) if pilots else 0.0
                ))

    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).groupby("k").agg(
        mean_mape=("mape","mean"),sd_mape=("mape","std"),mean_r2=("r2","mean"),
        mean_tau=("tau","mean"),mean_target_wh=("pilot_wh","mean")
    ).to_string())
    print(f"\nWrote {out}")

if __name__=="__main__":
    main()

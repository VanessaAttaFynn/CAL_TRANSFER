#!/usr/bin/env python3
"""Validate and extract the 88-architecture EC-NAS complete-case subset used by CAL-TRANSFER.

This script does not download EC-NAS. It operates on the processed hardware table already
included in this repository under data/processed/ec_nas/.

Expected input:
    data/processed/ec_nas/ec_nas_4v_hardware_features.csv

Outputs:
    data/processed/ec_nas/cal_transfer_complete_case_88.csv
    data/processed/ec_nas/cal_transfer_complete_case_88_pivot.csv
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

GPU_ORDER = ["Quadro RTX 6000", "RTX 3060", "RTX 3090", "Titan Xp"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/ec_nas/ec_nas_4v_hardware_features.csv")
    ap.add_argument("--out-long", default="data/processed/ec_nas/cal_transfer_complete_case_88.csv")
    ap.add_argument("--out-pivot", default="data/processed/ec_nas/cal_transfer_complete_case_88_pivot.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = {"architecture_hash", "gpu_type", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "epochs" in df.columns:
        df = df[df["epochs"].eq(4)]
    if "batch_size" in df.columns:
        df = df[df["batch_size"].eq(256)]

    df = df[df["gpu_type"].isin(GPU_ORDER)].copy()
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df[np.isfinite(df["target"]) & (df["target"] > 0)]

    dup = df.duplicated(["architecture_hash", "gpu_type"], keep=False)
    if dup.any():
        ex = df.loc[dup, ["architecture_hash", "gpu_type"]].head().to_dict("records")
        raise ValueError(
            "Duplicate architecture/GPU observations found. CAL-TRANSFER expects one retained "
            f"four-epoch observation per pair. Examples: {ex}"
        )

    counts = df.groupby("gpu_type")["architecture_hash"].nunique().reindex(GPU_ORDER)
    print("Retained architecture counts by GPU:")
    print(counts.to_string())

    wide = df.pivot(index="architecture_hash", columns="gpu_type", values="target")
    complete_ids = wide.dropna(subset=GPU_ORDER).index
    cc = df[df["architecture_hash"].isin(complete_ids)].copy()

    if len(complete_ids) != 88:
        raise ValueError(
            f"Expected 88 complete-case architectures, found {len(complete_ids)}. "
            "Check that the same processed EC-NAS snapshot used for the manuscript is present."
        )

    out_long = Path(args.out_long)
    out_pivot = Path(args.out_pivot)
    out_long.parent.mkdir(parents=True, exist_ok=True)

    cc = cc.sort_values(["architecture_hash", "gpu_type"])
    cc.to_csv(out_long, index=False)
    wide.loc[complete_ids, GPU_ORDER].sort_index().to_csv(out_pivot)

    print(f"Wrote {len(cc)} long-form rows ({len(complete_ids)} architectures x 4 GPUs) to {out_long}")
    print(f"Wrote energy pivot to {out_pivot}")

if __name__ == "__main__":
    main()

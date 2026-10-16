# Reproducibility guide

## 1. Data source

The processed EC-NAS hardware tables in `data/processed/ec_nas/` are taken from
https://github.com/VanessaAttaFynn/NN-Energy-Prediction-Framework (commit `f99b7bf8cd1b33dc3b39032c1e518ca9d1cd296d`),
where the raw EC-NAS TFRecords and the processing notebooks are kept.

## 2. Manuscript-specific data preparation

Run:

```bash
python scripts/prepare_ecnas_cal_transfer_subset.py
```

The script applies the manuscript's retained four-epoch / batch-size-256 conditions when those fields are
present, verifies one observation per architecture/GPU pair, and requires exactly 88 architectures with
complete observations on all four GPUs.

## 3. Primary matched-history rerun

Run:

```bash
python scripts/cal_transfer_fixed_test.py
```

The implementation follows the final Word equations:

- Eq. (5): geometric source prior;
- Eq. (6): median pilot log-residual;
- Eq. (7): global log-scale calibrated prediction;
- Eqs. (8)-(9): centroid-first / maximin coverage selection;
- Eqs. (10)-(14): local residual sensitivity adapter.

The target test set is fixed before pilot selection inside each outer split.

## 4. Auxiliary inductive stress test

Run:

```bash
python scripts/cal_transfer_inductive_reference.py
```

This is explicitly a **compound robustness test**. It combines source-predictor error and hardware-transfer
error and must not be interpreted as decomposing those two components.

## 5. Frozen manuscript results versus reruns

The CSV files under `results/manuscript/` are the values reported in the clean final manuscript.

The public repository snapshot did not include the original manuscript-specific split/seed manifest. For
that reason the reference scripts use deterministic base seed 42 for future reruns, while the frozen
manuscript CSVs remain the authoritative record of the submitted results.

A permanent archival release should add the exact historical seed/split manifest if it can be recovered.

## 6. Software versions

The repository records minimum package versions rather than the exact historical environment.
This update preserves that fact instead of inventing version numbers. For a DOI release, create and retain
a fully pinned environment from the machine used for the final rerun.

## 7. Provenance

EC-NAS is an external benchmark. CAL-TRANSFER's contribution is the authors' derived fixed-test
cross-GPU calibration analysis and associated statistical comparison. See `ORIGINALITY_AND_PROVENANCE.md`.

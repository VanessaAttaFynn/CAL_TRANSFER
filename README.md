# CAL-TRANSFER

Code and result tables for **CAL-TRANSFER: Few-Shot Cross-GPU Calibration for Neural Network Training-Energy Prediction**.

The analysis uses the directly measured four-epoch EC-NAS hardware records on four GPUs (Quadro RTX 6000,
RTX 3060, RTX 3090, Titan Xp), retaining the 88 architectures measured on all four devices.

## Repository layout

```
data/processed/ec_nas/
  ec_nas_4v_hardware_features.csv   Processed four-GPU EC-NAS hardware table
  ec_nas_features.csv               Architecture descriptors (used by the inductive stress test)
config/cal_transfer_config.yaml     Protocol settings (splits, budgets, seeds)
scripts/
  prepare_ecnas_cal_transfer_subset.py   Builds the 88-architecture complete-case subset
  cal_transfer_fixed_test.py             Primary matched-history fixed-test analysis
  cal_transfer_inductive_reference.py    Auxiliary inductive stress test
results/manuscript/                 Result tables reported in the manuscript
docs/                               Reproducibility and provenance notes
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Running

```bash
python scripts/prepare_ecnas_cal_transfer_subset.py
python scripts/cal_transfer_fixed_test.py
python scripts/cal_transfer_inductive_reference.py
```

The first command writes the complete-case subset to `data/processed/ec_nas/`. The analysis scripts write
their outputs to `results/cal_transfer_rerun/`. The full fixed-test run (200 outer splits × 100 random-pilot
draws) takes roughly 30–60 minutes on a laptop.

The scripts use base seed 42. The tables in `results/manuscript/` are the values reported in the paper; see
`docs/REPRODUCIBILITY.md` for how reruns relate to them.

## Data source

EC-NAS: P. Bakhtiarifard, C. Igel, R. Selvan, EC-NAS: Energy Consumption Aware Tabular Benchmarks for Neural
Architecture Search, ICASSP 2024, 5660–5664. https://doi.org/10.1109/ICASSP48485.2024.10448303

The processed tables come from https://github.com/VanessaAttaFynn/NN-Energy-Prediction-Framework, which
holds the raw EC-NAS records and processing notebooks.

## License

MIT License. See `LICENSE`.

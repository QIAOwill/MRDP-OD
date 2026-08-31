# MRDP-OD

This repository contains the authors' implementation of MRDP-OD for probabilistic
spatiotemporal OD-flow imputation. It includes the model, data pipeline, training
and evaluation code, configuration generators, and the six experiment modules used
in the accompanying study.

## Repository layout

```text
analysis/       Result aggregation and statistical analysis
configs/        MRDP-OD hyperparameter search space
data/           TSV readers, tensor construction, masking, and datasets
evaluation/     Imputation, calibration, network, and downstream metrics
experiments/    Protocol and generators for experiments 01--06
models/         MRDP-OD architecture
sample_data/    Small deterministic synthetic dataset in the required format
scripts/        Sample-data generator
training/       Training and experiment execution
utils/          Configuration, paths, seeds, and I/O helpers
run_experiments.py
```

Generated configurations, checkpoints, predictions, and summaries are written to
`outputs/`, which is excluded from version control.

## Environment

The release was validated on the following local environment:

- Windows 11, Python 3.13.12
- PyTorch 2.13.0+cu130 (CUDA 13.0 build)
- NumPy 2.4.4, pandas 3.0.3, tqdm 4.68.4
- NVIDIA GeForce RTX 5090 D v2, driver 595.97

A CUDA GPU is recommended for the full experiments. Device selection defaults to
`auto`, so protocol checks and small smoke tests can also run on CPU. Create and
activate an isolated environment, then install the pinned dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The CUDA-specific PyTorch wheel in `requirements.txt` follows the validated
machine. For a CPU-only machine or a different CUDA toolchain, install the matching
PyTorch build from the official PyTorch instructions, then install the remaining
packages.

## Data

The research dataset cannot be made public because the authors do not own the
rights required for redistribution. Therefore, this repository includes only
synthetic sample data. It has the same filenames, columns, delimiter, and key
structure expected by the code, but it contains no real observations and must not
be used to reproduce or interpret the paper's numerical results.

The bundled files are ready to use under `sample_data/`. They can be regenerated
deterministically with:

```bash
python scripts/generate_sample_data.py
```

To use authorized data, preserve the directory and filename convention shown in
`sample_data/`, or change `paths.dataset_dir` in `experiments/search_space.py`.
Each region requires five tab-separated UTF-8 files:

- `<region>_OD.txt`
- `<region>_city_static.txt`
- `<region>_city_dynamic.txt`
- `<region>_city_pair_static.txt`
- `<region>_pair_weather.txt`

## Running the code

Run commands from the repository root. The default command is deliberately safe:
it only reports current output status.

```bash
python run_experiments.py
python run_experiments.py --help
python run_experiments.py --steps audit
```

For a short integration run on the synthetic data:

```bash
python run_experiments.py --steps all --smoke
```

For a formal run, execute the stages in order. Hyperparameter selection produced
by `tune` is reused by subsequent experiment groups:

```bash
python run_experiments.py --steps audit tune select main
python run_experiments.py --steps 02_robustness
python run_experiments.py --steps 03_cross_pattern
python run_experiments.py --steps 04_ablation
python run_experiments.py --steps 05_calibration
python run_experiments.py --steps 06_network analyze verify
```

The six experiment identifiers are:

1. `01_overall`: overall performance and hyperparameter selection
2. `02_robustness`: robustness under extended missingness mechanisms
3. `03_cross_pattern`: train/test missingness-pattern transfer
4. `04_ablation`: MRDP-OD component ablations
5. `05_calibration`: validation-based uncertainty calibration
6. `06_network`: network reconstruction and downstream forecasting analysis

Use `--continue-on-error` only when intentionally collecting failures while a batch
continues. Run `python run_experiments.py --steps status` to inspect progress.

## Reproducibility notes

Random seeds, split rules, missingness scenarios, and hyperparameter grids are
defined in `experiments/protocol.py`, `experiments/configs.py`, and
`configs/search_spaces.json`. The synthetic dataset is intended for code-path and
schema validation; paper results require the original authorized data and the full
compute budget.

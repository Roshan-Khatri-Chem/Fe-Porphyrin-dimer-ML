# Copilot Instructions for this repo

This project is notebook-first and centered on a single ML workflow for Fe-porphyrin electronic coupling.

## Core structure
- `notebooks/` is the active workspace and contains the seven narrative notebooks.
- The actual code modules are imported from `../src/` via `sys.path.insert(0, '../src')` in notebooks.
- The primary dataset is `../data/master_dataset.csv`; this is the committed master CSV used by training and inference.
- Models are saved in `../models/`, with `stage2_rf_v4_srsh_aug.pkl` as the reference production model.

## Key notebooks and when to use them
- `01_overview.ipynb`: first checkpoint, dependency check, dataset summary, feature overview.
- `02_features_from_xyz.ipynb`: feature engineering from a dimer `.xyz` file; uses `geometry.load_dimer_xyz`, `monomer_frame`, `dimer_features`.
- `03_homo_and_overlap.ipynb`: optional electronic feature path; requires `pyscf` and `cclib` and an optional Q-Chem `.fchk` file at `../data/processed/fe3_sp.in.fchk`.
- `04_training_data.ipynb`: data pipeline for rebuilding `master_dataset.csv`; inspects `src/build_master_csv.py` and raw caches in `../data/processed/`.
- `05_train_and_predict.ipynb`: trains `v4_srsh_aug`, evaluates held-out test sets, and shows how to predict a new dimer.
- `06_md_validation.ipynb`: benchmark validation against MD reference values.
- `07_negative_results.ipynb`: explored alternatives and rejected methods.

## Workflow conventions
- Run from the notebook directory: `cd notebooks && jupyter lab`.
- Notebooks are already executed and committed; only re-run to verify changes or regenerate outputs.
- Use the committed master CSV unless specifically rebuilding the dataset via `04_training_data.ipynb` and `src/build_master_csv.py`.
- `v4_srsh_aug` uses 11 geometry features + `S12_srsh_cation` + `S12_srsh_cation**2` + `log10(|S12_srsh_cation|)`.
- Training uses `log10_J_cdft` as the target and filters on `J_cdft_eV >= 1e-4` plus non-missing `S12_srsh_cation`.

## Project-specific patterns
- Feature invariance is encoded in `geometry.dimer_features()`, not learned from raw Cartesian coordinates.
- The notebook code often loads relative paths like `../data/...` and `../models/...`, so maintain the notebook root when editing or running.
- Notebook 03 handles missing `.fchk` gracefully by falling back to precomputed HOMO coefficients; do not assume the raw file is always present.
- `master_dataset.csv` contains a mix of meta columns (`dataset`, `split`), reference labels, geometric features, electronic features, and rejected-method outputs.

## What to edit vs what to preserve
- Prefer changes in `../src/` helper modules for core logic and feature computations.
- Keep notebook narrative and explicit step-by-step cells intact, especially in `02_features_from_xyz.ipynb`, `04_training_data.ipynb`, and `05_train_and_predict.ipynb`.
- Avoid changing the data-loading conventions or the `sys.path.insert(0, '../src')` pattern unless the workspace layout is intentionally restructured.

## Validation hints
- If you change feature generation, re-run `02_features_from_xyz.ipynb` and `04_training_data.ipynb` to confirm the same master CSV structure.
- If you change the model definition, verify `05_train_and_predict.ipynb` still reproduces the expected `v4_srsh_aug` pipeline and metrics.
- Use `01_overview.ipynb` to confirm dependency versions and dataset column names after edits.

> Ask for clarification if any path, notebook role, or dataset assumption is unclear.
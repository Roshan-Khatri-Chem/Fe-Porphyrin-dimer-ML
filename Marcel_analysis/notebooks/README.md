# Notebooks: Reading Order

These seven notebooks document the **Fe-porphyrin electronic coupling ML model** from
the ground up. Each notebook is self-contained but builds on concepts from the
previous ones. Read them in order if this is your first encounter with the project;
jump straight to notebook 05 if you just want to use the trained model.

| # | Notebook | What it covers | Runtime |
|---|---|---|---|
| 01 | `01_overview.ipynb` | Project goals, dataset summary, dependency check | <30 s |
| 02 | `02_features_from_xyz.ipynb` | Load a dimer xyz → 11-D symmetry-adapted features | <30 s |
| 03 | `03_homo_and_overlap.ipynb` | Parse Q-Chem `.fchk` → HOMO → rotation-aware $S_{12}$ | ~1 min |
| 04 | `04_training_data.ipynb` | How `master_dataset.csv` was assembled from raw outputs | <30 s |
| 05 | `05_train_and_predict.ipynb` | Train `v4_srsh_aug`, predict on a new dimer | ~1 min |
| 06 | `06_md_validation.ipynb` | Validate against $\Gamma^*$ on 15 MD conformations | <30 s |
| 07 | `07_negative_results.ipynb` | EHT, ZINDO, DFTB+, HF: what was tried and why it didn't help | <30 s |

## How to run

```bash
cd notebooks/
jupyter lab     # or: jupyter notebook
```

All notebooks have already been executed and committed with their outputs visible —
you can read them in GitHub without running anything. Re-run only if you want to
modify code or verify reproducibility.

## Module dependencies

Notebooks import helper modules from `../src/`:

- `geometry.py` — `load_dimer_xyz`, `monomer_frame`, `dimer_features`
- `fchk_parser.py` — Q-Chem `.fchk` → PySCF basis + MO coefficients
- `srsh_s12.py` — `S12Calculator` (rotation-aware fragment-orbital overlap)

Notebook 03 additionally requires `pyscf` and `cclib`. All other notebooks need only
numpy, pandas, sklearn, scipy, joblib, ase, matplotlib (see `../requirements.txt`).

## Production model

After running notebook 05, the trained model is saved to
`../models/v4_srsh_aug_local.pkl`. The pre-trained reference model
(identical to what the notebook produces) is at
`../models/stage2_rf_v4_srsh_aug.pkl`.

To predict $J$ for a new dimer xyz, see the **last cell of notebook 05**.

## Citation

If you use this code, model, or dataset, please cite the report (`../latex/report.pdf`).

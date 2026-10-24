# Fe-porphyrin Electronic Coupling: ML Model and Code

This repository accompanies the report *"Symmetry-Adapted Machine Learning of Electronic
Coupling in Stacked Fe-Porphyrins"* (manuscript and figures hosted separately on Overleaf).

It contains everything needed to:

1. **Predict** the electronic coupling $J$ for a new Fe-porphyrin dimer geometry
   in milliseconds, given an xyz file.
2. **Reproduce** the model from the underlying training data.
3. **Audit** the negative results: which alternative methods (EHT, ZINDO, DFTB+, HF)
   were tested as cheap dimer-level alternatives and why none of them beat the
   recommended monomer-orbital ansatz.

## TL;DR for users who just want a prediction

```python
import sys, joblib
sys.path.insert(0, 'src')
from geometry import load_dimer_xyz, dimer_features
from srsh_s12 import S12Calculator

# (one-time setup: load the trained model and the SRSH HOMO calculator)
pipe = joblib.load('models/stage2_rf_v4_srsh_aug.pkl')
# ... see notebooks/05_train_and_predict.ipynb for the full helper

# Predict
import numpy as np
d = load_dimer_xyz('your_dimer.xyz')
J_eV = ...  # see notebook 05 for the full pipeline
```

The full prediction pipeline is implemented in **`notebooks/05_train_and_predict.ipynb`**,
last cell.

## Recommended reading order

If you're new to the project, walk through the seven Jupyter notebooks in
**`notebooks/`** in order. They build the full picture from scratch in ~10 minutes.

See **`notebooks/README.md`** for a one-line summary of each.

## Repository layout

```
.
├── README.md                   # this file
├── requirements.txt            # pip dependencies
├── notebooks/                  # 7 teaching notebooks (start here)
│   ├── README.md
│   ├── 01_overview.ipynb
│   ├── 02_features_from_xyz.ipynb
│   ├── 03_homo_and_overlap.ipynb
│   ├── 04_training_data.ipynb
│   ├── 05_train_and_predict.ipynb
│   ├── 06_md_validation.ipynb
│   └── 07_negative_results.ipynb
├── src/                        # Python modules imported by notebooks
│   ├── geometry.py             # symmetry-adapted feature engineering
│   ├── fchk_parser.py          # Q-Chem .fchk → PySCF MO conversion
│   ├── srsh_s12.py             # rotation-aware monomer-overlap S_12 calculator
│   ├── eht.py                  # Extended Hückel (negative result)
│   ├── zindo_orca.py           # ZINDO/S via ORCA 6 (negative result)
│   ├── dftb_runner.py          # DFTB+ via conda env (negative result)
│   ├── predict.py              # convenience: full xyz → J prediction
│   ├── build_master_csv.py     # rebuild the master dataset CSV
│   └── …                       # 27 numbered analysis scripts (research log)
├── data/
│   ├── master_dataset.csv      # ★ central data file: 830 rows × 23 columns
│   ├── structures/             # 704 systematic-scan dimer xyz files
│   ├── structures_slater/      # 111 MD-sampled Fe-porphine dimers (Slater set)
│   ├── md_extracted/           # 15 Fe-TPP+peptoid MD-validation xyz files
│   ├── coupling.csv            # raw CDFT-CI couplings for the 704 set
│   └── processed/              # cached feature arrays, model artefacts
└── models/
    ├── stage2_rf_v4_srsh_aug.pkl   # ★ production model
    ├── stage1_classifier.pkl       # near-zero-coupling classifier
    └── …                            # other model variants from the research history
```

(★ = the two files most users will need.)

## The 14-feature input vector

The recommended `v4_srsh_aug` model takes a 14-dimensional vector:

| Index | Name | Meaning |
|---|---|---|
| 0 | `dz` | interplanar distance (Å) |
| 1-2 | `dx`, `dy` | lateral slip in monomer-1 frame (Å) |
| 3 | `cos_theta` | dot product of monomer normals |
| 4-5 | `cos_4phi`, `sin_4phi` | in-plane twist with $C_4$ wrapping |
| 6-7 | `cos_4psi`, `sin_4psi` | tilt-axis direction |
| 8 | `d_lateral` | $\sqrt{dx^2 + dy^2}$ |
| 9-10 | `cos_4alpha`, `sin_4alpha` | slip-direction with $C_4$ wrapping |
| 11 | `S_12` | SRSH cation $\alpha$-SOMO overlap |
| 12 | `S_12^2` | square (proportional to $J^2$) |
| 13 | `log\|S_12\|` | log-magnitude (linearises the relationship) |

All features are invariant under translation, global rotation, and $C_4$ rotation
of either monomer about its porphyrin normal.

## Performance summary

| Set | n | log MAE | Pearson $r$ |
|---|---|---|---|
| 704 systematic test (held-out) | 120 | **0.27** | 0.83 |
| Slater 111 test (held-out) | 23 | 0.32 | 0.82 |
| MD validation (out-of-distribution) | 13 (J>0) | 0.83 | 0.36 |

The MD-validation gap is the open research question — see Section 11 of the
report for the recommended next steps.

## Installation

```bash
# create a clean environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# launch the notebooks
cd notebooks && jupyter lab
```

Tested on Python 3.11. `pyscf` and `cclib` are needed only for notebook 03
(re-computing $S_{12}$ from the raw `.fchk` file); the other notebooks read
from the precomputed master CSV and don't need them.

## License

Code: MIT. Data: see individual file headers.

## Citation

If you use this work, please cite:

> [report citation goes here when published]

## Contact

Open an issue or contact marcel.baer@pnnl.gov for questions.

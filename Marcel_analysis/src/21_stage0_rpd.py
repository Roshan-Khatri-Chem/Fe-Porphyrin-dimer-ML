"""
Stage 0: RPD interference node classifier — tuning and evaluation.
"""

import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'md_extracted')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

import geometry
from geometry import is_rpd_interference_node, dimer_features
from ase.io import read as ase_read

features_v2 = np.load(os.path.join(DATA_DIR, 'features_v2.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
d_lateral = features_v2[:, 8]

GAMMA_STAR_MEV = {
    1: 39.9246, 2: 0.1905, 3: 5.4858, 4: 5.5266, 5: 42.1205,
    6: 7.6328, 7: 22.4902, 8: 30.6074, 9: 0.0000, 10: 0.0136,
    11: 0.5987, 12: 0.3755, 13: 0.0190, 14: 10.1553, 15: 0.0000,
}

# ============================================================
# STEP 3: Tune tolerance on 704-structure dataset
# ============================================================
print("=" * 70)
print("STEP 3: Tune RPD interference tolerance on 704-structure dataset")
print("=" * 70)

results_tol = {}
for tol in [0.3, 0.5, 0.8]:
    fires = np.array([is_rpd_interference_node(d, tol) for d in d_lateral])
    n_fires = fires.sum()
    # FP: fires but coupling >= 1e-3 eV (clearly significant)
    fp = fires & (labels_ev >= 1e-3)
    # TP: fires and coupling < 1e-3 eV
    tp = fires & (labels_ev < 1e-3)
    fp_rate = fp.sum() / n_fires if n_fires > 0 else 0
    # Also check: how many structures with J < 1e-4 and near node are missed?
    near_node = np.array([abs(d - 2.0) < 1.5 or abs(d - 4.6) < 1.5 for d in d_lateral])
    missed = (~fires) & (labels_ev < 1e-4) & near_node

    results_tol[tol] = {'fires': int(n_fires), 'tp': int(tp.sum()),
                         'fp': int(fp.sum()), 'fp_rate': float(fp_rate),
                         'missed': int(missed.sum())}
    print(f"  Tolerance {tol:.1f} Å: fires={n_fires}, TP={tp.sum()}, "
          f"FP={fp.sum()}, FP rate={fp_rate:.2%}")

# Select tolerance: max TP while FP rate < 15%
best_tol = None
for tol in [0.8, 0.5, 0.3]:  # prefer larger if FP rate ok
    if results_tol[tol]['fp_rate'] < 0.15:
        best_tol = tol
        break
if best_tol is None:
    best_tol = 0.3  # fallback to most conservative
print(f"\n→ Selected tolerance: {best_tol} Å "
      f"(FP rate = {results_tol[best_tol]['fp_rate']:.2%})")

# Update the module-level tolerance
geometry.RPD_NODE_TOLERANCE = best_tol

# ============================================================
# STEP 4: Evaluate on 15 MD conformations
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Evaluate Stage 0 on MD conformations")
print("=" * 70)

import joblib
rf_model = joblib.load(os.path.join(MODEL_DIR, 'stage2_rf_final.pkl'))

md_results_no_s0 = {}
md_results_s0 = {}

for conf_id in range(1, 16):
    xyz_path = os.path.join(MD_DIR, f'conf_{conf_id:02d}.xyz')
    if not os.path.exists(xyz_path):
        continue

    atoms = ase_read(xyz_path)
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    M = len(symbols) // 2
    feat = dimer_features(positions[:M], symbols[:M], positions[M:], symbols[M:])
    d_lat = float(feat[8])
    X = feat[:11].reshape(1, -1)

    # Without Stage 0 (v2 baseline)
    log_J = rf_model.predict(X)[0]
    J_mev_no_s0 = 10**log_J * 1000
    md_results_no_s0[conf_id] = J_mev_no_s0

    # With Stage 0
    if is_rpd_interference_node(d_lat, best_tol):
        J_mev_s0 = 0.0
        s0_status = "FIRED"
    else:
        J_mev_s0 = J_mev_no_s0
        s0_status = "pass"
    md_results_s0[conf_id] = J_mev_s0

    gs = GAMMA_STAR_MEV[conf_id]
    print(f"  Conf {conf_id:2d}: d_lat={d_lat:5.2f} Å, Stage0={s0_status:>5}, "
          f"Γ*_ref={gs:8.3f}, pred_v2={J_mev_no_s0:8.2f}, "
          f"pred_s0={J_mev_s0:8.2f} meV")

# Compute log MAE for both (excluding confs 9, 15 for Gamma*=0)
def compute_md_metrics(preds, label=""):
    valid = {c: p for c, p in preds.items() if GAMMA_STAR_MEV[c] > 0}
    log_ratios = {c: np.log10(p / GAMMA_STAR_MEV[c]) if p > 0 else -6.0
                  for c, p in valid.items()}
    abs_lr = {c: abs(v) for c, v in log_ratios.items()}
    mae_all = np.mean(list(abs_lr.values()))

    # In-dist (excl. lateral-OOD confs 2, 10, 13)
    in_dist = {c: v for c, v in abs_lr.items() if c not in [2, 10, 13]}
    mae_id = np.mean(list(in_dist.values())) if in_dist else float('nan')

    # Pearson r (in-dist)
    id_confs = sorted(in_dist.keys())
    if len(id_confs) >= 3:
        log_star = [np.log10(GAMMA_STAR_MEV[c]) for c in id_confs]
        log_pred = [np.log10(max(preds[c], 1e-6)) for c in id_confs]
        r, p = pearsonr(log_star, log_pred)
    else:
        r, p = float('nan'), float('nan')

    return mae_all, mae_id, r

mae_all_no, mae_id_no, r_no = compute_md_metrics(md_results_no_s0)
mae_all_s0, mae_id_s0, r_s0 = compute_md_metrics(md_results_s0)

print(f"\n{'Model':<20} {'MD MAE':>8} {'ID MAE':>8} {'r(id)':>8} {'C9':>8} {'C15':>8}")
print("-" * 56)
print(f"{'v2 (no Stage 0)':<20} {mae_all_no:>8.3f} {mae_id_no:>8.3f} {r_no:>8.3f} "
      f"{md_results_no_s0[9]:>8.2f} {md_results_no_s0[15]:>8.2f}")
print(f"{'v2 + Stage 0':<20} {mae_all_s0:>8.3f} {mae_id_s0:>8.3f} {r_s0:>8.3f} "
      f"{md_results_s0[9]:>8.2f} {md_results_s0[15]:>8.2f}")

# Check false positives in MD set (in-dist confs that Stage 0 fires on)
in_dist_confs = [1, 3, 4, 5, 6, 7, 8, 11, 12, 14]
fp_md = [c for c in in_dist_confs
         if md_results_s0[c] == 0.0 and GAMMA_STAR_MEV[c] > 0.01]
if fp_md:
    print(f"\nWARNING: Stage 0 false positives on in-dist MD confs: {fp_md}")
else:
    print(f"\nNo Stage 0 false positives on in-dist MD conformations.")

# ============================================================
# STEP 5: Sensitivity to node position shifts
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Sensitivity analysis — shifted node positions")
print("=" * 70)

for shift in [-0.3, 0.0, +0.3]:
    nodes_shifted = [2.0 + shift, 4.6 + shift]
    c9_fires = any(abs(md_results_no_s0[9] and feat[8] - n) < best_tol for n in nodes_shifted)

    # Re-evaluate confs 9 and 15 specifically
    c9_d_lat = None
    c15_d_lat = None
    for cid in [9, 15]:
        xyz_path = os.path.join(MD_DIR, f'conf_{cid:02d}.xyz')
        atoms = ase_read(xyz_path)
        symbols = np.array(atoms.get_chemical_symbols())
        positions = atoms.get_positions()
        M = len(symbols) // 2
        f = dimer_features(positions[:M], symbols[:M], positions[M:], symbols[M:])
        dl = float(f[8])
        if cid == 9: c9_d_lat = dl
        if cid == 15: c15_d_lat = dl

    c9_fires = any(abs(c9_d_lat - n) < best_tol for n in nodes_shifted)
    c15_fires = any(abs(c15_d_lat - n) < best_tol for n in nodes_shifted)

    print(f"  Shift {shift:+.1f} Å: nodes at {nodes_shifted}")
    print(f"    Conf 9  (d_lat={c9_d_lat:.2f}): {'FIRES' if c9_fires else 'pass'}")
    print(f"    Conf 15 (d_lat={c15_d_lat:.2f}): {'FIRES' if c15_fires else 'pass'}")

# ============================================================
# Save results and update model config
# ============================================================
print("\n" + "=" * 70)
print("Saving results")
print("=" * 70)

config = {
    'version': 'v2+Stage0',
    'n_features': 11,
    'stage0': {
        'type': 'rpd_interference_node',
        'node_positions_ang': [2.0, 4.6],
        'tolerance_ang': float(best_tol),
        'source': 'Bakenhaster et al. Fig 4a, RFe-Fe=5.0 Ang',
        'fp_rate_704': float(results_tol[best_tol]['fp_rate']),
        'n_fires_704': int(results_tol[best_tol]['fires']),
    },
    'stage1': 'models/stage1_classifier.pkl',
    'stage2': 'models/stage2_rf_final.pkl',
    'performance': {
        'test_704_log_mae': 0.306,
        'md_log_mae_gamma_star': float(mae_all_s0),
        'md_in_dist_log_mae': float(mae_id_s0),
        'md_in_dist_pearson_r': float(r_s0),
    }
}
with open(os.path.join(MODEL_DIR, 'model_config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print("Saved: models/model_config.json")

results = {
    'tolerance_tuning': results_tol,
    'selected_tolerance': float(best_tol),
    'md_no_stage0': {'mae_all': float(mae_all_no), 'mae_id': float(mae_id_no), 'r_id': float(r_no)},
    'md_with_stage0': {'mae_all': float(mae_all_s0), 'mae_id': float(mae_id_s0), 'r_id': float(r_s0)},
}
with open(os.path.join(RESULTS_DIR, 'stage0_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: results/stage0_results.json")

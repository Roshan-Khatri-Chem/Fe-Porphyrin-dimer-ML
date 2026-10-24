"""
2D RPD interference node classifier — adds geometric conditions
(parallel, stacked, low-twist) to reduce the 77% FP rate of 1D.
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

from geometry import dimer_features
from ase.io import read as ase_read
import joblib

features_v2 = np.load(os.path.join(DATA_DIR, 'features_v2.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))

GAMMA_STAR_MEV = {
    1: 39.9246, 2: 0.1905, 3: 5.4858, 4: 5.5266, 5: 42.1205,
    6: 7.6328, 7: 22.4902, 8: 30.6074, 9: 0.0000, 10: 0.0136,
    11: 0.5987, 12: 0.3755, 13: 0.0190, 14: 10.1553, 15: 0.0000,
}


def is_rpd_node_2d(features, tolerance=0.5,
                    cos_theta_min=0.85, dz_max=5.5, cos4phi_min=0.0):
    """2D interference node classifier with geometric conditions."""
    d_lat = features[8]
    cos_theta = features[3]
    cos_4phi = features[4]
    dz = features[0]

    if not (cos_theta > cos_theta_min and dz < dz_max and cos_4phi > cos4phi_min):
        return False

    for node in [2.0, 4.6]:
        if abs(d_lat - node) < tolerance:
            return True
    return False


# ============================================================
# Evaluate 2D classifier on 704-structure dataset
# ============================================================
print("=" * 70)
print("2D Stage 0: Tolerance scan on 704-structure dataset")
print("=" * 70)

# First show the geometric condition filtering effect
all_parallel = features_v2[:, 3] > 0.85
all_stacked = features_v2[:, 0] < 5.5
all_lowtwist = features_v2[:, 4] > 0.0
geom_ok = all_parallel & all_stacked & all_lowtwist
print(f"Geometric conditions: parallel={all_parallel.sum()}, "
      f"stacked={all_stacked.sum()}, low-twist={all_lowtwist.sum()}, "
      f"ALL={geom_ok.sum()} / {len(features_v2)}")

results_2d = {}
for tol in [0.3, 0.5, 0.8]:
    fires = np.array([is_rpd_node_2d(f, tol) for f in features_v2])
    n_fires = fires.sum()
    fp = fires & (labels_ev >= 1e-3)
    tp = fires & (labels_ev < 1e-3)
    fp_rate = fp.sum() / n_fires if n_fires > 0 else 0

    results_2d[tol] = {'fires': int(n_fires), 'tp': int(tp.sum()),
                        'fp': int(fp.sum()), 'fp_rate': float(fp_rate)}
    print(f"  Tolerance {tol:.1f} Å: fires={n_fires}, TP={tp.sum()}, "
          f"FP={fp.sum()}, FP rate={fp_rate:.1%}")

# Also try tighter geometric conditions
print("\nTuning geometric thresholds:")
for ct_min, dz_max, c4p_min, label in [
    (0.85, 5.5, 0.0, "default"),
    (0.90, 5.0, 0.0, "tighter-parallel"),
    (0.85, 5.5, 0.3, "tighter-twist"),
    (0.90, 5.0, 0.3, "tight-all"),
    (0.95, 4.5, 0.3, "very-tight"),
]:
    fires = np.array([is_rpd_node_2d(f, 0.5, ct_min, dz_max, c4p_min)
                      for f in features_v2])
    n = fires.sum()
    fp = (fires & (labels_ev >= 1e-3)).sum()
    tp = (fires & (labels_ev < 1e-3)).sum()
    fpr = fp / n if n > 0 else 0
    print(f"  {label:>18}: cosθ>{ct_min}, dz<{dz_max}, cos4φ>{c4p_min} → "
          f"fires={n}, TP={tp}, FP={fp}, FPR={fpr:.1%}")

# Select best 2D configuration
# Need FP rate < 20%
best_config = None
for ct_min, dz_max, c4p_min, tol, label in [
    (0.95, 4.5, 0.3, 0.5, "very-tight-0.5"),
    (0.95, 4.5, 0.3, 0.3, "very-tight-0.3"),
    (0.90, 5.0, 0.3, 0.5, "tight-all-0.5"),
    (0.90, 5.0, 0.3, 0.3, "tight-all-0.3"),
    (0.85, 5.5, 0.3, 0.5, "tighter-twist-0.5"),
    (0.85, 5.5, 0.3, 0.3, "tighter-twist-0.3"),
    (0.85, 5.5, 0.0, 0.3, "default-0.3"),
]:
    fires = np.array([is_rpd_node_2d(f, tol, ct_min, dz_max, c4p_min)
                      for f in features_v2])
    n = fires.sum()
    if n == 0:
        continue
    fp = (fires & (labels_ev >= 1e-3)).sum()
    fpr = fp / n
    if fpr < 0.20 and n >= 3:
        best_config = (ct_min, dz_max, c4p_min, tol, label, n, fp, fpr)
        break

if best_config:
    ct_min, dz_max, c4p_min, tol, label, n, fp, fpr = best_config
    print(f"\n→ Selected 2D config: {label}")
    print(f"  cosθ > {ct_min}, dz < {dz_max}, cos4φ > {c4p_min}, tolerance = {tol} Å")
    print(f"  Fires: {n}, FP: {fp}, FPR: {fpr:.1%}")
else:
    print("\n→ No 2D configuration achieves FP rate < 20% with ≥3 fires.")
    print("  Trying ultra-tight configurations...")

    for ct_min, dz_max, c4p_min, tol in [
        (0.98, 4.2, 0.5, 0.3),
        (0.95, 4.5, 0.5, 0.3),
        (0.95, 4.5, 0.5, 0.5),
        (0.90, 5.0, 0.5, 0.3),
    ]:
        fires = np.array([is_rpd_node_2d(f, tol, ct_min, dz_max, c4p_min)
                          for f in features_v2])
        n = fires.sum()
        if n == 0: continue
        fp = (fires & (labels_ev >= 1e-3)).sum()
        tp = (fires & (labels_ev < 1e-3)).sum()
        fpr = fp / n
        print(f"  cosθ>{ct_min}, dz<{dz_max}, cos4φ>{c4p_min}, tol={tol}: "
              f"fires={n}, TP={tp}, FP={fp}, FPR={fpr:.1%}")
        if fpr < 0.20 and n >= 2:
            best_config = (ct_min, dz_max, c4p_min, tol, "ultra-tight", n, fp, fpr)
            break

if best_config:
    ct_min, dz_max, c4p_min, tol = best_config[:4]
    print(f"\n→ Final 2D config: cosθ>{ct_min}, dz<{dz_max}, cos4φ>{c4p_min}, tol={tol}")
else:
    ct_min, dz_max, c4p_min, tol = 0.95, 4.5, 0.3, 0.5
    print(f"\n→ Using best-effort 2D config (may have high FPR)")

# ============================================================
# Evaluate on 15 MD conformations
# ============================================================
print("\n" + "=" * 70)
print("2D Stage 0: MD conformation evaluation")
print("=" * 70)

rf_model = joblib.load(os.path.join(MODEL_DIR, 'stage2_rf_final.pkl'))

md_preds_none = {}
md_preds_1d = {}
md_preds_2d = {}
md_feats = {}

for cid in range(1, 16):
    xyz_path = os.path.join(MD_DIR, f'conf_{cid:02d}.xyz')
    atoms = ase_read(xyz_path)
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    M = len(symbols) // 2
    feat = dimer_features(positions[:M], symbols[:M], positions[M:], symbols[M:])
    md_feats[cid] = feat

    X = feat[:11].reshape(1, -1)
    log_J = rf_model.predict(X)[0]
    J_mev = 10**log_J * 1000
    md_preds_none[cid] = J_mev

    # 1D Stage 0 (tolerance=0.3 from previous analysis)
    from geometry import is_rpd_interference_node
    if is_rpd_interference_node(feat[8], 0.3):
        md_preds_1d[cid] = 0.0
    else:
        md_preds_1d[cid] = J_mev

    # 2D Stage 0
    if is_rpd_node_2d(feat, tol, ct_min, dz_max, c4p_min):
        md_preds_2d[cid] = 0.0
    else:
        md_preds_2d[cid] = J_mev

print(f"{'Conf':>4} {'d_lat':>6} {'cosθ':>6} {'cos4φ':>6} {'dz':>5} "
      f"{'1D':>5} {'2D':>5} {'Γ*ref':>8} {'pred_v2':>8} {'pred_2D':>8}")
print("-" * 75)

for cid in range(1, 16):
    f = md_feats[cid]
    s1d = "FIRE" if md_preds_1d[cid] == 0 else "pass"
    s2d = "FIRE" if md_preds_2d[cid] == 0 else "pass"
    gs = GAMMA_STAR_MEV[cid]
    print(f"{cid:>4} {f[8]:>6.2f} {f[3]:>6.3f} {f[4]:>6.3f} {f[0]:>5.2f} "
          f"{s1d:>5} {s2d:>5} {gs:>8.3f} {md_preds_none[cid]:>8.2f} {md_preds_2d[cid]:>8.2f}")

# Compute metrics
def md_metrics(preds, label):
    valid = {c: p for c, p in preds.items() if GAMMA_STAR_MEV[c] > 0}
    lr = {c: np.log10(max(p, 1e-6) / GAMMA_STAR_MEV[c]) for c, p in valid.items()}
    mae_all = np.mean([abs(v) for v in lr.values()])
    # In-dist (excl 2, 10, 13)
    id_lr = {c: v for c, v in lr.items() if c not in [2, 10, 13]}
    mae_id = np.mean([abs(v) for v in id_lr.values()])
    id_confs = sorted(id_lr.keys())
    log_s = [np.log10(GAMMA_STAR_MEV[c]) for c in id_confs]
    log_p = [np.log10(max(preds[c], 1e-6)) for c in id_confs]
    r, _ = pearsonr(log_s, log_p) if len(id_confs) >= 3 else (float('nan'), 0)
    return mae_all, mae_id, r

for label, preds in [("v2 (no Stage 0)", md_preds_none),
                      ("v2 + 1D Stage 0", md_preds_1d),
                      ("v2 + 2D Stage 0", md_preds_2d)]:
    mae_a, mae_i, r_i = md_metrics(preds, label)
    c9 = preds[9]; c15 = preds[15]
    print(f"\n{label:>20}: MD MAE={mae_a:.3f}, ID MAE={mae_i:.3f}, "
          f"r(id)={r_i:.3f}, C9={c9:.2f}, C15={c15:.2f}")

# Check: does conf 3 (the 1D false positive) pass the 2D filter?
f3 = md_feats[3]
print(f"\nConf 3 analysis (1D false positive):")
print(f"  d_lat={f3[8]:.2f}, cosθ={f3[3]:.3f}, cos4φ={f3[4]:.3f}, dz={f3[0]:.2f}")
print(f"  Passes cosθ>{ct_min}? {f3[3] > ct_min}")
print(f"  Passes dz<{dz_max}? {f3[0] < dz_max}")
print(f"  Passes cos4φ>{c4p_min}? {f3[4] > c4p_min}")
print(f"  2D verdict: {'FIRE' if md_preds_2d[3] == 0 else 'PASS (correctly)'}")

# Save results
results = {
    'best_2d_config': {
        'cos_theta_min': float(ct_min),
        'dz_max': float(dz_max),
        'cos4phi_min': float(c4p_min),
        'tolerance': float(tol),
    },
    '2d_fp_rate_704': float(best_config[7]) if best_config else None,
}

mae_a_2d, mae_i_2d, r_i_2d = md_metrics(md_preds_2d, "2D")
results['md_2d'] = {'mae_all': float(mae_a_2d), 'mae_id': float(mae_i_2d), 'r_id': float(r_i_2d)}

with open(os.path.join(RESULTS_DIR, 'stage0_2d_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: results/stage0_2d_results.json")

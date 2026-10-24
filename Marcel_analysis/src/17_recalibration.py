"""
v6: Recalibration and delta-ML correction for Fe-TPP domain shift.
"""

import sys, os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
MD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'md_extracted')

FEAT_NAMES = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi',
              'cos_4psi', 'sin_4psi', 'd_lateral', 'cos_4alpha', 'sin_4alpha']

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(os.path.join(RESULTS_DIR, 'md_validation.csv'))
ok = df[df['status'] == 'OK'].copy().reset_index(drop=True)
N = len(ok)

log_pred = np.log10(ok['J_pred_mev'].values)
log_ref = np.log10(ok['J_ref_mev'].values)
delta = log_ref - log_pred

print(f"Loaded {N} conformations")
print(f"Raw log MAE: {np.mean(np.abs(delta)):.3f}")

# Build feature matrix from the extracted XYZ files
from geometry import dimer_features
from ase.io import read as ase_read
import glob

feat_matrix = np.zeros((N, 11))
for i, row in ok.iterrows():
    conf_id = int(row['conf_id'])
    xyz_path = os.path.join(MD_DIR, f'conf_{conf_id:02d}.xyz')
    atoms = ase_read(xyz_path)
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    M = len(symbols) // 2
    feats = dimer_features(positions[:M], symbols[:M], positions[M:], symbols[M:])
    feat_matrix[i] = feats[:11]

# ============================================================
# PART 1: Global recalibration
# ============================================================
print("\n" + "=" * 70)
print("PART 1: Global recalibration")
print("=" * 70)

# Option A: global offset
offset_A = delta.mean()
print(f"\nOption A — Global offset:")
print(f"  Offset = {offset_A:.4f} log units = factor of {10**offset_A:.4f}")

# LOO-CV for Option A
loo_errors_A = []
for j in range(N):
    mask = np.ones(N, dtype=bool)
    mask[j] = False
    off_j = delta[mask].mean()
    pred_j = log_pred[j] + off_j
    loo_errors_A.append(abs(pred_j - log_ref[j]))
loo_mae_A = np.mean(loo_errors_A)
print(f"  LOO-CV log MAE: {loo_mae_A:.4f}")

# Option B: slope + intercept
coeffs_B = np.polyfit(log_pred, log_ref, 1)
slope_B, intercept_B = coeffs_B
print(f"\nOption B — Slope + intercept:")
print(f"  Slope = {slope_B:.4f}, Intercept = {intercept_B:.4f}")

# LOO-CV for Option B
loo_errors_B = []
for j in range(N):
    mask = np.ones(N, dtype=bool)
    mask[j] = False
    c_j = np.polyfit(log_pred[mask], log_ref[mask], 1)
    pred_j = c_j[0] * log_pred[j] + c_j[1]
    loo_errors_B.append(abs(pred_j - log_ref[j]))
loo_mae_B = np.mean(loo_errors_B)
print(f"  LOO-CV log MAE: {loo_mae_B:.4f}")

# Select best
if loo_mae_A <= loo_mae_B:
    best_method = 'offset'
    best_offset = offset_A
    best_slope = 1.0
    best_loo = loo_mae_A
    print(f"\n→ Selected: Option A (offset only), LOO-CV = {loo_mae_A:.4f}")
else:
    best_method = 'linear'
    best_offset = intercept_B
    best_slope = slope_B
    best_loo = loo_mae_B
    print(f"\n→ Selected: Option B (slope+intercept), LOO-CV = {loo_mae_B:.4f}")

# Save calibration
cal = {'method': best_method, 'offset': float(best_offset),
       'slope': float(best_slope), 'loo_cv_log_mae': float(best_loo)}
with open(os.path.join(MODEL_DIR, 'calibration.json'), 'w') as f:
    json.dump(cal, f, indent=2)
print(f"Saved: models/calibration.json")

# Apply calibration to all 15
log_pred_cal = best_slope * log_pred + best_offset
J_pred_cal_mev = 10 ** log_pred_cal

# ============================================================
# PART 2: Delta-ML model
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Delta-ML Ridge regression")
print("=" * 70)

scaler_delta = StandardScaler()
X_scaled = scaler_delta.fit_transform(feat_matrix)

ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                scoring='neg_mean_absolute_error')
ridge.fit(X_scaled, delta)

print(f"Best alpha: {ridge.alpha_:.2f}")
print(f"Ridge intercept: {ridge.intercept_:.4f}")
print(f"\nRidge coefficients (which features drive the porphine→TPP discrepancy):")
for name, coef in sorted(zip(FEAT_NAMES, ridge.coef_), key=lambda x: -abs(x[1])):
    print(f"  {name:<14}: {coef:+.4f}")

# LOO-CV for delta model
loo_errors_delta = []
for j in range(N):
    mask = np.ones(N, dtype=bool)
    mask[j] = False
    sc_j = StandardScaler()
    X_j = sc_j.fit_transform(feat_matrix[mask])
    ridge_j = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                      scoring='neg_mean_absolute_error')
    ridge_j.fit(X_j, delta[mask])
    X_test_j = sc_j.transform(feat_matrix[j:j+1])
    delta_pred_j = ridge_j.predict(X_test_j)[0]
    log_J_corrected = log_pred[j] + delta_pred_j
    loo_errors_delta.append(abs(log_J_corrected - log_ref[j]))
loo_mae_delta = np.mean(loo_errors_delta)
print(f"\nDelta-ML LOO-CV log MAE: {loo_mae_delta:.4f}")

# Full-data delta predictions (for table — not LOO)
delta_pred_full = ridge.predict(X_scaled)
log_pred_delta = log_pred + delta_pred_full
J_pred_delta_mev = 10 ** log_pred_delta

joblib.dump(ridge, os.path.join(MODEL_DIR, 'delta_ridge.pkl'))
joblib.dump(scaler_delta, os.path.join(MODEL_DIR, 'scaler_delta.pkl'))
print("Saved: models/delta_ridge.pkl, models/scaler_delta.pkl")

# ============================================================
# PART 3: Comparison table
# ============================================================
print("\n" + "=" * 70)
print("PART 3: Comparison table")
print("=" * 70)

header = (f"{'Conf':>4}  {'Γ_ref':>8}  {'Γ_raw':>8}  {'Γ_offset':>8}  {'Γ_delta':>8}  "
          f"{'lr_raw':>7}  {'lr_off':>7}  {'lr_dlt':>7}")
print(header)
print("-" * len(header))

lr_raw = log_pred - log_ref
lr_off = log_pred_cal - log_ref
lr_delta = log_pred_delta - log_ref

for i in range(N):
    conf = int(ok.loc[i, 'conf_id'])
    print(f"{conf:>4}  {ok.loc[i,'J_ref_mev']:>8.4f}  {ok.loc[i,'J_pred_mev']:>8.4f}  "
          f"{J_pred_cal_mev[i]:>8.4f}  {J_pred_delta_mev[i]:>8.4f}  "
          f"{lr_raw[i]:>+7.2f}  {lr_off[i]:>+7.2f}  {lr_delta[i]:>+7.2f}")

print(f"\n{'Overall log MAE':<20}")
print(f"  Raw:              {np.mean(np.abs(lr_raw)):.4f}")
print(f"  Offset:           {np.mean(np.abs(lr_off)):.4f}  (LOO-CV: {loo_mae_A:.4f})")
print(f"  Delta-ML:         {np.mean(np.abs(lr_delta)):.4f}  (LOO-CV: {loo_mae_delta:.4f})")

# Save results
results = {
    'raw_log_mae': float(np.mean(np.abs(lr_raw))),
    'offset_log_mae': float(np.mean(np.abs(lr_off))),
    'offset_loo_cv': float(loo_mae_A),
    'linear_loo_cv': float(loo_mae_B),
    'delta_log_mae_fit': float(np.mean(np.abs(lr_delta))),
    'delta_loo_cv': float(loo_mae_delta),
    'calibration': cal,
    'ridge_alpha': float(ridge.alpha_),
    'ridge_coefficients': dict(zip(FEAT_NAMES, ridge.coef_.tolist())),
}
with open(os.path.join(RESULTS_DIR, 'recalibration_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# ============================================================
# PART 3b: Updated validation figure
# ============================================================
print("\n" + "=" * 70)
print("Generating updated validation figure")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Color by coupling regime
def get_colors(refs):
    colors = []
    for j in refs:
        if j < 0.01: colors.append('#d62728')
        elif j < 0.1: colors.append('#ff7f0e')
        elif j < 1.0: colors.append('#2ca02c')
        else: colors.append('#1f77b4')
    return colors

colors = get_colors(ok['J_ref_mev'])
lims = [5e-4, 80]

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label=r'$\Gamma < 0.01$ meV'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=8, label=r'$0.01 \leq \Gamma < 0.1$ meV'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=8, label=r'$0.1 \leq \Gamma < 1$ meV'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label=r'$\Gamma \geq 1$ meV'),
]

for ax_idx, (ax, pred_vals, title, mae_val) in enumerate(zip(axes, [
    ok['J_pred_mev'].values, J_pred_cal_mev, J_pred_delta_mev
], [
    'Raw ML model', f'Offset-corrected', f'Delta-ML corrected'
], [
    np.mean(np.abs(lr_raw)), np.mean(np.abs(lr_off)), np.mean(np.abs(lr_delta))
])):
    ax.scatter(ok['J_ref_mev'], pred_vals, c=colors, s=80, zorder=3,
               edgecolors='k', linewidths=0.5)
    for _, row in ok.iterrows():
        i = row.name
        ax.annotate(f"{int(row['conf_id'])}", (row['J_ref_mev'], pred_vals[i]),
                    xytext=(4, 4), textcoords='offset points', fontsize=7)
    ax.plot(lims, lims, 'k-', linewidth=1)
    ax.fill_between(lims, [x / 3 for x in lims], [x * 3 for x in lims],
                    alpha=0.12, color='grey')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r'Reference $\Gamma$ (meV)')
    if ax_idx == 0:
        ax.set_ylabel(r'Predicted $\Gamma$ (meV)')
    loo_label = ""
    if ax_idx == 1:
        loo_label = f"\nLOO-CV: {loo_mae_A:.2f}"
    elif ax_idx == 2:
        loo_label = f"\nLOO-CV: {loo_mae_delta:.2f}"
    ax.text(0.05, 0.95, f'{title}\nlog MAE = {mae_val:.2f}{loo_label}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    if ax_idx == 2:
        ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

plt.tight_layout()
for ext in ['png', 'pdf']:
    plt.savefig(os.path.join(FIG_DIR, f'pub_fig5_md_validation.{ext}'))
plt.close()
print("Saved: pub_fig5_md_validation.png/pdf (updated with 3 panels)")

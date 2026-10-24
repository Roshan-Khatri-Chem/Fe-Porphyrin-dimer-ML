"""
Corrected validation: compare ML predictions against Gamma* (geometry-only
coupling from Bakenhaster et al. Table 1), not Gamma (full CDFT-CI coupling).
"""

import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

# ============================================================
# Step 1: Correct reference values (Gamma* from Table 1)
# ============================================================
GAMMA_STAR_MEV = {
    1: 39.9246,  2:  0.1905,  3:  5.4858,  4:  5.5266,
    5: 42.1205,  6:  7.6328,  7: 22.4902,  8: 30.6074,
    9:  0.0000, 10:  0.0136, 11:  0.5987, 12:  0.3755,
   13:  0.0190, 14: 10.1553, 15:  0.0000,
}

# Old Gamma reference (from Huang et al. Table 3) for comparison
GAMMA_MEV = {
    1: 0.0027,  2: 0.3755,  3: 0.4327,  4: 0.0435,
    5: 3.1157,  6: 3.1919,  7: 5.0096,  8: 0.9932,
    9: 0.1034, 10: 0.0027, 11: 0.0109, 12: 0.2259,
   13: 0.0163, 14: 0.0327, 15: 0.3510,
}

# ============================================================
# Step 2: Load predictions, compute against Gamma*
# ============================================================
df = pd.read_csv(os.path.join(RESULTS_DIR, 'md_validation.csv'))
ok = df[df['status'] == 'OK'].copy().reset_index(drop=True)

ok['gamma_star_ref'] = ok['conf_id'].map(GAMMA_STAR_MEV)
ok['gamma_ref'] = ok['conf_id'].map(GAMMA_MEV)

# Exclude confs 9 and 15 (Gamma* = 0, log undefined)
valid = ok[~ok['conf_id'].isin([9, 15])].copy()
valid['log_ratio_star'] = np.log10(valid['J_pred_mev'] / valid['gamma_star_ref'])
valid['abs_log_ratio_star'] = valid['log_ratio_star'].abs()

# Old comparison vs Gamma
valid['log_ratio_gamma'] = np.log10(valid['J_pred_mev'] / valid['gamma_ref'])

log_mae_star = valid['abs_log_ratio_star'].mean()
mean_signed = valid['log_ratio_star'].mean()

print("=" * 70)
print("CORRECTED VALIDATION: ML predictions vs Gamma* (geometry-only)")
print("=" * 70)
print(f"\n(a) Log MAE vs Gamma* (N=13): {log_mae_star:.3f}")
print(f"    Mean signed log ratio: {mean_signed:+.3f}")
print(f"    (positive = model overpredicts)")

# (b) Excluding confs 1 and 14 (large intramolecular distortion)
distortion_confs = []
for _, row in valid.iterrows():
    gs = row['gamma_star_ref']
    g = row['gamma_ref']
    if gs > 0 and g > 0:
        dist_ratio = np.log10(gs / g)
        if dist_ratio > 1:
            distortion_confs.append(int(row['conf_id']))

print(f"\n    Conformations with large intramolecular distortion "
      f"(log10(Gamma*/Gamma) > 1): {distortion_confs}")

clean = valid[~valid['conf_id'].isin(distortion_confs)]
log_mae_clean = clean['abs_log_ratio_star'].mean()
print(f"(b) Log MAE vs Gamma* (excl. {distortion_confs}, N={len(clean)}): {log_mae_clean:.3f}")

# (c) Conformations 9 and 15 — geometric nodes
print(f"\n(c) Geometric node conformations:")
for cid in [9, 15]:
    row = ok[ok['conf_id'] == cid].iloc[0]
    print(f"    Conf {cid}: Gamma*_ref = {row['gamma_star_ref']:.4f} meV, "
          f"Gamma_pred = {row['J_pred_mev']:.4f} meV, "
          f"cos_4phi = {row['cos_4phi']:.3f}")

# (d) Pearson r
log_pred_valid = np.log10(valid['J_pred_mev'].values)
log_star_valid = np.log10(valid['gamma_star_ref'].values)
r_star, p_star = pearsonr(log_star_valid, log_pred_valid)
print(f"\n(d) Pearson r(log Gamma*_ref, log Gamma_pred): {r_star:.4f} (p = {p_star:.2e})")

# ============================================================
# Step 3: Full per-conformation table
# ============================================================
print("\n" + "=" * 70)
print("PER-CONFORMATION TABLE")
print("=" * 70)

header = (f"{'Conf':>4}  {'Γ*_ref':>8}  {'Γ_ref':>8}  {'Γ_pred':>8}  "
          f"{'lr_star':>8}  {'lr_gamma':>9}  {'log(Γ*/Γ)':>10}  {'Notes'}")
print(header)
print("-" * len(header))

for _, row in ok.iterrows():
    cid = int(row['conf_id'])
    gs = row['gamma_star_ref']
    g = row['gamma_ref']
    pred = row['J_pred_mev']

    if gs > 0 and pred > 0:
        lr_star = f"{np.log10(pred/gs):>+8.2f}"
    else:
        lr_star = f"{'N/A':>8}"

    if g > 0 and pred > 0:
        lr_gamma = f"{np.log10(pred/g):>+9.2f}"
    else:
        lr_gamma = f"{'N/A':>9}"

    if gs > 0 and g > 0:
        dist = f"{np.log10(gs/g):>10.2f}"
    elif gs == 0:
        dist = f"{'−∞':>10}"
    else:
        dist = f"{'N/A':>10}"

    notes = ""
    if cid in [9, 15]:
        notes = "geometric node (Γ*=0)"
    elif cid in distortion_confs:
        notes = "large intramol. distortion"

    print(f"{cid:>4}  {gs:>8.4f}  {g:>8.4f}  {pred:>8.4f}  "
          f"{lr_star}  {lr_gamma}  {dist}  {notes}")

# ============================================================
# Step 4: Corrected parity plot (two panels)
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# --- Panel A: predicted vs Gamma (old, wrong) ---
ax = axes[0]
lims = [5e-4, 80]

for _, row in ok.iterrows():
    cid = int(row['conf_id'])
    ref = row['gamma_ref']
    pred = row['J_pred_mev']
    if ref <= 0 or pred <= 0:
        continue
    ax.scatter(ref, pred, c='#999999', s=60, zorder=3,
               edgecolors='k', linewidths=0.5)
    ax.annotate(f"{cid}", (ref, pred), xytext=(4, 4),
                textcoords='offset points', fontsize=7, color='#666666')

ax.plot(lims, lims, 'k-', linewidth=1)
ax.fill_between(lims, [x/3 for x in lims], [x*3 for x in lims],
                alpha=0.1, color='grey')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(r'$\Gamma$ (meV) [CDFT-CI, full]')
ax.set_ylabel(r'$\Gamma_\mathrm{pred}$ (meV) [ML model]')
old_mae = ok.dropna(subset=['J_pred_mev']).apply(
    lambda r: abs(np.log10(r['J_pred_mev']/r['gamma_ref']))
    if r['gamma_ref'] > 0 and r['J_pred_mev'] > 0 else np.nan, axis=1).mean()
ax.text(0.05, 0.95, f'vs $\\Gamma$ (wrong ref.)\nlog MAE = {old_mae:.2f}',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))

# --- Panel B: predicted vs Gamma* (correct) ---
ax = axes[1]

# Regular conformations
for _, row in valid.iterrows():
    cid = int(row['conf_id'])
    ref = row['gamma_star_ref']
    pred = row['J_pred_mev']
    if ref <= 0 or pred <= 0:
        continue

    if cid in distortion_confs:
        marker = 's'
        color = '#ff7f0e'
        label_suffix = ''
    else:
        marker = 'o'
        color = '#1f77b4'
        label_suffix = ''

    ax.scatter(ref, pred, c=color, s=70, marker=marker, zorder=3,
               edgecolors='k', linewidths=0.5)
    ax.annotate(f"{cid}", (ref, pred), xytext=(4, 4),
                textcoords='offset points', fontsize=7)

# Geometric nodes (confs 9, 15) — plot at bottom with special marker
for cid in [9, 15]:
    row = ok[ok['conf_id'] == cid].iloc[0]
    pred = row['J_pred_mev']
    ax.scatter(lims[0] * 1.5, pred, c='#d62728', s=80, marker='X', zorder=3,
               edgecolors='k', linewidths=0.5)
    ax.annotate(f"{cid}", (lims[0] * 1.5, pred), xytext=(4, 4),
                textcoords='offset points', fontsize=7, color='#d62728')

ax.plot(lims, lims, 'k-', linewidth=1)
ax.fill_between(lims, [x/3 for x in lims], [x*3 for x in lims],
                alpha=0.1, color='grey')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(r'$\Gamma^*$ (meV) [geometry-only ref.]')
ax.set_ylabel(r'$\Gamma_\mathrm{pred}$ (meV) [ML model]')

ax.text(0.05, 0.95,
        f'vs $\\Gamma^*$ (correct ref.)\nlog MAE = {log_mae_star:.2f}\n'
        f'$r$ = {r_star:.3f}',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.7))

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
           markersize=8, markeredgecolor='k', markeredgewidth=0.5,
           label=f'Standard (N={len(clean)})'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e',
           markersize=8, markeredgecolor='k', markeredgewidth=0.5,
           label=f'Intramol. distortion ({distortion_confs})'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#d62728',
           markersize=8, markeredgecolor='k', markeredgewidth=0.5,
           label=r'Geometric node ($\Gamma^*=0$)'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

plt.tight_layout()
for ext in ['png', 'pdf']:
    plt.savefig(os.path.join(FIG_DIR, f'pub_fig5_md_validation.{ext}'))
plt.close()
print(f"\nSaved: pub_fig5_md_validation.png/pdf")

# ============================================================
# Step 5: Intramolecular distortion analysis
# ============================================================
print("\n" + "=" * 70)
print("INTRAMOLECULAR DISTORTION ANALYSIS")
print("=" * 70)

print(f"\nlog10(Gamma*/Gamma) per conformation:")
print(f"{'Conf':>4}  {'Gamma*':>8}  {'Gamma':>8}  {'log(Γ*/Γ)':>10}")
for cid in range(1, 16):
    gs = GAMMA_STAR_MEV[cid]
    g = GAMMA_MEV[cid]
    if gs > 0 and g > 0:
        ratio = np.log10(gs / g)
        marker = " ***" if abs(ratio) > 1 else ""
        print(f"{cid:>4}  {gs:>8.4f}  {g:>8.4f}  {ratio:>+10.2f}{marker}")
    elif gs == 0 and g > 0:
        print(f"{cid:>4}  {gs:>8.4f}  {g:>8.4f}  {'−∞':>10}  (geometric node)")
    else:
        print(f"{cid:>4}  {gs:>8.4f}  {g:>8.4f}  {'N/A':>10}")

# ============================================================
# Step 6: Clean up v5 calibration artifacts
# ============================================================
print("\n" + "=" * 70)
print("CLEANUP: Removing v5 recalibration artifacts")
print("=" * 70)

for fname in ['calibration.json', 'delta_ridge.pkl', 'scaler_delta.pkl']:
    fpath = os.path.join(MODEL_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
        print(f"  Removed: {fpath}")

# Save corrected results
results = {
    'log_mae_vs_gamma_star_N13': float(log_mae_star),
    'log_mae_vs_gamma_star_clean': float(log_mae_clean),
    'n_clean': int(len(clean)),
    'excluded_distortion_confs': distortion_confs,
    'pearson_r': float(r_star),
    'pearson_p': float(p_star),
    'mean_signed_log_ratio': float(mean_signed),
}
with open(os.path.join(RESULTS_DIR, 'gamma_star_validation.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: results/gamma_star_validation.json")

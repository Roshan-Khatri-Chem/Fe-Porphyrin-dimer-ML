"""
v4 Task 2: Three publication-quality figures.
"""

import sys, os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'coupling.csv')

features = np.load(os.path.join(DATA_DIR, 'features_v2.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
meta = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

train_mask = (meta['split'] == 'train').values
feat_medians = np.median(features[train_mask], axis=0)

rf = joblib.load(os.path.join(MODEL_DIR, 'stage2_rf_v2.pkl'))

# ============================================================
# FIGURE 1 — 2D Coupling Landscape (dz vs cos_4phi)
# ============================================================
print("Generating Figure 1: 2D Coupling Landscape...")

dz_range = np.linspace(2.5, 8.0, 120)
phi_range = np.linspace(-1.0, 1.0, 120)
DZ, PHI = np.meshgrid(dz_range, phi_range)

grid_points = np.tile(feat_medians, (DZ.size, 1))
grid_points[:, 0] = DZ.ravel()
grid_points[:, 4] = PHI.ravel()

log_J_pred = rf.predict(grid_points)
log_J_grid = log_J_pred.reshape(DZ.shape)

fig, ax = plt.subplots(figsize=(6, 5))

cf = ax.contourf(DZ, PHI, log_J_grid, levels=20, cmap='viridis')

# Contour lines at specific levels
contour_levels = [-4, -3, -2, -1]
cs = ax.contour(DZ, PHI, log_J_grid, levels=contour_levels,
                colors='black', linewidths=0.5)
ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

# D4h node line
ax.axhline(-1.0, color='red', linestyle='--', linewidth=1.5)
ax.annotate(r'D$_{4h}$ node ($\varphi = 45°$)',
            xy=(7.9, -1.0), xytext=(7.9, -0.85),
            fontsize=9, color='red', ha='right', va='bottom')

ax.set_xlabel(r'Interplanar spacing $d_z$ (Å)')
ax.set_ylabel(r'In-plane twist $\cos(4\varphi)$')

# Colorbar
cb = plt.colorbar(cf, ax=ax, pad=0.12)
cb.set_label(r'log$_{10}$($J$ / eV)')

# Secondary y-axis: twist angle in degrees
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
# cos(4phi) = val => phi = arccos(val)/4
# cos_4phi = -1 -> phi=45, cos_4phi=0 -> phi=22.5, cos_4phi=+1 -> phi=0
tick_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
tick_labels = []
for v in tick_vals:
    phi_deg = np.degrees(np.arccos(np.clip(v, -1, 1))) / 4.0
    tick_labels.append(f'{phi_deg:.1f}')
ax2.set_yticks(tick_vals)
ax2.set_yticklabels(tick_labels)
ax2.set_ylabel(r'Twist angle $\varphi$ (°)')

plt.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(os.path.join(FIG_DIR, f'pub_fig1.{ext}'))
plt.close()
print("  Saved: pub_fig1.png, pub_fig1.pdf")

# ============================================================
# FIGURE 2 — Feature Importance
# ============================================================
print("Generating Figure 2: Feature Importance...")

importances = rf.named_steps['rf'].feature_importances_

display_labels = [
    r'$d_z$ (Å)',
    r'$d_x$ (Å)',
    r'$d_y$ (Å)',
    r'$\cos\theta$',
    r'$\cos 4\varphi$',
    r'$\sin 4\varphi$',
    r'$\cos 4\psi$',
    r'$\sin 4\psi$',
    r'$d_\mathrm{lat}$ (Å)',
    r'$\cos 4\alpha$',
    r'$\sin 4\alpha$',
]

# Color by group
colors_map = {
    0: '#1f4e79',  # dz - deep blue
    1: '#2e8b8b',  # dx - teal
    2: '#2e8b8b',  # dy - teal
    3: '#d46a00',  # cos_theta - orange
    4: '#2d6a2d',  # cos_4phi - green
    5: '#2d6a2d',  # sin_4phi - green
    6: '#666666',  # cos_4psi - grey
    7: '#666666',  # sin_4psi - grey
    8: '#2e8b8b',  # d_lat - teal
    9: '#2e8b8b',  # cos_4alpha - teal
    10: '#2e8b8b', # sin_4alpha - teal
}

# Sort by importance
sort_idx = np.argsort(importances)  # ascending (for bottom-to-top bar chart)
sorted_imp = importances[sort_idx]
sorted_labels = [display_labels[i] for i in sort_idx]
sorted_colors = [colors_map[i] for i in sort_idx]

fig, ax = plt.subplots(figsize=(5, 5))
y_pos = np.arange(len(importances))
bars = ax.barh(y_pos, sorted_imp, color=sorted_colors, edgecolor='white', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_labels)
ax.set_xlabel('Feature importance (mean decrease in impurity)')
ax.set_xlim(0, 0.45)

# Uniform baseline
ax.axvline(1.0/11, color='grey', linestyle='--', linewidth=1, alpha=0.7)
ax.annotate('uniform\nbaseline', xy=(1.0/11, len(importances) - 0.5),
            fontsize=8, color='grey', ha='center', va='bottom')

# Value labels
for i, (imp, bar) in enumerate(zip(sorted_imp, bars)):
    ax.text(imp + 0.005, i, f'{imp:.3f}', va='center', fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f4e79', label='Stacking distance'),
    Patch(facecolor='#2e8b8b', label='Lateral DOFs'),
    Patch(facecolor='#d46a00', label='Ring tilt'),
    Patch(facecolor='#2d6a2d', label='In-plane twist'),
    Patch(facecolor='#666666', label='Tilt direction'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          framealpha=0.9)

plt.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(os.path.join(FIG_DIR, f'pub_fig2.{ext}'))
plt.close()
print("  Saved: pub_fig2.png, pub_fig2.pdf")

# ============================================================
# FIGURE 3 — Energy vs Coupling
# ============================================================
print("Generating Figure 3: Energy vs Coupling...")

# Load energies
df_coup = pd.read_csv(CSV_PATH)
energy_dict = {}
for _, row in df_coup.iterrows():
    stem = row['Filename'].replace('.out', '')
    energy_dict[stem] = row['Total_Energy_Ha']

E_Ha = np.array([energy_dict[fn] for fn in meta['filename']])
HA_TO_EV = 27.211386245988
E_eV = E_Ha * HA_TO_EV
E_rel = (E_Ha - E_Ha.min()) * HA_TO_EV

cos_4phi = features[:, 4]
near_zero_mask = labels_ev < 1e-4
sig_mask = labels_ev >= 1e-4
zero_mask = labels_ev == 0.0
nz_nonzero_mask = near_zero_mask & ~zero_mask

# Pearson r on significant + nonzero near-zero
valid = labels_ev > 0
r_val, p_val = pearsonr(E_rel[valid], np.log10(labels_ev[valid]))

fig, ax = plt.subplots(figsize=(7, 5))

# Significant structures colored by cos_4phi
sc = ax.scatter(E_rel[sig_mask], np.log10(labels_ev[sig_mask]),
                c=cos_4phi[sig_mask], cmap='RdBu_r', vmin=-1, vmax=1,
                marker='o', s=20, alpha=0.6, zorder=3)

# Near-zero (> 0)
if nz_nonzero_mask.sum() > 0:
    ax.scatter(E_rel[nz_nonzero_mask], np.log10(labels_ev[nz_nonzero_mask]),
               marker='v', s=15, alpha=0.5, color='lightgrey', zorder=2,
               label=r'Near-zero ($J < 10^{-4}$ eV)')

# Zero coupling
if zero_mask.sum() > 0:
    ax.scatter(E_rel[zero_mask], np.full(zero_mask.sum(), -7.0),
               marker='x', s=20, alpha=0.7, color='black', zorder=2,
               label=r'Zero coupling ($J = 0$)')

# Colorbar
cb = plt.colorbar(sc, ax=ax, pad=0.02)
cb.set_label(r'$\cos(4\varphi)$')
# Annotate colorbar
cb.ax.text(1.5, -1.0, 'staggered\n(45°)', fontsize=8, ha='left', va='center',
           transform=cb.ax.get_yaxis_transform())
cb.ax.text(1.5, 1.0, 'eclipsed\n(0°)', fontsize=8, ha='left', va='center',
           transform=cb.ax.get_yaxis_transform())

# Mean E_rel lines
nz_mean_E = E_rel[near_zero_mask].mean()
sig_mean_E = E_rel[sig_mask].mean()

ax.axvline(nz_mean_E, color='grey', linestyle='--', linewidth=1, zorder=1)
ax.text(nz_mean_E, -0.3, 'near-zero\nmean', fontsize=8, ha='center', va='bottom',
        color='grey')

ax.axvline(sig_mean_E, color='grey', linestyle='--', linewidth=1, zorder=1)
ax.text(sig_mean_E, -0.3, 'significant\nmean', fontsize=8, ha='center', va='bottom',
        color='grey')

# Text box
textstr = f'$r = +{r_val:.2f}$, $p < 10^{{-20}}$\nHigher energy → stronger coupling'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

ax.set_xlabel(r'Relative energy $E_\mathrm{rel}$ (eV)')
ax.set_ylabel(r'log$_{10}$($J$ / eV)')
ax.set_xlim(0, 18)
ax.set_ylim(-7, 0)
ax.legend(loc='lower right', framealpha=0.9)

plt.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(os.path.join(FIG_DIR, f'pub_fig3.{ext}'))
plt.close()
print("  Saved: pub_fig3.png, pub_fig3.pdf")

# ============================================================
# Verification
# ============================================================
print("\nFile sizes:")
for i in range(1, 4):
    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'pub_fig{i}.{ext}')
        size = os.path.getsize(path)
        print(f"  pub_fig{i}.{ext}: {size/1024:.0f} KB")

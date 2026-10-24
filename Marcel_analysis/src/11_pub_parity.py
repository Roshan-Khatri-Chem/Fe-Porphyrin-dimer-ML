"""
Publication-quality parity plot: predicted vs measured coupling with GPR
confidence intervals.
"""

import sys, os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error, r2_score

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

# Load data
features = np.load(os.path.join(DATA_DIR, 'features_v2.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
meta = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

# Load models
rf = joblib.load(os.path.join(MODEL_DIR, 'stage2_rf_v2.pkl'))
gpr = joblib.load(os.path.join(MODEL_DIR, 'stage2_gpr_v2.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_stage2_v2.pkl'))

# Filter to significant coupling, test set
sig_mask = labels_ev >= 1e-4
X_sig = features[sig_mask]
y_log_sig = np.log10(labels_ev[sig_mask])
meta_sig = meta[sig_mask].reset_index(drop=True)

test_mask = (meta_sig['split'] == 'test').values
X_test = X_sig[test_mask]
y_true = y_log_sig[test_mask]

# RF predictions (primary model)
y_pred_rf = rf.predict(X_test)

# GPR predictions + uncertainty (for confidence intervals)
X_test_s = scaler.transform(X_test)
y_pred_gpr, y_std_gpr = gpr.predict(X_test_s, return_std=True)

# Metrics
log_mae = mean_absolute_error(y_true, y_pred_rf)
log_r2 = r2_score(y_true, y_pred_rf)

# Decade colors
decades = np.floor(y_true).astype(int)
decade_labels = {-4: r'$[10^{-4},\,10^{-3})$',
                 -3: r'$[10^{-3},\,10^{-2})$',
                 -2: r'$[10^{-2},\,10^{-1})$',
                 -1: r'$[10^{-1},\,1]$'}
decade_colors = {-4: '#440154', -3: '#31688e', -2: '#35b779', -1: '#fde725'}

fig, ax = plt.subplots(figsize=(6, 6))

# Sort by y_true so error bars don't overlap messily
sort_idx = np.argsort(y_true)
y_true_s = y_true[sort_idx]
y_pred_s = y_pred_rf[sort_idx]
y_std_s = y_std_gpr[sort_idx]
decades_s = decades[sort_idx]

# Plot GPR 95% confidence intervals as vertical error bars
# Use GPR std as uncertainty estimate for the RF prediction
ax.errorbar(y_true_s, y_pred_s, yerr=1.96 * y_std_s,
            fmt='none', ecolor='lightgrey', elinewidth=0.8, capsize=0,
            zorder=1, label=r'GPR 95\% CI')

# Scatter by decade
for dec in sorted(decade_labels.keys()):
    mask = decades_s == dec
    if mask.sum() > 0:
        ax.scatter(y_true_s[mask], y_pred_s[mask],
                   c=decade_colors[dec], s=35, alpha=0.8,
                   edgecolors='k', linewidth=0.3, zorder=3,
                   label=f'{decade_labels[dec]} (N={mask.sum()})')

# 1:1 line
lims = [-4.2, -0.2]
ax.plot(lims, lims, 'k-', linewidth=1.2, zorder=2)

# ±0.5 log-unit bands (factor of ~3)
ax.fill_between(lims, [l - 0.5 for l in lims], [l + 0.5 for l in lims],
                color='grey', alpha=0.08, zorder=0)
ax.plot(lims, [l + 0.5 for l in lims], 'k--', linewidth=0.6, alpha=0.4, zorder=2)
ax.plot(lims, [l - 0.5 for l in lims], 'k--', linewidth=0.6, alpha=0.4, zorder=2)

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.set_xlabel(r'Measured $\log_{10}(J\,/\,\mathrm{eV})$')
ax.set_ylabel(r'Predicted $\log_{10}(J\,/\,\mathrm{eV})$')

# Metrics text box
textstr = (f'Log-MAE $= {log_mae:.3f}$\n'
           f'$R^2 = {log_r2:.3f}$\n'
           f'$N = {len(y_true)}$')
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='grey')
ax.text(0.04, 0.96, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Annotation for the bands
ax.text(-0.55, -0.55 + 0.5 + 0.05, r'$\pm\,0.5$ log', fontsize=8,
        color='grey', rotation=45, ha='center', va='bottom')

ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

plt.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(os.path.join(FIG_DIR, f'pub_fig4.{ext}'))
plt.close()

print(f"Saved: pub_fig4.png ({os.path.getsize(os.path.join(FIG_DIR, 'pub_fig4.png'))//1024} KB)")
print(f"Saved: pub_fig4.pdf ({os.path.getsize(os.path.join(FIG_DIR, 'pub_fig4.pdf'))//1024} KB)")
print(f"Metrics: log-MAE = {log_mae:.3f}, R² = {log_r2:.3f}")

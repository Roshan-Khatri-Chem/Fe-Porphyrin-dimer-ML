"""
Phase 4: Exploratory Data Analysis

Produce 6 plots and save to results/figures/.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

FEATURE_NAMES = ['dz (Å)', 'dx (Å)', 'dy (Å)', 'cos_theta', 'cos_4phi', 'sin_4phi', 'cos_4psi', 'sin_4psi']

# Load data
features = np.load(os.path.join(DATA_DIR, 'features.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
labels_log = np.load(os.path.join(DATA_DIR, 'labels_log.npy'))

# Mask for finite log values
valid = np.isfinite(labels_log)

# Coupling decade for coloring
decades = np.full(len(labels_ev), -99)
for i, v in enumerate(labels_ev):
    if v > 0:
        decades[i] = int(np.floor(np.log10(v)))

# ---- Plot 1: Feature histograms ----
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.hist(features[:, i], bins=40, edgecolor='black', alpha=0.7)
    ax.set_title(FEATURE_NAMES[i])
    ax.set_ylabel('Count')
fig.suptitle('Feature Distributions', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_feature_histograms.png'), dpi=150)
plt.close()
print("Plot 1: eda_feature_histograms.png saved")

# ---- Plot 2: Feature vs coupling ----
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
cmap = plt.cm.viridis
for i, ax in enumerate(axes.flat):
    mask = valid
    x = features[mask, i]
    y = labels_log[mask]
    d = decades[mask]
    sc = ax.scatter(x, y, c=d, cmap=cmap, alpha=0.3, s=10)
    r, _ = pearsonr(x, y)
    ax.set_title(f'{FEATURE_NAMES[i]} (r={r:.3f})')
    ax.set_xlabel(FEATURE_NAMES[i])
    ax.set_ylabel('log10(J / eV)')
fig.suptitle('Features vs log10(Coupling)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_feature_vs_coupling.png'), dpi=150)
plt.close()
print("Plot 2: eda_feature_vs_coupling.png saved")

# Check dz-coupling anticorrelation
r_dz, _ = pearsonr(features[valid, 0], labels_log[valid])
print(f"  dz-coupling Pearson r = {r_dz:.4f}")
if r_dz > 0:
    print("  ⚠ WARNING: dz-coupling correlation is POSITIVE — expected negative!")
    print("  This may indicate a bug in geometry or data loading.")

# ---- Plot 3: Correlation matrix ----
fig, ax = plt.subplots(figsize=(10, 8))
corr = np.corrcoef(features.T)
short_names = ['dz', 'dx', 'dy', 'cos_θ', 'cos_4φ', 'sin_4φ', 'cos_4ψ', 'sin_4ψ']
sns.heatmap(corr, annot=True, fmt='.2f', xticklabels=short_names,
            yticklabels=short_names, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_correlation_matrix.png'), dpi=150)
plt.close()
print("Plot 3: eda_correlation_matrix.png saved")

# ---- Plot 4: Coupling distribution ----
fig, ax = plt.subplots(figsize=(10, 6))
log_vals = labels_log[valid]
ax.hist(log_vals, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.log10(1e-4), color='red', linestyle='--', label='1e-4 eV threshold')
ax.set_xlabel('log10(Coupling / eV)')
ax.set_ylabel('Count')
ax.set_title('Coupling Distribution')
# Annotate decades
for dec in range(-6, 0):
    count = np.sum((log_vals >= dec) & (log_vals < dec + 1))
    if count > 0:
        ax.annotate(f'n={count}', xy=(dec + 0.5, 0), xytext=(dec + 0.5, ax.get_ylim()[1] * 0.85),
                    ha='center', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_coupling_distribution.png'), dpi=150)
plt.close()
print("Plot 4: eda_coupling_distribution.png saved")

# ---- Plot 5: dz-slippage heatmap ----
fig, ax = plt.subplots(figsize=(10, 8))
dz = features[valid, 0]
d_lateral = np.sqrt(features[valid, 1]**2 + features[valid, 2]**2)
hb = ax.hexbin(dz, d_lateral, C=labels_log[valid], gridsize=25, cmap='viridis',
               reduce_C_function=np.mean)
cb = plt.colorbar(hb, ax=ax)
cb.set_label('mean log10(J / eV)')
ax.set_xlabel('dz (Å)')
ax.set_ylabel('d_lateral = sqrt(dx² + dy²) (Å)')
ax.set_title('Interplanar Distance vs Lateral Slippage')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_dz_slippage_heatmap.png'), dpi=150)
plt.close()
print("Plot 5: eda_dz_slippage_heatmap.png saved")

# ---- Plot 6: PCA ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
mask = valid
sc = ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=labels_log[mask],
                cmap='viridis', alpha=0.5, s=15)
cb = plt.colorbar(sc, ax=ax)
cb.set_label('log10(J / eV)')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA of Features, colored by Coupling')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_pca.png'), dpi=150)
plt.close()
print("Plot 6: eda_pca.png saved")
print(f"\nPCA explained variance (first 3): {pca.explained_variance_ratio_[:3]}")

print("\n✓ All 6 EDA plots saved to results/figures/")

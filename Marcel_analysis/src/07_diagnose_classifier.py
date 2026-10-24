"""
Diagnose Stage 1 classifier failure.
Plot features for near-zero vs significant test structures.
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')

FEATURE_NAMES = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi', 'cos_4psi', 'sin_4psi']

features = np.load(os.path.join(DATA_DIR, 'features.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
metadata = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

test_mask = metadata['split'] == 'test'
X_test = features[test_mask.values]
y_test = labels_ev[test_mask.values]

near_zero = y_test < 1e-4
significant = ~near_zero

print(f"Test set: {near_zero.sum()} near-zero, {significant.sum()} significant")

# What coupling values do the near-zero structures have?
print(f"\nNear-zero test coupling values:")
nz_vals = y_test[near_zero]
for v in sorted(nz_vals):
    print(f"  {v:.6e} eV")

# What about the borderline significant (1e-4 to 1e-3)?
borderline = (y_test >= 1e-4) & (y_test < 1e-3)
print(f"\nBorderline [1e-4, 1e-3) test coupling values ({borderline.sum()}):")
bl_vals = y_test[borderline]
for v in sorted(bl_vals)[:10]:
    print(f"  {v:.6e} eV")

# Feature comparison
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for i, ax in enumerate(axes.flat):
    nz_feat = X_test[near_zero, i]
    sig_feat = X_test[significant, i]
    bl_feat = X_test[borderline, i]

    ax.hist(sig_feat, bins=20, alpha=0.5, label=f'Significant (N={significant.sum()})',
            color='steelblue', density=True)
    ax.hist(nz_feat, bins=15, alpha=0.6, label=f'Near-zero (N={near_zero.sum()})',
            color='red', density=True)
    ax.hist(bl_feat, bins=10, alpha=0.4, label=f'Borderline (N={borderline.sum()})',
            color='orange', density=True)
    ax.set_title(FEATURE_NAMES[i])
    ax.set_ylabel('Density')
    if i == 0:
        ax.legend(fontsize=8)

fig.suptitle('Feature Distributions: Near-zero vs Significant vs Borderline [1e-4,1e-3)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'diagnose_classifier_features.png'), dpi=150)
plt.close()
print("\nSaved: diagnose_classifier_features.png")

# Print feature statistics
print("\nFeature statistics (test set):")
print(f"{'Feature':<12} {'NZ mean':>10} {'NZ std':>10} {'Sig mean':>10} {'Sig std':>10} {'Overlap?':>10}")
print("-" * 64)
for i, name in enumerate(FEATURE_NAMES):
    nz_m = X_test[near_zero, i].mean()
    nz_s = X_test[near_zero, i].std()
    sig_m = X_test[significant, i].mean()
    sig_s = X_test[significant, i].std()
    # Rough overlap: distance in stds
    pooled_s = max((nz_s + sig_s) / 2, 1e-10)
    sep = abs(nz_m - sig_m) / pooled_s
    overlap = "LOW" if sep > 1.5 else ("MEDIUM" if sep > 0.7 else "HIGH")
    print(f"{name:<12} {nz_m:>10.3f} {nz_s:>10.3f} {sig_m:>10.3f} {sig_s:>10.3f} {overlap:>10}")

# 2D scatter: dz vs d_lateral colored by class
fig, ax = plt.subplots(figsize=(10, 8))
d_lat_test = np.sqrt(X_test[:, 1]**2 + X_test[:, 2]**2)
ax.scatter(X_test[significant, 0], d_lat_test[significant], alpha=0.4, s=20,
           label='Significant', color='steelblue')
ax.scatter(X_test[near_zero, 0], d_lat_test[near_zero], alpha=0.7, s=40,
           label='Near-zero', color='red', marker='x', linewidths=2)
ax.scatter(X_test[borderline, 0], d_lat_test[borderline], alpha=0.7, s=40,
           label='Borderline [1e-4,1e-3)', color='orange', marker='^')
ax.set_xlabel('dz (Å)')
ax.set_ylabel('d_lateral (Å)')
ax.set_title('Test set: dz vs d_lateral by coupling class')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'diagnose_classifier_dz_dlat.png'), dpi=150)
plt.close()
print("Saved: diagnose_classifier_dz_dlat.png")

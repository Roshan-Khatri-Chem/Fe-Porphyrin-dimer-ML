"""
Phase 7: Physical Interpretation

Generate interpretation plots using the trained models.
"""

import sys
import os
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')

FEATURE_NAMES = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi', 'cos_4psi', 'sin_4psi']
FEATURE_UNITS = ['Å', 'Å', 'Å', '', '', '', '', '']

# Load data and models
features = np.load(os.path.join(DATA_DIR, 'features.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))

# Filter to significant
sig_mask = labels_ev >= 1e-4
X_sig = features[sig_mask]

scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_stage2.pkl'))
gpr_model = joblib.load(os.path.join(MODEL_DIR, 'stage2_gpr.pkl'))
rf_model = joblib.load(os.path.join(MODEL_DIR, 'stage2_rf.pkl'))
krr_model = joblib.load(os.path.join(MODEL_DIR, 'stage2_krr.pkl'))

# Medians and stds of training features
medians = np.median(X_sig, axis=0)
stds = np.std(X_sig, axis=0)
means = np.mean(X_sig, axis=0)

# ---- Plot 1: 1D profiles ----
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
dz_profile_x = None
dz_profile_y = None
dz_profile_std = None

for i, ax in enumerate(axes.flat):
    # Sweep feature i, fix others at median
    lo = means[i] - 2 * stds[i]
    hi = means[i] + 2 * stds[i]
    x_sweep = np.linspace(lo, hi, 100)

    X_sweep = np.tile(medians, (100, 1))
    X_sweep[:, i] = x_sweep

    X_sweep_s = scaler.transform(X_sweep)
    y_pred, y_std = gpr_model.predict(X_sweep_s, return_std=True)

    ax.plot(x_sweep, y_pred, 'b-', linewidth=2)
    ax.fill_between(x_sweep, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.2, color='blue')
    unit_str = f' ({FEATURE_UNITS[i]})' if FEATURE_UNITS[i] else ''
    ax.set_xlabel(f'{FEATURE_NAMES[i]}{unit_str}')
    ax.set_ylabel('log10(J / eV)')
    ax.set_title(FEATURE_NAMES[i])

    if i == 0:
        dz_profile_x = x_sweep
        dz_profile_y = y_pred
        dz_profile_std = y_std

fig.suptitle('1D Feature Profiles (GPR, others at median)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'interp_1d_profiles.png'), dpi=150)
plt.close()
print("Plot 1: interp_1d_profiles.png saved")

# ---- Plot 2: dz-phi surface ----
fig, ax = plt.subplots(figsize=(10, 8))
n_grid = 50
dz_range = np.linspace(means[0] - 2 * stds[0], means[0] + 2 * stds[0], n_grid)
phi_deg_range = np.linspace(0, 90, n_grid)  # degrees
phi_rad_range = np.deg2rad(phi_deg_range)

Z = np.zeros((n_grid, n_grid))
for j, phi in enumerate(phi_rad_range):
    X_grid = np.tile(medians, (n_grid, 1))
    X_grid[:, 0] = dz_range
    X_grid[:, 4] = np.cos(4.0 * phi)
    X_grid[:, 5] = np.sin(4.0 * phi)
    X_grid_s = scaler.transform(X_grid)
    Z[:, j] = gpr_model.predict(X_grid_s)

DZ, PHI = np.meshgrid(dz_range, phi_deg_range, indexing='ij')
ct = ax.contourf(DZ, PHI, Z, levels=30, cmap='viridis')
plt.colorbar(ct, label='log10(J / eV)')
ax.set_xlabel('dz (Å)')
ax.set_ylabel('φ (degrees, 0-90°)')
ax.set_title('Coupling vs dz and in-plane twist')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'interp_dz_phi_surface.png'), dpi=150)
plt.close()
print("Plot 2: interp_dz_phi_surface.png saved")

# ---- Plot 3: Slippage map ----
fig, ax = plt.subplots(figsize=(10, 8))
n_grid = 50
dx_range = np.linspace(-4, 4, n_grid)
dy_range = np.linspace(-4, 4, n_grid)
Z = np.zeros((n_grid, n_grid))
for j, dy in enumerate(dy_range):
    X_grid = np.tile(medians, (n_grid, 1))
    X_grid[:, 1] = dx_range
    X_grid[:, 2] = dy
    X_grid_s = scaler.transform(X_grid)
    Z[:, j] = gpr_model.predict(X_grid_s)

DX, DY = np.meshgrid(dx_range, dy_range, indexing='ij')
ct = ax.contourf(DX, DY, Z, levels=30, cmap='viridis')
plt.colorbar(ct, label='log10(J / eV)')
ax.set_xlabel('dx (Å)')
ax.set_ylabel('dy (Å)')
ax.set_title(f'Coupling vs Lateral Slippage (dz={medians[0]:.1f} Å)')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'interp_slippage_map.png'), dpi=150)
plt.close()
print("Plot 3: interp_slippage_map.png saved")

# ---- Plot 4: Feature importance comparison ----
fig, ax = plt.subplots(figsize=(10, 6))
rf_best = rf_model.named_steps['rf']
rf_imp = rf_best.feature_importances_

# KRR sensitivity: mean |ΔJ/Δf_i| where Δf_i = 1 std
krr_sens = np.zeros(8)
for i in range(8):
    X_plus = np.tile(medians, (1, 1))
    X_minus = np.tile(medians, (1, 1))
    X_plus[0, i] = medians[i] + stds[i]
    X_minus[0, i] = medians[i] - stds[i]
    y_plus = krr_model.predict(X_plus)
    y_minus = krr_model.predict(X_minus)
    krr_sens[i] = abs(y_plus[0] - y_minus[0]) / 2.0

# Sort by RF importance
sort_idx = np.argsort(rf_imp)[::-1]
y_pos = np.arange(8)

ax.barh(y_pos - 0.15, rf_imp[sort_idx], height=0.3, label='RF importance', color='steelblue')
# Normalize KRR sensitivity for comparison
krr_norm = krr_sens / krr_sens.max() * rf_imp.max()
ax.barh(y_pos + 0.15, krr_norm[sort_idx], height=0.3, label='KRR sensitivity (normalized)', color='coral')
ax.set_yticks(y_pos)
ax.set_yticklabels([FEATURE_NAMES[i] for i in sort_idx])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance Comparison')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'interp_feature_importance.png'), dpi=150)
plt.close()
print("Plot 4: interp_feature_importance.png saved")
print(f"  RF importance order: {[FEATURE_NAMES[i] for i in sort_idx]}")

# ---- Plot 5: Exponential decay fit ----
fig, ax = plt.subplots(figsize=(8, 6))

# Fit log10(J) = log10(A) - beta * dz * log10(e)
def decay_func(dz, log10_A, beta):
    return log10_A - beta * dz * np.log10(np.e)

# Use GPR dz profile
popt, pcov = curve_fit(decay_func, dz_profile_x, dz_profile_y, p0=[0, 1.5])
log10_A, beta = popt
A = 10 ** log10_A

ax.plot(dz_profile_x, dz_profile_y, 'b-', linewidth=2, label='GPR prediction')
ax.fill_between(dz_profile_x, dz_profile_y - 2 * dz_profile_std,
                dz_profile_y + 2 * dz_profile_std, alpha=0.2, color='blue')
ax.plot(dz_profile_x, decay_func(dz_profile_x, *popt), 'r--', linewidth=2,
        label=f'Fit: A={A:.3f} eV, β={beta:.2f} Å⁻¹')
ax.set_xlabel('dz (Å)')
ax.set_ylabel('log10(J / eV)')
ax.set_title('Exponential Decay of Coupling with Distance')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'interp_exponential_decay.png'), dpi=150)
plt.close()

print(f"\nPlot 5: interp_exponential_decay.png saved")
print(f"  Exponential decay fit: A = {A:.4f} eV, β = {beta:.4f} Å⁻¹")
if beta < 0.5 or beta > 5.0:
    print(f"  ⚠ WARNING: β = {beta:.2f} is outside expected range [0.5, 5.0] Å⁻¹!")
else:
    print(f"  β is within expected range [0.5, 5.0] Å⁻¹ ✓")

print("\n✓ All 5 interpretation plots saved to results/figures/")

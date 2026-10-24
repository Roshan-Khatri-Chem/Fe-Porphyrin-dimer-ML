"""
Phase 6: Evaluation

Load all Stage 2 models. Run full pipeline on test set.
Compute metrics and generate plots.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

FEATURE_NAMES = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi', 'cos_4psi', 'sin_4psi']

# Load data
features = np.load(os.path.join(DATA_DIR, 'features.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
labels_log = np.load(os.path.join(DATA_DIR, 'labels_log.npy'))
metadata = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

# Load models
scaler_s2 = joblib.load(os.path.join(MODEL_DIR, 'scaler_stage2.pkl'))
krr_model = joblib.load(os.path.join(MODEL_DIR, 'stage2_krr.pkl'))
gpr_model = joblib.load(os.path.join(MODEL_DIR, 'stage2_gpr.pkl'))
rf_model = joblib.load(os.path.join(MODEL_DIR, 'stage2_rf.pkl'))

# Filter to significant coupling
sig_mask = labels_ev >= 1e-4
X_all = features[sig_mask]
y_log_all = labels_log[sig_mask]
y_ev_all = labels_ev[sig_mask]
meta_sig = metadata[sig_mask].reset_index(drop=True)

test_mask = meta_sig['split'] == 'test'
X_test = X_all[test_mask.values]
y_log_test = y_log_all[test_mask.values]
y_ev_test = y_ev_all[test_mask.values]

X_test_s = scaler_s2.transform(X_test)

print(f"Test set: {len(X_test)} structures (coupling >= 1e-4 eV)")

# Predictions for all three models
models = {
    'KRR': ('pipeline', krr_model),
    'GPR': ('gpr', gpr_model),
    'RF': ('pipeline', rf_model),
}

all_metrics = {}

for name, (model_type, model) in models.items():
    if model_type == 'pipeline':
        y_pred_log = model.predict(X_test)
    else:  # GPR uses pre-scaled data
        y_pred_log, y_pred_std = model.predict(X_test_s, return_std=True)

    y_pred_ev = 10 ** y_pred_log

    # Log-space metrics
    log_mae = mean_absolute_error(y_log_test, y_pred_log)
    log_rmse = np.sqrt(mean_squared_error(y_log_test, y_pred_log))
    log_r2 = r2_score(y_log_test, y_pred_log)

    # Linear-space metrics
    lin_mae = mean_absolute_error(y_ev_test, y_pred_ev)
    lin_rmse = np.sqrt(mean_squared_error(y_ev_test, y_pred_ev))
    lin_r2 = r2_score(y_ev_test, y_pred_ev)

    all_metrics[name] = {
        'log_mae': float(log_mae),
        'log_rmse': float(log_rmse),
        'log_r2': float(log_r2),
        'linear_mae': float(lin_mae),
        'linear_rmse': float(lin_rmse),
        'linear_r2': float(lin_r2),
    }

    print(f"\n--- {name} ---")
    print(f"  Log MAE:  {log_mae:.4f}")
    print(f"  Log RMSE: {log_rmse:.4f}")
    print(f"  Log R²:   {log_r2:.4f}")
    print(f"  Lin MAE:  {lin_mae:.6f} eV")
    print(f"  Lin RMSE: {lin_rmse:.6f} eV")
    print(f"  Lin R²:   {lin_r2:.4f}")

    # Stratified by coupling decade
    decades = np.floor(y_log_test).astype(int)
    dec_bins = [(-4, -3), (-3, -2), (-2, -1), (-1, 0)]
    print(f"\n  Stratified log MAE:")
    for lo, hi in dec_bins:
        mask = (decades >= lo) & (decades < hi)
        if mask.sum() > 0:
            d_mae = mean_absolute_error(y_log_test[mask], y_pred_log[mask])
            print(f"    [{10**lo:.0e}, {10**hi:.0e}) eV: log MAE={d_mae:.4f}, N={mask.sum()}")

# Save metrics
with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("\nSaved: results/metrics.json")

# ---- Determine best model for plots ----
best_name = min(all_metrics, key=lambda k: all_metrics[k]['log_mae'])
print(f"\nBest model for plots: {best_name}")

if best_name == 'GPR':
    y_pred_log_best, y_pred_std_best = gpr_model.predict(X_test_s, return_std=True)
elif best_name == 'KRR':
    y_pred_log_best = krr_model.predict(X_test)
    y_pred_std_best = None
else:
    y_pred_log_best = rf_model.predict(X_test)
    y_pred_std_best = None

# ---- Plot 1: Parity plot (log scale) ----
fig, ax = plt.subplots(figsize=(8, 8))
decades_test = np.floor(y_log_test).astype(int)
scatter = ax.scatter(y_log_test, y_pred_log_best, c=decades_test, cmap='viridis',
                     alpha=0.6, s=30, edgecolors='k', linewidth=0.3)
lims = [min(y_log_test.min(), y_pred_log_best.min()) - 0.3,
        max(y_log_test.max(), y_pred_log_best.max()) + 0.3]
ax.plot(lims, lims, 'k-', linewidth=1.5, label='1:1')
ax.plot(lims, [l + 0.5 for l in lims], 'k--', alpha=0.5, linewidth=0.8, label='±0.5 log')
ax.plot(lims, [l - 0.5 for l in lims], 'k--', alpha=0.5, linewidth=0.8)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Actual log10(J / eV)')
ax.set_ylabel('Predicted log10(J / eV)')
log_mae_best = all_metrics[best_name]['log_mae']
log_r2_best = all_metrics[best_name]['log_r2']
ax.set_title(f'Test set ({best_name}): log MAE = {log_mae_best:.2f}, R² = {log_r2_best:.2f}')
ax.legend()
plt.colorbar(scatter, label='Coupling decade')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eval_parity_log.png'), dpi=150)
plt.close()
print("Plot 1: eval_parity_log.png saved")

# ---- Plot 2: Residuals vs features ----
residuals = y_pred_log_best - y_log_test
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    x_feat = X_test[:, i]
    ax.scatter(x_feat, residuals, alpha=0.4, s=10)
    ax.axhline(0, color='k', linewidth=0.8)
    # Simple moving average as lowess substitute
    sort_idx = np.argsort(x_feat)
    x_sorted = x_feat[sort_idx]
    r_sorted = residuals[sort_idx]
    window = max(len(r_sorted) // 10, 5)
    r_smooth = np.convolve(r_sorted, np.ones(window)/window, mode='valid')
    x_smooth = np.convolve(x_sorted, np.ones(window)/window, mode='valid')
    ax.plot(x_smooth, r_smooth, 'r-', linewidth=2)
    ax.set_xlabel(FEATURE_NAMES[i])
    ax.set_ylabel('Residual (log10)')
    ax.set_title(FEATURE_NAMES[i])
fig.suptitle(f'Residuals vs Features ({best_name})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eval_residuals.png'), dpi=150)
plt.close()
print("Plot 2: eval_residuals.png saved")

# ---- Plot 3: GPR uncertainty (if GPR available) ----
gpr_pred, gpr_std = gpr_model.predict(X_test_s, return_std=True)
gpr_abs_err = np.abs(gpr_pred - y_log_test)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(gpr_std, gpr_abs_err, alpha=0.5, s=20)
ax.set_xlabel('GPR posterior std')
ax.set_ylabel('|log10(pred) - log10(true)|')
ax.set_title('GPR Uncertainty Calibration')
# Add correlation
from scipy.stats import pearsonr
r_unc, _ = pearsonr(gpr_std, gpr_abs_err)
ax.annotate(f'Pearson r = {r_unc:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, va='top')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eval_gpr_uncertainty.png'), dpi=150)
plt.close()
print("Plot 3: eval_gpr_uncertainty.png saved")

# ---- Comparison table ----
print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)
print(f"{'Dataset':<10} {'Model':<12} {'Test log MAE':>14} {'Test lin MAE':>14} {'Test R² (lin)':>14}")
print("-" * 66)
print(f"{'597':<10} {'Student':<12} {'—':>14} {'0.00690':>14} {'0.863':>14}")
for name in ['KRR', 'GPR', 'RF']:
    m = all_metrics[name]
    print(f"{'597':<10} {name:<12} {m['log_mae']:>14.4f} {m['linear_mae']:>14.5f} {m['linear_r2']:>14.4f}")

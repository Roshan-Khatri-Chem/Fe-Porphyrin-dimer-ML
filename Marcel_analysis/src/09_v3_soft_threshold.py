"""
v3: Replace binary Stage 1 with physics-informed soft threshold.
Options A, B, C + Boltzmann weighting.
"""

import sys, os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (mean_absolute_error, r2_score,
                             confusion_matrix, recall_score, precision_score)
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'coupling.csv')

FEATURE_NAMES = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi',
                 'cos_4psi', 'sin_4psi', 'd_lateral', 'cos_4alpha', 'sin_4alpha']

# Load data
features = np.load(os.path.join(DATA_DIR, 'features_v2.npy'))  # (704, 11)
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
labels_log = np.load(os.path.join(DATA_DIR, 'labels_log.npy'))
metadata = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

train_mask = (metadata['split'] == 'train').values
test_mask = (metadata['split'] == 'test').values
is_near_zero = labels_ev < 1e-4

X_train, X_test = features[train_mask], features[test_mask]
y_ev_train, y_ev_test = labels_ev[train_mask], labels_ev[test_mask]
y_log_train, y_log_test = labels_log[train_mask], labels_log[test_mask]
nz_train, nz_test = is_near_zero[train_mask], is_near_zero[test_mask]

# Load energy data
df_coup = pd.read_csv(CSV_PATH)
energy_dict = {}
for _, row in df_coup.iterrows():
    stem = row['Filename'].replace('.out', '')
    energy_dict[stem] = row['Total_Energy_Ha']

energies_ha = np.array([energy_dict[fn] for fn in metadata['filename']])
HA_TO_EV = 27.211386245988
energies_ev = energies_ha * HA_TO_EV
E_rel_ev = (energies_ha - energies_ha.min()) * HA_TO_EV  # relative energy in eV

# Significant-only test mask (for apples-to-apples comparison)
sig_test = y_ev_test >= 1e-4
y_log_test_sig = np.log10(y_ev_test[sig_test])

RF_PARAMS = {
    'rf__n_estimators': [200, 500],
    'rf__max_features': ['sqrt', 0.5],
    'rf__min_samples_leaf': [1, 2, 3],
    'rf__max_depth': [None, 10, 20],
}

DEC_BINS = [(-4, -3), (-3, -2), (-2, -1), (-1, 0)]


def stratified_mae(y_true, y_pred):
    """Return dict of log MAE by decade."""
    decades = np.floor(y_true).astype(int)
    result = {}
    for lo, hi in DEC_BINS:
        m = (decades >= lo) & (decades < hi)
        if m.sum() > 0:
            result[f'[1e{lo},1e{hi})'] = {
                'log_mae': float(mean_absolute_error(y_true[m], y_pred[m])),
                'N': int(m.sum())
            }
    return result


def train_rf(X_tr, y_tr, sample_weight=None):
    """Train RF with RandomizedSearchCV, return fitted pipeline."""
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('rf', RandomForestRegressor(random_state=42))])
    search = RandomizedSearchCV(pipe, RF_PARAMS, n_iter=30, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1,
                                random_state=42, verbose=0)
    fit_params = {}
    if sample_weight is not None:
        fit_params['rf__sample_weight'] = sample_weight
    search.fit(X_tr, y_tr, **fit_params)
    return search.best_estimator_


# ============================================================
# BASELINE: v2 two-stage RF (11-feat) — reproduce for reference
# ============================================================
print("=" * 70)
print("BASELINE: Two-stage RF (11-feat, Stage 2 on >= 1e-4)")
print("=" * 70)

sig_train = ~nz_train
X_train_sig = X_train[sig_train]
y_train_sig = y_log_train[sig_train]

baseline_rf = train_rf(X_train_sig, y_train_sig)
baseline_pred_sig = baseline_rf.predict(X_test[sig_test])

baseline_mae = mean_absolute_error(y_log_test_sig, baseline_pred_sig)
baseline_r2 = r2_score(y_log_test_sig, baseline_pred_sig)
baseline_strat = stratified_mae(y_log_test_sig, baseline_pred_sig)
print(f"  Overall log MAE = {baseline_mae:.4f}, R² = {baseline_r2:.4f}")
for k, v in baseline_strat.items():
    print(f"    {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

# ============================================================
# OPTION A: Soft gate on cos_4phi (1D logistic)
# ============================================================
print("\n" + "=" * 70)
print("OPTION A: Soft gate on cos_4phi (1D logistic)")
print("=" * 70)

cos4phi_train = X_train[:, 4].reshape(-1, 1)
cos4phi_test = X_test[:, 4].reshape(-1, 1)

logreg = LogisticRegression(random_state=42)
logreg.fit(cos4phi_train, nz_train.astype(int))

# Predictions
nz_pred_logreg = logreg.predict(cos4phi_test)
nz_proba_logreg = logreg.predict_proba(cos4phi_test)[:, 1]

recall_logreg = recall_score(nz_test.astype(int), nz_pred_logreg)
precision_logreg = precision_score(nz_test.astype(int), nz_pred_logreg, zero_division=0)
cm_logreg = confusion_matrix(nz_test.astype(int), nz_pred_logreg)

# Compare to SVC from original
stage1_svc = joblib.load(os.path.join(MODEL_DIR, 'stage1_classifier.pkl'))
nz_pred_svc = stage1_svc.predict(X_test[:, :8])  # SVC trained on 8 features
recall_svc = recall_score(nz_test.astype(int), nz_pred_svc)

print(f"  Logistic (cos_4phi only):")
print(f"    Recall (near-zero): {recall_logreg:.3f} ({int(recall_logreg * nz_test.sum())}/{nz_test.sum()})")
print(f"    Precision (near-zero): {precision_logreg:.3f}")
print(f"    Confusion matrix:\n      {cm_logreg}")
print(f"  SVC (11 features, original):")
print(f"    Recall (near-zero): {recall_svc:.3f} ({int(recall_svc * nz_test.sum())}/{nz_test.sum()})")

# Option A pipeline: logistic gate -> Stage 2 RF
# Structures predicted significant go to Stage 2
optA_sig_pred = nz_pred_logreg == 0  # predicted as significant
# For significant test structures, predict with Stage 2 RF
optA_pred_log = np.full(len(X_test), np.nan)
if optA_sig_pred.any():
    optA_pred_log[optA_sig_pred] = baseline_rf.predict(X_test[optA_sig_pred])
# Near-zero predictions get log10(0) which is -inf; for comparison, use -6
optA_pred_log[~optA_sig_pred] = -6.0

# Evaluate on significant test structures only
optA_pred_sig = optA_pred_log[sig_test]
# Some significant structures may have been gated out (false positives of near-zero)
gated_out = np.isnan(optA_pred_sig) | (optA_pred_sig <= -5.5)
optA_n_gated = gated_out.sum()
print(f"\n  Option A pipeline: {optA_n_gated} significant test structures incorrectly gated as near-zero")

# For fair comparison, evaluate only structures NOT gated
optA_kept = sig_test.copy()
optA_kept_idx = np.where(sig_test)[0]
for i, idx in enumerate(optA_kept_idx):
    if nz_pred_logreg[idx]:
        optA_kept[idx] = False

optA_kept_test = optA_kept
if optA_kept_test.sum() > 0:
    y_true_A = np.log10(y_ev_test[optA_kept_test])
    y_pred_A = baseline_rf.predict(X_test[optA_kept_test])
    optA_mae_kept = mean_absolute_error(y_true_A, y_pred_A)
    print(f"  Log MAE on non-gated significant: {optA_mae_kept:.4f} (N={optA_kept_test.sum()})")

# Full evaluation including gated structures (they get wrong predictions)
optA_pred_full = np.where(nz_pred_logreg, -6.0, baseline_rf.predict(X_test))
optA_pred_full_sig = optA_pred_full[sig_test]
optA_mae_full = mean_absolute_error(y_log_test_sig, optA_pred_full_sig)
optA_r2_full = r2_score(y_log_test_sig, optA_pred_full_sig)
optA_strat = stratified_mae(y_log_test_sig, optA_pred_full_sig)
print(f"  Overall log MAE (full, with gated errors) = {optA_mae_full:.4f}, R² = {optA_r2_full:.4f}")
for k, v in optA_strat.items():
    print(f"    {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

# Plot logistic fit
fig, ax = plt.subplots(figsize=(8, 5))
cos4phi_range = np.linspace(-1, 1, 200).reshape(-1, 1)
proba_range = logreg.predict_proba(cos4phi_range)[:, 1]
ax.plot(cos4phi_range, proba_range, 'b-', linewidth=2, label='P(near-zero)')
ax.scatter(cos4phi_test[nz_test, 0], np.ones(nz_test.sum()) * 1.05, c='red',
           s=30, alpha=0.5, label='Near-zero (actual)')
ax.scatter(cos4phi_test[~nz_test, 0], np.zeros((~nz_test).sum()) - 0.05, c='steelblue',
           s=10, alpha=0.3, label='Significant (actual)')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('cos_4phi')
ax.set_ylabel('P(near-zero)')
ax.set_title('Option A: Logistic gate on cos_4phi')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'v3_optA_logistic_gate.png'), dpi=150)
plt.close()
print("  Saved: v3_optA_logistic_gate.png")

# ============================================================
# OPTION B: Eliminate Stage 1, physics-informed labels
# ============================================================
print("\n" + "=" * 70)
print("OPTION B: Single-model RF with physics-informed labels")
print("=" * 70)

# Build labels
y_optB = np.zeros(len(labels_ev))
for i in range(len(labels_ev)):
    if labels_ev[i] >= 1e-4:
        y_optB[i] = np.log10(labels_ev[i])
    else:
        cos4phi_i = features[i, 4]
        y_optB[i] = np.log10(1e-4) - 2.0 * (1.0 + cos4phi_i)

y_optB_train = y_optB[train_mask]
y_optB_test = y_optB[test_mask]

print(f"  Label range: [{y_optB.min():.2f}, {y_optB.max():.2f}]")
print(f"  Near-zero label range: [{y_optB[is_near_zero].min():.2f}, {y_optB[is_near_zero].max():.2f}]")
print(f"  Significant label range: [{y_optB[~is_near_zero].min():.2f}, {y_optB[~is_near_zero].max():.2f}]")

# Train RF on all 704
optB_rf = train_rf(X_train, y_optB_train)
optB_pred = optB_rf.predict(X_test)

# Evaluate on significant test structures
optB_pred_sig = optB_pred[sig_test]
optB_mae = mean_absolute_error(y_log_test_sig, optB_pred_sig)
optB_r2 = r2_score(y_log_test_sig, optB_pred_sig)
optB_strat = stratified_mae(y_log_test_sig, optB_pred_sig)
print(f"  Overall log MAE = {optB_mae:.4f}, R² = {optB_r2:.4f}")
for k, v in optB_strat.items():
    print(f"    {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

joblib.dump(optB_rf, os.path.join(MODEL_DIR, 'optB_rf.pkl'))

# Plot label distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y_optB[~is_near_zero], bins=30, alpha=0.6, label='Significant', color='steelblue')
ax.hist(y_optB[is_near_zero], bins=15, alpha=0.6, label='Near-zero (physics label)', color='red')
ax.set_xlabel('Label value')
ax.set_ylabel('Count')
ax.set_title('Option B: Physics-informed label distribution')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'v3_optB_label_dist.png'), dpi=150)
plt.close()
print("  Saved: v3_optB_label_dist.png")

# ============================================================
# OPTION C: GMM soft weight
# ============================================================
print("\n" + "=" * 70)
print("OPTION C: GMM soft weight on [cos_4phi, d_lateral]")
print("=" * 70)

gmm_features_all = features[:, [4, 8]]  # cos_4phi, d_lateral
gmm_train = gmm_features_all[train_mask]
gmm_test = gmm_features_all[test_mask]

gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full')
gmm.fit(gmm_train)

# Identify which component is "near-zero"
proba_train = gmm.predict_proba(gmm_train)
# The near-zero component should have higher proportion of near-zero labels
nz_rate_0 = nz_train[gmm.predict(gmm_train) == 0].mean()
nz_rate_1 = nz_train[gmm.predict(gmm_train) == 1].mean()
nz_component = 0 if nz_rate_0 > nz_rate_1 else 1
print(f"  Component 0 near-zero rate: {nz_rate_0:.3f}")
print(f"  Component 1 near-zero rate: {nz_rate_1:.3f}")
print(f"  Near-zero component: {nz_component}")

# Posterior P(near-zero) for each sample
P_nz_train = gmm.predict_proba(gmm_train)[:, nz_component]
P_nz_test = gmm.predict_proba(gmm_test)[:, nz_component]

# Use log10(coupling) for ALL structures (with floor for zero-coupling)
y_all_log = np.where(labels_ev > 0, np.log10(labels_ev), -7.0)
y_all_log_train = y_all_log[train_mask]
y_all_log_test = y_all_log[test_mask]

# Sample weights: 1 - P_near_zero (downweight near-zero structures)
weights_train = 1.0 - P_nz_train

# Train RF with sample weights
optC_rf = train_rf(X_train, y_all_log_train, sample_weight=weights_train)
optC_pred = optC_rf.predict(X_test)

optC_pred_sig = optC_pred[sig_test]
optC_mae = mean_absolute_error(y_log_test_sig, optC_pred_sig)
optC_r2 = r2_score(y_log_test_sig, optC_pred_sig)
optC_strat = stratified_mae(y_log_test_sig, optC_pred_sig)
print(f"  Overall log MAE = {optC_mae:.4f}, R² = {optC_r2:.4f}")
for k, v in optC_strat.items():
    print(f"    {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

# Plot GMM
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(gmm_features_all[:, 0], gmm_features_all[:, 1],
                     c=is_near_zero, cmap='coolwarm', alpha=0.4, s=15)
ax.set_xlabel('cos_4phi')
ax.set_ylabel('d_lateral (Å)')
ax.set_title('Option C: GMM components (red=near-zero, blue=significant)')
# Plot GMM contours
from matplotlib.patches import Ellipse
for k in range(2):
    mean = gmm.means_[k]
    cov = gmm.covariances_[k]
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for n_std in [1, 2]:
        ell = Ellipse(mean, 2 * n_std * np.sqrt(vals[0]), 2 * n_std * np.sqrt(vals[1]),
                      angle=angle, fill=False, edgecolor='green' if k == nz_component else 'black',
                      linewidth=2, linestyle='--' if n_std == 2 else '-')
        ax.add_patch(ell)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'v3_optC_gmm.png'), dpi=150)
plt.close()
print("  Saved: v3_optC_gmm.png")

# ============================================================
# BOLTZMANN WEIGHTING
# ============================================================
print("\n" + "=" * 70)
print("BOLTZMANN WEIGHTING")
print("=" * 70)

E_rel_train = E_rel_ev[train_mask]
E_rel_test = E_rel_ev[test_mask]

kT_300K = 0.026  # eV at 300K
boltz_weights = np.exp(-E_rel_train / kT_300K)
# Normalize to sum = N_train
boltz_weights *= len(boltz_weights) / boltz_weights.sum()

print(f"  E_rel range: {E_rel_ev.min():.4f} to {E_rel_ev.max():.4f} eV")
print(f"  Boltzmann weight range: {boltz_weights.min():.4e} to {boltz_weights.max():.4e}")
print(f"  N weights > 0.01: {(boltz_weights > 0.01).sum()} / {len(boltz_weights)}")
print(f"  Effective N (sum w²/N): {boltz_weights.sum()**2 / (boltz_weights**2).sum():.1f}")

# E_rel vs log10(coupling)
valid_mask_all = labels_ev > 0
r_E_coupling, p_E_coupling = pearsonr(E_rel_ev[valid_mask_all],
                                       np.log10(labels_ev[valid_mask_all]))
print(f"\n  Pearson r(E_rel, log10(J)): {r_E_coupling:.4f} (p={p_E_coupling:.2e})")

# Are high-energy structures the near-zero ones?
nz_mean_E = E_rel_ev[is_near_zero].mean()
sig_mean_E = E_rel_ev[~is_near_zero].mean()
print(f"  Mean E_rel (near-zero): {nz_mean_E:.4f} eV")
print(f"  Mean E_rel (significant): {sig_mean_E:.4f} eV")

# Plot E_rel vs coupling
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
valid = labels_ev > 0
ax.scatter(E_rel_ev[valid & ~is_near_zero], np.log10(labels_ev[valid & ~is_near_zero]),
           alpha=0.3, s=10, color='steelblue', label='Significant')
ax.scatter(E_rel_ev[valid & is_near_zero], np.log10(labels_ev[valid & is_near_zero]),
           alpha=0.5, s=20, color='red', label='Near-zero')
ax.set_xlabel('E_rel (eV)')
ax.set_ylabel('log10(J / eV)')
ax.set_title(f'E_rel vs Coupling (r = {r_E_coupling:.3f})')
ax.legend()

ax = axes[1]
ax.hist(E_rel_ev[~is_near_zero], bins=40, alpha=0.5, label='Significant', density=True, color='steelblue')
ax.hist(E_rel_ev[is_near_zero], bins=20, alpha=0.5, label='Near-zero', density=True, color='red')
ax.set_xlabel('E_rel (eV)')
ax.set_ylabel('Density')
ax.set_title('Energy Distribution by Class')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'v3_boltzmann_energy.png'), dpi=150)
plt.close()
print("  Saved: v3_boltzmann_energy.png")

# --- Boltzmann-weighted Stage 2 RF (on >= 1e-4 only, like baseline) ---
print("\n  Training Boltzmann-weighted Stage 2 RF (significant only)...")
sig_train_mask = ~nz_train
boltz_sig = boltz_weights[sig_train_mask]
boltz_sig *= len(boltz_sig) / boltz_sig.sum()  # renormalize

boltz_rf = train_rf(X_train[sig_train_mask], y_log_train[sig_train_mask],
                     sample_weight=boltz_sig)
boltz_pred_sig = boltz_rf.predict(X_test[sig_test])

boltz_mae = mean_absolute_error(y_log_test_sig, boltz_pred_sig)
boltz_r2 = r2_score(y_log_test_sig, boltz_pred_sig)
boltz_strat = stratified_mae(y_log_test_sig, boltz_pred_sig)
print(f"  Overall log MAE = {boltz_mae:.4f}, R² = {boltz_r2:.4f}")
for k, v in boltz_strat.items():
    print(f"    {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

# --- Boltzmann + Option B (physics labels on all 704) ---
print("\n  Training Boltzmann-weighted Option B RF (all 704)...")
boltz_all = np.exp(-E_rel_train / kT_300K)
boltz_all *= len(boltz_all) / boltz_all.sum()

boltzB_rf = train_rf(X_train, y_optB_train, sample_weight=boltz_all)
boltzB_pred = boltzB_rf.predict(X_test)
boltzB_pred_sig = boltzB_pred[sig_test]
boltzB_mae = mean_absolute_error(y_log_test_sig, boltzB_pred_sig)
boltzB_r2 = r2_score(y_log_test_sig, boltzB_pred_sig)
boltzB_strat = stratified_mae(y_log_test_sig, boltzB_pred_sig)
print(f"  Overall log MAE = {boltzB_mae:.4f}, R² = {boltzB_r2:.4f}")
for k, v in boltzB_strat.items():
    print(f"    {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
print("\n" + "=" * 70)
print("FULL COMPARISON TABLE (evaluated on significant test structures)")
print("=" * 70)

all_results = {
    'Baseline (two-stage RF 11f)':    (baseline_mae, baseline_r2, baseline_strat),
    'Option A (logistic gate)':       (optA_mae_full, optA_r2_full, optA_strat),
    'Option B (physics labels)':      (optB_mae, optB_r2, optB_strat),
    'Option C (GMM soft weight)':     (optC_mae, optC_r2, optC_strat),
    'Boltzmann Stage2':               (boltz_mae, boltz_r2, boltz_strat),
    'Boltzmann + Option B':           (boltzB_mae, boltzB_r2, boltzB_strat),
}

header = f"{'Approach':<35} {'log MAE':>8} {'R²':>7}"
for lo, hi in DEC_BINS:
    header += f" {'[1e'+str(lo)+',1e'+str(hi)+')':>14}"
print(header)
print("-" * len(header))

for name, (mae, r2, strat) in all_results.items():
    row = f"{name:<35} {mae:>8.4f} {r2:>7.3f}"
    for lo, hi in DEC_BINS:
        key = f'[1e{lo},1e{hi})'
        if key in strat:
            row += f" {strat[key]['log_mae']:>14.4f}"
        else:
            row += f" {'—':>14}"
    print(row)

# Save all results
save_results = {}
for name, (mae, r2, strat) in all_results.items():
    save_results[name] = {
        'log_mae': float(mae),
        'log_r2': float(r2),
        'stratified': strat,
    }
save_results['boltzmann_info'] = {
    'pearson_r_Erel_logJ': float(r_E_coupling),
    'mean_Erel_near_zero_eV': float(nz_mean_E),
    'mean_Erel_significant_eV': float(sig_mean_E),
    'effective_N_boltzmann': float(boltz_weights.sum()**2 / (boltz_weights**2).sum()),
}
save_results['optA_classifier'] = {
    'logistic_recall_near_zero': float(recall_logreg),
    'logistic_precision_near_zero': float(precision_logreg),
    'svc_recall_near_zero': float(recall_svc),
}
with open(os.path.join(RESULTS_DIR, 'metrics_v3.json'), 'w') as f:
    json.dump(save_results, f, indent=2)
print(f"\nSaved: results/metrics_v3.json")

# ============================================================
# PARITY PLOTS: best approaches side by side
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

plot_data = [
    ('Baseline\n(two-stage RF)', y_log_test_sig, baseline_pred_sig),
    ('Option B\n(physics labels)', y_log_test_sig, optB_pred_sig),
    ('Boltzmann + Option B', y_log_test_sig, boltzB_pred_sig),
]

for ax, (title, yt, yp) in zip(axes, plot_data):
    decades = np.floor(yt).astype(int)
    sc = ax.scatter(yt, yp, c=decades, cmap='viridis', alpha=0.6, s=30,
                    edgecolors='k', linewidth=0.3)
    lims = [min(yt.min(), yp.min()) - 0.3, max(yt.max(), yp.max()) + 0.3]
    ax.plot(lims, lims, 'k-', linewidth=1.5)
    ax.plot(lims, [l + 0.5 for l in lims], 'k--', alpha=0.4, linewidth=0.8)
    ax.plot(lims, [l - 0.5 for l in lims], 'k--', alpha=0.4, linewidth=0.8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual log10(J / eV)')
    ax.set_ylabel('Predicted log10(J / eV)')
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    ax.set_title(f'{title}\nlog MAE={mae:.3f}, R²={r2:.3f}')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'v3_parity_comparison.png'), dpi=150)
plt.close()
print("Saved: v3_parity_comparison.png")

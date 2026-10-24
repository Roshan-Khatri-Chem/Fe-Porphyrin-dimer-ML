"""
v5: Factorized coupling model.

Instead of predicting log10(J) directly, predict the ratio J / |S_HL|.
The Hückel overlap captures the orbital symmetry modulation (distance decay
+ D4h twist node), so the residual ratio should be smoother and easier to
learn.

J_pred = 10^(log_ratio_pred) * |S_HL|
"""

import sys, os, glob, json
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import geometry
from geometry import load_dimer_xyz, huckel_overlap, dimer_features

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
STRUCT_DIR = os.path.join(DATA_DIR, 'structures')
COUPLING_CSV = os.path.join(DATA_DIR, 'coupling.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

ALPHA_OPT = 0.10  # fitted in v4

GEOM_FEAT_NAMES = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi',
                    'cos_4psi', 'sin_4psi', 'd_lateral', 'cos_4alpha', 'sin_4alpha']

DEC_BINS = [(-4, -3), (-3, -2), (-2, -1), (-1, 0)]

RF_PARAMS = {
    'rf__n_estimators': [200, 500],
    'rf__max_features': ['sqrt', 0.5],
    'rf__min_samples_leaf': [1, 2, 3],
    'rf__max_depth': [None, 10, 20],
}


def stratified_mae(y_true, y_pred):
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


# ============================================================
# STEP 1: Compute |S_HL| for all structures
# ============================================================
print("=" * 70)
print("STEP 1: Compute |S_HL| for all structures (alpha = 0.10)")
print("=" * 70)

df_coup = pd.read_csv(COUPLING_CSV)
coupling_dict = {}
for _, row in df_coup.iterrows():
    stem = row['Filename'].replace('.out', '')
    coupling_dict[stem] = row['Coupling_eV']

xyz_files = sorted(glob.glob(os.path.join(STRUCT_DIR, '*.xyz')))

dimers = []
labels_ev_list = []
filenames_list = []
s_hl_list = []

for filepath in tqdm(xyz_files, desc="Loading dimers + S_HL"):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    if stem not in coupling_dict:
        continue
    d = load_dimer_xyz(filepath)
    dimers.append(d)
    labels_ev_list.append(coupling_dict[stem])
    filenames_list.append(stem)
    s = huckel_overlap(
        d['monomer1']['positions'], d['monomer1']['symbols'],
        d['monomer2']['positions'], d['monomer2']['symbols'],
        alpha=ALPHA_OPT)
    s_hl_list.append(s)

labels_ev = np.array(labels_ev_list, dtype=np.float64)
labels_log = np.where(labels_ev > 0, np.log10(labels_ev), np.nan)
s_hl = np.array(s_hl_list, dtype=np.float64)
s_hl_abs = np.abs(s_hl)
N = len(labels_ev)

np.save(os.path.join(PROCESSED_DIR, 's_hl.npy'), s_hl)
print(f"Saved s_hl.npy: {N} values")
print(f"|S_HL| range: [{s_hl_abs.min():.6e}, {s_hl_abs.max():.6e}]")
print(f"|S_HL| == 0: {(s_hl_abs == 0).sum()} structures")
print(f"|S_HL| < 1e-6: {(s_hl_abs < 1e-6).sum()} structures")

# Verify D4h node behavior
near_zero_mask = labels_ev < 1e-4
print(f"\nNear-zero structures: {near_zero_mask.sum()}")
print(f"  Mean |S_HL| (near-zero): {s_hl_abs[near_zero_mask].mean():.6e}")
print(f"  Mean |S_HL| (significant): {s_hl_abs[~near_zero_mask].mean():.6e}")

# Load 11-feature matrix (geometric features only)
features_11 = np.load(os.path.join(PROCESSED_DIR, 'features_v2.npy'))
meta = pd.read_csv(os.path.join(PROCESSED_DIR, 'metadata.csv'))

# Recreate splits
is_near_zero = labels_ev < 1e-4
strat_labels = np.zeros(N, dtype=int)
for i, val in enumerate(labels_ev):
    if val < 1e-4:
        strat_labels[i] = -99
    elif val > 0:
        strat_labels[i] = int(np.floor(np.log10(val)))
    else:
        strat_labels[i] = -99

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.zeros((N, 1)), strat_labels))
train_mask = np.zeros(N, dtype=bool)
train_mask[train_idx] = True
test_mask = ~train_mask

# ============================================================
# STEP 2: Compute target ratio
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Compute target ratio J / |S_HL|")
print("=" * 70)

EPS = 1e-6
usable = (s_hl_abs > EPS) & (labels_ev > 0)
print(f"Usable structures (|S_HL| > {EPS} and J > 0): {usable.sum()}")
print(f"Set aside (|S_HL| < {EPS}): {(s_hl_abs <= EPS).sum()}")

ratio = np.full(N, np.nan)
ratio[usable] = labels_ev[usable] / s_hl_abs[usable]
log_ratio = np.full(N, np.nan)
log_ratio[usable] = np.log10(ratio[usable])

print(f"log_ratio range: [{np.nanmin(log_ratio):.2f}, {np.nanmax(log_ratio):.2f}]")
print(f"log_ratio std: {np.nanstd(log_ratio):.3f}")
print(f"log10(J) std (usable): {np.nanstd(labels_log[usable]):.3f}")

# ============================================================
# STEP 3: Train Stage 2 RF on log_ratio (11 geometric features)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Train factorized RF on log_ratio")
print("=" * 70)

# Training: significant coupling AND usable S_HL
sig_mask = labels_ev >= 1e-4
train_fact = train_mask & sig_mask & usable
test_fact = test_mask & sig_mask & usable

# Also need test structures with |S_HL| < eps to check near-zero prediction
test_node = test_mask & sig_mask & (s_hl_abs <= EPS)

X_train_fact = features_11[train_fact]
y_train_fact = log_ratio[train_fact]
X_test_fact = features_11[test_fact]
y_test_fact_ratio = log_ratio[test_fact]
y_test_fact_ev = labels_ev[test_fact]
y_test_fact_log = labels_log[test_fact]
s_hl_test_fact = s_hl_abs[test_fact]

print(f"Factorized train: {train_fact.sum()}, test: {test_fact.sum()}")
print(f"Test structures at geometric node (|S_HL| < eps): {test_node.sum()}")

# Train factorized RF
pipe_fact = Pipeline([('scaler', StandardScaler()),
                      ('rf', RandomForestRegressor(random_state=42))])
search_fact = RandomizedSearchCV(pipe_fact, RF_PARAMS, n_iter=30, cv=5,
                                  scoring='neg_mean_absolute_error', n_jobs=-1,
                                  random_state=42, verbose=0)
search_fact.fit(X_train_fact, y_train_fact)
rf_fact = search_fact.best_estimator_
print(f"Best params: {search_fact.best_params_}")
print(f"CV MAE on log_ratio: {-search_fact.best_score_:.4f}")

# ============================================================
# STEP 4: Inference and evaluation
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Inference")
print("=" * 70)

# Predict log_ratio, then reconstruct J
log_ratio_pred = rf_fact.predict(X_test_fact)
J_pred_fact = (10 ** log_ratio_pred) * s_hl_test_fact
log_J_pred_fact = np.log10(np.maximum(J_pred_fact, 1e-12))

# Baseline: direct RF on log10(J) with 11 features
sig_train_base = train_mask & sig_mask
sig_test_base = test_mask & sig_mask

X_train_base = features_11[sig_train_base]
y_train_base = labels_log[sig_train_base]
X_test_base = features_11[sig_test_base]
y_test_base = labels_log[sig_test_base]

pipe_base = Pipeline([('scaler', StandardScaler()),
                      ('rf', RandomForestRegressor(random_state=42))])
search_base = RandomizedSearchCV(pipe_base, RF_PARAMS, n_iter=30, cv=5,
                                  scoring='neg_mean_absolute_error', n_jobs=-1,
                                  random_state=42, verbose=0)
search_base.fit(X_train_base, y_train_base)
rf_base = search_base.best_estimator_
log_J_pred_base = rf_base.predict(X_test_base)

# For fair comparison, evaluate both on the SAME test set (factorized test)
# The baseline can predict all sig_test structures; factorized only those with |S_HL| > eps
# First show factorized on its own test set
mae_fact = mean_absolute_error(y_test_fact_log, log_J_pred_fact)
r2_fact = r2_score(y_test_fact_log, log_J_pred_fact)
strat_fact = stratified_mae(y_test_fact_log, log_J_pred_fact)

# Baseline on same subset
log_J_pred_base_subset = rf_base.predict(X_test_fact)
mae_base_subset = mean_absolute_error(y_test_fact_log, log_J_pred_base_subset)
r2_base_subset = r2_score(y_test_fact_log, log_J_pred_base_subset)
strat_base_subset = stratified_mae(y_test_fact_log, log_J_pred_base_subset)

# Full baseline on all sig test
mae_base_full = mean_absolute_error(y_test_base, log_J_pred_base)
r2_base_full = r2_score(y_test_base, log_J_pred_base)
strat_base_full = stratified_mae(y_test_base, log_J_pred_base)

print(f"\n  Baseline (direct RF, 11-feat, all sig test N={sig_test_base.sum()}):")
print(f"    log MAE = {mae_base_full:.4f}, R² = {r2_base_full:.4f}")
for k, v in strat_base_full.items():
    print(f"      {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

print(f"\n  Baseline (same subset as factorized, N={test_fact.sum()}):")
print(f"    log MAE = {mae_base_subset:.4f}, R² = {r2_base_subset:.4f}")
for k, v in strat_base_subset.items():
    print(f"      {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

print(f"\n  Factorized (J = 10^log_ratio * |S_HL|, N={test_fact.sum()}):")
print(f"    log MAE = {mae_fact:.4f}, R² = {r2_fact:.4f}")
for k, v in strat_fact.items():
    print(f"      {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

# Check: structures at geometric node
if test_node.sum() > 0:
    print(f"\n  Structures at geometric node (|S_HL| < eps, N={test_node.sum()}):")
    print(f"    These get J_pred = 0 by construction (|S_HL| ≈ 0).")
    print(f"    Actual coupling: {labels_ev[test_node]}")
else:
    print(f"\n  No test structures at geometric node (|S_HL| < {EPS})")

# Check near-zero prediction via S_HL
# For ALL test structures (including near-zero), predict using factorized
test_all = test_mask
X_test_all = features_11[test_all]
s_hl_test_all = s_hl_abs[test_all]
y_ev_test_all = labels_ev[test_all]

log_ratio_pred_all = rf_fact.predict(X_test_all)
J_pred_all = (10 ** log_ratio_pred_all) * s_hl_test_all

# How many near-zero structures does the factorized model get right?
nz_test = (y_ev_test_all < 1e-4)
nz_pred_below_threshold = J_pred_all[nz_test] < 1e-4
print(f"\n  Near-zero detection via factorized model:")
print(f"    Near-zero test structures: {nz_test.sum()}")
print(f"    Predicted J < 1e-4 eV: {nz_pred_below_threshold.sum()} / {nz_test.sum()}")
print(f"    (Baseline Stage 1 SVC detected: 1 / {nz_test.sum()})")

# ============================================================
# STEP 5: Histogram comparison + distribution analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Distribution analysis")
print("=" * 70)

valid = usable & sig_mask
log_J_valid = labels_log[valid]
log_ratio_valid = log_ratio[valid]

print(f"  log10(J) — mean: {np.nanmean(log_J_valid):.3f}, std: {np.nanstd(log_J_valid):.3f}")
print(f"  log_ratio — mean: {np.nanmean(log_ratio_valid):.3f}, std: {np.nanstd(log_ratio_valid):.3f}")
print(f"  Variance reduction: {1 - np.nanvar(log_ratio_valid)/np.nanvar(log_J_valid):.1%}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(log_J_valid, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_xlabel(r'$\log_{10}(J\,/\,\mathrm{eV})$')
ax.set_ylabel('Count')
ax.set_title(f'Direct target (std = {np.nanstd(log_J_valid):.2f})')

ax = axes[1]
ax.hist(log_ratio_valid, bins=40, alpha=0.7, color='coral', edgecolor='black')
ax.set_xlabel(r'$\log_{10}(J\,/\,|S_\mathrm{HL}|)$')
ax.set_ylabel('Count')
ax.set_title(f'Factorized target (std = {np.nanstd(log_ratio_valid):.2f})')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'v5_target_distributions.png'), dpi=150)
plt.close()
print("  Saved: v5_target_distributions.png")

# ============================================================
# STEP 6: Sanity check — log_ratio vs geometric features
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Correlation of log_ratio vs geometric features")
print("=" * 70)

valid_train = usable & train_mask & sig_mask
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# Compare correlations for log10(J) and log_ratio

print(f"{'Feature':<14} {'r(log10 J)':>12} {'r(log_ratio)':>14} {'Reduction':>10}")
print("-" * 52)

for i, ax in enumerate(axes.flat[:8]):
    if i >= len(GEOM_FEAT_NAMES):
        ax.set_visible(False)
        continue

    x = features_11[valid_train, i]
    y_J = labels_log[valid_train]
    y_R = log_ratio[valid_train]
    valid_J = np.isfinite(y_J)
    valid_R = np.isfinite(y_R)

    r_J, _ = pearsonr(x[valid_J], y_J[valid_J])
    r_R, _ = pearsonr(x[valid_R], y_R[valid_R])

    reduction = (1 - abs(r_R) / max(abs(r_J), 1e-10)) * 100

    print(f"{GEOM_FEAT_NAMES[i]:<14} {r_J:>12.4f} {r_R:>14.4f} {reduction:>9.0f}%")

    ax.scatter(x[valid_R], y_R[valid_R], alpha=0.3, s=8, color='coral')
    ax.set_xlabel(GEOM_FEAT_NAMES[i])
    ax.set_ylabel('log_ratio')
    ax.set_title(f'{GEOM_FEAT_NAMES[i]}: r={r_R:.3f}')

# Fill remaining axes with the remaining features
fig.suptitle('log_ratio vs geometric features (training set)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'v5_ratio_vs_features.png'), dpi=150)
plt.close()
print("\n  Saved: v5_ratio_vs_features.png")

# Parity plot comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, (title, yt, yp) in zip(axes, [
    ('Baseline (direct RF)', y_test_fact_log, log_J_pred_base_subset),
    ('Factorized (ratio RF)', y_test_fact_log, log_J_pred_fact),
]):
    decades = np.floor(yt).astype(int)
    sc = ax.scatter(yt, yp, c=decades, cmap='viridis', alpha=0.6, s=30,
                    edgecolors='k', linewidth=0.3)
    lims = [min(yt.min(), yp.min()) - 0.3, max(yt.max(), yp.max()) + 0.3]
    ax.plot(lims, lims, 'k-', linewidth=1.5)
    ax.plot(lims, [l + 0.5 for l in lims], 'k--', alpha=0.4, linewidth=0.8)
    ax.plot(lims, [l - 0.5 for l in lims], 'k--', alpha=0.4, linewidth=0.8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r'Actual $\log_{10}(J\,/\,\mathrm{eV})$')
    ax.set_ylabel(r'Predicted $\log_{10}(J\,/\,\mathrm{eV})$')
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    ax.set_title(f'{title}\nlog MAE={mae:.3f}, R²={r2:.3f}')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'v5_parity_comparison.png'), dpi=150)
plt.close()
print("  Saved: v5_parity_comparison.png")

# ============================================================
# Final comparison table
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON TABLE (same test subset, N={})".format(test_fact.sum()))
print("=" * 70)

header = f"{'Model':<30} {'log MAE':>8} {'R²':>7}"
for lo, hi in DEC_BINS:
    header += f" {'[1e'+str(lo)+',1e'+str(hi)+')':>14}"
print(header)
print("-" * len(header))

for name, mae, r2, strat in [
    ("Baseline (direct RF 11f)", mae_base_subset, r2_base_subset, strat_base_subset),
    ("Factorized (ratio RF 11f)", mae_fact, r2_fact, strat_fact),
]:
    row = f"{name:<30} {mae:>8.4f} {r2:>7.3f}"
    for lo, hi in DEC_BINS:
        key = f'[1e{lo},1e{hi})'
        if key in strat:
            row += f" {strat[key]['log_mae']:>14.4f}"
        else:
            row += f" {'—':>14}"
    print(row)

# Save results
results = {
    'baseline_direct': {'log_mae': float(mae_base_subset), 'log_r2': float(r2_base_subset),
                        'stratified': strat_base_subset},
    'factorized': {'log_mae': float(mae_fact), 'log_r2': float(r2_fact),
                   'stratified': strat_fact},
    'target_stats': {
        'log10_J_std': float(np.nanstd(log_J_valid)),
        'log_ratio_std': float(np.nanstd(log_ratio_valid)),
        'variance_reduction_pct': float((1 - np.nanvar(log_ratio_valid)/np.nanvar(log_J_valid)) * 100),
    },
    'near_zero_detection': {
        'n_nz_test': int(nz_test.sum()),
        'n_detected': int(nz_pred_below_threshold.sum()),
    },
}
with open(os.path.join(RESULTS_DIR, 'metrics_v5_factorized.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: results/metrics_v5_factorized.json")

joblib.dump(rf_fact, os.path.join(MODEL_DIR, 'stage2_rf_factorized.pkl'))
print("Saved: models/stage2_rf_factorized.pkl")

"""
Re-run pipeline with 11-feature vector.
Then train single-model approach on all 704 structures with log10(J + 1e-6).
Compare to original two-stage approach.
"""

import sys, os, glob, json
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from geometry import features_from_xyz

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
STRUCT_DIR = os.path.join(DATA_DIR, 'structures')
COUPLING_CSV = os.path.join(DATA_DIR, 'coupling.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

FEATURE_NAMES_11 = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi',
                     'cos_4psi', 'sin_4psi', 'd_lateral', 'cos_4alpha', 'sin_4alpha']

# ============================================================
# STEP 1: Re-featurize with 11 features
# ============================================================
print("=" * 70)
print("STEP 1: Re-featurize with 11-feature vector")
print("=" * 70)

df_coup = pd.read_csv(COUPLING_CSV)
coupling_dict = {}
for _, row in df_coup.iterrows():
    stem = row['Filename'].replace('.out', '')
    coupling_dict[stem] = row['Coupling_eV']

xyz_files = sorted(glob.glob(os.path.join(STRUCT_DIR, '*.xyz')))

features_list = []
labels_ev_list = []
filenames_list = []

for filepath in tqdm(xyz_files, desc="Featurizing (11-feat)"):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    if stem not in coupling_dict:
        continue
    feat = features_from_xyz(filepath)
    features_list.append(feat)
    labels_ev_list.append(coupling_dict[stem])
    filenames_list.append(stem)

features = np.array(features_list, dtype=np.float64)
labels_ev = np.array(labels_ev_list, dtype=np.float64)
labels_log = np.where(labels_ev > 0, np.log10(labels_ev), np.nan)

print(f"Feature matrix shape: {features.shape}")
assert features.shape == (704, 11)
assert np.sum(~np.isfinite(features)) == 0

# Save v2 arrays
np.save(os.path.join(PROCESSED_DIR, 'features_v2.npy'), features)
np.save(os.path.join(PROCESSED_DIR, 'labels_ev.npy'), labels_ev)
np.save(os.path.join(PROCESSED_DIR, 'labels_log.npy'), labels_log)

# Re-create stratified split (same seed)
is_near_zero = labels_ev < 1e-4
strat_labels = np.zeros(len(labels_ev), dtype=int)
for i, val in enumerate(labels_ev):
    if val < 1e-4:
        strat_labels[i] = -99
    elif val > 0:
        strat_labels[i] = int(np.floor(np.log10(val)))
    else:
        strat_labels[i] = -99

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(features, strat_labels))
split = np.array(['train'] * len(labels_ev))
split[test_idx] = 'test'

metadata = pd.DataFrame({
    'idx': range(len(labels_ev)),
    'filename': filenames_list,
    'coupling_eV': labels_ev,
    'log10_coupling': labels_log,
    'is_near_zero': is_near_zero,
    'split': split,
})
metadata.to_csv(os.path.join(PROCESSED_DIR, 'metadata.csv'), index=False)

# ============================================================
# STEP 2: Two-stage pipeline with 11 features (Stage 2 only on >= 1e-4)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Two-stage pipeline (11-feature, Stage 2 RF on >= 1e-4)")
print("=" * 70)

sig_mask = labels_ev >= 1e-4
X_sig = features[sig_mask]
y_sig_log = labels_log[sig_mask]
y_sig_ev = labels_ev[sig_mask]
meta_sig = metadata[sig_mask].reset_index(drop=True)

train_sig = meta_sig['split'] == 'train'
test_sig = meta_sig['split'] == 'test'

X_train_sig = X_sig[train_sig.values]
y_train_sig = y_sig_log[train_sig.values]
X_test_sig = X_sig[test_sig.values]
y_test_sig = y_sig_log[test_sig.values]
y_test_sig_ev = y_sig_ev[test_sig.values]

print(f"Significant: {len(X_sig)} total, {len(X_train_sig)} train, {len(X_test_sig)} test")

# Train RF (best model from Phase 5)
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42)),
])
rf_params = {
    'rf__n_estimators': [200, 500],
    'rf__max_features': ['sqrt', 0.5],
    'rf__min_samples_leaf': [1, 2, 3],
    'rf__max_depth': [None, 10, 20],
}
rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=30, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1,
                                random_state=42, verbose=0)
rf_search.fit(X_train_sig, y_train_sig)
y_pred_twostage = rf_search.predict(X_test_sig)

twostage_log_mae = mean_absolute_error(y_test_sig, y_pred_twostage)
twostage_log_r2 = r2_score(y_test_sig, y_pred_twostage)
print(f"Two-stage RF (11-feat): log MAE = {twostage_log_mae:.4f}, log R² = {twostage_log_r2:.4f}")

# Stratified by decade
decades_test = np.floor(y_test_sig).astype(int)
dec_bins = [(-4, -3), (-3, -2), (-2, -1), (-1, 0)]
print("Stratified log MAE (two-stage, 11-feat):")
twostage_strat = {}
for lo, hi in dec_bins:
    mask = (decades_test >= lo) & (decades_test < hi)
    if mask.sum() > 0:
        d_mae = mean_absolute_error(y_test_sig[mask], y_pred_twostage[mask])
        twostage_strat[f"[1e{lo}, 1e{hi})"] = {'log_mae': float(d_mae), 'N': int(mask.sum())}
        print(f"  [{10**lo:.0e}, {10**hi:.0e}) eV: log MAE={d_mae:.4f}, N={mask.sum()}")

joblib.dump(rf_search.best_estimator_, os.path.join(MODEL_DIR, 'stage2_rf_v2.pkl'))

# Also train KRR and GPR with 11 features
print("\nTraining KRR (11-feat)...")
krr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('krr', KernelRidge(kernel='rbf')),
])
krr_params = {
    'krr__alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0],
    'krr__gamma': [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
}
krr_grid = GridSearchCV(krr_pipe, krr_params, cv=5, scoring='neg_mean_absolute_error',
                        n_jobs=-1, verbose=0)
krr_grid.fit(X_train_sig, y_train_sig)
y_pred_krr = krr_grid.predict(X_test_sig)
krr_log_mae = mean_absolute_error(y_test_sig, y_pred_krr)
print(f"  KRR (11-feat): log MAE = {krr_log_mae:.4f}")
joblib.dump(krr_grid.best_estimator_, os.path.join(MODEL_DIR, 'stage2_krr_v2.pkl'))

print("Training GPR (11-feat)...")
scaler_v2 = StandardScaler()
X_train_s = scaler_v2.fit_transform(X_train_sig)
X_test_s = scaler_v2.transform(X_test_sig)
kernel_matern = C(1.0) * Matern(length_scale=np.ones(11), nu=2.5) + WhiteKernel(1e-3)
gpr = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=10,
                                normalize_y=True, random_state=42)
gpr.fit(X_train_s, y_train_sig)
y_pred_gpr, y_std_gpr = gpr.predict(X_test_s, return_std=True)
gpr_log_mae = mean_absolute_error(y_test_sig, y_pred_gpr)
print(f"  GPR (11-feat): log MAE = {gpr_log_mae:.4f}")
joblib.dump(gpr, os.path.join(MODEL_DIR, 'stage2_gpr_v2.pkl'))
joblib.dump(scaler_v2, os.path.join(MODEL_DIR, 'scaler_stage2_v2.pkl'))

# ============================================================
# STEP 3: Single-model approach on ALL 704 structures
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Single-model approach (all 704, label = log10(J + 1e-6))")
print("=" * 70)

# Labels with softplus shift
labels_shifted = np.log10(labels_ev + 1e-6)

train_all = metadata['split'] == 'train'
test_all = metadata['split'] == 'test'

X_train_all = features[train_all.values]
y_train_all = labels_shifted[train_all.values]
X_test_all = features[test_all.values]
y_test_all = labels_shifted[test_all.values]
y_test_ev_all = labels_ev[test_all.values]

print(f"All structures: {len(X_train_all)} train, {len(X_test_all)} test")

# Train RF on all
print("\nTraining RF (single-model, all 704)...")
rf_all_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42)),
])
rf_all_search = RandomizedSearchCV(rf_all_pipe, rf_params, n_iter=30, cv=5,
                                    scoring='neg_mean_absolute_error', n_jobs=-1,
                                    random_state=42, verbose=0)
rf_all_search.fit(X_train_all, y_train_all)
y_pred_all_shifted = rf_all_search.predict(X_test_all)

# Convert back: J_pred = 10^pred - 1e-6, clipped at 0
y_pred_all_ev = np.maximum(10 ** y_pred_all_shifted - 1e-6, 0.0)

# For log-space comparison on significant test structures only
sig_test_mask = y_test_ev_all >= 1e-4
y_pred_all_log_sig = np.log10(np.maximum(y_pred_all_ev[sig_test_mask], 1e-12))
y_true_log_sig = np.log10(y_test_ev_all[sig_test_mask])

single_log_mae_sig = mean_absolute_error(y_true_log_sig, y_pred_all_log_sig)
single_log_r2_sig = r2_score(y_true_log_sig, y_pred_all_log_sig)
print(f"Single RF (on significant test): log MAE = {single_log_mae_sig:.4f}, log R² = {single_log_r2_sig:.4f}")

# Overall on shifted labels
single_overall_mae = mean_absolute_error(y_test_all, y_pred_all_shifted)
print(f"Single RF (all test, shifted labels): MAE = {single_overall_mae:.4f}")

# Stratified for single model on significant structures
decades_single = np.floor(y_true_log_sig).astype(int)
print("Stratified log MAE (single RF, 11-feat, significant test):")
single_strat = {}
for lo, hi in dec_bins:
    mask = (decades_single >= lo) & (decades_single < hi)
    if mask.sum() > 0:
        d_mae = mean_absolute_error(y_true_log_sig[mask], y_pred_all_log_sig[mask])
        single_strat[f"[1e{lo}, 1e{hi})"] = {'log_mae': float(d_mae), 'N': int(mask.sum())}
        print(f"  [{10**lo:.0e}, {10**hi:.0e}) eV: log MAE={d_mae:.4f}, N={mask.sum()}")

# Train GPR on all
print("\nTraining GPR (single-model, all 704)...")
scaler_all = StandardScaler()
X_train_all_s = scaler_all.fit_transform(X_train_all)
X_test_all_s = scaler_all.transform(X_test_all)
kernel_all = C(1.0) * Matern(length_scale=np.ones(11), nu=2.5) + WhiteKernel(1e-3)
gpr_all = GaussianProcessRegressor(kernel=kernel_all, n_restarts_optimizer=10,
                                    normalize_y=True, random_state=42)
gpr_all.fit(X_train_all_s, y_train_all)
y_pred_gpr_all_shifted = gpr_all.predict(X_test_all_s)

y_pred_gpr_all_ev = np.maximum(10 ** y_pred_gpr_all_shifted - 1e-6, 0.0)
y_pred_gpr_log_sig = np.log10(np.maximum(y_pred_gpr_all_ev[sig_test_mask], 1e-12))
gpr_single_mae_sig = mean_absolute_error(y_true_log_sig, y_pred_gpr_log_sig)
print(f"Single GPR (on significant test): log MAE = {gpr_single_mae_sig:.4f}")

print("Stratified log MAE (single GPR, 11-feat, significant test):")
gpr_single_strat = {}
for lo, hi in dec_bins:
    mask = (decades_single >= lo) & (decades_single < hi)
    if mask.sum() > 0:
        d_mae = mean_absolute_error(y_true_log_sig[mask], y_pred_gpr_log_sig[mask])
        gpr_single_strat[f"[1e{lo}, 1e{hi})"] = {'log_mae': float(d_mae), 'N': int(mask.sum())}
        print(f"  [{10**lo:.0e}, {10**hi:.0e}) eV: log MAE={d_mae:.4f}, N={mask.sum()}")

joblib.dump(rf_all_search.best_estimator_, os.path.join(MODEL_DIR, 'single_rf_all.pkl'))
joblib.dump(gpr_all, os.path.join(MODEL_DIR, 'single_gpr_all.pkl'))
joblib.dump(scaler_all, os.path.join(MODEL_DIR, 'scaler_single_all.pkl'))

# ============================================================
# STEP 4: Comparison table
# ============================================================
print("\n" + "=" * 70)
print("FULL COMPARISON TABLE")
print("=" * 70)

# Load original 8-feat results
orig_metrics = json.load(open(os.path.join(RESULTS_DIR, 'metrics.json')))

print(f"\n{'Approach':<40} {'log MAE':>10} {'log R²':>10} {'[1e-4,1e-3)':>14}")
print("-" * 76)

# Original 8-feat two-stage
print(f"{'Two-stage RF (8-feat, original)':<40} {orig_metrics['RF']['log_mae']:>10.4f} {orig_metrics['RF']['log_r2']:>10.4f} {'0.5249':>14}")

# 11-feat two-stage
ts_decade_1e4 = twostage_strat.get('[1e-4, 1e-3)', {}).get('log_mae', float('nan'))
print(f"{'Two-stage RF (11-feat)':<40} {twostage_log_mae:>10.4f} {twostage_log_r2:>10.4f} {ts_decade_1e4:>14.4f}")

# 11-feat two-stage KRR
print(f"{'Two-stage KRR (11-feat)':<40} {krr_log_mae:>10.4f} {'':>10} {'':>14}")

# 11-feat two-stage GPR
print(f"{'Two-stage GPR (11-feat)':<40} {gpr_log_mae:>10.4f} {'':>10} {'':>14}")

# Single RF
single_decade_1e4 = single_strat.get('[1e-4, 1e-3)', {}).get('log_mae', float('nan'))
print(f"{'Single RF (11-feat, all 704)':<40} {single_log_mae_sig:>10.4f} {single_log_r2_sig:>10.4f} {single_decade_1e4:>14.4f}")

# Single GPR
gpr_single_decade_1e4 = gpr_single_strat.get('[1e-4, 1e-3)', {}).get('log_mae', float('nan'))
print(f"{'Single GPR (11-feat, all 704)':<40} {gpr_single_mae_sig:>10.4f} {'':>10} {gpr_single_decade_1e4:>14.4f}")

# ============================================================
# STEP 5: Parity plot for best model
# ============================================================
best_approaches = {
    'Two-stage RF 11-feat': (y_test_sig, y_pred_twostage),
    'Single RF 11-feat': (y_true_log_sig, y_pred_all_log_sig),
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, (name, (y_true, y_pred)) in zip(axes, best_approaches.items()):
    decades = np.floor(y_true).astype(int)
    sc = ax.scatter(y_true, y_pred, c=decades, cmap='viridis', alpha=0.6, s=30,
                    edgecolors='k', linewidth=0.3)
    lims = [min(y_true.min(), y_pred.min()) - 0.3,
            max(y_true.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, 'k-', linewidth=1.5)
    ax.plot(lims, [l + 0.5 for l in lims], 'k--', alpha=0.5, linewidth=0.8)
    ax.plot(lims, [l - 0.5 for l in lims], 'k--', alpha=0.5, linewidth=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual log10(J / eV)')
    ax.set_ylabel('Predicted log10(J / eV)')
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f'{name}\nlog MAE={mae:.3f}, R²={r2:.3f}')
    plt.colorbar(sc, ax=ax, label='Decade')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eval_parity_v2_comparison.png'), dpi=150)
plt.close()
print("\nSaved: eval_parity_v2_comparison.png")

# Save updated metrics
v2_metrics = {
    'two_stage_rf_11feat': {
        'log_mae': float(twostage_log_mae),
        'log_r2': float(twostage_log_r2),
        'stratified': twostage_strat,
    },
    'two_stage_krr_11feat': {'log_mae': float(krr_log_mae)},
    'two_stage_gpr_11feat': {'log_mae': float(gpr_log_mae)},
    'single_rf_11feat_all704': {
        'log_mae': float(single_log_mae_sig),
        'log_r2': float(single_log_r2_sig),
        'stratified': single_strat,
    },
    'single_gpr_11feat_all704': {
        'log_mae': float(gpr_single_mae_sig),
        'stratified': gpr_single_strat,
    },
}
with open(os.path.join(RESULTS_DIR, 'metrics_v2.json'), 'w') as f:
    json.dump(v2_metrics, f, indent=2)
print("Saved: results/metrics_v2.json")

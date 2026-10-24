"""
Fit Hückel overlap decay parameter alpha, refeaturize with 13 features,
retrain Stage 2 RF, and compare stratified log-MAE.
"""

import sys, os, glob, json
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from geometry import load_dimer_xyz, get_alpha_carbons, huckel_overlap, dimer_features

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
STRUCT_DIR = os.path.join(DATA_DIR, 'structures')
COUPLING_CSV = os.path.join(DATA_DIR, 'coupling.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

FEATURE_NAMES_13 = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi',
                     'cos_4psi', 'sin_4psi', 'd_lateral', 'cos_4alpha', 'sin_4alpha',
                     '|S_HL|', 'S_HL^2']

DEC_BINS = [(-4, -3), (-3, -2), (-2, -1), (-1, 0)]

RF_PARAMS = {
    'rf__n_estimators': [200, 500],
    'rf__max_features': ['sqrt', 0.5],
    'rf__min_samples_leaf': [1, 2, 3],
    'rf__max_depth': [None, 10, 20],
}

# ============================================================
# STEP 1: Load all dimers and compute S_HL at varying alpha
# ============================================================
print("=" * 70)
print("STEP 1: Load dimers and fit alpha")
print("=" * 70)

df_coup = pd.read_csv(COUPLING_CSV)
coupling_dict = {}
for _, row in df_coup.iterrows():
    stem = row['Filename'].replace('.out', '')
    coupling_dict[stem] = row['Coupling_eV']

xyz_files = sorted(glob.glob(os.path.join(STRUCT_DIR, '*.xyz')))

# Pre-load all dimer data
dimers = []
labels_ev_list = []
filenames_list = []
for filepath in tqdm(xyz_files, desc="Loading dimers"):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    if stem not in coupling_dict:
        continue
    d = load_dimer_xyz(filepath)
    dimers.append(d)
    labels_ev_list.append(coupling_dict[stem])
    filenames_list.append(stem)

labels_ev = np.array(labels_ev_list, dtype=np.float64)
labels_log = np.where(labels_ev > 0, np.log10(labels_ev), np.nan)
N = len(labels_ev)
print(f"Loaded {N} dimers")

# Recreate train/test split (same as before)
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
# Need a dummy X for the split
dummy_X = np.zeros((N, 1))
train_idx, test_idx = next(sss.split(dummy_X, strat_labels))
train_mask = np.zeros(N, dtype=bool)
train_mask[train_idx] = True
test_mask = ~train_mask

# Filter to significant coupling for alpha fitting
sig_mask = labels_ev >= 1e-4
sig_train = sig_mask & train_mask
valid_train = (labels_ev > 0) & train_mask  # for correlation


# ============================================================
# STEP 2: Fit alpha by maximizing |pearsonr(|S_HL|, log10(J))|
# ============================================================
print("\nFitting alpha...")

def compute_S_HL_all(alpha):
    """Compute |S_HL| for all dimers at a given alpha."""
    S_vals = np.zeros(N)
    for i, d in enumerate(dimers):
        S_vals[i] = abs(huckel_overlap(
            d['monomer1']['positions'], d['monomer1']['symbols'],
            d['monomer2']['positions'], d['monomer2']['symbols'],
            alpha=alpha))
    return S_vals

def neg_pearson_r(alpha):
    """Negative Pearson r between |S_HL| and log10(J) on significant training set."""
    S_vals = compute_S_HL_all(alpha)
    mask = valid_train
    S_train = S_vals[mask]
    y_train = labels_log[mask]
    valid = np.isfinite(y_train)
    r, _ = pearsonr(S_train[valid], y_train[valid])
    return -abs(r)  # minimize negative |r|

# Grid search then refine
print("  Grid search over alpha in [0.1, 3.0]...")
alphas = np.linspace(0.1, 3.0, 30)
rs = []
for a in alphas:
    r_val = -neg_pearson_r(a)
    rs.append(r_val)
    if r_val == max(rs):
        print(f"    alpha={a:.2f}: |r| = {r_val:.4f} *")

best_grid_alpha = alphas[np.argmax(rs)]
print(f"  Best grid alpha = {best_grid_alpha:.2f}, |r| = {max(rs):.4f}")

# Refine with scipy
result = minimize_scalar(neg_pearson_r, bounds=(0.1, 3.0), method='bounded')
alpha_opt = result.x
r_opt = -result.fun
print(f"  Optimized alpha = {alpha_opt:.4f}, |r| = {r_opt:.4f}")

# ============================================================
# STEP 3: Refeaturize with optimal alpha (13 features)
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 3: Refeaturize with alpha = {alpha_opt:.4f}")
print("=" * 70)

# Monkey-patch the default alpha in huckel_overlap for dimer_features
import geometry
_original_huckel = geometry.huckel_overlap

def huckel_overlap_fitted(pos1, sym1, pos2, sym2, alpha=alpha_opt):
    return _original_huckel(pos1, sym1, pos2, sym2, alpha=alpha)

geometry.huckel_overlap = huckel_overlap_fitted

features_list = []
for d in tqdm(dimers, desc="Featurizing (13-feat)"):
    feat = dimer_features(
        d['monomer1']['positions'], d['monomer1']['symbols'],
        d['monomer2']['positions'], d['monomer2']['symbols'])
    features_list.append(feat)

features = np.array(features_list, dtype=np.float64)
print(f"Feature matrix shape: {features.shape}")
assert features.shape == (704, 13)
assert np.sum(~np.isfinite(features)) == 0

# Save
np.save(os.path.join(PROCESSED_DIR, 'features_v3.npy'), features)

# Pearson r between |S_HL| and log10(J) on training set
S_HL_train = features[valid_train, 11]
y_log_train = labels_log[valid_train]
valid = np.isfinite(y_log_train)
r_train, p_train = pearsonr(S_HL_train[valid], y_log_train[valid])
print(f"\n(b) Pearson r(|S_HL|, log10(J)) on training set: {r_train:.4f} (p={p_train:.2e})")

# Also check S_HL^2
r_sq, _ = pearsonr(features[valid_train, 12][valid], y_log_train[valid])
print(f"    Pearson r(S_HL^2, log10(J)) on training set: {r_sq:.4f}")

# ============================================================
# STEP 4: Train Stage 2 RF with 13 features vs 11 features
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Train and compare RF models")
print("=" * 70)

# Load 11-feature baseline for comparison
features_11 = np.load(os.path.join(PROCESSED_DIR, 'features_v2.npy'))

X_train_sig_13 = features[sig_train, :]
X_train_sig_11 = features_11[sig_train, :]
y_train_sig = labels_log[sig_train]

sig_test = sig_mask & test_mask
X_test_sig_13 = features[sig_test, :]
X_test_sig_11 = features_11[sig_test, :]
y_test_sig = labels_log[sig_test]

print(f"Training: {sig_train.sum()} structures, Test: {sig_test.sum()} structures")

def train_and_eval(X_tr, y_tr, X_te, y_te, label):
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('rf', RandomForestRegressor(random_state=42))])
    search = RandomizedSearchCV(pipe, RF_PARAMS, n_iter=30, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1,
                                random_state=42, verbose=0)
    search.fit(X_tr, y_tr)
    y_pred = search.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)

    decades = np.floor(y_te).astype(int)
    strat = {}
    for lo, hi in DEC_BINS:
        m = (decades >= lo) & (decades < hi)
        if m.sum() > 0:
            strat[f'[1e{lo},1e{hi})'] = {
                'log_mae': float(mean_absolute_error(y_te[m], y_pred[m])),
                'N': int(m.sum())
            }

    print(f"\n  {label}:")
    print(f"    Overall log MAE = {mae:.4f}, R² = {r2:.4f}")
    for k, v in strat.items():
        print(f"      {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

    return search.best_estimator_, mae, r2, strat, y_pred

# 11-feature baseline
rf_11, mae_11, r2_11, strat_11, pred_11 = train_and_eval(
    X_train_sig_11, y_train_sig, X_test_sig_11, y_test_sig, "RF 11-feat (baseline)")

# 13-feature with S_HL
rf_13, mae_13, r2_13, strat_13, pred_13 = train_and_eval(
    X_train_sig_13, y_train_sig, X_test_sig_13, y_test_sig, "RF 13-feat (+|S_HL|, S_HL²)")

# Save 13-feat model
joblib.dump(rf_13, os.path.join(MODEL_DIR, 'stage2_rf_v3.pkl'))

# Feature importances
imp = rf_13.named_steps['rf'].feature_importances_
print("\n  RF 13-feat feature importances:")
for name, importance in sorted(zip(FEATURE_NAMES_13, imp), key=lambda x: -x[1]):
    print(f"    {name:<14}: {importance:.4f}")

# ============================================================
# STEP 5: Summary comparison table
# ============================================================
print("\n" + "=" * 70)
print("(c) COMPARISON TABLE")
print("=" * 70)

header = f"{'Model':<25} {'log MAE':>8} {'R²':>7}"
for lo, hi in DEC_BINS:
    header += f" {'[1e'+str(lo)+',1e'+str(hi)+')':>14}"
print(header)
print("-" * len(header))

for name, mae, r2, strat in [
    ("RF 11-feat (baseline)", mae_11, r2_11, strat_11),
    ("RF 13-feat (+Hückel)", mae_13, r2_13, strat_13),
]:
    row = f"{name:<25} {mae:>8.4f} {r2:>7.3f}"
    for lo, hi in DEC_BINS:
        key = f'[1e{lo},1e{hi})'
        if key in strat:
            row += f" {strat[key]['log_mae']:>14.4f}"
        else:
            row += f" {'—':>14}"
    print(row)

# Improvement in the target decade
key_1e4 = '[1e-4,1e-3)'
if key_1e4 in strat_11 and key_1e4 in strat_13:
    old = strat_11[key_1e4]['log_mae']
    new = strat_13[key_1e4]['log_mae']
    pct = (old - new) / old * 100
    print(f"\n[1e-4, 1e-3) improvement: {old:.4f} → {new:.4f} ({pct:+.1f}%)")

# Save results
results = {
    'alpha_opt': float(alpha_opt),
    'pearson_r_abs_SHL_logJ_train': float(r_train),
    'pearson_r_SHL2_logJ_train': float(r_sq),
    'rf_11feat': {'log_mae': float(mae_11), 'log_r2': float(r2_11), 'stratified': strat_11},
    'rf_13feat': {'log_mae': float(mae_13), 'log_r2': float(r2_13), 'stratified': strat_13},
    'feature_importances_13': dict(zip(FEATURE_NAMES_13, imp.tolist())),
}
with open(os.path.join(RESULTS_DIR, 'metrics_v3_huckel.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: results/metrics_v3_huckel.json")

# ============================================================
# Plot: |S_HL| vs log10(J) with fitted alpha
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
valid_all = labels_ev > 0
ax.scatter(features[valid_all & ~is_near_zero, 11],
           labels_log[valid_all & ~is_near_zero],
           alpha=0.4, s=15, c='steelblue', label='Significant')
ax.scatter(features[valid_all & is_near_zero, 11],
           labels_log[valid_all & is_near_zero],
           alpha=0.6, s=25, c='red', marker='x', label='Near-zero')
ax.set_xlabel(r'$|S_\mathrm{HL}|$')
ax.set_ylabel(r'$\log_{10}(J\,/\,\mathrm{eV})$')
ax.set_title(f'Hückel overlap vs coupling (α = {alpha_opt:.3f} Å⁻², r = {r_train:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'huckel_overlap_vs_coupling.png'), dpi=150)
plt.close()
print("Saved: huckel_overlap_vs_coupling.png")

print(f"\n(a) Fitted alpha = {alpha_opt:.4f} Å⁻²")
print(f"(b) Pearson r(|S_HL|, log10(J)) = {r_train:.4f}")

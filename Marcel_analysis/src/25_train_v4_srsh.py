"""Train v4_srsh: 14-feature RF (geometry + SRSH cation alpha SOMO S_12) on
815 = 704 systematic + 111 Slater MD structures, then evaluate on test/MD."""
import os, sys, json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
ROOT = os.path.join(THIS, '..')
DATA = os.path.join(ROOT, 'data', 'processed')
MODELS = os.path.join(ROOT, 'models')
RESULTS = os.path.join(ROOT, 'results')

RF_PARAMS = dict(n_estimators=500, max_features=0.5, max_depth=20,
                 min_samples_leaf=1, random_state=42)


def make_v4_features(features_v2, s12):
    """Append [s12, s12^2, log|s12|] to the 11-feature v2 vector."""
    s12 = np.asarray(s12)
    return np.column_stack([features_v2, s12, s12**2, np.log10(np.abs(s12) + 1e-12)])


def align_s12_to_meta(s12_data, meta_names):
    """Align s12_704 (709) to metadata names (704)."""
    name_to_s12 = dict(zip(s12_data['names_704'], s12_data['s12_704']))
    return np.array([name_to_s12.get(n, np.nan) for n in meta_names])


# ----------------------------------------------------------------------------
print("=" * 70)
print("Loading data")
print("=" * 70)
meta = pd.read_csv(os.path.join(DATA, 'metadata.csv'))
features_v2 = np.load(os.path.join(DATA, 'features_v2.npy'))    # (704, 11)
labels_log  = np.load(os.path.join(DATA, 'labels_log.npy'))     # (704,)
labels_ev   = np.load(os.path.join(DATA, 'labels_ev.npy'))      # (704,)

features_slater = np.load(os.path.join(DATA, 'features_slater.npy'))  # (111, 11)
labels_log_slater = np.load(os.path.join(DATA, 'labels_log_slater.npy'))
labels_ev_slater  = np.load(os.path.join(DATA, 'labels_ev_slater.npy'))

# SRSH S_12 — use the cation alpha SOMO ansatz (best correlation with J on Slater MD)
s12_srsh = np.load(os.path.join(DATA, 's12_srsh_cation.npz'), allow_pickle=True)

s12_704 = align_s12_to_meta(s12_srsh, meta['filename'].str.replace('.out','', regex=False).str.replace('.xyz','', regex=False).values)
s12_slater = s12_srsh['s12_slater']
s12_md     = s12_srsh['s12_md']
md_names   = s12_srsh['md_names']
print(f"  704 systematic: {(~np.isnan(s12_704)).sum()}/{len(s12_704)} valid SRSH S_12")
print(f"  111 Slater   : {(~np.isnan(s12_slater)).sum()}/{len(s12_slater)} valid SRSH S_12")
print(f"  15 MD        : {(~np.isnan(s12_md)).sum()}/{len(s12_md)} valid SRSH S_12")

# Build augmented training arrays (704 + 111)
X_704 = make_v4_features(features_v2, s12_704)
X_slater = make_v4_features(features_slater, s12_slater)
md_feat_path = os.path.join(DATA, 'features_md.npy')
if not os.path.exists(md_feat_path):
    print("  Computing MD geometry features on the fly")
    from geometry import dimer_features, load_dimer_xyz
    import glob
    md_files = sorted(glob.glob(os.path.join(ROOT, 'data', 'md_extracted', '*.xyz')))
    feats_md = []
    for fn in md_files:
        d = load_dimer_xyz(fn)
        feats_md.append(dimer_features(d['monomer1']['positions'], d['monomer1']['symbols'],
                                       d['monomer2']['positions'], d['monomer2']['symbols'])[:11])
    feats_md = np.array(feats_md)
    np.save(md_feat_path, feats_md)
X_md = make_v4_features(np.load(md_feat_path), s12_md)

# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Define splits")
print("=" * 70)
train_mask = (meta['split'] == 'train').values if 'split' in meta.columns else np.ones(len(meta), bool)
test_mask  = (meta['split'] == 'test').values if 'split' in meta.columns else np.zeros(len(meta), bool)
sig_mask   = labels_ev >= 1e-4

valid_704 = np.isfinite(s12_704) & sig_mask
valid_slater = np.isfinite(s12_slater) & (labels_ev_slater >= 1e-4)

# If no split column, use all 704 as train
if not test_mask.any():
    print("  No test split column found — using a fixed 80/20 split on 704")
    rng = np.random.default_rng(42)
    idx = np.where(sig_mask)[0]
    rng.shuffle(idx)
    n_test = len(idx) // 5
    test_mask = np.zeros(len(meta), bool); test_mask[idx[:n_test]] = True
    train_mask = np.zeros(len(meta), bool); train_mask[idx[n_test:]] = True

print(f"  704 train: {(train_mask & valid_704).sum()}, "
      f"test: {(test_mask & valid_704).sum()}")
print(f"  111 Slater (all train for v4_srsh_aug): {valid_slater.sum()}")

# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Train and evaluate models")
print("=" * 70)

results = {}

def train_eval(name, X_train, y_train, X_test, y_test, X_extra=None, y_extra=None,
               X_md_=None, y_md_=None):
    pipe = Pipeline([('s', StandardScaler()),
                     ('rf', RandomForestRegressor(**RF_PARAMS))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred)
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    mae_cv = -cv.mean()
    r_test, _ = pearsonr(y_pred, y_test) if len(y_test) > 2 else (np.nan, np.nan)

    extras = {}
    if X_extra is not None:
        y_extra_pred = pipe.predict(X_extra)
        extras['mae_extra'] = mean_absolute_error(y_extra, y_extra_pred)
        if len(y_extra) > 2:
            r_e, _ = pearsonr(y_extra_pred, y_extra)
            extras['r_extra'] = r_e
    if X_md_ is not None and (~np.isnan(y_md_)).any():
        m = ~np.isnan(y_md_) & (y_md_ > -np.inf)
        if m.sum() > 2:
            y_md_pred = pipe.predict(X_md_[m])
            extras['mae_md'] = mean_absolute_error(y_md_[m], y_md_pred)
            r_md, _ = pearsonr(y_md_pred, y_md_[m])
            extras['r_md']   = r_md

    print(f"\n  {name}")
    print(f"    CV log MAE  : {mae_cv:.4f}")
    print(f"    Test log MAE: {mae_test:.4f}  (Pearson r={r_test:.4f})")
    for k, v in extras.items():
        print(f"    {k:12}: {v:.4f}")
    return pipe, mae_cv, mae_test, extras


# v2 baseline (geometry only) on 704 train
print("\n--- v2 baseline (geometry only, 704 train) ---")
pipe_v2, cv_v2, test_v2, ex_v2 = train_eval(
    'v2 baseline',
    features_v2[train_mask & valid_704], labels_log[train_mask & valid_704],
    features_v2[test_mask  & valid_704], labels_log[test_mask  & valid_704],
    None, None,
    X_md[:, :11], np.full(len(X_md), np.nan))  # no MD labels here
results['v2'] = {'cv': cv_v2, 'test': test_v2}

# v4_srsh on 704 train alone
print("\n--- v4_srsh (geometry + SRSH S_12, 704 train) ---")
pipe_v4_704, cv_v4_704, test_v4_704, _ = train_eval(
    'v4_srsh (704 only)',
    X_704[train_mask & valid_704], labels_log[train_mask & valid_704],
    X_704[test_mask  & valid_704], labels_log[test_mask  & valid_704])
results['v4_srsh_704'] = {'cv': cv_v4_704, 'test': test_v4_704}

# v4_srsh_aug on 704 train + 111 Slater
print("\n--- v4_srsh_aug (geometry + SRSH S_12, 704 train + 111 Slater) ---")
X_aug = np.vstack([X_704[train_mask & valid_704], X_slater[valid_slater]])
y_aug = np.concatenate([labels_log[train_mask & valid_704], labels_log_slater[valid_slater]])

# Predict on Slater 111 separately to evaluate
pipe_v4_aug, cv_v4_aug, test_v4_aug, ex_aug = train_eval(
    'v4_srsh_aug',
    X_aug, y_aug,
    X_704[test_mask & valid_704], labels_log[test_mask & valid_704])
results['v4_srsh_aug'] = {'cv': cv_v4_aug, 'test': test_v4_aug}

# Save
joblib.dump(pipe_v4_aug, os.path.join(MODELS, 'stage2_rf_v4_srsh_aug.pkl'))
print(f"\nSaved: models/stage2_rf_v4_srsh_aug.pkl")

# ----------------------------------------------------------------------------
# MD validation: predict & compare against truth
print("\n" + "=" * 70)
print("MD validation predictions (15 conformations)")
print("=" * 70)

# MD truths from previous summary
GAMMA_STAR_MEV = {
    1: 39.9246, 2: 0.1905, 3: 5.4858, 4: 5.5266, 5: 42.1205,
    6: 7.6328, 7: 22.4902, 8: 30.6074, 9: 0.0000, 10: 0.0136,
    11: 0.5987, 12: 0.3755, 13: 0.0190, 14: 10.1553, 15: 0.0000,
}
md_J_mev = np.array([GAMMA_STAR_MEV.get(int(name.split('_')[-1]), np.nan) for name in md_names])

valid_md = np.isfinite(s12_md) & (md_J_mev > 0)
y_md_true_log = np.log10(md_J_mev[valid_md] / 1000.0)  # convert mV -> V (so log10(eV))
y_md_pred_v4_srsh = pipe_v4_aug.predict(X_md[valid_md])
y_md_pred_v2 = pipe_v2.predict(X_md[valid_md, :11])

mae_md_v2 = mean_absolute_error(y_md_true_log, y_md_pred_v2)
mae_md_v4 = mean_absolute_error(y_md_true_log, y_md_pred_v4_srsh)
print(f"  In-distribution MD ({valid_md.sum()} confs):")
print(f"    v2 baseline       : log MAE = {mae_md_v2:.4f}")
print(f"    v4_srsh_aug       : log MAE = {mae_md_v4:.4f}")

# Pearson on log
r_md_v2, _ = pearsonr(y_md_pred_v2, y_md_true_log)
r_md_v4, _ = pearsonr(y_md_pred_v4_srsh, y_md_true_log)
print(f"    v2 baseline Pearson r = {r_md_v2:.4f}")
print(f"    v4_srsh_aug Pearson r = {r_md_v4:.4f}")

# Per-conformation diagnostics for "interesting" structures
print("\n  Per-conformation predictions vs ground truth:")
print(f"    {'Conf':<6} {'J_true (meV)':>12} {'J_v2 (meV)':>12} {'J_v4_srsh (meV)':>18}  notes")
for i, name in enumerate(md_names):
    conf = int(name.split('_')[-1])
    J_true = GAMMA_STAR_MEV.get(conf, np.nan)
    pred_v2 = 10**pipe_v2.predict(X_md[i:i+1, :11])[0] * 1000.0
    if np.isfinite(s12_md[i]):
        pred_v4 = 10**pipe_v4_aug.predict(X_md[i:i+1])[0] * 1000.0
    else:
        pred_v4 = np.nan
    note = ''
    if conf == 9 or conf == 15:
        note = ' <-- known geometric node'
    elif conf in (2, 10, 13):
        note = ' <-- lateral disp outlier'
    print(f"    {conf:<6} {J_true:12.4f} {pred_v2:12.4f} {pred_v4:18.4f} {note}")

# Save summary
summary = {
    'v2_baseline':   {'cv': float(cv_v2), 'test': float(test_v2),
                      'md_mae': float(mae_md_v2), 'md_pearson': float(r_md_v2)},
    'v4_srsh_704':   {'cv': float(cv_v4_704), 'test': float(test_v4_704)},
    'v4_srsh_aug':   {'cv': float(cv_v4_aug), 'test': float(test_v4_aug),
                      'md_mae': float(mae_md_v4), 'md_pearson': float(r_md_v4)},
    'n_train_aug':   int(len(y_aug)),
    'n_test':        int((test_mask & valid_704).sum()),
    'n_md_indist':   int(valid_md.sum()),
}
with open(os.path.join(RESULTS, 'v4_srsh_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: results/v4_srsh_results.json")

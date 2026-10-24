"""
v7: Lateral displacement oscillation features.
Approach A (Fourier) and Approach B (polynomial), comparison and selection.
"""

import sys, os, glob, json
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

mpl.rcParams.update({'font.family': 'serif', 'font.size': 11,
                     'axes.labelsize': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'md_extracted')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

GAMMA_STAR_MEV = {
    1: 39.9246, 2: 0.1905, 3: 5.4858, 4: 5.5266, 5: 42.1205,
    6: 7.6328, 7: 22.4902, 8: 30.6074, 9: 0.0000, 10: 0.0136,
    11: 0.5987, 12: 0.3755, 13: 0.0190, 14: 10.1553, 15: 0.0000,
}

# Load base data
features_v2 = np.load(os.path.join(DATA_DIR, 'features_v2.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
labels_log = np.load(os.path.join(DATA_DIR, 'labels_log.npy'))
meta = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
N = len(labels_ev)

dz = features_v2[:, 0]
dx = features_v2[:, 1]
dy = features_v2[:, 2]
d_lat = features_v2[:, 8]

# Splits
sig_mask = labels_ev >= 1e-4
train_mask = (meta['split'] == 'train').values
test_mask = (meta['split'] == 'test').values

# ============================================================
# STEP 1A: Fit interference wavelength
# ============================================================
print("=" * 70)
print("STEP 1A: Fit interference wavelength from training data")
print("=" * 70)

regime = (dz < 5.5) & (d_lat > 1.0) & sig_mask & np.isfinite(labels_log) & train_mask
d_regime = d_lat[regime]
J_regime = labels_log[regime]
print(f"Fitting regime: {regime.sum()} structures")

def neg_r_fourier(params):
    lam1 = params[0]
    feat = np.cos(2 * np.pi * d_regime / lam1)
    r, _ = pearsonr(feat, J_regime)
    return -abs(r)

result = differential_evolution(neg_r_fourier, bounds=[(1.0, 6.0)], seed=42, maxiter=200)
lambda1_fit = result.x[0]
r_fit = -result.fun

# Also try lambda2 = 2*lambda1
lambda2_fit = 2.0 * lambda1_fit

# Compare to paper values
feat_paper = np.cos(2 * np.pi * d_regime / 2.3)
r_paper, _ = pearsonr(feat_paper, J_regime)

print(f"Fitted λ1 = {lambda1_fit:.2f} Å (r = {r_fit:.3f})")
print(f"λ2 = 2×λ1 = {lambda2_fit:.2f} Å")
print(f"Paper λ1 = 2.3 Å (r = {abs(r_paper):.3f})")

# Use fitted values if significantly better, otherwise paper
if r_fit - abs(r_paper) > 0.05:
    LAMBDA1, LAMBDA2 = lambda1_fit, lambda2_fit
    print(f"→ Using fitted values: λ1={LAMBDA1:.2f}, λ2={LAMBDA2:.2f}")
else:
    LAMBDA1, LAMBDA2 = 2.3, 4.6
    print(f"→ Using paper values: λ1={LAMBDA1}, λ2={LAMBDA2} (fitted improvement < 0.05)")

# ============================================================
# Build v3A and v3B feature matrices
# ============================================================
print("\n" + "=" * 70)
print("Building v3A (Fourier) and v3B (polynomial) feature matrices")
print("=" * 70)

def fourier_features(dx_arr, dy_arr, d_lat_arr, lam1, lam2):
    """6 Fourier oscillation features for all structures."""
    return np.column_stack([
        np.cos(2 * np.pi * d_lat_arr / lam1),
        np.sin(2 * np.pi * d_lat_arr / lam1),
        np.cos(2 * np.pi * d_lat_arr / lam2),
        np.sin(2 * np.pi * d_lat_arr / lam2),
        np.cos(2 * np.pi * dx_arr / lam2),
        np.cos(2 * np.pi * dy_arr / lam2),
    ])

def polynomial_features(dx_arr, dy_arr):
    """6 polynomial expansion features for all structures."""
    return np.column_stack([
        dx_arr**2,
        dy_arr**2,
        dx_arr * dy_arr,
        dx_arr**2 + dy_arr**2,
        np.abs(dx_arr),
        np.abs(dy_arr),
    ])

four_feats = fourier_features(dx, dy, d_lat, LAMBDA1, LAMBDA2)
poly_feats = polynomial_features(dx, dy)

features_v3a = np.hstack([features_v2, four_feats])  # (704, 17)
features_v3b = np.hstack([features_v2, poly_feats])  # (704, 17)

np.save(os.path.join(DATA_DIR, 'features_v3a.npy'), features_v3a)
np.save(os.path.join(DATA_DIR, 'features_v3b.npy'), features_v3b)
print(f"v3A shape: {features_v3a.shape}, v3B shape: {features_v3b.shape}")

# ============================================================
# STEP 2: Train RF for v2, v3A, v3B
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Train RF models")
print("=" * 70)

RF_PARAMS = dict(n_estimators=500, max_features=0.5, max_depth=20,
                 min_samples_leaf=1, random_state=42)

results_all = {}
models = {}

for name, feats in [('v2', features_v2), ('v3A', features_v3a), ('v3B', features_v3b)]:
    X_train = feats[train_mask & sig_mask]
    y_train = labels_log[train_mask & sig_mask]
    X_test = feats[test_mask & sig_mask]
    y_test = labels_log[test_mask & sig_mask]

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('rf', RandomForestRegressor(**RF_PARAMS))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5,
                                scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    # Feature importances
    imp = pipe.named_steps['rf'].feature_importances_

    results_all[name] = {'cv_mae': cv_mae, 'test_mae': test_mae, 'test_r2': test_r2,
                          'importances': imp}
    models[name] = pipe

    print(f"\n{name}: CV log MAE = {cv_mae:.4f}, Test log MAE = {test_mae:.4f}, R² = {test_r2:.4f}")
    joblib.dump(pipe, os.path.join(MODEL_DIR, f'stage2_rf_{name.lower()}.pkl'))

# ============================================================
# STEP 3: Evaluate on MD conformations
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: MD validation (vs Gamma*)")
print("=" * 70)

from geometry import dimer_features
from ase.io import read as ase_read

md_results = {}
for name, feats_train, pipe in [
    ('v2', features_v2, models['v2']),
    ('v3A', features_v3a, models['v3A']),
    ('v3B', features_v3b, models['v3B']),
]:
    preds = {}
    for cid in range(1, 16):
        xyz_path = os.path.join(MD_DIR, f'conf_{cid:02d}.xyz')
        if not os.path.exists(xyz_path):
            continue
        atoms = ase_read(xyz_path)
        symbols = np.array(atoms.get_chemical_symbols())
        positions = atoms.get_positions()
        M = len(symbols) // 2
        f = dimer_features(positions[:M], symbols[:M], positions[M:], symbols[M:])
        f11 = f[:11]

        if name == 'v3A':
            dl = np.sqrt(f11[1]**2 + f11[2]**2)
            osc = np.array([
                np.cos(2*np.pi*dl/LAMBDA1), np.sin(2*np.pi*dl/LAMBDA1),
                np.cos(2*np.pi*dl/LAMBDA2), np.sin(2*np.pi*dl/LAMBDA2),
                np.cos(2*np.pi*f11[1]/LAMBDA2), np.cos(2*np.pi*f11[2]/LAMBDA2),
            ])
            X = np.concatenate([f11, osc]).reshape(1, -1)
        elif name == 'v3B':
            poly = np.array([f11[1]**2, f11[2]**2, f11[1]*f11[2],
                             f11[1]**2+f11[2]**2, abs(f11[1]), abs(f11[2])])
            X = np.concatenate([f11, poly]).reshape(1, -1)
        else:
            X = f11.reshape(1, -1)

        log_J = pipe.predict(X)[0]
        J_mev = 10**log_J * 1000
        preds[cid] = J_mev

    # Compute log MAE vs Gamma*
    valid_confs = [c for c in preds if GAMMA_STAR_MEV[c] > 0]
    log_ratios = [np.log10(preds[c] / GAMMA_STAR_MEV[c]) for c in valid_confs]
    md_mae = np.mean(np.abs(log_ratios))

    # In-dist subset (excl. lateral-OOD confs 2,10,13 and nodes 9,15)
    in_dist_confs = [c for c in valid_confs if c not in [2, 10, 13]]
    if in_dist_confs:
        id_ratios = [np.log10(preds[c] / GAMMA_STAR_MEV[c]) for c in in_dist_confs]
        id_mae = np.mean(np.abs(id_ratios))
        log_star = [np.log10(GAMMA_STAR_MEV[c]) for c in in_dist_confs]
        log_pred = [np.log10(preds[c]) for c in in_dist_confs]
        r_id, _ = pearsonr(log_star, log_pred)
    else:
        id_mae = float('nan')
        r_id = float('nan')

    md_results[name] = {
        'md_mae': md_mae, 'id_mae': id_mae, 'r_id': r_id,
        'preds': preds,
        'conf9': preds.get(9, float('nan')),
        'conf15': preds.get(15, float('nan')),
    }
    print(f"{name}: MD log MAE(Γ*) = {md_mae:.3f}, "
          f"in-dist log MAE = {id_mae:.3f}, r(in-dist) = {r_id:.3f}, "
          f"Conf9={preds.get(9,0):.2f} meV, Conf15={preds.get(15,0):.2f} meV")

# ============================================================
# Master comparison table
# ============================================================
print("\n" + "=" * 70)
print("MASTER COMPARISON TABLE")
print("=" * 70)

header = (f"{'Model':<6} {'Feat':>4} {'704 CV':>8} {'704 test':>9} "
          f"{'MD Γ*':>7} {'MD r':>6} {'ID MAE':>7} {'C9':>7} {'C15':>7}")
print(header)
print("-" * len(header))
for name in ['v2', 'v3A', 'v3B']:
    r = results_all[name]
    m = md_results[name]
    nf = 11 if name == 'v2' else 17
    print(f"{name:<6} {nf:>4} {r['cv_mae']:>8.4f} {r['test_mae']:>9.4f} "
          f"{m['md_mae']:>7.3f} {m['r_id']:>6.3f} {m['id_mae']:>7.3f} "
          f"{m['conf9']:>7.2f} {m['conf15']:>7.2f}")

# ============================================================
# STEP 4: Interpretability — 1D profiles and 2D surface
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Interpretability analysis")
print("=" * 70)

# Feature importances
feat_names_v2 = ['dz','dx','dy','cosθ','cos4φ','sin4φ','cos4ψ','sin4ψ','d_lat','cos4α','sin4α']
feat_names_osc = ['cos(d/λ1)','sin(d/λ1)','cos(d/λ2)','sin(d/λ2)','cos(dx/λ2)','cos(dy/λ2)']
feat_names_poly = ['dx²','dy²','dx·dy','d_lat²','|dx|','|dy|']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: Feature importances
ax = axes[0]
for i, (name, feat_n) in enumerate([
    ('v2', feat_names_v2),
    ('v3A', feat_names_v2 + feat_names_osc),
    ('v3B', feat_names_v2 + feat_names_poly),
]):
    imp = results_all[name]['importances']
    nf = len(imp)
    y = np.arange(nf) + i * 0.25
    colors = ['#1f77b4'] * 11 + ['#d62728'] * (nf - 11)
    ax.barh(y, imp, height=0.22, color=colors, alpha=0.8, label=name)

ax.set_yticks(np.arange(17) + 0.25)
ax.set_yticklabels((feat_names_v2 + feat_names_osc)[:17], fontsize=7)
ax.set_xlabel('Feature importance')
ax.set_title('Feature importances')
ax.legend(fontsize=8)

# Panel B: 1D coupling profile (log10(J) vs d_lat at median features)
ax = axes[1]
medians = np.median(features_v2[train_mask & sig_mask], axis=0)
d_lat_sweep = np.linspace(0, 10, 200)

for name, pipe, color, ls in [
    ('v2', models['v2'], '#999999', '--'),
    ('v3A', models['v3A'], '#1f77b4', '-'),
    ('v3B', models['v3B'], '#ff7f0e', '-'),
]:
    preds = []
    for dl in d_lat_sweep:
        x = medians.copy()
        x[1] = dl  # dx = d_lat (along x axis for simplicity)
        x[2] = 0   # dy = 0
        x[8] = dl  # d_lateral

        if name == 'v3A':
            osc = np.array([
                np.cos(2*np.pi*dl/LAMBDA1), np.sin(2*np.pi*dl/LAMBDA1),
                np.cos(2*np.pi*dl/LAMBDA2), np.sin(2*np.pi*dl/LAMBDA2),
                np.cos(2*np.pi*dl/LAMBDA2), np.cos(0),
            ])
            X = np.concatenate([x, osc]).reshape(1, -1)
        elif name == 'v3B':
            poly = np.array([dl**2, 0, 0, dl**2, dl, 0])
            X = np.concatenate([x, poly]).reshape(1, -1)
        else:
            X = x.reshape(1, -1)

        preds.append(pipe.predict(X)[0])

    ax.plot(d_lat_sweep, preds, color=color, linestyle=ls, linewidth=2, label=name)

# Overlay training data at dz ≈ median
dz_med = medians[0]
near_med = (np.abs(dz - dz_med) < 0.5) & sig_mask
ax.scatter(d_lat[near_med], labels_log[near_med], c='steelblue', s=8, alpha=0.3, zorder=0)

ax.axvline(2.0, color='red', ls='--', lw=0.8, alpha=0.5)
ax.axvline(4.6, color='orange', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('d_lateral (Å)')
ax.set_ylabel('log10(J / eV)')
ax.set_title(f'1D profile (dz={dz_med:.1f} Å)')
ax.legend(fontsize=8)

# Panel C: 2D surface (v3A)
ax = axes[2]
n_grid = 60
dz_range = np.linspace(3.0, 6.0, n_grid)
dl_range = np.linspace(0.0, 8.0, n_grid)
Z = np.zeros((n_grid, n_grid))

for i, dz_val in enumerate(dz_range):
    for j, dl_val in enumerate(dl_range):
        x = medians.copy()
        x[0] = dz_val
        x[1] = dl_val; x[2] = 0; x[8] = dl_val
        osc = np.array([
            np.cos(2*np.pi*dl_val/LAMBDA1), np.sin(2*np.pi*dl_val/LAMBDA1),
            np.cos(2*np.pi*dl_val/LAMBDA2), np.sin(2*np.pi*dl_val/LAMBDA2),
            np.cos(2*np.pi*dl_val/LAMBDA2), np.cos(0),
        ])
        X = np.concatenate([x, osc]).reshape(1, -1)
        Z[i, j] = models['v3A'].predict(X)[0]

DZ, DL = np.meshgrid(dz_range, dl_range, indexing='ij')
ct = ax.contourf(DZ, DL, Z, levels=20, cmap='viridis')
plt.colorbar(ct, ax=ax, label='log10(J/eV)')
ax.axhline(2.0, color='red', ls='--', lw=1, alpha=0.7)
ax.axhline(4.6, color='orange', ls='--', lw=1, alpha=0.7)
ax.set_xlabel('dz (Å)'); ax.set_ylabel('d_lateral (Å)')
ax.set_title('v3A coupling surface')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'interpret_v3_profiles.png'), dpi=200)
plt.close()
print("Saved: interpret_v3_profiles.png")

# New oscillation features importance
for name in ['v3A', 'v3B']:
    imp = results_all[name]['importances']
    old_total = imp[:11].sum()
    new_total = imp[11:].sum()
    print(f"\n{name} importance budget: existing={old_total:.3f}, new={new_total:.3f} ({new_total/(old_total+new_total)*100:.1f}%)")

# ============================================================
# STEP 5: Publication figure
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Publication figure")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: coupling vs d_lat with model predictions
ax = axes[0]
short_dz_mask = (np.abs(dz - 4.0) < 0.5) & sig_mask
ax.scatter(d_lat[short_dz_mask], labels_log[short_dz_mask],
           c='steelblue', s=12, alpha=0.4, label='Data (dz≈4 Å)')
for name, color, ls in [('v2','#999999','--'), ('v3A','#1f77b4','-'), ('v3B','#ff7f0e','-')]:
    med4 = medians.copy()
    med4[0] = 4.0
    preds = []
    for dl in d_lat_sweep:
        x = med4.copy(); x[1]=dl; x[2]=0; x[8]=dl
        if name == 'v3A':
            osc = np.array([np.cos(2*np.pi*dl/LAMBDA1),np.sin(2*np.pi*dl/LAMBDA1),
                            np.cos(2*np.pi*dl/LAMBDA2),np.sin(2*np.pi*dl/LAMBDA2),
                            np.cos(2*np.pi*dl/LAMBDA2),np.cos(0)])
            X = np.concatenate([x, osc]).reshape(1,-1)
        elif name == 'v3B':
            poly = np.array([dl**2,0,0,dl**2,dl,0])
            X = np.concatenate([x, poly]).reshape(1,-1)
        else:
            X = x.reshape(1,-1)
        preds.append(models[name].predict(X)[0])
    ax.plot(d_lat_sweep, preds, color=color, ls=ls, lw=2, label=name)
ax.axvline(2.0, color='red', ls='--', lw=0.8, alpha=0.5)
ax.axvline(4.6, color='orange', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel(r'$d_\mathrm{lateral}$ (Å)'); ax.set_ylabel(r'$\log_{10}(J/\mathrm{eV})$')
ax.legend(fontsize=8)

# Panel B: grouped importance
ax = axes[1]
groups = {'Distance': [0,3], 'Angular': [4,5,6,7], 'Slippage': [1,2,8,9,10], 'New': list(range(11,17))}
x_pos = np.arange(len(groups))
width = 0.25
for i, (name, color) in enumerate([('v2','#999999'), ('v3A','#1f77b4'), ('v3B','#ff7f0e')]):
    imp = results_all[name]['importances']
    vals = []
    for gname, indices in groups.items():
        valid_idx = [j for j in indices if j < len(imp)]
        vals.append(sum(imp[j] for j in valid_idx))
    ax.bar(x_pos + i*width, vals, width, color=color, label=name, edgecolor='k', lw=0.3)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(groups.keys())
ax.set_ylabel('Summed importance')
ax.legend(fontsize=8)

# Panel C: 2D surface v3A
ax = axes[2]
ct = ax.contourf(DZ, DL, Z, levels=20, cmap='viridis')
plt.colorbar(ct, ax=ax, label=r'$\log_{10}(J/\mathrm{eV})$')
ax.axhline(2.0, color='red', ls='--', lw=1)
ax.axhline(4.6, color='orange', ls='--', lw=1)
ax.set_xlabel(r'$d_z$ (Å)'); ax.set_ylabel(r'$d_\mathrm{lateral}$ (Å)')

plt.tight_layout()
for ext in ['png', 'pdf']:
    plt.savefig(os.path.join(FIG_DIR, f'pub_fig7_lateral_oscillation.{ext}'))
plt.close()
print("Saved: pub_fig7_lateral_oscillation.png/pdf")

# ============================================================
# STEP 6: Model selection
# ============================================================
print("\n" + "=" * 70)
print("MODEL SELECTION")
print("=" * 70)

v2_test = results_all['v2']['test_mae']
for name in ['v3A', 'v3B']:
    delta = results_all[name]['test_mae'] - v2_test
    status = "PASS" if delta <= 0.02 else "FAIL"
    print(f"Criterion 1 (704 test MAE): {name} delta = {delta:+.4f} → {status}")

print(f"\nCriterion 2 (MD Γ* MAE): v2={md_results['v2']['md_mae']:.3f}, "
      f"v3A={md_results['v3A']['md_mae']:.3f}, v3B={md_results['v3B']['md_mae']:.3f}")
print(f"Criterion 2 (in-dist): v2={md_results['v2']['id_mae']:.3f}, "
      f"v3A={md_results['v3A']['id_mae']:.3f}, v3B={md_results['v3B']['id_mae']:.3f}")

print(f"\nCriterion 3 (geometric nodes):")
for name in ['v2', 'v3A', 'v3B']:
    m = md_results[name]
    print(f"  {name}: Conf9={m['conf9']:.2f} meV, Conf15={m['conf15']:.2f} meV (target: ~0)")

# Select best
candidates = []
for name in ['v2', 'v3A', 'v3B']:
    if results_all[name]['test_mae'] <= v2_test + 0.02:
        candidates.append(name)
print(f"\nCandidates passing criterion 1: {candidates}")

best = min(candidates, key=lambda n: md_results[n]['id_mae'])
print(f"\n→ Selected model: {best} (best in-dist MD MAE = {md_results[best]['id_mae']:.3f})")

# Save final model
final_pipe = models[best]
joblib.dump(final_pipe, os.path.join(MODEL_DIR, 'stage2_rf_final.pkl'))

config = {
    'version': best,
    'n_features': 17 if best != 'v2' else 11,
    'lambda1': float(LAMBDA1) if best == 'v3A' else None,
    'lambda2': float(LAMBDA2) if best == 'v3A' else None,
    'feature_names': (feat_names_v2 + (feat_names_osc if best=='v3A' else feat_names_poly if best=='v3B' else [])),
    'test_log_mae': float(results_all[best]['test_mae']),
    'md_log_mae_gamma_star': float(md_results[best]['md_mae']),
    'md_in_dist_log_mae': float(md_results[best]['id_mae']),
    'selection_rationale': f"{best} selected: best in-dist MD performance while maintaining 704-structure test accuracy.",
}
with open(os.path.join(MODEL_DIR, 'model_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Save comparison results
with open(os.path.join(RESULTS_DIR, 'final_model_comparison.json'), 'w') as f:
    json.dump({
        name: {
            'cv_mae': float(results_all[name]['cv_mae']),
            'test_mae': float(results_all[name]['test_mae']),
            'md_mae': float(md_results[name]['md_mae']),
            'id_mae': float(md_results[name]['id_mae']),
            'r_id': float(md_results[name]['r_id']),
        } for name in ['v2', 'v3A', 'v3B']
    }, f, indent=2)

print(f"\nSaved: models/stage2_rf_final.pkl, models/model_config.json")
print(f"Saved: results/final_model_comparison.json")

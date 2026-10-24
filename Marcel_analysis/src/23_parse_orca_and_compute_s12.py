"""
v8: Parse ORCA HOMO, compute DFT S12 for all structures, diagnose and retrain.
Runs Steps 2-6 of prompt_v8.md after ORCA calculation completes.
"""

import sys, os, glob, json
import numpy as np
import pandas as pd
import cclib
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import joblib

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
STRUCT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'structures')
MD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'md_extracted')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
ORCA_DIR = os.path.join(os.path.dirname(__file__), '..', 'orca')

GAMMA_STAR_MEV = {
    1: 39.9246, 2: 0.1905, 3: 5.4858, 4: 5.5266, 5: 42.1205,
    6: 7.6328, 7: 22.4902, 8: 30.6074, 9: 0.0000, 10: 0.0136,
    11: 0.5987, 12: 0.3755, 13: 0.0190, 14: 10.1553, 15: 0.0000,
}

RF_PARAMS = dict(n_estimators=500, max_features=0.5, max_depth=20,
                 min_samples_leaf=1, random_state=42)

from geometry import load_dimer_xyz, dimer_features
from ase.io import read as ase_read

# ============================================================
# STEP 2: Parse ORCA HOMO — try all available multiplicities
# ============================================================
print("=" * 70)
print("STEP 2: Parse ORCA output and identify π-dominated HOMO")
print("=" * 70)

best_mult = None
best_cn_frac = 0
best_data = None

for mult in [4, 2, 6]:
    out_file = os.path.join(ORCA_DIR, f'fe_porphine_mult{mult}.out')
    if not os.path.exists(out_file):
        print(f"  mult={mult}: output file not found, skipping")
        continue

    try:
        data = cclib.io.ccread(out_file)
        if data is None or not hasattr(data, 'mocoeffs'):
            print(f"  mult={mult}: cclib could not parse MO coefficients")
            continue

        homo_idx = data.homos[0]  # alpha HOMO
        homo_coeffs = data.mocoeffs[0][homo_idx]
        homo_energy = data.moenergies[0][homo_idx]

        # Compute atom contributions
        fe_weight = 0
        cn_weight = 0
        h_weight = 0
        for atom_idx, ao_indices in enumerate(data.atombasis):
            atom_w = np.sum(homo_coeffs[ao_indices] ** 2)
            sym = cclib.parser.utils.PeriodicTable().element[data.atomnos[atom_idx]]
            if sym == 'Fe':
                fe_weight += atom_w
            elif sym in ('C', 'N'):
                cn_weight += atom_w
            elif sym == 'H':
                h_weight += atom_w

        total = fe_weight + cn_weight + h_weight
        cn_frac = cn_weight / total if total > 0 else 0

        print(f"  mult={mult}: HOMO energy={homo_energy:.3f} eV, "
              f"Fe={fe_weight/total*100:.1f}%, C+N={cn_frac*100:.1f}%, "
              f"H={h_weight/total*100:.1f}%")

        if cn_frac > best_cn_frac:
            best_cn_frac = cn_frac
            best_mult = mult
            best_data = data

    except Exception as e:
        print(f"  mult={mult}: FAILED — {e}")

if best_data is None:
    print("\nERROR: No ORCA output successfully parsed. Cannot proceed.")
    print("Run ORCA first: see orca/fe_porphine_mult*.inp")
    sys.exit(1)

print(f"\n→ Selected mult={best_mult} (C+N fraction = {best_cn_frac*100:.1f}%)")
data = best_data
homo_idx = data.homos[0]
homo_coeffs = data.mocoeffs[0][homo_idx]

if best_cn_frac < 0.6:
    print("WARNING: HOMO is not clearly π-dominated (C+N < 60%).")
    print("Proceeding with best available, but results may be suboptimal.")

# Save full HOMO coefficients
np.save(os.path.join(DATA_DIR, 'homo_dft_coeffs.npy'), homo_coeffs)
print(f"Saved HOMO coefficients: {len(homo_coeffs)} AOs")

# ============================================================
# STEP 3: Compute S12 for all 704 structures using PySCF
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Compute DFT S12 for all structures")
print("=" * 70)

from pyscf import gto

def build_pyscf_basis_from_cclib(data):
    """Extract per-element basis from cclib's gbasis."""
    basis = {}
    for atom_idx, gbasis_atom in enumerate(data.gbasis):
        sym = cclib.parser.utils.PeriodicTable().element[data.atomnos[atom_idx]]
        if sym in basis:
            continue
        basis[sym] = []
        for shell_type, primitives in gbasis_atom:
            shell_data = [shell_type] + [[e, c] for e, c in primitives]
            basis[sym].append(shell_data)
    return basis

pyscf_basis = build_pyscf_basis_from_cclib(data)

# Map from reference monomer heavy atoms to HOMO coefficients
ref_heavy_indices = [i for i, z in enumerate(data.atomnos)
                     if cclib.parser.utils.PeriodicTable().element[z] != 'H']
ref_n_heavy = len(ref_heavy_indices)
print(f"Reference heavy atoms: {ref_n_heavy}")

def compute_s12_dft(m1_pos, m1_sym, m2_pos, m2_sym):
    """Compute HOMO-HOMO overlap using DFT coefficients + PySCF integrals."""
    # Build dimer with heavy atoms only
    atom_list = []
    for s, p in zip(m1_sym, m1_pos):
        if s != 'H':
            atom_list.append((s, p.tolist()))
    n_m1_heavy = len(atom_list)
    for s, p in zip(m2_sym, m2_pos):
        if s != 'H':
            atom_list.append((s, p.tolist()))

    mol = gto.M(atom=atom_list, basis=pyscf_basis, unit='Angstrom',
                verbose=0, cart=True)
    S_ao = mol.intor('int1e_ovlp')

    # Build coefficient vectors
    n_ao = S_ao.shape[0]
    c1 = np.zeros(n_ao)
    c2 = np.zeros(n_ao)

    # Map reference HOMO coefficients to dimer AOs
    # Reference heavy atom k → dimer atom k (M1) and k + n_m1_heavy (M2)
    slices = mol.aoslice_by_atom()
    ref_heavy_k = 0
    for atom_idx in range(data.natom):
        sym = cclib.parser.utils.PeriodicTable().element[data.atomnos[atom_idx]]
        if sym == 'H':
            continue

        ref_ao_list = data.atombasis[atom_idx]
        ref_coeffs_atom = homo_coeffs[ref_ao_list]

        if ref_heavy_k < n_m1_heavy:
            m1_start = slices[ref_heavy_k, 2]
            m1_end = slices[ref_heavy_k, 3]
            n_ao_dimer = m1_end - m1_start
            if n_ao_dimer == len(ref_coeffs_atom):
                c1[m1_start:m1_end] = ref_coeffs_atom

        m2_idx = n_m1_heavy + ref_heavy_k
        if m2_idx < mol.natm:
            m2_start = slices[m2_idx, 2]
            m2_end = slices[m2_idx, 3]
            n_ao_dimer2 = m2_end - m2_start
            if n_ao_dimer2 == len(ref_coeffs_atom):
                c2[m2_start:m2_end] = ref_coeffs_atom

        ref_heavy_k += 1

    return float(c1 @ S_ao @ c2)

# Test on first structure
xyz_files = sorted(glob.glob(os.path.join(STRUCT_DIR, '*.xyz')))
d_test = load_dimer_xyz(xyz_files[0])
s12_test = compute_s12_dft(
    d_test['monomer1']['positions'], d_test['monomer1']['symbols'],
    d_test['monomer2']['positions'], d_test['monomer2']['symbols'])
print(f"Test S12 (first structure): {s12_test:.6f}")

# Self-overlap check
ref_sym = np.array([cclib.parser.utils.PeriodicTable().element[z]
                     for z in data.atomnos])
ref_pos = data.atomcoords[-1]
ref_heavy_mask = ref_sym != 'H'
ref_mol = gto.M(
    atom=[(s, p.tolist()) for s, p in zip(ref_sym[ref_heavy_mask], ref_pos[ref_heavy_mask])],
    basis=pyscf_basis, unit='Angstrom', verbose=0, cart=True)
S_self = ref_mol.intor('int1e_ovlp')
ref_slices = ref_mol.aoslice_by_atom()
c_self = np.zeros(S_self.shape[0])
ref_heavy_k = 0
for atom_idx in range(data.natom):
    sym_a = cclib.parser.utils.PeriodicTable().element[data.atomnos[atom_idx]]
    if sym_a == 'H':
        continue
    ref_ao_list = data.atombasis[atom_idx]
    ref_coeffs_atom = homo_coeffs[ref_ao_list]
    start = ref_slices[ref_heavy_k, 2]
    end = ref_slices[ref_heavy_k, 3]
    if end - start == len(ref_coeffs_atom):
        c_self[start:end] = ref_coeffs_atom
    ref_heavy_k += 1
self_overlap = float(c_self @ S_self @ c_self)
print(f"HOMO self-overlap: {self_overlap:.4f} (should be ~1.0)")

# Compute for all 704 structures
cache_file = os.path.join(DATA_DIR, 's12_dft_orca.npz')
if os.path.exists(cache_file):
    print("Loading cached DFT S12...")
    cached = np.load(cache_file)
    s12_arr = cached['s12']
    print(f"Loaded {len(s12_arr)} values")
else:
    coupling_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'coupling.csv'))
    coupling_dict = {}
    for _, row in coupling_csv.iterrows():
        coupling_dict[row['Filename'].replace('.out', '')] = row['Coupling_eV']

    s12_list = []
    filenames = []
    for filepath in tqdm(xyz_files, desc="Computing DFT S12"):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        if stem not in coupling_dict:
            continue
        try:
            d = load_dimer_xyz(filepath)
            s12 = compute_s12_dft(
                d['monomer1']['positions'], d['monomer1']['symbols'],
                d['monomer2']['positions'], d['monomer2']['symbols'])
            s12_list.append(s12)
        except Exception as e:
            s12_list.append(np.nan)
            tqdm.write(f"FAILED {stem}: {e}")
        filenames.append(stem)

    s12_arr = np.array(s12_list)
    np.savez(cache_file, s12=s12_arr, filenames=np.array(filenames))
    print(f"\nSaved: {cache_file}")

labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
labels_log = np.load(os.path.join(DATA_DIR, 'labels_log.npy'))
features_v2 = np.load(os.path.join(DATA_DIR, 'features_v2.npy'))
meta = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

print(f"|S12_DFT| range: [{np.nanmin(np.abs(s12_arr)):.6e}, {np.nanmax(np.abs(s12_arr)):.6e}]")
print(f"NaN count: {np.isnan(s12_arr).sum()}")

# ============================================================
# STEP 4: Diagnostics
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Diagnostics — DFT S12 vs Hückel vs xTB")
print("=" * 70)

sig = (labels_ev >= 1e-4) & np.isfinite(labels_log) & np.isfinite(s12_arr)

r_dft, p_dft = pearsonr(np.abs(s12_arr[sig]), labels_log[sig])
print(f"Pearson r(|S12_DFT|, log10(J)): {r_dft:.3f} (p={p_dft:.2e})")
print(f"  Hückel:  r = 0.462")
print(f"  xTB:     r = 0.361")

# Cos_4phi signal retention
cos_4phi = features_v2[:, 4]
dz = features_v2[:, 0]
ratio = labels_log[sig] - np.log10(np.abs(s12_arr[sig]) + 1e-12)
r_phi_raw, _ = pearsonr(cos_4phi[sig], labels_log[sig])
r_phi_ratio, _ = pearsonr(cos_4phi[sig], ratio)
phi_retained = abs(r_phi_ratio) / abs(r_phi_raw) * 100

r_dz_raw, _ = pearsonr(dz[sig], labels_log[sig])
r_dz_ratio, _ = pearsonr(dz[sig], ratio)
dz_retained = abs(r_dz_ratio) / abs(r_dz_raw) * 100

print(f"\nAngular signal: cos_4phi retained = {phi_retained:.1f}% (Hückel: 60%, xTB: 23%)")
print(f"Distance signal: dz retained = {dz_retained:.1f}% (Hückel: 33%, xTB: 87%)")

# Decision gate
gate1 = r_dft > 0.462
gate2 = phi_retained < 20
gate3 = dz_retained < 15
print(f"\nDecision gate:")
print(f"  1. r > 0.462: {r_dft:.3f} → {'PASS' if gate1 else 'FAIL'}")
print(f"  2. cos_4phi retained < 20%: {phi_retained:.1f}% → {'PASS' if gate2 else 'FAIL'}")
print(f"  3. dz retained < 15%: {dz_retained:.1f}% → {'PASS' if gate3 else 'FAIL'}")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
d_lat = features_v2[:, 8]
pd_scan = (dz >= 3.0) & (dz <= 5.5) & np.isfinite(s12_arr)

axes[0].scatter(d_lat[pd_scan], s12_arr[pd_scan], c=labels_log[pd_scan],
                cmap='viridis', s=15, alpha=0.6)
axes[0].axhline(0, color='k', lw=0.8)
axes[0].axvline(2.0, color='red', ls='--', lw=1)
axes[0].axvline(4.6, color='orange', ls='--', lw=1)
axes[0].set_xlabel('d_lateral (Å)'); axes[0].set_ylabel('S₁₂ (DFT)')
axes[0].set_title('S₁₂ vs d_lateral')

axes[1].scatter(d_lat[pd_scan & sig], labels_log[pd_scan & sig],
                c='steelblue', s=15, alpha=0.5)
axes[1].axvline(2.0, color='red', ls='--'); axes[1].axvline(4.6, color='orange', ls='--')
axes[1].set_xlabel('d_lateral (Å)'); axes[1].set_ylabel('log10(J/eV)')
axes[1].set_title('Reference coupling')

axes[2].scatter(np.abs(s12_arr[sig]), labels_log[sig], s=10, alpha=0.4)
axes[2].set_xlabel('|S₁₂| (DFT)'); axes[2].set_ylabel('log10(J/eV)')
axes[2].set_title(f'r = {r_dft:.3f}')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'diagnose_s12_dft_orca.png'), dpi=200)
plt.close()
print("Saved: diagnose_s12_dft_orca.png")

# Save diagnostic results
diag = {
    'multiplicity': int(best_mult),
    'cn_fraction': float(best_cn_frac),
    'pearson_r_dft': float(r_dft),
    'cos4phi_retained_pct': float(phi_retained),
    'dz_retained_pct': float(dz_retained),
    'self_overlap': float(self_overlap),
    'gate1_pass': bool(gate1),
    'gate2_pass': bool(gate2),
    'gate3_pass': bool(gate3),
}
with open(os.path.join(RESULTS_DIR, 'orca_s12_diagnostics.json'), 'w') as f:
    json.dump(diag, f, indent=2)
print(f"\nSaved: results/orca_s12_diagnostics.json")

if not (gate1 or gate2 or gate3):
    print("\nAll gates FAILED. Stopping — DFT S12 does not improve on Hückel.")
    print("Possible causes: wrong spin state, basis mismatch, or AO mapping error.")
    sys.exit(0)

print("\nAt least one gate passed — proceeding to retrain.")

# ============================================================
# STEP 5: Retrain with DFT S12 as features
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Retrain with DFT S12 features")
print("=" * 70)

# Add S12, S12², log|S12| as features 11, 12, 13
s12_abs = np.abs(s12_arr)
log_s12 = np.log10(s12_abs + 1e-12)

features_v4 = np.column_stack([features_v2, s12_arr, s12_arr**2, log_s12])
np.save(os.path.join(DATA_DIR, 'features_v4_dft.npy'), features_v4)
print(f"features_v4 shape: {features_v4.shape}")

# Splits
train_mask = (meta['split'] == 'train').values
test_mask = (meta['split'] == 'test').values
sig_mask = labels_ev >= 1e-4

# Train v4 RF
valid_mask = np.isfinite(s12_arr)
X_train = features_v4[train_mask & sig_mask & valid_mask]
y_train = labels_log[train_mask & sig_mask & valid_mask]
X_test = features_v4[test_mask & sig_mask & valid_mask]
y_test = labels_log[test_mask & sig_mask & valid_mask]

pipe = Pipeline([('scaler', StandardScaler()),
                 ('rf', RandomForestRegressor(**RF_PARAMS))])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

cv_scores = cross_val_score(pipe, X_train, y_train, cv=5,
                            scoring='neg_mean_absolute_error')
cv_mae = -cv_scores.mean()
test_mae = mean_absolute_error(y_test, y_pred)

# v2 baseline
pipe_v2 = Pipeline([('scaler', StandardScaler()),
                     ('rf', RandomForestRegressor(**RF_PARAMS))])
pipe_v2.fit(features_v2[train_mask & sig_mask], labels_log[train_mask & sig_mask])
y_pred_v2 = pipe_v2.predict(features_v2[test_mask & sig_mask])
mae_v2 = mean_absolute_error(labels_log[test_mask & sig_mask], y_pred_v2)

print(f"v2 (11-feat): test log MAE = {mae_v2:.4f}")
print(f"v4 (14-feat + DFT S12): CV log MAE = {cv_mae:.4f}, test log MAE = {test_mae:.4f}")

# Feature importances
imp = pipe.named_steps['rf'].feature_importances_
feat_names = ['dz','dx','dy','cosθ','cos4φ','sin4φ','cos4ψ','sin4ψ',
              'd_lat','cos4α','sin4α','S12','S12²','log|S12|']
print("\nFeature importances:")
for n, v in sorted(zip(feat_names, imp), key=lambda x: -x[1]):
    print(f"  {n:<10}: {v:.4f}")

s12_importance = imp[11] + imp[12] + imp[13]
print(f"\nTotal S12 importance: {s12_importance:.3f} ({s12_importance*100:.1f}%)")

joblib.dump(pipe, os.path.join(MODEL_DIR, 'stage2_rf_v4_dft.pkl'))
print("Saved: models/stage2_rf_v4_dft.pkl")

# Save summary
summary = {
    'v2_test_mae': float(mae_v2),
    'v4_cv_mae': float(cv_mae),
    'v4_test_mae': float(test_mae),
    's12_total_importance': float(s12_importance),
}
with open(os.path.join(RESULTS_DIR, 'orca_v4_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved: results/orca_v4_results.json")

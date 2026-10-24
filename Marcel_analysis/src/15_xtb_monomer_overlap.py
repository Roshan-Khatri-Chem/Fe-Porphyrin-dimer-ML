"""
xTB experiment Part 2: Compute proper two-centre overlap S12 using
xTB HOMO coefficients and PySCF overlap integrals.

The xTB Molden file uses Cartesian d-functions (6 components) rather than
spherical (5), so PySCF must be configured with cart=True for d-shells.
"""

import sys, os, glob, json, tempfile, subprocess, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(__file__))
from geometry import load_dimer_xyz

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
STRUCT_DIR = os.path.join(DATA_DIR, 'structures')
COUPLING_CSV = os.path.join(DATA_DIR, 'coupling.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

XTB = os.path.expanduser('~/miniconda3/bin/xtb')
BOHR_TO_ANG = 0.529177210903

DEC_BINS = [(-4, -3), (-3, -2), (-2, -1), (-1, 0)]
RF_PARAMS = {
    'rf__n_estimators': [200, 500],
    'rf__max_features': ['sqrt', 0.5],
    'rf__min_samples_leaf': [1, 2, 3],
    'rf__max_depth': [None, 10, 20],
}

GEOM_FEAT_NAMES = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi',
                    'cos_4psi', 'sin_4psi', 'd_lateral', 'cos_4alpha', 'sin_4alpha']


def parse_molden(filepath):
    """Parse Molden file -> atoms, GTO basis, MO coefficients."""
    with open(filepath) as f:
        content = f.read()

    # Atoms (Bohr -> Angstrom)
    atoms_section = re.search(r'\[Atoms\]\s+AU\n(.*?)\n\[', content, re.DOTALL)
    atoms = []
    for line in atoms_section.group(1).strip().split('\n'):
        parts = line.split()
        xyz = np.array([float(parts[3]), float(parts[4]), float(parts[5])]) * BOHR_TO_ANG
        atoms.append((parts[0], xyz))

    # GTO basis per atom
    gto_section = re.search(r'\[GTO\]\n(.*?)\n\[', content, re.DOTALL)
    basis_per_atom = {}  # atom_idx -> list of (shell_type, [(exp, coef), ...])
    current_atom = None
    current_shell = None
    primitives = []
    for line in gto_section.group(1).strip().split('\n'):
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 2 and parts[1] == '0':
            if current_atom is not None and current_shell is not None:
                basis_per_atom[current_atom].append((current_shell, primitives))
            current_atom = int(parts[0]) - 1
            basis_per_atom[current_atom] = []
            current_shell = None
            primitives = []
        elif len(parts) >= 3 and parts[0] in ('s', 'p', 'd', 'f'):
            if current_shell is not None:
                basis_per_atom[current_atom].append((current_shell, primitives))
            current_shell = parts[0]
            primitives = []
        elif len(parts) >= 2:
            try:
                primitives.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    if current_atom is not None and current_shell is not None:
        basis_per_atom[current_atom].append((current_shell, primitives))

    # MOs
    mo_section = re.search(r'\[MO\]\n(.*?)$', content, re.DOTALL)
    mos = []
    current_mo = {}
    coeffs = []
    for line in mo_section.group(1).strip().split('\n'):
        line = line.strip()
        if line.startswith('Sym='):
            if coeffs:
                current_mo['coeffs'] = np.array(coeffs)
                mos.append(current_mo)
            current_mo = {'sym': line.split('=')[1].strip()}
            coeffs = []
        elif line.startswith('Ene='):
            current_mo['energy'] = float(line.split('=')[1])
        elif line.startswith('Occup='):
            current_mo['occup'] = float(line.split('=')[1])
        elif line.startswith('Spin='):
            pass
        else:
            parts = line.split()
            if len(parts) == 2:
                try:
                    int(parts[0])
                    coeffs.append(float(parts[1]))
                except ValueError:
                    pass
    if coeffs:
        current_mo['coeffs'] = np.array(coeffs)
        mos.append(current_mo)

    return atoms, basis_per_atom, mos


def build_pyscf_mol(positions, symbols, basis_per_atom, mol_atoms):
    """
    Build a PySCF Mole object with the xTB GTO basis.
    Uses Cartesian d-functions (cart=True) to match xTB's Molden output.
    """
    from pyscf import gto

    # Build per-element basis from the parsed data
    # Each element shares the same basis across all atoms of that type
    basis = {}
    for atom_idx, shells in basis_per_atom.items():
        sym = mol_atoms[atom_idx][0]
        if sym in basis:
            continue
        basis[sym] = []
        for shell_type, prims in shells:
            ang_mom = {'s': 0, 'p': 1, 'd': 2, 'f': 3}[shell_type]
            shell_data = [ang_mom] + [[e, c] for e, c in prims]
            basis[sym].append(shell_data)

    atom_list = [(s, p.tolist()) for s, p in zip(symbols, positions)]
    mol = gto.M(atom=atom_list, basis=basis, unit='Angstrom', verbose=0, cart=True)
    return mol, basis


def compute_S12(pos1, sym1, pos2, sym2, homo_coeffs, basis):
    """Compute HOMO-HOMO overlap between two monomers via PySCF."""
    from pyscf import gto

    # Build dimer molecule (both monomers in one system)
    atom_list = []
    for s, p in zip(sym1, pos1):
        atom_list.append((s, p.tolist()))
    for s, p in zip(sym2, pos2):
        atom_list.append((s, p.tolist()))

    mol = gto.M(atom=atom_list, basis=basis, unit='Angstrom', verbose=0, cart=True)
    S = mol.intor('int1e_ovlp')

    n_mono = len(homo_coeffs)
    C1 = np.zeros(S.shape[0])
    C1[:n_mono] = homo_coeffs
    C2 = np.zeros(S.shape[0])
    C2[n_mono:] = homo_coeffs

    return float(C1 @ S @ C2)


def strat_mae(yt, yp):
    decades = np.floor(yt).astype(int)
    r = {}
    for lo, hi in DEC_BINS:
        m = (decades >= lo) & (decades < hi)
        if m.sum() > 0:
            r[f'[1e{lo},1e{hi})'] = {
                'log_mae': float(mean_absolute_error(yt[m], yp[m])),
                'N': int(m.sum())
            }
    return r


# ============================================================
# STEP 1: Run xTB on isolated monomer, extract HOMO
# ============================================================
print("=" * 70)
print("STEP 1: xTB monomer calculation")
print("=" * 70)

from ase.io import write as ase_write
from ase import Atoms

d = load_dimer_xyz(os.path.join(STRUCT_DIR, '35_ang_02.xyz'))
mono_atoms = Atoms(symbols=d['monomer1']['symbols'],
                   positions=d['monomer1']['positions'])
mono_path = '/tmp/monomer_xtb.xyz'
ase_write(mono_path, mono_atoms)

workdir = tempfile.mkdtemp(prefix='xtb_mono_')
result = subprocess.run(
    [XTB, mono_path, '--gfn', '2', '--chrg', '0', '--uhf', '0', '--molden'],
    capture_output=True, text=True, cwd=workdir, timeout=120
)
print(f"xTB return code: {result.returncode}")

molden_file = os.path.join(workdir, 'molden.input')
mol_atoms, basis_per_atom, mos = parse_molden(molden_file)
print(f"Parsed: {len(mol_atoms)} atoms, {len(mos)} MOs")

# Find HOMO: last MO with occupation > 0.5
homo_idx = None
for i in range(len(mos) - 1, -1, -1):
    if mos[i].get('occup', 0) > 0.5:
        homo_idx = i
        break

homo_mo = mos[homo_idx]
homo_coeffs = homo_mo['coeffs']
print(f"HOMO: MO {homo_idx + 1}, energy = {homo_mo['energy']:.6f} Ha, "
      f"occup = {homo_mo['occup']:.4f}, n_coeffs = {len(homo_coeffs)}")

# Build PySCF basis and verify self-overlap
mono_sym = np.array([a[0] for a in mol_atoms])
mono_pos = np.array([a[1] for a in mol_atoms])

pyscf_mol, pyscf_basis = build_pyscf_mol(mono_pos, mono_sym, basis_per_atom, mol_atoms)
print(f"PySCF basis functions: {pyscf_mol.nao}")
print(f"HOMO coefficients: {len(homo_coeffs)}")

S_self = pyscf_mol.intor('int1e_ovlp')
self_overlap = float(homo_coeffs @ S_self @ homo_coeffs)
print(f"Monomer HOMO self-overlap: {self_overlap:.6f} (should be ~1.0)")

# ============================================================
# STEP 2: Compute S12 for all 704 dimers
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Compute S12 for all dimers")
print("=" * 70)

df_coup = pd.read_csv(COUPLING_CSV)
coupling_dict = {}
for _, row in df_coup.iterrows():
    stem = row['Filename'].replace('.out', '')
    coupling_dict[stem] = row['Coupling_eV']

xyz_files = sorted(glob.glob(os.path.join(STRUCT_DIR, '*.xyz')))

cache_file = os.path.join(PROCESSED_DIR, 'xtb_S12.npz')
if os.path.exists(cache_file):
    print("Loading cached S12 results...")
    cached = np.load(cache_file, allow_pickle=True)
    filenames = cached['filenames'].tolist()
    S12_vals = cached['S12']
    labels_ev = cached['labels_ev']
    print(f"Loaded {len(filenames)} cached results")
else:
    filenames = []
    S12_list = []
    labels_ev_list = []
    n_failed = 0

    for filepath in tqdm(xyz_files, desc="Computing S12"):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        if stem not in coupling_dict:
            continue
        try:
            dd = load_dimer_xyz(filepath)
            s12 = compute_S12(
                dd['monomer1']['positions'], dd['monomer1']['symbols'],
                dd['monomer2']['positions'], dd['monomer2']['symbols'],
                homo_coeffs, pyscf_basis)
            filenames.append(stem)
            S12_list.append(s12)
            labels_ev_list.append(coupling_dict[stem])
        except Exception as e:
            n_failed += 1
            if n_failed <= 3:
                tqdm.write(f"  ERROR {stem}: {e}")

    S12_vals = np.array(S12_list)
    labels_ev = np.array(labels_ev_list)
    print(f"\nCompleted: {len(filenames)}, failed: {n_failed}")

    np.savez(cache_file, filenames=np.array(filenames), S12=S12_vals,
             labels_ev=labels_ev)

labels_log = np.where(labels_ev > 0, np.log10(labels_ev), np.nan)
S12_abs = np.abs(S12_vals)
N = len(filenames)

print(f"|S12| range: [{S12_abs.min():.6e}, {S12_abs.max():.6e}]")
print(f"|S12| < 1e-6: {(S12_abs < 1e-6).sum()}")

# ============================================================
# STEP 3: Correlations
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Correlation analysis")
print("=" * 70)

valid = (labels_ev > 0) & (S12_abs > 0)
r_S12, p_S12 = pearsonr(np.log10(S12_abs[valid]), labels_log[valid])
print(f"Pearson r(log|S12|, log10(J)): {r_S12:.4f} (p = {p_S12:.2e})")

r_S12_lin, _ = pearsonr(S12_abs[valid], labels_log[valid])
print(f"Pearson r(|S12|, log10(J)): {r_S12_lin:.4f}")
print(f"  (Gaussian S_HL was: r = 0.462)")

# Near-zero vs significant
is_near_zero = labels_ev < 1e-4
sig_mask = labels_ev >= 1e-4
print(f"\nMean |S12| near-zero: {S12_abs[is_near_zero].mean():.6e}")
print(f"Mean |S12| significant: {S12_abs[sig_mask].mean():.6e}")

# ============================================================
# STEP 4: Factorized model J/|S12|
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Factorized model J/|S12|")
print("=" * 70)

features_11 = np.load(os.path.join(PROCESSED_DIR, 'features_v2.npy'))

EPS = 1e-8
usable = (S12_abs > EPS) & (labels_ev > 0)
log_ratio = np.full(N, np.nan)
log_ratio[usable] = np.log10(labels_ev[usable] / S12_abs[usable])

print(f"Usable: {usable.sum()}, set aside: {(~usable).sum()}")
if usable.sum() > 0:
    print(f"log_ratio std: {np.nanstd(log_ratio):.3f} (log10(J) std: {np.nanstd(labels_log[valid]):.3f})")

# Splits
strat_labels = np.zeros(N, dtype=int)
for i, val in enumerate(labels_ev):
    if val < 1e-4: strat_labels[i] = -99
    elif val > 0: strat_labels[i] = int(np.floor(np.log10(val)))
    else: strat_labels[i] = -99

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.zeros((N, 1)), strat_labels))
train_mask = np.zeros(N, dtype=bool)
train_mask[train_idx] = True
test_mask = ~train_mask

# Correlation of log_ratio with cos_4phi (the key diagnostic)
train_usable = train_mask & usable & sig_mask
print(f"\nCorrelation: log_ratio vs geometric features (train, sig, usable):")
print(f"{'Feature':<14} {'r(logJ)':>10} {'r(ratio_S12)':>14} {'Reduction':>10}")
print("-" * 50)
for i, name in enumerate(GEOM_FEAT_NAMES[:8]):  # core 8 features
    x = features_11[train_usable, i]
    y_J = labels_log[train_usable]
    y_R = log_ratio[train_usable]
    v = np.isfinite(y_R)
    r_J, _ = pearsonr(x[v], y_J[v])
    r_R, _ = pearsonr(x[v], y_R[v])
    reduct = (1 - abs(r_R) / max(abs(r_J), 1e-10)) * 100
    print(f"{name:<14} {r_J:>10.4f} {r_R:>14.4f} {reduct:>9.0f}%")

# Key check: cos_4phi signal
x_phi = features_11[train_usable, 4]
y_R_phi = log_ratio[train_usable]
v = np.isfinite(y_R_phi)
r_phi_orig, _ = pearsonr(x_phi[v], labels_log[train_usable][v])
r_phi_ratio, _ = pearsonr(x_phi[v], y_R_phi[v])
pct_retained = abs(r_phi_ratio) / abs(r_phi_orig) * 100
print(f"\ncos_4phi signal retained: {pct_retained:.0f}% (target: < 20%, Gaussian was: 60%)")

# Train factorized RF
train_fact = train_mask & sig_mask & usable
test_fact = test_mask & sig_mask & usable

if train_fact.sum() > 0 and test_fact.sum() > 0:
    X_train = features_11[train_fact]
    y_train = log_ratio[train_fact]
    X_test = features_11[test_fact]
    y_test_log = labels_log[test_fact]
    s12_test = S12_abs[test_fact]

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('rf', RandomForestRegressor(random_state=42))])
    search = RandomizedSearchCV(pipe, RF_PARAMS, n_iter=30, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1,
                                random_state=42, verbose=0)
    search.fit(X_train, y_train)
    log_ratio_pred = search.predict(X_test)
    J_pred = (10 ** log_ratio_pred) * s12_test
    log_J_pred = np.log10(np.maximum(J_pred, 1e-12))

    mae_fact = mean_absolute_error(y_test_log, log_J_pred)
    r2_fact = r2_score(y_test_log, log_J_pred)
    sf = strat_mae(y_test_log, log_J_pred)

    # Baseline
    pipe_b = Pipeline([('scaler', StandardScaler()),
                       ('rf', RandomForestRegressor(random_state=42))])
    search_b = RandomizedSearchCV(pipe_b, RF_PARAMS, n_iter=30, cv=5,
                                  scoring='neg_mean_absolute_error', n_jobs=-1,
                                  random_state=42, verbose=0)
    search_b.fit(features_11[train_mask & sig_mask], labels_log[train_mask & sig_mask])
    log_J_pred_base = search_b.predict(X_test)
    mae_base = mean_absolute_error(y_test_log, log_J_pred_base)
    sb = strat_mae(y_test_log, log_J_pred_base)

    print(f"\n{'Model':<30} {'log MAE':>8}")
    print("-" * 40)
    print(f"{'Baseline (direct RF 11f)':<30} {mae_base:>8.4f}")
    print(f"{'Factorized (S12, RF 11f)':<30} {mae_fact:>8.4f}")

    print(f"\nStratified:")
    for lo, hi in DEC_BINS:
        key = f'[1e{lo},1e{hi})'
        b = sb.get(key, {}).get('log_mae', float('nan'))
        f_val = sf.get(key, {}).get('log_mae', float('nan'))
        n = sb.get(key, {}).get('N', 0)
        print(f"  {key}: baseline={b:.4f}, factorized={f_val:.4f}, N={n}")

    # Near-zero detection
    test_all_mask = test_mask
    X_test_all = features_11[test_all_mask]
    s12_test_all = S12_abs[test_all_mask]
    y_ev_test_all = labels_ev[test_all_mask]
    lr_pred_all = search.predict(X_test_all)
    J_pred_all = (10 ** lr_pred_all) * s12_test_all
    nz_test = y_ev_test_all < 1e-4
    nz_detected = J_pred_all[nz_test] < 1e-4
    print(f"\nNear-zero detection: {nz_detected.sum()}/{nz_test.sum()} predicted J < 1e-4")

    results = {
        'pearson_r_logS12_logJ': float(r_S12),
        'pearson_r_S12_logJ': float(r_S12_lin),
        'cos4phi_pct_retained': float(pct_retained),
        'factorized_log_mae': float(mae_fact),
        'baseline_log_mae': float(mae_base),
        'self_overlap': float(self_overlap),
        'stratified_factorized': sf,
        'stratified_baseline': sb,
    }
else:
    results = {
        'pearson_r_logS12_logJ': float(r_S12),
        'pearson_r_S12_logJ': float(r_S12_lin),
        'self_overlap': float(self_overlap),
        'error': 'insufficient usable structures',
    }

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
v_sig = valid & sig_mask
v_nz = valid & is_near_zero
ax.scatter(np.log10(S12_abs[v_sig]), labels_log[v_sig],
           alpha=0.4, s=15, c='steelblue', label='Significant')
if v_nz.sum() > 0:
    ax.scatter(np.log10(S12_abs[v_nz]), labels_log[v_nz],
               alpha=0.5, s=20, c='red', marker='x', label='Near-zero')
ax.set_xlabel(r'$\log_{10}|S_{12}|$ (xTB HOMO overlap)')
ax.set_ylabel(r'$\log_{10}(J\,/\,\mathrm{eV})$')
ax.set_title(f'xTB HOMO overlap vs coupling (r = {r_S12:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'xtb_S12_analysis.png'), dpi=150)
plt.close()
print("\nSaved: xtb_S12_analysis.png")

with open(os.path.join(RESULTS_DIR, 'xtb_part2_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: results/xtb_part2_results.json")

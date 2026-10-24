"""
v5: Extract Fe-TPP dimers from MD GRO files, run predict.py, validate
against 15 reference CDFT-CI couplings from Huang et al.
"""

import sys, os, glob, json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DIR = '/Users/mbaer/Projects/ML_for_coupling/data/from_bradley/raw'
STRUCT_DIR = os.path.join(DATA_DIR, 'structures')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MD_OUT_DIR = os.path.join(DATA_DIR, 'md_extracted')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

SOLVENT_RESNAMES = {'ACN', 'ACE', 'MeCN', 'CH3CN', 'SOL', 'WAT', 'TIP',
                    'SPC', 'OPC', 'MOH'}

REF_COUPLING_MEV = {
    1: 0.0027, 2: 0.3755, 3: 0.4327, 4: 0.0435,
    5: 3.1157, 6: 3.1919, 7: 5.0096, 8: 0.9932,
    9: 0.1034, 10: 0.0027, 11: 0.0109, 12: 0.2259,
    13: 0.0163, 14: 0.0327, 15: 0.3510,
}


def parse_gro(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    title = lines[0].strip()
    n_atoms = int(lines[1].strip())
    atoms = []
    for line in lines[2:2 + n_atoms]:
        if len(line) < 44:
            continue
        atoms.append({
            'resnum': int(line[0:5]),
            'resname': line[5:10].strip(),
            'atomname': line[10:15].strip(),
            'atomnum': int(line[15:20]),
            'x': float(line[20:28]) * 10,  # nm -> Å
            'y': float(line[28:36]) * 10,
            'z': float(line[36:44]) * 10,
        })
    box_line = lines[2 + n_atoms].split()
    box = np.array([float(x) for x in box_line[:3]]) * 10  # nm -> Å
    return {'title': title, 'n_atoms': n_atoms, 'atoms': atoms, 'box': box}


def filter_solvent(atoms):
    res_groups = defaultdict(list)
    for a in atoms:
        res_groups[a['resnum']].append(a)
    kept = []
    removed = set()
    for resnum, res_atoms in res_groups.items():
        resname = res_atoms[0]['resname'].strip().upper()
        if resname in SOLVENT_RESNAMES:
            removed.add(resname)
            continue
        atomnames = [a['atomname'].strip().upper() for a in res_atoms]
        has_fe = any('FE' in n for n in atomnames)
        n_heavy = sum(1 for n in atomnames if not n.startswith('H'))
        if not has_fe and n_heavy <= 3 and len(res_atoms) <= 6:
            removed.add(resname)
            continue
        kept.extend(res_atoms)
    return kept, removed


def find_tpp_residue_groups(atoms):
    fe_atoms = [a for a in atoms if 'FE' in a['atomname'].upper()]
    if len(fe_atoms) != 2:
        raise ValueError(f"Expected 2 Fe atoms, found {len(fe_atoms)}")
    fe1, fe2 = fe_atoms[0], fe_atoms[1]
    if fe1['resnum'] != fe2['resnum']:
        m1 = [a for a in atoms if a['resnum'] == fe1['resnum']]
        m2 = [a for a in atoms if a['resnum'] == fe2['resnum']]
    else:
        fe1_pos = np.array([fe1['x'], fe1['y'], fe1['z']])
        fe2_pos = np.array([fe2['x'], fe2['y'], fe2['z']])
        all_tpp = [a for a in atoms if a['resnum'] == fe1['resnum']]
        m1, m2 = [], []
        for a in all_tpp:
            pos = np.array([a['x'], a['y'], a['z']])
            if np.linalg.norm(pos - fe1_pos) < np.linalg.norm(pos - fe2_pos):
                m1.append(a)
            else:
                m2.append(a)
    return m1, m2


def atomname_to_symbol(atomname, resname):
    name = atomname.strip().upper()
    if name.startswith('FE'):
        return 'Fe'
    if name in ('NA', 'NB', 'NC', 'ND') and resname.upper() in (
            'HEM', 'POR', 'TPP', 'FEP', 'HEA', 'HEB'):
        return 'N'
    name = name.lstrip('0123456789')
    letters = ''.join(c for c in name if c.isalpha())
    if not letters:
        raise ValueError(f"Cannot determine element from '{atomname}'")
    return letters[0].capitalize()


def unwrap_molecule(atoms, box):
    positions = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    atomnames = [a['atomname'] for a in atoms]
    fe_idx = next(i for i, n in enumerate(atomnames) if 'FE' in n.upper())
    anchor = positions[fe_idx].copy()
    changed = True
    while changed:
        changed = False
        for i in range(len(positions)):
            if i == fe_idx:
                continue
            diff = positions[i] - anchor
            for dim in range(3):
                if abs(diff[dim]) > box[dim] / 2:
                    positions[i, dim] -= np.sign(diff[dim]) * box[dim]
                    changed = True
    for i, a in enumerate(atoms):
        a['x'], a['y'], a['z'] = positions[i]
    return atoms


def kabsch_align(mobile_pos, target_pos):
    mc = mobile_pos - mobile_pos.mean(axis=0)
    tc = target_pos - target_pos.mean(axis=0)
    H = mc.T @ tc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    aligned = (R @ mc.T).T + target_pos.mean(axis=0)
    rmsd = np.sqrt(np.mean(np.sum((aligned - target_pos) ** 2, axis=1)))
    return aligned, rmsd


# ============================================================
# STEP 1: Reference monomer
# ============================================================
print("=" * 70)
print("STEP 1: Reference monomer")
print("=" * 70)

from ase.io import read as ase_read, write as ase_write

ref_path = os.path.join(PROCESSED_DIR, 'reference_monomer.xyz')
if not os.path.exists(ref_path):
    first_xyz = sorted(glob.glob(os.path.join(STRUCT_DIR, '*.xyz')))[0]
    dimer = ase_read(first_xyz)
    n = len(dimer) // 2
    monomer = dimer[:n]
    ase_write(ref_path, monomer)
    print(f"Created reference from {first_xyz}")

ref = ase_read(ref_path)
ref_symbols = ref.get_chemical_symbols()
ref_positions = ref.get_positions()
print(f"Reference monomer: {len(ref)} atoms")
print(f"  Composition: {dict(Counter(ref_symbols))}")
print(f"  Fe: {ref_symbols.count('Fe')}, N: {ref_symbols.count('N')}")

# ============================================================
# STEP 2: Extract TPP from all GRO files
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Extract Fe-TPP from GRO files")
print("=" * 70)

gro_files = sorted(glob.glob(os.path.join(RAW_DIR, '*/eq.gro')),
                   key=lambda p: int(p.split('/')[-2]))
print(f"Found {len(gro_files)} GRO files")

os.makedirs(MD_OUT_DIR, exist_ok=True)
records = []

# Inspect first GRO to understand TPP composition
gro0 = parse_gro(gro_files[0])
kept0, removed0 = filter_solvent(gro0['atoms'])
m1_0, m2_0 = find_tpp_residue_groups(kept0)
sym0 = [atomname_to_symbol(a['atomname'], a['resname']) for a in m1_0]
print(f"Fe-TPP monomer composition: {dict(Counter(sym0))}")
print(f"Atoms per TPP monomer: {len(sym0)}")
print(f"  (Reference Fe-porphine monomer: {len(ref_symbols)} atoms)")
print(f"  → Fe-TPP ≠ Fe-porphine; geometry library uses Fe+4N only (OK)")

# For RMSD validation: extract heavy-atom core (Fe + 4N + ring C)
# Since TPP has extra phenyl C/H, only compare the porphyrin core
# Identify core atoms: Fe and atoms bonded to N (alpha-C positions)
# Simpler: just check Fe count == 1, N count == 4 per monomer

for gro_path in tqdm(gro_files, desc="Extracting"):
    conf_id = int(gro_path.split('/')[-2])
    try:
        gro = parse_gro(gro_path)
        box = gro['box']
        kept, _ = filter_solvent(gro['atoms'])
        m1_atoms, m2_atoms = find_tpp_residue_groups(kept)

        # Unwrap PBC
        m1_atoms = unwrap_molecule(m1_atoms, box)
        m2_atoms = unwrap_molecule(m2_atoms, box)

        # Convert to symbols and positions
        sym1 = [atomname_to_symbol(a['atomname'], a['resname']) for a in m1_atoms]
        pos1 = np.array([[a['x'], a['y'], a['z']] for a in m1_atoms])
        sym2 = [atomname_to_symbol(a['atomname'], a['resname']) for a in m2_atoms]
        pos2 = np.array([[a['x'], a['y'], a['z']] for a in m2_atoms])

        # Validate: Fe==1, N>=4 per monomer (TPP has extra amide N)
        for mon_id, (s, p) in enumerate([(sym1, pos1), (sym2, pos2)], 1):
            fe_c = s.count('Fe')
            n_c = s.count('N')
            if fe_c != 1:
                raise ValueError(f"Conf {conf_id} M{mon_id}: {fe_c} Fe (expected 1)")
            if n_c < 4:
                raise ValueError(f"Conf {conf_id} M{mon_id}: {n_c} N (expected >= 4)")

        # Fe-Fe distance
        fe1_idx = sym1.index('Fe')
        fe2_idx = sym2.index('Fe')
        fe_fe = np.linalg.norm(pos1[fe1_idx] - pos2[fe2_idx])

        # Write dimer XYZ
        n_total = len(sym1) + len(sym2)
        xyz_path = os.path.join(MD_OUT_DIR, f'conf_{conf_id:02d}.xyz')
        with open(xyz_path, 'w') as f:
            f.write(f"{n_total}\n")
            f.write(f"Conf {conf_id} Fe-TPP dimer from {gro_path}\n")
            for s, p in zip(sym1, pos1):
                f.write(f"{s:2s}  {p[0]:12.6f}  {p[1]:12.6f}  {p[2]:12.6f}\n")
            for s, p in zip(sym2, pos2):
                f.write(f"{s:2s}  {p[0]:12.6f}  {p[1]:12.6f}  {p[2]:12.6f}\n")

        records.append({
            'conf_id': conf_id, 'status': 'OK',
            'n_atoms_m1': len(sym1), 'n_atoms_m2': len(sym2),
            'fe_fe_dist_ang': fe_fe, 'xyz_path': xyz_path,
        })
    except Exception as e:
        print(f"FAILED conf {conf_id}: {e}")
        records.append({'conf_id': conf_id, 'status': f'FAILED: {e}'})

summary = pd.DataFrame(records)
summary.to_csv(os.path.join(MD_OUT_DIR, 'extraction_summary.csv'), index=False)
ok = summary[summary['status'] == 'OK']
print(f"\nExtraction: {len(ok)} OK, {len(summary) - len(ok)} failed")
if len(ok) > 0:
    print(f"Fe-Fe distances: mean={ok['fe_fe_dist_ang'].mean():.2f}, "
          f"min={ok['fe_fe_dist_ang'].min():.2f}, max={ok['fe_fe_dist_ang'].max():.2f} Å")

# ============================================================
# STEP 3: Run predict.py on extracted structures
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: ML predictions on extracted structures")
print("=" * 70)

# Need to modify geometry.py's load_dimer_xyz to handle TPP (unequal composition)
# Instead, call dimer_features directly with the arrays
from geometry import dimer_features, monomer_frame, load_dimer_xyz

xyz_files_md = sorted(glob.glob(os.path.join(MD_OUT_DIR, 'conf_*.xyz')))
results = []

for xyz_path in xyz_files_md:
    conf_id = int(os.path.basename(xyz_path).split('_')[1].split('.')[0])
    try:
        # Load the XYZ (can't use load_dimer_xyz — composition mismatch for TPP)
        atoms = ase_read(xyz_path)
        symbols = np.array(atoms.get_chemical_symbols())
        positions = atoms.get_positions()
        N = len(symbols)
        M = N // 2
        m1_sym, m1_pos = symbols[:M], positions[:M]
        m2_sym, m2_pos = symbols[M:], positions[M:]

        # Compute features (geometry.py only needs Fe+4N)
        feats = dimer_features(m1_pos, m1_sym, m2_pos, m2_sym)

        # Predict using the RF pipeline directly (skip Stage 1 SVC which expects 8 features)
        import joblib
        rf_model = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'stage2_rf_v2.pkl'))
        X = feats[:11].reshape(1, -1)  # Use first 11 features (v2 model)
        log_J_pred = rf_model.predict(X)[0]
        J_ev = 10 ** log_J_pred
        J_mev = J_ev * 1000

        ref = REF_COUPLING_MEV.get(conf_id)
        log_ratio = np.log10(J_mev / ref) if ref and J_mev > 0 and ref > 0 else None

        fe_fe = np.sqrt(feats[0]**2 + feats[1]**2 + feats[2]**2)

        results.append({
            'conf_id': conf_id, 'J_pred_mev': J_mev, 'J_ref_mev': ref,
            'log10_ratio': log_ratio,
            'abs_log_ratio': abs(log_ratio) if log_ratio is not None else None,
            'dz_ang': feats[0], 'dx_ang': feats[1], 'dy_ang': feats[2],
            'cos_theta': feats[3], 'cos_4phi': feats[4], 'd_lateral': feats[8],
            'fe_fe_dist': fe_fe, 'status': 'OK',
        })
        lr_str = f"  log_ratio={log_ratio:+.2f}" if log_ratio is not None else ""
        print(f"Conf {conf_id:2d}: pred={J_mev:8.4f} meV  ref={ref:8.4f} meV{lr_str}  "
              f"dz={feats[0]:.2f}  FeFe={fe_fe:.2f} Å")
    except Exception as e:
        print(f"Conf {conf_id}: FAILED — {e}")
        results.append({'conf_id': conf_id, 'status': f'FAILED: {e}'})

df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, 'md_validation.csv'), index=False)

ok_pred = df[df['status'] == 'OK'].copy()
print(f"\n{'='*60}")
print("EXTERNAL VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"Configurations evaluated: {len(ok_pred)}")
if 'abs_log_ratio' in ok_pred.columns and ok_pred['abs_log_ratio'].notna().any():
    valid = ok_pred['abs_log_ratio'].notna()
    print(f"Log MAE: {ok_pred.loc[valid, 'abs_log_ratio'].mean():.3f}")
    above_01 = ok_pred['J_ref_mev'] > 0.1
    if (valid & above_01).any():
        print(f"Log MAE (ref > 0.1 meV): {ok_pred.loc[valid & above_01, 'abs_log_ratio'].mean():.3f}")
    print(f"Max |log ratio|: {ok_pred.loc[valid, 'abs_log_ratio'].max():.3f} "
          f"(conf {ok_pred.loc[ok_pred['abs_log_ratio'].idxmax(), 'conf_id']})")

print(f"\nFull results table:")
cols = ['conf_id', 'J_ref_mev', 'J_pred_mev', 'log10_ratio', 'dz_ang', 'd_lateral', 'cos_4phi']
print(ok_pred[cols].to_string(index=False))

# ============================================================
# STEP 4: Validation figure
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Validation figure")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ok_v = ok_pred.dropna(subset=['J_ref_mev', 'J_pred_mev'])

# Panel A: Parity
ax = axes[0]
colors = []
for j in ok_v['J_ref_mev']:
    if j < 0.01: colors.append('#d62728')
    elif j < 0.1: colors.append('#ff7f0e')
    elif j < 1.0: colors.append('#2ca02c')
    else: colors.append('#1f77b4')

ax.scatter(ok_v['J_ref_mev'], ok_v['J_pred_mev'],
           c=colors, s=80, zorder=3, edgecolors='k', linewidths=0.5)
for _, row in ok_v.iterrows():
    ax.annotate(f"{int(row['conf_id'])}", (row['J_ref_mev'], row['J_pred_mev']),
                xytext=(4, 4), textcoords='offset points', fontsize=8)

lims = [1e-4, 50]
ax.plot(lims, lims, 'k-', linewidth=1)
ax.fill_between(lims, [x / 3 for x in lims], [x * 3 for x in lims],
                alpha=0.15, color='grey')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(r'Reference $\Gamma$ (meV) [CDFT-CI/SRSH-PCM]')
ax.set_ylabel(r'Predicted $\Gamma$ (meV) [ML model]')

log_mae = ok_v['abs_log_ratio'].mean() if ok_v['abs_log_ratio'].notna().any() else float('nan')
ax.text(0.05, 0.95, f'Log MAE = {log_mae:.2f}\n(N = {len(ok_v)})',
        transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=8, label=r'$\Gamma < 0.01$ meV'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=8, label=r'$0.01 \leq \Gamma < 0.1$ meV'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=8, label=r'$0.1 \leq \Gamma < 1$ meV'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label=r'$\Gamma \geq 1$ meV'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

# Panel B: Log ratio vs Fe-Fe distance
ax2 = axes[1]
ok_v2 = ok_v.dropna(subset=['log10_ratio', 'fe_fe_dist'])
ax2.scatter(ok_v2['fe_fe_dist'], ok_v2['log10_ratio'],
            c=[colors[i] for i in ok_v2.index - ok_v2.index[0]],
            s=80, edgecolors='k', linewidths=0.5)
for _, row in ok_v2.iterrows():
    ax2.annotate(f"{int(row['conf_id'])}", (row['fe_fe_dist'], row['log10_ratio']),
                 xytext=(4, 4), textcoords='offset points', fontsize=8)
ax2.axhline(0, color='k', linewidth=1)
ax2.axhline(0.5, color='grey', linewidth=0.8, linestyle='--')
ax2.axhline(-0.5, color='grey', linewidth=0.8, linestyle='--')
ax2.set_xlabel(r'Fe–Fe distance (Å)')
ax2.set_ylabel(r'$\log_{10}(\Gamma_\mathrm{pred}/\Gamma_\mathrm{ref})$')

plt.tight_layout()
for ext in ['png', 'pdf']:
    plt.savefig(os.path.join(FIG_DIR, f'pub_fig5_md_validation.{ext}'))
plt.close()
print("Saved: pub_fig5_md_validation.png/pdf")

# ============================================================
# STEP 5: Domain analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Domain analysis")
print("=" * 70)

features_train = np.load(os.path.join(PROCESSED_DIR, 'features_v2.npy'))
labels_ev_train = np.load(os.path.join(PROCESSED_DIR, 'labels_ev.npy'))
meta_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'metadata.csv'))

fe_fe_train = np.sqrt(features_train[:, 0]**2 + features_train[:, 1]**2 + features_train[:, 2]**2)
fe_fe_md = ok_pred['fe_fe_dist'].values

print(f"Training set Fe-Fe range: [{fe_fe_train.min():.2f}, {fe_fe_train.max():.2f}] Å")
print(f"MD structures Fe-Fe range: [{fe_fe_md.min():.2f}, {fe_fe_md.max():.2f}] Å")

# Check if MD structures are in-distribution
in_dist = (fe_fe_md >= fe_fe_train.min()) & (fe_fe_md <= fe_fe_train.max())
print(f"MD structures in training Fe-Fe range: {in_dist.sum()}/{len(fe_fe_md)}")

# Feature range comparison
print(f"\nFeature range comparison (training vs MD):")
feat_names = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi',
              'cos_4psi', 'sin_4psi', 'd_lateral', 'cos_4alpha', 'sin_4alpha']
md_feats = ok_pred[['dz_ang', 'dx_ang', 'dy_ang', 'cos_theta', 'cos_4phi']].values
for i, name in enumerate(feat_names[:5]):
    tr_min, tr_max = features_train[:, i].min(), features_train[:, i].max()
    if i < md_feats.shape[1]:
        md_min, md_max = md_feats[:, i].min(), md_feats[:, i].max()
        ood = "OOD!" if md_min < tr_min or md_max > tr_max else "OK"
        print(f"  {name:<12}: train=[{tr_min:.2f}, {tr_max:.2f}], MD=[{md_min:.2f}, {md_max:.2f}] {ood}")

if log_mae > 0.5:
    print(f"\nLog MAE = {log_mae:.2f} > 0.5 → retraining may help")
else:
    print(f"\nLog MAE = {log_mae:.2f} ≤ 0.5 → no retraining needed")

print(f"\nDone. Results in: results/md_validation.csv")

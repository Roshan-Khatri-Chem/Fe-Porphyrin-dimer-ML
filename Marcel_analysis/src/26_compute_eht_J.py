"""Compute EHT-FO coupling for all 704 systematic + 111 Slater + 15 MD structures."""
import sys, os, glob, time, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
ROOT = os.path.join(THIS, '..')
DATA = os.path.join(ROOT, 'data', 'processed')
RESULTS = os.path.join(ROOT, 'results')

from eht import EHTCoupling
from geometry import load_dimer_xyz

# Reference monomer
with open(os.path.join(DATA, 'reference_monomer.xyz')) as f:
    n = int(f.readline().strip()); f.readline()
    atoms = [(p[0], [float(x) for x in p[1:4]]) for p in [f.readline().split() for _ in range(n)]]

eht = EHTCoupling(atoms, basis='sto-3g', target_a2u=True)
print(f"Reference orbital: idx={eht.homo_idx}  E={eht.ref_eigvals[eht.homo_idx]*27.2114:+.3f} eV")
print(f"  C+N={eht.homo_C_N_frac*100:.1f}%, Fe={eht.homo_Fe_frac*100:.1f}%")
print(f"  (strict HOMO would be idx={eht.homo_idx_strict})")

def compute_set(name, file_list, label_dict=None):
    print(f"\n{name}: {len(file_list)} structures")
    rows = []
    t0 = time.time()
    for fn in tqdm(file_list, desc=name):
        stem = os.path.splitext(os.path.basename(fn))[0]
        try:
            d = load_dimer_xyz(fn)
            J, S12, H12 = eht.compute(d['monomer1']['positions'], d['monomer1']['symbols'],
                                      d['monomer2']['positions'], d['monomer2']['symbols'])
            rows.append({'name': stem, 'J_eht_eV': J, 'S12_eht': S12,
                         'H12_eht_eV': H12,
                         'J_truth_eV': label_dict.get(stem) if label_dict else None})
        except Exception as e:
            rows.append({'name': stem, 'J_eht_eV': np.nan, 'S12_eht': np.nan,
                         'H12_eht_eV': np.nan,
                         'J_truth_eV': label_dict.get(stem) if label_dict else None})
    print(f"  {len(rows)} done in {time.time()-t0:.0f}s")
    return rows


# ----- 704 systematic -----
coup_csv = pd.read_csv(os.path.join(ROOT, 'data', 'coupling.csv'))
coup_dict = {row['Filename'].replace('.out',''): row['Coupling_eV']
             for _, row in coup_csv.iterrows()}
files_704 = sorted(glob.glob(os.path.join(ROOT, 'data', 'structures', '*.xyz')))
rows_704 = compute_set('704 systematic', files_704, coup_dict)

# ----- Slater 111 -----
slater_records = json.load(open(os.path.join(DATA, 'slater_records.json')))
slater_dict = {rec['name']: rec['J_eV'] for rec in slater_records}
files_slater = [os.path.join(ROOT, 'data', 'structures_slater', rec['name']+'.xyz')
                for rec in slater_records]
rows_slater = compute_set('Slater 111 MD', files_slater, slater_dict)

# ----- MD validation 15 -----
GAMMA_STAR_MEV = {1:39.9246,2:0.1905,3:5.4858,4:5.5266,5:42.1205,6:7.6328,7:22.4902,
                  8:30.6074,9:0.0000,10:0.0136,11:0.5987,12:0.3755,13:0.0190,
                  14:10.1553,15:0.0000}
md_files = sorted(glob.glob(os.path.join(ROOT, 'data', 'md_extracted', '*.xyz')))
md_dict = {f'conf_{k:02d}': v/1000.0 for k, v in GAMMA_STAR_MEV.items()}
rows_md = compute_set('MD validation 15', md_files, md_dict)

# ----- Save -----
def save(prefix, rows):
    arr = {}
    for k in rows[0].keys():
        if k == 'name':
            arr[k] = np.array([r[k] for r in rows])
        else:
            # Replace None with NaN to keep dtype float
            arr[k] = np.array([(np.nan if r[k] is None else r[k]) for r in rows], dtype=float)
    np.savez(os.path.join(DATA, f'{prefix}.npz'), **arr)
    return arr
arr_704 = save('eht_J_704', rows_704)
arr_slater = save('eht_J_slater', rows_slater)
arr_md = save('eht_J_md', rows_md)

# ----- Correlation analysis -----
print("\n" + "="*70)
print("Correlation: log|J_EHT_FO| vs log|J_CDFT|")
print("="*70)

for label, arr in [('704 systematic', arr_704),
                   ('Slater 111 MD', arr_slater),
                   ('MD validation 15', arr_md)]:
    valid = (np.isfinite(arr['J_eht_eV']) & (np.abs(arr['J_eht_eV']) > 1e-12)
             & np.isfinite(arr['J_truth_eV']) & (arr['J_truth_eV'] > 1e-6))
    if valid.sum() < 3:
        print(f"  {label}: only {valid.sum()} valid -- skip")
        continue
    log_eht = np.log10(np.abs(arr['J_eht_eV'][valid]))
    log_t   = np.log10(arr['J_truth_eV'][valid])
    r, _ = pearsonr(log_eht, log_t)
    print(f"  {label}: n={valid.sum()}  r = {r:+.3f}  log10|J| MAE = {np.mean(np.abs(log_eht - log_t)):.3f}")

# Per-conf MD predictions vs truth
print(f"\nMD validation per-conformation:")
print(f"  {'name':<10} {'J_truth (meV)':>14} {'J_EHT (meV)':>14}")
for r in rows_md:
    t = r['J_truth_eV']*1000 if r['J_truth_eV'] is not None else float('nan')
    e = abs(r['J_eht_eV'])*1000 if np.isfinite(r['J_eht_eV']) else float('nan')
    print(f"  {r['name']:<10} {t:14.3f} {e:14.3f}")

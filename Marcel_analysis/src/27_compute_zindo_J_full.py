"""Run ZINDO/S on all 704 + 111 + 15 dimers; compute Lowdin-diabatic J;
save and analyze with focus on Slater 111 (the realistic MD test set)."""
import os, sys, glob, time, shutil, re, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
ROOT = os.path.join(THIS, '..')
DATA = os.path.join(ROOT, 'data', 'processed')
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

from zindo_orca import run_zindo, parse_zindo_output, element_decomp
from geometry import load_dimer_xyz


def m12_weights(c, labels, npm):
    w1 = w2 = 0
    for lab, ci in zip(labels, c):
        m = re.match(r'(\d+)([A-Z][a-z]?)', lab.split()[0])
        ai = int(m.group(1))
        if ai < npm: w1 += ci**2
        else: w2 += ci**2
    return w1, w2


def find_pi_pair_J_lowdin(parsed, npm, cn_thresh=0.95, search_window=8):
    """Find the highest occupied pair of pi orbitals (>=95% C+N) and compute J
    via Lowdin diabatic formula that accounts for site-energy localization.

    Returns (J_eV, idx_high, idx_low, candidate_metadata)."""
    e = parsed['eigvals_eV']
    mo = parsed['mocoeffs']
    labels = parsed['ao_labels']
    homo = parsed['homo_idx']

    cands = []
    for i in range(homo, max(homo - search_window, -1), -1):
        if i < 0 or i >= mo.shape[1]:
            continue
        elem = element_decomp(mo[:, i], labels)
        tot = sum(elem.values())
        if tot < 0.1: continue
        cn = (elem.get('C', 0) + elem.get('N', 0)) / tot
        if cn < cn_thresh: continue
        w1, w2 = m12_weights(mo[:, i], labels, npm)
        s = w1 + w2
        if s < 0.5: continue
        cands.append({'idx': i, 'E': e[i], 'cn': cn, 'w1': w1/s, 'w2': w2/s})
    if len(cands) < 2:
        return np.nan, -1, -1, cands
    cands.sort(key=lambda c: -c['idx'])
    for j in range(len(cands) - 1):
        ch = cands[j]; cl = cands[j+1]
        if abs(ch['idx'] - cl['idx']) > 2: continue
        eps1 = ch['w1'] * ch['E'] + cl['w1'] * cl['E']
        eps2 = ch['w2'] * ch['E'] + cl['w2'] * cl['E']
        delta = (ch['E'] - cl['E']) / 2
        site = (eps1 - eps2) / 2
        if abs(delta) <= abs(site):
            continue  # not a real pi pair
        J = float(np.sqrt(delta ** 2 - site ** 2))
        return J, ch['idx'], cl['idx'], cands[:4]
    return np.nan, -1, -1, cands


def compute_zindo_for_set(name, file_list, label_dict=None):
    print(f"\n=== {name}: {len(file_list)} dimers ===")
    rows = []
    t0 = time.time()
    for fn in tqdm(file_list, desc=name):
        stem = os.path.splitext(os.path.basename(fn))[0]
        rec = {'name': stem, 'J_zindo_eV': np.nan, 'idx_high': -1, 'idx_low': -1,
               'J_truth_eV': label_dict.get(stem) if label_dict else None,
               'wall_sec': np.nan}
        try:
            d = load_dimer_xyz(fn)
            npm = len(d['monomer1']['symbols'])
            atoms = [(s, p.tolist()) for s, p in zip(
                list(d['monomer1']['symbols']) + list(d['monomer2']['symbols']),
                list(d['monomer1']['positions']) + list(d['monomer2']['positions']))]
            t_start = time.time()
            out_path, work_dir, _ = run_zindo(atoms)
            parsed = parse_zindo_output(out_path)
            rec['wall_sec'] = time.time() - t_start
            J, ih, il, cands = find_pi_pair_J_lowdin(parsed, npm)
            rec['J_zindo_eV'] = J
            rec['idx_high'] = ih
            rec['idx_low'] = il
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            tqdm.write(f"  failed {stem}: {type(e).__name__}: {e}")
        rows.append(rec)
    print(f"  done in {time.time()-t0:.0f}s, mean per-structure {np.mean([r['wall_sec'] for r in rows if not np.isnan(r['wall_sec'])]):.1f}s")
    return rows


# ----- 704 systematic -----
coup_csv = pd.read_csv(os.path.join(ROOT, 'data', 'coupling.csv'))
coup_dict = {row['Filename'].replace('.out', ''): row['Coupling_eV']
             for _, row in coup_csv.iterrows()}
files_704 = sorted(glob.glob(os.path.join(ROOT, 'data', 'structures', '*.xyz')))
rows_704 = compute_zindo_for_set('704 systematic', files_704, coup_dict)

# ----- Slater 111 MD -----
slater_records = json.load(open(os.path.join(DATA, 'slater_records.json')))
slater_dict = {rec['name']: rec['J_eV'] for rec in slater_records}
files_slater = [os.path.join(ROOT, 'data', 'structures_slater', rec['name']+'.xyz')
                for rec in slater_records]
rows_slater = compute_zindo_for_set('Slater 111 MD', files_slater, slater_dict)

# ----- MD validation 15 -----
GAMMA_STAR_MEV = {1:39.9246,2:0.1905,3:5.4858,4:5.5266,5:42.1205,6:7.6328,7:22.4902,
                  8:30.6074,9:0.0000,10:0.0136,11:0.5987,12:0.3755,13:0.0190,
                  14:10.1553,15:0.0000}
md_dict = {f'conf_{k:02d}': v / 1000.0 for k, v in GAMMA_STAR_MEV.items()}
md_files = sorted(glob.glob(os.path.join(ROOT, 'data', 'md_extracted', '*.xyz')))
rows_md = compute_zindo_for_set('MD validation 15', md_files, md_dict)


def save(prefix, rows):
    arr = {}
    for k in rows[0].keys():
        if k == 'name':
            arr[k] = np.array([r[k] for r in rows])
        else:
            arr[k] = np.array([(np.nan if r[k] is None else r[k]) for r in rows], dtype=float)
    np.savez(os.path.join(DATA, f'{prefix}.npz'), **arr)
    return arr


arr_704 = save('zindo_J_704', rows_704)
arr_slater = save('zindo_J_slater', rows_slater)
arr_md = save('zindo_J_md', rows_md)

# ----- Analysis -----
print('\n' + '=' * 78)
print('ZINDO/S DIMER-LEVEL J (Lowdin diabatic) — FINAL EVALUATION')
print('=' * 78)

for label, arr in [('704 systematic',  arr_704),
                   ('Slater 111 MD',   arr_slater),
                   ('MD validation 15', arr_md)]:
    Jz = arr['J_zindo_eV']
    Jt = arr['J_truth_eV']
    valid = (np.isfinite(Jz) & (Jz > 0)
             & np.isfinite(Jt) & (Jt > 1e-6))
    print(f"\n  {label}:  n_total={len(Jz)}  n_zindo_valid={(np.isfinite(Jz) & (Jz > 0)).sum()}"
          f"  n_for_corr={valid.sum()}")
    if valid.sum() < 3:
        print('    too few valid for correlation')
        continue
    log_z = np.log10(Jz[valid])
    log_t = np.log10(Jt[valid])
    r, _ = pearsonr(log_z, log_t)
    mae = float(np.mean(np.abs(log_z - log_t)))
    print(f"    Pearson r(log|J_ZINDO|, log|J_CDFT|) = {r:+.4f}")
    print(f"    log10|J| MAE                         = {mae:.4f}")

# Compare on Slater 111 to monomer ansatz baselines
print('\n' + '-' * 70)
print('Comparison on Slater 111 MD (the realistic test set):')
print('-' * 70)
slater_S12_b3lyp = np.load(os.path.join(DATA, 'slater_S12_b3lyp.npy'))
srsh = np.load(os.path.join(DATA, 's12_srsh_cation.npz'), allow_pickle=True)
J_truth_arr = arr_slater['J_truth_eV']
mask = (J_truth_arr > 1e-6) & np.isfinite(J_truth_arr)
log_t = np.log10(J_truth_arr[mask])

# B3LYP
m_b = mask & np.isfinite(slater_S12_b3lyp)
log_b = np.log10(np.abs(slater_S12_b3lyp[m_b]) + 1e-12)
log_t_b = np.log10(J_truth_arr[m_b])
r_b, _ = pearsonr(log_b, log_t_b)
print(f"  B3LYP S_12 monomer ansatz:    r = {r_b:.4f}  (n={m_b.sum()})")

# SRSH
s12_srsh_slater = srsh['s12_slater']
m_s = mask & np.isfinite(s12_srsh_slater)
log_s = np.log10(np.abs(s12_srsh_slater[m_s]) + 1e-12)
log_t_s = np.log10(J_truth_arr[m_s])
r_s, _ = pearsonr(log_s, log_t_s)
print(f"  SRSH S_12 monomer ansatz:     r = {r_s:.4f}  (n={m_s.sum()})")

# ZINDO
Jz = arr_slater['J_zindo_eV']
m_z = mask & np.isfinite(Jz) & (Jz > 0)
log_z = np.log10(Jz[m_z])
log_t_z = np.log10(J_truth_arr[m_z])
r_z, _ = pearsonr(log_z, log_t_z)
mae_z = float(np.mean(np.abs(log_z - log_t_z)))
print(f"  ZINDO Lowdin diabatic J:      r = {r_z:.4f}  (n={m_z.sum()})  log MAE = {mae_z:.3f}")

# Save summary
out = {
    'slater_111': {
        'b3lyp_r': float(r_b), 'b3lyp_n': int(m_b.sum()),
        'srsh_r':  float(r_s), 'srsh_n':  int(m_s.sum()),
        'zindo_r': float(r_z), 'zindo_n': int(m_z.sum()),
        'zindo_log_mae': mae_z,
    }
}
with open(os.path.join(RESULTS, 'zindo_validation.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: results/zindo_validation.json, data/processed/zindo_J_*.npz")

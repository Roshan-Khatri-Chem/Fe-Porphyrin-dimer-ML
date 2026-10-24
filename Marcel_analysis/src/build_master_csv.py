"""Consolidate all data sources into a single master_dataset.csv with one row per
structure. This is the central data file for the notebooks-based handoff."""
import os, json
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(ROOT, 'data', 'processed')

FEATURE_NAMES = ['dz', 'dx', 'dy', 'cos_theta',
                 'cos_4phi', 'sin_4phi', 'cos_4psi', 'sin_4psi',
                 'd_lateral', 'cos_4alpha', 'sin_4alpha']

# ----- 704 systematic -----
meta = pd.read_csv(f'{DATA}/metadata.csv')
features_v2 = np.load(f'{DATA}/features_v2.npy')
labels_ev = np.load(f'{DATA}/labels_ev.npy')
s12_b3lyp = np.load(f'{DATA}/s12_molden.npy')

# Map SRSH cation S_12 from npz (709 names) into the 704 metadata order
srsh = np.load(f'{DATA}/s12_srsh_cation.npz', allow_pickle=True)
srsh_singlet = np.load(f'{DATA}/s12_srsh_singlet.npz', allow_pickle=True)
srsh_cation_map = dict(zip(srsh['names_704'], srsh['s12_704']))
srsh_singlet_map = dict(zip(srsh_singlet['names_704'], srsh_singlet['s12_704']))

eht = np.load(f'{DATA}/eht_J_704.npz', allow_pickle=True)
eht_map = dict(zip(eht['name'], eht['J_eht_eV']))

zindo = np.load(f'{DATA}/zindo_J_704.npz', allow_pickle=True)
zindo_map = dict(zip(zindo['name'], zindo['J_zindo_eV']))

dftb_smoke = np.load(f'{DATA}/dftb_J_30_smoke.npz', allow_pickle=True)
dftb_map = dict(zip(dftb_smoke['name'], dftb_smoke['J_dftb_eV']))

rows_704 = []
for i, row in meta.iterrows():
    name = row['filename'].replace('.out', '').replace('.xyz', '')
    rec = {
        'name': name,
        'dataset': '704_systematic',
        'split': row['split'],
        'J_cdft_eV': float(row['coupling_eV']),
    }
    for j, fname in enumerate(FEATURE_NAMES):
        rec[fname] = float(features_v2[i, j])
    rec['S12_b3lyp'] = float(s12_b3lyp[i])
    rec['S12_srsh_singlet'] = float(srsh_singlet_map.get(name, np.nan))
    rec['S12_srsh_cation'] = float(srsh_cation_map.get(name, np.nan))
    rec['S12_qchem_truth'] = np.nan  # only available for Slater set
    rec['J_eht_eV'] = float(eht_map.get(name, np.nan))
    rec['J_zindo_eV'] = float(zindo_map.get(name, np.nan))
    rec['J_dftb_eV'] = float(dftb_map.get(name, np.nan))
    rows_704.append(rec)

# ----- 111 Slater -----
slater_recs = json.load(open(f'{DATA}/slater_records.json'))
features_slater = np.load(f'{DATA}/features_slater.npy')
slater_S12_qchem = np.load(f'{DATA}/slater_S12_qchem.npy')
slater_S12_b3lyp = np.load(f'{DATA}/slater_S12_b3lyp.npy')

eht_s = np.load(f'{DATA}/eht_J_slater.npz', allow_pickle=True)
eht_s_map = dict(zip(eht_s['name'], eht_s['J_eht_eV']))
zindo_s = np.load(f'{DATA}/zindo_J_slater.npz', allow_pickle=True)
zindo_s_map = dict(zip(zindo_s['name'], zindo_s['J_zindo_eV']))

# All 111 Slater are training data (none held out separately in master CSV)
rng = np.random.default_rng(42)
slater_split = ['train'] * 111
test_idx = rng.choice(111, size=23, replace=False)
for ti in test_idx:
    slater_split[int(ti)] = 'test'

rows_slater = []
for i, rec in enumerate(slater_recs):
    name = rec['name']
    r = {
        'name': name,
        'dataset': 'slater_111_md',
        'split': slater_split[i],
        'J_cdft_eV': float(rec['J_eV']),
    }
    for j, fname in enumerate(FEATURE_NAMES):
        r[fname] = float(features_slater[i, j])
    r['S12_b3lyp'] = float(slater_S12_b3lyp[i])
    r['S12_srsh_singlet'] = float(srsh_singlet['s12_slater'][i])
    r['S12_srsh_cation'] = float(srsh['s12_slater'][i])
    r['S12_qchem_truth'] = float(slater_S12_qchem[i])
    r['J_eht_eV'] = float(eht_s_map.get(name, np.nan))
    r['J_zindo_eV'] = float(zindo_s_map.get(name, np.nan))
    r['J_dftb_eV'] = np.nan
    rows_slater.append(r)

# ----- 15 MD validation -----
features_md = np.load(f'{DATA}/features_md.npy')
GAMMA_STAR_MEV = {1:39.9246, 2:0.1905, 3:5.4858, 4:5.5266, 5:42.1205, 6:7.6328,
                  7:22.4902, 8:30.6074, 9:0.0000, 10:0.0136, 11:0.5987, 12:0.3755,
                  13:0.0190, 14:10.1553, 15:0.0000}

eht_md = np.load(f'{DATA}/eht_J_md.npz', allow_pickle=True)
eht_md_map = dict(zip(eht_md['name'], eht_md['J_eht_eV']))
zindo_md = np.load(f'{DATA}/zindo_J_md.npz', allow_pickle=True)
zindo_md_map = dict(zip(zindo_md['name'], zindo_md['J_zindo_eV']))

rows_md = []
for i in range(15):
    name = f'conf_{i+1:02d}'
    J_truth = GAMMA_STAR_MEV[i+1] / 1000.0  # mV to V (eV)
    r = {
        'name': name,
        'dataset': 'md_validation',
        'split': 'validation',
        'J_cdft_eV': J_truth,
    }
    for j, fname in enumerate(FEATURE_NAMES):
        r[fname] = float(features_md[i, j])
    r['S12_b3lyp'] = np.nan
    r['S12_srsh_singlet'] = float(srsh_singlet['s12_md'][i])
    r['S12_srsh_cation'] = float(srsh['s12_md'][i])
    r['S12_qchem_truth'] = np.nan
    r['J_eht_eV'] = float(eht_md_map.get(name, np.nan))
    r['J_zindo_eV'] = float(zindo_md_map.get(name, np.nan))
    r['J_dftb_eV'] = np.nan
    rows_md.append(r)

# Combine
df = pd.DataFrame(rows_704 + rows_slater + rows_md)

# Add log10_J for convenience
df['log10_J_cdft'] = np.log10(np.maximum(df['J_cdft_eV'].abs(), 1e-12))
df['log10_J_cdft'] = df['log10_J_cdft'].where(df['J_cdft_eV'].abs() > 1e-12, np.nan)

# Order columns
col_order = ['name', 'dataset', 'split',
             'J_cdft_eV', 'log10_J_cdft'] + FEATURE_NAMES + \
            ['S12_b3lyp', 'S12_srsh_singlet', 'S12_srsh_cation', 'S12_qchem_truth',
             'J_eht_eV', 'J_zindo_eV', 'J_dftb_eV']
df = df[col_order]

out_path = f'{DATA}/master_dataset.csv'
df.to_csv(out_path, index=False, float_format='%.6e')
print(f'Wrote {out_path}: {len(df)} rows, {len(df.columns)} columns')
print(f'\nDataset counts:')
print(df['dataset'].value_counts())
print(f'\nSplit counts (within each dataset):')
print(df.groupby(['dataset', 'split']).size())
print(f'\nJ_cdft_eV stats:')
print(df['J_cdft_eV'].describe())
print(f'\nFirst row:')
print(df.iloc[0])

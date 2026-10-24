"""
Recompute S_12 for all training + MD validation structures using the SRSH-PCM
HOMO from Slater's monomer fchk files (instead of the B3LYP molden HOMO),
with proper Wigner-D rotation handling for arbitrarily oriented monomers.

Outputs:
  data/processed/homo_coeffs_srsh_singlet_sph.npy
  data/processed/homo_coeffs_srsh_cation_sph.npy
  data/processed/s12_srsh_singlet.npz   -- 815 train + 15 MD
  data/processed/s12_srsh_cation.npz
  results/srsh_s12_validation.json      -- correlation with Slater ground truth
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(THIS, '..'))
from fchk_parser import (parse_fchk, build_pyscf_basis, reorder_mo_coeffs_to_pyscf,
                         normalize_to_pyscf_basis, PERIODIC)
from srsh_s12 import S12Calculator, load_srsh_homo_sph
from geometry import load_dimer_xyz

ROOT = os.path.join(THIS, '..')
DATA = os.path.join(ROOT, 'data', 'processed')
STRUCT = os.path.join(ROOT, 'data', 'structures')
SLATER_ADD = '/Users/mbaer/Projects/ML_for_coupling/data/from_slater/additional'
SLATER_XYZ = os.path.join(ROOT, 'data', 'structures_slater')
RESULTS = os.path.join(ROOT, 'results')
MD_DIR = os.path.join(ROOT, 'data', 'md_extracted')

# ----------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Load SRSH-PCM monomer HOMOs (singlet + cation alpha)")
print("=" * 70)

singlet_path = os.path.join(SLATER_ADD, 'fe_mon_fchk', 'fe2_sp.in.fchk')
cation_path  = os.path.join(SLATER_ADD, 'fe_mon_fchk', 'fe3_sp.in.fchk')

c_singlet, basis, atom_list_ref, fchk_singlet = load_srsh_homo_sph(
    singlet_path, normalize_to_pyscf_basis, reorder_mo_coeffs_to_pyscf,
    build_pyscf_basis, parse_fchk, PERIODIC)
c_cation, _, _, fchk_cation = load_srsh_homo_sph(
    cation_path, normalize_to_pyscf_basis, reorder_mo_coeffs_to_pyscf,
    build_pyscf_basis, parse_fchk, PERIODIC)

# Verify they are the same orbital (sign may differ)
print(f"  Singlet HOMO    E = {fchk_singlet['alpha_e'][fchk_singlet['nalpha']-1]*27.2114:+.3f} eV")
print(f"  Cation alpha SOMO E = {fchk_cation['alpha_e'][fchk_cation['nalpha']-1]*27.2114:+.3f} eV")

# Save sph coefficients
np.save(os.path.join(DATA, 'homo_coeffs_srsh_singlet_sph.npy'), c_singlet)
np.save(os.path.join(DATA, 'homo_coeffs_srsh_cation_sph.npy'), c_cation)
print(f"  Saved homo_coeffs_srsh_*_sph.npy (n_ao = {len(c_singlet)})")

# Build calculators
calc_singlet = S12Calculator(basis, atom_list_ref, c_singlet)
calc_cation  = S12Calculator(basis, atom_list_ref, c_cation)

# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Validate S_12 ansatz against Slater CDFT diabatic |S_12|")
print("=" * 70)

slater_records = json.load(open(os.path.join(DATA, 'slater_records.json')))
slater_S12_qchem = np.load(os.path.join(DATA, 'slater_S12_qchem.npy'))   # ground-truth
slater_S12_b3lyp = np.load(os.path.join(DATA, 'slater_S12_b3lyp.npy'))   # B3LYP ansatz baseline
print(f"  {len(slater_records)} Slater structures, "
      f"|S12_qchem| range [{np.min(np.abs(slater_S12_qchem)):.2e}, "
      f"{np.max(np.abs(slater_S12_qchem)):.2e}]")

slater_s12_singlet = np.full(len(slater_records), np.nan)
slater_s12_cation  = np.full(len(slater_records), np.nan)
for i, rec in enumerate(tqdm(slater_records, desc="Slater 111")):
    xyz_path = os.path.join(SLATER_XYZ, rec['name'] + '.xyz')
    try:
        d = load_dimer_xyz(xyz_path)
        slater_s12_singlet[i] = calc_singlet.compute(
            d['monomer1']['positions'], d['monomer1']['symbols'],
            d['monomer2']['positions'], d['monomer2']['symbols'])
        slater_s12_cation[i] = calc_cation.compute(
            d['monomer1']['positions'], d['monomer1']['symbols'],
            d['monomer2']['positions'], d['monomer2']['symbols'])
    except Exception as e:
        tqdm.write(f"  failed {rec['name']}: {e}")

valid = (np.isfinite(slater_s12_singlet) & np.isfinite(slater_S12_qchem)
         & (np.abs(slater_S12_qchem) > 1e-10))
print(f"  Valid pairs: {valid.sum()}/{len(slater_records)}")
log_qchem  = np.log10(np.abs(slater_S12_qchem[valid]))
log_b3lyp  = np.log10(np.abs(slater_S12_b3lyp[valid]) + 1e-12)
log_srsh_s = np.log10(np.abs(slater_s12_singlet[valid]) + 1e-12)
log_srsh_c = np.log10(np.abs(slater_s12_cation[valid]) + 1e-12)

r_b3lyp,  _ = pearsonr(log_b3lyp,  log_qchem)
r_srsh_s, _ = pearsonr(log_srsh_s, log_qchem)
r_srsh_c, _ = pearsonr(log_srsh_c, log_qchem)
print("\n  Pearson r between log10|S_12_ansatz| and log10|S_12_CDFT_truth| on Slater 111:")
print(f"    B3LYP molden ansatz                : r = {r_b3lyp:.4f}")
print(f"    SRSH singlet HOMO (rotation-aware) : r = {r_srsh_s:.4f}")
print(f"    SRSH cation alpha SOMO (rot-aware) : r = {r_srsh_c:.4f}")

# Also show vs log|J|
slater_J = np.array([rec['J_eV'] for rec in slater_records])
J_valid = valid & (slater_J > 0) & np.isfinite(np.log10(slater_J + 1e-30))
log_J = np.log10(slater_J[J_valid])
r_b3lyp_J,  _ = pearsonr(np.log10(np.abs(slater_S12_b3lyp[J_valid]) + 1e-12), log_J)
r_qchem_J,  _ = pearsonr(np.log10(np.abs(slater_S12_qchem[J_valid]) + 1e-12), log_J)
r_srsh_s_J, _ = pearsonr(np.log10(np.abs(slater_s12_singlet[J_valid]) + 1e-12), log_J)
r_srsh_c_J, _ = pearsonr(np.log10(np.abs(slater_s12_cation [J_valid]) + 1e-12), log_J)
print(f"\n  Pearson r between log10|S_12| and log10|J| on Slater 111 ({J_valid.sum()} pts):")
print(f"    QChem CDFT (truth)                 : r = {r_qchem_J:.4f}")
print(f"    B3LYP molden ansatz                : r = {r_b3lyp_J:.4f}")
print(f"    SRSH singlet HOMO (rotation-aware) : r = {r_srsh_s_J:.4f}")
print(f"    SRSH cation alpha SOMO             : r = {r_srsh_c_J:.4f}")

# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Compute SRSH S_12 for full training + MD set")
print("=" * 70)

xyz_files_704 = sorted(glob.glob(os.path.join(STRUCT, '*.xyz')))
print(f"  {len(xyz_files_704)} systematic structures")
coupling_csv = pd.read_csv(os.path.join(ROOT, 'data', 'coupling.csv'))
coupling_dict = {row['Filename'].replace('.out',''): row['Coupling_eV']
                 for _, row in coupling_csv.iterrows()}

s12_704_singlet = np.full(len(xyz_files_704), np.nan)
s12_704_cation  = np.full(len(xyz_files_704), np.nan)
filenames_704 = []
for i, fn in enumerate(tqdm(xyz_files_704, desc="704 systematic")):
    stem = os.path.splitext(os.path.basename(fn))[0]
    filenames_704.append(stem)
    try:
        d = load_dimer_xyz(fn)
        s12_704_singlet[i] = calc_singlet.compute(
            d['monomer1']['positions'], d['monomer1']['symbols'],
            d['monomer2']['positions'], d['monomer2']['symbols'])
        s12_704_cation[i] = calc_cation.compute(
            d['monomer1']['positions'], d['monomer1']['symbols'],
            d['monomer2']['positions'], d['monomer2']['symbols'])
    except Exception as e:
        tqdm.write(f"  failed {stem}: {e}")

# MD 15 structures
md_files = sorted(glob.glob(os.path.join(MD_DIR, '*.xyz')))
print(f"  {len(md_files)} MD validation structures")
s12_md_singlet = np.full(len(md_files), np.nan)
s12_md_cation  = np.full(len(md_files), np.nan)
md_names = []
for i, fn in enumerate(tqdm(md_files, desc="MD 15")):
    md_names.append(os.path.splitext(os.path.basename(fn))[0])
    try:
        d = load_dimer_xyz(fn)
        s12_md_singlet[i] = calc_singlet.compute(
            d['monomer1']['positions'], d['monomer1']['symbols'],
            d['monomer2']['positions'], d['monomer2']['symbols'])
        s12_md_cation[i] = calc_cation.compute(
            d['monomer1']['positions'], d['monomer1']['symbols'],
            d['monomer2']['positions'], d['monomer2']['symbols'])
    except Exception as e:
        tqdm.write(f"  failed {fn}: {e}")

np.savez(os.path.join(DATA, 's12_srsh_singlet.npz'),
         s12_704=s12_704_singlet, names_704=np.array(filenames_704),
         s12_slater=slater_s12_singlet,
         s12_md=s12_md_singlet, md_names=np.array(md_names))
np.savez(os.path.join(DATA, 's12_srsh_cation.npz'),
         s12_704=s12_704_cation, names_704=np.array(filenames_704),
         s12_slater=slater_s12_cation,
         s12_md=s12_md_cation, md_names=np.array(md_names))
print(f"  Saved s12_srsh_*.npz")

out = {
    'singlet_HOMO': {
        'energy_eV': float(fchk_singlet['alpha_e'][fchk_singlet['nalpha']-1]*27.2114),
        'r_log_S12_vs_qchem_truth_slater111': float(r_srsh_s),
        'r_log_S12_vs_logJ_slater':           float(r_srsh_s_J),
    },
    'cation_alpha_SOMO': {
        'energy_eV': float(fchk_cation['alpha_e'][fchk_cation['nalpha']-1]*27.2114),
        'r_log_S12_vs_qchem_truth_slater111': float(r_srsh_c),
        'r_log_S12_vs_logJ_slater':           float(r_srsh_c_J),
    },
    'b3lyp_baseline': {
        'r_log_S12_vs_qchem_truth_slater111': float(r_b3lyp),
        'r_log_S12_vs_logJ_slater':           float(r_b3lyp_J),
    },
    'qchem_truth': {
        'r_log_S12_vs_logJ_slater':           float(r_qchem_J),
    },
    'n_valid_slater': int(valid.sum()),
}
with open(os.path.join(RESULTS, 'srsh_s12_validation.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved: results/srsh_s12_validation.json")

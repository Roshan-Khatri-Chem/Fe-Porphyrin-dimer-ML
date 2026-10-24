"""
xTB experiment Part 1: Run xTB on all 704 dimers, extract HOMO-HOMO-1 splitting.
Part 2: Monomer HOMO overlap using xTB MO coefficients.
"""

import sys, os, glob, json, re, subprocess, tempfile, shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
STRUCT_DIR = os.path.join(DATA_DIR, 'structures')
COUPLING_CSV = os.path.join(DATA_DIR, 'coupling.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

XTB = os.path.expanduser('~/miniconda3/bin/xtb')

DEC_BINS = [(-4, -3), (-3, -2), (-2, -1), (-1, 0)]


def run_xtb_dimer(xyz_path, workdir):
    """Run xTB on a dimer XYZ file, return HOMO and HOMO-1 energies in eV."""
    result = subprocess.run(
        [XTB, xyz_path, '--gfn', '2', '--chrg', '0', '--uhf', '0'],
        capture_output=True, text=True, cwd=workdir, timeout=120
    )
    output = result.stdout

    # Parse the FINAL orbital table (after convergence)
    # Find the HOMO line and the line before it
    homo_energy = None
    homo_m1_energy = None
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if '(HOMO)' in line:
            # Parse HOMO energy
            parts = line.split()
            homo_energy = float(parts[-2])  # Energy/eV column
            # HOMO-1 is the previous line
            if i > 0:
                prev = lines[i - 1].split()
                if len(prev) >= 3:
                    homo_m1_energy = float(prev[-1])  # Energy/eV

    return homo_energy, homo_m1_energy


def stratified_mae(y_true, y_pred):
    decades = np.floor(y_true).astype(int)
    result = {}
    for lo, hi in DEC_BINS:
        m = (decades >= lo) & (decades < hi)
        if m.sum() > 0:
            result[f'[1e{lo},1e{hi})'] = {
                'log_mae': float(mean_absolute_error(y_true[m], y_pred[m])),
                'N': int(m.sum())
            }
    return result


# ============================================================
# PART 1: xTB dimer calculations
# ============================================================
print("=" * 70)
print("PART 1: xTB dimer calculations")
print("=" * 70)

df_coup = pd.read_csv(COUPLING_CSV)
coupling_dict = {}
for _, row in df_coup.iterrows():
    stem = row['Filename'].replace('.out', '')
    coupling_dict[stem] = row['Coupling_eV']

xyz_files = sorted(glob.glob(os.path.join(STRUCT_DIR, '*.xyz')))

# Cache file to avoid rerunning
cache_file = os.path.join(PROCESSED_DIR, 'xtb_dimer_results.npz')
if os.path.exists(cache_file):
    print("Loading cached xTB results...")
    cached = np.load(cache_file, allow_pickle=True)
    filenames = cached['filenames'].tolist()
    J_ref = cached['J_ref']
    J_xtb = cached['J_xtb']
    homo_energies = cached['homo_energies']
    homo_m1_energies = cached['homo_m1_energies']
    print(f"Loaded {len(filenames)} cached results")
else:
    filenames = []
    J_ref_list = []
    J_xtb_list = []
    homo_list = []
    homo_m1_list = []
    n_failed = 0

    workdir = tempfile.mkdtemp(prefix='xtb_dimer_')
    print(f"Working directory: {workdir}")

    for filepath in tqdm(xyz_files, desc="xTB dimers"):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        if stem not in coupling_dict:
            continue

        try:
            homo_e, homo_m1_e = run_xtb_dimer(filepath, workdir)
            if homo_e is not None and homo_m1_e is not None:
                delta_E = abs(homo_e - homo_m1_e)
                J_xtb_val = delta_E / 2.0
                filenames.append(stem)
                J_ref_list.append(coupling_dict[stem])
                J_xtb_list.append(J_xtb_val)
                homo_list.append(homo_e)
                homo_m1_list.append(homo_m1_e)
            else:
                n_failed += 1
                tqdm.write(f"  PARSE FAIL: {stem}")
        except Exception as e:
            n_failed += 1
            tqdm.write(f"  ERROR: {stem}: {e}")

    # Cleanup
    shutil.rmtree(workdir, ignore_errors=True)

    J_ref = np.array(J_ref_list)
    J_xtb = np.array(J_xtb_list)
    homo_energies = np.array(homo_list)
    homo_m1_energies = np.array(homo_m1_list)

    print(f"\nCompleted: {len(filenames)} structures, {n_failed} failures")

    # Cache results
    np.savez(cache_file, filenames=np.array(filenames), J_ref=J_ref, J_xtb=J_xtb,
             homo_energies=homo_energies, homo_m1_energies=homo_m1_energies)
    print(f"Cached to {cache_file}")

# ============================================================
# Analysis
# ============================================================
print("\n" + "=" * 70)
print("PART 1 ANALYSIS")
print("=" * 70)

print(f"Structures with results: {len(filenames)}")
print(f"J_xTB range: [{J_xtb.min():.6e}, {J_xtb.max():.6e}] eV")
print(f"J_ref range: [{J_ref.min():.6e}, {J_ref.max():.6e}] eV")

# Filter to structures where both J_ref and J_xtb > 0 for log comparison
valid = (J_ref > 0) & (J_xtb > 0)
log_J_ref = np.log10(J_ref[valid])
log_J_xtb = np.log10(J_xtb[valid])

r_log, p_log = pearsonr(log_J_xtb, log_J_ref)
print(f"\nPearson r(log J_xTB, log J_ref): {r_log:.4f} (p = {p_log:.2e})")

# Log-MAE if J_xTB used directly as predictor
# Need to evaluate on structures where J_ref >= 1e-4 (the significant range)
sig_valid = valid & (J_ref >= 1e-4)
log_J_ref_sig = np.log10(J_ref[sig_valid])
log_J_xtb_sig = np.log10(J_xtb[sig_valid])

mae_xtb = mean_absolute_error(log_J_ref_sig, log_J_xtb_sig)
r2_xtb = r2_score(log_J_ref_sig, log_J_xtb_sig)
strat_xtb = stratified_mae(log_J_ref_sig, log_J_xtb_sig)

print(f"\nIf J_xTB used directly (significant structures, N={sig_valid.sum()}):")
print(f"  log MAE = {mae_xtb:.4f}")
print(f"  R² = {r2_xtb:.4f}")
for k, v in strat_xtb.items():
    print(f"    {k}: log MAE = {v['log_mae']:.4f}, N = {v['N']}")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

fig, ax = plt.subplots(figsize=(7, 7))
mask_sig = (J_ref > 0) & (J_xtb > 0) & (J_ref >= 1e-4)
mask_nz = (J_ref > 0) & (J_xtb > 0) & (J_ref < 1e-4)
ax.scatter(np.log10(J_ref[mask_sig]), np.log10(J_xtb[mask_sig]),
           alpha=0.5, s=15, c='steelblue', label=f'Significant (N={mask_sig.sum()})')
if mask_nz.sum() > 0:
    ax.scatter(np.log10(J_ref[mask_nz]), np.log10(J_xtb[mask_nz]),
               alpha=0.5, s=15, c='red', marker='x', label=f'Near-zero (N={mask_nz.sum()})')
all_valid = (J_ref > 0) & (J_xtb > 0)
lims = [min(np.log10(J_ref[all_valid]).min(), np.log10(J_xtb[all_valid]).min()) - 0.3,
        max(np.log10(J_ref[all_valid]).max(), np.log10(J_xtb[all_valid]).max()) + 0.3]
ax.plot(lims, lims, 'k-', linewidth=1.2)
ax.plot(lims, [l + 1 for l in lims], 'k--', alpha=0.3, linewidth=0.8)
ax.plot(lims, [l - 1 for l in lims], 'k--', alpha=0.3, linewidth=0.8)
ax.set_xlabel(r'Reference $\log_{10}(J\,/\,\mathrm{eV})$')
ax.set_ylabel(r'xTB $\log_{10}(J_\mathrm{xTB}\,/\,\mathrm{eV})$')
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='grey')
ax.text(0.04, 0.96, f'$r = {r_log:.3f}$\nlog-MAE $= {mae_xtb:.3f}$',
        transform=ax.transAxes, fontsize=11, va='top', bbox=props)
ax.legend(loc='lower right')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'xtb_dimer_parity.png'), dpi=300)
plt.savefig(os.path.join(FIG_DIR, 'xtb_dimer_parity.pdf'))
plt.close()
print("\nSaved: xtb_dimer_parity.png/pdf")

# Save results
results = {
    'n_structures': len(filenames),
    'pearson_r_log': float(r_log),
    'log_mae_as_predictor': float(mae_xtb),
    'log_r2_as_predictor': float(r2_xtb),
    'stratified': strat_xtb,
}
with open(os.path.join(RESULTS_DIR, 'xtb_part1_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: results/xtb_part1_results.json")

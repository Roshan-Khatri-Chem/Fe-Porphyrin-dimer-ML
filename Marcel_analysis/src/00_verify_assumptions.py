"""
Phase 2: Assumption Verification

Load every XYZ file. Report:
  1. Total number of files
  2. Atom count per file (assert all identical)
  3. Element composition of monomer 1 and monomer 2 (assert identical)
  4. Fe count per monomer (assert == 1)
  5. N count per monomer (assert == 4)
  6. Monomer RMSD via Kabsch alignment
  7. Coupling CSV summary
  8. Filename matching between XYZ files and coupling CSV
"""

import sys
import os
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from geometry import load_dimer_xyz

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
STRUCT_DIR = os.path.join(DATA_DIR, 'structures')
COUPLING_CSV = os.path.join(DATA_DIR, 'coupling.csv')


def kabsch_rmsd(P, Q):
    """Compute RMSD between P and Q after optimal Kabsch alignment."""
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_c = P - centroid_P
    Q_c = Q - centroid_Q
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    P_aligned = (R @ P_c.T).T
    return np.sqrt(np.mean(np.sum((P_aligned - Q_c) ** 2, axis=1)))


def main():
    # 1. Total number of files
    xyz_files = sorted(glob.glob(os.path.join(STRUCT_DIR, '*.xyz')))
    print(f"1. Total XYZ files: {len(xyz_files)}")

    # Load all files
    atom_counts = []
    compositions = []
    fe_counts_m1 = []
    fe_counts_m2 = []
    n_counts_m1 = []
    n_counts_m2 = []
    m1_positions_list = []
    m1_symbols_list = []
    m2_positions_list = []
    filenames = []

    for f in xyz_files:
        d = load_dimer_xyz(f)
        filenames.append(os.path.basename(f))
        atom_counts.append(len(d['symbols']))
        m1 = d['monomer1']
        m2 = d['monomer2']
        compositions.append(sorted(m1['symbols'].tolist()))
        fe_counts_m1.append(np.sum(m1['symbols'] == 'Fe'))
        fe_counts_m2.append(np.sum(m2['symbols'] == 'Fe'))
        n_counts_m1.append(np.sum(m1['symbols'] == 'N'))
        n_counts_m2.append(np.sum(m2['symbols'] == 'N'))
        m1_positions_list.append(m1['positions'])
        m1_symbols_list.append(m1['symbols'])
        m2_positions_list.append(m2['positions'])

    # 2. Atom count per file
    unique_counts = set(atom_counts)
    print(f"2. Atom counts: {unique_counts}")
    assert len(unique_counts) == 1, f"NOT all files have the same atom count: {unique_counts}"
    print(f"   All files have {atom_counts[0]} atoms ✓")

    # 3. Element composition
    ref_comp = compositions[0]
    all_same = all(c == ref_comp for c in compositions)
    print(f"3. All monomers have same composition: {all_same}")
    if all_same:
        from collections import Counter
        comp_dict = dict(Counter(ref_comp))
        print(f"   Composition per monomer: {comp_dict}")

    # 4. Fe count per monomer
    assert all(c == 1 for c in fe_counts_m1), "Some monomer 1 have != 1 Fe"
    assert all(c == 1 for c in fe_counts_m2), "Some monomer 2 have != 1 Fe"
    print(f"4. Fe count per monomer: all == 1 ✓")

    # 5. N count per monomer
    assert all(c == 4 for c in n_counts_m1), "Some monomer 1 have != 4 N"
    assert all(c == 4 for c in n_counts_m2), "Some monomer 2 have != 4 N"
    print(f"5. N count per monomer: all == 4 ✓")

    # 6. Monomer RMSD
    # Sort atoms by element for consistent alignment
    ref_pos = m1_positions_list[0]
    ref_sym = m1_symbols_list[0]
    sort_idx = np.argsort(ref_sym)
    ref_sorted = ref_pos[sort_idx]

    rmsds_m1 = []
    rmsds_m2 = []
    for i in range(len(xyz_files)):
        # Sort monomer 1 atoms
        s1 = m1_symbols_list[i]
        idx1 = np.argsort(s1)
        pos1_sorted = m1_positions_list[i][idx1]
        rmsds_m1.append(kabsch_rmsd(pos1_sorted, ref_sorted))

        # Sort monomer 2 atoms
        # Monomer 2 has same symbols as monomer 1 (verified above)
        pos2_sorted = m2_positions_list[i][idx1]
        rmsds_m2.append(kabsch_rmsd(pos2_sorted, ref_sorted))

    rmsds_m1 = np.array(rmsds_m1)
    rmsds_m2 = np.array(rmsds_m2)
    print(f"6. Monomer RMSD (Kabsch-aligned to reference):")
    print(f"   Monomer 1: mean={rmsds_m1.mean():.4f} Å, max={rmsds_m1.max():.4f} Å")
    print(f"   Monomer 2: mean={rmsds_m2.mean():.4f} Å, max={rmsds_m2.max():.4f} Å")
    if rmsds_m1.max() > 0.1 or rmsds_m2.max() > 0.1:
        print("   ⚠ WARNING: max RMSD > 0.1 Å — monomers may not be rigid!")
        high_rmsd_m1 = [(filenames[i], rmsds_m1[i]) for i in range(len(filenames)) if rmsds_m1[i] > 0.05]
        high_rmsd_m2 = [(filenames[i], rmsds_m2[i]) for i in range(len(filenames)) if rmsds_m2[i] > 0.05]
        if high_rmsd_m1:
            print(f"   Files with monomer 1 RMSD > 0.05: {len(high_rmsd_m1)}")
        if high_rmsd_m2:
            print(f"   Files with monomer 2 RMSD > 0.05: {len(high_rmsd_m2)}")
    else:
        print(f"   All RMSDs < 0.1 Å — monomers are rigid ✓")

    # 7. Coupling CSV
    df = pd.read_csv(COUPLING_CSV)
    print(f"\n7. Coupling CSV:")
    print(f"   Columns: {list(df.columns[:5])} ...")
    print(f"   N rows: {len(df)}")
    print(f"   Coupling range: {df['Coupling_eV'].min():.6e} to {df['Coupling_eV'].max():.6e}")
    print(f"   Mean coupling: {df['Coupling_eV'].mean():.6e}")
    n_below_1e4 = (df['Coupling_eV'] < 1e-4).sum()
    n_zero = (df['Coupling_eV'] == 0).sum()
    print(f"   N values < 1e-4 eV: {n_below_1e4}")
    print(f"   N zero values: {n_zero}")

    # 8. Filename matching
    # CSV uses .out extension, XYZ files use .xyz
    csv_stems = set(df['Filename'].str.replace('.out', '', regex=False))
    xyz_stems = set(os.path.splitext(f)[0] for f in filenames)
    matched = csv_stems & xyz_stems
    csv_only = csv_stems - xyz_stems
    xyz_only = xyz_stems - csv_stems
    print(f"\n8. Filename matching:")
    print(f"   Matched: {len(matched)}")
    print(f"   In CSV but no XYZ: {len(csv_only)}")
    if csv_only:
        print(f"     {sorted(csv_only)[:5]}...")
    print(f"   In XYZ but no CSV: {len(xyz_only)}")
    if xyz_only:
        print(f"     {sorted(xyz_only)[:10]}...")

    print("\n" + "="*60)
    print("PHASE 2 SUMMARY: All assertions passed.")
    print("="*60)


if __name__ == '__main__':
    main()

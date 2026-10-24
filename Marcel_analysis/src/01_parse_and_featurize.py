"""
Phase 3: Feature Matrix Construction

Build feature matrix from XYZ files and coupling CSV.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from geometry import features_from_xyz

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
STRUCT_DIR = os.path.join(DATA_DIR, 'structures')
COUPLING_CSV = os.path.join(DATA_DIR, 'coupling.csv')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

FEATURE_NAMES = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi', 'cos_4psi', 'sin_4psi']


def main():
    # 1. Load coupling CSV
    df_coup = pd.read_csv(COUPLING_CSV)
    # Build dict: filename_stem -> coupling_eV
    coupling_dict = {}
    for _, row in df_coup.iterrows():
        stem = row['Filename'].replace('.out', '')
        coupling_dict[stem] = row['Coupling_eV']
    print(f"Loaded {len(coupling_dict)} coupling entries")

    # 2. Glob and sort XYZ files
    xyz_files = sorted(glob.glob(os.path.join(STRUCT_DIR, '*.xyz')))
    print(f"Found {len(xyz_files)} XYZ files")

    # 3. Featurize each file that has a coupling entry
    features_list = []
    labels_ev_list = []
    filenames_list = []
    skipped = 0

    for filepath in tqdm(xyz_files, desc="Featurizing"):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        if stem not in coupling_dict:
            skipped += 1
            continue
        feat = features_from_xyz(filepath)
        features_list.append(feat)
        labels_ev_list.append(coupling_dict[stem])
        filenames_list.append(stem)

    print(f"Skipped {skipped} files without coupling data")

    # 4. Stack arrays
    features = np.array(features_list, dtype=np.float64)
    labels_ev = np.array(labels_ev_list, dtype=np.float64)
    labels_log = np.where(labels_ev > 0, np.log10(labels_ev), np.nan)

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Labels shape: {labels_ev.shape}")

    # 5. Save arrays
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, 'features.npy'), features)
    np.save(os.path.join(PROCESSED_DIR, 'labels_ev.npy'), labels_ev)
    np.save(os.path.join(PROCESSED_DIR, 'labels_log.npy'), labels_log)

    # 6. Build metadata
    is_near_zero = labels_ev < 1e-4

    # Data split: StratifiedShuffleSplit
    from sklearn.model_selection import StratifiedShuffleSplit

    # Create stratification bins
    # For log-space: bin by decade
    strat_labels = np.zeros(len(labels_ev), dtype=int)
    for i, val in enumerate(labels_ev):
        if val < 1e-4:
            strat_labels[i] = -99  # near-zero bin
        elif val > 0:
            strat_labels[i] = int(np.floor(np.log10(val)))
        else:
            strat_labels[i] = -99

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(features, strat_labels))
    split = np.array(['train'] * len(labels_ev))
    split[test_idx] = 'test'

    metadata = pd.DataFrame({
        'idx': range(len(labels_ev)),
        'filename': filenames_list,
        'coupling_eV': labels_ev,
        'log10_coupling': labels_log,
        'is_near_zero': is_near_zero,
        'split': split,
    })
    metadata.to_csv(os.path.join(PROCESSED_DIR, 'metadata.csv'), index=False)

    # 7. Summary
    n_total = len(labels_ev)
    n_near_zero = is_near_zero.sum()
    n_usable_log = np.sum(labels_ev > 0)
    n_nan = np.sum(~np.isfinite(features))
    n_train = (split == 'train').sum()
    n_test = (split == 'test').sum()

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"N total:          {n_total}")
    print(f"N near-zero (<1e-4 eV): {n_near_zero}")
    print(f"N usable for log-space: {n_usable_log}")
    print(f"N train / N test: {n_train} / {n_test}")
    print(f"NaN/Inf in features: {n_nan}")
    print()

    print("Feature statistics:")
    print(f"{'Feature':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 54)
    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]
        print(f"{name:<12} {col.mean():>10.4f} {col.std():>10.4f} {col.min():>10.4f} {col.max():>10.4f}")


if __name__ == '__main__':
    main()

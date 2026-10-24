"""
Self-contained inference module for Fe-porphyrin dimer coupling prediction.

Uses the two-stage pipeline:
  Stage 1: SVC classifier (near-zero vs significant) on 8 core features
  Stage 2: RF regressor (11-feature v2 pipeline) on log10(coupling)

Public API:
    import predict as coupling_model
    J, J_std = coupling_model.predict_from_arrays(m1_positions, m2_positions,
                                                   m1_symbols, m2_symbols)
    J, J_std = coupling_model.predict_from_xyz(filepath)
"""

import os
import numpy as np
import joblib
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'src'))
from geometry import dimer_features, load_dimer_xyz, is_rpd_interference_node

# Load models once at import time
_MODEL_DIR = os.path.join(_THIS_DIR, 'models')
_stage1_classifier = joblib.load(os.path.join(_MODEL_DIR, 'stage1_classifier.pkl'))
_stage2_rf = joblib.load(os.path.join(_MODEL_DIR, 'stage2_rf_v2.pkl'))
_scaler_stage2 = joblib.load(os.path.join(_MODEL_DIR, 'scaler_stage2_v2.pkl'))

# Also load GPR for uncertainty estimates
try:
    _stage2_gpr = joblib.load(os.path.join(_MODEL_DIR, 'stage2_gpr_v2.pkl'))
    _has_gpr = True
except Exception:
    _has_gpr = False


def predict_from_arrays(m1_positions, m2_positions, m1_symbols, m2_symbols):
    """
    Predict electronic coupling from monomer positions and symbols.

    Parameters
    ----------
    m1_positions : (M, 3) float64, Angstrom
    m2_positions : (M, 3) float64, Angstrom
    m1_symbols   : (M,) element symbols as strings
    m2_symbols   : (M,) element symbols as strings

    Returns
    -------
    J     : float, predicted coupling in eV
    J_std : float, GPR posterior std in eV (0.0 if GPR unavailable)
    """
    m1_positions = np.asarray(m1_positions, dtype=np.float64)
    m2_positions = np.asarray(m2_positions, dtype=np.float64)
    m1_symbols = np.asarray(m1_symbols)
    m2_symbols = np.asarray(m2_symbols)

    # 1. Compute feature vector
    feat = dimer_features(m1_positions, m1_symbols, m2_positions, m2_symbols)
    X = feat[:11].reshape(1, -1)  # first 11 features for v2 model

    # Stage 0 (RPD interference node classifier) was evaluated and rejected:
    # 77% false-positive rate on training set. The interference zeros depend
    # on the full multi-dimensional geometry, not just d_lateral alone.

    # Stage 1: learned near-zero classifier (D4h twist node, 8 features)
    pred_class = _stage1_classifier.predict(X[:, :8])[0]
    if pred_class == 1:  # near-zero
        return 0.0, 0.0

    # 3. Stage 2: predict log10(coupling) with RF pipeline (11 features)
    mu = _stage2_rf.predict(X)[0]
    J = 10 ** mu

    # 4. Optional GPR uncertainty estimate
    J_std = 0.0
    if _has_gpr:
        X_scaled = _scaler_stage2.transform(X)
        mu_gpr, std_gpr = _stage2_gpr.predict(X_scaled, return_std=True)
        J_std = float(10 ** (mu_gpr[0] + std_gpr[0]) - 10 ** mu_gpr[0])

    return float(J), float(J_std)


def predict_from_xyz(filepath):
    """
    Predict coupling from an XYZ file.

    Parameters
    ----------
    filepath : str, path to dimer XYZ file

    Returns
    -------
    J     : float, predicted coupling in eV
    J_std : float, GPR posterior std in eV
    """
    d = load_dimer_xyz(filepath)
    return predict_from_arrays(
        d['monomer1']['positions'], d['monomer2']['positions'],
        d['monomer1']['symbols'], d['monomer2']['symbols']
    )


if __name__ == '__main__':
    import sys as _sys
    import time
    import glob

    if len(_sys.argv) > 1:
        filepath = _sys.argv[1]
        J, J_std = predict_from_xyz(filepath)
        if J_std > 0:
            print(f"Predicted coupling: {J:.3e} eV (±{J_std:.2e} eV)")
        else:
            print(f"Predicted coupling: {J:.3e} eV")
    else:
        # Timing benchmark using a real structure
        print("Running timing benchmark with real monomer arrays...")
        struct_dir = os.path.join(_THIS_DIR, 'data', 'structures')
        xyz_files = sorted(glob.glob(os.path.join(struct_dir, '*.xyz')))
        d = load_dimer_xyz(xyz_files[0])
        pos1 = d['monomer1']['positions']
        pos2 = d['monomer2']['positions']
        sym1 = d['monomer1']['symbols']
        sym2 = d['monomer2']['symbols']

        # Warmup
        for _ in range(5):
            predict_from_arrays(pos1, pos2, sym1, sym2)

        # Time 1000 calls
        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            predict_from_arrays(pos1, pos2, sym1, sym2)
            times.append(time.perf_counter() - t0)

        times = np.array(times) * 1000  # ms
        print(f"Timing: {times.mean():.3f} ± {times.std():.3f} ms per call (N=1000)")

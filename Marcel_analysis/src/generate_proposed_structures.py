"""
Generate two sets of optimized Fe-porphine dimer geometries as XYZ files:
1) md_proposed_160  – synthetic stand‑ins for the ~160 MD-sampled structures
2) rpd_scan_240     – focused lateral displacement scan (R_PD grid)

Each XYZ comment line stores the 13 geometry features from geometry.dimer_features.
The monomers are rigid copies of the optimized reference monomer
(data/processed/reference_monomer.xyz) placed according to requested dz/dx/dy/phi/tilt.
"""
from __future__ import annotations
import os, math
import numpy as np
from pathlib import Path
from ase.io import read as ase_read

import sys
sys.path.append(os.path.dirname(__file__))
import geometry

ROOT = Path(__file__).resolve().parent.parent
REF_PATH = ROOT / "data" / "processed" / "reference_monomer.xyz"
OUT_MD = ROOT / "data" / "generated_md160"
OUT_RPD = ROOT / "data" / "generated_rpd_scan"

np.set_printoptions(suppress=True, precision=6)


def load_reference_monomer():
    atoms = ase_read(REF_PATH)
    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    # Build canonical frame: centroid at origin, x->+x, z->+z
    centroid, normal, x_axis = geometry.monomer_frame(positions, symbols)
    y_axis = np.cross(normal, x_axis)
    # Orthonormal rotation matrix with columns = [x,y,z]; maps local to lab.
    O = np.stack([x_axis, y_axis, normal], axis=1)
    # To map lab coords into canonical frame, use O.T and subtract centroid
    positions_centered = (positions - centroid) @ O
    return positions_centered, symbols


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0],[s, c, 0],[0,0,1]])

def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def build_dimer(dx, dy, dz, phi=0.0, tilt_deg=0.0):
    """Place monomer2 relative to monomer1 using canonical frame."""
    mono_pos, mono_sym = load_reference_monomer()
    # Monomer 1 at origin, canonical orientation
    m1_pos, m1_sym = mono_pos.copy(), mono_sym
    # Monomer 2: rotate about z by phi, optional tilt about y, then translate
    R = rot_y(math.radians(tilt_deg)) @ rot_z(phi)
    m2_pos = (R @ mono_pos.T).T + np.array([dx, dy, dz])
    m2_sym = mono_sym
    return m1_pos, m1_sym, m2_pos, m2_sym


def write_xyz(path: Path, m1_pos, m1_sym, m2_pos, m2_sym):
    features = geometry.dimer_features(m1_pos, m1_sym, m2_pos, m2_sym)
    # comment with feature names
    names = ["dz","dx","dy","cos_theta","cos_4phi","sin_4phi",
             "cos_4psi","sin_4psi","d_lateral","cos_4alpha","sin_4alpha",
             "abs_S_HL","S_HL_sq"]
    feat_str = " ".join(f"{n}={v:.6f}" for n,v in zip(names, features))
    all_pos = np.vstack([m1_pos, m2_pos])
    all_sym = np.concatenate([m1_sym, m2_sym])
    with open(path, "w") as f:
        f.write(f"{len(all_sym)}\n")
        f.write(feat_str + "\n")
        for s,(x,y,z) in zip(all_sym, all_pos):
            f.write(f"{s:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")


def generate_md_like(n_total=160, seed=42):
    rng = np.random.default_rng(seed)
    OUT_MD.mkdir(parents=True, exist_ok=True)
    records = []
    # Core set (~140): broad coverage
    for i in range(140):
        dz = rng.uniform(3.0, 8.5)
        d_lat = rng.uniform(0.0, 6.0)
        alpha = rng.uniform(0, 2*math.pi)
        dx = d_lat * math.cos(alpha)
        dy = d_lat * math.sin(alpha)
        phi = rng.uniform(0, math.pi/2)  # D4h fundamental domain
        tilt = rng.uniform(-15, 15)
        records.append((dz, dx, dy, phi, tilt))
    # Interference-focused (~20): around 2.0 and 4.6 Å, low dz
    for center in (2.0, 4.6):
        for _ in range(10):
            dz = rng.uniform(3.0, 5.5)
            d_lat = rng.normal(center, 0.3)
            d_lat = max(0.0, d_lat)
            alpha = rng.uniform(0, 2*math.pi)
            dx = d_lat * math.cos(alpha)
            dy = d_lat * math.sin(alpha)
            phi = rng.uniform(0, math.pi/2)
            tilt = rng.uniform(-10, 10)
            records.append((dz, dx, dy, phi, tilt))
    assert len(records) == n_total
    for idx, (dz, dx, dy, phi, tilt) in enumerate(records, 1):
        m1_pos, m1_sym, m2_pos, m2_sym = build_dimer(dx, dy, dz, phi, tilt)
        write_xyz(OUT_MD / f"md_{idx:03d}.xyz", m1_pos, m1_sym, m2_pos, m2_sym)


def generate_rpd_grid():
    OUT_RPD.mkdir(parents=True, exist_ok=True)
    # 0.20 Å spacing gives ~246 structures (close to the requested ~240)
    d_lats = np.arange(0.0, 8.0001, 0.20)
    dz_list = [3.5, 4.0, 4.5]
    tilts = [0.0, 11.0]
    idx = 1
    for dz in dz_list:
        for tilt in tilts:
            for d_lat in d_lats:
                dx, dy = d_lat, 0.0  # slippage along +x
                phi = 0.0
                m1_pos, m1_sym, m2_pos, m2_sym = build_dimer(dx, dy, dz, phi, tilt)
                write_xyz(OUT_RPD / f"rpd_{idx:04d}.xyz", m1_pos, m1_sym, m2_pos, m2_sym)
                idx += 1


def main():
    generate_md_like()
    generate_rpd_grid()
    print(f"Wrote {len(list(OUT_MD.glob('*.xyz')))} MD-like structures to {OUT_MD}")
    print(f"Wrote {len(list(OUT_RPD.glob('*.xyz')))} RPD scan structures to {OUT_RPD}")


if __name__ == "__main__":
    main()

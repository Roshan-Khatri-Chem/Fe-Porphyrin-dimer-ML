#!/usr/bin/env python3
"""Generate distance/angle-based geometry features for a folder of dimer XYZ files.

The features are derived from the monomer Fe positions, porphyrin-plane normals,
and heavy-atom pairwise distances.

Example:
    /Users/Roshan/opt/anaconda3/bin/python src/generate_distance_features_from_folder.py \
        --input-dir data/Roshan-test_dimer \
        --output-csv data/Roshan-test_dimer_distance_features.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ase.io import read as ase_read


def _best_fit_plane_normal(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    n = vh[-1]
    n /= np.linalg.norm(n)
    return n


def _principal_inplane_axis(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    proj = centered - np.outer(centered @ normal, normal)
    _, _, vh = np.linalg.svd(proj, full_matrices=False)
    e1 = vh[0]
    e1 -= (e1 @ normal) * normal
    e1 /= np.linalg.norm(e1)
    return e1


def compute_distance_features(xyz_path: Path, anchor_symbol: str = "N") -> dict:
    atoms = ase_read(str(xyz_path))
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()

    if len(pos) % 2 != 0:
        raise ValueError(f"Expected an even atom count for a dimer XYZ, got {len(pos)}")

    n_atoms_per_monomer = len(pos) // 2
    mon1_idx = list(range(0, n_atoms_per_monomer))
    mon2_idx = list(range(n_atoms_per_monomer, len(pos)))

    fe1 = np.array([pos[i] for i in mon1_idx if sym[i] == "Fe"])[0]
    fe2 = np.array([pos[i] for i in mon2_idx if sym[i] == "Fe"])[0]

    n1 = np.array([pos[i] for i in mon1_idx if sym[i] == anchor_symbol])
    n2 = np.array([pos[i] for i in mon2_idx if sym[i] == anchor_symbol])

    if len(n1) < 3 or len(n2) < 3:
        raise ValueError("Not enough anchor atoms")

    n1_norm = _best_fit_plane_normal(n1)
    n2_norm = _best_fit_plane_normal(n2)

    if n1_norm @ n2_norm < 0:
        n2_norm = -n2_norm

    n_avg = (n1_norm + n2_norm)
    n_avg /= np.linalg.norm(n_avg)

    v = fe2 - fe1
    d_perp = abs(v @ n_avg)
    v_para = v - (v @ n_avg) * n_avg
    s_mag = np.linalg.norm(v_para)

    e1 = _principal_inplane_axis(n1, n1_norm)
    e2 = np.cross(n_avg, e1)
    e2 /= np.linalg.norm(e2)

    s1 = v_para @ e1
    s2 = v_para @ e2
    phi = np.degrees(np.arctan2(s2, s1))
    tilt = np.degrees(np.arccos(np.clip(n1_norm @ n2_norm, -1.0, 1.0)))

    heavy_elements = {"C", "N", "Fe"}
    heavy1 = [i for i in mon1_idx if sym[i] in heavy_elements]
    heavy2 = [i for i in mon2_idx if sym[i] in heavy_elements]

    min_dist = np.nan
    max_dist = np.nan
    if heavy1 and heavy2:
        P1 = pos[heavy1]
        P2 = pos[heavy2]
        diff = P1[:, None, :] - P2[None, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        i_min, j_min = np.unravel_index(np.argmin(d), d.shape)
        i_max, j_max = np.unravel_index(np.argmax(d), d.shape)
        min_dist = float(d[i_min, j_min])
        max_dist = float(d[i_max, j_max])

    return {
        "name": xyz_path.stem,
        "Interlayer_Dist_Ang": float(d_perp),
        "Lateral_Slippage_Ang": float(s_mag),
        "Slippage_X_Ang": float(s1),
        "Slippage_Y_Ang": float(s2),
        "Slippage_Angle_Deg": float(phi),
        "Plane_Tilt_Deg": float(tilt),
        "FeFe_Dist_Ang": float(np.linalg.norm(v)),
        "Min_HeavyDist_Ang": float(min_dist),
        "Max_HeavyDist_Ang": float(max_dist),
    }


def list_xyz_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.xyz"))
    if not files:
        raise FileNotFoundError(f"No .xyz files found in {input_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate distance/angle geometry features for a folder of dimer XYZ files")
    parser.add_argument("--input-dir", required=True, help="Folder containing one or more .xyz dimer files")
    parser.add_argument("--output-csv", required=True, help="Path to save the output CSV")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    xyz_files = list_xyz_files(input_dir)
    rows = []
    for xyz_path in xyz_files:
        try:
            rows.append(compute_distance_features(xyz_path))
            print(f"Processed {xyz_path.name}")
        except Exception as exc:
            print(f"Skipped {xyz_path.name}: {exc}")

    if not rows:
        raise RuntimeError("No files were processed successfully")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "Interlayer_Dist_Ang",
        "Lateral_Slippage_Ang",
        "Slippage_X_Ang",
        "Slippage_Y_Ang",
        "Slippage_Angle_Deg",
        "Plane_Tilt_Deg",
        "FeFe_Dist_Ang",
        "Min_HeavyDist_Ang",
        "Max_HeavyDist_Ang",
    ]
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()

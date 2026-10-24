#!/usr/bin/env python3
"""Planarize MD-extracted dimer XYZ files using feature-table pose information.

The workflow is:
1. Load a reference monomer from data/processed/reference_monomer.xyz.
2. For each dimer XYZ file, split the dimer into two monomers.
3. Align each monomer to the reference monomer with a Kabsch RMSD fit.
4. Reconstruct a planarized dimer by placing monomer 1 at the origin and
   monomer 2 relative to it using the feature-table values for dx, dy, dz,
   and the recovered in-plane twist angle phi from cos_4phi/sin_4phi.
5. Write a new XYZ file with the same atom ordering and a simple comment line.

Usage:
    /path/to/.venv/bin/python src/planarize_md_xyz_from_features.py \
        --input-dir Bradly_MD_data_July-15/20000-xyz-files/P1_xyz-files \
        --features-file Bradly_MD_data_July-15/20000-xyz-files/P1_dimer_full_features.csv \
        --output-dir Bradly_MD_data_July-15/20000-xyz-files/P1_planar_xyz \
        --reference data/processed/reference_monomer.xyz
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from ase.io import read as ase_read, write as ase_write

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import geometry


def load_reference_monomer(reference_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    atoms = ase_read(str(reference_path))
    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    return positions, symbols


def kabsch_align(source_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Return an orientation matrix that maps source -> target by RMSD minimization."""
    src = np.asarray(source_pos, dtype=float)
    tgt = np.asarray(target_pos, dtype=float)
    src_centered = src - src.mean(axis=0)
    tgt_centered = tgt - tgt.mean(axis=0)
    covariance = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return rotation


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def recover_phi_from_features(row: Dict[str, str]) -> float:
    cos_4phi = float(row["cos_4phi"])
    sin_4phi = float(row["sin_4phi"])
    return 0.25 * math.atan2(sin_4phi, cos_4phi)


def load_features(features_file: Path) -> Dict[str, Dict[str, str]]:
    with features_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        result = {}
        for row in reader:
            key = row["Filename"].strip()
            if key.endswith(".xyz"):
                key = Path(key).stem
            result[key] = row
        return result


def planarize_dimer_from_row(
    source_dimer: Dict[str, np.ndarray],
    reference_pos: np.ndarray,
    reference_sym: np.ndarray,
    feature_row: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    m1 = source_dimer["monomer1"]
    m2 = source_dimer["monomer2"]

    # Align both monomers to the reference monomer by RMSD (Kabsch).
    rot_m1 = kabsch_align(m1["positions"], reference_pos)
    rot_m2 = kabsch_align(m2["positions"], reference_pos)

    m1_aligned = (m1["positions"] - m1["positions"].mean(axis=0)) @ rot_m1
    m2_aligned = (m2["positions"] - m2["positions"].mean(axis=0)) @ rot_m2

    # Use the reference monomer frame as the canonical basis.
    _, normal, x_axis = geometry.monomer_frame(reference_pos, reference_sym)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    dx = float(feature_row["dx"])
    dy = float(feature_row["dy"])
    dz = float(feature_row["dz"])
    translation = dx * x_axis + dy * y_axis + dz * normal

    phi = recover_phi_from_features(feature_row)
    twist = rotation_about_axis(normal, phi)

    m1_centered = m1_aligned - m1_aligned.mean(axis=0)
    m2_centered = m2_aligned - m2_aligned.mean(axis=0)

    m1_out = m1_centered
    m2_out = (m2_centered @ twist.T) + translation

    symbols = np.concatenate([m1["symbols"], m2["symbols"]])
    positions = np.vstack([m1_out, m2_out])
    return positions, symbols


def write_xyz(path: Path, positions: np.ndarray, symbols: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write("planarized_from_features\n")
        for symbol, (x, y, z) in zip(symbols, positions):
            handle.write(f"{symbol:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--features-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    features_file = args.features_file.resolve()
    output_dir = args.output_dir.resolve()
    reference_path = args.reference.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not features_file.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_file}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference monomer not found: {reference_path}")

    reference_pos, reference_sym = load_reference_monomer(reference_path)
    features = load_features(features_file)

    xyz_files = sorted(input_dir.glob("*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"No XYZ files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for xyz_path in xyz_files:
        stem = xyz_path.stem
        feature_row = features.get(stem)
        if feature_row is None:
            print(f"Skipping {xyz_path.name}: no feature row for {stem}")
            continue

        source_dimer = geometry.load_dimer_xyz(str(xyz_path))
        positions, symbols = planarize_dimer_from_row(
            source_dimer,
            reference_pos,
            reference_sym,
            feature_row,
        )
        out_path = output_dir / f"{stem}_planar.xyz"
        write_xyz(out_path, positions, symbols)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

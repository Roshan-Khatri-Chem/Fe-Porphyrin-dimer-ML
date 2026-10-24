#!/usr/bin/env python3
"""
Convert an MD-distorted dimer XYZ into a planar dimer by:
1) computing each monomer's molecular frame from the porphyrin-core nitrogens,
2) aligning monomer 2 so its plane normal matches monomer 1's plane normal,
3) placing monomer 2 at a chosen inter-monomer distance along monomer 1's normal
   while preserving the in-plane lateral offset from the original geometry.

Example:
    python src/planarize_md_dimer.py input.xyz output.xyz --dz 3.5
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

sys.path.append(os.path.dirname(__file__))
import geometry


def rotation_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps v1 -> v2."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.dot(v1, v2)
    if np.isclose(dot, 1.0):
        return np.eye(3)
    if np.isclose(dot, -1.0):
        # 180-degree rotation around any axis perpendicular to v1
        axis = np.array([1.0, 0.0, 0.0])
        if np.isclose(abs(np.dot(axis, v1)), 1.0):
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, v1) * v1
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        return np.eye(3) + 2.0 * K @ K

    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + dot))


def planarize_dimer(
    input_xyz: str,
    output_xyz: str,
    dz: float | None = None,
) -> dict:
    """
    Read a dimer XYZ, align monomer 2 to monomer 1 so the two planes are parallel,
    and write a planarized dimer.

    The new position of monomer 2 keeps the original in-plane lateral components
    relative to monomer 1 and uses the requested dz (interplanar distance) along
    monomer 1's normal.
    """
    atoms = ase_read(input_xyz)
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()

    if len(symbols) % 2 != 0:
        raise ValueError(f"Expected an even number of atoms, got {len(symbols)}")

    m = len(symbols) // 2
    sym1 = symbols[:m]
    pos1 = positions[:m]
    sym2 = symbols[m:]
    pos2 = positions[m:]

    c1, n1, x1 = geometry.monomer_frame(pos1, sym1)
    c2, n2, x2 = geometry.monomer_frame(pos2, sym2)

    disp_old = c2 - c1
    current_dist = np.linalg.norm(disp_old)
    current_angle_deg = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))

    # Preserve lateral components in monomer 1's frame and set the out-of-plane component.
    y1 = np.cross(n1, x1)
    dx = np.dot(disp_old, x1)
    dy = np.dot(disp_old, y1)
    if dz is None:
        dz = abs(np.dot(disp_old, n1))

    new_disp = dx * x1 + dy * y1 + float(dz) * n1
    new_centroid2 = c1 + new_disp

    # Rotate monomer 2 so its normal aligns with monomer 1's normal.
    rot = rotation_matrix_from_vectors(n2, n1)
    pos2_rot = (rot @ (pos2 - c2).T).T + new_centroid2

    new_positions = np.vstack([pos1, pos2_rot])
    new_symbols = np.concatenate([sym1, sym2])
    out_atoms = Atoms(symbols=new_symbols, positions=new_positions)
    ase_write(output_xyz, out_atoms)

    return {
        "input": input_xyz,
        "output": output_xyz,
        "current_distance": float(current_dist),
        "current_angle_deg": float(current_angle_deg),
        "new_distance": float(np.linalg.norm(new_centroid2 - c1)),
        "new_angle_deg": 0.0,
        "dz_used": float(dz),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Planarize a distorted dimer XYZ")
    parser.add_argument("input_xyz", help="Input dimer XYZ file")
    parser.add_argument("output_xyz", help="Output planarized XYZ file")
    parser.add_argument("--dz", type=float, default=None,
                        help="Desired inter-monomer distance along monomer 1 normal (Å)")
    args = parser.parse_args()

    if not os.path.exists(args.input_xyz):
        raise FileNotFoundError(f"Input file not found: {args.input_xyz}")

    result = planarize_dimer(args.input_xyz, args.output_xyz, dz=args.dz)
    print(f"Wrote planarized dimer to {args.output_xyz}")
    print(f"  current distance = {result['current_distance']:.3f} Å")
    print(f"  current angle    = {result['current_angle_deg']:.3f} deg")
    print(f"  new distance     = {result['new_distance']:.3f} Å")
    print(f"  new angle        = {result['new_angle_deg']:.3f} deg")


if __name__ == "__main__":
    main()

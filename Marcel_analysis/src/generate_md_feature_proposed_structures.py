#!/usr/bin/env python3
"""Generate new dimer XYZs from MD-derived feature rows.

This script keeps the existing generate_proposed_structures.py intact and
adds a new workflow that builds new dimer geometries from the MD feature CSV.
It produces:
  - a completely planar set where the second monomer remains parallel to
    the first monomer (cos_theta = 1)
  - a distorted set where the second monomer is tilted using MD cos_theta
    and MD tilt direction psi

Both sets are generated from the same reference monomer stored in
data/processed/reference_monomer.xyz.

The script also computes RMSD between original MD dimer geometries and the
newly generated structures for comparison.
Run with:
python3 src/generate_md_feature_proposed_structures.py \
  --features-file Bradly_MD_data_July-15/20000-xyz-files/P1_dimer_full_features.csv \
  --input-dir Bradly_MD_data_July-15/20000-xyz-files/P1_xyz-files \
  --output-dir data/generated_md_from_features \
  --mode both 
  """
from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ase.io import read as ase_read

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import geometry

ROOT = Path(__file__).resolve().parent.parent
REF_PATH = ROOT / "data" / "processed" / "reference_monomer.xyz"


def load_reference_monomer(reference_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    atoms = ase_read(str(reference_path))
    positions = atoms.get_positions()
    symbols = np.array(atoms.get_chemical_symbols())
    return positions, symbols


def canonicalize_monomer(positions: np.ndarray, symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid, normal, x_axis = geometry.monomer_frame(positions, symbols)
    y_axis = np.cross(normal, x_axis)
    O = np.stack([x_axis, y_axis, normal], axis=1)
    positions_centered = (positions - centroid) @ O
    return positions_centered, symbols


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


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


def decode_angle_from_4fold(cos_4x: float, sin_4x: float) -> float:
    return 0.25 * math.atan2(sin_4x, cos_4x)


def build_planar_dimer(
    mono_pos: np.ndarray,
    mono_sym: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    phi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m1_pos = mono_pos.copy()
    m1_sym = mono_sym
    R_phi = rot_z(phi)
    m2_pos = (mono_pos @ R_phi.T) + np.array([dx, dy, dz])
    m2_sym = mono_sym
    return m1_pos, m1_sym, m2_pos, m2_sym


def build_distorted_dimer(
    mono_pos: np.ndarray,
    mono_sym: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    phi: float,
    cos_theta: float,
    psi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tilt_angle = math.acos(np.clip(cos_theta, -1.0, 1.0))
    m1_pos = mono_pos.copy()
    m1_sym = mono_sym
    R_phi = rot_z(phi)
    tilt_axis = np.array([math.cos(psi), math.sin(psi), 0.0], dtype=float)
    R_tilt = rotation_about_axis(tilt_axis, tilt_angle)
    R = R_tilt @ R_phi
    m2_pos = (mono_pos @ R.T) + np.array([dx, dy, dz])
    m2_sym = mono_sym
    return m1_pos, m1_sym, m2_pos, m2_sym


def load_features(features_file: Path) -> Dict[str, Dict[str, str]]:
    with features_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        result = {}
        for row in reader:
            key = (
                row.get("Filename")
                or row.get("filename")
                or row.get("Name")
                or row.get("name")
                or ""
            )
            key = key.strip()
            if key.endswith(".xyz"):
                key = Path(key).stem
            result[key] = row
    return result


def parse_float(value: str, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    return float(value)


def write_xyz(path: Path, m1_pos: np.ndarray, m1_sym: np.ndarray, m2_pos: np.ndarray, m2_sym: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_pos = np.vstack([m1_pos, m2_pos])
    all_sym = np.concatenate([m1_sym, m2_sym])
    with path.open("w") as handle:
        handle.write(f"{len(all_sym)}\n")
        handle.write(comment + "\n")
        for symbol, (x, y, z) in zip(all_sym, all_pos):
            handle.write(f"{symbol:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")


def copy_original_xyz(source_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source_path.name
    shutil.copy2(source_path, destination_path)
    return destination_path


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    PC = P - P.mean(axis=0)
    QC = Q - Q.mean(axis=0)
    H = PC.T @ QC
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    M = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ M @ U.T
    P_rot = (R @ PC.T).T
    return float(np.sqrt(np.mean(np.sum((P_rot - QC) ** 2, axis=1))))


def rmsd_for_monomers(m1_a: np.ndarray, sym_a: np.ndarray, m1_b: np.ndarray, sym_b: np.ndarray) -> float:
    if len(sym_a) != len(sym_b):
        raise ValueError("Monomers must have the same atom count")
    sort_idx = np.argsort(sym_a)
    if not np.array_equal(sym_a[sort_idx], sym_b[np.argsort(sym_b)]):
        raise ValueError("Monomer symbol sets do not match")
    return kabsch_rmsd(m1_a[sort_idx], m1_b[np.argsort(sym_b)])


def decode_angles_from_row(row: Dict[str, str]) -> Tuple[float, float]:
    phi = decode_angle_from_4fold(parse_float(row.get("cos_4phi", "0")), parse_float(row.get("sin_4phi", "0")))
    psi = decode_angle_from_4fold(parse_float(row.get("cos_4psi", "0")), parse_float(row.get("sin_4psi", "0")))
    return phi, psi


def build_comment(row: Dict[str, str], mode: str) -> str:
    values = {
        "mode": mode,
        "dz": row.get("dz", "nan"),
        "dx": row.get("dx", "nan"),
        "dy": row.get("dy", "nan"),
        "cos_theta": row.get("cos_theta", "nan"),
        "Lateral_Slippage_Ang": row.get("Lateral_Slippage_Ang", "nan"),
        "Min_HeavyDist_Ang": row.get("Min_HeavyDist_Ang", "nan"),
        "S12_srsh_cation": row.get("S12_srsh_cation", "nan"),
    }
    return " ".join(f"{k}={v}" for k, v in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-file", required=True, type=Path, help="MD feature CSV with per-geometry rows")
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder with original MD XYZ files")
    parser.add_argument("--output-dir", default=ROOT / "data" / "generated_md_from_features", type=Path)
    parser.add_argument("--reference", default=REF_PATH, type=Path, help="Reference monomer XYZ")
    parser.add_argument("--mode", choices=["planar", "distorted", "both"], default="both", help="Which generated sets to write")
    args = parser.parse_args()

    features_file = args.features_file.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()

    if not features_file.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_file}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference monomer not found: {reference_path}")

    ref_pos, ref_sym = load_reference_monomer(reference_path)
    ref_mono_pos, ref_sym = canonicalize_monomer(ref_pos, ref_sym)
    features = load_features(features_file)

    output_planar = output_dir / "planar"
    output_distorted = output_dir / "distorted"
    summary_rows: List[Dict[str, str]] = []

    xyz_files = sorted(input_dir.glob("*.xyz"))
    for xyz_path in xyz_files:
        stem = xyz_path.stem
        feature_row = features.get(stem)
        if feature_row is None:
            print(f"Skipping {xyz_path.name}: no feature row")
            continue

        dx = parse_float(feature_row.get("dx", "nan"))
        dy = parse_float(feature_row.get("dy", "nan"))
        dz = parse_float(feature_row.get("dz", "nan"))
        cos_theta = parse_float(feature_row.get("cos_theta", "1.0"))
        phi, psi = decode_angles_from_row(feature_row)

        source_dimer = geometry.load_dimer_xyz(str(xyz_path))
        md_m1 = source_dimer["monomer1"]["positions"]
        md_m2 = source_dimer["monomer2"]["positions"]
        md_sym1 = source_dimer["monomer1"]["symbols"]
        md_sym2 = source_dimer["monomer2"]["symbols"]

        if args.mode in {"distorted", "both"}:
            copy_original_xyz(xyz_path, output_distorted)

        planar_rmsd_m1 = planar_rmsd_m2 = float("nan")
        rmsd_diff = float("nan")

        if args.mode in {"planar", "both"}:
            p1, syms1, p2, syms2 = build_distorted_dimer(
                ref_mono_pos, ref_sym, dx, dy, dz, phi, 1.0, 0.0
            )
            out_path = output_planar / f"{stem}_planar.xyz"
            comment = build_comment(feature_row, "planar")
            write_xyz(out_path, p1, syms1, p2, syms2, comment)
            planar_rmsd_m1 = rmsd_for_monomers(p1, syms1, md_m1, md_sym1)
            planar_rmsd_m2 = rmsd_for_monomers(p2, syms2, md_m2, md_sym2)
            rmsd_diff = abs(planar_rmsd_m1 - planar_rmsd_m2)
            print(
                f"{stem}: monomer1 RMSD={planar_rmsd_m1:.6f}, monomer2 RMSD={planar_rmsd_m2:.6f}, diff={rmsd_diff:.6f}"
            )

        summary_rows.append(
            {
                "Filename": stem,
                "dz": str(dz),
                "dx": str(dx),
                "dy": str(dy),
                "cos_theta": str(cos_theta),
                "Lateral_Slippage_Ang": feature_row.get("Lateral_Slippage_Ang", ""),
                "Min_HeavyDist_Ang": feature_row.get("Min_HeavyDist_Ang", ""),
                "S12_srsh_cation": feature_row.get("S12_srsh_cation", ""),
                "planar_rmsd_m1": f"{planar_rmsd_m1:.6f}",
                "planar_rmsd_m2": f"{planar_rmsd_m2:.6f}",
                "rmsd_diff": f"{rmsd_diff:.6f}",
            }
        )

    summary_file = output_dir / "rmsd_summary.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Filename", "dz", "dx", "dy", "cos_theta",
        "Lateral_Slippage_Ang", "Min_HeavyDist_Ang", "S12_srsh_cation",
        "planar_rmsd_m1", "planar_rmsd_m2", "rmsd_diff",
    ]
    with summary_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Generated {len(summary_rows)} geometries from {features_file}")
    print(f"Planar outputs in: {output_planar}")
    print(f"Original geometries copied to: {output_distorted}")
    print(f"Summary written to: {summary_file}")


if __name__ == "__main__":
    main()

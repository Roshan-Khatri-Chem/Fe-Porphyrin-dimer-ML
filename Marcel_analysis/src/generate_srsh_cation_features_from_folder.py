#!/usr/bin/env python3
"""Compute SRSH cation S12 values for a folder of dimer XYZ files.

Example:
    /Users/Roshan/opt/anaconda3/bin/python src/generate_srsh_cation_features_from_folder.py \
        --input-dir data/Roshan-test_dimer \
        --output-csv data/Roshan-test_dimer_s12_srsh_cation.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fchk_parser import (
    PERIODIC,
    build_pyscf_basis,
    normalize_to_pyscf_basis,
    parse_fchk,
    reorder_mo_coeffs_to_pyscf,
)
from geometry import load_dimer_xyz
from srsh_s12 import S12Calculator, load_srsh_homo_sph


def list_xyz_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.xyz"))
    if not files:
        raise FileNotFoundError(f"No .xyz files found in {input_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SRSH cation S12 for a set of dimer XYZ files")
    parser.add_argument("--input-dir", required=True, help="Folder containing one or more .xyz dimer files")
    parser.add_argument("--output-csv", required=True, help="Path to save the output CSV")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    fchk_path = ROOT / "Qchem-output" / "fe_mon_fchk" / "fe3_sp.in.fchk"
    if not fchk_path.exists():
        raise FileNotFoundError(f"FCHK file not found: {fchk_path}")

    print(f"Loading cation HOMO from {fchk_path}")
    c_cation, basis, atom_list_ref, _ = load_srsh_homo_sph(
        str(fchk_path),
        normalize_to_pyscf_basis,
        reorder_mo_coeffs_to_pyscf,
        build_pyscf_basis,
        parse_fchk,
        PERIODIC,
    )
    calc_cation = S12Calculator(basis, atom_list_ref, c_cation)

    xyz_files = list_xyz_files(input_dir)
    rows = []
    failures = 0
    for xyz_path in xyz_files:
        try:
            dimer = load_dimer_xyz(str(xyz_path))
            s12 = calc_cation.compute(
                dimer["monomer1"]["positions"],
                dimer["monomer1"]["symbols"],
                dimer["monomer2"]["positions"],
                dimer["monomer2"]["symbols"],
            )
            rows.append({"name": xyz_path.stem, "S12_srsh_cation": float(s12)})
            print(f"Processed {xyz_path.name}: {s12:.6f}")
        except Exception as exc:
            failures += 1
            rows.append({"name": xyz_path.stem, "S12_srsh_cation": np.nan})
            print(f"Skipped {xyz_path.name}: {exc}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "S12_srsh_cation"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")
    print(f"Failures: {failures}")


if __name__ == "__main__":
    main()

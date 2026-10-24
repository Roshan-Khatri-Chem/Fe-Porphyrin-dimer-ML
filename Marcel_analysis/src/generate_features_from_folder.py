#!/usr/bin/env python3
"""Generate 11-D geometry features for a folder of dimer XYZ files.

Example:
    python src/generate_features_from_folder.py \
        --input-dir data/Roshan-test_dimer \
        --output-csv data/Roshan-test_dimer/Roshan-test_dimer_features.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from geometry import dimer_features, load_dimer_xyz

FEATURE_NAMES = [
    "dz",
    "dx",
    "dy",
    "cos_theta",
    "cos_4phi",
    "sin_4phi",
    "cos_4psi",
    "sin_4psi",
    "d_lateral",
    "cos_4alpha",
    "sin_4alpha",
]


def list_xyz_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.xyz"))
    if not files:
        raise FileNotFoundError(f"No .xyz files found in {input_dir}")
    return files


def build_row(xyz_path: Path) -> dict:
    dimer = load_dimer_xyz(str(xyz_path))
    features = dimer_features(
        dimer["monomer1"]["positions"],
        dimer["monomer1"]["symbols"],
        dimer["monomer2"]["positions"],
        dimer["monomer2"]["symbols"],
    )
    row = {"name": xyz_path.stem}
    row.update({name: float(val) for name, val in zip(FEATURE_NAMES, features[:11])})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 11-D dimer features from XYZ files")
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
            rows.append(build_row(xyz_path))
            print(f"Processed {xyz_path.name}")
        except Exception as exc:
            print(f"Skipped {xyz_path.name}: {exc}")

    if not rows:
        raise RuntimeError("No files were processed successfully")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", *FEATURE_NAMES]
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved {len(rows)} rows to {output_csv}")
    print("Columns:", ", ".join(fieldnames))


if __name__ == "__main__":
    main()

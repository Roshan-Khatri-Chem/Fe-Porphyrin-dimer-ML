#!/usr/bin/env python3
"""Run the full dimer feature pipeline from one command.

This launcher executes the geometry-feature generator, distance/angle-feature
generator, SRSH cation S12 generator, and the merge step in sequence.

Example:
    /Users/Roshan/opt/anaconda3/bin/python src/run_full_feature_pipeline.py \
        --input-dir data/Roshan-test_dimer \
        --geometry-output data/Roshan-test_dimer_features.csv \
        --distance-output data/Roshan-test_dimer_distance_features.csv \
        --s12-output data/Roshan-test_dimer_s12_srsh_cation.csv \
        --merged-output data/Roshan-test_dimer_full_features.csv

    /Users/Roshan/opt/anaconda3/bin/python src/run_full_feature_pipeline.py \
  --input-dir data/generated_tanfunction_planar \
  --geometry-output data/generated_tanfunction_planar/P1_planar_features.csv \
  --distance-output data/generated_tanfunction_planar/P1_planar_distance_features.csv \
  --s12-output data/generated_tanfunction_planar/P1_planar_s12_srsh_cation.csv \
  --merged-output data/generated_tanfunction_planar/P1_planar_full_features.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(py_executable: str, script_path: Path, args: list[str], cwd: Path) -> None:
    cmd = [py_executable, str(script_path), *args]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full dimer feature pipeline")
    parser.add_argument("--input-dir", default="data/Roshan-test_dimer", help="Folder containing .xyz dimer files")
    parser.add_argument("--geometry-output", default="data/Roshan-test_dimer_features.csv", help="Output CSV for the 11 geometry features")
    parser.add_argument("--distance-output", default="data/Roshan-test_dimer_distance_features.csv", help="Output CSV for distance/angle features")
    parser.add_argument("--s12-output", default="data/Roshan-test_dimer_s12_srsh_cation.csv", help="Output CSV for SRSH cation S12 values")
    parser.add_argument("--merged-output", default="data/Roshan-test_dimer_full_features.csv", help="Merged full feature CSV")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    py_executable = sys.executable

    geometry_script = root / "src" / "generate_features_from_folder.py"
    distance_script = root / "src" / "generate_distance_features_from_folder.py"
    s12_script = root / "src" / "generate_srsh_cation_features_from_folder.py"
    merge_script = root / "src" / "merge_feature_csvs.py"

    run_step(
        py_executable,
        geometry_script,
        ["--input-dir", str(Path(args.input_dir).expanduser().resolve()), "--output-csv", str(Path(args.geometry_output).expanduser().resolve())],
        root,
    )
    run_step(
        py_executable,
        distance_script,
        ["--input-dir", str(Path(args.input_dir).expanduser().resolve()), "--output-csv", str(Path(args.distance_output).expanduser().resolve())],
        root,
    )
    run_step(
        py_executable,
        s12_script,
        ["--input-dir", str(Path(args.input_dir).expanduser().resolve()), "--output-csv", str(Path(args.s12_output).expanduser().resolve())],
        root,
    )
    run_step(
        py_executable,
        merge_script,
        [
            "--geometry-csv",
            str(Path(args.geometry_output).expanduser().resolve()),
            "--distance-csv",
            str(Path(args.distance_output).expanduser().resolve()),
            "--s12-csv",
            str(Path(args.s12_output).expanduser().resolve()),
            "--output-csv",
            str(Path(args.merged_output).expanduser().resolve()),
        ],
        root,
    )

    print("Full feature pipeline completed successfully")
    print(f"Merged output: {Path(args.merged_output).expanduser().resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge geometry, distance/angle, and SRSH cation feature CSVs.

Example:
    /Users/Roshan/opt/anaconda3/bin/python src/merge_feature_csvs.py \
        --geometry-csv data/Roshan-test_dimer_features.csv \
        --distance-csv data/Roshan-test_dimer_distance_features.csv \
        --s12-csv data/Roshan-test_dimer_s12_srsh_cation.csv \
        --output-csv data/Roshan-test_dimer_full_features.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge geometry, distance/angle, and S12 feature CSVs")
    parser.add_argument("--geometry-csv", required=True, help="CSV containing the geometry features")
    parser.add_argument("--distance-csv", help="Optional CSV containing distance/angle features")
    parser.add_argument("--s12-csv", help="Optional CSV containing the S12 feature")
    parser.add_argument("--output-csv", required=True, help="Path to the merged output CSV")
    args = parser.parse_args()

    geometry_csv = Path(args.geometry_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not geometry_csv.exists():
        raise FileNotFoundError(f"Geometry CSV not found: {geometry_csv}")

    with open(geometry_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))

    if args.distance_csv:
        distance_csv = Path(args.distance_csv).expanduser().resolve()
        if not distance_csv.exists():
            raise FileNotFoundError(f"Distance CSV not found: {distance_csv}")
        with open(distance_csv, newline="") as handle:
            distance_rows = list(csv.DictReader(handle))
        distance_map = {row["name"]: row for row in distance_rows}
        for row in rows:
            dist_row = distance_map.get(row.get("name", ""), {})
            for key, value in dist_row.items():
                if key != "name":
                    row[key] = value

    if args.s12_csv:
        s12_csv = Path(args.s12_csv).expanduser().resolve()
        if not s12_csv.exists():
            raise FileNotFoundError(f"S12 CSV not found: {s12_csv}")
        with open(s12_csv, newline="") as handle:
            s12_rows = list(csv.DictReader(handle))
        s12_map = {row["name"]: row for row in s12_rows}
        for row in rows:
            s12_row = s12_map.get(row.get("name", ""), {})
            for key, value in s12_row.items():
                if key != "name":
                    row[key] = value

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(rows)} rows into {output_csv}")


if __name__ == "__main__":
    main()

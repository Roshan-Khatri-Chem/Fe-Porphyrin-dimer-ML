#!/usr/bin/env python3
"""Convert raw MD geometry XYZ files into cleaned XYZ files.

This script reads the raw geometry files from the MD export folder,
removes the unnecessary atoms using the same trimming logic as the
existing cleanup scripts, and writes standard XYZ output files into
an output directory.

Example:
python Bradly_MD_data_July-15/20000-xyz-files/process_geom_files_to_xyz.py \
  --input-dir Bradly_MD_data_July-15/20000-xyz-files/frameP1_extract \
  --output-dir Bradly_MD_data_July-15/20000-xyz-files/P1_xyz-files
  
    python process_geom_files_to_xyz.py \
        --input-dir frameP1_extract \
        --output-dir P1_xyz-files
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
import geometry


def parse_xyz_atoms(input_file: Path) -> List[Dict[str, Any]]:
    atoms: List[Dict[str, Any]] = []
    with input_file.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        elem = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        atoms.append({"elem": elem, "pos": np.array([x, y, z], dtype=float)})

    return atoms


def clean_element_name(elem: str) -> str:
    return re.sub(r"[0-9]", "", elem)


def process_single_file(input_file: Path, output_file: Path, ref_mono_pos: np.ndarray, ref_mono_sym: List[str]) -> bool:
    raw_atoms = parse_xyz_atoms(input_file)

    if len(raw_atoms) < 154:
        print(f"[!] Warning: {input_file} has {len(raw_atoms)} atoms (expected 154). Skipping.")
        return False

    m1_c_remove = {1, 2, 3, 5, 6, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 43, 44, 45, 46, 47}
    m1_h_remove = {49, 50, 51, 52, 76, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75}
    m1_replace_h = {4, 28, 34, 42}

    m2_c_remove = {78, 79, 80, 82, 83, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 120, 121, 122, 123, 124}
    m2_h_remove = {126, 127, 128, 129, 153, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 148, 149, 150, 151, 152}
    m2_replace_h = {81, 105, 111, 119}

    to_remove = m1_c_remove | m1_h_remove | m2_c_remove | m2_h_remove
    to_replace = m1_replace_h | m2_replace_h

    intermediate_atoms = []
    for idx, atom in enumerate(raw_atoms, start=1):
        if idx in to_remove:
            continue

        elem = atom["elem"]
        is_replaced_h = False
        if idx in to_replace:
            elem = "H"
            is_replaced_h = True

        intermediate_atoms.append(
            {"elem": elem, "pos": atom["pos"].copy(), "is_replaced_h": is_replaced_h}
        )

    # coords/elements will be computed after any reordering below
    # Reorder atoms within each monomer to match reference monomer ordering.
    # We assume intermediate_atoms contains monomer1 then monomer2 (equal halves).
    M = len(intermediate_atoms) // 2
    def reorder_monomer(atoms_slice, ref_pos, ref_sym):
        pos = np.array([a["pos"] for a in atoms_slice], dtype=float)
        sym = [clean_element_name(a["elem"]) for a in atoms_slice]
        # Build monomer frame and orient into local canonical frame
        try:
            centroid, normal, x_axis = geometry.monomer_frame(pos, np.array(sym))
        except Exception:
            # fallback: no reordering
            return atoms_slice
        y_axis = np.cross(normal, x_axis)
        O = np.stack([x_axis, y_axis, normal], axis=1)
        pos_local = (pos - centroid) @ O
        # Reference canonical positions: assume ref_pos already canonicalized in caller
        ref_pos_local = ref_pos
        ref_sym_list = list(ref_sym)

        assigned = set()
        order = []
        for r_sym, r_pos in zip(ref_sym_list, ref_pos_local):
            # find nearest unassigned atom in pos_local with matching element
            best_idx = None
            best_d = float('inf')
            for i, (p, s) in enumerate(zip(pos_local, sym)):
                if i in assigned:
                    continue
                if s != r_sym:
                    continue
                d = np.linalg.norm(p - r_pos)
                if d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx is not None:
                assigned.add(best_idx)
                order.append(atoms_slice[best_idx])
        # append any unassigned atoms at end (preserve original order)
        for i, a in enumerate(atoms_slice):
            if i not in assigned:
                order.append(a)
        return order

    if len(intermediate_atoms) % 2 == 0 and len(intermediate_atoms) >= 2 * M:
        m1_slice = intermediate_atoms[:M]
        m2_slice = intermediate_atoms[M:2 * M]
        new_m1 = reorder_monomer(m1_slice, ref_mono_pos, ref_mono_sym)
        new_m2 = reorder_monomer(m2_slice, ref_mono_pos, ref_mono_sym)
        intermediate_atoms = new_m1 + new_m2

    # Recompute coords/elements after potential reordering so indices match
    coords = np.array([a["pos"] for a in intermediate_atoms], dtype=float)
    elements = [a["elem"] for a in intermediate_atoms]
    c_indices = [i for i, e in enumerate(elements) if e == "C"]

    target_ch_dist = 1.08
    for i, atom in enumerate(intermediate_atoms):
        if atom["is_replaced_h"]:
            if not c_indices:
                continue
            dists = [np.linalg.norm(coords[i] - coords[j]) for j in c_indices]
            nearest_c_idx = c_indices[int(np.argmin(dists))]
            c_pos = coords[nearest_c_idx]
            h_pos = coords[i]
            vec = h_pos - c_pos
            norm = np.linalg.norm(vec)
            if norm == 0.0:
                continue
            unit_vec = vec / norm
            new_h_pos = c_pos + unit_vec * target_ch_dist
            intermediate_atoms[i]["pos"] = new_h_pos

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(intermediate_atoms)}\n")
        handle.write(f"Cleaned iron-porphyrin dimer frame ({input_file.name})\n")
        for atom in intermediate_atoms:
            elem = clean_element_name(atom["elem"])
            p = atom["pos"]
            handle.write(f"{elem:<3} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw MD geometry files into cleaned XYZ files")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw geom_*.xyz files")
    parser.add_argument("--output-dir", required=True, help="Directory to write cleaned XYZ files")
    parser.add_argument("--reference", required=False, help="Reference monomer XYZ to match ordering", default=str(Path(__file__).resolve().parents[2] / "data" / "processed" / "reference_monomer.xyz"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    reference_path = Path(args.reference).expanduser().resolve()
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference monomer not found: {reference_path}")

    # Load and canonicalize reference monomer positions/symbols into monomer frame
    ref_atoms = parse_xyz_atoms(reference_path)
    ref_positions = np.array([a["pos"] for a in ref_atoms], dtype=float)
    ref_symbols = np.array([clean_element_name(a["elem"]) for a in ref_atoms])
    try:
        centroid, normal, x_axis = geometry.monomer_frame(ref_positions, ref_symbols)
        y_axis = np.cross(normal, x_axis)
        O = np.stack([x_axis, y_axis, normal], axis=1)
        ref_pos_local = (ref_positions - centroid) @ O
    except Exception:
        ref_pos_local = ref_positions

    input_files = sorted(input_dir.glob("geom_*.xyz"))
    if not input_files:
        print(f"No geom_*.xyz files found in {input_dir}")
        return

    print(f"Found {len(input_files)} geometry file(s) to process")

    processed_count = 0
    for input_file in input_files:
        output_file = output_dir / input_file.name
        success = process_single_file(input_file, output_file, ref_pos_local, ref_symbols)
        if success:
            print(f"[✓] {input_file.name} -> {output_file.name}")
            processed_count += 1

    print(f"\nDone! Wrote {processed_count} cleaned XYZ files to {output_dir}")


if __name__ == "__main__":
    main()

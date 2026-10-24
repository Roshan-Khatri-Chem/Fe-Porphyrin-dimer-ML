#!/usr/bin/env python3
import os
import re
import pandas as pd

input_dir = "../"
output_filename = "CDFT_CI_output_analysis.csv"

conversion_factor = 27.2114  # Hartree → eV


def extract_cdft_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    # =========================
    # 1️⃣ Coupling (eV)
    # =========================
    coupling = None
    match = re.search(
        r"CDFT-CI Hamiltonian matrix in orthogonalized basis.*?\n\s*1.*?\s+([-+]?\d*\.\d+)",
        text, re.DOTALL)
    if match:
        coupling = abs(float(match.group(1))) * conversion_factor

    # =========================
    # 2️⃣ Diabatic Energies
    # =========================
    diabatic_sections = re.split(
        r"beginning diabatic state SCF calculation\.",
        text)

    diabatic_energies = []
    for section in diabatic_sections[1:]:
        m = re.search(r"SCF Energy \(H0 \+ V/2\)\s*=\s*([-]?\d+\.\d+)", section)
        if m:
            diabatic_energies.append(float(m.group(1)))

    E1 = E2 = gap = total_energy = None
    if len(diabatic_energies) >= 2:
        E1, E2 = diabatic_energies[:2]
        gap = (E2 - E1) * conversion_factor
        total_energy = E1 + E2   # ✅ YOUR DEFINITION

    # =========================
    # 3️⃣ Overlap (S12)
    # =========================
    overlap = None
    match = re.search(
        r"CDFT-CI overlap matrix.*?\n\s*1\s+1\.0+\s+([-+]?\d*\.\d+)",
        text, re.DOTALL)
    if match:
        overlap = float(match.group(1))

    return coupling, E1, E2, gap, total_energy, overlap


# =========================
# Loop over files
# =========================
results = []

for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".out"):
        filepath = os.path.join(input_dir, filename)

        coupling, E1, E2, gap, total_energy, overlap = extract_cdft_data(filepath)

        results.append((
            filename,
            coupling,
            E1,
            E2,
            gap,
            total_energy,
            overlap
        ))


# =========================
# Save results
# =========================
df = pd.DataFrame(results, columns=[
    "Filename",
    "Coupling_eV",
    "Diabatic_E1_Ha",
    "Diabatic_E2_Ha",
    "Energy_Gap_eV",
    "Total_Energy_Ha",
    "Overlap_S12"
])

df_sorted = df.sort_values(by="Coupling_eV", ascending=False)
df_sorted.to_csv(output_filename, index=False)

print(f"✅ Done! Data saved to {output_filename}")

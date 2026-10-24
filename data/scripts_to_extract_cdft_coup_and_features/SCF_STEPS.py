import os
import re
import pandas as pd

folder = "../"
results = []

for file in sorted(os.listdir(folder)):
    if not file.endswith(".out"):
        continue

    filepath = os.path.join(folder, file)

    # Store results
    data = {
        "Filename": file,

        "SCF1_steps": 0,
        "SCF2_steps": 0,
        "SCF1_energy": None,
        "SCF2_energy": None,

        "D1_steps": 0,
        "D2_steps": 0,
        "D1_energy": None,
        "D2_energy": None,
    }

    current_state = None
    current_mode = None   # "CDFT" or "DIAB"
    counting = False

    with open(filepath, "r") as f:
        for line in f:

            # -------------------------------
            # Detect CDFT-CI SCF start
            # -------------------------------
            if "Starting CDFT-CI SCF calculation on state" in line:
                match = re.search(r"state\s+(\d+)", line)
                if match:
                    current_state = int(match.group(1))
                    current_mode = "CDFT"
                    counting = False
                continue

            # -------------------------------
            # Detect diabatic SCF start
            # -------------------------------
            if "beginning diabatic state SCF calculation" in line:
                current_mode = "DIAB"
                counting = False
                continue

            # -------------------------------
            # Detect SCF table start
            # -------------------------------
            if "Cycle" in line and "Energy" in line:
                if current_state in [1, 2] and current_mode:
                    counting = True
                continue

            # -------------------------------
            # Count cycles
            # -------------------------------
            if counting and current_state in [1, 2]:
                if re.match(r"\s*\d+\s+", line):
                    if current_mode == "CDFT":
                        data[f"SCF{current_state}_steps"] += 1
                    elif current_mode == "DIAB":
                        data[f"D{current_state}_steps"] += 1

                # -------------------------------
                # Detect convergence + energy
                # -------------------------------
                if "Convergence criterion met" in line:
                    energy_match = re.search(r"(-\d+\.\d+)", line)

                    if energy_match:
                        energy = float(energy_match.group(1))

                        if current_mode == "CDFT":
                            data[f"SCF{current_state}_energy"] = energy
                        elif current_mode == "DIAB":
                            data[f"D{current_state}_energy"] = energy

                    counting = False
                    # NOTE: do NOT reset state here (diabatic follows same state)

    results.append(data)

# Convert to DataFrame
df = pd.DataFrame(results)

# Save
df.to_csv("cdftci_diabatic_scf_summary.csv", index=False)

print(df.head())

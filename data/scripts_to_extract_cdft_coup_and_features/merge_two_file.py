import pandas as pd

# Load both files
df_main = pd.read_csv("111_md_added.csv")
df_geom = pd.read_csv("geometry_parameters.csv")

# =========================
# Merge on Filename
# =========================
df_merged = df_main.merge(df_geom, on="XYZ_File", how="left")

# =========================
# Move theta, psi, phi to the far right (optional but requested)
# =========================
angle_cols = ["theta", "psi", "phi"]

# Reorder columns
cols = [c for c in df_merged.columns if c not in angle_cols] + angle_cols
df_merged = df_merged[cols]

# =========================
# Save
# =========================
df_merged.to_csv("111_final_dataset_with_angles.csv", index=False)

print("✅ Saved -> final_dataset_with_angles.csv")

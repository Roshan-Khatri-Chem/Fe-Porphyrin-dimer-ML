#!/usr/bin/env python3

import pandas as pd

# =========================================================
# Load files
# =========================================================
df1 = pd.read_csv("808_final_with_log_sqrt.csv")
df2 = pd.read_csv("master_dataset.csv")

# =========================================================
# Remove .out extension from df1 filenames
# Example:
# geom_188.out  -> geom_188
# =========================================================
df1["Filename_clean"] = df1["Filename"].str.replace(
    ".out",
    "",
    regex=False
)

# =========================================================
# Merge using cleaned filename
# =========================================================
merged_df = pd.merge(
    df1,
    df2,
    left_on="Filename_clean",
    right_on="Filename",
    how="left",
    suffixes=("", "_master")
)

# =========================================================
# Optional:
# remove duplicate filename columns
# =========================================================
merged_df.drop(columns=["Filename_clean", "Filename_master"],
               inplace=True,
               errors="ignore")

# =========================================================
# Save merged file
# =========================================================
output_file = "808_master_dataset.csv"

merged_df.to_csv(output_file, index=False)

print(f"Merged file saved as: {output_file}")

# =========================================================
# Check matching
# =========================================================
matched = merged_df.notna().all(axis=1).sum()

print(f"Total rows in df1: {len(df1)}")
print(f"Rows with matches: {matched}")

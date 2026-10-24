import os
import pandas as pd
import numpy as np
from ase.io import read

# =========================
# Geometry helper functions
# =========================
def _best_fit_plane_normal(points):
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    n = vh[-1]
    n /= np.linalg.norm(n)
    return n

def _principal_inplane_axis(points, normal):
    centered = points - points.mean(axis=0)
    proj = centered - np.outer(centered @ normal, normal)
    _, _, vh = np.linalg.svd(proj, full_matrices=False)
    e1 = vh[0]
    e1 -= (e1 @ normal) * normal
    e1 /= np.linalg.norm(e1)
    return e1


# =========================
# Main feature extraction
# =========================
def compute_all_features(xyz_path, mon1_idx=range(0,37), mon2_idx=range(37,74), anchor_symbol="N"):
    try:
        atoms = read(xyz_path)
        pos = atoms.get_positions()
        sym = atoms.get_chemical_symbols()

        # =========================
        # Fe positions
        # =========================
        fe1 = np.array([pos[i] for i in mon1_idx if sym[i] == "Fe"])[0]
        fe2 = np.array([pos[i] for i in mon2_idx if sym[i] == "Fe"])[0]

        # =========================
        # Anchor atoms (N)
        # =========================
        n1 = np.array([pos[i] for i in mon1_idx if sym[i] == anchor_symbol])
        n2 = np.array([pos[i] for i in mon2_idx if sym[i] == anchor_symbol])

        if len(n1) < 3 or len(n2) < 3:
            raise ValueError("Not enough anchor atoms")

        # =========================
        # Plane normals
        # =========================
        n1_norm = _best_fit_plane_normal(n1)
        n2_norm = _best_fit_plane_normal(n2)

        if n1_norm @ n2_norm < 0:
            n2_norm = -n2_norm

        n_avg = (n1_norm + n2_norm)
        n_avg /= np.linalg.norm(n_avg)

        # =========================
        # Distances & slippage
        # =========================
        v = fe2 - fe1
        d_perp = abs(v @ n_avg)
        v_para = v - (v @ n_avg) * n_avg
        s_mag = np.linalg.norm(v_para)

        e1 = _principal_inplane_axis(n1, n1_norm)
        e2 = np.cross(n_avg, e1)
        e2 /= np.linalg.norm(e2)

        s1 = v_para @ e1
        s2 = v_para @ e2

        phi = np.degrees(np.arctan2(s2, s1))
        tilt = np.degrees(np.arccos(np.clip(n1_norm @ n2_norm, -1.0, 1.0)))

        # =========================
        # Centroid features
        # =========================
        centroid1 = np.mean(pos[mon1_idx], axis=0)
        centroid2 = np.mean(pos[mon2_idx], axis=0)

        centroid_vec = centroid2 - centroid1
        centroid_dist = np.linalg.norm(centroid_vec)

        centroid_unit = centroid_vec / centroid_dist if centroid_dist != 0 else np.zeros(3)

        angle_centroid_plane = np.degrees(np.arccos(
            np.clip(np.dot(centroid_vec, n1_norm)/centroid_dist, -1.0, 1.0)
        )) if centroid_dist != 0 else np.nan

        # =========================
        # N–N distances (16)
        # =========================
        nn_distances = []
        nn_pairs = []

        for i, ni in enumerate(n1):
            for j, nj in enumerate(n2):
                dist = np.linalg.norm(ni - nj)
                nn_distances.append(dist)
                nn_pairs.append(f"{i}-{j}")

        idx = np.argsort(nn_distances)
        nn_distances = [nn_distances[i] for i in idx]
        nn_pairs = [nn_pairs[i] for i in idx]

        # =========================
        # Heavy atom min/max
        # =========================
        heavy_elements = {"C", "N", "Fe"}

        heavy1 = [i for i in mon1_idx if sym[i] in heavy_elements]
        heavy2 = [i for i in mon2_idx if sym[i] in heavy_elements]

        min_dist = max_dist = np.nan
        min_pair = max_pair = ""
        min_elem = max_elem = ""

        if heavy1 and heavy2:
            P1 = pos[heavy1]
            P2 = pos[heavy2]

            diff = P1[:, None, :] - P2[None, :, :]
            d = np.sqrt(np.sum(diff**2, axis=2))

            i_min, j_min = np.unravel_index(np.argmin(d), d.shape)
            i_max, j_max = np.unravel_index(np.argmax(d), d.shape)

            min_dist = d[i_min, j_min]
            max_dist = d[i_max, j_max]

            min_pair = f"{heavy1[i_min]}-{heavy2[j_min]}"
            max_pair = f"{heavy1[i_max]}-{heavy2[j_max]}"

            min_elem = f"{sym[heavy1[i_min]]}-{sym[heavy2[j_min]]}"
            max_elem = f"{sym[heavy1[i_max]]}-{sym[heavy2[j_max]]}"

        # =========================
        # Final dictionary
        # =========================
        result = {
            "Interlayer_Dist_Ang": d_perp,
            "Lateral_Slippage_Ang": s_mag,
            "Slippage_X_Ang": s1,
            "Slippage_Y_Ang": s2,
            "Slippage_Angle_Deg": phi,
            "Plane_Tilt_Deg": tilt,
            "FeFe_Dist_Ang": np.linalg.norm(v),

            "Centroid_Dist_Ang": centroid_dist,
            "Centroid_dx": centroid_vec[0],
            "Centroid_dy": centroid_vec[1],
            "Centroid_dz": centroid_vec[2],
            "Centroid_dx_norm": centroid_unit[0],
            "Centroid_dy_norm": centroid_unit[1],
            "Centroid_dz_norm": centroid_unit[2],
            "Centroid_PlaneAngle_Deg": angle_centroid_plane,

            # NEW FEATURES
            "Min_HeavyDist_Ang": min_dist,
            "Max_HeavyDist_Ang": max_dist,
            "MinPair": min_pair,
            "MaxPair": max_pair,
            "MinElements": min_elem,
            "MaxElements": max_elem
        }

        # Add NN distances
        for k, (p, dval) in enumerate(zip(nn_pairs, nn_distances), start=1):
            result[f"NN_Dist_{k}_Ang"] = dval
            result[f"NN_Pair_{k}"] = p

        return result

    except Exception as e:
        print(f"Error processing {xyz_path}: {e}")
        return {}


# =========================
# Batch processing
# =========================
df = pd.read_csv("CDFT_CI_output_analysis.csv")

records = []

for fname in df["Filename"]:
    xyz_file = fname.replace(".out", ".xyz")

    geom = compute_all_features(xyz_file)

    row = {"Filename": fname, "XYZ_File": xyz_file}
    row.update(geom)

    records.append(row)

geom_df = pd.DataFrame(records)

# Merge
out = df.merge(geom_df, on="Filename", how="left")

out.to_csv("PD_final_dataset_with_all_features.csv", index=False)

print("✅ Saved -> final_dataset_with_all_features.csv")

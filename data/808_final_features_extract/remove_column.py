import pandas as pd

df = pd.read_csv("808_final.csv")

df = df.drop(columns=[
    "NN_Pair_16",
    "NN_Pair_15",
    "NN_Pair_14",
   "NN_Pair_13",
    "NN_Pair_12",
    "NN_Pair_11",
    "NN_Pair_10",
    "NN_Pair_9",
    "NN_Pair_8",
   "NN_Pair_7",
    "NN_Pair_6",
    "NN_Pair_5",
    "NN_Pair_4",
    "NN_Pair_3",
    "NN_Pair_2",
    "NN_Pair_1",
    "MinPair",
    "MaxPair",
    "MinElements",
    "MaxElements",
    "XYZ_File",
    "Centroid_dx_norm",
    "Centroid_dy_norm",
    "Centroid-dz_norm",
    #"NN_Dist_1_Ang",
    "NN_Dist_2_Ang",
    "NN_Dist_3_Ang",
    "NN_Dist_4_Ang",
    "NN_Dist_5_Ang",
    "NN_Dist_6_Ang",
    "NN_Dist_7_Ang",
    "NN_Dist_8_Ang",
    "NN_Dist_9_Ang",
    "NN_Dist_10_Ang",
   "NN_Dist_11_Ang",
    "NN_Dist_12_Ang",
   "NN_Dist_13_Ang",
    "NN_Dist_14_Ang",
    "NN_Dist_15_Ang",
    "NN_Dist_16_Ang"
], errors="ignore")

df.to_csv("808_master_dataset_v2.csv", index=False)

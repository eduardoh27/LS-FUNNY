import os
import numpy as np
import pandas as pd
import argparse

# --------------------------------------------------
# Parameter configuration
# --------------------------------------------------

parser = argparse.ArgumentParser(description="Divides features in N windows.")
parser.add_argument("-n", "--num_windows", type=int, default=5, help="Number of windows (default: 5)")
args = parser.parse_args()

N = args.num_windows

# List of features to extract
feature_list = (
    "confidence, success, gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, gaze_angle_x, gaze_angle_y, eye_lmk_x_0, eye_lmk_x_1, eye_lmk_x_2, eye_lmk_x_3, eye_lmk_x_4, eye_lmk_x_5, eye_lmk_x_6, eye_lmk_x_7, eye_lmk_x_8, eye_lmk_x_9, eye_lmk_x_10, eye_lmk_x_11, eye_lmk_x_12, eye_lmk_x_13, eye_lmk_x_14, eye_lmk_x_15, eye_lmk_x_16, eye_lmk_x_17, eye_lmk_x_18, eye_lmk_x_19, eye_lmk_x_20, eye_lmk_x_21, eye_lmk_x_22, eye_lmk_x_23, eye_lmk_x_24, eye_lmk_x_25, eye_lmk_x_26, eye_lmk_x_27, eye_lmk_x_28, eye_lmk_x_29, eye_lmk_x_30, eye_lmk_x_31, eye_lmk_x_32, eye_lmk_x_33, eye_lmk_x_34, eye_lmk_x_35, eye_lmk_x_36, eye_lmk_x_37, eye_lmk_x_38, eye_lmk_x_39, eye_lmk_x_40, eye_lmk_x_41, eye_lmk_x_42, eye_lmk_x_43, eye_lmk_x_44, eye_lmk_x_45, eye_lmk_x_46, eye_lmk_x_47, eye_lmk_x_48, eye_lmk_x_49, eye_lmk_x_50, eye_lmk_x_51, eye_lmk_x_52, eye_lmk_x_53, eye_lmk_x_54, eye_lmk_x_55, eye_lmk_y_0, eye_lmk_y_1, eye_lmk_y_2, eye_lmk_y_3, eye_lmk_y_4, eye_lmk_y_5, eye_lmk_y_6, eye_lmk_y_7, eye_lmk_y_8, eye_lmk_y_9, eye_lmk_y_10, eye_lmk_y_11, eye_lmk_y_12, eye_lmk_y_13, eye_lmk_y_14, eye_lmk_y_15, eye_lmk_y_16, eye_lmk_y_17, eye_lmk_y_18, eye_lmk_y_19, eye_lmk_y_20, eye_lmk_y_21, eye_lmk_y_22, eye_lmk_y_23, eye_lmk_y_24, eye_lmk_y_25, eye_lmk_y_26, eye_lmk_y_27, eye_lmk_y_28, eye_lmk_y_29, eye_lmk_y_30, eye_lmk_y_31, eye_lmk_y_32, eye_lmk_y_33, eye_lmk_y_34, eye_lmk_y_35, eye_lmk_y_36, eye_lmk_y_37, eye_lmk_y_38, eye_lmk_y_39, eye_lmk_y_40, eye_lmk_y_41, eye_lmk_y_42, eye_lmk_y_43, eye_lmk_y_44, eye_lmk_y_45, eye_lmk_y_46, eye_lmk_y_47, eye_lmk_y_48, eye_lmk_y_49, eye_lmk_y_50, eye_lmk_y_51, eye_lmk_y_52, eye_lmk_y_53, eye_lmk_y_54, eye_lmk_y_55, eye_lmk_X_0, eye_lmk_X_1, eye_lmk_X_2, eye_lmk_X_3, eye_lmk_X_4, eye_lmk_X_5, eye_lmk_X_6, eye_lmk_X_7, eye_lmk_X_8, eye_lmk_X_9, eye_lmk_X_10, eye_lmk_X_11, eye_lmk_X_12, eye_lmk_X_13, eye_lmk_X_14, eye_lmk_X_15, eye_lmk_X_16, eye_lmk_X_17, eye_lmk_X_18, eye_lmk_X_19, eye_lmk_X_20, eye_lmk_X_21, eye_lmk_X_22, eye_lmk_X_23, eye_lmk_X_24, eye_lmk_X_25, eye_lmk_X_26, eye_lmk_X_27, eye_lmk_X_28, eye_lmk_X_29, eye_lmk_X_30, eye_lmk_X_31, eye_lmk_X_32, eye_lmk_X_33, eye_lmk_X_34, eye_lmk_X_35, eye_lmk_X_36, eye_lmk_X_37, eye_lmk_X_38, eye_lmk_X_39, eye_lmk_X_40, eye_lmk_X_41, eye_lmk_X_42, eye_lmk_X_43, eye_lmk_X_44, eye_lmk_X_45, eye_lmk_X_46, eye_lmk_X_47, eye_lmk_X_48, eye_lmk_X_49, eye_lmk_X_50, eye_lmk_X_51, eye_lmk_X_52, eye_lmk_X_53, eye_lmk_X_54, eye_lmk_X_55, eye_lmk_Y_0, eye_lmk_Y_1, eye_lmk_Y_2, eye_lmk_Y_3, eye_lmk_Y_4, eye_lmk_Y_5, eye_lmk_Y_6, eye_lmk_Y_7, eye_lmk_Y_8, eye_lmk_Y_9, eye_lmk_Y_10, eye_lmk_Y_11, eye_lmk_Y_12, eye_lmk_Y_13, eye_lmk_Y_14, eye_lmk_Y_15, eye_lmk_Y_16, eye_lmk_Y_17, eye_lmk_Y_18, eye_lmk_Y_19, eye_lmk_Y_20, eye_lmk_Y_21, eye_lmk_Y_22, eye_lmk_Y_23, eye_lmk_Y_24, eye_lmk_Y_25, eye_lmk_Y_26, eye_lmk_Y_27, eye_lmk_Y_28, eye_lmk_Y_29, eye_lmk_Y_30, eye_lmk_Y_31, eye_lmk_Y_32, eye_lmk_Y_33, eye_lmk_Y_34, eye_lmk_Y_35, eye_lmk_Y_36, eye_lmk_Y_37, eye_lmk_Y_38, eye_lmk_Y_39, eye_lmk_Y_40, eye_lmk_Y_41, eye_lmk_Y_42, eye_lmk_Y_43, eye_lmk_Y_44, eye_lmk_Y_45, eye_lmk_Y_46, eye_lmk_Y_47, eye_lmk_Y_48, eye_lmk_Y_49, eye_lmk_Y_50, eye_lmk_Y_51, eye_lmk_Y_52, eye_lmk_Y_53, eye_lmk_Y_54, eye_lmk_Y_55, eye_lmk_Z_0, eye_lmk_Z_1, eye_lmk_Z_2, eye_lmk_Z_3, eye_lmk_Z_4, eye_lmk_Z_5, eye_lmk_Z_6, eye_lmk_Z_7, eye_lmk_Z_8, eye_lmk_Z_9, eye_lmk_Z_10, eye_lmk_Z_11, eye_lmk_Z_12, eye_lmk_Z_13, eye_lmk_Z_14, eye_lmk_Z_15, eye_lmk_Z_16, eye_lmk_Z_17, eye_lmk_Z_18, eye_lmk_Z_19, eye_lmk_Z_20, eye_lmk_Z_21, eye_lmk_Z_22, eye_lmk_Z_23, eye_lmk_Z_24, eye_lmk_Z_25, eye_lmk_Z_26, eye_lmk_Z_27, eye_lmk_Z_28, eye_lmk_Z_29, eye_lmk_Z_30, eye_lmk_Z_31, eye_lmk_Z_32, eye_lmk_Z_33, eye_lmk_Z_34, eye_lmk_Z_35, eye_lmk_Z_36, eye_lmk_Z_37, eye_lmk_Z_38, eye_lmk_Z_39, eye_lmk_Z_40, eye_lmk_Z_41, eye_lmk_Z_42, eye_lmk_Z_43, eye_lmk_Z_44, eye_lmk_Z_45, eye_lmk_Z_46, eye_lmk_Z_47, eye_lmk_Z_48, eye_lmk_Z_49, eye_lmk_Z_50, eye_lmk_Z_51, eye_lmk_Z_52, eye_lmk_Z_53, eye_lmk_Z_54, eye_lmk_Z_55, pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz, p_scale, p_rx, p_ry, p_rz, p_tx, p_ty, p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9, p_10, p_11, p_12, p_13, p_14, p_15, p_16, p_17, p_18, p_19, p_20, p_21, p_22, p_23, p_24, p_25, p_26, p_27, p_28, p_29, p_30, p_31, p_32, p_33, AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r, AU01_c, AU02_c, AU04_c, AU05_c, AU06_c, AU07_c, AU09_c, AU10_c, AU12_c, AU14_c, AU15_c, AU17_c, AU20_c, AU23_c, AU25_c, AU26_c, AU28_c, AU45_c"
).split(", ")

# Input and output directories
input_dir = "features/open_face_raw"
output_dir = f"features/open_face_{N}_full"
os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------------
# Processing each CSV file
# --------------------------------------------------

for fname in os.listdir(input_dir):
    if not fname.endswith(".csv"):
        continue

    # Read only relevant columns
    path_in = os.path.join(input_dir, fname)
    df = pd.read_csv(path_in, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[feature_list]

    # Convert everything to numeric; unconvertible values become NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Calculate split indices for N windows
    total_frames = len(df)
    cuts = np.linspace(0, total_frames, N + 1, dtype=int)

    # For each window, compute the mean of each feature
    window_means = []
    for i in range(N):
        start_idx, end_idx = cuts[i], cuts[i + 1]
        window = df.iloc[start_idx:end_idx]
        window_means.append(window.mean().values)

    arr = np.stack(window_means)
    # Flatten to a single vector (1, N * num_features)
    flat = arr.reshape(1, -1)

    base_name = os.path.splitext(fname)[0]
    out_path = os.path.join(output_dir, base_name + ".npy")
    np.save(out_path, flat)

    print(f"Processed {fname} → {out_path} (shape {flat.shape})")

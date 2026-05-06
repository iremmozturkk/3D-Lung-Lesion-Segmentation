import os
import time
import numpy as np
import pandas as pd
import nibabel as nib

# =====================================================
# BASE DIRECTORY
# =====================================================

BASE_DIR = r"D:/GÜNCEL STAJ PROJESİ/MediBoxLesion"

# =====================================================
# PATHS
# =====================================================

CSV_PATH = os.path.join(
    BASE_DIR,
    "data/mela/annotations/mela_master_dataframe.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results/performance"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATAFRAME
# =====================================================

print("\nLoading dataframe...")

df = pd.read_csv(CSV_PATH)

print(f"Total rows: {len(df)}")

# =====================================================
# REMOVE MISSING PATHS
# =====================================================

df = df[df["image_path"].notna()]

print(f"Valid rows: {len(df)}")

# =====================================================
# RESULTS
# =====================================================

results = []

# =====================================================
# TOTAL TIMER
# =====================================================

total_start = time.time()

# =====================================================
# LOOP
# =====================================================

for idx, row in df.iterrows():

    try:

        # -------------------------------------------------
        # CASE ID
        # -------------------------------------------------

        public_id = str(row["public_id"])

        split = row["split"]

        filename = f"{public_id}.nii.gz"

        # -------------------------------------------------
        # LOCAL IMAGE PATH
        # -------------------------------------------------

        image_path = os.path.join(
            BASE_DIR,
            f"data/mela/images/{split}",
            filename
        )

        print("\n================================")
        print(f"Processing: {public_id}")
        print(image_path)

        # -------------------------------------------------
        # FILE CHECK
        # -------------------------------------------------

        if not os.path.exists(image_path):

            print("File not found.")

            continue

        # =================================================
        # CASE TIMER
        # =================================================

        case_start = time.time()

        # =================================================
        # VOLUME LOADING
        # =================================================

        load_start = time.time()

        volume = nib.load(image_path).get_fdata(dtype=np.float32)

        load_end = time.time()

        volume_load_time = (
            load_end - load_start
        )

        # =================================================
        # ROI EXTRACTION
        # =================================================

        roi_start = time.time()

        z_center = int(row["center_z"])

        z_min = max(0, z_center - 5)
        z_max = min(volume.shape[2], z_center + 5) 


        roi = volume[:, :, z_min:z_max]

        roi_end = time.time()

        roi_time = roi_end - roi_start

        # =================================================
        # SIMULATED MODEL INFERENCE
        # =================================================

        inference_start = time.time()

        dummy_prediction = np.mean(roi)

        inference_end = time.time()

        inference_time = (
            inference_end - inference_start
        )

        # =================================================
        # TOTAL CASE TIME
        # =================================================

        case_end = time.time()

        runtime = case_end - case_start

        # =================================================
        # SAVE RESULT
        # =================================================

        results.append({

            "public_id": public_id,

            "split": split,

            "runtime_sec": runtime,

            "volume_load_sec": volume_load_time,

            "roi_sec": roi_time,

            "inference_sec": inference_time,

            "volume_shape": str(volume.shape),

            "roi_shape": str(roi.shape)
        })

        print(f"Runtime: {runtime:.2f} sec")

    except Exception as e:

        print(f"ERROR in {public_id}: {e}")

# =====================================================
# TOTAL RUNTIME
# =====================================================

total_end = time.time()

# =====================================================
# SAVE CSV
# =====================================================

timing_df = pd.DataFrame(results)

output_csv = os.path.join(
    OUTPUT_DIR,
    "inference_timing.csv"
)

timing_df.to_csv(output_csv, index=False)

# =====================================================
# SUMMARY
# =====================================================

print("\n================================")
print("FINAL SUMMARY")
print("================================")

print(f"Saved CSV: {output_csv}")

print(f"Processed cases: {len(timing_df)}")

print(
    f"Total runtime: "
    f"{total_end-total_start:.2f} sec"
)

if len(timing_df) > 0:

    print(
        f"Average runtime: "
        f"{timing_df['runtime_sec'].mean():.2f} sec"
    )

    print(
        f"Average volume loading: "
        f"{timing_df['volume_load_sec'].mean():.2f} sec"
    )

    print(
        f"Average ROI extraction: "
        f"{timing_df['roi_sec'].mean():.4f} sec"
    )

    print(
        f"Average inference: "
        f"{timing_df['inference_sec'].mean():.4f} sec"
    )
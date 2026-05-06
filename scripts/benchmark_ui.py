import os
import time
import numpy as np
import pandas as pd
import nibabel as nib

# =====================================================
# BASE DIR
# =====================================================

BASE_DIR = r"D:/GÜNCEL STAJ PROJESİ/MediBoxLesion"

# =====================================================
# CSV
# =====================================================

batch_csv = os.path.join(
    BASE_DIR,
    "results/mela/mela_batch_inference_summary.csv"
)

df = pd.read_csv(batch_csv)

# =====================================================
# SAMPLE CASES
# =====================================================

sample_cases = df.head(5)

# =====================================================
# RESULTS
# =====================================================

results = []

# =====================================================
# LOOP
# =====================================================

for idx, row in sample_cases.iterrows():

    try:

        public_id = row["public_id"]

        split = row["split"]

        # =================================================
        # IMAGE PATH
        # =================================================

        image_path = os.path.join(

            BASE_DIR,

            f"data/mela/images/{split}",

            f"{public_id}.nii.gz"
        )

        # =================================================
        # PRED PATH FIX
        # =================================================

        pred_filename = os.path.basename(
            row["pred_path"]
        )

        pred_path = os.path.join(

            BASE_DIR,

            "results/mela/mela_batch_predictions",

            pred_filename
        )

        print("\n========================")
        print(f"Case: {public_id}")

        print("Image:")
        print(image_path)

        print("Prediction:")
        print(pred_path)

        # =================================================
        # FILE CHECKS
        # =================================================

        if not os.path.exists(image_path):

            print("Image not found.")

            continue

        if not os.path.exists(pred_path):

            print("Prediction not found.")

            continue

        # =================================================
        # FIRST LOAD
        # =================================================

        start = time.time()

        volume = nib.load(image_path).get_fdata(dtype=np.float32)

        pred = np.load(pred_path)

        end = time.time()

        first_load = end - start

        # =================================================
        # SECOND LOAD
        # =================================================

        start2 = time.time()

        volume2 = nib.load(
            image_path
        ).get_fdata()

        pred2 = np.load(pred_path)

        end2 = time.time()

        second_load = end2 - start2

        # =================================================
        # CACHE GAIN
        # =================================================

        cache_gain = (
            first_load - second_load
        )

        # =================================================
        # SAVE RESULT
        # =================================================

        results.append({

            "public_id": public_id,

            "first_load_sec": first_load,

            "second_load_sec": second_load,

            "cache_gain_sec": cache_gain
        })

        # =================================================
        # PRINT
        # =================================================

        print(
            f"First Load: "
            f"{first_load:.2f} sec"
        )

        print(
            f"Second Load: "
            f"{second_load:.2f} sec"
        )

        print(
            f"Cache Gain: "
            f"{cache_gain:.2f} sec"
        )

    except Exception as e:

        print(f"ERROR: {e}")

# =====================================================
# SAVE CSV
# =====================================================

results_df = pd.DataFrame(results)

output_csv = os.path.join(

    BASE_DIR,

    "results/performance/ui_performance.csv"
)

results_df.to_csv(
    output_csv,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n========================")
print("FINAL SUMMARY")
print("========================")

print(results_df)

print(f"\nSaved: {output_csv}")
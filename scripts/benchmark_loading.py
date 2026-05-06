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
# OUTPUT DIR
# =====================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results/performance"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# RESULTS
# =====================================================

results = []

# =====================================================
# SAMPLE FILES
# =====================================================

sample_nifti = os.path.join(
    BASE_DIR,
    "data/mela/images/train/mela_0001.nii.gz"
)

sample_csv = os.path.join(
    BASE_DIR,
    "data/mela/annotations/mela_master_dataframe.csv"
)

# =====================================================
# AUTOMATIC PREDICTION FILE SELECTION
# =====================================================

prediction_dir = os.path.join(
    BASE_DIR,
    "results/mela/mela_batch_predictions"
)

prediction_files = [

    f for f in os.listdir(prediction_dir)

    if f.endswith(".npy")
]

if len(prediction_files) == 0:

    raise FileNotFoundError(
        "No .npy prediction files found."
    )

sample_prediction = os.path.join(
    prediction_dir,
    prediction_files[0]
)

print("\nSelected prediction file:")
print(sample_prediction)

# =====================================================
# CSV LOADING TEST
# =====================================================

print("\n================================")
print("CSV LOADING TEST")
print("================================")

csv_start = time.time()

df = pd.read_csv(sample_csv)

csv_end = time.time()

csv_time = csv_end - csv_start

csv_size = os.path.getsize(sample_csv) / (1024**2)

results.append({

    "operation": "csv_load",

    "file": os.path.basename(sample_csv),

    "file_size_mb": csv_size,

    "load_time_sec": csv_time
})

print(f"CSV Load Time: {csv_time:.4f} sec")

# =====================================================
# NIFTI LOADING TEST
# =====================================================

print("\n================================")
print("NIFTI LOADING TEST")
print("================================")

nifti_start = time.time()

volume = nib.load(
    sample_nifti
).get_fdata(dtype=np.float32)

nifti_end = time.time()

nifti_time = nifti_end - nifti_start

nifti_size = os.path.getsize(sample_nifti) / (1024**2)

results.append({

    "operation": "nifti_load",

    "file": os.path.basename(sample_nifti),

    "file_size_mb": nifti_size,

    "load_time_sec": nifti_time
})

print(f"NIfTI Load Time: {nifti_time:.4f} sec")

print(f"Volume Shape: {volume.shape}")

# =====================================================
# NPY LOADING TEST
# =====================================================

print("\n================================")
print("NPY LOADING TEST")
print("================================")

npy_start = time.time()

prediction = np.load(sample_prediction)

npy_end = time.time()

npy_time = npy_end - npy_start

npy_size = os.path.getsize(sample_prediction) / (1024**2)

results.append({

    "operation": "npy_load",

    "file": os.path.basename(sample_prediction),

    "file_size_mb": npy_size,

    "load_time_sec": npy_time
})

print(f"NPY Load Time: {npy_time:.4f} sec")

print(f"Prediction Shape: {prediction.shape}")

# =====================================================
# REPEATED NIFTI LOADING TEST
# =====================================================

print("\n================================")
print("REPEATED NIFTI LOADING TEST")
print("================================")

repeat_times = []

for i in range(5):

    start = time.time()

    vol = nib.load(sample_nifti).get_fdata()

    end = time.time()

    runtime = end - start

    repeat_times.append(runtime)

    print(f"Run {i+1}: {runtime:.4f} sec")

avg_repeat = np.mean(repeat_times)

results.append({

    "operation": "repeated_nifti_load",

    "file": os.path.basename(sample_nifti),

    "file_size_mb": nifti_size,

    "load_time_sec": avg_repeat
})

# =====================================================
# SAVE CSV
# =====================================================

results_df = pd.DataFrame(results)

output_csv = os.path.join(
    OUTPUT_DIR,
    "loading_times.csv"
)

results_df.to_csv(output_csv, index=False)

# =====================================================
# SUMMARY
# =====================================================

print("\n================================")
print("FINAL SUMMARY")
print("================================")

print(results_df)

print(f"\nSaved CSV: {output_csv}")
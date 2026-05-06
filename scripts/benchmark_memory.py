import os
import gc
import time
import psutil
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

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
# MEMORY HELPER
# =====================================================

process = psutil.Process(os.getpid())

def get_memory_mb():

    return (
        process.memory_info().rss
        / 1024**2
    )

# =====================================================
# SAMPLE FILES
# =====================================================

sample_nifti = os.path.join(

    BASE_DIR,

    "data/mela/images/train/mela_0001.nii.gz"
)

sample_prediction = os.path.join(

    BASE_DIR,

    "results/mela/mela_batch_predictions/mela_0001_pred.npy"
)

# =====================================================
# RESULTS
# =====================================================

results = []

# =====================================================
# INITIAL MEMORY
# =====================================================

initial_memory = get_memory_mb()

print("\n================================")
print("INITIAL MEMORY")
print("================================")

print(f"Initial RAM: {initial_memory:.2f} MB")

results.append({

    "operation": "initial_memory",

    "memory_mb": initial_memory
})

# =====================================================
# NIFTI LOADING MEMORY
# =====================================================

print("\n================================")
print("NIFTI MEMORY TEST")
print("================================")

before_nifti = get_memory_mb()

volume = nib.load(
    sample_nifti
).get_fdata(dtype=np.float32)

after_nifti = get_memory_mb()

nifti_memory = (
    after_nifti - before_nifti
)

print(f"Before: {before_nifti:.2f} MB")

print(f"After: {after_nifti:.2f} MB")

print(f"Used: {nifti_memory:.2f} MB")

results.append({

    "operation": "nifti_loading",

    "memory_mb": nifti_memory
})

# =====================================================
# PREDICTION MEMORY
# =====================================================

print("\n================================")
print("PREDICTION MEMORY TEST")
print("================================")

before_pred = get_memory_mb()

prediction = np.load(
    sample_prediction
)

after_pred = get_memory_mb()

pred_memory = (
    after_pred - before_pred
)

print(f"Before: {before_pred:.2f} MB")

print(f"After: {after_pred:.2f} MB")

print(f"Used: {pred_memory:.2f} MB")

results.append({

    "operation": "prediction_loading",

    "memory_mb": pred_memory
})

# =====================================================
# MATPLOTLIB MEMORY
# =====================================================

print("\n================================")
print("MATPLOTLIB MEMORY TEST")
print("================================")

before_plot = get_memory_mb()

fig, ax = plt.subplots(
    1,
    1,
    figsize=(6,6)
)

ax.imshow(
    volume[:, :, 100],
    cmap="gray"
)

ax.imshow(
    prediction[100],
    alpha=0.4
)

after_plot = get_memory_mb()

plot_memory = (
    after_plot - before_plot
)

print(f"Before: {before_plot:.2f} MB")

print(f"After: {after_plot:.2f} MB")

print(f"Used: {plot_memory:.2f} MB")

results.append({

    "operation": "matplotlib_render",

    "memory_mb": plot_memory
})

# =====================================================
# MEMORY CLEANUP
# =====================================================

print("\n================================")
print("MEMORY CLEANUP TEST")
print("================================")

before_cleanup = get_memory_mb()

plt.close(fig)

del volume
del prediction

gc.collect()

after_cleanup = get_memory_mb()

cleanup_gain = (
    before_cleanup - after_cleanup
)

print(f"Before: {before_cleanup:.2f} MB")

print(f"After: {after_cleanup:.2f} MB")

print(f"Freed: {cleanup_gain:.2f} MB")

results.append({

    "operation": "memory_cleanup",

    "memory_mb": cleanup_gain
})

# =====================================================
# SAVE CSV
# =====================================================

results_df = pd.DataFrame(results)

output_csv = os.path.join(

    OUTPUT_DIR,

    "memory_usage.csv"
)

results_df.to_csv(
    output_csv,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n================================")
print("FINAL SUMMARY")
print("================================")

print(results_df)

print(f"\nSaved: {output_csv}")
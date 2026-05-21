# ============================================================
# MELA Dashboard V2
# Clean Medical Imaging Viewer
# ============================================================

from pathlib import Path
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st

from utils.mask_refinement import refine_prediction

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MELA Candidate Mask Dashboard",
    layout="wide",
)

# ============================================================
# THEME
# ============================================================

st.markdown(
    """
<style>
    .stApp {
        background: #0b0f16;
        color: #e8edf5;
    }

    section[data-testid="stSidebar"] {
        background: #242732;
        border-right: 1px solid #343946;
    }

    section[data-testid="stSidebar"] * {
        color: #f4f6fb;
    }

    div[data-testid="stMetric"] {
        background: #151b25;
        border: 1px solid #293242;
        border-radius: 8px;
        padding: 14px 16px;
    }

    div[data-testid="stMetric"] label {
        color: #aab6c7;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    .hero-title {
        color: #f7f9ff;
        font-size: 2.1rem;
        font-weight: 750;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #9fb0c8;
        font-size: 1.02rem;
        margin-bottom: 1.1rem;
    }

    .legend-row {
        color: #c8d2e1;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin: 0.6rem 0 1.1rem 0;
    }

    .legend-item {
        align-items: center;
        display: inline-flex;
        font-size: 0.9rem;
        gap: 7px;
    }

    .swatch {
        border-radius: 3px;
        display: inline-block;
        height: 14px;
        width: 14px;
    }

    .swatch-cyan { background: #28d7ff; }
    .swatch-yellow { background: #f4e525; }
    .swatch-red { background: #7e1f27; }
    .swatch-orange { background: #f97316; }
</style>

<div class="hero-title">MELA Candidate Mask Dashboard</div>
<div class="hero-subtitle">
Bounding-box destekli aday lezyon analizi ve model maske inceleme arayüzü
</div>
"""
    ,
    unsafe_allow_html=True,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"

MELA_DIR = DATA_DIR / "mela"

ANNOT_DIR = MELA_DIR / "annotations"
IMAGE_DIR = MELA_DIR / "images"
MASK_DIR = MELA_DIR / "masks"

MELA_RESULTS_DIR = RESULTS_DIR / "mela"
PRED_DIR = MELA_RESULTS_DIR / "mela_batch_predictions"
DEFAULT_MODEL_PATH = MODEL_DIR / "best_model.pt"

MASTER_CSV = ANNOT_DIR / "mela_master_dataframe.csv"
BATCH_SUMMARY = MELA_RESULTS_DIR / "mela_batch_inference_summary.csv"
FEEDBACK_PATH = RESULTS_DIR / "user_feedback.csv"

# ============================================================
# CACHE FUNCTIONS
# ============================================================


@st.cache_data
def load_tables():
    master_df = pd.read_csv(MASTER_CSV)
    batch_df = pd.read_csv(BATCH_SUMMARY)

    return master_df, batch_df


@st.cache_data
def load_nifti(path):
    nii = nib.load(str(path))
    vol = nii.get_fdata(dtype=np.float32)

    # Convert from (X, Y, Z) to (Z, Y, X) for slice viewing.
    return np.transpose(vol, (2, 1, 0))


@st.cache_data
def load_prediction(path):
    return np.load(path)


@st.cache_data
def load_nifti_raw(path):
    nii = nib.load(str(path))

    return nii.get_fdata(dtype=np.float32)


# ============================================================
# HELPERS
# ============================================================


def resolve_project_path(path_value):
    if pd.isna(path_value):
        return None

    path_text = str(path_value).strip()

    if not path_text:
        return None

    direct_path = Path(path_text)

    if direct_path.exists():
        return direct_path

    normalized = path_text.replace("\\", "/")

    for marker, base_dir in [
        ("/data/", DATA_DIR),
        ("data/", DATA_DIR),
        ("/results/", RESULTS_DIR),
        ("results/", RESULTS_DIR),
    ]:
        if marker in normalized:
            relative_tail = normalized.split(marker, 1)[1]
            candidate = base_dir / Path(relative_tail)

            if candidate.exists():
                return candidate

    return None


def get_row_value(row, column):
    if row is None:
        return None

    if hasattr(row, "index") and column in row.index:
        return row[column]

    if hasattr(row, column):
        return getattr(row, column)

    return None


def find_image_path(public_id, row_batch=None, row_ann=None):
    for row in [row_batch, row_ann]:
        if row is None:
            continue

        for column in ["img_path", "image_path"]:
            path_value = get_row_value(row, column)

            if path_value is not None:
                resolved = resolve_project_path(path_value)

                if resolved is not None:
                    split = resolved.parent.name
                    return resolved, split

    for split in ["train", "val"]:
        candidate = IMAGE_DIR / split / f"{public_id}.nii.gz"

        if candidate.exists():
            return candidate, split

    return None, None


def find_pseudo_mask_path(public_id, split, row_ann=None):
    mask_path = get_row_value(row_ann, "mask_path")

    if mask_path is not None:
        resolved = resolve_project_path(mask_path)

        if resolved is not None:
            return resolved

    splits_to_try = []

    if split:
        splits_to_try.append(split)

    for fallback_split in ["train", "val"]:
        if fallback_split not in splits_to_try:
            splits_to_try.append(fallback_split)

    candidates = []

    for candidate_split in splits_to_try:
        candidates.extend(
            [
                MASK_DIR / candidate_split / f"{public_id}_mask.nii.gz",
                MASK_DIR / candidate_split / f"{public_id}.nii.gz",
            ]
        )

    for path in candidates:
        if path.exists():
            return path

    return None


def find_prediction_path(public_id, row_batch):
    local_path = PRED_DIR / f"{public_id}_pred.npy"

    if local_path.exists():
        return local_path

    resolved = resolve_project_path(get_row_value(row_batch, "pred_path"))

    if resolved is not None:
        return resolved

    csv_path = Path(str(row_batch.pred_path))
    fallback = PRED_DIR / csv_path.name

    if fallback.exists():
        return fallback

    return None


def get_nonzero_slices(mask):
    sums = mask.reshape(mask.shape[0], -1).sum(axis=1)

    return np.where(sums > 0)[0]


def choose_best_slice(mask, fallback):
    sums = mask.reshape(mask.shape[0], -1).sum(axis=1)

    if sums.max() == 0:
        return clamp_index(fallback, mask.shape[0])

    return int(np.argmax(sums))


def clamp_index(value, size):
    return max(0, min(int(round(float(value))), size - 1))


def normalize_ct(img, hu_min=-1000, hu_max=400):
    clipped = np.clip(img, hu_min, hu_max)

    return (clipped - hu_min) / (hu_max - hu_min + 1e-8)


def get_annotation_center_z(row, shape):
    center_z = get_row_value(row, "center_z")

    if center_z is not None:
        return clamp_index(center_z, shape[0])

    z1 = get_row_value(row, "z1")
    z2 = get_row_value(row, "z2")

    if z1 is not None and z2 is not None:
        return clamp_index((float(z1) + float(z2)) / 2, shape[0])

    return clamp_index(get_row_value(row, "coordZ"), shape[0])


def has_value(value):
    return value is not None and not pd.isna(value)


def build_bbox(row, shape, margin=20):
    _, y_max, x_max = shape

    csv_x1 = get_row_value(row, "x1")
    csv_x2 = get_row_value(row, "x2")
    csv_y1 = get_row_value(row, "y1")
    csv_y2 = get_row_value(row, "y2")

    if all(has_value(value) for value in [csv_x1, csv_x2, csv_y1, csv_y2]):
        x1 = int(float(csv_x1))
        x2 = int(float(csv_x2))
        y1 = int(float(csv_y1))
        y2 = int(float(csv_y2))
    else:
        center_x = get_row_value(row, "center_x")
        center_y = get_row_value(row, "center_y")

        if center_x is None:
            center_x = get_row_value(row, "coordX")

        if center_y is None:
            center_y = get_row_value(row, "coordY")

        cx = int(float(center_x))
        cy = int(float(center_y))

        xl = int(float(get_row_value(row, "x_length")))
        yl = int(float(get_row_value(row, "y_length")))

        x1 = cx - xl // 2
        x2 = cx + xl // 2
        y1 = cy - yl // 2
        y2 = cy + yl // 2

    x1 = max(0, min(x1, x_max))
    x2 = max(0, min(x2, x_max))
    y1 = max(0, min(y1, y_max))
    y2 = max(0, min(y2, y_max))

    rx1 = max(0, x1 - margin)
    rx2 = min(x_max, x2 + margin)

    ry1 = max(0, y1 - margin)
    ry2 = min(y_max, y2 + margin)

    return (x1, x2, y1, y2), (rx1, rx2, ry1, ry2)


def build_bbox_3d(row, shape):
    z_max, y_max, x_max = shape
    bbox, _ = build_bbox(row, shape, margin=0)
    x1, x2, y1, y2 = bbox

    z1 = get_row_value(row, "z1")
    z2 = get_row_value(row, "z2")

    if not has_value(z1) or not has_value(z2):
        center_z = get_annotation_center_z(row, shape)
        z_length = get_row_value(row, "z_length")
        half_length = int(float(z_length)) // 2 if has_value(z_length) else 5
        z1 = center_z - half_length
        z2 = center_z + half_length

    z1 = max(0, min(int(float(z1)), z_max))
    z2 = max(0, min(int(float(z2)), z_max))

    return z1, z2, y1, y2, x1, x2


def bbox_overlap_voxels(mask, bbox_3d):
    z1, z2, y1, y2, x1, x2 = bbox_3d

    if z2 <= z1 or y2 <= y1 or x2 <= x1:
        return 0

    return int((mask[z1:z2, y1:y2, x1:x2] > 0).sum())


def load_mask_aligned(path, target_shape, bbox_3d):
    raw_mask = load_nifti_raw(path)
    candidates = []

    for axes in [
        (2, 1, 0),
        (2, 0, 1),
        (1, 2, 0),
        (1, 0, 2),
        (0, 2, 1),
        (0, 1, 2),
    ]:
        if tuple(raw_mask.shape[axis] for axis in axes) != tuple(target_shape):
            continue

        oriented = np.transpose(raw_mask, axes)
        candidates.append((f"transpose{axes}", oriented))

    if not candidates:
        raise ValueError(
            "Mask shape cannot be aligned with image shape: "
            f"raw_mask={raw_mask.shape}, image={target_shape}"
        )

    best_name = None
    best_mask = None
    best_score = -1

    for name, candidate in candidates:
        for flip_z in [False, True]:
            for flip_y in [False, True]:
                for flip_x in [False, True]:
                    aligned = candidate
                    flip_suffix = []

                    if flip_z:
                        aligned = np.flip(aligned, axis=0)
                        flip_suffix.append("flip_z")

                    if flip_y:
                        aligned = np.flip(aligned, axis=1)
                        flip_suffix.append("flip_y")

                    if flip_x:
                        aligned = np.flip(aligned, axis=2)
                        flip_suffix.append("flip_x")

                    score = bbox_overlap_voxels(aligned, bbox_3d)

                    if score > best_score:
                        best_name = (
                            name
                            if not flip_suffix
                            else f"{name}+{'+'.join(flip_suffix)}"
                        )
                        best_mask = aligned.copy()
                        best_score = score

    return (best_mask > 0).astype(np.uint8), best_name, best_score


def build_roi_mask(shape, roi):
    roi_mask = np.zeros(shape, dtype=np.uint8)
    x1, x2, y1, y2 = roi

    roi_mask[:, y1:y2, x1:x2] = 1

    return roi_mask


# ============================================================
# PLOTTING
# ============================================================


def draw_panel(
    ax,
    ct,
    mask,
    bbox,
    roi,
    title,
    alpha,
    show_bbox,
    show_roi,
    mask_color="autumn",
    dim_background=False,
):
    if dim_background:
        ct = np.clip(ct * 0.55 + 0.28, 0, 1)

    ax.imshow(ct, cmap="gray")

    if mask is not None and mask.sum() > 0:
        ax.imshow(
            np.ma.masked_where(mask == 0, mask),
            cmap=mask_color,
            alpha=alpha,
        )

    x1, x2, y1, y2 = bbox
    rx1, rx2, ry1, ry2 = roi

    if show_bbox:
        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="#28d7ff",
                linewidth=2.4,
            )
        )

    if show_roi:
        ax.add_patch(
            plt.Rectangle(
                (rx1, ry1),
                rx2 - rx1,
                ry2 - ry1,
                fill=False,
                edgecolor="#f4e525",
                linewidth=2.4,
            )
        )

    ax.set_title(
        title,
        color="#1f2937",
        fontsize=11,
        fontweight="bold",
        pad=6,
    )
    ax.axis("off")


def create_comparison_figure(
    ct,
    pseudo_mask,
    final_pred_mask,
    bbox,
    roi,
    alpha,
    show_bbox,
    show_roi,
    case_id,
    slice_index,
    prediction_title,
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.1))
    fig.patch.set_facecolor("white")

    draw_panel(
        axes[0],
        ct,
        None,
        bbox,
        roi,
        f"{case_id} | slice {slice_index} | CT Slice",
        alpha,
        False,
        False,
    )

    draw_panel(
        axes[1],
        ct,
        None,
        bbox,
        roi,
        "Annotation Box",
        alpha,
        show_bbox,
        show_roi,
    )

    draw_panel(
        axes[2],
        ct,
        pseudo_mask,
        bbox,
        roi,
        "Pseudo-mask Overlay",
        alpha,
        show_bbox,
        show_roi,
        mask_color="Reds",
        dim_background=True,
    )

    draw_panel(
        axes[3],
        ct,
        final_pred_mask,
        bbox,
        roi,
        prediction_title,
        alpha,
        show_bbox,
        show_roi,
        mask_color="autumn",
        dim_background=True,
    )

    plt.tight_layout(pad=1.1, w_pad=2.2)

    return fig


def create_gallery_figure(
    ct,
    mask,
    bbox,
    roi,
    title,
    alpha,
    show_bbox,
    show_roi,
):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    fig.patch.set_facecolor("white")

    draw_panel(
        ax,
        ct,
        mask,
        bbox,
        roi,
        title,
        alpha,
        show_bbox,
        show_roi,
        mask_color="autumn",
        dim_background=True,
    )

    plt.tight_layout(pad=0.5)

    return fig


# ============================================================
# LOAD TABLES
# ============================================================

try:
    master_df, batch_df = load_tables()
except FileNotFoundError as exc:
    st.error(f"Required table not found: {exc}")
    st.stop()

valid_df = batch_df[batch_df.status == "ok"].copy()

if valid_df.empty:
    st.error("No valid inference cases found in the batch summary.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Controls")

st.sidebar.header("Filters")

case_filter = st.sidebar.selectbox(
    "Case filter",
    [
        "all",
        "empty",
        "small",
        "medium",
        "large",
    ],
)

if case_filter == "all":
    filtered_df = valid_df
else:
    filtered_df = valid_df[valid_df.prediction_type == case_filter]

if filtered_df.empty:
    st.warning("No cases found for the selected prediction type.")
    st.stop()

public_ids = filtered_df.public_id.tolist()

if "selected_case_id" not in st.session_state:
    st.session_state.selected_case_id = public_ids[0]

if st.session_state.selected_case_id not in public_ids:
    st.session_state.selected_case_id = public_ids[0]

current_index = public_ids.index(st.session_state.selected_case_id)

selected_id = st.sidebar.selectbox(
    "Select public_id",
    public_ids,
    index=current_index,
    key="selected_case_id",
)

nav_prev, nav_next = st.sidebar.columns(2)

if nav_prev.button("◀ Previous", use_container_width=True):
    current_index = max(0, public_ids.index(selected_id) - 1)
    st.session_state.selected_case_id = public_ids[current_index]
    st.rerun()

if nav_next.button("Next ▶", use_container_width=True):
    current_index = min(len(public_ids) - 1, public_ids.index(selected_id) + 1)
    st.session_state.selected_case_id = public_ids[current_index]
    st.rerun()

st.sidebar.header("Visualization")

overlay_alpha = st.sidebar.slider(
    "Mask opacity",
    0.1,
    1.0,
    0.65,
)

roi_margin = st.sidebar.slider(
    "ROI margin",
    0,
    50,
    20,
)

show_bbox = st.sidebar.checkbox(
    "Show true bbox",
    value=True,
)

show_roi = st.sidebar.checkbox(
    "Show ROI box",
    value=True,
)

st.sidebar.header("Refinement")

apply_refinement = st.sidebar.checkbox(
    "Apply refinement",
    value=False,
)

threshold = st.sidebar.slider(
    "Prediction threshold",
    0.1,
    0.95,
    0.70,
)

# ============================================================
# LOAD CASE
# ============================================================

load_start = time.time()

row_batch = valid_df[valid_df.public_id == selected_id].iloc[0]
matching_annotations = master_df[master_df.public_id == selected_id]

if matching_annotations.empty:
    st.error(
        "Annotation row not found for the selected case. "
        f"Case ID: {selected_id}"
    )
    st.stop()

row_ann = matching_annotations.iloc[0]

img_path, split = find_image_path(
    selected_id,
    row_batch=row_batch,
    row_ann=row_ann,
)

if img_path is None:
    st.error("Image file not found.")
    st.stop()

pred_path = find_prediction_path(selected_id, row_batch)

if pred_path is None:
    st.error("Prediction file not found.")
    st.stop()

pseudo_path = find_pseudo_mask_path(
    selected_id,
    split,
    row_ann=row_ann,
)

with st.spinner("Loading data..."):
    try:
        volume = load_nifti(img_path)
        raw_pred = load_prediction(pred_path)
        pred = raw_pred.copy()

    except Exception as exc:
        st.error(f"Case could not be loaded: {exc}")
        st.stop()

if pred.shape != volume.shape:
    st.error(
        "Prediction shape does not match image shape: "
        f"image={volume.shape}, prediction={pred.shape}"
    )
    st.stop()

loading_time = time.time() - load_start

# ============================================================
# BBOX AND REFINEMENT
# ============================================================

bbox, roi = build_bbox(
    row_ann,
    volume.shape,
    roi_margin,
)

bbox_3d = build_bbox_3d(row_ann, volume.shape)
pseudo_mask = None
pseudo_alignment = None
pseudo_bbox_voxels = 0

if pseudo_path is not None:
    try:
        pseudo_mask, pseudo_alignment, pseudo_bbox_voxels = load_mask_aligned(
            pseudo_path,
            volume.shape,
            bbox_3d,
        )
    except Exception as exc:
        st.warning(f"Pseudo-mask could not be aligned: {exc}")
        pseudo_mask = None

if apply_refinement:
    pred = refine_prediction(
        pred_probs=pred,
        roi_mask=build_roi_mask(pred.shape, roi),
        threshold=threshold,
    )

# ============================================================
# BEST SLICE
# ============================================================

nonzero = get_nonzero_slices(pred)
pseudo_nonzero = (
    get_nonzero_slices(pseudo_mask)
    if pseudo_mask is not None
    else np.array([])
)

annotation_slice = get_annotation_center_z(row_ann, volume.shape)

if len(nonzero) > 0 and len(pseudo_nonzero) > 0:
    combined_mask = (pred > 0).astype(np.uint8) + (pseudo_mask > 0).astype(np.uint8)
    best_slice = choose_best_slice(
        combined_mask,
        annotation_slice,
    )
elif len(nonzero) > 0:
    best_slice = choose_best_slice(
        pred,
        annotation_slice,
    )
elif len(pseudo_nonzero) > 0:
    best_slice = choose_best_slice(
        pseudo_mask,
        annotation_slice,
    )
else:
    best_slice = annotation_slice

if len(nonzero) == 0:
    st.warning("No candidate lesion detected for the current settings.")

slice_mode = st.sidebar.radio(
    "Slice mode",
    [
        "Best Slice",
        "Manual",
    ],
)

if slice_mode == "Best Slice":
    z = best_slice
else:
    z = st.sidebar.slider(
        "Slice",
        0,
        volume.shape[0] - 1,
        best_slice,
    )

if pseudo_mask is None:
    st.warning(
        "Pseudo-mask file could not be found for this case. "
        "The app searched data/mela/masks/train and data/mela/masks/val."
    )
elif z not in set(pseudo_nonzero.tolist()):
    st.info(
        "Pseudo-mask exists, but it is empty on the selected slice. "
        "Use Best Slice or inspect the pseudo-mask gallery below."
    )

# ============================================================
# DASHBOARD METRICS
# ============================================================

st.subheader(f"Case: {selected_id}")

metric_cols = st.columns(4)

metric_cols[0].metric(
    "Prediction Volume",
    int(pred.sum()),
)

metric_cols[1].metric(
    "Prediction Type",
    row_batch.prediction_type,
)

metric_cols[2].metric(
    "Prediction Slices",
    len(nonzero),
)

metric_cols[3].metric(
    "Dataset Split",
    split,
)

mask_status_cols = st.columns(3)

mask_status_cols[0].metric(
    "Pseudo-mask",
    "Found" if pseudo_mask is not None else "Missing",
)

mask_status_cols[1].metric(
    "Pseudo-mask Slices",
    len(pseudo_nonzero),
)

mask_status_cols[2].metric(
    "Selected Slice",
    z,
)

with st.expander("Loaded file paths and sources", expanded=True):
    source_rows = [
        {
            "source": "CT image",
            "path": str(img_path),
            "exists": img_path.exists(),
        },
        {
            "source": "Pseudo-mask",
            "path": str(pseudo_path) if pseudo_path is not None else "Not found",
            "exists": pseudo_path.exists() if pseudo_path is not None else False,
        },
        {
            "source": "Batch model prediction",
            "path": str(pred_path),
            "exists": pred_path.exists(),
        },
        {
            "source": "Model checkpoint reference",
            "path": str(DEFAULT_MODEL_PATH),
            "exists": DEFAULT_MODEL_PATH.exists(),
        },
    ]

    st.dataframe(
        pd.DataFrame(source_rows),
        use_container_width=True,
        hide_index=True,
    )

    if pseudo_mask is not None:
        st.caption(
            "Pseudo-mask alignment: "
            f"{pseudo_alignment} | bbox-overlap voxels: {pseudo_bbox_voxels}"
        )

    st.caption(
        "Model output is loaded from the batch prediction .npy file. "
        "The checkpoint path is shown as the model artifact reference."
    )

# ============================================================
# COMPARISON VIEW
# ============================================================

st.subheader("Main Comparison")

st.markdown(
    """
<div class="legend-row">
    <span class="legend-item"><span class="swatch swatch-cyan"></span>True bbox</span>
    <span class="legend-item"><span class="swatch swatch-yellow"></span>ROI box</span>
    <span class="legend-item"><span class="swatch swatch-red"></span>Pseudo-mask</span>
    <span class="legend-item"><span class="swatch swatch-orange"></span>Model mask</span>
</div>
""",
    unsafe_allow_html=True,
)

render_start = time.time()

fig = create_comparison_figure(
    normalize_ct(volume[z]),
    pseudo_mask[z] if pseudo_mask is not None else None,
    pred[z],
    bbox,
    roi,
    overlay_alpha,
    show_bbox,
    show_roi,
    selected_id,
    z,
    "Refined Prediction" if apply_refinement else "Model Prediction",
)

st.pyplot(fig)
plt.close(fig)

render_time = time.time() - render_start

# ============================================================
# SLICE GALLERY
# ============================================================

st.markdown("---")
st.subheader("Mask-visible slices")

gallery_mask = pred
gallery_slices = nonzero
gallery_title = "Prediction"

if len(gallery_slices) == 0 and pseudo_mask is not None:
    gallery_mask = pseudo_mask
    gallery_slices = pseudo_nonzero
    gallery_title = "Pseudo-mask"

if len(gallery_slices) > 0:
    st.caption(f"Showing {gallery_title} visible slices.")
    selected_slices = gallery_slices

    if len(selected_slices) > 3:
        idx = np.linspace(
            0,
            len(selected_slices) - 1,
            3,
        ).astype(int)

        selected_slices = selected_slices[idx]

    cols = st.columns(len(selected_slices))

    for col, slice_index in zip(cols, selected_slices):
        fig2 = create_gallery_figure(
            normalize_ct(volume[slice_index]),
            gallery_mask[slice_index],
            bbox,
            roi,
            f"{selected_id} | slice {slice_index}",
            overlay_alpha,
            show_bbox,
            show_roi,
        )

        col.pyplot(fig2)
        plt.close(fig2)

else:
    st.info("No mask-visible slices found.")

if pseudo_mask is not None:
    st.markdown("---")
    st.subheader("Pseudo-mask visible slices")

    if len(pseudo_nonzero) > 0:
        st.caption(f"Showing slices from {pseudo_path}")
        selected_pseudo_slices = pseudo_nonzero

        if len(selected_pseudo_slices) > 3:
            idx = np.linspace(
                0,
                len(selected_pseudo_slices) - 1,
                3,
            ).astype(int)

            selected_pseudo_slices = selected_pseudo_slices[idx]

        pseudo_cols = st.columns(len(selected_pseudo_slices))

        for col, slice_index in zip(pseudo_cols, selected_pseudo_slices):
            fig3 = create_gallery_figure(
                normalize_ct(volume[slice_index]),
                pseudo_mask[slice_index],
                bbox,
                roi,
                f"{selected_id} | pseudo slice {slice_index}",
                overlay_alpha,
                show_bbox,
                show_roi,
            )

            col.pyplot(fig3)
            plt.close(fig3)

    else:
        st.info("Pseudo-mask file exists, but it has no nonzero slices.")

# ============================================================
# CASE SUMMARY
# ============================================================

with st.expander(
    "Case Summary",
    expanded=False,
):
    st.write(f"Public ID: {selected_id}")
    st.write(f"Prediction Type: {row_batch.prediction_type}")
    st.write(f"Prediction Volume: {int(row_batch.pred_sum)}")
    st.write(f"Prediction Slice Count: {len(nonzero)}")
    st.write(f"Pseudo-mask Slice Count: {len(pseudo_nonzero)}")

    st.write(f"Dataset Split: {split}")
    st.write(f"Selected Slice: {z}")
    st.write(f"Annotation Center Slice: {annotation_slice}")
    st.write(f"Volume Shape: {volume.shape}")
    st.write(f"Coordinates: ({row_ann.coordX}, {row_ann.coordY}, {row_ann.coordZ})")
    st.write(f"BBox Size: {row_ann.x_length} x {row_ann.y_length} x {row_ann.z_length}")
    st.write(f"Image Path: {img_path}")
    st.write(f"Prediction Path: {pred_path}")

    if pseudo_path is not None:
        st.write(f"Pseudo-mask Path: {pseudo_path}")
        st.write(f"Pseudo-mask Alignment: {pseudo_alignment}")
        st.write(f"Pseudo-mask BBox Overlap Voxels: {pseudo_bbox_voxels}")
    else:
        st.write("Pseudo-mask Path: Not found")

    st.write(f"Model Checkpoint Reference: {DEFAULT_MODEL_PATH}")

# ============================================================
# DOWNLOAD REPORT
# ============================================================

report_df = pd.DataFrame(
    [
        {
            "public_id": selected_id,
            "split": split,
            "slice": z,
            "roi_margin": roi_margin,
            "pseudo_mask_found": pseudo_mask is not None,
            "pseudo_mask_path": str(pseudo_path) if pseudo_path is not None else None,
            "pseudo_mask_slices": len(pseudo_nonzero),
            "pseudo_mask_alignment": pseudo_alignment,
            "pseudo_mask_bbox_voxels": pseudo_bbox_voxels,
            "image_path": str(img_path),
            "prediction_path": str(pred_path),
            "model_checkpoint_reference": str(DEFAULT_MODEL_PATH),
            "prediction_volume": int(row_batch.pred_sum),
            "prediction_type": row_batch.prediction_type,
            "mask_slices": len(nonzero),
            "apply_refinement": apply_refinement,
            "threshold": threshold,
            "loading_time_sec": loading_time,
            "render_time_sec": render_time,
        }
    ]
)

st.download_button(
    "Download Case Report",
    report_df.to_csv(index=False),
    file_name=f"{selected_id}_report.csv",
    mime="text/csv",
)

# ============================================================
# FEEDBACK SYSTEM
# ============================================================

st.subheader("User Feedback")

feedback = st.radio(
    "Prediction Quality",
    [
        "Correct",
        "Partially Correct",
        "Incorrect",
    ],
)

note = st.text_area("Optional Notes")

if st.button("Save Feedback"):
    feedback_row = pd.DataFrame(
        [
            {
                "public_id": selected_id,
                "feedback": feedback,
                "note": note,
                "slice": z,
                "prediction_type": row_batch.prediction_type,
                "apply_refinement": apply_refinement,
                "threshold": threshold,
                "saved_at": pd.Timestamp.now().isoformat(),
            }
        ]
    )

    if FEEDBACK_PATH.exists():
        old_feedback = pd.read_csv(FEEDBACK_PATH)
        feedback_row = pd.concat(
            [old_feedback, feedback_row],
            ignore_index=True,
        )

    feedback_row.to_csv(
        FEEDBACK_PATH,
        index=False,
    )

    st.success("Feedback saved successfully.")

# ============================================================
# FOOTER
# ============================================================

st.sidebar.success(f"Loading Time: {loading_time:.2f} sec")
st.sidebar.success(f"Render Time: {render_time:.2f} sec")

st.markdown("---")
st.info(
    "This system is a bounding-box guided candidate lesion analysis prototype "
    "and not a final clinical segmentation tool."
)

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_opening
from scipy.ndimage import binary_closing


def apply_threshold(
    pred,
    threshold=0.5
):

    return (
        pred > threshold
    ).astype(np.uint8)


def keep_largest_component(mask):

    labeled, num = ndimage.label(mask)

    if num == 0:
        return mask

    sizes = ndimage.sum(
        mask,
        labeled,
        range(1, num + 1)
    )

    largest_label = (
        np.argmax(sizes) + 1
    )

    refined = (
        labeled == largest_label
    )

    return refined.astype(np.uint8)


def morphology_cleanup(mask):

    opened = binary_opening(mask)

    closed = binary_closing(opened)

    return closed.astype(np.uint8)


def apply_roi_constraint(
    mask,
    roi_mask
):

    return (
        mask * roi_mask
    ).astype(np.uint8)


def refine_prediction(
    pred_probs,
    roi_mask,
    threshold=0.7
):

    # threshold
    mask = apply_threshold(
        pred_probs,
        threshold
    )

    # largest component
    mask = keep_largest_component(
        mask
    )

    # morphology
    mask = morphology_cleanup(
        mask
    )

    # roi constraint
    mask = apply_roi_constraint(
        mask,
        roi_mask
    )

    return mask
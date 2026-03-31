from typing import Tuple

import cv2
import numpy as np

from webcam_cv.config import AppConfig
from webcam_cv.models.sam.mask_candidate import MaskCandidate


min_area_ratio: float = 0.01
max_area_ratio: float = 0.70


def compute_mask_area_ratio(mask: np.ndarray) -> float:
    """Compute the fraction of image pixels covered by the mask."""
    return float(mask.mean())


def is_mask_area_valid(area_ratio: float) -> bool:
    """Return whether the mask area falls within the accepted ratio range."""
    return min_area_ratio <= area_ratio <= max_area_ratio


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute intersection over union of two boolean masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_containment_ratio(inner: np.ndarray, outer: np.ndarray) -> float:
    """Compute how much of the first mask is covered by the second."""
    intersection = np.logical_and(inner, outer).sum()
    inner_area = inner.sum()

    if inner_area == 0:
        return 0.0

    return float(intersection / inner_area)


def suppress_contained_masks(
    masks: list[np.ndarray],
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.90,
) -> list[np.ndarray]:
    """Remove masks that are near-duplicates or strongly contained in larger masks."""
    suppressed: set[int] = set()

    for i, mask_a in enumerate(masks):
        if i in suppressed:
            continue

        area_a = mask_a.sum()

        for j in range(i + 1, len(masks)):
            if j in suppressed:
                continue

            mask_b = masks[j]
            area_b = mask_b.sum()

            iou = compute_iou(mask_a, mask_b)
            containment_a_in_b = compute_containment_ratio(mask_a, mask_b)
            containment_b_in_a = compute_containment_ratio(mask_b, mask_a)

            # Near-duplicate masks: keep the earlier one (higher-ranked).
            if iou >= iou_threshold:
                suppressed.add(j)
                continue

            # Containment: suppress the smaller contained mask.
            if containment_a_in_b >= containment_threshold and area_a <= area_b:
                suppressed.add(i)
                break

            if containment_b_in_a >= containment_threshold and area_b <= area_a:
                suppressed.add(j)

    return [mask for idx, mask in enumerate(masks) if idx not in suppressed]


# def suppress_contained_masks(masks: list[np.ndarray], iou_threshold: float = 0.85,
#                              containment_threshold: float = 0.90) -> list[np.ndarray]:
#     """Remove masks that are near-duplicates or strongly contained in larger masks."""
#     kept: list[np.ndarray] = []
#     nb_skipped_masks = 0
#
#     for mask in masks:
#         should_keep = True
#
#         for kept_mask in kept:
#             iou = compute_iou(mask, kept_mask)
#             containment = compute_containment_ratio(mask, kept_mask)
#
#             if iou >= iou_threshold or containment >= containment_threshold:
#                 should_keep = False
#                 nb_skipped_masks += 1
#                 break
#
#         if should_keep:
#             kept.append(mask)
#
#     return kept


def compute_mask_center(mask: np.ndarray) -> Tuple[int, int]:
    """Compute a stable integer center point for drawing inside a mask.

    This uses the point farthest from the mask boundary rather than the
    arithmetic mean of mask pixels. For irregular masks, this usually gives
    a more visually relevant position for debug text.
    """
    mask_uint8 = mask.astype(np.uint8)

    if not mask_uint8.any():
        return 0, 0

    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)

    return int(x), int(y)


def compute_mask_centroid(mask: np.ndarray) -> Tuple[int, int]:
    """Compute the centroid of a binary mask using image moments."""
    mask_uint8 = mask.astype(np.uint8)

    if not mask_uint8.any():
        return 0, 0

    moments = cv2.moments(mask_uint8)
    if moments['m00'] == 0:
        return 0, 0

    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return cx, cy


def compute_mask_center_distance(mask: np.ndarray) -> float:
    """Compute the normalized distance between the mask centroid and image center."""
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return 1.0

    h, w = mask.shape
    cx = xs.mean()
    cy = ys.mean()

    image_cx = w / 2.0
    image_cy = h / 2.0

    dx = (cx - image_cx) / w
    dy = (cy - image_cy) / h

    return float(np.sqrt(dx * dx + dy * dy))


def mask_touches_border(mask: np.ndarray) -> bool:
    """Return whether the mask touches any image border."""
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def ndarray_to_mask_candidate(mask: np.ndarray) -> MaskCandidate:
    """Wrap the mask array into MaskCandidate dataclass."""
    area_ratio = compute_mask_area_ratio(mask)

    mask_center = compute_mask_center(mask)
    h, w = mask.shape[:2]
    cx, cy = mask_center

    margin = 10
    is_center_safe = margin <= cx < (w - margin) and margin <= cy < (h - margin)

    # Check if computed center is within the image borders.
    # If not, fallback to mask centroid.
    if not is_center_safe:
        mask_center = compute_mask_centroid(mask)

    candidate = MaskCandidate(
        mask=mask,
        area_ratio=area_ratio,
        mask_center=mask_center,
        center_distance=compute_mask_center_distance(mask),
        touches_border=mask_touches_border(mask),
    )
    candidate.score = score_mask_candidate(candidate)

    return candidate


def score_mask_candidate(candidate: MaskCandidate) -> float:
    """Compute a heuristic score for a valid mask candidate."""
    area_score = 1.0 - abs(candidate.area_ratio - 0.20)
    center_score = 1.0 - candidate.center_distance
    border_penalty = 0.25 if candidate.touches_border else 0.0

    return area_score + center_score - border_penalty


def rank_masks(config: AppConfig, masks: list[np.ndarray]) -> list[MaskCandidate]:
    """Filter, score, and rank SAM masks."""
    candidates: list[MaskCandidate] = []

    # filtered_masks, nb_skipped_masks = suppress_contained_masks(masks)

    for mask in masks:
        mask_area = compute_mask_area_ratio(mask)
        if not is_mask_area_valid(mask_area):
            continue

        candidate = ndarray_to_mask_candidate(mask)

        candidate.score = score_mask_candidate(candidate)
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.score, reverse=True)
    # candidates = candidates[:config.sam_top_k_masks]

    return candidates

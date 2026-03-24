from typing import Tuple, cast

import cv2
import numpy as np

from webcam_cv.models.sam.mask_candidate import MaskCandidate


min_area_ratio: float = 0.01
max_area_ratio: float = 0.70


def compute_mask_area_ratio(mask: np.ndarray) -> float:
    """Compute the fraction of image pixels covered by the mask."""
    return float(mask.mean())


def is_mask_area_valid(area_ratio: float) -> bool:
    """Return whether the mask area falls within the accepted ratio range."""
    return min_area_ratio <= area_ratio <= max_area_ratio


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

# def compute_mask_center(mask: np.ndarray) -> Tuple[float, float]:
#     """Compute the center point of the mask."""
#     ys, xs = np.where(mask)
#
#     mask_cx = int(xs.mean())
#     mask_cy = int(ys.mean())
#
#     return mask_cx, mask_cy


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


def score_mask_candidate(candidate: MaskCandidate) -> float:
    """Compute a heuristic score for a valid mask candidate."""
    area_score = 1.0 - abs(candidate.area_ratio - 0.20)
    center_score = 1.0 - candidate.center_distance
    border_penalty = 0.25 if candidate.touches_border else 0.0

    return area_score + center_score - border_penalty


def rank_masks(masks: list[dict]) -> list[MaskCandidate]:
    """Filter, score, and rank SAM masks."""
    candidates: list[MaskCandidate] = []

    for mask in masks:
        mask_bool = np.asarray(mask).astype(bool)

        area_ratio = compute_mask_area_ratio(mask_bool)
        if not is_mask_area_valid(area_ratio):
            continue

        candidate = MaskCandidate(
            mask=mask_bool,
            area_ratio=area_ratio,
            mask_center= compute_mask_center(mask_bool),
            center_distance=compute_mask_center_distance(mask_bool),
            touches_border=mask_touches_border(mask_bool),
        )
        candidate.score = score_mask_candidate(candidate)
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates

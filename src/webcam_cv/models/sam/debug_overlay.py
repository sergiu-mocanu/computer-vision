import random

import cv2
import numpy as np
from typing import Tuple

from webcam_cv.models.sam.mask_candidate import MaskCandidate
from webcam_cv.models.sam.mask_ranker import compute_mask_center


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay a boolean mask on the frame."""
    result = frame.copy()
    result[mask] = (0.6 * result[mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    return result



def draw_mask_metadata(frame: np.ndarray, candidate: MaskCandidate, rank: int, y: int) -> None:
    """Draw ranking metadata for a mask candidate on screen."""
    text = (
        f'#{rank} '
        f'area={candidate.area_ratio:.3f} '
        f'center={candidate.center_distance:.3f} '
        f'border={candidate.touches_border} '
        f'score={candidate.score:.3f}'
    )

    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 0, 200),
        1,
        cv2.LINE_AA,
    )


def draw_mask_center(frame: np.ndarray, candidate: MaskCandidate, rank: int) -> None:
    text = f'#{rank}'

    mask_cx, mask_cy = candidate.mask_center

    cv2.putText(
        frame,
        text,
        (mask_cx, mask_cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 0, 200),
        1,
        cv2.LINE_AA,
    )


def draw_mask_contour(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw the contour of a binary mask on the frame."""
    ran_r = random.randint(0, 255)
    ran_g = random.randint(0, 255)
    ran_b = random.randint(0, 255)
    color = (ran_g, ran_b, ran_r)

    result = frame.copy()
    mask_uint8 = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, thickness=2)

    return result
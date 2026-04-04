from typing import Tuple

import cv2
import numpy as np
from distinctipy import distinctipy

from webcam_cv.pipeline.sam.mask_candidate import MaskCandidate


contour_color = Tuple[int, int, int]


def generate_distinct_colors(nb_colors: int) -> list[contour_color]:
    """Generate visually distinct colors for mask contour overlay."""
    colors = distinctipy.get_colors(nb_colors)
    colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

    return colors_rgb


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay a boolean mask on the frame."""
    result = frame.copy()
    result[mask] = (0.6 * result[mask] + 0.4 * np.array([0, 150, 0])).astype(np.uint8)
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


def draw_mask_center(frame: np.ndarray, candidate: MaskCandidate, rank: int, color: Tuple[int, int, int]) -> None:
    """Draw the ranking of a mask on the debugging display at its center."""
    text = f'#{rank}'

    mask_cx, mask_cy = candidate.mask_center

    cv2.putText(
        frame,
        text,
        (mask_cx, mask_cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_mask_contour(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Draw the contour of a binary mask on the frame."""

    result = frame.copy()
    mask_uint8 = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, thickness=2)

    return result


def draw_masks(preview_frame: np.ndarray, masks: list[MaskCandidate],
               text_y: int, draw_metadata: bool = True) -> np.ndarray:
    """Display all the relevant mask information on the debugging overlay."""

    distinct_colors = generate_distinct_colors(len(masks))
    current_y = text_y

    for idx, candidate in enumerate(masks):
        current_mask = masks[idx].mask
        preview_frame = draw_mask_contour(preview_frame, current_mask, distinct_colors[idx])

        if draw_metadata:
            draw_mask_metadata(preview_frame, candidate, idx, current_y)

        current_y += text_y

    for idx, candidate in enumerate(masks):
        draw_mask_center(preview_frame, candidate, idx, distinct_colors[idx])

    return preview_frame




import cv2
import numpy as np
from distinctipy import distinctipy

from webcam_cv.pipeline.sam.crop_utils import compute_mask_bbox
from webcam_cv.pipeline.sam.mask_candidate import MaskCandidate


color = tuple[int, int, int]


def generate_distinct_colors(nb_colors: int) -> list[color]:
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


def draw_mask_center(frame: np.ndarray, candidate: MaskCandidate, rank: int, font_color: color) -> None:
    """Draw the ranking of a mask on the debugging display at its center."""
    text = f'#{rank}'

    mask_cx, mask_cy = candidate.mask_center

    cv2.putText(
        frame,
        text,
        (mask_cx, mask_cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        font_color,
        1,
        cv2.LINE_AA,
    )


def draw_mask_contour(frame: np.ndarray, mask: np.ndarray, contour_color: color) -> np.ndarray:
    """Draw the contour of a binary mask on the frame."""

    result = frame.copy()
    mask_uint8 = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, contour_color, thickness=2)

    return result


def draw_mask_bbox(frame: np.ndarray, candidate: MaskCandidate, outline_color: color, thickness: int = 2) -> None:
    """Draw the bounding box of a mask on the frame."""
    mask = candidate.mask
    x_min, y_min, x_max, y_max = compute_mask_bbox(mask)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), outline_color, thickness)


def draw_masks(frame: np.ndarray, candidates: list[MaskCandidate],
               text_y: int, draw_metadata: bool = True) -> np.ndarray:
    """Display all the relevant mask information on the window overlay."""
    result = frame.copy()

    distinct_colors = generate_distinct_colors(len(candidates))
    current_y = text_y

    for idx, candidate in enumerate(candidates):
        current_mask = candidates[idx].mask
        result = draw_mask_contour(result, current_mask, distinct_colors[idx])

        if draw_metadata:
            draw_mask_metadata(frame, candidate, idx, current_y)

        current_y += text_y

    for idx, candidate in enumerate(candidates):
        draw_mask_center(result, candidate, idx, distinct_colors[idx])

    return result

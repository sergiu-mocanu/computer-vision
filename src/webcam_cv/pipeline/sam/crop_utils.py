from typing import Tuple
import numpy as np


def compute_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute the bounding box of a boolean mask."""
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError('Cannot compute bounding box of an empty mask.')

    x_min = int(xs.min())
    x_max = int(xs.max()) + 1
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1

    return x_min, y_min, x_max, y_max


def crop_frame_to_mask_bbox(frame: np.ndarray, mask: np.ndarray, padding: int = 10) -> np.ndarray:
    """Crop a frame to the padded bounding box of a mask."""
    h, w = frame.shape[:2]
    x_min, y_min, x_max, y_max = compute_mask_bbox(mask)

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    return frame[y_min:y_max, x_min:x_max].copy()
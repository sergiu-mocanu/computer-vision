import cv2
from PIL import Image
import numpy as np
from typing import Tuple


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Converts a BGR image to PIL Image (red, green, blue)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def reduce_res(frame_bgr: np.ndarray, new_res: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Reduce image resolution. Useful for saving computational resources."""
    return cv2.resize(frame_bgr, new_res)


def is_image_unchanged(current_frame: np.ndarray, previous_frame: np.ndarray, threshold: int = 2) -> bool:
    """Determine if a frame has been changed significantly.

    Skipping inference when change is negligible reduces computational resources.
    """
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, prev_gray)
    motion_score = diff.mean()

    return motion_score < threshold
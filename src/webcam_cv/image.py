import time
import os

import cv2
from PIL import Image
import numpy as np

from webcam_cv.config import AppConfig


def bgr_2_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Converts a BGR image to PIL Image (red, green, blue)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def apply_gamma(config: AppConfig, frame: np.ndarray):
    """Apply image gamma correction to customize brightness."""
    inv_gamma = 1.0 / config.gamma
    table = ((np.arange(256) / 255.0) ** inv_gamma) * 255
    table = table.astype('uint8')
    return cv2.LUT(frame, table)


def adjust_brightness_contrast(config: AppConfig, frame: np.ndarray) -> np.ndarray:
    """Adjust global contrast and brightness."""
    return cv2.convertScaleAbs(frame, alpha=config.contrast, beta=config.brightness)


def reduce_res(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Reduce image resolution while keeping the aspect ratio.

    Reducing the resolution saves computational resources.
    Keeping the aspect ratio avoid distortion in object proportions.
    """
    h, w = frame.shape[:2]

    if w <= max_width:
        return frame

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def is_scene_static(current_frame: np.ndarray, previous_frame: np.ndarray, threshold: float = 1.5) -> bool:
    """Determine if the captured scene has been changed significantly.

    Skipping inference when change is negligible reduces computational resources.
    """
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, prev_gray)
    motion_score = diff.mean()

    return motion_score < threshold


def write_image_locally(config: AppConfig, frame: np.ndarray) -> None:
    """Save captured frame locally."""
    filename = f'snapshot_{int(time.time())}.jpg'
    folder_path = AppConfig.saved_photos_folder
    filepath = os.path.join(folder_path, filename)

    if not os.path.exists(folder_path):
        os.makedirs(config.saved_photos_folder)

    cv2.imwrite(filename=filepath, img=frame)
    print(f'Saved {filepath}')
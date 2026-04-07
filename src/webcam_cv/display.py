import cv2
import numpy as np

from webcam_cv.config import AppConfig


def init_window(config: AppConfig) -> None:
    """Initialize display window."""
    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.window_name, config.window_width, config.window_height)


def draw_text(frame: np.ndarray, text: str, y: int, scale: float = 0.6) -> None:
    """Draw status text on top of the webcam image."""
    cv2.putText(frame,
                text=text,
                org=(20, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=scale,
                color=(200, 0, 200),
                thickness=2,
                lineType=cv2.LINE_AA
                )


def show(config: AppConfig, frame: np.ndarray) -> None:
    """Display a frame in an OpenCV window."""
    cv2.imshow(config.window_name, frame)

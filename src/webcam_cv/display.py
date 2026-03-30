import cv2
import numpy as np

from webcam_cv.config import AppConfig


debug_window_name = "Debug"


def init_window(config: AppConfig, debug_mode: bool = False) -> None:
    """Initialize display window."""
    if debug_mode:
        cv2.namedWindow(debug_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(debug_window_name, config.window_width, config.window_height)
    else:
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.window_name, config.window_width, config.window_height)


def draw_text(frame: np.ndarray, text: str, y: int, scale: float = 0.7) -> None:
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


def show(config: AppConfig, frame: np.ndarray, debug_mode: bool = False) -> None:
    """Display a frame in an OpenCV window."""
    if debug_mode:
        cv2.imshow(debug_window_name, frame)
    else:
        cv2.imshow(config.window_name, frame)
import cv2
import numpy as np


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


def show(window_name: str, frame: np.ndarray) -> None:
    """Display a frame in an OpenCV window."""
    cv2.imshow(window_name, frame)
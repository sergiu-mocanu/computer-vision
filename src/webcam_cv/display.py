import cv2
import numpy as np

from webcam_cv.config import AppConfig


text_color = (200, 0, 200)


def init_window(config: AppConfig) -> None:
    """Initialize display window."""
    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.window_name, config.window_width, config.window_height)


def draw_text(frame: np.ndarray, text: str, y: int, scale: float = 0.6) -> None:
    """Draw status text on top of the webcam image."""
    cv2.putText(frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                text_color,
                2,
                cv2.LINE_AA
                )


def draw_text_top_right(frame: np.ndarray, text: str, y: int, scale: float = 0.7, margin: int = 20) -> None:
    """Draw right-aligned text near the top-right corner of the frame."""
    (text_w, _), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        2,
    )
    x = frame.shape[1] - text_w - margin

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        2,
        cv2.LINE_AA,
    )


def draw_label_line(
    frame: np.ndarray,
    idx: int,
    label: str,
    confidence: float,
    y: int,
    label_color: tuple[int, int, int],
    scale: float = 0.6,
    x: int = 20,
) -> None:
    """Draw a label line whose prefix color matches the mask box color."""
    prefix = f'Label #{idx}: '
    suffix = f'{label} | Confidence: {confidence:.3f}'

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    cv2.putText(
        frame,
        prefix,
        (x, y),
        font,
        scale,
        label_color,
        thickness,
        cv2.LINE_AA,
    )

    (prefix_width, _), _ = cv2.getTextSize(prefix, font, scale, thickness)

    cv2.putText(
        frame,
        suffix,
        (x + prefix_width, y),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def show(config: AppConfig, frame: np.ndarray) -> None:
    """Display a frame in an OpenCV window."""
    cv2.imshow(config.window_name, frame)

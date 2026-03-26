import cv2

from webcam_cv.config import AppConfig
from webcam_cv.utils.image import apply_gamma, adjust_brightness_contrast


class Camera:
    """Simple wrapper around OpenCV VideoCapture for webcam frame acquisition."""

    def __init__(self, index: int = 0) -> None:
        """Initialize the webcam capture device."""
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError('Could not open webcam')


    def read(self, config: AppConfig):
        """Read a single frame from the webcam and apply brightness, contrast and gamma correction."""
        ok, frame = self.cap.read()
        frame = adjust_brightness_contrast(config, frame)
        frame = apply_gamma(config, frame)
        return ok, frame


    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        self.cap.release()
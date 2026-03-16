import cv2


class Camera:
    """Simple wrapper around OpenCV VideoCapture for webcam frame acquisition."""

    def __init__(self, index: int = 0) -> None:
        """Initialize the webcam capture device."""
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError('Could not open webcam')


    def read(self):
        """Read a single frame from the webcam."""
        ok, frame = self.cap.read()
        return ok, frame


    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        self.cap.release()
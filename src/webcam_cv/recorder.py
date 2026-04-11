import os
import time

import cv2
import numpy as np

from webcam_cv.config import AppConfig


class VideoRecorder:
    """Write displayed frames to a video file."""

    def __init__(self, config: AppConfig, frame: np.ndarray, codec: str = 'mp4v') -> None:
        """Initialize the video recorder."""
        fourcc = cv2.VideoWriter_fourcc(*codec)

        folder_path = config.saved_photos_folder
        if not os.path.exists(folder_path):
            os.makedirs(config.saved_photos_folder)

        h, w = frame.shape[:2]

        record_output_path: str = os.path.join(config.saved_photos_folder, f'pipeline_demo_{int(time.time())}.mp4')
        self.writer = cv2.VideoWriter(record_output_path, fourcc, config.record_output_fps, (w, h))

        if not self.writer.isOpened():
            raise RuntimeError(f'Could not open video writer for {record_output_path!r}.')

    def write(self, frame: np.ndarray) -> None:
        """Write one BGR frame to the output video."""
        self.writer.write(frame)

    def release(self) -> None:
        """Release the underlying video writer."""
        self.writer.release()


def ensure_recorder(config: AppConfig, recorder: VideoRecorder | None, display: np.ndarray) -> VideoRecorder | None:
    """Initialize the recorder lazily if recording is enabled."""
    if recorder is None and config.record_output:
        return VideoRecorder(config, display)
    return recorder

from typing import cast, Optional

import cv2
import numpy as np

from webcam_cv.config import AppConfig
from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.models.factory import create_model_from_spec
from webcam_cv.models.sam_segmenter import SamSegmenter
from webcam_cv.pipeline.segmentation_stage import generate_ranked_masks

from webcam_cv.camera import Camera
from webcam_cv.pipeline.sam.mask_overlay import draw_masks
from webcam_cv.image import write_image_locally
from webcam_cv.display import init_window, draw_text, show, debug_window_name


def run_segmentation_app(config: AppConfig) -> None:
    """Run interactive SAM segmentation on frozen webcam frames."""

    # --------------------------------------------------------
    # Initialize components (camera, model, anomaly scorer)
    # --------------------------------------------------------
    camera = Camera()

    init_window(config)

    mode_spec = MODE_REGISTRY[config.app_mode]
    segmenter = cast(SamSegmenter, create_model_from_spec(config, mode_spec))

    frame: Optional[np.ndarray] = None
    frozen_frame: Optional[np.ndarray] = None
    preview_frame: Optional[np.ndarray] = None
    last_infer_ms = 0.0

    print(f'Running Segmentation mode on device: {config.gpu_name if config.gpu_name else 'CPU'}')
    print(f'Model: {segmenter.model_name}\n')

    print('Controls:')
    print('  f = freeze current frame and segment')
    print('  r = return to live view')
    print('  s = save current frame')
    print('  q = quit')

    # --------------------------------------------------------
    # Main realtime loop
    # --------------------------------------------------------
    while True:
        if frozen_frame is None:
            ok, frame = camera.read(config)
            if frame is None:
                break
            display = frame.copy()
        else:
            if preview_frame is None:
                break
            display = preview_frame.copy()

        key = cv2.waitKey(1) & 0xFF

        # --------------------------------------------------------
        # Handle user input (freeze image, save frame, etc.)
        # --------------------------------------------------------
        if key == ord('q'):
            break

        if key == ord('s') and frozen_frame is not None:
            write_image_locally(config, display)

        if key == ord('r'):
            frozen_frame = None
            preview_frame = None
            cv2.destroyWindow(debug_window_name)

        # --------------------------------------------------------
        # Run inference, segment and rank areas of the image
        # --------------------------------------------------------
        if key == ord('f') and frozen_frame is None:
            if frame is None:
                break
            frozen_frame = frame.copy()

            if frozen_frame is None:
                break
            ranked_masks, last_infer_ms = generate_ranked_masks(config, segmenter, frozen_frame)

            text_y = 25
            preview_frame = draw_masks(display, ranked_masks, text_y)

        draw_text(display, 'Mode: SAM', 30)
        draw_text(display, f'Model: {segmenter.model_name}', 60)

        if frozen_frame is None:
            draw_text(display, 'Press f to freeze and segment', 100)
        else:
            draw_text(display, 'Frozen frame segmented', 100)
            draw_text(display, f'Inference: {last_infer_ms:.1f} ms', 130)

        if frozen_frame is None:
            show(config, display)
        else:
            if preview_frame is None:
                break
            show(config, preview_frame)

    camera.release()
    cv2.destroyAllWindows()

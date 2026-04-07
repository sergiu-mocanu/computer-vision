from typing import Optional, cast

import cv2
import numpy as np

from webcam_cv.config import AppConfig
from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.models.factory import create_model_from_spec
from webcam_cv.models.clip_embedder import ClipEmbedder
from webcam_cv.pipeline.labeling_stage import select_best_image_prompts

from webcam_cv.camera import Camera
from webcam_cv.display import draw_text, show, init_window
from webcam_cv.image import is_scene_static, write_image_locally


def run_labelling_app(config: AppConfig) -> None:
    """Run the real-time webcam image-prompt similarity using CLIP model.

    Initializes the camera and the CV model.
    Captures frames in a loop, compute similarity
    and renders the result in a GUI window.
    """

    # --------------------------------------------------------
    # Initialize components (camera, model)
    # --------------------------------------------------------
    camera = Camera()

    init_window(config)

    mode_spec = MODE_REGISTRY[config.app_mode]
    embedder = cast(ClipEmbedder, create_model_from_spec(config, mode_spec))

    frame_index = 0
    last_infer_ms = 0.0
    prompt_scores: list[tuple[str, float]] = []
    best_prompt: Optional[str] = None
    best_score: Optional[str] = None
    previous_frame: Optional[np.ndarray] = None

    print(f'Running Labeling mode on device: {config.gpu_name if config.gpu_name else 'CPU'}')
    print(f'Model: {embedder.model_name}\n')

    print('Controls:')
    print('  s = save current frame')
    print('  q = quit')

    # --------------------------------------------------------
    # Main realtime loop
    # --------------------------------------------------------
    while True:
        ok, frame = camera.read(config)
        if not ok:
            break

        frame_index = (frame_index + 1) % config.inference_frame_stride
        display = frame.copy()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # --------------------------------------------------------
        # Compute image-prompt similarity
        # --------------------------------------------------------
        if frame_index % config.inference_frame_stride == 0:
            if previous_frame is not None:
                if is_scene_static(frame, previous_frame):
                    continue

            prompt_scores, last_infer_ms = select_best_image_prompts(config, embedder, frame)
            best_prompt, best_score = prompt_scores[0]

            previous_frame = frame

        # --------------------------------------------------------
        # Render overlay and display frame
        # --------------------------------------------------------
        draw_text(display, f'Inference: {last_infer_ms:.1f} ms', 30)

        if best_prompt:
            draw_text(display, f'Best prompt: {best_prompt}', 70)
            draw_text(display, f'Confidence: {best_score:.3f}', 100)

            y = 140
            for prompt, score in prompt_scores[1:]:
                draw_text(display, f'{prompt}: {score:.3f}', y, scale=0.6)
                y += 20

        show(config, display)

        if key == ord('s'):
            write_image_locally(config, display)

    camera.release()
    cv2.destroyAllWindows()
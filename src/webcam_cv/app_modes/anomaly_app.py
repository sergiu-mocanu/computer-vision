import time
from typing import cast

import cv2

from webcam_cv.config import AppConfig
from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.models.dinov2_embedder import DinoV2Embedder
from webcam_cv.models.factory import create_model_from_spec
from webcam_cv.anomaly.scorer import AnomalyScorer
from webcam_cv.camera import Camera
from webcam_cv.display import draw_text, show, init_window
from webcam_cv.utils.image import is_scene_static, write_image_locally


def run_anomaly_app(config: AppConfig) -> None:
    """Run the real-time webcam anomaly detection application using DINOv2 model.

    Initializes the camera, embedding model, and anomaly scorer.
    Captures frames in a loop, processes embeddings at the configured
    stride, computes anomaly scores, and renders the result in a GUI window.
    """

    # --------------------------------------------------------
    # Initialize components (camera, model, anomaly scorer)
    # --------------------------------------------------------
    camera = Camera()

    mode_spec = MODE_REGISTRY[config.app_mode]
    embedder = cast(DinoV2Embedder, create_model_from_spec(config, mode_spec))
    scorer = AnomalyScorer(
        z_threshold=config.anomaly_z_threshold,
        ema_alpha=config.ema_alpha
    )

    print(f'Running Anomaly mode on device: {config.gpu_name if config.gpu_name else 'CPU'}')
    print(f'Model: {embedder.model_name}\n')

    print('Controls:')
    print('  r = record normal reference frames')
    print('  c = clear reference')
    print('  s = save current frame')
    print('  q = quit')

    frame_index = 0
    last_infer_ms = 0.0
    previous_frame = None

    init_window(config)

    # --------------------------------------------------------
    # Main realtime loop
    # --------------------------------------------------------
    while True:
        ok, frame = camera.read(config)
        if not ok:
            break

        frame_index += 1
        display = frame.copy()

        # --------------------------------------------------------
        # Handle user input (record reference, save frame, etc.)
        # --------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c'):
            scorer.clear()
            print('Reference cleared')

        # --------------------------------------------------------
        # Collect reference embeddings (normal scene modeling)
        # --------------------------------------------------------
        if key == ord('r'):
            embeddings = embedder.collect_normal_frames(camera=camera, config=config)

            if embeddings:
                scorer.fit_reference(embeddings)
                print('Reference embedding created')

        # --------------------------------------------------------
        # Run inference and compute anomaly score
        # --------------------------------------------------------
        if scorer.reference_embedding is not None and frame_index % config.inference_frame_stride == 0:
            if previous_frame is not None:
                if is_scene_static(frame, previous_frame):
                    continue

            previous_frame = frame

            start = time.perf_counter()
            embedding = embedder.embed(frame)
            scorer.score(embedding)
            last_infer_ms = (time.perf_counter() - start) * 1000

        # --------------------------------------------------------
        # Render overlay and display frame
        # --------------------------------------------------------
        if scorer.reference_embedding is None:
            draw_text(display, 'Status: no reference yet (press r)', 30)

        else:
            draw_text(display, 'Reference: ready', 30)

            score = scorer.smoothed_score

            if score is not None:
                status = 'ANOMALY' if scorer.is_anomaly(score) else 'NORMAL'
                draw_text(display, f'Status: {status}', 60)
                draw_text(display, f'Anomaly score: {score:.4f}', 100)
                draw_text(display, f'Threshold: {config.anomaly_z_threshold:.4f}', 130)
                draw_text(display, f'Inference: {last_infer_ms:.1f} ms', 160)

        show(config, display)

        if key == ord('s'):
            write_image_locally(config, display)

    camera.release()
    cv2.destroyAllWindows()
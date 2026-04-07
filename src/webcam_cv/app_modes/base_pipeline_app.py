from typing import Optional, cast

import cv2
import numpy as np

from webcam_cv.config import AppConfig
from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.models.factory import create_model_from_spec
from webcam_cv.pipeline.dino.anomaly_scorer import AnomalyScorer
from webcam_cv.models.dinov2_embedder import DinoV2Embedder
from webcam_cv.models.clip_embedder import ClipEmbedder
from webcam_cv.pipeline.anomaly_stage import score_frame_anomaly
from webcam_cv.pipeline.labeling_stage import select_best_image_prompts

from webcam_cv.camera import Camera
from webcam_cv.display import draw_text, show, init_window
from webcam_cv.image import write_image_locally, is_scene_static


def run_base_pipeline_app(config: AppConfig) -> None:
    """Run in parallel the real-time webcam anomaly detection application (DINOv2)
    and the text-image similarity application (CLIP).

    Initializes the camera, embedding model, and anomaly scorer.
    Capture a set of normal frames as a baseline as well as detects the best prompt
    for the normal scene. Text-image similarity is then computed only when an anomaly is detected
    in order to save computational resources.
    """

    # --------------------------------------------------------
    # Initialize components (camera, model, anomaly scorer)
    # --------------------------------------------------------
    camera = Camera()

    init_window(config)

    mode_spec = MODE_REGISTRY[config.app_mode]

    detector_role = 'detector'
    classifier_role = 'classifier'
    detector = cast(DinoV2Embedder, create_model_from_spec(config, mode_spec, detector_role))
    classifier = cast(ClipEmbedder, create_model_from_spec(config, mode_spec, classifier_role))

    scorer = AnomalyScorer(config)

    frame_index = 0
    previous_frame: Optional[np.ndarray] = None
    last_detector_ms = 0.0
    last_classifier_ms = 0.0

    score = None
    is_anomaly = False

    best_prompt: Optional[str] = None
    best_score: Optional[str] = None

    print('Mode: pipeline')
    print(f'Running Base Pipeline mode on device: {config.gpu_name if config.gpu_name else 'CPU'}')
    print(f'Detector: {detector.model_name}')
    print(f'Classifier: {classifier.model_name}\n')

    print('Controls:')
    print('  r = record normal reference frames')
    print('  c = clear reference')
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

        # --------------------------------------------------------
        # Handle user input (record reference, save frame, reset normal stage)
        # --------------------------------------------------------
        if key == ord('c'):
            scorer.clear()
            best_prompt = None
            best_score = None
            print('Reference cleared')

        if key == ord('q'):
            break

        # --------------------------------------------------------
        # Collect reference embeddings (normal scene modeling)
        # --------------------------------------------------------
        if key == ord('r'):
            embeddings = detector.collect_normal_frames(config, camera)

            scorer.fit_reference(embeddings)
            print('Reference embedding created')

            prompt_scores, last_classifier_ms = select_best_image_prompts(config, classifier, frame)

            best_prompt, best_score = prompt_scores[0]

        # --------------------------------------------------------
        # Run inference and compute anomaly score
        # --------------------------------------------------------
        if scorer.reference_embedding is not None and frame_index % config.inference_frame_stride == 0:

            if previous_frame is not None:
                if is_scene_static(frame, previous_frame):
                    continue

            score, is_anomaly, last_detector_ms = score_frame_anomaly(detector, scorer, frame)

            # --------------------------------------------------------
            # Only run CLIP when anomaly is detected.
            # --------------------------------------------------------
            if is_anomaly:
                prompt_scores, last_classifier_ms = select_best_image_prompts(config, classifier, frame)

                if prompt_scores:
                    best_prompt, best_score = prompt_scores[0]

            previous_frame = frame

        # --------------------------------------------------------
        # Render overlay and display frame
        # --------------------------------------------------------
        if scorer.reference_embedding is None:
            draw_text(display, 'Status: no reference yet (press r)', 30)

        else:
            if score is not None:
                status = 'ANOMALY' if is_anomaly else 'NORMAL'
                draw_text(display, f'Status: {status}', 30)
                draw_text(display, f'Anomaly score: {score:.4f}', 60)
                draw_text(display, f'Threshold: {config.anomaly_z_threshold:.2f}', 90)
                draw_text(display, f'Detector inference: {last_detector_ms:.1f} ms', 120)

                if best_prompt is not None and best_score is not None:
                    draw_text(display, f'Label: {best_prompt}', 160, scale=0.6)
                    draw_text(display, f'Confidence: {best_score:.3f}', 190, scale=0.6)
                    draw_text(display, f'Classifier inference: {last_classifier_ms:.1f} ms', 220, scale=0.6)

        show(config, display)

        if key == ord('s'):
            write_image_locally(config, display)

    camera.release()
    cv2.destroyAllWindows()

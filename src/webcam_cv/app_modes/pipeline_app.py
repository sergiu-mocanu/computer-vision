import time
from typing import Optional, cast

import cv2

from webcam_cv.config import AppConfig
from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.models.factory import create_model_from_spec
from webcam_cv.anomaly.scorer import AnomalyScorer
from webcam_cv.models.dinov2_embedder import DinoV2Embedder
from webcam_cv.models.clip_embedder import ClipEmbedder
from webcam_cv.camera import Camera
from webcam_cv.display import draw_text, show, init_window
from webcam_cv.utils.image import write_image_locally
from webcam_cv.utils.image import is_image_unchanged

def run_pipeline_app(config: AppConfig) -> None:
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

    mode_spec = MODE_REGISTRY[config.app_mode]

    detector_role = 'detector'
    classifier_role = 'classifier'

    detector_spec = mode_spec['models'][detector_role]
    classifier_spec = mode_spec['models'][classifier_role]

    detector = cast(DinoV2Embedder, create_model_from_spec(detector_spec, role=detector_role))
    classifier = cast(ClipEmbedder, create_model_from_spec(classifier_spec, role=classifier_role))

    scorer = AnomalyScorer(
        z_threshold=config.anomaly_z_threshold,
        ema_alpha=config.ema_alpha
    )

    frame_index = 0
    previous_frame = None
    last_detector_ms = 0.0
    last_classifier_ms = 0.0

    smoothed_score = None
    is_anomaly = False

    best_prompt: Optional[str] = None
    best_score: Optional[str] = None

    print('Mode: pipeline')
    print(f'Detector: {detector.model_name}')
    print(f'Classifier: {classifier.model_name}\n')

    print('Controls:')
    print('  r = record normal reference frames')
    print('  c = clear reference')
    print('  s = save current frame')
    print('  q = quit')

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

        key = cv2.waitKey(1) & 0xFF

        # --------------------------------------------------------
        # Handle user input (record reference, save frame, etc.)
        # --------------------------------------------------------
        if key == ord('q'):
            break

        if key == ord('c'):
            scorer.clear()
            best_prompt = None
            best_score = None
            print('Reference cleared')

        if key == ord('s'):
            write_image_locally(frame)

        # --------------------------------------------------------
        # Collect reference embeddings (normal scene modeling)
        # --------------------------------------------------------
        if key == ord('r'):
            embeddings = detector.collect_normal_frames(camera=camera, config=config)

            if embeddings:
                scorer.fit_reference(embeddings)
                best_prompt = None
                best_score = None
                print('Reference embedding created')

            classifier_start = time.perf_counter()
            prompt_scores = classifier.score_prompts(frame, config.clip_prompts)
            last_classifier_ms = (time.perf_counter() - classifier_start) * 1000.0

            if prompt_scores:
                best_prompt, best_score = prompt_scores[0]

        # --------------------------------------------------------
        # Run inference and compute anomaly score
        # --------------------------------------------------------
        if scorer.reference_embedding is not None and frame_index % config.inference_frame_stride == 0:
            if previous_frame is not None:
                if is_image_unchanged(frame, previous_frame):
                    continue

            previous_frame = frame

            detector_start = time.perf_counter()
            embedding = detector.embed(frame)
            smoothed_score = scorer.score(embedding)
            last_detector_ms = (time.perf_counter() - detector_start) * 1000.0

            if smoothed_score is not None:
                is_anomaly = scorer.is_anomaly(smoothed_score)

            # --------------------------------------------------------
            # Only run CLIP when anomaly is detected.
            # --------------------------------------------------------
            if is_anomaly:
                classifier_start = time.perf_counter()
                prompt_scores = classifier.score_prompts(frame, config.clip_prompts)
                last_classifier_ms = (time.perf_counter() - classifier_start) * 1000.0

                if prompt_scores:
                    best_prompt, best_score = prompt_scores[0]

        # --------------------------------------------------------
        # Render overlay and display frame
        # --------------------------------------------------------
        if scorer.reference_embedding is None:
            draw_text(display, 'Status: no reference yet (press r)', 30)

        else:
            if smoothed_score is not None:
                status = 'ANOMALY' if is_anomaly else 'NORMAL'
                draw_text(display, f'Status: {status}', 30)
                draw_text(display, f'Anomaly score: {smoothed_score:.4f}', 60)
                draw_text(display, f'Threshold: {config.anomaly_z_threshold:.2f}', 90)
                draw_text(display, f'Detector inference: {last_detector_ms:.1f} ms', 120)

                if best_prompt is not None and best_score is not None:
                    draw_text(display, f'Label: {best_prompt}', 160, scale=0.6)
                    draw_text(display, f'Confidence: {best_score:.3f}', 190, scale=0.6)
                    draw_text(display, f'Classifier inference: {last_classifier_ms:.1f} ms', 220, scale=0.6)

        show(config, display)

    camera.release()
    cv2.destroyAllWindows()

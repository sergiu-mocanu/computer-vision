import time
from typing import cast, Optional
from dataclasses import dataclass, field

import cv2
import numpy as np

from webcam_cv.camera import Camera
from webcam_cv.config import AppConfig
from webcam_cv.display import init_window, draw_text, show

from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.image import write_image_locally, is_scene_static
from webcam_cv.models.factory import create_model_from_spec
from webcam_cv.models.clip_embedder import ClipEmbedder
from webcam_cv.models.dinov2_embedder import DinoV2Embedder
from webcam_cv.models.sam_segmenter import SamSegmenter

from webcam_cv.pipeline.dino.anomaly_scorer import AnomalyScorer
from webcam_cv.pipeline.anomaly_stage import score_frame_anomaly
from webcam_cv.pipeline.sam.mask_candidate import MaskCandidate
from webcam_cv.pipeline.labeling_stage import select_best_mask_with_clip
from webcam_cv.pipeline.sam.mask_overlay import draw_masks_bbox
from webcam_cv.pipeline.segmentation_stage import generate_ranked_masks


@dataclass
class PipelineValues:
    latest_score: Optional[float] = None
    latest_is_anomaly = False
    latest_best_label: Optional[str] = None
    latest_best_confidence: Optional[float] = None
    latest_best_candidate = None
    latest_ranked_masks: list[MaskCandidate] = field(default_factory=list)

    last_detector_ms = 0.0
    last_segmenter_ms = 0.0
    last_classifier_ms = 0.0


    def reset_latest_values(self):
        self.latest_score = None
        self.latest_is_anomaly = False
        self.latest_best_label = None
        self.latest_best_confidence = None
        self.latest_best_candidate = None
        self.latest_ranked_masks = []


    def reset_segmenter_classifier_values(self):
        self.latest_best_label = None
        self.latest_best_confidence = None
        self.latest_best_candidate = None
        self.latest_ranked_masks = []
        self.last_segmenter_ms = 0.0
        self.last_classifier_ms = 0.0


def run_segmented_pipeline_app(config: AppConfig) -> None:
    """Run the three-stage DINOv2 + SAM + CLIP pipeline.

    Pipeline:
    1. DINOv2 scores the frame for anomaly detection
    2. SAM proposes and ranks candidate masks if the frame is anomalous
    3. CLIP scores cropped candidate regions and selects the best one
    """

    # --------------------------------------------------------
    # Initialize components (camera, model, anomaly scorer)
    # --------------------------------------------------------
    camera = Camera()

    init_window(config)

    detector_role = 'detector'
    segmenter_role = 'segmenter'
    classifier_role = 'classifier'

    mode_spec = MODE_REGISTRY[config.app_mode]
    detector = cast(DinoV2Embedder, create_model_from_spec(config, mode_spec, detector_role))
    segmenter = cast(SamSegmenter, create_model_from_spec(config, mode_spec, segmenter_role))
    classifier = cast(ClipEmbedder, create_model_from_spec(config, mode_spec, classifier_role))

    scorer = AnomalyScorer(config)

    frame_index = 0

    plval = PipelineValues()

    previous_frame: Optional[np.ndarray] = None
    last_anomaly_processing: Optional[float] = None
    anomaly_processing_delay_s = 4.0

    print(f'Running Segmented Pipeline mode on device: {config.gpu_name if config.gpu_name else 'CPU'}')
    print(f'Detector: {detector.model_name}')
    print(f'Segmenter: {segmenter.model_name}')
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
            plval.reset_latest_values()
            print('Reference cleared')

        if key == ord('q'):
            break

        if key == ord('r'):
            embeddings = detector.collect_normal_frames(config, camera)
            scorer.fit_reference(embeddings)

            plval.reset_latest_values()
            print('Reference embedding created')

        if (scorer.reference_embedding is not None
                and frame_index % config.inference_frame_stride == 0):

            if previous_frame is not None:
                if is_scene_static(frame, previous_frame, threshold=2):
                    continue

            (
                plval.latest_score,
                plval.latest_is_anomaly,
                plval.last_detector_ms
            ) = score_frame_anomaly(detector, scorer, frame)

            now = time.perf_counter()

            previous_frame = frame

            should_run_sam_clip = False

            if plval.latest_is_anomaly:
                if last_anomaly_processing is None:
                    should_run_sam_clip = True
                    last_anomaly_processing = now

                elif now - last_anomaly_processing >= anomaly_processing_delay_s:
                    should_run_sam_clip = True
                    last_anomaly_processing = now

                if should_run_sam_clip:
                    plval.latest_ranked_masks, plval.last_segmenter_ms = generate_ranked_masks(config, segmenter, frame)

                    (
                        plval.latest_best_candidate,
                        plval.latest_best_label,
                        plval.latest_best_confidence,
                        plval.last_classifier_ms,
                    ) = select_best_mask_with_clip(
                        classifier=classifier,
                        frame_bgr=frame,
                        mask_candidates=plval.latest_ranked_masks,
                        prompts=config.clip_prompts,
                    )

        if scorer.reference_embedding is None:
            draw_text(display, 'Status: no reference yet (press r)', 30)

        else:
            if plval.latest_score is not None:
                if plval.latest_is_anomaly:
                    draw_text(display, f'Segmenter inference: {plval.last_segmenter_ms:.1f} ms', 150)
                    draw_text(display, f'Classifier inference: {plval.last_classifier_ms:.1f} ms', 180)

                    if plval.latest_best_candidate is not None:
                        display = draw_masks_bbox(display, [plval.latest_best_candidate])

                    if plval.latest_best_label is not None and plval.latest_best_confidence is not None:
                        draw_text(display, f'Label: {plval.latest_best_label}', 250, scale=0.6)
                        draw_text(display, f'Confidence: {plval.latest_best_confidence:.3f}', 500, scale=0.6)

                status = 'ANOMALY' if plval.latest_is_anomaly else 'NORMAL'
                draw_text(display, f'Status: {status}', 30)
                draw_text(display, f'Anomaly score: {plval.latest_score:.4f}', 60)
                draw_text(display, f'Threshold: {config.anomaly_z_threshold:.2f}', 90)
                draw_text(display, f'Detector inference: {plval.last_detector_ms:.1f} ms', 120)

        show(config, display)

        if key == ord('s'):
            write_image_locally(config, display)

    camera.release()
    cv2.destroyAllWindows()


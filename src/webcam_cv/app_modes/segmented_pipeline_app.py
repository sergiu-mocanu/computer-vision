import time
from typing import cast
from dataclasses import dataclass, field

import cv2
import numpy as np

from webcam_cv.config import AppConfig
from webcam_cv.camera import Camera
from webcam_cv.recorder import VideoRecorder, ensure_recorder
from webcam_cv.display import init_window, draw_text, show, draw_text_top_right, draw_label_line
from webcam_cv.image import write_image_locally, is_scene_static

from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.models.factory import create_model_from_spec
from webcam_cv.models.clip_embedder import ClipEmbedder
from webcam_cv.models.dinov2_embedder import DinoV2Embedder

from webcam_cv.pipeline.dino.anomaly_scorer import AnomalyScorer
from webcam_cv.pipeline.anomaly_stage import score_frame_anomaly
from webcam_cv.pipeline.sam.mask_candidate import MaskCandidate
from webcam_cv.pipeline.labeling_stage import select_best_mask_with_clip
from webcam_cv.pipeline.sam.mask_overlay import draw_mask_bbox, generate_distinct_colors, draw_mask_center
from webcam_cv.pipeline.segmentation_stage import generate_ranked_masks


@dataclass
class PipelineValues:
    latest_score: float | None = None
    latest_is_anomaly = False
    mask_prompt_sim: tuple[MaskCandidate, str, float] | None = None
    latest_ranked_masks: list[MaskCandidate] = field(default_factory=list)

    last_detector_ms = 0.0
    last_segmenter_ms = 0.0
    last_classifier_ms = 0.0


    def reset_latest_values(self):
        self.latest_score = None
        self.latest_is_anomaly = False
        self.mask_prompt_sim = None
        self.latest_ranked_masks = []


    def reset_segmenter_classifier_values(self):
        self.mask_prompt_sim = None
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

    recorder: VideoRecorder | None = None

    init_window(config)

    detector_role = 'detector'
    segmenter_role = 'segmenter'
    classifier_role = 'classifier'

    mode_spec = MODE_REGISTRY[config.app_mode]
    detector = cast(DinoV2Embedder, create_model_from_spec(config, mode_spec, detector_role))
    segmenter = create_model_from_spec(config, mode_spec, segmenter_role)
    classifier = cast(ClipEmbedder, create_model_from_spec(config, mode_spec, classifier_role))

    scorer = AnomalyScorer(config)

    frame_index = 0

    plval = PipelineValues()

    previous_frame: np.ndarray | None = None

    now: float | None = None
    next_seglabel_time: float | None = None
    seglabel_delay_s: float = 4.0

    distinct_colors = generate_distinct_colors(config.sam_top_k_masks, exclude_text_color=True)

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

        recorder = ensure_recorder(config, recorder, display)

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
            embeddings = detector.collect_normal_frames(config, camera, recorder)
            scorer.fit_reference(embeddings)

            plval.reset_latest_values()
            print('Reference embedding created')

        if (scorer.reference_embedding is not None
                and frame_index % config.inference_frame_stride == 0):

            if previous_frame is not None:
                if is_scene_static(frame, previous_frame) and not plval.latest_is_anomaly:
                    continue

            (
                plval.latest_score,
                plval.latest_is_anomaly,
                plval.last_detector_ms
            ) = score_frame_anomaly(detector, scorer, frame)

            previous_frame = frame

            now = time.perf_counter()

            if plval.latest_is_anomaly:
                if next_seglabel_time is None:
                    next_seglabel_time = now + seglabel_delay_s

                if now >= next_seglabel_time:
                    plval.latest_ranked_masks, plval.last_segmenter_ms = generate_ranked_masks(config, segmenter, frame)

                    (
                        plval.mask_prompt_sim,
                        plval.last_classifier_ms
                    ) = select_best_mask_with_clip(
                        classifier=classifier,
                        frame_bgr=frame,
                        mask_candidates=plval.latest_ranked_masks,
                        prompts=config.seg_pipe_prompts,
                    )

                    now = time.perf_counter()
                    next_seglabel_time = now + seglabel_delay_s + 2

            else:
                next_seglabel_time = None
                plval.reset_segmenter_classifier_values()

        if scorer.reference_embedding is None:
            draw_text(display, 'Status: no reference yet (press r)', 30)

        else:
            if plval.latest_score is not None:
                if plval.latest_is_anomaly:
                    if plval.mask_prompt_sim is not None:
                        text_y = 60

                        for idx, (candidate, label, confidence) in enumerate(plval.mask_prompt_sim):
                            current_color = distinct_colors[idx]

                            draw_mask_bbox(display, candidate, current_color)
                            draw_mask_center(display, candidate, idx+1, current_color)

                            draw_label_line(
                                display,
                                idx=idx + 1,
                                label=label,
                                confidence=confidence,
                                y=text_y,
                                label_color=current_color,
                                scale=0.6,
                            )
                            text_y += 30

                status = 'ANOMALY' if plval.latest_is_anomaly else 'NORMAL'
                draw_text(display, f'Status: {status}', 30)

                if plval.latest_is_anomaly and next_seglabel_time is not None:
                    remaining_time = max(0.0, next_seglabel_time - now)
                    draw_text_top_right(
                        display,
                        f'Anomaly processing in {remaining_time:.2f}s',
                        y=30
                    )

        if recorder is not None:
            recorder.write(display)

        show(config, display)

        if key == ord('s'):
            write_image_locally(config, display)

    if recorder is not None:
        recorder.release()

    camera.release()
    cv2.destroyAllWindows()


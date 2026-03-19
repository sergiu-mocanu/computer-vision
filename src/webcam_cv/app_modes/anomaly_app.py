import time
import os
import cv2

from webcam_cv.config import AppConfig
from webcam_cv.camera import Camera
from webcam_cv.display import draw_text, show
from webcam_cv.utils.image import is_image_unchanged
from webcam_cv.models.factory import create_embedder
from webcam_cv.anomaly.scorer import AnomalyScorer


def run_anomaly_app(config: AppConfig) -> None:
    """Run the real-time webcam anomaly detection application using DINOv2 model.

    Initializes the camera, embedding model, and anomaly scorer.
    Captures frames in a loop, processes embeddings at the configured
    stride, computes anomaly scores, and renders the result in a GUI
    window.
    """

    # --------------------------------------------------------
    # Initialize components (camera, model, anomaly scorer)
    # --------------------------------------------------------
    camera = Camera()
    embedder = create_embedder(config)
    scorer = AnomalyScorer(config.anomaly_threshold)

    print(f'Running anomaly mode on device: {config.gpu_name if config.gpu_name else 'CPU'} ')
    print(f'Model: {embedder.model_name}\n')

    print('Controls:')
    print('  r = record normal reference frames')
    print('  c = clear reference')
    print('  s = save current frame')
    print('  q = quit')

    frame_index = 0
    last_infer_ms = 0.0
    smoothed_score = None
    previous_frame = None

    # --------------------------------------------------------
    # Main realtime loop
    # --------------------------------------------------------
    while True:
        ok, frame = camera.read()
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
            smoothed_score = None
            print('Reference cleared')

        if key == ord('s'):
            filename = f'snapshot_{int(time.time())}.jpg'
            folder_path = config.saved_photos_folder
            filepath = os.path.join(folder_path, filename)
            if not os.path.exists(folder_path):
                os.makedirs(config.saved_photos_folder)
            cv2.imwrite(filepath, frame)
            print(f'Saved {filepath}')

        # --------------------------------------------------------
        # Collect reference embeddings (normal scene modeling)
        # --------------------------------------------------------
        if key == ord('r'):
            print(f'Recording {config.normal_frames_target} frames')

            embeddings = []
            collected = 0

            while collected < config.normal_frames_target and frame_index % config.frame_stride == 0:
                ok_ref, ref_frame = camera.read()
                if not ok_ref:
                    break

                if collected % config.frame_stride == 0:
                    embeddings.append(embedder.embed(ref_frame))

                collected += 1

                ref_display = ref_frame.copy()
                draw_text(
                    ref_display,
                    f'Recording normal frames: {collected} / {config.normal_frames_target}',
                    30
                )

                show(config.window_name, ref_display)
                cv2.waitKey(1)

            if embeddings:
                scorer.fit_reference(embeddings)
                print('Reference embedding created')

        # --------------------------------------------------------
        # Run inference and compute anomaly score
        # --------------------------------------------------------
        if scorer.reference_embedding is not None and frame_index % config.frame_stride == 0:
            if previous_frame is not None:
                if is_image_unchanged(frame, previous_frame):
                    continue

            previous_frame = frame

            start = time.perf_counter()
            embedding = embedder.embed(frame)
            smoothed_score = scorer.score(embedding)
            last_infer_ms = (time.perf_counter() - start) * 1000

        # --------------------------------------------------------
        # Render overlay and display frame
        # --------------------------------------------------------
        if scorer.reference_embedding is None:
            draw_text(display, 'Status: no reference yet (press r)', 30)

        else:
            draw_text(display, 'Reference: ready', 30)

            if smoothed_score is not None:
                status = 'ANOMALY' if scorer.is_anomaly(smoothed_score) else 'NORMAL'
                draw_text(display, f'Status: {status}', 60)
                draw_text(display, f'Anomaly score: {smoothed_score:.4f}', 90)
                draw_text(display, f'Threshold: {config.anomaly_threshold:.4f}', 120)
                draw_text(display, f'Inference: {last_infer_ms:.1f} ms', 150)

        show(config.window_name, display)

    camera.release()
    cv2.destroyAllWindows()
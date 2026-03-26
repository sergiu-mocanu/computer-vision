import time

from webcam_cv.camera import Camera
from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.models.factory import create_model_from_spec
from webcam_cv.display import init_window, draw_text, show
from webcam_cv.utils.image import write_image_locally

from webcam_cv.models.sam.debug_overlay import *
from webcam_cv.models.sam.mask_ranker import rank_masks


def run_segmentation_app(config: AppConfig) -> None:
    """Run interactive SAM segmentation on frozen webcam frames."""

    # --------------------------------------------------------
    # Initialize components (camera, model, anomaly scorer)
    # --------------------------------------------------------
    camera = Camera()

    mode_spec = MODE_REGISTRY[config.app_mode]
    segmenter = create_model_from_spec(config=config, mode_spec=mode_spec)

    init_window(config)

    frame = None
    frozen_frame = None
    preview_frame = None
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
            if not ok:
                break
            display = frame.copy()
        else:
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

        # --------------------------------------------------------
        # Run inference and segment areas of the image
        # --------------------------------------------------------
        if key == ord('f') and frozen_frame is None:
            assert frame is not None
            frozen_frame = frame.copy()

            start = time.perf_counter()
            masks = segmenter.generate_masks(frozen_frame)
            last_infer_ms = (time.perf_counter() - start) * 1000.0

            preview_frame = frozen_frame.copy()

            # --------------------------------------------------------
            # Select top-k masks (regions)
            # --------------------------------------------------------
            if masks:
                ranked_masks = rank_masks(masks)
                text_y = 25

                preview_frame = draw_best_masks(config, display, ranked_masks, text_y)

        draw_text(display, 'Mode: SAM', 30)
        draw_text(display, f'Model: {segmenter.model_name}', 60)

        if frozen_frame is None:
            draw_text(display, 'Press f to freeze and segment', 100)
        else:
            draw_text(display, 'Frozen frame segmented', 100)
            draw_text(display, f'Inference: {last_infer_ms:.1f} ms', 130)

        show(config, display if frozen_frame is None else preview_frame)

    camera.release()
    cv2.destroyAllWindows()

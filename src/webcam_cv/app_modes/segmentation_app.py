import time

from webcam_cv.camera import Camera
from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.models.factory import create_model_from_spec
from webcam_cv.display import init_window, draw_text, show, debug_window_name
from webcam_cv.utils.image import write_image_locally

from webcam_cv.models.sam.debug_overlay import *
from webcam_cv.models.sam.mask_ranker import rank_masks, ndarray_to_mask_candidate


def run_segmentation_app(config: AppConfig) -> None:
    """Run interactive SAM segmentation on frozen webcam frames."""

    # --------------------------------------------------------
    # Initialize components (camera, model, anomaly scorer)
    # --------------------------------------------------------
    camera = Camera()

    init_window(config)

    mode_spec = MODE_REGISTRY[config.app_mode]
    segmenter = create_model_from_spec(config=config, mode_spec=mode_spec)

    frame = None
    frozen_frame = None
    preview_frame = None
    last_infer_ms = 0.0
    debug_preview_frame = None
    freeze_mode_enabled = False

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
            freeze_mode_enabled = False
            cv2.destroyWindow(debug_window_name)

        # --------------------------------------------------------
        # Run inference and segment areas of the image
        # --------------------------------------------------------
        if key == ord('f') and frozen_frame is None:
            freeze_mode_enabled = True

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
                ranked_masks, nb_filtered_masks = rank_masks(config, masks)
                text_y = 25

                preview_frame = draw_masks(display, ranked_masks, text_y)

                if config.sam_debug_enabled and freeze_mode_enabled:
                    init_window(config, debug_mode=config.sam_debug_enabled)

                    debug_nb_masks =  nb_filtered_masks + len(ranked_masks)
                    ranked_and_filtered_masks = masks[:debug_nb_masks]
                    ranked_and_filtered_masks = list(
                        map(
                            lambda m: ndarray_to_mask_candidate(m),
                            ranked_and_filtered_masks)
                    )

                    debug_preview_frame = draw_masks(display, ranked_and_filtered_masks, text_y, draw_metadata=False)


        draw_text(display, 'Mode: SAM', 30)
        draw_text(display, f'Model: {segmenter.model_name}', 60)

        if frozen_frame is None:
            draw_text(display, 'Press f to freeze and segment', 100)
        else:
            draw_text(display, 'Frozen frame segmented', 100)
            draw_text(display, f'Inference: {last_infer_ms:.1f} ms', 130)

        show(config, display if frozen_frame is None else preview_frame)
        if config.sam_debug_enabled and freeze_mode_enabled:
            if debug_preview_frame is not None:
                show(
                    config,
                    display if frozen_frame is None else debug_preview_frame,
                    debug_mode=config.sam_debug_enabled
                )

    camera.release()
    cv2.destroyAllWindows()

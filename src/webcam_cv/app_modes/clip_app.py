import time
from typing import Optional

import cv2

from webcam_cv.camera import Camera
from webcam_cv.display import draw_text, show
from webcam_cv.models.factory import create_embedder
from webcam_cv.config import AppConfig
from webcam_cv.utils.image import is_image_unchanged


def run_clip_app(config: AppConfig) -> None:
    print(f'Running clip mode on device: {config.device}')

    camera = Camera()
    embedder = create_embedder(config)

    frame_index = 0
    last_infer_ms = 0.0
    prompt_scores: list[tuple[str, float]] = []
    best_prompt: Optional[str] = None
    best_score: Optional[str] = None
    previous_frame = None

    print('Controls:')
    print('  s = save current frame')
    print('  q = quit')

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame_index += 1
        display = frame.copy()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            filename = f'snapshot_{int(time.time())}.jpg'
            cv2.imwrite(filename, frame)
            print(f'Saved {filename}')

        if frame_index % config.frame_stride == 0:
            if previous_frame is not None:
                if is_image_unchanged(frame, previous_frame):
                    continue

            start = time.perf_counter()
            prompt_scores = embedder.score_prompts(frame, config.clip_prompts)
            last_infer_ms = (time.perf_counter() - start) * 1000

            best_prompt, best_score = prompt_scores[0]

            previous_frame = frame

        draw_text(display, 'Mode: clip', 30)
        draw_text(display, f'Model: {embedder.model_name}', 60)
        draw_text(display, f'Inference: {last_infer_ms:.1f} ms', 90)

        if best_prompt:
            draw_text(display, f'Best prompt: {best_prompt}', 120)
            draw_text(display, f'Confidence: {best_score:.3f}', 150)

            top_k = prompt_scores[:3]
            y = 190
            for prompt, score in top_k:
                draw_text(display, f'{prompt}: {score:.3f}', y, scale=0.6)
                y += 30


        show(config.window_name, display)

    camera.release()
    cv2.destroyAllWindows()
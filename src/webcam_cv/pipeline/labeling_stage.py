import time
import numpy as np

from webcam_cv.config import AppConfig
from webcam_cv.models.clip_embedder import ClipEmbedder


best_prompt_scores = list[tuple[str, float]]


def select_best_image_prompts(
        config: AppConfig,
        classifier: ClipEmbedder,
        frame_bgr: np.ndarray
) -> tuple[best_prompt_scores, float]:
    """Select best prompts based on image similarity."""
    start = time.perf_counter()
    prompt_scores = classifier.score_prompts(frame_bgr, config.clip_prompts)
    last_infer_ms = (time.perf_counter() - start) * 1000

    return prompt_scores, last_infer_ms
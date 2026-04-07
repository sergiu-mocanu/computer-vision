import time
from typing import Optional

import numpy as np

from webcam_cv.config import AppConfig
from webcam_cv.models.clip_embedder import ClipEmbedder
from webcam_cv.pipeline.sam.crop_utils import crop_frame_to_mask_bbox
from webcam_cv.pipeline.sam.mask_candidate import MaskCandidate

best_prompt_scores = list[tuple[str, float]]
mask_prompt_similarities = tuple[MaskCandidate, str, float]


def select_best_image_prompts(
        config: AppConfig,
        classifier: ClipEmbedder,
        frame_bgr: np.ndarray
) -> tuple[best_prompt_scores, float]:
    """Select best prompts based on image similarity."""
    start = time.perf_counter()
    prompt_scores = classifier.score_prompts(frame_bgr, config.clip_prompts)
    last_infer_ms = (time.perf_counter() - start) * 1000

    return prompt_scores[:config.clip_top_k_prompts], last_infer_ms


def select_best_mask_with_clip(
        classifier: ClipEmbedder,
        frame_bgr: np.ndarray,
        mask_candidates: list[MaskCandidate],
        prompts: list[str],
) -> tuple[list[mask_prompt_similarities], float]:
    """Select the best SAM candidate using CLIP prompt scoring.

    Each candidate mask is converted to a bounding-box crop, then scored
    against the provided prompts. The candidate with the highest prompt
    confidence is selected.
    """
    mask_candidates_copy = mask_candidates.copy()
    prompts_copy = prompts.copy()

    best_similarities: list[mask_prompt_similarities] = []

    start = time.perf_counter()

    while mask_candidates_copy:
        best_candidate: Optional[MaskCandidate] = None
        best_label: Optional[str] = None
        best_confidence: Optional[float] = None

        for candidate in mask_candidates_copy:
            crop = crop_frame_to_mask_bbox(frame_bgr, candidate.mask)

            if crop.size == 0:
                continue

            prompt_scores = classifier.score_prompts(crop, prompts_copy)
            if not prompt_scores:
                continue

            label, confidence = prompt_scores[0]

            if best_confidence is None or confidence > best_confidence:
                best_candidate = candidate
                best_label = label
                best_confidence = confidence

        if best_candidate is None or best_label is None or best_confidence is None:
            break

        best_similarities.append((best_candidate, best_label, best_confidence))
        mask_candidates_copy.remove(best_candidate)
        prompts_copy.remove(best_label)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    best_similarities.sort(key=lambda x: x[2], reverse=True)
    return best_similarities, elapsed_ms

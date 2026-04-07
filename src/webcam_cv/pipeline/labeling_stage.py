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
) -> tuple[MaskCandidate | None, str | None, float | None, float]:
   """Select the best SAM candidate using CLIP prompt scoring.

   Each candidate mask is converted to a bounding-box crop, then scored
   against the provided prompts. The candidate with the highest prompt
   confidence is selected.
   """
   best_candidate = None
   best_label = None
   best_confidence = None


   start = time.perf_counter()


   for candidate in mask_candidates:
       crop = crop_frame_to_mask_bbox(frame_bgr, candidate.mask)


       if crop.size == 0:
           continue


       prompt_scores = classifier.score_prompts(crop, prompts)
       if not prompt_scores:
           continue


       label, confidence = prompt_scores[0]


       if best_confidence is None or confidence > best_confidence:
           best_candidate = candidate
           best_label = label
           best_confidence = confidence


   elapsed_ms = (time.perf_counter() - start) * 1000.0
   return best_candidate, best_label, best_confidence, elapsed_ms

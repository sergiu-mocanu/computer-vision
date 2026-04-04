import time
import cv2
import numpy as np

from webcam_cv.config import AppConfig
from webcam_cv.models.sam_segmenter import SamSegmenter
from webcam_cv.pipeline.sam.mask_ranker import rank_masks, suppress_contained_masks


def generate_ranked_masks(config: AppConfig, segmenter: SamSegmenter, frame_bgr: np.ndarray) -> tuple[list, float]:
    """Run SAM on a frame, suppress identical or contained masks, and return the top-k ranked mask candidates."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    start = time.perf_counter()
    raw_masks = segmenter.generate_masks(frame_rgb)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    filtered_masks = suppress_contained_masks(raw_masks)

    ranked_masks = rank_masks(filtered_masks)
    return ranked_masks[:config.sam_top_k_masks], elapsed_ms
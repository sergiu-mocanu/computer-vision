import time
import numpy as np

from webcam_cv.models.base import BaseEmbedder
from webcam_cv.pipeline.dino.anomaly_scorer import AnomalyScorer


def score_frame_anomaly(
        detector: BaseEmbedder,
        scorer: AnomalyScorer,
        frame_bgr: np.ndarray
) -> tuple[float | None, bool, float]:
    """Run the anomaly stage on a frame."""

    start = time.perf_counter()
    embedding = detector.embed(frame_bgr)
    score = scorer.score(embedding)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    is_anomaly = False
    if score is not None:
        is_anomaly = scorer.is_anomaly(score)

    return score, is_anomaly, elapsed_ms
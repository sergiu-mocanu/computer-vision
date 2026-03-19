from collections import deque
import numpy as np
import torch
import torch.nn.functional as F


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine distance between two embedding vectors."""
    similarity = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    distance = 1.0 - similarity
    return distance


class AnomalyScorer:
    """Lightweight anomaly detector using cosine distance in embedding space.

    Learns a reference embedding from normal frames and scores new embeddings
    by measuring their deviation from this reference. Used by DINOv2 model.
    """

    def __init__(self, threshold: float, smoothing_window: int = 20) -> None:
        """Initialize the anomaly scorer."""
        self.threshold = threshold
        self.history = deque(maxlen=smoothing_window)
        self.reference_embedding = None


    def clear(self) -> None:
        """Reset the reference embedding and score history."""
        self.history.clear()
        self.reference_embedding = None


    def fit_reference(self, embeddings: list[torch.Tensor]) -> None:
        """Compute and store the normalized mean embedding representing the normal scene."""
        reference = torch.stack(embeddings, dim=0).mean(dim=0)
        self.reference_embedding = F.normalize(reference, dim=0)


    def score(self, embedding: torch.Tensor) -> float | None:
        """Compute a smoothed anomaly score for a new embedding."""
        if self.reference_embedding is None:
            return None

        score = cosine_distance(embedding, self.reference_embedding)
        self.history.append(score)
        return float(np.mean(self.history))


    def is_anomaly(self, score: float) -> bool:
        """Determine whether a score exceeds the anomaly threshold."""
        return score > self.threshold
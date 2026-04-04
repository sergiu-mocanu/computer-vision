import numpy as np
import torch
import torch.nn.functional as F


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine distance between two embedding vectors."""
    similarity = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    distance = 1.0 - similarity
    return distance


class AnomalyScorer:
    """
    Scores anomalies by comparing the current embedding to the reference
    embedding, then normalizing that distance relative to the distribution
    observed on the reference frames.

    Pipeline:
    1. Build reference mean embedding from normal samples
    2. Compute normal distance distribution around that mean
    3. Score new frames with a z-score
    4. Smooth z-score with EMA
    """

    def __init__(self, z_threshold: float, ema_alpha: float = 0.2, eps: float = 1e-6) -> None:
        """Initialize the anomaly scorer."""
        self.z_threshold = z_threshold
        self.ema_alpha = ema_alpha
        self.eps = eps

        self.reference_embedding: torch.Tensor | None = None
        self.reference_mean_distance: float | None = None
        self.reference_std_distance: float | None = None

        self.smoothed_score: float | None = None


    def clear(self) -> None:
        """Reset the internal state."""
        self.reference_embedding = None
        self.reference_mean_distance = None
        self.reference_std_distance = None
        self.smoothed_score = None


    def fit_reference(self, embeddings: list[torch.Tensor]) -> None:
        """
        Build the reference model from normal embeddings.

        Computes:
        - reference embedding (mean of normalized embeddings)
        - distribution of distances of normal samples to this reference
          (mean and standard deviation)

        Important:
        Reference frames should include natural variability of the scene
        (e.g., minor movements or slight lighting changes). If the reference
        set is too uniform, the estimated standard deviation may be near zero,
        making the detector overly sensitive.
        """
        # Compute the center of the normal embedding cloud
        reference = torch.stack(embeddings, dim=0).mean(dim=0)
        reference = F.normalize(reference, dim=0)
        self.reference_embedding = reference

        assert self.reference_embedding is not None
        # Measure how far each normal embedding is from the center
        distances = [
            cosine_distance(embedding, self.reference_embedding)
            for embedding in embeddings
        ]

        # Learn the normal distance distribution
        self.reference_mean_distance = float(np.mean(distances))
        self.reference_std_distance = float(np.std(distances))

        # Reset smoothing when reference changes
        self.smoothed_score = None


    def raw_distance(self, embedding: torch.Tensor) -> float | None:
        """Compute the cosine distance between the current embedding and the reference embedding."""
        if self.reference_embedding is None:
            return None

        return cosine_distance(embedding, self.reference_embedding)


    def raw_z_score(self, embedding: torch.Tensor) -> float | None:
        if (
                self.reference_embedding is None
                or self.reference_mean_distance is None
                or self.reference_std_distance is None
        ):
            return None

        raw_distance = self.raw_distance(embedding)
        assert raw_distance is not None

        # Protect against near-zero std when the reference scene is extremely stable
        denom = max(self.reference_std_distance, self.eps)
        z_score = (raw_distance - self.reference_mean_distance) / denom
        return float(z_score)


    def score(self, embedding: torch.Tensor) -> float | None:
        """Returns the EMA-smoothed z-score."""
        raw_z = self.raw_z_score(embedding)
        if raw_z is None:
            return None

        if self.smoothed_score is None:
            self.smoothed_score = raw_z
        else:
            self.smoothed_score = (
                self.ema_alpha * raw_z
                + (1 - self.ema_alpha) * self.smoothed_score
            )

        return self.smoothed_score


    def is_anomaly(self, score: float) -> bool:
        """Determine whether a score exceeds the anomaly threshold."""
        return score > self.z_threshold

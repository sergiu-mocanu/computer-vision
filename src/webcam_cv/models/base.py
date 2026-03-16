from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseEmbedder(ABC):
    """Abstract interface for models that convert images into feature embeddings."""

    @abstractmethod
    def embed(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Compute an embedding vector from an input frame."""
        pass

from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseEmbedder(ABC):
    """Abstract interface for models that convert images into feature embeddings."""
    model_name: str = None

    @abstractmethod
    def embed(self, frame_bgr: np.ndarray, reduce_img_size: bool) -> torch.Tensor:
        """Compute an embedding vector from an input frame."""
        pass

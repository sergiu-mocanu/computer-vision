from abc import ABC, abstractmethod
import torch
import numpy as np

from webcam_cv.utils.image import reduce_res


def prepare_frame(frame: np.ndarray, reduce_img_size: bool = True) -> np.ndarray:
    if reduce_img_size:
        return reduce_res(frame)
    return frame


class BaseEmbedder(ABC):
    """Abstract interface for models that convert images into feature embeddings."""
    model_name: str = None

    @abstractmethod
    def embed(self, frame_bgr: np.ndarray, size: str = None, reduce_img_size: bool = True) -> torch.Tensor:
        """Compute an embedding vector from an input frame."""
        pass

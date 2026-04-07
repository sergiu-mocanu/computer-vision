from abc import ABC, abstractmethod
import torch
import numpy as np


from webcam_cv.image import reduce_res


def prepare_frame(frame: np.ndarray, reduce_img_size: bool = True, max_width: int = 384) -> np.ndarray:
    if reduce_img_size:
        return reduce_res(frame, max_width=max_width)
    return frame


class BaseEmbedder(ABC):
    model_name: str

    @abstractmethod
    def embed(self, frame_bgr: np.ndarray, size: str = None, reduce_img_size: bool = True) -> torch.Tensor:
        """Compute an embedding vector from an input frame."""
        pass
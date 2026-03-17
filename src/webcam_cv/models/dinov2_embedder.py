import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from webcam_cv.models.base import BaseEmbedder
from webcam_cv.utils.image import bgr_2_pil, reduce_res


DINOv2_MODEL_NAMES = {
    'small': 'facebook/dinov2-small',
    'base': 'facebook/dinov2-base',
    'large': 'facebook/dinov2-large',
    'giant': 'facebook/dinov2-giant',
}


class DinoV2Embedder(BaseEmbedder):
    """Vision transformer embedder using the DINOv2 foundation model."""

    MODEL_TYPE = 'dinov2'
    DEFAULT_SIZE = 'base'

    def __init__(self, device: str, size: str | None = None, use_fast: bool = False):
        """Load the DINOv2 model and associated preprocessing pipeline."""
        self.device = device
        self.size = size or self.DEFAULT_SIZE

        if self.size not in DINOv2_MODEL_NAMES:
            raise ValueError(
                f'Unknown DINOv2 model size: {self.size}. '
                f'Expected one of {DINOv2_MODEL_NAMES}.'
            )

        self.model_name = DINOv2_MODEL_NAMES[self.size]

        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=use_fast)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.model.eval()


    @torch.inference_mode()
    def embed(self, frame_bgr: np.ndarray, reduce_img_size: bool = True) -> torch.Tensor:
        """Extract a normalized global embedding from a frame.

        The frame is resized, preprocessed with the model image processor,
        and passed through the transformer. The CLS token is used as the
        global image representation.
        """
        if reduce_img_size:
            frame = reduce_res(frame_bgr)
        else:
            frame = frame_bgr

        image = bgr_2_pil(frame)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        # Run the model on the image
        outputs = self.model(pixel_values=pixel_values)

        cls_token = outputs.last_hidden_state[:, 0, :]

        # Normalize the embedding in order to obtain the direction of the feature vector
        embedding = F.normalize(cls_token, dim=-1)

        # Remove batch dimension and move back to CPU
        return embedding.squeeze(0).detach().cpu()
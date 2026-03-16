import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from webcam_cv.models.base import BaseEmbedder
from webcam_cv.utils.image import bgr_to_pil, reduce_res


class DinoV2Embedder(BaseEmbedder):
    """Vision transformer embedder using the DINOv2 foundation model."""

    def __init__(self, model_name: str, device: str, use_fast: bool = False):
        """Load the DINOv2 model and associated preprocessing pipeline."""
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=use_fast)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Extract a normalized global embedding from a frame.

        The frame is resized, preprocessed with the model image processor,
        and passed through the transformer. The CLS token is used as the
        global image representation.
        """
        frame_small = reduce_res(frame_bgr)
        image = bgr_to_pil(frame_small)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        # Run the model on the image
        outputs = self.model(pixel_values=pixel_values)

        cls_token = outputs.last_hidden_state[:, 0, :]

        # Normalize the embedding in order to obtain the direction of the feature vector
        embedding = F.normalize(cls_token, dim=-1)

        # Remove batch dimension and move back to CPU
        return embedding.squeeze(0).detach().cpu()
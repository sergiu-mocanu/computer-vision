import numpy as np
import torch.nn.functional as F

import cv2
import torch
from transformers import AutoImageProcessor, AutoModel

from webcam_cv.config import AppConfig
from webcam_cv.camera import Camera
from webcam_cv.display import draw_text, show
from webcam_cv.models.base import BaseEmbedder, prepare_frame
from webcam_cv.utils.image import bgr_2_pil


DINOv2_MODEL_NAMES = {
    'small': 'facebook/dinov2-small',
    'base': 'facebook/dinov2-base',
    'large': 'facebook/dinov2-large',
    'giant': 'facebook/dinov2-giant',
}


class DinoV2Embedder(BaseEmbedder):
    """Vision transformer embedder using the DINOv2 foundation model."""
    MODEL_TYPE: str = 'dinov2'
    DEFAULT_SIZE: str = 'base'
    AVAILABLE_SIZES: list[str] = tuple(DINOv2_MODEL_NAMES.keys())

    def __init__(self, device: str, size: str | None = None, use_fast: bool = False):
        self.device = device
        self.size = size or self.DEFAULT_SIZE

        if self.size not in DINOv2_MODEL_NAMES:
            raise ValueError(
                f'Unknown DINOv2 model size: {self.size}. '
                f'Expected one of {self.AVAILABLE_SIZES}.'
            )

        self.model_name = DINOv2_MODEL_NAMES[self.size]

        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=use_fast)
        self.model = AutoModel.from_pretrained(self.model_name).to(device)
        self.model.eval()


    @torch.inference_mode()
    def embed(self, frame_bgr: np.ndarray, reduce_img_size: bool = True) -> torch.Tensor:
        frame = prepare_frame(frame_bgr, reduce_img_size)

        image = bgr_2_pil(frame)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        outputs = self.model(pixel_values=pixel_values)

        cls_token = outputs.last_hidden_state[:, 0, :]

        embedding = F.normalize(cls_token, dim=-1)

        return embedding.squeeze(0).detach().cpu()


    def collect_normal_frames(self, camera: Camera, config: AppConfig) -> list[torch.Tensor]:
        """Collect normal frames as a baseline for the anomaly measurement."""
        embeddings = []
        collected = 0
        frames = 0

        while collected < config.normal_frames_target:
            ok_ref, ref_frame = camera.read(config)
            if not ok_ref:
                break

            if frames % config.reference_frame_stride == 0:
                embeddings.append(self.embed(ref_frame))
                collected += 1

            frames += 1

            ref_display = ref_frame.copy()
            draw_text(
                ref_display,
                f'Recording normal frames: {collected}/{config.normal_frames_target}',
                30
            )
            show(config, ref_display)
            cv2.waitKey(1)

        return embeddings

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from webcam_cv.models.base import BaseEmbedder, prepare_frame
from webcam_cv.image import bgr_2_pil

CLIP_MODEL_NAMES = {
    'base': 'openai/clip-vit-base-patch32',
    'large': 'openai/clip-vit-large-patch14',
}


class ClipEmbedder(BaseEmbedder):
    MODEL_TYPE: str = 'clip'
    DEFAULT_SIZE: str = 'base'
    AVAILABLE_SIZES: list[str] = list(CLIP_MODEL_NAMES.keys())

    def __init__(self, device: str, size: str | None = None, use_fast: bool = False) -> None:
        self.device = device
        self.size = size or self.DEFAULT_SIZE

        if self.size not in CLIP_MODEL_NAMES:
            raise ValueError(
                f'Unknown CLIP size: {self.size}\n'
                f'Expected one of: {self.AVAILABLE_SIZES}'
            )

        self.model_name = CLIP_MODEL_NAMES[self.size]
        self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=use_fast)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()


    @torch.inference_mode()
    def embed(self, frame_bgr: np.ndarray, reduce_img_size: bool = True) -> torch.Tensor:
        frame: np.ndarray = prepare_frame(frame_bgr, reduce_img_size)

        image = bgr_2_pil(frame)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        image_features = self.model.get_image_features(pixel_values=pixel_values)
        image_features = F.normalize(image_features, dim=-1)

        return image_features.squeeze(0).detach().cpu()


    @torch.inference_mode()
    def score_prompts(self, frame_bgr: np.ndarray, prompts: list[str],
                      reduce_img_size: bool = True) -> list[tuple[str, float]]:
        """Score text prompts against one image and return them sorted by match probability."""
        frame = prepare_frame(frame_bgr, reduce_img_size)

        image = bgr_2_pil(frame)

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Move all input tensors to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run CLIP on the text-image pairs
        outputs = self.model(**inputs)

        # Convert image-to-text logits into probabilities over the prompt list
        probs = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu().tolist()

        # Pair each prompt with its probability and sort from best to worst match
        pairs = list(zip(prompts, probs, strict=False))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

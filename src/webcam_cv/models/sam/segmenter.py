import numpy as np
from transformers import pipeline

from webcam_cv.models.base import prepare_frame
from webcam_cv.utils.image import bgr_2_pil

SAM_MODEL_NAMES = {
    'base': 'facebook/sam-vit-base',
    'large': 'facebook/sam-vit-large',
    'huge': 'facebook/sam-vit-huge',
}


class SamSegmenter:
    """Run SAM automatic mask generation on a single image."""

    MODEL_TYPE = 'sam'
    DEFAULT_SIZE = 'base'
    AVAILABLE_SIZES: list[str] = tuple(SAM_MODEL_NAMES.keys())

    def __init__(self, device: str, size: str | None = None):
        self.device = device
        self.size = size or self.DEFAULT_SIZE

        if self.size not in SAM_MODEL_NAMES:
            raise ValueError(
                f'Unknown SAM size: {self.size}. '
                f'Expected one of: {list(SAM_MODEL_NAMES)}'
            )

        self.model_name = SAM_MODEL_NAMES[self.size]

        self.generator = pipeline(
            'mask-generation',
            model=self.model_name,
            device=device,
        )


    def generate_masks(self, frame_bgr: np.ndarray, reduce_img_size: bool = True) -> list[dict]:
        """Generate segmentation masks for the input frame."""
        frame = prepare_frame(frame_bgr, reduce_img_size, max_width=1024)
        image = bgr_2_pil(frame)
        outputs = self.generator(image)
        return outputs['masks']


from webcam_cv.config import AppConfig
from webcam_cv.models.base import BaseEmbedder
from webcam_cv.models.dinov2_embedder import DinoV2Embedder


def create_embedder(config: AppConfig) -> BaseEmbedder:
    if config.model_type == 'dinov2':
        return DinoV2Embedder(
            device=config.device,
            size=config.model_size,
        )

    raise ValueError(f'Unsupported model_type: {config.model_type}')

from webcam_cv.config import AppConfig
from webcam_cv.models.registry import MODEL_REGISTRY
from webcam_cv.models.base import BaseEmbedder


def create_embedder(config: AppConfig) -> BaseEmbedder:
    models = list_models()

    if config.model_type not in models:
        raise ValueError(f'Unsupported model_type: {config.model_type}'
                         f'Available models: {", ".join(models)}')

    model_cls = MODEL_REGISTRY[config.model_type]

    return model_cls(
        device=config.device,
        size=config.model_size
    )


def list_models():
    return MODEL_REGISTRY.keys()

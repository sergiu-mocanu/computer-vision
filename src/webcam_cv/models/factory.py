from webcam_cv.config import AppConfig
from webcam_cv.models.base import BaseEmbedder


def create_model_for_mode(mode_spec: dict, role: str = 'primary') -> BaseEmbedder:
    config = AppConfig()

    if role not in mode_spec['models']:
        raise ValueError(f"Mode '{config.app_mode}' does not define model role '{role}'")

    model_spec = mode_spec['models'][role]

    model_cls = model_spec['model_cls']
    model_size = model_spec.get('size')

    if config.model_size is not None:
        model_size = config.model_size

    return model_cls(device=config.device, size=model_size)
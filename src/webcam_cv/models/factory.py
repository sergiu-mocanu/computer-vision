from webcam_cv.config import AppConfig
from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.models.base import BaseEmbedder


def create_model_from_spec(mode_spec: dict, role: str = 'primary') -> BaseEmbedder:
    config = AppConfig()

    app_mode_models = MODE_REGISTRY[config.app_mode]['models']

    if role not in app_mode_models:
        print(app_mode_models)
        raise ValueError(f"Mode '{config.app_mode}' does not define model role '{role}'")

    model_spec = app_mode_models[role]

    model_cls = model_spec['model_cls']
    model_size = model_spec.get('size')

    if config.model_size is not None:
        model_size = config.model_size

    return model_cls(device=config.device, size=model_size)
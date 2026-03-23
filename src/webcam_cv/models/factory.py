from typing import Union

from webcam_cv.config import AppConfig
from webcam_cv.models.base import BaseEmbedder
from webcam_cv.models.sam_segmenter import SamSegmenter


def create_model_from_spec(config: AppConfig, mode_spec: dict,
                           role: str = 'primary') -> Union[BaseEmbedder, SamSegmenter]:
    """Create CV model from app-mode specification."""
    app_mode_models = mode_spec['models']

    if role not in app_mode_models:
        print(app_mode_models)
        raise ValueError(f"Mode '{config.app_mode}' does not define model role '{role}'\n"
                         f"Available modes: {list(app_mode_models)}")

    model_spec = app_mode_models[role]

    model_cls = model_spec['model_cls']
    model_size = model_spec.get('size')

    if config.model_size is not None:
        model_size = config.model_size

    return model_cls(device=config.device, size=model_size)
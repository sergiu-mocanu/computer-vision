from webcam_cv.config import AppConfig
from webcam_cv.app_modes.anomaly_app import run_anomaly_app
from webcam_cv.app_modes.clip_app import run_clip_app


def run() -> None:
    config = AppConfig()

    if config.model_type == 'dinov2':
        run_anomaly_app(config)
        return

    if config.model_type == 'clip':
        run_clip_app(config)
        return
from webcam_cv.config import AppConfig
from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.app_modes.anomaly_app import run_anomaly_app
from webcam_cv.app_modes.clip_app import run_clip_app
from webcam_cv.app_modes.pipeline_app import run_pipeline_app


RUNNERS = {
    'anomaly': run_anomaly_app,
    'labeling': run_clip_app,
    'pipeline': run_pipeline_app,
}


def run() -> None:
    config = AppConfig()

    if config.app_mode not in MODE_REGISTRY:
        raise ValueError(f'Invalid app_mode: {config.app_mode}'
                         f'Available modes: {list(RUNNERS.keys())}')

    runner = RUNNERS[config.app_mode]
    runner(config)

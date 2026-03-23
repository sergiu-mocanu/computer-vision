from webcam_cv.config import AppConfig
from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.app_modes.anomaly_app import run_anomaly_app
from webcam_cv.app_modes.labeling_app import run_labelling_app
from webcam_cv.app_modes.pipeline_app import run_pipeline_app
from webcam_cv.app_modes.segmentation_app import run_segmentation_app


RUNNERS = {
    'anomaly': run_anomaly_app,
    'labeling': run_labelling_app,
    'pipeline': run_pipeline_app,
    'segmentation': run_segmentation_app
}


def run() -> None:
    config = AppConfig()

    if config.app_mode not in MODE_REGISTRY:
        raise ValueError(f'Invalid app_mode: {config.app_mode}\n'
                         f'Available modes: {list(RUNNERS.keys())}')

    runner = RUNNERS[config.app_mode]
    runner(config)

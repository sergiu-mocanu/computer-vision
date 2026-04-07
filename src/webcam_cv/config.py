from dataclasses import dataclass, field
from pathlib import Path
import os
import torch


@dataclass
class AppConfig:
    app_mode: str = 'segmented_pipeline'

    model_size: str = None

    reference_frame_stride: int = 2
    inference_frame_stride: int = 2

    normal_frames_target: int = 50

    anomaly_z_threshold: float = 3.5
    ema_alpha: float = 0.2

    window_name: str = 'Webcam CV Prototype'
    window_width: int = 900
    window_height: int = 720

    gamma: float = 1.2
    contrast: float = 0.95
    brightness: int = -3

    saved_photos_folder: str = os.path.join(str(Path.home()), 'com-vis-photos')

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name: str = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None

    clip_prompts: list[str] = field(default_factory=lambda: [
        'a hand in front of the camera',
        'a person in front of the camera',
        'a chair',
        'a phone',
        'a door'
    ])
    clip_top_k_prompts: int = 3

    sam_top_k_masks: int = 4

from dataclasses import dataclass, field
from pathlib import Path
import os
import torch


@dataclass
class AppConfig:
    model_type: str = 'clip'
    model_size: str = None

    normal_frames_target: int = 50
    anomaly_threshold: float = 0.12
    frame_stride: int = 2

    window_name: str = 'Webcam CV Prototype'
    saved_photos_folder: str = os.path.join(str(Path.home()), 'com-vis-photos')

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name: str = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None

    clip_prompts: list[str] = field(default_factory=lambda: [
        'a hand in front of the camera',
        'a person in front of the camera',
        'an empty chair',
        'a mirror',
        'a car'
    ])
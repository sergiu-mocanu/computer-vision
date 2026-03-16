from dataclasses import dataclass
from pathlib import Path
import os
import torch

@dataclass
class AppConfig:
    model_name: str = 'facebook/dinov2-base'
    normal_frames_target: int = 50
    anomaly_threshold: float = 0.12
    frame_stride: int = 2
    window_name: str = 'DINOv2 Webcam Anomaly Prototype'
    saved_photos_folder: str = os.path.join(str(Path.home()), 'com-vis-photos')
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name: str = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
from pathlib import Path
import logging
from typing import cast

from webcam_cv.config import AppConfig
from webcam_cv.camera import Camera

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from transformers import CLIPModel, CLIPProcessor


# Suppress external warnings
logging.getLogger('torch.onnx').setLevel(logging.ERROR)


MODEL_NAME = 'openai/clip-vit-base-patch32'
OUTPUT_PATH = Path('exports/clip_image_encoder.onnx')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ClipImageEncoder(torch.nn.Module):
    """Wrap CLIP image feature extraction for ONNX export."""

    def __init__(self, model: CLIPModel):
        """Initialize the wrapper with a Hugging Face CLIP model."""
        super().__init__()
        self.model = model


    # Note: FloatTensor instead of Tensor?
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Return normalized CLIP image features."""
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        return F.normalize(image_features, dim=-1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D numpy vectors."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    """Export CLIP image encoder to ONNX and validate against PyTorch output."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = cast(CLIPModel, CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE))
    model.eval()

    wrapper = ClipImageEncoder(model).to(DEVICE)
    wrapper.eval()


    config = AppConfig()
    camera = Camera()

    ok, frame = camera.read(config)
    if not ok:
        exit(1)

    image_size = 384
    test_images = [
        np.zeros((image_size, image_size, 3), dtype=np.uint8),
        np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8),
        frame,
    ]

    for image in test_images:
        inputs = processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.inference_mode():
            torch_output = wrapper(pixel_values).detach().cpu().numpy()

        torch.onnx.export(
            wrapper,
            pixel_values,
            OUTPUT_PATH,
            input_names=['pixel_values'],
            output_names=['image_features'],
            opset_version=18,
            dynamo=True,
        )

        session = ort.InferenceSession(
            str(OUTPUT_PATH),
            providers=['CPUExecutionProvider']
        )

        onnx_output = session.run(
            ['image_features'],
            {'pixel_values': pixel_values.detach().cpu().numpy()}
        )[0]

        similarity = cosine_similarity(torch_output, onnx_output)
        max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))

        print(f'Exported ONNX model to: {OUTPUT_PATH}')
        print(f'PyTorch output shape: {torch_output.shape}')
        print(f'ONNX output shape: {onnx_output.shape}')
        print(f'Cosine similarity: {similarity:.6f}')
        print(f'Max absolute difference: {max_abs_diff:.6f}')
        print('_' * 80)


if __name__ == '__main__':
    main()











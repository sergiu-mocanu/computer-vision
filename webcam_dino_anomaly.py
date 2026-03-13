import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = 'facebook/dinov2-base'
NORMAL_FRAMES_TARGET = 50
ANOMALY_THRESHOLD = 0.12
FRAME_STRIDE = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SAVE_PATH = './saved_images/'

print(torch.cuda.get_device_name())

def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Converts a BGR image to PIL Image (red, green, blue)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


class DinoEmbedder:
    """Load the vision model and convert images into embeddings."""

    def __init__(self, model_name: str, device: str):
        self.device = device

        # The processor handles image preparation:
        # resizing, normalization, tensor conversion
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()


    @torch.inference_mode()
    def embed(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Convert one webcam frame into an embedding."""
        image = bgr_to_pil(frame_bgr)

        inputs = self.processor(images=image, return_tensors="pt")

        # 'pixel_values' is the tensor version of the image.
        # Shape is typically:
        #   [batch_size, color_channels, height, width]
        pixel_values = inputs['pixel_values'].to(self.device)


        # Run the model on the image
        outputs = self.model(pixel_values=pixel_values)

        cls_token = outputs.last_hidden_state[:, 0, :]

        # Normalize the embedding in order to obtain the direction of the feature vector
        embedding = F.normalize(cls_token, dim=-1)

        # Remove batch dimension and move back to CPU
        return embedding.squeeze(0).detach().cpu()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Measure the distance between two embeddings."""
    similarity = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    distance = 1.0 - similarity
    return distance


def draw_text(frame: np.ndarray,text: str, y: int, scale: float = 0.7) -> None:
    """Draw status text on top of the webcam image."""
    cv2.putText(frame,
                text=text,
                org=(20, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=scale,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA
                )

def main() -> None:
    print(f'Using device: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')

    embedder = DinoEmbedder(MODEL_NAME, DEVICE)

    # Open webcam 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam')

    print('Controls:')
    print('  r = record normal reference frames')
    print('  c = clear reference')
    print('  s = save current frame')
    print('  q = quit')

    # Store embeddings of normal (baseline) frames
    reference_embeddings: list[torch.Tensor] = []

    # Final average representation of "normal"
    reference_embedding: Optional[torch.Tensor] = None

    # Small smoothing buffer for recent anomaly scores.
    # This reduces flickering in real-time predictions.
    score_history = deque(maxlen=20)

    frame_index = 0
    last_infer_ms = 0.0
    current_score = None

    while True:
        # Read one frame from webcam
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1

        display = frame.copy()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c'):
            reference_embeddings.clear()
            reference_embedding = None
            score_history.clear()
            print('Reference cleared')

        if key == ord('s'):
            filename = f'snapshot_{int(time.time())}.jpg'
            filepath = os.path.join(IMAGE_SAVE_PATH, filename)
            cv2.imwrite(filepath, frame)
            print(f'Saved {filepath}')

        # REFERENCE COLLECTION PHASE
        if key == ord('r'):
            print(f'Recording {NORMAL_FRAMES_TARGET} normal frames...')
            collected = 0
            temp_embeddings: list[torch.Tensor] = []

            while collected < NORMAL_FRAMES_TARGET:
                ok_ref, ref_frame = cap.read()
                if not ok_ref:
                    break

                # Process every 'FRAME_STRIDE' in order to reduce GPU load
                if collected % FRAME_STRIDE == 0:
                    emb = embedder.embed(ref_frame)
                    temp_embeddings.append(emb)

                collected += 1

                ref_display = ref_frame.copy()
                draw_text(
                    frame=ref_display,
                    text=f'Recording normal frames: {collected}/{NORMAL_FRAMES_TARGET}',
                    y=30
                )
                cv2.imshow('DINOv2 Webcam Anomaly Prototype', ref_display)
                cv2.waitKey(1)

            if temp_embeddings:
                # Average all "normal" embeddings to produce a single reference vector representing the normal scene
                reference_embedding = torch.stack(temp_embeddings, dim=0).mean(dim=0)
                reference_embedding = F.normalize(reference_embedding, dim=0)
                reference_embeddings = temp_embeddings
                print('Reference embeddings created')

        # LIVE ANOMALY SCORING
        if reference_embedding is not None and frame_index % FRAME_STRIDE == 0:
            start = time.perf_counter()

            # Extract visual features from current frame
            emb = embedder.embed(frame)

            # Measure distance between learned normal state and current frame
            current_score = cosine_distance(reference_embedding, emb)

            # Smooth the score over several recent frames
            score_history.append(current_score)

            last_infer_ms = (time.perf_counter() - start) * 1000

        smoothed_score = None
        if score_history:
            smoothed_score = float(np.mean(score_history))

        draw_text(display, f'Device: {DEVICE}', y=30)
        draw_text(display, f'MODEL: {MODEL_NAME}', y=60)

        if reference_embedding is None:
            draw_text(display, 'Status: no reference yet (press r)', y=100)
        else:
            draw_text(display, 'Reference ready', y=100)

            if smoothed_score is not None:
                status = 'ANOMALY' if smoothed_score > ANOMALY_THRESHOLD else 'NORMAL'
                draw_text(display, f'Stats: {status}', y=140)
                draw_text(display, f'Anomaly score: {smoothed_score:.4f}', y=170)
                draw_text(display, f'Threshold: {ANOMALY_THRESHOLD:.4f}', y=200)
                draw_text(display, f'Inference time: {last_infer_ms:.1f} ms', y=250)

        cv2.imshow('DINOv2 Webcam Anomaly Prototype', display)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



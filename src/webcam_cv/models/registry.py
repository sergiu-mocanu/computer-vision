from webcam_cv.models.dinov2_embedder import DinoV2Embedder
from webcam_cv.models.clip_embedder import ClipEmbedder

MODEL_REGISTRY = {
    'dinov2': DinoV2Embedder,
    'clip': ClipEmbedder
}
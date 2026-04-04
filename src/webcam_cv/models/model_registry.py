from webcam_cv.models.dinov2_embedder import DinoV2Embedder
from webcam_cv.models.clip_embedder import ClipEmbedder
from webcam_cv.models.sam_segmenter import SamSegmenter

MODEL_REGISTRY = {
    'dinov2': DinoV2Embedder,
    'clip': ClipEmbedder,
    'sam': SamSegmenter
}
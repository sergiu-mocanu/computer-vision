from webcam_cv.models.dinov2_embedder import DinoV2Embedder
from webcam_cv.models.clip_embedder import ClipEmbedder
from webcam_cv.models.sam_segmenter import SamSegmenter

MODE_REGISTRY = {
    'anomaly': {
        'models': {
            'primary': {'model_cls': DinoV2Embedder},
        }
    },
    'labeling': {
        'models': {
            'primary': {'model_cls': ClipEmbedder},
        }
    },
    'pipeline': {
        'models': {
            'detector': {'model_cls': DinoV2Embedder},
            'classifier': {'model_cls': ClipEmbedder, 'size': 'large'}
        }
    },
    'segmentation': {
        'models': {
            'primary': {'model_cls': SamSegmenter}
        }
    }
}

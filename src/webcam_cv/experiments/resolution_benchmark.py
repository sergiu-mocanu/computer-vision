from typing import List, Tuple

import cv2
import time

import numpy as np
import torch.nn.functional as F

from webcam_cv.app_modes.mode_registry import MODE_REGISTRY
from webcam_cv.config import AppConfig
from webcam_cv.display import draw_text, show
from webcam_cv.image import reduce_res
from webcam_cv.models.base import BaseEmbedder
from webcam_cv.models.factory import create_model_from_spec

image_sizes = [1080, 768, 640, 512, 384, 256, 224]

exp_res_type = dict[float, list[float]]


def run_benchmark(embedder: BaseEmbedder, nb_frames: int = 20,
                  delay_ms: float = 100, nb_runs:int = 5) -> Tuple[exp_res_type, exp_res_type]:
    """Estimate the optimal input resolution for a vision embedding model by balancing
    computational cost and embedding fidelity.

    This provides a practical trade-off between speed and representation quality
    for real-time or resource-constrained pipelines.

    Note: during the capture time, it is important to add a natural change of scenery (e.g., hand movement).
    """
    print(f'Computing embedding time for CV model: {embedder.model_name}\n')

    # Capture multiple frames with a little spacing between them
    cap = cv2.VideoCapture(0)
    frames: List[np.ndarray] = []

    for current_frame in range(nb_frames):
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame.copy())

        frame_display = frame.copy()
        draw_text(
            frame_display,
            f'Capturing frame: {current_frame + 1}/{nb_frames}',
            30
        )
        show(AppConfig(), frame_display)

        cv2.waitKey(delay_ms)

    cap.release()

    if not frames:
        raise RuntimeError('Could not capture frames')

    baseline_size = image_sizes[0]
    times = {s: [] for s in image_sizes}
    sims = {s: [] for s in image_sizes if s != baseline_size}

    for frame in frames:
        resized_versions = {s: reduce_res(frame, s) for s in image_sizes}

        # Reference embedding for this frame
        base = embedder.embed(resized_versions[baseline_size], reduce_img_size=False)

        # Run embedding simulations: measure time and embedding similarity
        for s in image_sizes:
            for _ in range(nb_runs):
                start = time.perf_counter()
                emb = embedder.embed(resized_versions[s], reduce_img_size=False)
                elapsed = time.perf_counter() - start
                times[s].append(elapsed)

                if s != baseline_size:
                    sim = F.cosine_similarity(
                        base.unsqueeze(0),
                        emb.unsqueeze(0),
                    ).item()
                    sims[s].append(sim)


    print('Average results across frames:\n')
    for s in image_sizes:
        avg_time_ms = sum(times[s]) / len(times[s]) * 1000
        if s == baseline_size:
            print(f'max_dim={s:4d} | similarity=1.0000 | avg_time={avg_time_ms:.2f} ms')
        else:
            avg_sim = sum(sims[s]) / len(sims[s])
            print(f'max_dim={s:4d} | similarity={avg_sim:.4f} | avg_time={avg_time_ms:.2f} ms')

    return times, sims


def compute_optimal_resolution(list_time: exp_res_type, list_sim: exp_res_type, threshold: float = 0.995) -> None:
    """Compute the optimal input resolution based on compute time and fidelity loss.

    Accepted distance threshold:
        >= 0.995 for a conservative choice
        >= 0.99 for a more speed-oriented choice
    """
    base_size = image_sizes[0]
    base_time = sum(list_time[base_size]) / len(list_time[base_size])

    best_size = None
    best_time = float('inf')

    print('\nResolution ranking:\n')
    for s in image_sizes[1:]:
        avg_time = sum(list_time[s]) / len(list_time[s])
        avg_sim = sum(list_sim[s]) / len(list_sim[s])

        time_gain = (base_time - avg_time) / base_time
        fidelity_loss = 1.0 - avg_sim
        ratio_score = time_gain / max(fidelity_loss, 1e-6) # add epsilon to account for low sim value

        print(
            f'max_dim={s:4d} | '
            f'sim={avg_sim:.4f} | '
            f'time={avg_time * 1000:.2f} ms | '
            f'time_gain={time_gain:.2%} | '
            f'fid_loss={fidelity_loss:.4f} | '
            f'score={ratio_score:.2f}'
        )

        if avg_sim >= threshold and avg_time < best_time:
            best_time = avg_time
            best_size = s

    print(f'\nBest size for similarity threshold {threshold}: {best_size}\n')


def run_and_compute(app_mode: str, model_size: str = None):
    config = AppConfig(app_mode=app_mode, model_size=model_size)
    mode_spec = MODE_REGISTRY[app_mode]

    embedder = create_model_from_spec(config, mode_spec)
    list_times, list_sims = run_benchmark(embedder)
    compute_optimal_resolution(list_times, list_sims)


run_and_compute('anomaly')

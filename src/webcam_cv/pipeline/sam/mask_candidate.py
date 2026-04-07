from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(eq=False)
class MaskCandidate:
    """Container for a SAM-generated mask and its ranking metadata."""
    mask: np.ndarray
    area_ratio: float
    mask_center: Tuple[float, float]
    center_distance: float
    touches_border: bool
    score: float = None

"""Reference-v2 MAPLES FOV engineering candidate; not a frozen training mask."""
import numpy as np
from scipy import ndimage

RULE = {
    "version": "maples_fov_reference_v2_candidate",
    "input": "native RGB uint8, before normalization",
    "threshold": "max(R,G,B) > 10",
    "foreground_connectivity": 8,
    "component": "largest",
    "equal_area_tie": "first component in row-major scan order",
    "fill_holes_background_connectivity": 4,
    "roi_intersection": False,
    "circle_assumption": False,
    "uses_labels": False,
    "review_status": "engineering_candidate_requires_human_review",
}


def candidate_fov(rgb):
    """Return native-size bool FOV using only raw RGB pixels."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) == 0:
        raise ValueError("raw_rgb: expected nonempty HWC RGB uint8")
    foreground = rgb.max(axis=2) > 10
    components, count = ndimage.label(foreground, structure=np.ones((3, 3), bool))
    if count == 0:
        raise ValueError("empty_fov: no max-RGB > 10 component; no fallback mask")
    areas = np.bincount(components.ravel())
    areas[0] = 0
    # SciPy numbers components by row-major scan; argmax returns the first tie.
    selected = components == int(np.argmax(areas))
    filled = ndimage.binary_fill_holes(
        selected, structure=ndimage.generate_binary_structure(2, 1))
    return np.ascontiguousarray(filled, dtype=bool)

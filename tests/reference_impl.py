"""Pre-optimization implementations kept as a correctness oracle.

The speed work on ``VoronoiPainting`` is meant to be behaviour preserving: the
same genome must produce exactly the same pixels and exactly the same fitness as
the original implementation. These functions are literal copies of that original
code - allocating fresh arrays, carrying the target image around, indexing with
fancy indexing - so ``tests/test_equivalence.py`` can assert the optimized code
matches them byte for byte.

Do not "clean up" or speed up anything in this module. Its whole value is being
an independent, unoptimized second opinion.
"""

import cv2
import numpy as np


def reference_render(painting, scale=1) -> np.ndarray:
    """The original ``VoronoiPainting._render_array``."""
    w = painting.get_img_width * scale
    h = painting.get_img_height * scale
    coords = np.array([p.coordinates for p in painting.points], dtype=np.int32) * scale
    xs = np.clip(coords[:, 0], 0, w - 1)
    ys = np.clip(coords[:, 1], 0, h - 1)
    binary = np.full((h, w), 255, dtype=np.uint8)
    binary[ys, xs] = 0
    _, labels = cv2.distanceTransformWithLabels(
        binary, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    colors = np.array([p.color[:3] for p in painting.points], dtype=np.uint8)
    seed_labels = labels[ys, xs]
    lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
    lut[seed_labels] = colors

    return lut[labels]


def reference_render_with_lines(
    painting, scale=1, line_width=2, line_color=(0, 0, 0)
) -> np.ndarray:
    """The original ``VoronoiPainting._render_array_with_lines``."""
    w = painting.get_img_width * scale
    h = painting.get_img_height * scale
    coords = np.array([p.coordinates for p in painting.points], dtype=np.int32) * scale
    xs = np.clip(coords[:, 0], 0, w - 1)
    ys = np.clip(coords[:, 1], 0, h - 1)
    binary = np.full((h, w), 255, dtype=np.uint8)
    binary[ys, xs] = 0
    _, labels = cv2.distanceTransformWithLabels(
        binary, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    colors = np.array([p.color[:3] for p in painting.points], dtype=np.uint8)
    seed_labels = labels[ys, xs]
    lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
    lut[seed_labels] = colors

    result = lut[labels]

    label_u = np.roll(labels, -1, axis=0)
    label_d = np.roll(labels, 1, axis=0)
    label_l = np.roll(labels, -1, axis=1)
    label_r = np.roll(labels, 1, axis=1)

    is_edge = (
        (labels != label_u)
        | (labels != label_d)
        | (labels != label_l)
        | (labels != label_r)
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (line_width * 2 - 1, line_width * 2 - 1)
    )
    is_edge = cv2.dilate(is_edge.astype(np.uint8), kernel).astype(bool)

    result[is_edge] = line_color

    return result


def reference_image_diff(painting, target) -> float:
    """The original ``VoronoiPainting.image_diff``."""
    target_np = np.array(target.convert("RGB"), dtype=np.uint8)
    return cv2.norm(reference_render(painting), target_np, cv2.NORM_L1)

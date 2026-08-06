"""
Equivalence tests for the optimized rendering and genome paths.

The speed work on ``VoronoiPainting`` is meant to be behavior preserving: the
same genome must produce exactly the same pixels and exactly the same fitness as
the original implementation. This file keeps a literal copy of the original
rendering code as a reference oracle and asserts that the optimized code matches
it byte for byte on random genomes.

Run directly (no pytest required):

    python test_equivalence.py
"""

import io
import pickle
import random
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image

from voronoi_painting import ColoredPoint, VoronoiPainting

TARGET_PATH = "./img/car.jpeg"


# ---------------------------------------------------------------------------
# Reference implementations, copied verbatim from the pre-optimization code.
# ---------------------------------------------------------------------------


def reference_render(painting, scale=1) -> np.ndarray:
    """The original ``_render_array``."""
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
    """The original ``_render_array_with_lines``."""
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
    """The original ``image_diff``."""
    target_np = np.array(target.convert("RGB"), dtype=np.uint8)
    return cv2.norm(reference_render(painting), target_np, cv2.NORM_L1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_painting(target, num_points, seed):
    """Build a painting with a reproducible random genome."""
    random.seed(seed)
    painting = VoronoiPainting(num_points, target, background_color=(128, 128, 128))
    # Push some points out of bounds and to the exact edges, so clipping and
    # duplicate-seed handling are covered too.
    painting.points[0].coordinates = (-25, -40)
    painting.points[1].coordinates = (10**6, 10**6)
    painting.points[2].coordinates = painting.points[3].coordinates
    return painting


def check(name, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")
    return bool(condition)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_render_matches_reference(target):
    print("render matches the original implementation")
    ok = True
    for seed in range(5):
        for num_points in (50, 250, 800):
            painting = random_painting(target, num_points, seed)
            ok &= check(
                f"seed={seed} points={num_points}",
                np.array_equal(painting._render_array(), reference_render(painting)),
            )
    for scale in (2, 3):
        painting = random_painting(target, 120, seed=7)
        ok &= check(
            f"scale={scale}",
            np.array_equal(
                painting._render_array(scale=scale),
                reference_render(painting, scale=scale),
            ),
        )
    return ok


def test_render_with_lines_matches_reference(target):
    print("outlined render matches the original implementation")
    ok = True
    for line_width in (1, 2, 4):
        painting = random_painting(target, 250, seed=3)
        ok &= check(
            f"line_width={line_width}",
            np.array_equal(
                painting._render_array_with_lines(line_width=line_width),
                reference_render_with_lines(painting, line_width=line_width),
            ),
        )
    painting = random_painting(target, 250, seed=4)
    ok &= check(
        "line_color=(255,0,0)",
        np.array_equal(
            painting._render_array_with_lines(line_width=2, line_color=(255, 0, 0)),
            reference_render_with_lines(painting, line_width=2, line_color=(255, 0, 0)),
        ),
    )
    return ok


def test_image_diff_matches_reference(target):
    print("fitness matches the original implementation")
    ok = True
    for seed in range(5):
        painting = random_painting(target, 250, seed)
        ok &= check(
            f"seed={seed}",
            painting.image_diff(target) == reference_image_diff(painting, target),
        )
    return ok


def test_buffer_reuse_is_safe(target):
    print("shared render buffers do not corrupt earlier results")
    a = random_painting(target, 250, seed=11)
    b = random_painting(target, 250, seed=12)
    first = a._render_array()
    expected = reference_render(a)
    # Rendering another painting reuses the same scratch buffer.
    b._render_array()
    b.image_diff(target)
    # Encoding forces PIL to read through the pixel buffer it was handed.
    a.draw().save(io.BytesIO(), format="PNG")
    ok = check("copy survives later renders", np.array_equal(first, expected))
    ok &= check(
        "draw() output is independent",
        np.array_equal(np.array(a.draw()), expected),
    )
    return ok


def test_deepcopy_is_independent(target):
    print("deepcopy detaches the genome but shares the target")
    painting = random_painting(target, 250, seed=21)
    clone = deepcopy(painting)
    ok = check(
        "same pixels after copy",
        np.array_equal(painting._render_array(), clone._render_array()),
    )
    clone.mutate_points(rate=1.0, sigma=1.0)
    ok &= check(
        "mutating the copy leaves the original untouched",
        np.array_equal(painting._render_array(), reference_render(painting)),
    )
    ok &= check(
        "points are distinct objects",
        all(p is not q for p, q in zip(painting.points, clone.points)),
    )
    ok &= check("target image is shared", clone.target_image is painting.target_image)
    return ok


def test_pickle_round_trip(target):
    print("pickling preserves the genome exactly")
    painting = random_painting(target, 250, seed=31)
    restored = pickle.loads(pickle.dumps(painting, protocol=pickle.HIGHEST_PROTOCOL))
    ok = check(
        "coordinates preserved",
        [p.coordinates for p in painting.points]
        == [p.coordinates for p in restored.points],
    )
    ok &= check(
        "colors preserved",
        [p.color for p in painting.points] == [p.color for p in restored.points],
    )
    ok &= check(
        "same pixels after round trip",
        np.array_equal(painting._render_array(), restored._render_array()),
    )
    ok &= check(
        "same fitness after round trip",
        painting.image_diff(target) == restored.image_diff(restored.target_image),
    )
    ok &= check("dimensions preserved", restored.get_img_width == painting.get_img_width)
    ok &= check(
        "background preserved",
        restored.get_background_color == painting.get_background_color,
    )
    return ok


def test_operators_produce_independent_children(target):
    print("crossover operators return detached children")
    mom = random_painting(target, 250, seed=41)
    dad = random_painting(target, 250, seed=42)

    child = VoronoiPainting.breed(mom, dad)
    ok = check("breed keeps the point count", child.num_points == mom.num_points)
    ok &= check(
        "breed takes every point from a parent",
        all(
            point.coordinates in (m.coordinates, d.coordinates)
            for point, m, d in zip(child.points, mom.points, dad.points)
        ),
    )
    child.mutate_points(rate=1.0, sigma=2.0)
    ok &= check(
        "mutating the child leaves parents untouched",
        np.array_equal(mom._render_array(), reference_render(mom)),
    )

    merged = VoronoiPainting.merge(mom, dad)
    ok &= check("merge doubles the point count", merged.num_points == 2 * mom.num_points)
    merged.mutate_points(rate=1.0, sigma=2.0)
    ok &= check(
        "mutating the merged child leaves parents untouched",
        np.array_equal(dad._render_array(), reference_render(dad)),
    )

    child_a, child_b = VoronoiPainting.mate(mom, dad)
    ok &= check(
        "mate returns complementary children",
        all(
            {a.coordinates, b.coordinates} == {m.coordinates, d.coordinates}
            for a, b, m, d in zip(child_a.points, child_b.points, mom.points, dad.points)
        ),
    )
    return ok


def test_mutation_rate_is_respected(target):
    print("mutation touches the expected number of points")
    ok = True
    for rate in (0.0, 0.03, 0.05, 1.0):
        painting = random_painting(target, 400, seed=51)
        before = [(p.coordinates, p.color) for p in painting.points]
        painting.mutate_points(rate=rate, sigma=0.5)
        after = [(p.coordinates, p.color) for p in painting.points]
        changed = sum(1 for x, y in zip(before, after) if x != y)
        # A mutation can be a no-op (e.g. a zero-sized shift), so the count of
        # observed changes is an upper bound check, not an equality.
        ok &= check(f"rate={rate}", changed <= int(rate * 400))
    return ok


def test_colored_point_bounds(target):
    print("freshly created points stay inside the valid color range")
    random.seed(61)
    points = [ColoredPoint(100, 100) for _ in range(2000)]
    ok = check(
        "channels within 0-255",
        all(0 <= channel <= 255 for point in points for channel in point.color),
    )
    ok &= check("alpha fixed at 255", all(point.color[3] == 255 for point in points))
    return ok


def main():
    target = Image.open(TARGET_PATH).convert("RGBA")
    tests = [
        test_render_matches_reference,
        test_render_with_lines_matches_reference,
        test_image_diff_matches_reference,
        test_buffer_reuse_is_safe,
        test_deepcopy_is_independent,
        test_pickle_round_trip,
        test_operators_produce_independent_children,
        test_mutation_rate_is_respected,
        test_colored_point_bounds,
    ]

    passed = True
    for test in tests:
        passed &= bool(test(target))
        print()

    print("ALL TESTS PASSED" if passed else "SOME TESTS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

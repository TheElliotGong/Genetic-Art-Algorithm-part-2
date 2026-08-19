"""Unit tests for the genome, its operators and the renderer."""

import random

import numpy as np
import pytest
from PIL import Image

import voronoi_painting
from tests.conftest import build_painting, make_target
from voronoi_painting import ColoredPoint, VoronoiPainting


# --- ColoredPoint ---------------------------------------------------------


def test_new_points_stay_inside_the_image_and_colour_range():
    random.seed(61)
    points = [ColoredPoint(100, 80) for _ in range(500)]

    assert all(0 <= point.coordinates[0] <= 100 for point in points)
    assert all(0 <= point.coordinates[1] <= 80 for point in points)
    assert all(0 <= channel <= 255 for point in points for channel in point.color)
    assert all(point.color[3] == 255 for point in points)


def test_point_copy_is_independent():
    random.seed(1)
    point = ColoredPoint(50, 50)
    clone = point.copy()

    assert clone is not point
    assert clone.coordinates == point.coordinates
    assert clone.color == point.color

    before = (point.coordinates, point.color)
    # Mutating hard enough that a no-op is vanishingly unlikely; whatever it
    # changes must not reach the original.
    for _ in range(20):
        clone.mutate(sigma=5.0)
    assert (point.coordinates, point.color) == before
    assert (clone.coordinates, clone.color) != before


@pytest.mark.parametrize("sigma", [0.5, 1.0, 4.0])
def test_colour_mutation_stays_clamped(sigma):
    """Colour jitter must never push a channel outside 0-255."""
    random.seed(7)
    point = ColoredPoint(10, 10)
    point.color = (255, 0, 255, 255)
    for _ in range(500):
        point.mutate(sigma=sigma)
        assert all(0 <= channel <= 255 for channel in point.color)
        assert point.color[3] == 255


def test_point_deepcopy_is_detached_and_memoized():
    from copy import deepcopy

    random.seed(2)
    point = ColoredPoint(40, 40)
    pair = deepcopy([point, point])

    assert pair[0] is pair[1], "deepcopy memo should collapse the shared point"
    assert pair[0] is not point
    assert (pair[0].coordinates, pair[0].color) == (point.coordinates, point.color)


def test_point_pickle_state_round_trips():
    random.seed(3)
    point = ColoredPoint(30, 30)
    restored = ColoredPoint.__new__(ColoredPoint)
    restored.__setstate__(point.__getstate__())

    assert restored.coordinates == point.coordinates
    assert restored.color == point.color


def test_str_describes_the_point():
    random.seed(3)
    point = ColoredPoint(30, 30)
    point.coordinates = (4, 5)
    point.color = (10, 20, 30, 255)
    assert "(4, 5)" in str(point)
    assert "(10, 20, 30)" in str(point)


# --- VoronoiPainting basics -----------------------------------------------


def test_painting_reports_its_geometry(target):
    painting = build_painting(target, 40, seed=2)

    assert painting.num_points == 40
    assert (painting.get_img_width, painting.get_img_height) == target.size
    assert painting.get_background_color == (128, 128, 128)
    # The registry is content addressed, so this resolves to *an* image with the
    # target's pixels - not necessarily the same object.
    assert np.array_equal(np.array(painting.target_image), np.array(target))
    assert "40" in repr(painting)


def test_render_shape_and_scaling(target):
    painting = build_painting(target, 30, seed=5)
    width, height = target.size

    assert painting._render_array().shape == (height, width, 3)
    assert painting._render_array(scale=2).shape == (height * 2, width * 2, 3)
    assert painting.draw().size == (width, height)
    assert painting.draw(scale=3).size == (width * 3, height * 3)
    assert painting.draw().mode == "RGB"


def test_render_uses_only_genome_colours(target):
    """Every rendered pixel must come from some point's colour."""
    painting = build_painting(target, 12, seed=9)
    rendered = painting._render_array()

    used = {tuple(colour) for colour in rendered.reshape(-1, 3)}
    available = {point.color[:3] for point in painting.points}
    assert used <= available


def test_outlines_darken_region_boundaries(target):
    painting = build_painting(target, 40, seed=13)
    plain = painting._render_array()
    outlined = painting._render_array_with_lines(line_width=2, line_color=(0, 0, 0))

    changed = np.any(plain != outlined, axis=2)
    assert changed.any(), "outlining changed nothing"
    assert np.all(outlined[changed] == 0)


def test_wider_outlines_cover_more_pixels(target):
    painting = build_painting(target, 40, seed=13)
    plain = painting._render_array()

    covered = []
    for width in (1, 2, 4):
        outlined = painting._render_array_with_lines(line_width=width)
        covered.append(int(np.any(plain != outlined, axis=2).sum()))

    assert covered[0] < covered[1] < covered[2]


def test_draw_lines_returns_an_image(target):
    painting = build_painting(target, 20, seed=17)
    image = painting.draw_lines(scale=2, line_width=1, line_color=(255, 0, 0))

    assert isinstance(image, Image.Image)
    assert image.size == (target.size[0] * 2, target.size[1] * 2)


def test_shrink_removes_exactly_one_point(target):
    painting = build_painting(target, 25, seed=19)
    painting.shrink_points()
    assert painting.num_points == 24


@pytest.mark.parametrize("rate", [0.0, 0.03, 0.05, 1.0])
def test_mutation_touches_at_most_the_requested_fraction(target, rate):
    painting = build_painting(target, 400, seed=51)
    before = [(p.coordinates, p.color) for p in painting.points]
    painting.mutate_points(rate=rate, sigma=0.5)
    after = [(p.coordinates, p.color) for p in painting.points]

    changed = sum(1 for x, y in zip(before, after) if x != y)
    # A mutation can be a no-op (a zero-sized shift, or a clamped colour), so
    # the requested count is an upper bound rather than an exact figure.
    assert changed <= int(rate * 400)


def test_zero_rate_mutation_is_a_no_op(target):
    painting = build_painting(target, 10, seed=53)
    before = [(p.coordinates, p.color) for p in painting.points]
    painting.mutate_points(rate=0.0, sigma=1.0)
    assert [(p.coordinates, p.color) for p in painting.points] == before


def test_full_rate_mutation_changes_most_points(target):
    painting = build_painting(target, 200, seed=55)
    before = [(p.coordinates, p.color) for p in painting.points]
    painting.mutate_points(rate=1.0, sigma=2.0)
    after = [(p.coordinates, p.color) for p in painting.points]

    changed = sum(1 for x, y in zip(before, after) if x != y)
    assert changed > 150


# --- Fitness ---------------------------------------------------------------


def test_image_diff_is_zero_for_a_perfect_match():
    """A single point covering a flat target reproduces it exactly."""
    flat = Image.new("RGBA", (16, 12), (40, 60, 80, 255))
    painting = VoronoiPainting(1, flat)
    painting.points[0].coordinates = (5, 5)
    painting.points[0].color = (40, 60, 80, 255)

    assert painting.image_diff(flat) == 0.0


def test_image_diff_grows_with_error():
    flat = Image.new("RGBA", (16, 12), (0, 0, 0, 255))
    painting = VoronoiPainting(1, flat)
    painting.points[0].coordinates = (5, 5)

    painting.points[0].color = (1, 0, 0, 255)
    near = painting.image_diff(flat)
    painting.points[0].color = (255, 255, 255, 255)
    far = painting.image_diff(flat)

    assert 0 < near < far
    assert far == 16 * 12 * 255 * 3


# --- Crossover -------------------------------------------------------------


def test_breed_falls_back_to_the_larger_parent_when_sizes_differ(target):
    small = build_painting(target, 10, seed=61)
    large = build_painting(target, 30, seed=62)

    child = VoronoiPainting.breed(small, large)
    assert child.num_points == 30
    assert child.points[0] is not large.points[0]


def test_mate_falls_back_to_the_larger_parent_when_sizes_differ(target):
    small = build_painting(target, 10, seed=63)
    large = build_painting(target, 30, seed=64)

    first, second = VoronoiPainting.mate(small, large)
    assert first is large and second is large

    first, second = VoronoiPainting.mate(large, small)
    assert first is large and second is large


def test_mating_is_refused_across_image_sizes(target):
    other = make_target(20, 20)
    a = build_painting(target, 10, seed=65)
    b = build_painting(other, 10, seed=66)

    assert not VoronoiPainting._mate_possible(a, b)


def test_children_average_the_parent_backgrounds(target):
    a = build_painting(target, 8, seed=67)
    b = build_painting(target, 8, seed=68)
    a._background_color = (0, 0, 0, 255)
    b._background_color = (100, 200, 50, 255)

    child = VoronoiPainting.breed(a, b)
    assert child.get_background_color == (50, 100, 25)


def test_merge_interleaves_both_genomes(target):
    a = build_painting(target, 5, seed=69)
    b = build_painting(target, 5, seed=70)

    merged = VoronoiPainting.merge(a, b)
    assert [p.coordinates for p in merged.points[::2]] == [
        p.coordinates for p in a.points
    ]
    assert [p.coordinates for p in merged.points[1::2]] == [
        p.coordinates for p in b.points
    ]


# --- Render buffers --------------------------------------------------------


def test_render_buffer_cache_is_bounded(target):
    """Many distinct tile sizes must not grow the scratch cache without bound."""
    saved = dict(voronoi_painting._RENDER_BUFFERS)
    voronoi_painting._RENDER_BUFFERS.clear()
    try:
        for size in range(8, 8 + voronoi_painting._MAX_BUFFER_SHAPES + 4):
            painting = build_painting(make_target(size, size), 6, seed=size)
            painting._render_array()
        assert (
            len(voronoi_painting._RENDER_BUFFERS)
            <= voronoi_painting._MAX_BUFFER_SHAPES
        )
    finally:
        voronoi_painting._RENDER_BUFFERS.clear()
        voronoi_painting._RENDER_BUFFERS.update(saved)


def test_buffers_are_reused_for_a_repeated_shape():
    first = voronoi_painting._render_buffers(9, 11)
    second = voronoi_painting._render_buffers(9, 11)
    assert all(a is b for a, b in zip(first, second))

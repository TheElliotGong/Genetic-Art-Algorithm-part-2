"""The optimized painting code must behave exactly like the original.

Every assertion here compares against ``tests.reference_impl``, a verbatim copy
of the pre-optimization implementation. If one of these fails, the speed work
changed what the algorithm produces, not just how fast it produces it.
"""

import io
import pickle
from copy import deepcopy

import numpy as np
import pytest

from tests.conftest import build_painting
from tests.reference_impl import (
    reference_image_diff,
    reference_render,
    reference_render_with_lines,
)
from voronoi_painting import VoronoiPainting


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("num_points", [50, 250, 800])
def test_render_matches_reference(target, seed, num_points):
    painting = build_painting(target, num_points, seed)
    assert np.array_equal(painting._render_array(), reference_render(painting))


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_render_matches_reference_at_scale(target, scale):
    painting = build_painting(target, 120, seed=7)
    assert np.array_equal(
        painting._render_array(scale=scale), reference_render(painting, scale=scale)
    )


@pytest.mark.parametrize("line_width", [1, 2, 4])
def test_outlined_render_matches_reference(target, line_width):
    painting = build_painting(target, 250, seed=3)
    assert np.array_equal(
        painting._render_array_with_lines(line_width=line_width),
        reference_render_with_lines(painting, line_width=line_width),
    )


def test_outlined_render_honours_line_colour(target):
    painting = build_painting(target, 250, seed=4)
    assert np.array_equal(
        painting._render_array_with_lines(line_width=2, line_color=(255, 0, 0)),
        reference_render_with_lines(painting, line_width=2, line_color=(255, 0, 0)),
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_image_diff_matches_reference(target, seed):
    painting = build_painting(target, 250, seed)
    assert painting.image_diff(target) == reference_image_diff(painting, target)


def test_shared_render_buffers_do_not_corrupt_earlier_results(target):
    """A returned render must survive later renders in the same process."""
    a = build_painting(target, 250, seed=11)
    b = build_painting(target, 250, seed=12)

    first = a._render_array()
    expected = reference_render(a)

    # Both of these reuse the scratch buffer `first` was rendered through.
    b._render_array()
    b.image_diff(target)
    # Encoding forces PIL to read through the pixel buffer it was handed.
    a.draw().save(io.BytesIO(), format="PNG")

    assert np.array_equal(first, expected)
    assert np.array_equal(np.array(a.draw()), expected)


def test_deepcopy_detaches_the_genome_but_shares_the_target(target):
    painting = build_painting(target, 250, seed=21)
    clone = deepcopy(painting)

    assert np.array_equal(painting._render_array(), clone._render_array())
    assert all(p is not q for p, q in zip(painting.points, clone.points))
    assert clone.target_image is painting.target_image

    clone.mutate_points(rate=1.0, sigma=1.0)
    assert np.array_equal(painting._render_array(), reference_render(painting))


def test_pickle_round_trip_preserves_the_genome(target):
    painting = build_painting(target, 250, seed=31)
    restored = pickle.loads(pickle.dumps(painting, protocol=pickle.HIGHEST_PROTOCOL))

    assert [p.coordinates for p in painting.points] == [
        p.coordinates for p in restored.points
    ]
    assert [p.color for p in painting.points] == [p.color for p in restored.points]
    assert np.array_equal(painting._render_array(), restored._render_array())
    assert painting.image_diff(target) == restored.image_diff(restored.target_image)
    assert restored.get_img_width == painting.get_img_width
    assert restored.get_img_height == painting.get_img_height
    assert restored.get_background_color == painting.get_background_color


def test_breed_returns_a_detached_child(target):
    mom = build_painting(target, 250, seed=41)
    dad = build_painting(target, 250, seed=42)

    child = VoronoiPainting.breed(mom, dad)
    assert child.num_points == mom.num_points
    assert all(
        point.coordinates in (m.coordinates, d.coordinates)
        for point, m, d in zip(child.points, mom.points, dad.points)
    )

    child.mutate_points(rate=1.0, sigma=2.0)
    assert np.array_equal(mom._render_array(), reference_render(mom))


def test_merge_doubles_the_genome_and_detaches_it(target):
    mom = build_painting(target, 250, seed=41)
    dad = build_painting(target, 250, seed=42)

    merged = VoronoiPainting.merge(mom, dad)
    assert merged.num_points == 2 * mom.num_points

    merged.mutate_points(rate=1.0, sigma=2.0)
    assert np.array_equal(dad._render_array(), reference_render(dad))


def test_mate_returns_complementary_children(target):
    mom = build_painting(target, 250, seed=41)
    dad = build_painting(target, 250, seed=42)

    child_a, child_b = VoronoiPainting.mate(mom, dad)
    assert all(
        {a.coordinates, b.coordinates} == {m.coordinates, d.coordinates}
        for a, b, m, d in zip(child_a.points, child_b.points, mom.points, dad.points)
    )

"""Tests for the divide-and-conquer tiling: splitting, feathering, stitching."""

import numpy as np
import pytest
from PIL import Image

from evolve_tiled import (
    TILE_OVERLAP_PIXELS,
    TileSpec,
    feather_blend_tiles,
    split_into_tiles,
    stitch,
)
from tests.conftest import build_painting, make_target


# --- Splitting -------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "cols"), [(1, 1), (1, 3), (3, 1), (2, 2), (3, 3), (4, 5)]
)
def test_split_produces_one_tile_per_cell(rows, cols):
    image = make_target(97, 71)
    tiles = split_into_tiles(image, rows, cols, overlap_pixels=8)
    assert len(tiles) == rows * cols


def test_tile_cores_tile_the_image_exactly():
    """The cores must cover every pixel exactly once - no gaps, no overlaps."""
    width, height = 97, 71
    image = make_target(width, height)
    coverage = np.zeros((height, width), dtype=np.int32)

    for tile in split_into_tiles(image, 3, 4, overlap_pixels=9):
        coverage[tile.y0 : tile.y1, tile.x0 : tile.x1] += 1

    assert np.array_equal(coverage, np.ones_like(coverage))


def test_crops_extend_by_the_overlap_but_stay_inside_the_image():
    width, height = 97, 71
    overlap = 9
    image = make_target(width, height)

    for tile in split_into_tiles(image, 3, 3, overlap_pixels=overlap):
        assert 0 <= tile.crop_x0 <= tile.x0
        assert tile.x1 <= tile.crop_x1 <= width
        assert 0 <= tile.crop_y0 <= tile.y0
        assert tile.y1 <= tile.crop_y1 <= height
        assert tile.x0 - tile.crop_x0 <= overlap
        assert tile.crop_x1 - tile.x1 <= overlap
        assert tile.image.size == tile.crop_size


def test_tile_pixels_match_the_source_region():
    image = make_target(64, 48)
    source = np.array(image.convert("RGB"))

    for tile in split_into_tiles(image, 2, 2, overlap_pixels=6):
        expected = source[tile.crop_y0 : tile.crop_y1, tile.crop_x0 : tile.crop_x1]
        assert np.array_equal(np.array(tile.image.convert("RGB")), expected)


def test_zero_overlap_makes_crops_equal_to_cores():
    image = make_target(40, 40)
    for tile in split_into_tiles(image, 2, 2, overlap_pixels=0):
        assert tile.crop_size == tile.core_size
        assert (tile.crop_x0, tile.crop_y0) == (tile.x0, tile.y0)


def test_a_single_tile_covers_the_whole_image():
    image = make_target(40, 30)
    (tile,) = split_into_tiles(image, 1, 1, overlap_pixels=TILE_OVERLAP_PIXELS)

    assert tile.core_size == (40, 30)
    assert tile.crop_size == (40, 30)
    assert tile.image.size == image.size


def test_tile_spec_is_immutable():
    image = make_target(16, 16)
    (tile,) = split_into_tiles(image, 1, 1, overlap_pixels=0)
    with pytest.raises(AttributeError):
        tile.x0 = 5


# --- Feathering ------------------------------------------------------------


def test_feather_mask_has_the_crop_shape_and_unit_range():
    image = make_target(64, 48)
    for tile in split_into_tiles(image, 2, 2, overlap_pixels=8):
        mask = feather_blend_tiles(tile)
        width, height = tile.crop_size

        assert mask.shape == (height, width)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


def test_feather_mask_is_opaque_over_the_tile_core():
    image = make_target(64, 48)
    for tile in split_into_tiles(image, 2, 2, overlap_pixels=8):
        mask = feather_blend_tiles(tile)
        core = mask[
            tile.y0 - tile.crop_y0 : tile.y1 - tile.crop_y0,
            tile.x0 - tile.crop_x0 : tile.x1 - tile.crop_x0,
        ]
        assert np.allclose(core, 1.0)


def test_feather_mask_fades_only_into_the_overlap():
    image = make_target(64, 48)
    tiles = split_into_tiles(image, 2, 2, overlap_pixels=8)
    # The top-left tile has no left/top margin (image border) but does have a
    # right/bottom one.
    mask = feather_blend_tiles(tiles[0])

    assert mask[0, 0] == pytest.approx(1.0)
    assert mask[-1, -1] < 1.0
    assert mask[0, -1] < 1.0


def test_feather_weights_sum_to_one_everywhere():
    """The stitched canvas divides by these weights, so they must not vanish."""
    image = make_target(80, 60)
    tiles = split_into_tiles(image, 3, 3, overlap_pixels=7)
    total = np.zeros((60, 80), dtype=np.float32)

    for tile in tiles:
        mask = feather_blend_tiles(tile)
        total[tile.crop_y0 : tile.crop_y1, tile.crop_x0 : tile.crop_x1] += mask

    assert total.min() > 0.0


def test_a_single_tile_mask_is_fully_opaque():
    image = make_target(32, 24)
    (tile,) = split_into_tiles(image, 1, 1, overlap_pixels=8)
    assert np.allclose(feather_blend_tiles(tile), 1.0)


# --- Stitching -------------------------------------------------------------


def test_stitch_returns_a_full_size_image(target):
    tiles = split_into_tiles(target, 2, 2, overlap_pixels=6)
    results = [(tile, build_painting(tile.image, 12, seed=i)) for i, tile in enumerate(tiles)]

    stitched = stitch(results, target.size, fade_pixels=6)

    assert isinstance(stitched, Image.Image)
    assert stitched.size == target.size
    assert stitched.mode == "RGB"


def test_stitching_uniform_tiles_reproduces_the_colour():
    """Feathered blending of identical tiles must not darken the seams.

    Each tile gets a single Voronoi cell, so it renders flat with no outlines
    and the only thing under test is the weighted blend.
    """
    image = Image.new("RGBA", (48, 32), (10, 120, 200, 255))
    tiles = split_into_tiles(image, 2, 2, overlap_pixels=6)

    results = []
    for index, tile in enumerate(tiles):
        painting = build_painting(tile.image, 1, seed=index)
        painting.points[0].color = (10, 120, 200, 255)
        results.append((tile, painting))

    stitched = np.array(stitch(results, image.size, fade_pixels=6)).astype(np.int16)

    # The blend accumulates in float32 and truncates on the way back to uint8,
    # so a single unit of rounding is expected; a darkened seam would not be.
    assert np.abs(stitched - np.array([10, 120, 200])).max() <= 1


def test_stitch_of_one_tile_matches_that_tile(target):
    (tile,) = split_into_tiles(target, 1, 1, overlap_pixels=0)
    painting = build_painting(tile.image, 20, seed=3)

    stitched = np.array(stitch([(tile, painting)], target.size, fade_pixels=0))
    direct = np.array(painting.draw_lines(scale=1, line_width=1).convert("RGB"))

    assert np.allclose(stitched, direct, atol=1)


@pytest.mark.slow
def test_evolve_tile_runs_end_to_end(tmp_path, capsys):
    """The per-tile driver, at the smallest settings that still exercise it."""
    from evolve_tiled import evolve_tile

    tile = make_target(32, 24)
    # ``evolve_tile`` hard-codes ``survive(fraction=0.025)``, so the population
    # has to be at least 40 for anyone to survive a generation.
    best = evolve_tile(
        tile,
        tile_id="00",
        output_dir=str(tmp_path),
        points_initial=6,
        population_size=40,
        workers=1,
        gens_phase1=2,
        gens_phase2=2,
    )

    assert (best.get_img_width, best.get_img_height) == tile.size
    # Survivors keep their 6 points while merged children carry 12, so the best
    # individual is one or the other - but never something else.
    assert best.num_points in (6, 12)
    # Phase 1, the duplication generation and phase 2 all ran, and the final
    # outlined render was written under the last generation number.
    assert (tmp_path / "tile_00_gen_00005.png").is_file()
    assert "[Tile 00] 32x24px" in capsys.readouterr().out


@pytest.mark.slow
def test_evolve_tile_writes_a_renderable_final_image(tmp_path):
    from evolve_tiled import evolve_tile

    best = evolve_tile(
        make_target(24, 24),
        tile_id="07",
        output_dir=str(tmp_path),
        points_initial=4,
        population_size=40,
        workers=1,
        gens_phase1=1,
        gens_phase2=1,
    )

    written = sorted(path.name for path in tmp_path.glob("tile_07_gen_*.png"))
    assert written == ["tile_07_gen_00003.png"]
    with Image.open(tmp_path / written[0]) as image:
        assert image.size == (24, 24)
    assert best.num_points in (4, 8)


def test_tile_spec_size_properties():
    spec = TileSpec(
        x0=10, y0=20, x1=30, y1=50, crop_x0=5, crop_y0=15, crop_x1=35, crop_y1=55,
        image=Image.new("RGB", (30, 40)),
    )
    assert spec.core_size == (20, 30)
    assert spec.crop_size == (30, 40)

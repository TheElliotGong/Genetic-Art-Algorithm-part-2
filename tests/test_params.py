"""Tests for the run-parameter model.

``RunParams`` is both the request validator and the description the front-end
builds its controls from, so its bounds and derived properties are load-bearing.
"""

import pytest
from pydantic import ValidationError

from webapp.params import RunParams


def test_defaults_are_valid_and_self_consistent():
    params = RunParams()

    assert params.mode == "single"
    assert params.tile_count == 1
    assert params.total_generations == params.generations_per_target
    assert 2 <= params.survivor_count <= params.population_size


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generations", 0),
        ("generations", 20_001),
        ("population_size", 3),
        ("population_size", 1001),
        ("num_points", 3),
        ("mutation_rate", -0.1),
        ("mutation_rate", 1.1),
        ("mutation_sigma", 0.0),
        ("survive_fraction", 0.0),
        ("survive_fraction", 1.0),
        ("region_bias", 1.5),
        ("initial_colors", 1),
        ("final_colors", 0),
        ("max_dimension", 32),
        ("max_dimension", 2001),
        ("tile_rows", 0),
        ("tile_cols", 9),
        ("tile_overlap", -1),
        ("outline_width", 0),
        ("render_scale", 9),
        ("workers", 0),
        ("workers", 33),
        ("preview_every", 0),
        ("mode", "spiral"),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        RunParams(**{field: value})


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("#1A2B3C", "#1a2b3c"),
        ("1a2b3c", "#1a2b3c"),
        ("#abc", "#aabbcc"),
        ("  #FFF  ", "#ffffff"),
    ],
)
def test_outline_colour_is_normalised(supplied, expected):
    assert RunParams(outline_color=supplied).outline_color == expected


@pytest.mark.parametrize("bad", ["#12345", "not-a-colour", "#gggggg", ""])
def test_bad_outline_colours_are_rejected(bad):
    with pytest.raises(ValidationError):
        RunParams(outline_color=bad)


def test_outline_rgb_decodes_the_hex_colour():
    assert RunParams(outline_color="#ff8000").outline_rgb == (255, 128, 0)
    assert RunParams(outline_color="#000").outline_rgb == (0, 0, 0)


def test_tile_count_follows_the_grid():
    assert RunParams(mode="single", tile_rows=3, tile_cols=3).tile_count == 1
    assert RunParams(mode="tiled", tile_rows=3, tile_cols=4).tile_count == 12


def test_total_generations_covers_every_tile():
    params = RunParams(mode="tiled", tile_rows=2, tile_cols=3, generations=100)

    assert params.generations_per_target == 101  # duplication costs one generation
    assert params.total_generations == 101 * 6


def test_duplication_can_be_switched_off():
    params = RunParams(generations=100, genome_duplication=False)

    assert params.generations_per_target == 100
    assert params.phase_plan() == [("explore", 100)]


def test_phase_plan_splits_the_run_in_half():
    params = RunParams(generations=100)
    assert params.phase_plan() == [("explore", 50), ("duplicate", 1), ("refine", 50)]


@pytest.mark.parametrize("generations", [1, 2, 3, 7, 999, 1000])
def test_phase_plan_always_spends_the_requested_generations(generations):
    """Rounding must not lose or invent generations, and no phase may be negative."""
    params = RunParams(generations=generations)
    plan = params.phase_plan()

    evolving = sum(count for label, count in plan if label != "duplicate")
    assert evolving == generations
    assert all(count >= 0 for _, count in plan)
    assert sum(count for _, count in plan) == params.generations_per_target


@pytest.mark.parametrize(
    ("population_size", "fraction", "expected"),
    [
        (60, 0.025, 2),  # rounds to 1, floored at 2 so two parents exist
        (1000, 0.025, 25),
        (100, 0.5, 50),
        (4, 0.9, 4),
        (10, 0.001, 2),
    ],
)
def test_survivor_count_stays_usable(population_size, fraction, expected):
    params = RunParams(population_size=population_size, survive_fraction=fraction)
    assert params.survivor_count == expected
    assert params.survivor_count <= params.population_size


def test_model_dump_round_trips():
    params = RunParams(mode="tiled", generations=42, outline_color="#ABC")
    assert RunParams(**params.model_dump()) == params

"""Tests for the GA building blocks: palettes, region seeding and selection."""

import random
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image
from skimage import feature
import cv2

from evolve_voronoi import (
    build_region_groups,
    clone,
    condense_palette,
    create_region_seeded_population,
    map_pixels_to_palette,
    mate,
    merge,
    mutate_painting,
    pick_best,
    pick_best_and_random,
    pick_random,
    score,
    shrink_painting,
    simplify_palette,
)
from tests.conftest import build_painting, make_target
from voronoi_painting import VoronoiPainting


class FakeIndividual:
    """Stands in for ``evol.Individual`` in the parent-picker tests."""

    def __init__(self, fitness, name):
        self.fitness = fitness
        self.name = name

    def __repr__(self):
        return f"<{self.name} fitness={self.fitness}>"


# --- Palette ---------------------------------------------------------------


def test_condense_palette_drops_near_duplicates():
    colors = [(0, 0, 0), (1, 1, 1), (255, 255, 255), (250, 250, 250)]
    condensed = condense_palette(colors, threshold=30)

    assert condensed == [(0, 0, 0), (255, 255, 255)]


def test_condense_palette_keeps_distinct_colours():
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    assert condense_palette(colors, threshold=30) == colors


def test_condense_palette_threshold_is_honoured():
    colors = [(0, 0, 0), (40, 0, 0)]
    assert len(condense_palette(colors, threshold=30)) == 2
    assert len(condense_palette(colors, threshold=50)) == 1


def test_simplify_palette_returns_at_most_the_target_count():
    colors = [(i * 8, 0, 255 - i * 8) for i in range(32)]
    for target_count in (1, 2, 5, 12):
        simplified = simplify_palette(colors, target_count)
        assert len(simplified) <= target_count
        assert all(color in colors for color in simplified)


def test_simplify_palette_keeps_a_short_palette_intact():
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
    assert simplify_palette(colors, 10) == colors


def test_simplify_palette_spans_the_condensed_range():
    """The subsample must keep both ends, not just the first N colours."""
    colors = [(i * 8, 0, 0) for i in range(32)]
    simplified = simplify_palette(colors, 4)

    condensed = condense_palette(colors)
    assert simplified[0] == condensed[0]
    assert simplified[-1] == condensed[-1]


def test_map_pixels_to_palette_returns_one_label_per_pixel():
    palette = [(0, 0, 0), (90, 90, 90), (255, 255, 255)]
    image = np.array(
        [[[10, 10, 10], [240, 240, 240]], [[128, 128, 128], [200, 200, 200]]],
        dtype=np.uint8,
    )
    labels = map_pixels_to_palette(image, palette)

    assert labels.shape == (2, 2)
    assert labels.min() >= 0
    assert labels.max() < len(palette)


def test_map_pixels_to_palette_picks_the_nearest_colour_when_close():
    palette = [(0, 0, 0), (100, 100, 100)]
    image = np.array([[[10, 10, 10], [95, 95, 95]]], dtype=np.uint8)

    labels = map_pixels_to_palette(image, palette)

    assert labels[0, 0] == 0
    assert labels[0, 1] == 1


def test_map_pixels_to_palette_handles_far_apart_colours():
    """Regression: squaring the difference must not overflow its integer type.

    A channel gap over 181 overflows int16 into a negative "distance", which
    made ``argmin`` prefer the furthest colour - black pixels mapped to white.
    """
    palette = [(0, 0, 0), (255, 255, 255)]
    image = np.array([[[10, 10, 10], [240, 240, 240]]], dtype=np.uint8)

    labels = map_pixels_to_palette(image, palette)

    assert labels[0, 0] == 0
    assert labels[0, 1] == 1


def test_map_pixels_to_palette_is_correct_across_the_whole_range():
    """Exhaustive check against a plain nearest-neighbour computed in Python."""
    palette = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 128, 255), (90, 200, 30)]
    values = np.array(
        [[[r, g, b] for b in (0, 60, 127, 200, 255)] for r in (0, 127, 255) for g in (0, 255)],
        dtype=np.uint8,
    )

    labels = map_pixels_to_palette(values, palette)

    for y, row in enumerate(values):
        for x, pixel in enumerate(row):
            expected = min(
                range(len(palette)),
                key=lambda index: sum(
                    (int(a) - int(b)) ** 2 for a, b in zip(pixel, palette[index])
                ),
            )
            assert labels[y, x] == expected, f"pixel {tuple(pixel)}"


def test_region_colour_labels_respect_a_dark_and_light_palette():
    """The overflow surfaced here: a black region grouped with the white one."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:] = 255

    labels = map_pixels_to_palette(image, [(0, 0, 0), (255, 255, 255)])

    assert (labels[:, :4] == 0).all()
    assert (labels[:, 4:] == 1).all()


# --- Region detection ------------------------------------------------------


def edges_for(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    return rgb, feature.canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), sigma=1.2)


def test_regions_are_found_in_a_structured_image(target):
    rgb, edges = edges_for(target)
    palette = simplify_palette([(200, 30, 40), (20, 90, 220), (128, 128, 128)], 3)

    regions = build_region_groups(rgb, palette, edges, min_area=4)

    assert regions
    for region in regions:
        assert region["area"] >= 4
        assert len(region["x"]) == len(region["y"])
        assert len(region["color"]) == 3
        assert all(0 <= channel <= 255 for channel in region["color"])
        assert region["x"].max() < rgb.shape[1]
        assert region["y"].max() < rgb.shape[0]


def test_a_flat_image_falls_back_to_one_whole_image_region():
    """Segmentation that finds nothing must still hand back something usable."""
    flat = Image.new("RGB", (24, 16), (77, 77, 77))
    rgb, edges = edges_for(flat)

    regions = build_region_groups(rgb, [(77, 77, 77)], edges, min_area=10_000)

    assert len(regions) == 1
    assert regions[0]["area"] == 24 * 16
    assert regions[0]["color"] == (77, 77, 77)


def test_min_area_filters_small_regions(photo_target):
    rgb, edges = edges_for(photo_target)
    palette = simplify_palette(
        [tuple(int(v) for v in color) for color in np.array(
            photo_target.convert("P", palette=Image.ADAPTIVE, colors=16).getpalette()[:48]
        ).reshape(-1, 3)],
        8,
    )

    loose = build_region_groups(rgb, palette, edges, min_area=4)
    strict = build_region_groups(rgb, palette, edges, min_area=200)

    assert len(strict) <= len(loose)
    assert all(region["area"] >= 200 for region in strict)


# --- Seeding ---------------------------------------------------------------


def test_seeded_population_has_the_requested_shape(target):
    rgb, edges = edges_for(target)
    palette = [(200, 30, 40), (20, 90, 220)]
    regions = build_region_groups(rgb, palette, edges, min_area=4)

    population = create_region_seeded_population(
        7, 15, target, regions, palette, region_bias=0.85
    )

    assert len(population) == 7
    assert all(painting.num_points == 15 for painting in population)
    assert all(painting.get_img_width == target.size[0] for painting in population)
    assert all(painting.get_img_height == target.size[1] for painting in population)


def test_seeding_places_points_inside_the_image(target):
    rgb, edges = edges_for(target)
    palette = [(200, 30, 40), (20, 90, 220)]
    regions = build_region_groups(rgb, palette, edges, min_area=4)

    population = create_region_seeded_population(
        4, 40, target, regions, palette, region_bias=1.0
    )

    width, height = target.size
    for painting in population:
        for point in painting.points:
            assert 0 <= point.coordinates[0] < width
            assert 0 <= point.coordinates[1] < height
            assert all(0 <= channel <= 255 for channel in point.color)
            assert point.color[3] == 255


def test_zero_region_bias_only_uses_the_fallback_palette(target):
    palette = [(10, 20, 30), (200, 210, 220)]
    rgb, edges = edges_for(target)
    regions = build_region_groups(rgb, palette, edges, min_area=4)

    population = create_region_seeded_population(
        3, 30, target, regions, palette, region_bias=0.0
    )

    for painting in population:
        for point in painting.points:
            assert point.color[:3] in palette


def test_seeding_survives_an_empty_region_list(target):
    palette = [(10, 20, 30)]
    population = create_region_seeded_population(2, 5, target, [], palette, region_bias=1.0)

    assert len(population) == 2
    assert all(point.color[:3] == (10, 20, 30) for p in population for point in p.points)


def test_seeded_population_beats_a_random_one(photo_target):
    """Region seeding exists to start closer to the target; check that it does."""
    rgb, edges = edges_for(photo_target)
    converted = photo_target.convert("P", palette=Image.ADAPTIVE, colors=24)
    colors = [tuple(converted.getpalette()[i : i + 3]) for i in range(0, 72, 3)]
    palette = simplify_palette(colors, 12)
    regions = build_region_groups(rgb, palette, edges, min_area=10)

    random.seed(99)
    seeded = create_region_seeded_population(
        6, 80, photo_target, regions, palette, region_bias=0.85
    )
    random.seed(99)
    uniform = [VoronoiPainting(80, photo_target) for _ in range(6)]

    seeded_best = min(score(painting) for painting in seeded)
    uniform_best = min(score(painting) for painting in uniform)

    assert seeded_best < uniform_best


# --- Scoring and operators -------------------------------------------------


def test_score_matches_image_diff(target):
    painting = build_painting(target, 30, seed=5)
    assert score(painting) == painting.image_diff(target)


def test_score_can_print_progress_dots(monkeypatch, capsys, target):
    monkeypatch.setattr("evolve_voronoi.SHOW_EVAL_PROGRESS", True)
    score(build_painting(target, 5, seed=5))
    assert capsys.readouterr().out == "."


def test_print_summary_reports_and_checkpoints(tmp_path, capsys, target):
    """The CLI callback: log every generation, snapshot on the round numbers."""
    from evolve_voronoi import print_summary

    painting = build_painting(target, 12, seed=79)
    checkpoints = []

    class FakePopulation:
        generation = 50
        individuals = [FakeIndividual(10.0, "a"), FakeIndividual(30.0, "b")]
        current_best = individuals[0]

        def __init__(self):
            for individual in self.individuals:
                individual.chromosome = painting

        def checkpoint(self, target, method):
            checkpoints.append((target, method))

    population = FakePopulation()
    template = str(tmp_path / "gen_%05d.png")

    returned = print_summary(population, template, checkpoint_path=str(tmp_path))

    output = capsys.readouterr().out
    assert returned is population
    assert "Current generation 50" in output
    assert "best score 10" in output
    assert (tmp_path / "gen_00050.png").is_file()
    assert checkpoints == [(str(tmp_path), "pickle")]


def test_print_summary_skips_snapshots_off_the_schedule(tmp_path, capsys, target):
    from evolve_voronoi import print_summary

    painting = build_painting(target, 12, seed=80)

    class FakePopulation:
        generation = 7
        individuals = [FakeIndividual(5.0, "a")]
        current_best = individuals[0]

        def __init__(self):
            self.individuals[0].chromosome = painting

        def checkpoint(self, target, method):  # pragma: no cover - must not run
            raise AssertionError("checkpointed off-schedule")

    print_summary(FakePopulation(), str(tmp_path / "gen_%05d.png"), str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_mutate_painting_leaves_the_parent_alone(target):
    parent = build_painting(target, 50, seed=71)
    before = [(p.coordinates, p.color) for p in parent.points]

    child = mutate_painting(parent, rate=1.0, sigma=1.0)

    assert child is not parent
    assert [(p.coordinates, p.color) for p in parent.points] == before
    assert child.num_points == parent.num_points


def test_shrink_painting_leaves_the_parent_alone(target):
    parent = build_painting(target, 20, seed=73)
    child = shrink_painting(parent)

    assert parent.num_points == 20
    assert child.num_points == 19


def test_clone_copies_the_genome(target):
    parent = build_painting(target, 12, seed=75)
    copy = clone(parent)

    assert copy is not parent
    assert [p.coordinates for p in copy.points] == [p.coordinates for p in parent.points]
    assert all(a is not b for a, b in zip(copy.points, parent.points))


def test_mate_and_merge_delegate_to_the_painting(target):
    mom = build_painting(target, 10, seed=77)
    dad = build_painting(target, 10, seed=78)

    assert mate(mom, dad).num_points == 10
    assert merge(mom, dad).num_points == 20


# --- Parent pickers --------------------------------------------------------


def test_pick_best_and_random_returns_the_fittest_as_mom():
    population = [
        FakeIndividual(50.0, "a"),
        FakeIndividual(10.0, "best"),
        FakeIndividual(30.0, "c"),
    ]
    mom, dad = pick_best_and_random(population)

    assert mom.name == "best"
    assert dad in population


def test_pick_best_and_random_maximizes_when_asked():
    population = [FakeIndividual(50.0, "best"), FakeIndividual(10.0, "b")]
    mom, _ = pick_best_and_random(population, maximize=True)
    assert mom.name == "best"


def test_pickers_cope_with_an_unevaluated_population():
    population = [FakeIndividual(None, "a"), FakeIndividual(None, "b")]

    mom, dad = pick_best_and_random(population)
    assert mom in population and dad in population
    assert pick_best(population) in population


def test_pick_best_returns_the_single_fittest():
    population = [FakeIndividual(5.0, "best"), FakeIndividual(9.0, "b")]
    assert pick_best(population).name == "best"
    assert pick_best(population, maximize=True).name == "b"


def test_pick_random_returns_two_members():
    population = [FakeIndividual(float(i), str(i)) for i in range(5)]
    mom, dad = pick_random(population)
    assert mom in population and dad in population


# --- A very small end-to-end evolution -------------------------------------


@pytest.mark.slow
def test_a_short_evolution_improves_the_fitness(target):
    """The loop as ``evol`` drives it: fitness must go down, not up."""
    from evol import Evolution, Population

    random.seed(5)
    chromosomes = [VoronoiPainting(30, target) for _ in range(8)]
    population = Population(
        chromosomes=chromosomes, eval_function=score, maximize=False, concurrent_workers=1
    )
    population.evaluate()
    start = population.current_best.fitness

    evolution = (
        Evolution()
        .survive(n=2)
        .breed(parent_picker=pick_best_and_random, combiner=mate, population_size=8)
        .mutate(mutate_function=mutate_painting, rate=0.2, sigma=1.0)
        .evaluate(lazy=False)
    )
    population = population.evolve(evolution, n=15)

    assert population.generation == 15
    assert population.current_best.fitness < start


@pytest.mark.slow
def test_genome_duplication_doubles_the_point_count(target):
    from evol import Evolution, Population

    random.seed(6)
    chromosomes = [VoronoiPainting(20, target) for _ in range(6)]
    population = Population(
        chromosomes=chromosomes, eval_function=score, maximize=False, concurrent_workers=1
    )

    duplication = (
        Evolution()
        .survive(n=2)
        .breed(parent_picker=pick_best_and_random, combiner=merge, population_size=6)
        .mutate(mutate_function=mutate_painting, rate=0.05, sigma=0.5)
        .evaluate(lazy=False)
    )
    population = population.evolve(duplication, n=1)

    # Two survivors keep their original genome; the four children merged two
    # parents each and so carry twice as many points.
    counts = sorted(i.chromosome.num_points for i in population.individuals)
    assert counts == [20, 20, 40, 40, 40, 40]

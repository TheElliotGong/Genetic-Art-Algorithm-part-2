from PIL import Image, ImageDraw, ImageFilter
from evol import Evolution, Population
from copy import deepcopy
from skimage import feature

import random
import os
import cv2
import numpy as np


from voronoi_painting import VoronoiPainting


def score(x: VoronoiPainting) -> float:
    current_score = x.image_diff(x.target_image)
    print(".", end="", flush=True)
    return current_score


def pick_best_and_random(pop, maximize=False):
    evaluated_individuals = tuple(filter(lambda x: x.fitness is not None, pop))
    if len(evaluated_individuals) > 0:
        mom = max(
            evaluated_individuals, key=lambda x: x.fitness if maximize else -x.fitness
        )
    else:
        mom = random.choice(pop)
    dad = random.choice(pop)
    return mom, dad


def pick_best(pop, maximize=False):
    evaluated_individuals = tuple(filter(lambda x: x.fitness is not None, pop))
    if len(evaluated_individuals) > 0:
        mom = max(
            evaluated_individuals, key=lambda x: x.fitness if maximize else -x.fitness
        )
    else:
        mom = random.choice(pop)
    return mom


def pick_random(pop):
    mom = random.choice(pop)
    dad = random.choice(pop)
    return mom, dad


# The mutation function, it mutates the painting by mutating the points, the mutation is done in place, but we return a deepcopy of the painting to avoid issues with multiprocessing and to make sure we don't mutate the original painting in place
def mutate_painting(x: VoronoiPainting, rate=0.04, sigma=1) -> VoronoiPainting:
    x.mutate_points(rate=rate, sigma=sigma)
    return deepcopy(x)


def shrink_painting(x: VoronoiPainting) -> VoronoiPainting:
    x.shrink_points()
    return deepcopy(x)


# The crossover function, it creates a child painting by combining the points of the mom and dad paintings, the crossover is done in place, but we return a deepcopy of the child painting to avoid issues with multiprocessing and to make sure we don't mutate the original paintings in place
def mate(mom: VoronoiPainting, dad: VoronoiPainting):
    child_a, child_b = VoronoiPainting.mate(mom, dad)

    return deepcopy(child_a)


def clone(mom: VoronoiPainting):
    return deepcopy(mom)


def merge(mom: VoronoiPainting, dad: VoronoiPainting):
    child_a = VoronoiPainting.merge(mom, dad)

    return deepcopy(child_a)


def print_summary(
    pop, img_template="output%d.png", checkpoint_path="output"
) -> Population:
    avg_fitness = sum([i.fitness for i in pop.individuals]) / len(pop.individuals)
    chromosome_length = pop.individuals[0].chromosome.num_points
    print(
        "\nCurrent generation %d, best score %f, pop. avg. %f. Chromosome length %d"
        % (pop.generation, pop.current_best.fitness, avg_fitness, chromosome_length)
    )
    img = pop.current_best.chromosome.draw(scale=3)
    img.save(img_template % pop.generation, "PNG")

    if pop.generation % 50 == 0:
        pop.checkpoint(target=checkpoint_path, method="pickle")

    return pop


def scale_generations(count, scale):
    return max(1, int(round(count * scale)))


def evolve_phase_with_early_stop(
    pop,
    evolution_step,
    generations,
    label,
    improvement_window=200,
    min_improvement_ratio=0.0075,
):
    best_fitness = None
    window_reference_best = None

    for generation_idx in range(generations):
        pop = pop.evolve(evolution_step, n=1)
        current_best = pop.current_best.fitness

        if best_fitness is None or current_best < best_fitness:
            best_fitness = current_best

        if (generation_idx + 1) % improvement_window == 0:
            if window_reference_best is None:
                window_reference_best = best_fitness
                continue

            baseline = max(abs(window_reference_best), 1e-12)
            improvement = (window_reference_best - best_fitness) / baseline
            if improvement < min_improvement_ratio:
                print(
                    f"\nEarly stop in {label} at local generation {generation_idx + 1}: "
                    f"{improvement * 100:.3f}% improvement over last {improvement_window} generations"
                )
                break

            window_reference_best = best_fitness

    return pop


# Condense similar colors in the palette to create a more distinct set of colors
def condense_palette(colors, threshold=30):
    condensed = []
    for color in colors:
        if all(
            sum((c1 - c2) ** 2 for c1, c2 in zip(color, other)) ** 0.5 > threshold
            for other in condensed
        ):
            condensed.append(color)
    return condensed


# Reduce the condensed palette to the requested number of colors
def simplify_palette(colors, target_count):
    condensed = condense_palette(colors)
    if len(condensed) <= target_count:
        return condensed

    if target_count == 1:
        return [condensed[0]]

    step = (len(condensed) - 1) / (target_count - 1)
    return [condensed[round(i * step)] for i in range(target_count)]


def map_pixels_to_palette(rgb_image, palette):
    palette_np = np.array(palette, dtype=np.int16)
    pixels = rgb_image.reshape(-1, 3).astype(np.int16)
    distances = np.sum((pixels[:, None, :] - palette_np[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels.reshape(rgb_image.shape[0], rgb_image.shape[1])


def build_region_groups(rgb_image, palette, edges, texture_bins=4, min_area=40):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    # Use local variance proxy as texture signal.
    texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = cv2.GaussianBlur(np.abs(texture), (5, 5), 0)

    quantiles = np.linspace(0.0, 1.0, texture_bins + 1)
    boundaries = np.quantile(texture, quantiles)
    # Ensure strictly monotonic bins for digitize.
    boundaries = np.maximum.accumulate(boundaries)
    texture_labels = np.digitize(texture, boundaries[1:-1], right=False)

    color_labels = map_pixels_to_palette(rgb_image, palette)
    joint_labels = color_labels * texture_bins + texture_labels

    edge_mask = edges.astype(np.uint8)
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)

    regions = []
    max_joint = int(joint_labels.max())
    for label in range(max_joint + 1):
        mask = ((joint_labels == label) & (edge_mask == 0)).astype(np.uint8)
        if mask.sum() < min_area:
            continue

        components, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for component_id in range(1, components):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            ys, xs = np.where(cc_labels == component_id)
            if len(xs) == 0:
                continue

            dominant_color = tuple(
                int(v) for v in np.median(rgb_image[ys, xs], axis=0).astype(np.uint8)
            )
            regions.append(
                {
                    "x": xs,
                    "y": ys,
                    "area": area,
                    "color": dominant_color,
                }
            )

    # Fallback when segmentation is too fragmented or strict.
    if not regions:
        h, w, _ = rgb_image.shape
        ys, xs = np.indices((h, w))
        regions = [
            {
                "x": xs.flatten(),
                "y": ys.flatten(),
                "area": h * w,
                "color": tuple(
                    int(v) for v in np.median(rgb_image.reshape(-1, 3), axis=0)
                ),
            }
        ]

    return regions


# Create initial population with points seeded in detected color-texture regions to improve early convergence
def create_region_seeded_population(
    population_size,
    num_points,
    target_image,
    region_groups,
    fallback_palette,
    region_bias=0.8,
):
    weighted_regions = [r for r in region_groups if r["area"] > 0]
    weights = [r["area"] for r in weighted_regions]

    chromosomes = []
    for _ in range(population_size):
        painting = VoronoiPainting(
            num_points, target_image, background_color=(128, 128, 128)
        )

        for point in painting.points:
            if random.random() < region_bias and weighted_regions:
                region = random.choices(weighted_regions, weights=weights, k=1)[0]
                idx = random.randrange(len(region["x"]))
                point.coordinates = (int(region["x"][idx]), int(region["y"][idx]))

                base_color = region["color"]
                jitter = np.random.randint(-16, 17, size=3)
                color = np.clip(np.array(base_color) + jitter, 0, 255).astype(np.uint8)
                point.color = (int(color[0]), int(color[1]), int(color[2]), 255)
            else:
                palette_color = random.choice(fallback_palette)
                point.color = (
                    int(palette_color[0]),
                    int(palette_color[1]),
                    int(palette_color[2]),
                    255,
                )

        chromosomes.append(painting)

    return chromosomes


if __name__ == "__main__":
    target_image_path = "./img/battleship.jpeg"
    checkpoint_path = "./output/"
    image_template = os.path.join(checkpoint_path, "drawing_%05d.png")
    target_image = Image.open(target_image_path).convert("RGBA")

    num_points = 250
    population_size = 250

    generation_scale = float(os.getenv("GENERATION_SCALE", "0.6"))
    early_stop_window = int(os.getenv("EARLY_STOP_WINDOW", "200"))
    min_improvement_ratio = float(os.getenv("MIN_IMPROVEMENT_RATIO", "0.0075"))

    initialColorCount = 60
    finalColorCount = 20

    # Extract color palette from target image, we will use this palette to initialize the points with colors that are present in the target image, this will help the algorithm to converge faster
    # palette = target_image.getcolors(maxcolors=1000)
    converted_img = target_image.convert(
        "P", palette=Image.ADAPTIVE, colors=initialColorCount
    )
    # converted_img.show()
    palette = converted_img.getpalette()[: initialColorCount * 3]  # Get the RGB values
    colors = [tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)]

    # Visualize the color palette as a set of colored squares
    palette_img = Image.new("RGB", (initialColorCount * 20, 20))
    draw = ImageDraw.Draw(palette_img)
    for i, color in enumerate(colors):
        draw.rectangle([i * 20, 0, (i + 1) * 20, 20], fill=color)
    # palette_img.show()
    # Condense the palette and visualize the condensed colors
    condensed_colors = simplify_palette(colors, finalColorCount)
    print(f"Original colors: {len(colors)}, Condensed colors: {len(condensed_colors)}")
    palette_img = Image.new("RGB", (finalColorCount * 20, 20))
    draw = ImageDraw.Draw(palette_img)
    for i, color in enumerate(condensed_colors):
        draw.rectangle([i * 20, 0, (i + 1) * 20, 20], fill=color)
    # palette_img.show()

    # Use Canny edges as hard separators between regions.
    target_rgb = np.array(target_image.convert("RGB"))
    edges = feature.canny(cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY), sigma=1.2)

    # Group pixels by both palette color and local texture, while avoiding edge boundaries.
    region_groups = build_region_groups(target_rgb, condensed_colors, edges)
    print(
        f"Detected {len(region_groups)} color-texture regions for seeded initialization"
    )

    # Initialize population with region-aware points/colors to improve early convergence.
    seeded_chromosomes = create_region_seeded_population(
        population_size,
        num_points,
        target_image,
        region_groups,
        condensed_colors,
        region_bias=0.85,
    )

    pop = Population(
        chromosomes=seeded_chromosomes,
        eval_function=score,
        maximize=False,
        concurrent_workers=4,
    )

    # Code to load a pickled/stored version, each 50 generation the population is written to disk
    # stored_pop = Population.load('./output/20200207-223736.187164.pkl', eval_function=score, maximize=False)
    # # Create new population from stored one, trick to get multiprocessing working after
    # pop = Population(chromosomes=[deepcopy(a) for a in stored_pop.chromosomes],
    #                  eval_function=score, maximize=False, concurrent_workers=4, generation=4550)

    print(f"Staring with {pop.concurrent_workers} workers")

    genome_duplication = (
        Evolution()
        .survive(fraction=0.025)
        .breed(
            parent_picker=pick_best_and_random,
            combiner=merge,
            population_size=population_size,
        )
        .mutate(mutate_function=mutate_painting, rate=0.05, sigma=0.5)
        .evaluate(lazy=False)
        .callback(
            print_summary, img_template=image_template, checkpoint_path=checkpoint_path
        )
    )

    evo_step_1 = (
        Evolution()
        .survive(fraction=0.025)
        .breed(
            parent_picker=pick_best_and_random,
            combiner=mate,
            population_size=population_size,
        )
        .mutate(mutate_function=mutate_painting, rate=0.05, sigma=0.5)
        .evaluate(lazy=False)
        .callback(
            print_summary, img_template=image_template, checkpoint_path=checkpoint_path
        )
    )

    evo_step_2 = (
        Evolution()
        .survive(fraction=0.025)
        .breed(
            parent_picker=pick_best_and_random,
            combiner=mate,
            population_size=population_size,
        )
        .mutate(mutate_function=mutate_painting, rate=0.03, sigma=0.4)
        .evaluate(lazy=False)
        .callback(
            print_summary, img_template=image_template, checkpoint_path=checkpoint_path
        )
    )

    evo_step_3 = (
        Evolution()
        .survive(fraction=0.025)
        .breed(
            parent_picker=pick_best_and_random,
            combiner=mate,
            population_size=population_size,
        )
        .mutate(mutate_function=mutate_painting, rate=0.005, sigma=0.4)
        .evaluate(lazy=False)
        .callback(
            print_summary, img_template=image_template, checkpoint_path=checkpoint_path
        )
    )

    shrink_step = (
        Evolution()
        .survive(n=1)
        .breed(parent_picker=pick_best, combiner=clone, population_size=population_size)
        .mutate(mutate_function=shrink_painting)
        .evaluate(lazy=False)
        .callback(
            print_summary, img_template=image_template, checkpoint_path=checkpoint_path
        )
    )

    phase_250 = scale_generations(999, generation_scale)
    phase_500_main = scale_generations(899, generation_scale)
    phase_800_main_1 = scale_generations(900, generation_scale)
    phase_800_main_2 = scale_generations(900, generation_scale)
    phase_final_refine = scale_generations(1000, generation_scale)
    shrink_generations = scale_generations(100, generation_scale)

    total_scaled_generations = (
        phase_250
        + phase_500_main
        + shrink_generations
        + phase_800_main_1
        + shrink_generations
        + phase_800_main_2
        + shrink_generations
        + phase_final_refine
        + 2
    )
    print(
        "Evolution schedule: "
        f"scale={generation_scale}, total generations={total_scaled_generations}, "
        f"early-stop window={early_stop_window}, min-improvement={min_improvement_ratio}"
    )

    # 250 points
    pop = evolve_phase_with_early_stop(
        pop,
        evo_step_1,
        phase_250,
        label="250-point exploration",
        improvement_window=early_stop_window,
        min_improvement_ratio=min_improvement_ratio,
    )
    pop = pop.evolve(genome_duplication, n=1)
    # 500 points
    pop = evolve_phase_with_early_stop(
        pop,
        evo_step_1,
        phase_500_main,
        label="500-point exploration",
        improvement_window=early_stop_window,
        min_improvement_ratio=min_improvement_ratio,
    )
    pop = pop.evolve(shrink_step, n=shrink_generations)
    pop = pop.evolve(genome_duplication, n=1)
    # 800 points
    pop = evolve_phase_with_early_stop(
        pop,
        evo_step_2,
        phase_800_main_1,
        label="800-point exploration A",
        improvement_window=early_stop_window,
        min_improvement_ratio=min_improvement_ratio,
    )
    pop = pop.evolve(shrink_step, n=shrink_generations)
    pop = evolve_phase_with_early_stop(
        pop,
        evo_step_2,
        phase_800_main_2,
        label="800-point exploration B",
        improvement_window=early_stop_window,
        min_improvement_ratio=min_improvement_ratio,
    )
    pop = pop.evolve(shrink_step, n=shrink_generations)
    pop = evolve_phase_with_early_stop(
        pop,
        evo_step_3,
        phase_final_refine,
        label="final refinement",
        improvement_window=early_stop_window,
        min_improvement_ratio=min_improvement_ratio,
    )

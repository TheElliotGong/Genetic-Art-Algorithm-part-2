from PIL import Image, ImageDraw, ImageFilter
from evol import Evolution, Population
from copy import deepcopy
from skimage import feature

import random
import os
import cv2
import numpy as np
from multiprocessing import Pool
from functools import partial

from voronoi_painting import VoronoiPainting

# Cache scaled targets per (target identity, size, scale) to avoid rebuilding arrays.
_SCALED_TARGET_CACHE = {}


def score(x: VoronoiPainting) -> float:
    # Approximate fitness: render at reduced scale and compute vectorized MSE
    global FITNESS_SCALE, _SCALED_TARGET_CACHE
    try:
        scale = FITNESS_SCALE
    except NameError:
        scale = float(os.getenv("FITNESS_SCALE", "1.0"))
        FITNESS_SCALE = scale

    target_image = x.target_image

    # Render the candidate at the fitness scale
    src_img = x.draw(scale=scale).convert("RGB")
    src_np = np.array(src_img, dtype=np.float32)

    # Lazily prepare the scaled target (cached for all evaluations)
    cache_key = (id(target_image), target_image.size, float(scale))
    if cache_key not in _SCALED_TARGET_CACHE:
        if scale == 1.0:
            tgt = target_image.convert("RGB")
        else:
            tgt = target_image.resize(
                (int(target_image.width * scale), int(target_image.height * scale)),
                resample=Image.BILINEAR,
            ).convert("RGB")
        _SCALED_TARGET_CACHE[cache_key] = np.array(tgt, dtype=np.float32)
    tgt_np = _SCALED_TARGET_CACHE[cache_key]

    # Defensive resize when dimensions differ due to rounding.
    if src_np.shape != tgt_np.shape:
        src_h, src_w = src_np.shape[:2]
        tgt_np = np.array(
            target_image.resize((src_w, src_h), resample=Image.BILINEAR).convert("RGB"),
            dtype=np.float32,
        )

    # Mean squared error as fitness (lower is better)
    diff = src_np - tgt_np
    mse = float(np.mean(np.square(diff)))
    print(".", end="", flush=True)
    return mse


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
    pop, img_template="output%d.png", checkpoint_path="output", output_scale=1.0
) -> Population:
    avg_fitness = sum([i.fitness for i in pop.individuals]) / len(pop.individuals)
    chromosome_length = pop.individuals[0].chromosome.num_points
    print(
        "\nCurrent generation %d, best score %f, pop. avg. %f. Chromosome length %d"
        % (pop.generation, pop.current_best.fitness, avg_fitness, chromosome_length)
    )

    # Save only on generation 1 and every 50 generations thereafter
    if pop.generation == 1 or (pop.generation % 50) == 0:
        img = pop.current_best.chromosome.draw(scale=output_scale)
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
    output_scale=1.0,
):
    weighted_regions = [r for r in region_groups if r["area"] > 0]
    weights = [r["area"] for r in weighted_regions]

    chromosomes = []
    for _ in range(population_size):
        painting = VoronoiPainting(
            num_points,
            target_image,
            background_color=(128, 128, 128),
            output_scale=output_scale,
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


def split_image_into_grid(image: Image, rows=3, cols=3, overlap_pixels=20):
    width, height = image.size
    xs = [int(round(i * width / cols)) for i in range(cols + 1)]
    ys = [int(round(i * height / rows)) for i in range(rows + 1)]

    tiles = []
    for row in range(rows):
        for col in range(cols):
            # Crop bounds for the actual tile (no overlap)
            left, right = xs[col], xs[col + 1]
            top, bottom = ys[row], ys[row + 1]

            # Expand bounds with overlap margin (clamped to image bounds)
            expand_left = max(0, left - overlap_pixels) if col > 0 else left
            expand_right = (
                min(width, right + overlap_pixels) if col < cols - 1 else right
            )
            expand_top = max(0, top - overlap_pixels) if row > 0 else top
            expand_bottom = (
                min(height, bottom + overlap_pixels) if row < rows - 1 else bottom
            )

            expand_bbox = (expand_left, expand_top, expand_right, expand_bottom)
            tiles.append(
                {
                    "row": row,
                    "col": col,
                    "bbox": (left, top, right, bottom),  # Core tile bounds (no overlap)
                    "expand_bbox": expand_bbox,  # Expanded bounds for rendering
                    "image": image.crop(expand_bbox).convert("RGBA"),
                    "overlap_margins": {
                        "left": left - expand_left,
                        "top": top - expand_top,
                        "right": expand_right - right,
                        "bottom": expand_bottom - bottom,
                    },
                }
            )
    return tiles


def feather_blend_tiles(
    canvas, patch, top_left, overlap_margins, output_scale=1.0, feather_width=20
):
    """Blend patch onto canvas with feathering at overlap zones."""
    x, y = top_left
    h, w = patch.shape[:2]
    roi = canvas[y : y + h, x : x + w].astype(np.float32)
    src = patch.astype(np.float32)

    # Create mask with soft edges at overlap zones
    mask = np.ones((h, w), dtype=np.float32)

    left_margin = int(overlap_margins.get("left", 0) * output_scale)
    top_margin = int(overlap_margins.get("top", 0) * output_scale)
    right_margin = int(overlap_margins.get("right", 0) * output_scale)
    bottom_margin = int(overlap_margins.get("bottom", 0) * output_scale)

    # Feather out at margins
    if left_margin > 0:
        for i in range(left_margin):
            mask[:, i] *= i / max(1, left_margin)
    if top_margin > 0:
        for i in range(top_margin):
            mask[i, :] *= i / max(1, top_margin)
    if right_margin > 0:
        for i in range(right_margin):
            mask[:, -(i + 1)] *= i / max(1, right_margin)
    if bottom_margin > 0:
        for i in range(bottom_margin):
            mask[-(i + 1), :] *= i / max(1, bottom_margin)

    mask = np.expand_dims(mask, axis=2)
    out = src * mask + roi * (1 - mask)
    canvas[y : y + h, x : x + w] = np.clip(out, 0, 255).astype(np.uint8)
    return canvas


def stitch_tile_results(
    tile_results, original_size, output_scale=1.0, use_feather_blend=True
):
    """Stitch tile results with optional feather blending at overlaps."""
    full_width, full_height = original_size
    canvas_cv = np.zeros(
        (int(full_height * output_scale), int(full_width * output_scale), 4),
        dtype=np.uint8,
    )

    # Sort tiles by row, col to paste in order
    sorted_tiles = sorted(tile_results, key=lambda t: (t["row"], t["col"]))

    for tile in sorted_tiles:
        left, top, _, _ = tile["bbox"]
        paste_x = int(round(left * output_scale))
        paste_y = int(round(top * output_scale))

        # Convert PIL RGBA to numpy BGRA for OpenCV
        render_pil = tile["render"]
        render_rgb = np.array(render_pil.convert("RGB"), dtype=np.uint8)
        render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)

        if use_feather_blend and ("overlap_margins" in tile):
            # Add alpha channel for blending
            render_bgra = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2BGRA)
            canvas_bgra = canvas_cv
            feather_blend_tiles(
                canvas_bgra,
                render_bgra,
                (paste_x, paste_y),
                tile["overlap_margins"],
                output_scale=output_scale,
            )
        else:
            # Simple paste
            h, w = render_bgr.shape[:2]
            canvas_cv[paste_y : paste_y + h, paste_x : paste_x + w] = cv2.cvtColor(
                render_bgr, cv2.COLOR_BGR2BGRA
            )

    # Convert back to PIL RGBA
    canvas_bgr = cv2.cvtColor(canvas_cv, cv2.COLOR_BGRA2BGR)
    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(canvas_rgb, "RGB")


def run_evolution(
    target_image,
    checkpoint_path,
    image_template,
    num_points,
    population_size,
    generation_scale,
    early_stop_window,
    min_improvement_ratio,
    output_scale,
):
    initialColorCount = 60
    finalColorCount = 20

    # Extract palette and simplify colors for region-aware seeding.
    converted_img = target_image.convert(
        "P", palette=Image.ADAPTIVE, colors=initialColorCount
    )
    palette = converted_img.getpalette()[: initialColorCount * 3]
    colors = [tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)]
    condensed_colors = simplify_palette(colors, finalColorCount)
    print(f"Original colors: {len(colors)}, Condensed colors: {len(condensed_colors)}")

    # Use Canny edges as separators between color-texture regions.
    target_rgb = np.array(target_image.convert("RGB"))
    edges = feature.canny(cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY), sigma=1.2)
    region_groups = build_region_groups(target_rgb, condensed_colors, edges)
    print(
        f"Detected {len(region_groups)} color-texture regions for seeded initialization"
    )

    seeded_chromosomes = create_region_seeded_population(
        population_size,
        num_points,
        target_image,
        region_groups,
        condensed_colors,
        region_bias=0.85,
        output_scale=output_scale,
    )

    pop = Population(
        chromosomes=seeded_chromosomes,
        eval_function=score,
        maximize=False,
        concurrent_workers=6,
    )

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
            print_summary,
            img_template=image_template,
            checkpoint_path=checkpoint_path,
            output_scale=output_scale,
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
            print_summary,
            img_template=image_template,
            checkpoint_path=checkpoint_path,
            output_scale=output_scale,
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
            print_summary,
            img_template=image_template,
            checkpoint_path=checkpoint_path,
            output_scale=output_scale,
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
            print_summary,
            img_template=image_template,
            checkpoint_path=checkpoint_path,
            output_scale=output_scale,
        )
    )

    shrink_step = (
        Evolution()
        .survive(n=1)
        .breed(parent_picker=pick_best, combiner=clone, population_size=population_size)
        .mutate(mutate_function=shrink_painting)
        .evaluate(lazy=False)
        .callback(
            print_summary,
            img_template=image_template,
            checkpoint_path=checkpoint_path,
            output_scale=output_scale,
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
        f"early-stop window={early_stop_window}, min-improvement={min_improvement_ratio}, "
        f"output-scale={output_scale}"
    )

    pop = evolve_phase_with_early_stop(
        pop,
        evo_step_1,
        phase_250,
        label="250-point exploration",
        improvement_window=early_stop_window,
        min_improvement_ratio=min_improvement_ratio,
    )
    pop = pop.evolve(genome_duplication, n=1)
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

    return pop.current_best.chromosome.draw(scale=output_scale)


def evolve_tile_worker(tile_spec):
    """Worker function for parallel tile evolution. Returns tile with evolved render."""
    tile = tile_spec["tile"]
    args = tile_spec["args"]

    row, col = tile["row"], tile["col"]
    checkpoint_path = args["checkpoint_path"]

    tile_checkpoint_path = os.path.join(checkpoint_path, f"tile_r{row}_c{col}")
    os.makedirs(tile_checkpoint_path, exist_ok=True)
    tile_template = os.path.join(tile_checkpoint_path, "drawing_%05d.png")

    print(f"[Worker] Starting tile ({row}, {col}) with size={tile['image'].size}")

    best_tile_render = run_evolution(
        target_image=tile["image"],
        checkpoint_path=tile_checkpoint_path,
        image_template=tile_template,
        num_points=args["num_points"],
        population_size=args["population_size"],
        generation_scale=args["generation_scale"],
        early_stop_window=args["early_stop_window"],
        min_improvement_ratio=args["min_improvement_ratio"],
        output_scale=args["output_scale"],
    )

    print(f"[Worker] Finished tile ({row}, {col})")

    return {
        "row": row,
        "col": col,
        "bbox": tile["bbox"],
        "overlap_margins": tile.get("overlap_margins", {}),
        "render": best_tile_render,
    }


if __name__ == "__main__":
    target_image_path = "./img/girl_half.jpg"
    checkpoint_path = "./output/girl"
    os.makedirs(checkpoint_path, exist_ok=True)
    image_template = os.path.join(checkpoint_path, "drawing_%05d.png")
    target_image = Image.open(target_image_path).convert("RGBA")

    num_points = 250
    population_size = 250

    generation_scale = float(os.getenv("GENERATION_SCALE", "0.6"))
    early_stop_window = int(os.getenv("EARLY_STOP_WINDOW", "200"))
    min_improvement_ratio = float(os.getenv("MIN_IMPROVEMENT_RATIO", "0.01"))
    output_scale = float(os.getenv("OUTPUT_SCALE", "2.0"))
    use_divide_and_conquer = os.getenv(
        "USE_DIVIDE_AND_CONQUER", "0"
    ).strip().lower() in ("1", "true", "yes", "on")

    if use_divide_and_conquer:
        grid_rows = int(os.getenv("GRID_ROWS", "3"))
        grid_cols = int(os.getenv("GRID_COLS", "3"))
        tile_points = int(
            os.getenv(
                "TILE_NUM_POINTS",
                str(max(25, int(round(num_points / max(1, grid_rows * grid_cols))))),
            )
        )
        tile_population_size = int(
            os.getenv("TILE_POPULATION_SIZE", str(population_size))
        )
        tile_generation_scale = float(
            os.getenv("TILE_GENERATION_SCALE", str(generation_scale))
        )
        overlap_pixels = int(os.getenv("TILE_OVERLAP_PIXELS", "20"))
        num_workers = int(os.getenv("NUM_WORKERS", "6"))
        use_feather_blend = os.getenv("USE_FEATHER_BLEND", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        print(
            "Running divide-and-conquer evolution: "
            f"grid={grid_rows}x{grid_cols}, tile_points={tile_points}, "
            f"tile_population_size={tile_population_size}, tile_generation_scale={tile_generation_scale}, "
            f"overlap_pixels={overlap_pixels}, num_workers={num_workers}, use_feather_blend={use_feather_blend}"
        )

        tiles = split_image_into_grid(
            target_image, rows=grid_rows, cols=grid_cols, overlap_pixels=overlap_pixels
        )

        # Prepare worker args
        worker_args = {
            "checkpoint_path": checkpoint_path,
            "num_points": tile_points,
            "population_size": tile_population_size,
            "generation_scale": tile_generation_scale,
            "early_stop_window": early_stop_window,
            "min_improvement_ratio": min_improvement_ratio,
            "output_scale": output_scale,
        }

        # Create worker task specs
        worker_tasks = [{"tile": tile, "args": worker_args} for tile in tiles]

        # Run in parallel or sequentially
        if num_workers > 1:
            print(
                f"Evolving {len(tiles)} tiles in parallel with {num_workers} workers..."
            )
            with Pool(processes=num_workers) as pool:
                tile_results = pool.map(evolve_tile_worker, worker_tasks)
        else:
            print(f"Evolving {len(tiles)} tiles sequentially...")
            tile_results = [evolve_tile_worker(task) for task in worker_tasks]

        stitched = stitch_tile_results(
            tile_results,
            original_size=target_image.size,
            output_scale=output_scale,
            use_feather_blend=use_feather_blend,
        )
        stitched_path = os.path.join(checkpoint_path, "drawing_divide_and_conquer.png")
        stitched.save(stitched_path, "PNG")
        print(f"Saved stitched divide-and-conquer result to {stitched_path}")
    else:
        run_evolution(
            target_image=target_image,
            checkpoint_path=checkpoint_path,
            image_template=image_template,
            num_points=num_points,
            population_size=population_size,
            generation_scale=generation_scale,
            early_stop_window=early_stop_window,
            min_improvement_ratio=min_improvement_ratio,
            output_scale=output_scale,
        )

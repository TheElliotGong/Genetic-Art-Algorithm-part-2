from PIL import Image
from evol import Evolution, Population
from datetime import datetime
from skimage import feature

import os
import cv2
import numpy as np

from voronoi_painting import VoronoiPainting
from evolve_voronoi import (
    score,
    pick_best_and_random,
    mutate_painting,
    mate,
    merge,
    simplify_palette,
    build_region_groups,
    create_region_seeded_population,
)


def split_into_tiles(target_image, n_rows=3, n_cols=3):
    w, h = target_image.size
    col_edges = [(w * i) // n_cols for i in range(n_cols + 1)]
    row_edges = [(h * i) // n_rows for i in range(n_rows + 1)]
    tiles = []
    for ty in range(n_rows):
        for tx in range(n_cols):
            x0, x1 = col_edges[tx], col_edges[tx + 1]
            y0, y1 = row_edges[ty], row_edges[ty + 1]
            tile = target_image.crop((x0, y0, x1, y1))
            tiles.append((x0, y0, tile))
    return tiles


def evolve_tile(
    target_tile,
    tile_id,
    output_dir,
    points_initial=50,
    population_size=100,
    workers=4,
    gens_phase1=999,
    gens_phase2=1000,
):
    tile_w, tile_h = target_tile.size

    initial_color_count = 30
    final_color_count = 12
    converted = target_tile.convert("P", palette=Image.ADAPTIVE, colors=initial_color_count)
    palette = converted.getpalette()[: initial_color_count * 3]
    colors = [tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)]
    condensed = simplify_palette(colors, final_color_count)

    target_rgb = np.array(target_tile.convert("RGB"))
    edges = feature.canny(cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY), sigma=1.2)
    region_groups = build_region_groups(target_rgb, condensed, edges, min_area=10)

    seeded = create_region_seeded_population(
        population_size,
        points_initial,
        target_tile,
        region_groups,
        condensed,
        region_bias=0.85,
    )

    pop = Population(
        chromosomes=seeded,
        eval_function=score,
        maximize=False,
        concurrent_workers=workers,
    )

    print(
        f"  [Tile {tile_id}] {tile_w}x{tile_h}px, {len(region_groups)} regions, "
        f"{points_initial} points -> {points_initial * 2} after duplication"
    )

    img_template = os.path.join(output_dir, f"tile_{tile_id}_gen_%05d.png")

    def tile_callback(p):
        if p.generation % 100 == 0:
            print(
                f"  [Tile {tile_id}] gen {p.generation}, "
                f"best {p.current_best.fitness:.0f}, "
                f"points {p.individuals[0].chromosome.num_points}"
            )
        if p.generation % 500 == 0:
            img = p.current_best.chromosome.draw(scale=1)
            img.save(img_template % p.generation, "PNG")
        return p

    evo_step_1 = (
        Evolution()
        .survive(fraction=0.025)
        .breed(parent_picker=pick_best_and_random, combiner=mate, population_size=population_size)
        .mutate(mutate_function=mutate_painting, rate=0.05, sigma=0.5)
        .evaluate(lazy=False)
        .callback(tile_callback)
    )

    duplication = (
        Evolution()
        .survive(fraction=0.025)
        .breed(parent_picker=pick_best_and_random, combiner=merge, population_size=population_size)
        .mutate(mutate_function=mutate_painting, rate=0.05, sigma=0.5)
        .evaluate(lazy=False)
        .callback(tile_callback)
    )

    evo_step_2 = (
        Evolution()
        .survive(fraction=0.025)
        .breed(parent_picker=pick_best_and_random, combiner=mate, population_size=population_size)
        .mutate(mutate_function=mutate_painting, rate=0.03, sigma=0.4)
        .evaluate(lazy=False)
        .callback(tile_callback)
    )

    pop = pop.evolve(evo_step_1, n=gens_phase1)
    pop = pop.evolve(duplication, n=1)
    pop = pop.evolve(evo_step_2, n=gens_phase2)

    final_img = pop.current_best.chromosome.draw(scale=1)
    final_img.save(img_template % pop.generation, "PNG")

    return pop.current_best.chromosome


def stitch(tile_results, full_size):
    canvas = Image.new("RGB", full_size, (0, 0, 0))
    for x_offset, y_offset, painting in tile_results:
        tile_img = painting.draw(scale=1).convert("RGB")
        canvas.paste(tile_img, (x_offset, y_offset))
    return canvas


if __name__ == "__main__":
    target_image_path = "./img/car.jpeg"
    target_image = Image.open(target_image_path).convert("RGBA")

    n_rows, n_cols = 3, 3
    tiles = split_into_tiles(target_image, n_rows, n_cols)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("./output", f"tiled-{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Target: {target_image.size}")
    print(f"Splitting into {n_rows}x{n_cols} = {len(tiles)} tiles")
    print(f"Output: {output_dir}\n")

    target_image.save(os.path.join(output_dir, "_target.png"))

    tile_results = []
    start = datetime.now()
    for i, (x_offset, y_offset, tile) in enumerate(tiles):
        tile_id = f"{i:02d}"
        print(
            f"=== Tile {i + 1}/{len(tiles)} (id {tile_id}) "
            f"at ({x_offset},{y_offset}), size {tile.size} ==="
        )
        tile_start = datetime.now()
        best = evolve_tile(
            tile,
            tile_id=tile_id,
            output_dir=output_dir,
            points_initial=50,
            population_size=100,
            workers=4,
            gens_phase1=999,
            gens_phase2=1000,
        )
        tile_results.append((x_offset, y_offset, best))
        elapsed = (datetime.now() - tile_start).total_seconds()
        print(f"  [Tile {tile_id}] done in {elapsed:.0f}s\n")

        partial = stitch(tile_results, target_image.size)
        partial.save(os.path.join(output_dir, f"_partial_after_{tile_id}.png"))

    final = stitch(tile_results, target_image.size)
    final.save(os.path.join(output_dir, "_final.png"))
    total_elapsed = (datetime.now() - start).total_seconds()
    print(f"Done. Total time: {total_elapsed / 60:.1f} minutes")
    print(f"Final: {os.path.join(output_dir, '_final.png')}")
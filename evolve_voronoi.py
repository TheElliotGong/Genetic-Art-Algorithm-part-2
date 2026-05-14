from PIL import Image, ImageDraw, ImageFilter
from evol import Evolution, Population
from copy import deepcopy
from datetime import datetime
from skimage import feature

import random
import os
import cv2
import numpy as np


from voronoi_painting import VoronoiPainting


def score(x: VoronoiPainting) -> float:
    """
    The score function, it calculates the difference between the target image and the current painting, the lower the score the better,
    we will use the mean squared error as the score, we can also add a penalty for having too many points in the painting to encourage simpler paintings.
    """
    current_score = x.image_diff(x.target_image)
    print(".", end="", flush=True)
    return current_score


def pick_best_and_random(pop, maximize=False):
    """Pick the best individual from the population and a random individual from the population to be the parents for the next generation.
    :param pop: The population to pick from.
    :param maximize: Whether to maximize or minimize the score. If True, the best individual will be the one with the highest score, otherwise it will be the one with the lowest score.
    """
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
    """Pick the best individual from the population to be the parent for the next generation.
    :param pop: The population to pick from.
    :param maximize: Whether to maximize or minimize the score. If True, the best individual will be the one with the highest score, otherwise it will be the one with the lowest score.
    """
    evaluated_individuals = tuple(filter(lambda x: x.fitness is not None, pop))
    if len(evaluated_individuals) > 0:
        mom = max(
            evaluated_individuals, key=lambda x: x.fitness if maximize else -x.fitness
        )
    else:
        mom = random.choice(pop)
    return mom


def pick_random(pop):
    """Pick two random individuals from the population to be the parents for the next generation.
    :param pop: The population to pick from."""
    mom = random.choice(pop)
    dad = random.choice(pop)
    return mom, dad


def mutate_painting(x: VoronoiPainting, rate=0.04, sigma=1) -> VoronoiPainting:
    """Mutate the painting by mutating a percentage of the points in the painting based on the given mutation rate and sigma for mutation strength.
    :param x: The painting to mutate.
    :param rate: The percentage of points to mutate (between 0 and 1).
    :param sigma: The standard deviation for the mutation strength. Higher values will result in more significant mutations.
    """
    x.mutate_points(rate=rate, sigma=sigma)
    return deepcopy(x)


def shrink_painting(x: VoronoiPainting) -> VoronoiPainting:
    """Shrink the painting by removing a random point.
    :param x: The painting to shrink.
    """
    x.shrink_points()
    return deepcopy(x)


def mate(mom: VoronoiPainting, dad: VoronoiPainting):
    """Mate two paintings by combining their points to create a child painting. We only save one of the children to keep the population size constant,
       we can also add some mutation to the child painting to encourage diversity in the population.
    :param mom: The first parent painting.
    :param dad: The second parent painting."""
    child_a, child_b = VoronoiPainting.mate(mom, dad)

    return deepcopy(child_a)


def clone(mom: VoronoiPainting):
    return deepcopy(mom)


def merge(mom: VoronoiPainting, dad: VoronoiPainting):
    """Merge two paintings by combining their points to create a child painting with more points than either parent, this is used for genome duplication to increase the number of points in the painting."""
    child_a = VoronoiPainting.merge(mom, dad)

    return deepcopy(child_a)


def print_summary(
    pop, img_template="output%d.png", checkpoint_path="output"
) -> Population:
    """Print a summary of the current population, including the generation number, best score, average score, and chromosome length.
    Also save an image of the best painting every 10 generations and checkpoint the population every 50 generations.
    """
    avg_fitness = sum([i.fitness for i in pop.individuals]) / len(pop.individuals)
    chromosome_length = pop.individuals[0].chromosome.num_points
    print(
        "\nCurrent generation %d, best score %f, pop. avg. %f. Chromosome length %d"
        % (pop.generation, pop.current_best.fitness, avg_fitness, chromosome_length)
    )
    if pop.generation % 10 == 0:
        img = pop.current_best.chromosome.draw(scale=3)
        img.save(img_template % pop.generation, "PNG")

    if pop.generation % 50 == 0:
        pop.checkpoint(target=checkpoint_path, method="pickle")

    return pop


def condense_palette(colors, threshold=30):
    """Condense the palette by removing colors that are too similar to each other based on a distance threshold in RGB space.
    :param colors: A list of RGB color tuples to condense.
    :param threshold: The distance threshold in RGB space for considering colors as similar.
    Colors that are closer than this threshold will be considered similar and only one of them will be kept in the condensed palette.
    """
    condensed = []
    for color in colors:
        if all(
            sum((c1 - c2) ** 2 for c1, c2 in zip(color, other)) ** 0.5 > threshold
            for other in condensed
        ):
            condensed.append(color)
    return condensed


def simplify_palette(colors, target_count):
    """Simplify the palette by reducing the number of colors to the target count.
    :param colors: A list of RGB color tuples to simplify.
    :param target_count: The desired number of colors in the simplified palette."""
    condensed = condense_palette(colors)
    if len(condensed) <= target_count:
        return condensed

    if target_count == 1:
        return [condensed[0]]

    step = (len(condensed) - 1) / (target_count - 1)
    return [condensed[round(i * step)] for i in range(target_count)]


def map_pixels_to_palette(rgb_image, palette):
    """Map each pixel in the RGB image to the closest color in the palette and return a 2D array of color indices corresponding to the palette.
    :param rgb_image: A 3D NumPy array representing the RGB image (height x width x 3).
    :param palette: A list of RGB color tuples representing the palette to map to."""
    palette_np = np.array(palette, dtype=np.int16)
    pixels = rgb_image.reshape(-1, 3).astype(np.int16)
    distances = np.sum((pixels[:, None, :] - palette_np[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels.reshape(rgb_image.shape[0], rgb_image.shape[1])


def build_region_groups(rgb_image, palette, edges, texture_bins=4, min_area=40):
    """Build region groups by segmenting the image based on both color and local texture, while respecting edge boundaries to avoid crossing strong edges.
    :param rgb_image: A 3D NumPy array representing the RGB image (height x width x 3).
    :param palette: A list of RGB color tuples representing the color palette to use for grouping.
    :param edges: A 2D binary NumPy array representing edge locations in the image (height x width), where 1 indicates an edge and 0 indicates no edge.
    :param texture_bins: The number of bins to use for local texture quantization. Higher values will create more texture-based groups.
    :param min_area: The minimum area (in pixels) for a region to be considered valid. Regions smaller than this will be discarded.
    """
    # Compute a simple texture measure using the Laplacian of the grayscale image, then quantize the texture into discrete bins to combine with color-based grouping.
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = cv2.GaussianBlur(np.abs(texture), (5, 5), 0)
    # Quantize texture into bins based on quantiles to ensure a more balanced distribution of texture groups, this allows us to combine texture information with color-based grouping while respecting edge boundaries.
    quantiles = np.linspace(0.0, 1.0, texture_bins + 1)
    boundaries = np.quantile(texture, quantiles)
    # Ensure strictly monotonic bins for digitize.
    boundaries = np.maximum.accumulate(boundaries)
    texture_labels = np.digitize(texture, boundaries[1:-1], right=False)
    # Create the labels for grouping by combining the color labels from the palette mapping and the texture labels, this allows us to create groups that are based on both color and local texture characteristics of the image.
    color_labels = map_pixels_to_palette(rgb_image, palette)
    joint_labels = color_labels * texture_bins + texture_labels
    # Create a mask for edges to ensure that we do not group pixels across strong edges, this helps to preserve important boundaries in the image when creating the region groups.
    edge_mask = edges.astype(np.uint8)
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
    # Iterate through each unique combination of color and texture labels, create a mask for that group while excluding edge pixels, and
    # then find connected components within that mask to identify distinct regions. We filter out small regions based on the min_area parameter and calculate the
    #  dominant color for each valid region to create a list of region groups that can be used for seeded initialization of the population.
    regions = []
    max_joint = int(joint_labels.max())
    # For each unique combination of color and texture labels, we create a mask that includes pixels with that combination while excluding edge pixels.
    # We then find connected components within that mask to identify distinct regions, filtering out small regions based on the min_area parameter.
    # For each valid region, we calculate the dominant color and store the pixel coordinates, area, and dominant color in the regions list for use in seeded initialization of the population.
    for label in range(max_joint + 1):
        # For each unique combination of color and texture labels, we create a mask that includes pixels with that combination while excluding edge pixels.
        mask = ((joint_labels == label) & (edge_mask == 0)).astype(np.uint8)
        if mask.sum() < min_area:
            continue
        # Create a mask for the current group and find connected components to identify distinct regions, filtering out small regions based on the min_area parameter.
        components, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        # Iterate through the connected components, filter out small regions based on the min_area parameter, and calculate the dominant color for each valid region to
        #  create a list of region groups for seeded initialization.
        for component_id in range(1, components):
            # Find the area of the connected component and filter out small regions based on the min_area parameter, this helps to ensure that we only consider regions
            #  that are large enough to be meaningful for seeded initialization.
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            # Calculate the dominant color for the region by taking the median color of the pixels in the region, this provides a representative color for the region
            # that can be used for seeding points in the population.
            ys, xs = np.where(cc_labels == component_id)
            if len(xs) == 0:
                continue

            dominant_color = tuple(
                int(v) for v in np.median(rgb_image[ys, xs], axis=0).astype(np.uint8)
            )
            # Store the pixel coordinates, area, and dominant color of the region in the regions list for use in seeded initialization of the population.
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


def create_region_seeded_population(
    population_size,
    num_points,
    target_image,
    region_groups,
    fallback_palette,
    region_bias=0.8,
):
    """Create an initial population of Voronoi paintings with points seeded in detected color-texture regions to improve early convergence.
    :param population_size: The number of paintings in the population.
    :param num_points: The number of points to use in each painting.
    :param target_image: The target image that the paintings are trying to evolve towards, used for determining the image dimensions and color palette.
    :param region_groups: A list of region groups, where each group is a dictionary containing 'x', 'y', 'area', and 'color' keys that describe the pixel coordinates, area, and dominant color of the region.
    :param fallback_palette: A list of RGB color tuples to use for points that are not seeded in any region, this ensures that all points have a color even if the region-based seeding is not used for some points.
    :param region_bias: The probability (between 0 and 1) that a point will be seeded in a region as opposed to being placed randomly.
    Higher values will result in more points being seeded in regions, which can improve convergence but may reduce diversity in the initial population.
    """
    # Filter regions to only those with positive area and prepare weights for random selection based on area size,
    # this allows us to bias point placement towards larger regions which may be more visually significant in the target image.
    weighted_regions = [r for r in region_groups if r["area"] > 0]
    weights = [r["area"] for r in weighted_regions]

    chromosomes = []
    # Create individual paintings by seeding points in regions based on the specified bias, and assigning colors based on the region's dominant color with some random jitter for variation.
    for _ in range(population_size):
        painting = VoronoiPainting(
            num_points, target_image, background_color=(128, 128, 128)
        )
        # For each individual's chromosome, we iterate through its points and decide whether to seed it in a region or place it randomly based on the region_bias.
        # If seeding in a region, we randomly select a region weighted by area size, then randomly select a pixel within that region for the point's coordinates.
        # The point's color is set to the region's dominant color with some random jitter added to introduce variation.
        # If not seeding in a region, the point's color is randomly chosen from the fallback palette.
        for point in painting.points:
            # With a probability defined by region_bias, we attempt to seed the point in a region. If there are valid regions available,
            # we randomly select one weighted by area size, then randomly select a pixel from that region for the point's coordinates.
            # The point's color is set to the region's dominant color with some random jitter added to introduce variation.
            # If we do not seed in a region (either due to the bias or lack of valid regions), we assign the point a random color from the fallback palette.
            if random.random() < region_bias and weighted_regions:
                region = random.choices(weighted_regions, weights=weights, k=1)[0]
                idx = random.randrange(len(region["x"]))
                point.coordinates = (int(region["x"][idx]), int(region["y"][idx]))

                base_color = region["color"]
                jitter = np.random.randint(-16, 17, size=3)
                color = np.clip(np.array(base_color) + jitter, 0, 255).astype(np.uint8)
                point.color = (int(color[0]), int(color[1]), int(color[2]), 255)
            # Otherwise, we assign the point a random color from the fallback palette to ensure that all points have a color even if they are not seeded in a region.
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
    target_image_path = "./img/car.jpeg"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join("./output", run_id)
    os.makedirs(checkpoint_path, exist_ok=True)
    image_template = os.path.join(checkpoint_path, "drawing_%05d.png")
    target_image = Image.open(target_image_path).convert("RGBA")
    # Variables to control the evolution process, including the number of points in each painting, the population size, and the number of generations for each phase of evolution.
    num_points = 250
    population_size = 250

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
    # Run the evolution in two phases: first with mating and mutation, then with duplication and mutation, and finally with more mating and mutation.
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

    # 250 points
    pop = pop.evolve(evo_step_1, n=999)
    pop = pop.evolve(genome_duplication, n=1)
    # 500 points
    pop = pop.evolve(evo_step_1, n=899)
    pop = pop.evolve(shrink_step, n=100)
    pop = pop.evolve(genome_duplication, n=1)
    # 800 points
    pop = pop.evolve(evo_step_2, n=900)
    pop = pop.evolve(shrink_step, n=100)
    pop = pop.evolve(evo_step_2, n=900)
    pop = pop.evolve(shrink_step, n=100)
    pop = pop.evolve(evo_step_3, n=1000)

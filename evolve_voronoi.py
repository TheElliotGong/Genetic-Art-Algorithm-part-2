from PIL import Image, ImageDraw
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
    print(".", end='', flush=True)
    return current_score


def pick_best_and_random(pop, maximize=False):
    evaluated_individuals = tuple(filter(lambda x: x.fitness is not None, pop))
    if len(evaluated_individuals) > 0:
        mom = max(evaluated_individuals, key=lambda x: x.fitness if maximize else -x.fitness)
    else:
        mom = random.choice(pop)
    dad = random.choice(pop)
    return mom, dad


def pick_best(pop, maximize=False):
    evaluated_individuals = tuple(filter(lambda x: x.fitness is not None, pop))
    if len(evaluated_individuals) > 0:
        mom = max(evaluated_individuals, key=lambda x: x.fitness if maximize else -x.fitness)
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


def print_summary(pop, img_template="output%d.png", checkpoint_path="output") -> Population:
    avg_fitness = sum([i.fitness for i in pop.individuals])/len(pop.individuals)
    chromosome_length = pop.individuals[0].chromosome.num_points
    print("\nCurrent generation %d, best score %f, pop. avg. %f. Chromosome length %d" % (pop.generation,
                                                                                          pop.current_best.fitness,
                                                                                          avg_fitness,
                                                                                          chromosome_length))
    img = pop.current_best.chromosome.draw(scale=3)
    img.save(img_template % pop.generation, 'PNG')

    if pop.generation % 50 == 0:
        pop.checkpoint(target=checkpoint_path, method='pickle')

    return pop

# Condense similar colors in the palette to create a more distinct set of colors
def condense_palette(colors, threshold=30):
    condensed = []
    for color in colors:
        if all(sum((c1 - c2) ** 2 for c1, c2 in zip(color, other)) ** 0.5 > threshold for other in condensed):
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


if __name__ == "__main__":
    target_image_path = "./img/girl_with_pearl_earring_half.jpg"
    checkpoint_path = "./output/"
    image_template = os.path.join(checkpoint_path, "drawing_%05d.png")
    target_image = Image.open(target_image_path).convert('RGBA')

    num_points = 250
    population_size = 250

    initialColorCount = 60
    finalColorCount = 20

    #Extract color palette from target image, we will use this palette to initialize the points with colors that are present in the target image, this will help the algorithm to converge faster
    # palette = target_image.getcolors(maxcolors=1000)
    converted_img = target_image.convert("P", palette=Image.ADAPTIVE, colors=initialColorCount)
    # converted_img.show()
    palette = converted_img.getpalette()[:initialColorCount * 3]  # Get the RGB values
    colors = [tuple(palette[i:i + 3]) for i in range(0, len(palette), 3)]

    # Visualize the color palette as a set of colored squares
    palette_img = Image.new("RGB", (initialColorCount * 20, 20))
    draw = ImageDraw.Draw(palette_img)
    for i, color in enumerate(colors):
        draw.rectangle([i * 20, 0, (i + 1) * 20, 20], fill=color)
    palette_img.show()
    # Condense the palette and visualize the condensed colors
    condensed_colors = simplify_palette(colors, finalColorCount)
    print(f"Original colors: {len(colors)}, Condensed colors: {len(condensed_colors)}")
    palette_img = Image.new("RGB", (finalColorCount * 20, 20))
    draw = ImageDraw.Draw(palette_img)
    for i, color in enumerate(condensed_colors):
        draw.rectangle([i * 20, 0, (i + 1) * 20, 20], fill=color)
    palette_img.show()
    pop = Population(chromosomes=[VoronoiPainting(num_points, target_image, background_color=(128, 128, 128)) for _ in
                                  range(population_size)],
                     eval_function=score, maximize=False, concurrent_workers=4)

    # Code to load a pickled/stored version, each 50 generation the population is written to disk
    # stored_pop = Population.load('./output/20200207-223736.187164.pkl', eval_function=score, maximize=False)
    # # Create new population from stored one, trick to get multiprocessing working after
    # pop = Population(chromosomes=[deepcopy(a) for a in stored_pop.chromosomes],
    #                  eval_function=score, maximize=False, concurrent_workers=4, generation=4550)

    print(f"Staring with {pop.concurrent_workers} workers")

    genome_duplication = (Evolution()
                          .survive(fraction=0.025)
                          .breed(parent_picker=pick_best_and_random, combiner=merge, population_size=population_size)
                          .mutate(mutate_function=mutate_painting, rate=0.05, sigma=0.5)
                          .evaluate(lazy=False)
                          .callback(print_summary,
                                    img_template=image_template,
                                    checkpoint_path=checkpoint_path))

    evo_step_1 = (Evolution()
                  .survive(fraction=0.025)
                  .breed(parent_picker=pick_best_and_random, combiner=mate, population_size=population_size)
                  .mutate(mutate_function=mutate_painting, rate=0.05, sigma=0.5)
                  .evaluate(lazy=False)
                  .callback(print_summary,
                            img_template=image_template,
                            checkpoint_path=checkpoint_path))

    evo_step_2 = (Evolution()
                  .survive(fraction=0.025)
                  .breed(parent_picker=pick_best_and_random, combiner=mate, population_size=population_size)
                  .mutate(mutate_function=mutate_painting, rate=0.03, sigma=0.4)
                  .evaluate(lazy=False)
                  .callback(print_summary,
                            img_template=image_template,
                            checkpoint_path=checkpoint_path))

    evo_step_3 = (Evolution()
                  .survive(fraction=0.025)
                  .breed(parent_picker=pick_best_and_random, combiner=mate, population_size=population_size)
                  .mutate(mutate_function=mutate_painting, rate=0.005, sigma=0.4)
                  .evaluate(lazy=False)
                  .callback(print_summary,
                            img_template=image_template,
                            checkpoint_path=checkpoint_path))

    shrink_step = (Evolution()
                   .survive(n=1)
                   .breed(parent_picker=pick_best, combiner=clone, population_size=population_size)
                   .mutate(mutate_function=shrink_painting)
                   .evaluate(lazy=False)
                   .callback(print_summary,
                             img_template=image_template,
                             checkpoint_path=checkpoint_path))

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

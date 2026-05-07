# Genetic Art Algorithm - Part 2

To cope with some issues the previous version had I tried a different approach using Voronoi diagrams/partitions/cells. 
This allows points to be duplicated (which initially doesn't have an effect) and removed more easily. This allows some
new evolutionary steps that mimic biology (like whole genome duplications). You can see the progression from the 1 
generation to generation 5000 below. Between the later images there usually is a duplication step, followed by a number
of normal evolutionary steps, followed by a reduction step. 

![Evolving Cells into Girl with a Pearl Earring](./vermeer_evolution.png)

The final result after 5600 generations you can check out below

![Final result after 5600 generations](./vermeer_generation_05600.png)

Note that running this code will take a long time, running 5000 generations took several days to complete on my machine.

## Running the code

To run the code in this repository clone it, set up a virtual environment and install the required packages from 
requirements.txt

```bash

git clone https://github.com/4dcu-be/Genetic-Art-Algorithm-part-2
cd Genetic-Art-Algorithm-part-2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

Next you can run evolve.py and evolve_simple.py using

```bash

python evolve_voronoi.py

```

## Divide and conquer mode (3x3 grid)

You can split the target image into a grid and run the GA per tile, then stitch the final result with optional overlap blending.

### Basic usage (sequential, no overlap blending):

```bash
USE_DIVIDE_AND_CONQUER=1
GRID_ROWS=3
GRID_COLS=3
TILE_NUM_POINTS=30
TILE_POPULATION_SIZE=250
TILE_GENERATION_SCALE=0.6
python evolve_voronoi.py
```

### Advanced usage (parallel with overlap and blending):

```bash
USE_DIVIDE_AND_CONQUER=1
GRID_ROWS=3
GRID_COLS=3
TILE_NUM_POINTS=30
TILE_POPULATION_SIZE=250
TILE_GENERATION_SCALE=0.6
TILE_OVERLAP_PIXELS=20
NUM_WORKERS=4
USE_FEATHER_BLEND=1
python evolve_voronoi.py
```

**New environment variables:**
- `TILE_OVERLAP_PIXELS`: Margin (pixels) to expand tile rendering beyond grid boundaries to blend seams (default: 20)
- `NUM_WORKERS`: Number of parallel workers for tile evolution (default: 1 = sequential). Set to 4–8 for typical modern CPUs.
- `USE_FEATHER_BLEND`: Enable feather blending at overlaps (default: 1 = enabled). Reduces visible seams; disable if you prefer simple paste.

### Output:

- Stitched final image:
  ```bash
  ./output/battleship/drawing_divide_and_conquer.png
  ```

- Per-tile checkpoints and renders:
  ```bash
  ./output/battleship/tile_r{row}_c{col}/
  ```

### Performance tips:

- **Faster iteration:** Reduce `TILE_NUM_POINTS` and `TILE_POPULATION_SIZE` to test overlap/blending quickly.
- **Parallelism:** Set `NUM_WORKERS` to your CPU core count (or slightly higher for I/O-bound operations).
- **Quality vs. speed:** Higher `TILE_OVERLAP_PIXELS` (30–50) gives better blending but increases per-tile render cost.
- **Disable blending:** Set `USE_FEATHER_BLEND=0` if tiles look correct without it (saves a small amount of I/O time).

The paths to the target image and output directory are hard-coded, but can easily be changed. the lines are in the main
routine.

```python
    target_image_path = "./img/girl_with_pearl_earring_half.jpg"
    checkpoint_path = "./output/"
```


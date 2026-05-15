# Stained Glass Art Genetic Algorithm

This project is forked from Sebastian Proost's Voronoi-based Genetic Art Algorithm and seeks to improve upon it. We added image pre-processing, Voronoi shape outlining, and a divide-and-conquer approach. Will potentially update with additional features/optimization.

Contributors:
- Elliot Gong
- Cesar Ramirez
- Md Islam
- Alex Alcazar

## Software Requirements

For the best possible experience, the following is required:
 - Python 3.10 or later
 - Git 2.0 or later
 - Operating System: Windows, macOS, or Linux
 - Text Editor or IDE (recommended): VS Code, PyCharm, Sublime Text, or any text editor of your choice

## Set Up Instructions
To run the code in this repository, clone it, create a virtual environment, and install the required packages from requirements.txt.

```bash
git clone https://github.com/TheElliotGong/Genetic-Art-Algorithm-part-2
cd Genetic-Art-Algorithm-part-2
python -m venv venv
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.\venv\Scripts\Activate.ps1
```

On macOS or Linux, use:

```bash
source venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Instructions on how to run the project
To run the genetic algorithm, use one of the following commands:

```bash
python evolve_tiled.py  # For the divide-and-conquer approach. Recommended.
python evolve_voronoi.py # For the brute force approach.
```

The target image path and output directory are hard-coded in the main routine, but they can be changed easily:

```python
target_image_path = "./img/girl_with_pearl_earring_half.jpg"
checkpoint_path = "./output/"
```

Here are the hyperparameter settings/adjustments you can make regarding the genetic algorithm in both evolve_tiled.py and evolve_voronoi.py:

### `evolve_voronoi.py` (whole-image / brute-force)
- `num_points`: Number of Voronoi points per painting (controls detail). Default: `250`.
- `population_size`: Population size for the GA. Default: `250`.
- `initialColorCount`: Colors used when extracting the initial palette from the image. Default: `60`.
- `finalColorCount`: Number of colors after condensing the palette. Default: `20`.
- `concurrent_workers`: Number of worker processes for parallel evaluation. Default: `4`.
- Evolution operators / schedule (tunable inside the Evolution pipeline):
	- `survive fraction`: fraction of population retained each generation. Example used: `0.025`.
	- `mutation rate`: typically `0.03`–`0.05` (sigma roughly `0.4`–`0.5`).
	- `breed` / `combiner`: uses `mate` and `merge` functions to combine parents.

### `evolve_tiled.py` (divide-and-conquer tiled runs)
- `n_rows`, `n_cols`: Tile grid size (e.g., `3 x 3`). Defaults used in main: `3, 3`.
- `points_initial`: Initial number of points seeded per tile. Default in `evolve_tile()`: `50`.
- `population_size`: Population size per tile. Default in `evolve_tile()`: `100`.
- `workers`: Concurrent workers passed to per-tile population evaluation. Function default: `4`; example run uses `8`.
- `gens_phase1`, `gens_phase2`: Number of generations for the two evolution phases. Defaults in `evolve_tile()`: `gens_phase1=999`, `gens_phase2=1000`.
- `initial_color_count` / `final_color_count`: Palette extraction for a tile. Defaults: `30` -> `12`.
- `region_bias`: Probability to seed points inside detected regions (helps convergence). Example value: `0.85`.
- `min_area`: Minimum pixel area for detected regions (used in `build_region_groups`). Example value: `10` for tiles.
- Evolution operators / schedule (per-tile Evolution pipelines):
	- `survive fraction`: `0.025` (example).
	- `mutation rate`: `0.03`–`0.05` (sigma `0.4`–`0.5`).
	- duplication step: a dedicated `merge`/duplication evolution stage is used to increase point counts.

Notes:
- Defaults above are taken from the current code and from the example runs in the `__main__` sections. You can tune these values in the respective scripts to trade quality for runtime.
- For large images prefer the tiled workflow for lower memory usage and easier parallelism.




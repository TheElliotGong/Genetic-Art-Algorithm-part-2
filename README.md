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

## Web interface

The quickest way to use the project is the browser interface, which wraps the
same algorithm in an upload form, a hyperparameter panel and a live progress
view:

```bash
python run_web.py
```

Then open <http://127.0.0.1:8000>. Pass `--host` / `--port` to change where it
binds, e.g. `python run_web.py --host 0.0.0.0 --port 8080` to reach it from
another machine on your network.

What it gives you:

- **A target image** from your machine (drag-and-drop or file picker), from a
  URL the server fetches for you, or one of the samples in `img/`.
- **Every hyperparameter** described below, as sliders with live bounds pulled
  from the server, plus *Quick look* / *Balanced* / *Detailed* presets. Both the
  whole-image and the tiled strategy are available.
- **Live progress**: a progress bar over the whole run (with tile boundaries
  marked in tiled mode), elapsed time and a running estimate of the time
  remaining, generation counter, similarity to the target, and current point
  count.
- **A live preview** of the best individual, refreshed every *N* generations,
  with a slider that wipes between the target and the evolved painting.
- **Stop** at any time, then download the final PNG. Finished runs stay in the
  *Recent runs* list and can be reopened.

Runs execute one at a time on a background worker thread; anything submitted
while one is in flight is queued and reports its position. Output is written to
`runs/` (override with `VORONOI_WEB_DATA`).

The REST API behind the page is documented at `/docs` if you would rather drive
it from a script.

### Running the algorithm directly

To run the genetic algorithm from the command line, use one of the following
commands:

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

## Performance

The evolution loop was optimized without changing what it produces: for a given
genome the renderer, the outlined renderer and the fitness score are identical
to the original implementation, which `test_equivalence.py` asserts directly
against a copy of the original code.

Measured on `img/girl_with_pearl_earring_half.jpg` (400x469), 250 points,
population 60, on a 2 core machine:

| operation | before | after | speedup |
| --- | --- | --- | --- |
| render | 2.60 ms | 1.42 ms | 1.8x |
| fitness (`image_diff`) | 2.59 ms | 1.49 ms | 1.7x |
| `deepcopy` of a painting | 1.076 ms | 0.022 ms | 48x |
| serialize one individual | 3.66 ms | 0.18 ms | 20x |
| serialized size per individual | 1290.8 KiB | 3.4 KiB | 384x |
| **end to end** | **1.0 gen/s** | **9.7 gen/s** | **9.7x** |

Where the time went:

- **The target image was stored on every painting.** `evol` serializes whole
  individuals on each `evaluate`, so a 250 individual population shipped the
  target image 250 times per generation, and every `deepcopy` in the mutation
  and crossover operators duplicated it again. Targets now live in a
  per-process registry (`target_cache.py`) and paintings only carry a short
  content key.
- **The genome was serialized as objects.** `evol` serializes with `dill`,
  which dispatches per object; paintings now transport their points as two
  numpy arrays and rebuild them on arrival.
- **Rendering reallocated four full size arrays per call.** They are now
  reused per process, and the color lookup writes into a preallocated buffer.
- **Operators copied more than they needed to.** Crossover and merge hand back
  children that already own their points, so the extra `deepcopy` is gone, and
  mutation samples the indices it needs instead of shuffling all of them.

### Benchmarking

```bash
python benchmark.py --image ./img/car.jpeg --label after --save-json bench_after.json
python benchmark.py --compare bench_before.json          # A/B against a saved run
python benchmark.py --sweep-workers 1,2,4,8              # find the best worker count
python test_equivalence.py                               # confirm output is unchanged
```

Run the worker sweep on your own machine before picking `concurrent_workers` /
`workers`. Now that individuals are small on the wire, worker processes only pay
off once the per-individual rendering work exceeds the cost of moving
individuals between processes - for small tiles a lower worker count can be
faster than a higher one.

### Environment variables

- `VORONOI_PROGRESS=1`: restore the per-evaluation progress dots (off by
  default; they cost a flushed write per individual per generation).
- `VORONOI_TARGET_CACHE=<dir>`: where target pixels are spooled so that worker
  processes started with `spawn` (the default on Windows) can load them.
  Defaults to a `voronoi-targets` folder in the system temp directory.
- `VORONOI_EMBED_TARGET=1`: skip the spool file and embed target pixels in every
  serialized painting. Slower, but makes population checkpoints self-contained
  and loadable on another machine.
- `VORONOI_WEB_DATA=<dir>`: where the web interface stores uploads and run
  output. Defaults to `runs/` next to the code.
- `VORONOI_ALLOW_PRIVATE_URLS=1`: allow "from URL" fetches that resolve to
  private, loopback or link-local addresses. Blocked by default, because the
  server fetches whatever URL it is handed - only enable it if you trust
  everyone who can reach the app.




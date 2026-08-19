# Stained Glass Genetic Art

[![tests](https://github.com/TheElliotGong/Genetic-Art-Algorithm-part-2/actions/workflows/tests.yml/badge.svg)](https://github.com/TheElliotGong/Genetic-Art-Algorithm-part-2/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Evolves a stained-glass style painting toward any target image using a genetic
algorithm over a Voronoi diagram, and serves the whole thing as a REST-backed
web application with live progress streaming.

![Evolution of a painting toward the target image](vermeer_evolution.png)

*A population evolving toward Vermeer's* Girl with a Pearl Earring. *Each frame
is the best individual at that point in the run.*

---

## At a glance

| | |
| --- | --- |
| **Core** | Genetic algorithm over a Voronoi tessellation, written in Python |
| **Architecture** | Client/server web application - REST API + Server-Sent Events, vanilla-JS front end |
| **Performance** | **9.7x end-to-end speedup**, proven output-identical by a dedicated equivalence test suite |
| **Testing** | **288 automated tests**, running on 3 operating systems x 2 Python versions in CI |
| **Scale** | ~2,750 lines of application code across 10 modules, backed by ~2,860 lines of tests |

This started as a fork of Sebastian Proost's Voronoi genetic art algorithm. The
work that followed - image pre-processing, cell outlining, a divide-and-conquer
tiled solver, a full profiling-and-optimization pass, the web application, and
the test suite - is documented below.

---

## Architecture

The project is layered so the algorithm never depends on the web tier, which
keeps it usable from the command line and testable without a running server.

```
Browser (HTML5 / CSS / vanilla JS)
        |  REST + Server-Sent Events
webapp/app.py        - 18 HTTP endpoints, request validation, streaming
webapp/params.py     - typed parameter model, the single validation authority
webapp/runner.py     - job queue, worker thread, progress + lifecycle
webapp/imaging.py    - decoding, EXIF handling, SSRF-guarded URL fetch
        |  plain function calls
evolve_voronoi.py    - whole-image solver, seeding, palette, selection
evolve_tiled.py      - divide-and-conquer tiled solver
voronoi_painting.py  - the genome: VoronoiPainting / ColoredPoint
target_cache.py      - process-global content-addressed image registry
```

### Object-oriented design

The genome is a small, deliberately-designed class hierarchy, because it is
copied and serialized millions of times per run - every design decision here
shows up directly in the profiler.

**`ColoredPoint`** ([voronoi_painting.py:45](voronoi_painting.py:45)) - one
Voronoi cell: a coordinate and an RGBA color. It declares `__slots__` to drop
the per-instance `__dict__`, which matters because a single population holds
hundreds of thousands of these. It implements `__getstate__` / `__setstate__`
so `__slots__` and pickling coexist.

**`VoronoiPainting`** ([voronoi_painting.py:111](voronoi_painting.py:111)) - one
candidate solution. It encapsulates its point list behind read-only `@property`
accessors and exposes the genetic operators as `@staticmethod` factories
(`breed`, `mate`, `merge`). Its notable feature is a custom serialization
contract: `__getstate__` converts the genome into two flat NumPy arrays rather
than letting the serializer walk thousands of Python objects, and
`__setstate__` rebuilds them on arrival.

**`Job` and `JobManager`** ([webapp/runner.py:148](webapp/runner.py:148)) -
`Job` is a state machine (`queued -> running -> done | error | cancelled`)
holding progress, history and cancellation state behind a lock. `JobManager`
owns the queue, the worker thread, and eviction of old runs.

**`RunParams`** ([webapp/params.py:13](webapp/params.py:13)) - a Pydantic model
that is simultaneously the API request schema, the validation layer, and the
description the front end builds its controls from. Derived values
(`survivor_count`, `total_generations`, `phase_plan`) are computed properties,
so the progress bar and the solver can never disagree about what a run means.

Design patterns applied where they earned their place:

| Pattern | Where | Why |
| --- | --- | --- |
| **Object pool** | `_render_buffers` ([voronoi_painting.py:22](voronoi_painting.py:22)) | Rendering needs 4 scratch arrays; pooling them by shape removed a per-render allocation of 12 bytes per output pixel |
| **Flyweight / identity map** | [target_cache.py](target_cache.py) | One shared copy of the target image per process, keyed by content hash, instead of one per individual |
| **Prototype** | `__deepcopy__` ([voronoi_painting.py:202](voronoi_painting.py:202)) | Clones the genome and *shares* the immutable target - a 48x faster copy |
| **Strategy** | `parent_picker` / `combiner` ([evolve_voronoi.py:397](evolve_voronoi.py:397)) | Selection (`pick_best`, `pick_random`, `pick_best_and_random`) and recombination (`mate`, `merge`) are swapped per evolution phase |
| **Producer/consumer** | `JobManager` ([webapp/runner.py:316](webapp/runner.py:316)) | HTTP threads enqueue; one worker thread executes, so a long run never blocks the API |

### Data structures and algorithms

**Voronoi assignment via distance transform.** The naive way to color a Voronoi
diagram is, for every pixel, to scan every seed - `O(width x height x points)`.
Instead, seeds are written into a binary image and a single distance transform
labels every pixel with its nearest seed in `O(width x height)`, independent of
the point count ([voronoi_painting.py:254](voronoi_painting.py:254)). At 200
points that removes two orders of magnitude of work from the innermost loop of
the entire program.

**Color lookup table.** Labels are turned into pixels through a LUT indexed by
label, written into a preallocated buffer with `np.take`. That is `O(1)` per
pixel with no per-pixel branching, and it replaced fancy indexing that
allocated a fresh result array on every one of the millions of renders per run.

**Region-seeded initialization.** Rather than scattering starting points
uniformly, the target is segmented first: a palette is extracted and condensed,
pixels are mapped to it, Canny edges and a Laplacian texture measure split those
into regions, and connected-component labeling groups them
([evolve_voronoi.py:173](evolve_voronoi.py:173)). Points are then seeded inside
detected regions with probability `region_bias`. `tests/test_evolve_voronoi.py`
asserts this beats uniform random seeding rather than assuming it.

**Palette reduction.** `condense_palette` greedily filters colors below a
Euclidean distance threshold in RGB space; `simplify_palette` then strides
uniformly to hit an exact target count; `map_pixels_to_palette` assigns every
pixel via a vectorized broadcast-and-`argmin` nearest-neighbor search
([evolve_voronoi.py:131](evolve_voronoi.py:131)).

**Divide and conquer.** [evolve_tiled.py](evolve_tiled.py) partitions the image
into an `n_rows x n_cols` grid, evolves each tile as an independent population,
then feather-blends overlapping borders so no seam is visible. Smaller
subproblems converge faster and use less memory than one global population.
`tests/test_evolve_tiled.py` verifies the grid covers every pixel exactly once
and that stitching uniform tiles leaves no visible seam.

**Content-addressed cache.** [target_cache.py](target_cache.py) keys target
images by a hash of their pixels, so identical images collapse to one entry, and
spools them to disk so `spawn`-based worker processes (the default on Windows)
can rehydrate without the image crossing the wire.

**Complexity fix in mutation.** `mutate_points` samples the `k` indices it needs
in `O(k)` instead of shuffling all `n` and taking a prefix - at a 3% mutation
rate the shuffle dominated
([voronoi_painting.py:213](voronoi_painting.py:213)).

---

## Web interface

```bash
python run_web.py
```

Then open <http://127.0.0.1:8000>. Pass `--host` / `--port` to change where it
binds, e.g. `python run_web.py --host 0.0.0.0 --port 8080` to reach it from
another machine on your network.

The API is a REST service over JSON; the browser client is dependency-free
HTML5, CSS and JavaScript.

| Concern | Implementation |
| --- | --- |
| Target selection | Multipart upload, server-side URL fetch, or bundled samples |
| Validation | One Pydantic model serves as request schema, bounds check, and the source the UI builds its sliders from |
| Live progress | Server-Sent Events (`/api/jobs/{id}/events`) push generation, similarity, ETA and point count without polling |
| Live preview | Best individual re-rendered every *N* generations, with a wipe slider against the target |
| Concurrency | Runs execute one at a time on a background worker thread; queued requests are told their position |
| Cancellation | Cooperative, via a per-job `threading.Event` checked in the evolution loop |
| Security | URL fetches are refused when the hostname resolves to a private, loopback or link-local address (SSRF guard) |

Interactive API documentation is generated at `/docs`. Output is written to
`runs/` (override with `VORONOI_WEB_DATA`).

---

## Performance engineering

The evolution loop was profiled and optimized **without changing what it
produces**. For a given genome, the renderer, the outlined renderer and the
fitness score are byte-identical to the original implementation - and
`tests/test_equivalence.py` asserts exactly that against a verbatim copy of the
pre-optimization code kept in `tests/reference_impl.py`.

Measured on `img/girl_with_pearl_earring_half.jpg` (400x469), 250 points,
population 60, on a 2 core machine:

| operation | before | after | speedup |
| --- | --- | --- | --- |
| render | 2.60 ms | 1.42 ms | 1.8x |
| fitness (`image_diff`) | 2.59 ms | 1.49 ms | 1.7x |
| `deepcopy` of a painting | 1.076 ms | 0.022 ms | **48x** |
| serialize one individual | 3.66 ms | 0.18 ms | **20x** |
| serialized size per individual | 1290.8 KiB | 3.4 KiB | **384x** |
| **end to end** | **1.0 gen/s** | **9.7 gen/s** | **9.7x** |

Where the time was going:

- **The target image was stored on every painting.** `evol` serializes whole
  individuals on each `evaluate`, so a 250 individual population shipped the
  target image 250 times per generation, and every `deepcopy` in the mutation
  and crossover operators duplicated it again. Targets now live in a
  per-process registry ([target_cache.py](target_cache.py)) and paintings carry
  only a short content key.
- **The genome was serialized as objects.** `evol` serializes with `dill`,
  which dispatches per object; paintings now transport their points as two
  NumPy arrays and rebuild them on arrival.
- **Rendering reallocated four full-size arrays per call.** They are now pooled
  per process and reused, and the color lookup writes into a preallocated
  buffer.
- **Operators copied more than they needed to.** Crossover and merge hand back
  children that already own their points, so the extra `deepcopy` is gone, and
  mutation samples the indices it needs instead of shuffling all of them.

### Benchmarking

```bash
python benchmark.py --image ./img/car.jpeg --label after --save-json bench_after.json
python benchmark.py --compare bench_before.json          # A/B against a saved run
python benchmark.py --sweep-workers 1,2,4,8              # find the best worker count
pytest tests/test_equivalence.py                         # confirm output is unchanged
```

Run the worker sweep on your own machine before picking `concurrent_workers` /
`workers`. Now that individuals are small on the wire, worker processes only pay
off once the per-individual rendering work exceeds the cost of moving
individuals between processes - for small tiles a lower worker count can be
faster than a higher one.

---

## Testing

288 tests across 10 files, running in under 5 seconds with no network access, no
GPU and no long evolution runs.

```bash
pip install -r requirements-dev.txt
pytest
```

| file | what it pins down |
| --- | --- |
| `tests/test_equivalence.py` | The optimized renderer, outliner, fitness, genome copying and pickling produce byte-identical results to the pre-optimization code, kept verbatim in `tests/reference_impl.py` |
| `tests/test_voronoi_painting.py` | Point and painting behaviour: colour clamping, mutation rates, crossover, merge, render scaling, the shared scratch buffers |
| `tests/test_target_cache.py` | Content keying, spooling, rehydration in a *separate process*, and that a serialized individual carries no target pixels |
| `tests/test_evolve_voronoi.py` | Palette condensing, region detection, region-seeded initialization (including that it beats uniform random seeding), parent pickers, and a short real evolution that has to improve fitness |
| `tests/test_evolve_tiled.py` | Tile grids cover the image exactly once, feather masks stay opaque over tile cores, stitching uniform tiles leaves no visible seam |
| `tests/test_params.py` | Every hyperparameter bound, the outline-colour parser, and the phase plan / survivor-count arithmetic the progress bar depends on |
| `tests/test_imaging.py` | Decoding, EXIF rotation, size caps, and the SSRF guard that refuses URLs resolving to private addresses |
| `tests/test_runner.py` | Job lifecycle: queueing, cancellation, eviction, progress and ETA, thread-safe snapshots, and whole runs in both single and tiled mode |
| `tests/test_api.py` | Every HTTP endpoint, including uploads, validation errors, the SSE progress stream and downloading a finished run |
| `tests/test_entrypoints.py` | Every script still imports, and none of them start work at import time |

Useful variations:

```bash
pytest -m "not slow"                     # skip the real-evolution tests
pytest tests/test_api.py -k upload       # one file, one pattern
pytest --cov --cov-report=term-missing   # coverage report
python scripts/run_tests.py --fast       # same thing, platform independent
```

Tests that run a real evolution loop are marked `slow`.

### Continuous integration

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs the suite on
every push and pull request, nightly at 06:17 UTC, and on demand - across
**Linux, Windows and macOS on Python 3.12 and 3.14**. Test results and coverage
are uploaded as artifacts, and the `all tests passed` job is the one to mark as
a required status check on the default branch.

Locally, install the git hooks once and the suite runs before every push:

```bash
python scripts/install_hooks.py
```

`git push --no-verify` skips it for one push, and
`python scripts/install_hooks.py --uninstall` removes it.

---

## Setup

Requirements: Python 3.12 or later (CI covers 3.12 and 3.14; the pinned NumPy
and scikit-image releases do not support anything older than 3.11), Git 2.0 or
later, on Windows, macOS or Linux.

```bash
git clone https://github.com/TheElliotGong/Genetic-Art-Algorithm-part-2
cd Genetic-Art-Algorithm-part-2
python -m venv venv
```

Activate it - on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt      # runtime
pip install -r requirements-dev.txt  # plus everything the tests need
```

---

## Command line usage

```bash
python evolve_tiled.py    # divide-and-conquer tiled approach. Recommended.
python evolve_voronoi.py  # whole-image brute force approach.
```

The target image path and output directory are set in the main routine:

```python
target_image_path = "./img/girl_with_pearl_earring_half.jpg"
checkpoint_path = "./output/"
```

### `evolve_voronoi.py` (whole-image)

- `num_points`: Voronoi points per painting, controlling detail. Default `250`.
- `population_size`: GA population size. Default `250`.
- `initialColorCount`: colors extracted for the initial palette. Default `60`.
- `finalColorCount`: colors kept after condensing. Default `20`.
- `concurrent_workers`: worker processes for parallel evaluation. Default `4`.
- Operators and schedule: `survive fraction` (e.g. `0.025`), `mutation rate`
  (typically `0.03`-`0.05`, sigma `0.4`-`0.5`), and `breed` / `combiner` using
  the `mate` and `merge` functions.

### `evolve_tiled.py` (divide and conquer)

- `n_rows`, `n_cols`: tile grid size. Defaults `3, 3`.
- `points_initial`: points seeded per tile. Default `50`.
- `population_size`: population per tile. Default `100`.
- `workers`: concurrent workers per tile. Default `4`; the example run uses `8`.
- `gens_phase1`, `gens_phase2`: generations for the two phases. Defaults `999`
  and `1000`.
- `initial_color_count` / `final_color_count`: per-tile palette. Defaults `30`
  and `12`.
- `region_bias`: probability of seeding inside a detected region. Example `0.85`.
- `min_area`: minimum pixel area for detected regions. Example `10` for tiles.
- Operators: `survive fraction` `0.025`, mutation rate `0.03`-`0.05` (sigma
  `0.4`-`0.5`), plus a dedicated merge/duplication stage that increases the
  point count mid-run.

Prefer the tiled workflow for large images - lower memory use and easier
parallelism.

---

## Environment variables

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

---

## Credits

Forked from Sebastian Proost's Voronoi-based genetic art algorithm, then
extended with image pre-processing, cell outlining, the tiled solver, the
performance work, the web application and the test suite.

Contributors: Elliot Gong, Cesar Ramirez, Md Islam, Alex Alcazar.

Licensed under the MIT License - see [LICENSE](LICENSE).

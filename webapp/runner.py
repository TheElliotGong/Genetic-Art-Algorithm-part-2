"""Background execution of evolution runs for the web interface.

Runs are queued and executed one at a time by a single worker thread. That is
deliberate: scoring is CPU bound and ``voronoi_painting`` renders through
process-global scratch buffers, so two runs sharing a process would both fight
for the CPU and trample each other's buffers.

The job object is the bridge between that worker thread and the HTTP handlers -
the worker writes progress into it, the handlers read snapshots out of it.
"""

import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from evol import Evolution, Population
from PIL import Image
from skimage import feature

from evolve_tiled import feather_blend_tiles, split_into_tiles
from evolve_voronoi import (
    build_region_groups,
    create_region_seeded_population,
    mate,
    merge,
    mutate_painting,
    pick_best_and_random,
    score,
    simplify_palette,
)
from voronoi_painting import VoronoiPainting

from .imaging import fit_within, to_png_bytes
from .params import RunParams

# Fitness is an L1 norm over the RGB channels, so this is the worst score a
# painting of a given size could possibly get. Dividing by it turns the raw
# number into a 0-1 similarity that means something to a user.
_MAX_CHANNEL_ERROR = 255 * 3

# Preview refreshes are cheap but not free; the history feeding the chart is
# capped so a 20k generation run does not accumulate an unbounded series.
_MAX_HISTORY_POINTS = 400


class RunCancelled(Exception):
    """Raised inside the evolution loop when the user cancels a run."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_painting(painting: VoronoiPainting, params: RunParams, scale: int) -> Image.Image:
    """Render a painting with the run's output settings.

    :param painting: The painting to render.
    :param params: The run's hyperparameters.
    :param scale: Supersampling factor for the render.
    """
    if params.outline:
        return painting.draw_lines(
            scale=scale,
            line_width=params.outline_width,
            line_color=params.outline_rgb,
        )
    return painting.draw(scale=scale)


def _stitch_tiles(tile_results, full_size, scale, params: RunParams) -> Image.Image:
    """Feather-blend evolved tiles back into a single canvas.

    This is ``evolve_tiled.stitch`` generalised over the render scale and the
    outline settings, so the same code path serves both the live preview (scale
    1, fast) and the final image (the run's render scale).

    :param tile_results: ``(TileSpec, painting)`` pairs for the finished tiles.
    :param full_size: ``(width, height)`` of the un-tiled target.
    :param scale: Supersampling factor applied to every tile.
    :param params: The run's hyperparameters.
    """
    canvas_width, canvas_height = full_size[0] * scale, full_size[1] * scale
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weights = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    for tile_spec, painting in tile_results:
        tile_array = np.asarray(
            _render_painting(painting, params, scale).convert("RGB"), dtype=np.float32
        )
        mask = feather_blend_tiles(tile_spec)
        if scale != 1:
            mask = cv2.resize(
                mask,
                (tile_array.shape[1], tile_array.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        y0, y1 = tile_spec.crop_y0 * scale, tile_spec.crop_y1 * scale
        x0, x1 = tile_spec.crop_x0 * scale, tile_spec.crop_x1 * scale
        canvas[y0:y1, x0:x1] += tile_array * mask[..., None]
        weights[y0:y1, x0:x1] += mask

    weights = np.maximum(weights, 1e-6)
    blended = np.clip(canvas / weights[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _seed_population(params: RunParams, target_image: Image.Image, num_points: int, min_area: int):
    """Build the region-seeded starting population for one target.

    :param params: The run's hyperparameters.
    :param target_image: The image this population evolves towards.
    :param num_points: Points each starting painting is given.
    :param min_area: Smallest region, in pixels, worth seeding from.
    :return: ``(chromosomes, region_count)``.
    """
    converted = target_image.convert(
        "P", palette=Image.ADAPTIVE, colors=params.initial_colors
    )
    palette = converted.getpalette()[: params.initial_colors * 3]
    colors = [tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)]
    condensed = simplify_palette(colors, params.final_colors)
    # A flat or tiny target can condense down to nothing, and the seeding code
    # picks from this palette unconditionally.
    if not condensed:
        condensed = [(128, 128, 128)]

    target_rgb = np.array(target_image.convert("RGB"))
    edges = feature.canny(cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY), sigma=1.2)
    region_groups = build_region_groups(target_rgb, condensed, edges, min_area=min_area)

    chromosomes = create_region_seeded_population(
        params.population_size,
        num_points,
        target_image,
        region_groups,
        condensed,
        region_bias=params.region_bias,
    )
    return chromosomes, len(region_groups)


class Job:
    """One queued or running evolution, plus everything the UI reads from it."""

    def __init__(self, job_id: str, params: RunParams, source_path: Path, run_dir: Path):
        self.id = job_id
        self.params = params
        self.source_path = source_path
        self.run_dir = run_dir

        self._lock = threading.Lock()
        self.cancel_event = threading.Event()

        self.status = "queued"
        self.stage = "Waiting for a free worker"
        self.error = None

        self.generation = 0
        self.tile_index = 0
        self.best_fitness = None
        self.avg_fitness = None
        self.similarity = None
        self.num_points = params.num_points
        self.region_count = None
        self.history = []

        self.created_at = _utc_now()
        self.started_at = None
        self.finished_at = None
        self._start_monotonic = None
        self._elapsed = 0.0

        self.preview_png = None
        self.preview_version = 0
        self.result_path = None
        self.target_path = None

    # --- Worker-side mutation -------------------------------------------------

    def set_stage(self, stage: str, status: str = None) -> None:
        """Update the human readable stage label, and optionally the status."""
        with self._lock:
            self.stage = stage
            if status is not None:
                self.status = status

    def mark_started(self) -> None:
        with self._lock:
            self.status = "running"
            self.started_at = _utc_now()
            self._start_monotonic = time.monotonic()

    def mark_finished(self, status: str, stage: str, error: str = None) -> None:
        with self._lock:
            self.status = status
            self.stage = stage
            self.error = error
            self.finished_at = _utc_now()
            if self._start_monotonic is not None:
                self._elapsed = time.monotonic() - self._start_monotonic

    def set_region_count(self, region_count: int) -> None:
        """Record how many seeding regions were detected for the current target."""
        with self._lock:
            self.region_count = region_count

    def set_preview(self, image: Image.Image) -> None:
        """Publish a new live preview frame."""
        png = to_png_bytes(image)
        with self._lock:
            self.preview_png = png
            self.preview_version += 1

    def record_generation(self, population, tile_index: int, preview_builder) -> None:
        """Record one generation's statistics and refresh the preview if due.

        :param population: The ``evol`` population that just finished a generation.
        :param tile_index: Index of the tile being evolved (0 in single mode).
        :param preview_builder: Callable turning the best chromosome into the
            image shown as the live preview.
        """
        params = self.params
        best = population.current_best
        individuals = population.individuals
        avg = sum(i.fitness for i in individuals) / len(individuals)
        chromosome = best.chromosome
        pixels = chromosome.get_img_width * chromosome.get_img_height
        similarity = max(0.0, 1.0 - best.fitness / (pixels * _MAX_CHANNEL_ERROR))
        generation = tile_index * params.generations_per_target + population.generation

        with self._lock:
            self.generation = generation
            self.tile_index = tile_index
            self.best_fitness = float(best.fitness)
            self.avg_fitness = float(avg)
            self.similarity = similarity
            self.num_points = chromosome.num_points
            if self._start_monotonic is not None:
                self._elapsed = time.monotonic() - self._start_monotonic

        due = population.generation % params.preview_every == 0
        if due:
            with self._lock:
                self.history.append(
                    {
                        "generation": generation,
                        "similarity": round(similarity, 6),
                        "best": float(best.fitness),
                    }
                )
                # Keep every other sample once the series gets long, so the
                # chart stays cheap to serialize without losing its shape.
                if len(self.history) > _MAX_HISTORY_POINTS:
                    self.history = self.history[::2]
            self.set_preview(preview_builder(chromosome))

    # --- Reader-side snapshot -------------------------------------------------

    def elapsed_seconds(self) -> float:
        if self._start_monotonic is None:
            return 0.0
        if self.status in ("running",):
            return time.monotonic() - self._start_monotonic
        return self._elapsed

    def snapshot(self, queue_position: int = None) -> dict:
        """Return a JSON-serializable view of the job's current state."""
        with self._lock:
            total = self.params.total_generations
            elapsed = self.elapsed_seconds()
            progress = min(1.0, self.generation / total) if total else 0.0
            if self.status == "done":
                progress = 1.0

            eta = None
            if self.status == "running" and self.generation > 0 and progress > 0:
                eta = max(0.0, elapsed / progress - elapsed)

            return {
                "id": self.id,
                "status": self.status,
                "stage": self.stage,
                "error": self.error,
                "queue_position": queue_position,
                "mode": self.params.mode,
                "generation": self.generation,
                "total_generations": total,
                "progress": progress,
                "tile_index": self.tile_index,
                "tile_count": self.params.tile_count,
                "best_fitness": self.best_fitness,
                "avg_fitness": self.avg_fitness,
                "similarity": self.similarity,
                "num_points": self.num_points,
                "region_count": self.region_count,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "preview_version": self.preview_version,
                "has_preview": self.preview_png is not None,
                "has_target": self.target_path is not None,
                "has_result": self.result_path is not None,
                "history": list(self.history),
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "params": self.params.model_dump(),
            }


class JobManager:
    """Queues jobs and runs them one at a time on a dedicated worker thread."""

    def __init__(self, runs_dir: Path, max_jobs: int = 50):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.max_jobs = max_jobs

        self._jobs = {}
        self._order = []
        self._pending = []
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop, name="evolution-worker", daemon=True
        )
        self._worker.start()

    # --- Public API -----------------------------------------------------------

    def submit(self, params: RunParams, source_path: Path) -> Job:
        """Queue a new run against an already-stored source image."""
        job_id = uuid.uuid4().hex[:12]
        run_dir = self.runs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        job = Job(job_id, params, source_path, run_dir)

        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._pending.append(job_id)
            self._evict_old_jobs()

        self._queue.put(job_id)
        return job

    def get(self, job_id: str):
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job_id: str):
        """1-based position in the queue, or None if the job is not waiting."""
        with self._lock:
            if job_id not in self._pending:
                return None
            return self._pending.index(job_id) + 1

    def snapshot(self, job: Job) -> dict:
        return job.snapshot(queue_position=self.queue_position(job.id))

    def list_jobs(self) -> list:
        with self._lock:
            jobs = [self._jobs[job_id] for job_id in reversed(self._order)]
        return [self.snapshot(job) for job in jobs]

    def cancel(self, job_id: str) -> bool:
        """Ask a queued or running job to stop. Returns whether it was active."""
        job = self.get(job_id)
        if job is None or job.status in ("done", "error", "cancelled"):
            return False
        job.cancel_event.set()
        if job.status == "queued":
            job.set_stage("Cancelled before it started", status="cancelled")
            with self._lock:
                if job_id in self._pending:
                    self._pending.remove(job_id)
        else:
            job.set_stage("Stopping...")
        return True

    def remove(self, job_id: str) -> None:
        """Forget a job entirely, cancelling it first if it is still active."""
        self.cancel(job_id)
        with self._lock:
            self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)
            if job_id in self._pending:
                self._pending.remove(job_id)

    # --- Worker ---------------------------------------------------------------

    def _evict_old_jobs(self) -> None:
        """Drop the oldest finished jobs once the history grows past the cap."""
        while len(self._order) > self.max_jobs:
            oldest = self._order[0]
            job = self._jobs.get(oldest)
            if job is not None and job.status in ("queued", "running"):
                break
            self._order.pop(0)
            self._jobs.pop(oldest, None)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            with self._lock:
                if job_id in self._pending:
                    self._pending.remove(job_id)
            if job is None or job.cancel_event.is_set():
                if job is not None and job.status != "cancelled":
                    job.mark_finished("cancelled", "Cancelled before it started")
                continue
            try:
                self._run(job)
            except RunCancelled:
                job.mark_finished("cancelled", "Cancelled")
            except Exception as error:  # surfaced to the user rather than swallowed
                job.mark_finished("error", "Run failed", error=f"{type(error).__name__}: {error}")

    def _run(self, job: Job) -> None:
        """Execute one job end to end."""
        params = job.params
        job.mark_started()
        job.set_stage("Preparing target image")

        target_image = fit_within(
            Image.open(job.source_path).convert("RGBA"), params.max_dimension
        )
        target_path = job.run_dir / "target.png"
        target_image.convert("RGB").save(target_path, "PNG")
        job.target_path = target_path

        if params.mode == "single":
            painting = self._evolve_target(
                job,
                target_image,
                tile_index=0,
                num_points=params.num_points,
                min_area=40,
                preview_builder=lambda chromosome: _render_painting(chromosome, params, 1),
            )
            job.set_stage("Rendering final image")
            final = _render_painting(painting, params, params.render_scale)
        else:
            tiles = split_into_tiles(
                target_image,
                n_rows=params.tile_rows,
                n_cols=params.tile_cols,
                overlap_pixels=params.tile_overlap,
            )
            finished = []
            for index, tile_spec in enumerate(tiles):

                def preview_builder(chromosome, spec=tile_spec, done=finished):
                    return _stitch_tiles(
                        done + [(spec, chromosome)], target_image.size, 1, params
                    )

                best = self._evolve_target(
                    job,
                    tile_spec.image,
                    tile_index=index,
                    num_points=params.num_points,
                    min_area=10,
                    preview_builder=preview_builder,
                )
                finished.append((tile_spec, best))
                job.set_preview(_stitch_tiles(finished, target_image.size, 1, params))

            job.set_stage("Stitching final image")
            final = _stitch_tiles(finished, target_image.size, params.render_scale, params)

        result_path = job.run_dir / "result.png"
        final.save(result_path, "PNG")
        job.result_path = result_path
        job.set_preview(final)
        job.mark_finished("done", "Finished")

    def _evolve_target(
        self, job: Job, target_image, *, tile_index, num_points, min_area, preview_builder
    ) -> VoronoiPainting:
        """Evolve a single target (the whole image, or one tile) to completion.

        :param job: The job this work belongs to.
        :param target_image: The image being approximated.
        :param tile_index: Index of the tile, used to offset global progress.
        :param num_points: Points each starting painting is given.
        :param min_area: Smallest region, in pixels, worth seeding from.
        :param preview_builder: Turns a chromosome into the live preview image.
        """
        params = job.params
        if job.cancel_event.is_set():
            raise RunCancelled

        label = (
            "Analysing image"
            if params.mode == "single"
            else f"Analysing tile {tile_index + 1}/{params.tile_count}"
        )
        job.set_stage(label)
        chromosomes, region_count = _seed_population(
            params, target_image, num_points, min_area
        )
        job.set_region_count(region_count)

        population = Population(
            chromosomes=chromosomes,
            eval_function=score,
            maximize=False,
            concurrent_workers=params.workers,
        )
        # ``evolve`` returns new populations that share this pool, so the handle
        # has to be kept here to be shut down once the target is finished.
        pool = population.pool

        def callback(current):
            if job.cancel_event.is_set():
                raise RunCancelled
            job.record_generation(current, tile_index, preview_builder)
            return current

        try:
            for phase, generations in params.phase_plan():
                if generations <= 0:
                    continue

                stage = phase.capitalize()
                if params.mode == "tiled":
                    stage = f"{stage} - tile {tile_index + 1}/{params.tile_count}"
                job.set_stage(stage)

                if phase == "duplicate":
                    combiner, rate, sigma = merge, params.mutation_rate, params.mutation_sigma
                elif phase == "refine":
                    # The refinement phase deliberately mutates less, so late
                    # generations polish the painting instead of scrambling it.
                    combiner = mate
                    rate = params.mutation_rate * 0.6
                    sigma = params.mutation_sigma * 0.8
                else:
                    combiner, rate, sigma = mate, params.mutation_rate, params.mutation_sigma

                evolution = (
                    Evolution()
                    .survive(n=params.survivor_count)
                    .breed(
                        parent_picker=pick_best_and_random,
                        combiner=combiner,
                        population_size=params.population_size,
                    )
                    .mutate(mutate_function=mutate_painting, rate=rate, sigma=sigma)
                    .evaluate(lazy=False)
                    .callback(callback)
                )
                population = population.evolve(evolution, n=generations)
        finally:
            if pool is not None:
                pool.terminate()

        return population.current_best.chromosome

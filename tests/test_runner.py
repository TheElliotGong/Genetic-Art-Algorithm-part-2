"""Tests for the background job runner behind the web interface."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evolve_tiled import split_into_tiles
from tests.conftest import build_painting, make_target, tiny_run_params
from webapp import runner
from webapp.params import RunParams
from webapp.runner import Job, JobManager, RunCancelled, _render_painting, _stitch_tiles

TERMINAL = ("done", "error", "cancelled")


def wait_for(job: Job, statuses=TERMINAL, timeout: float = 180.0) -> str:
    """Block until ``job`` reaches one of ``statuses``.

    :param job: The job to watch.
    :param statuses: Statuses that end the wait.
    :param timeout: Seconds to wait before failing the test.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.status in statuses:
            return job.status
        time.sleep(0.02)
    pytest.fail(f"job stayed in {job.status!r} (stage {job.stage!r}) for {timeout}s")


@pytest.fixture
def manager(tmp_path: Path):
    """A job manager writing into the test's own temporary directory."""
    return JobManager(tmp_path / "jobs")


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    path = tmp_path / "source.png"
    make_target(64, 48).convert("RGB").save(path, "PNG")
    return path


class FakeIndividual:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness


class FakePopulation:
    """Enough of an ``evol`` population for ``Job.record_generation``."""

    def __init__(self, chromosome, generation, fitness=1000.0):
        self.generation = generation
        self.individuals = [
            FakeIndividual(chromosome, fitness),
            FakeIndividual(chromosome, fitness * 2),
        ]

    @property
    def current_best(self):
        return self.individuals[0]


# --- Rendering helpers -----------------------------------------------------


def test_render_painting_follows_the_outline_setting(target):
    painting = build_painting(target, 30, seed=1)

    plain = np.array(_render_painting(painting, tiny_run_params(outline=False), 1))
    outlined = np.array(
        _render_painting(
            painting, tiny_run_params(outline=True, outline_color="#ff0000"), 1
        )
    )

    assert plain.shape == outlined.shape
    changed = np.any(plain != outlined, axis=2)
    assert changed.any()
    assert np.array_equal(np.unique(outlined[changed], axis=0), [[255, 0, 0]])


def test_render_painting_applies_the_scale(target):
    painting = build_painting(target, 10, seed=2)
    image = _render_painting(painting, tiny_run_params(), 3)
    assert image.size == (target.size[0] * 3, target.size[1] * 3)


@pytest.mark.parametrize("scale", [1, 2])
def test_stitch_tiles_produces_a_full_canvas(target, scale):
    params = tiny_run_params(mode="tiled", tile_rows=2, tile_cols=2, tile_overlap=6)
    tiles = split_into_tiles(target, 2, 2, overlap_pixels=6)
    results = [
        (tile, build_painting(tile.image, 8, seed=index))
        for index, tile in enumerate(tiles)
    ]

    stitched = _stitch_tiles(results, target.size, scale, params)

    assert stitched.size == (target.size[0] * scale, target.size[1] * scale)
    assert stitched.mode == "RGB"


def test_stitch_tiles_works_with_only_some_tiles_finished(target):
    """The live preview stitches whatever is done so far."""
    params = tiny_run_params(mode="tiled", tile_rows=2, tile_cols=2, tile_overlap=6)
    tiles = split_into_tiles(target, 2, 2, overlap_pixels=6)
    partial = [(tiles[0], build_painting(tiles[0].image, 8, seed=0))]

    stitched = _stitch_tiles(partial, target.size, 1, params)

    assert stitched.size == target.size
    # Areas no tile covers stay black rather than blowing up on a zero weight.
    assert np.array(stitched)[-1, -1].tolist() == [0, 0, 0]


# --- Seeding ---------------------------------------------------------------


def test_seed_population_returns_a_full_population(target):
    params = tiny_run_params(population_size=5, initial_colors=8, final_colors=4)
    chromosomes, region_count = runner._seed_population(params, target, 12, min_area=4)

    assert len(chromosomes) == 5
    assert all(painting.num_points == 12 for painting in chromosomes)
    assert region_count >= 1


def test_seed_population_copes_with_a_flat_target():
    """A featureless image still has to yield a usable starting population."""
    flat = Image.new("RGBA", (24, 24), (90, 90, 90, 255))
    params = tiny_run_params(population_size=4)

    chromosomes, region_count = runner._seed_population(params, flat, 6, min_area=4)

    assert len(chromosomes) == 4
    assert region_count >= 1


# --- Job bookkeeping -------------------------------------------------------


def test_a_new_job_starts_queued(tmp_path):
    job = Job("abc", tiny_run_params(), tmp_path / "src.png", tmp_path)

    snapshot = job.snapshot(queue_position=2)
    assert snapshot["status"] == "queued"
    assert snapshot["queue_position"] == 2
    assert snapshot["progress"] == 0.0
    assert snapshot["eta_seconds"] is None
    assert snapshot["has_preview"] is False
    assert snapshot["params"]["mode"] == "single"


def test_snapshot_tracks_progress_and_eta(tmp_path, target):
    params = tiny_run_params(generations=10)
    job = Job("abc", params, tmp_path / "src.png", tmp_path)
    job.mark_started()

    painting = build_painting(target, 8, seed=1)
    job.record_generation(FakePopulation(painting, 5), 0, lambda c: c.draw())

    snapshot = job.snapshot()
    assert snapshot["status"] == "running"
    assert snapshot["generation"] == 5
    assert 0 < snapshot["progress"] < 1
    assert snapshot["eta_seconds"] is not None
    assert snapshot["num_points"] == 8
    assert 0.0 <= snapshot["similarity"] <= 1.0
    assert snapshot["has_preview"] is True


def test_finishing_a_job_pins_progress_to_one(tmp_path):
    job = Job("abc", tiny_run_params(), tmp_path / "src.png", tmp_path)
    job.mark_started()
    job.mark_finished("done", "Finished")

    snapshot = job.snapshot()
    assert snapshot["progress"] == 1.0
    assert snapshot["finished_at"] is not None
    assert snapshot["elapsed_seconds"] >= 0


def test_tiled_progress_accounts_for_earlier_tiles(tmp_path, target):
    params = tiny_run_params(mode="tiled", tile_rows=2, tile_cols=2, generations=10)
    job = Job("abc", params, tmp_path / "src.png", tmp_path)
    job.mark_started()
    painting = build_painting(target, 8, seed=1)

    job.record_generation(FakePopulation(painting, 4), 2, lambda c: c.draw())

    # Two whole tiles plus four generations into the third.
    assert job.generation == 2 * params.generations_per_target + 4
    assert job.tile_index == 2


def test_history_is_capped(tmp_path, target):
    params = tiny_run_params(generations=20000, preview_every=1)
    job = Job("abc", params, tmp_path / "src.png", tmp_path)
    job.mark_started()
    painting = build_painting(target, 4, seed=1)

    for generation in range(1, runner._MAX_HISTORY_POINTS + 50):
        job.record_generation(FakePopulation(painting, generation), 0, lambda c: c.draw())

    assert len(job.history) <= runner._MAX_HISTORY_POINTS


def test_preview_refreshes_only_when_due(tmp_path, target):
    params = tiny_run_params(preview_every=5)
    job = Job("abc", params, tmp_path / "src.png", tmp_path)
    job.mark_started()
    painting = build_painting(target, 4, seed=1)

    for generation in range(1, 11):
        job.record_generation(FakePopulation(painting, generation), 0, lambda c: c.draw())

    assert job.preview_version == 2


def test_job_state_is_safe_to_read_while_it_is_written(tmp_path, target):
    """Snapshots are taken from HTTP handlers while the worker mutates the job."""
    job = Job("abc", tiny_run_params(generations=100), tmp_path / "src.png", tmp_path)
    job.mark_started()
    painting = build_painting(target, 4, seed=1)
    stop = threading.Event()
    errors = []

    def writer():
        generation = 0
        while not stop.is_set():
            generation += 1
            try:
                job.record_generation(
                    FakePopulation(painting, generation), 0, lambda c: c.draw()
                )
            except Exception as error:  # pragma: no cover - only on a real bug
                errors.append(error)
                return

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            snapshot = job.snapshot()
            for entry in snapshot["history"]:
                assert set(entry) == {"generation", "similarity", "best"}
    finally:
        stop.set()
        thread.join(timeout=10)

    assert not errors


# --- JobManager ------------------------------------------------------------


def test_unknown_jobs_are_none(manager):
    assert manager.get("nope") is None
    assert manager.queue_position("nope") is None


def test_submitting_creates_a_run_directory(manager, source_image):
    job = manager.submit(tiny_run_params(), source_image)

    assert job.run_dir.is_dir()
    assert manager.get(job.id) is job
    assert [snapshot["id"] for snapshot in manager.list_jobs()] == [job.id]


def test_cancelling_a_queued_job_never_runs_it(manager, source_image):
    # Occupy the worker so the second job is still queued when it is cancelled.
    blocker = manager.submit(tiny_run_params(generations=200), source_image)
    queued = manager.submit(tiny_run_params(), source_image)

    assert manager.cancel(queued.id) is True
    assert queued.status == "cancelled"
    assert manager.queue_position(queued.id) is None

    manager.cancel(blocker.id)
    wait_for(blocker)
    assert queued.result_path is None


def test_cancelling_an_unknown_or_finished_job_is_a_no_op(manager, source_image):
    assert manager.cancel("nope") is False

    job = manager.submit(tiny_run_params(), source_image)
    wait_for(job)
    assert manager.cancel(job.id) is False


def test_removing_a_job_forgets_it(manager, source_image):
    job = manager.submit(tiny_run_params(), source_image)
    wait_for(job)

    manager.remove(job.id)

    assert manager.get(job.id) is None
    assert manager.list_jobs() == []


def test_old_finished_jobs_are_evicted(tmp_path, source_image):
    manager = JobManager(tmp_path / "jobs", max_jobs=3)
    jobs = []
    for _ in range(5):
        job = manager.submit(tiny_run_params(generations=1), source_image)
        wait_for(job)
        jobs.append(job)

    assert len(manager.list_jobs()) <= 3
    assert manager.get(jobs[-1].id) is not None


def test_a_missing_source_image_is_reported_as_an_error(manager, tmp_path):
    job = manager.submit(tiny_run_params(), tmp_path / "does-not-exist.png")

    wait_for(job)
    assert job.status == "error"
    assert "FileNotFoundError" in job.error
    assert manager.snapshot(job)["error"] == job.error


@pytest.mark.slow
def test_a_whole_single_mode_run_produces_a_result(manager, source_image):
    job = manager.submit(tiny_run_params(generations=3), source_image)

    assert wait_for(job) == "done", job.error
    assert job.result_path.is_file()
    assert job.target_path.is_file()
    assert job.preview_png is not None

    snapshot = manager.snapshot(job)
    assert snapshot["progress"] == 1.0
    assert snapshot["generation"] == job.params.total_generations
    assert snapshot["similarity"] > 0
    assert snapshot["history"]

    with Image.open(job.result_path) as result:
        assert result.size == Image.open(job.target_path).size


@pytest.mark.slow
def test_a_whole_tiled_run_stitches_every_tile(manager, source_image):
    params = tiny_run_params(
        mode="tiled", tile_rows=2, tile_cols=2, tile_overlap=4, generations=2
    )
    job = manager.submit(params, source_image)

    assert wait_for(job) == "done", job.error
    assert job.tile_index == 3
    with Image.open(job.result_path) as result, Image.open(job.target_path) as target:
        assert result.size == target.size


@pytest.mark.slow
def test_a_run_can_be_cancelled_mid_flight(manager, source_image):
    job = manager.submit(tiny_run_params(generations=5000), source_image)

    wait_for(job, statuses=("running",), timeout=60)
    # Let it get past seeding and into the evolution loop.
    deadline = time.monotonic() + 60
    while job.generation == 0 and time.monotonic() < deadline:
        time.sleep(0.02)

    assert manager.cancel(job.id) is True
    assert wait_for(job) == "cancelled"
    assert job.result_path is None


@pytest.mark.slow
def test_queued_runs_execute_one_at_a_time(manager, source_image):
    first = manager.submit(tiny_run_params(generations=4), source_image)
    second = manager.submit(tiny_run_params(generations=4), source_image)

    assert manager.queue_position(second.id) in (1, None)

    assert wait_for(first) == "done", first.error
    assert wait_for(second) == "done", second.error
    assert first.finished_at <= second.started_at


def test_run_cancelled_is_an_exception():
    assert issubclass(RunCancelled, Exception)

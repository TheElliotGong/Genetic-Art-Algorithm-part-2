"""Shared fixtures.

Two things have to happen before anything under test is imported, which is why
they run at module import time rather than inside a fixture:

  * ``webapp.app`` creates its data directory and starts its worker thread at
    import time, so ``VORONOI_WEB_DATA`` has to point somewhere disposable
    already, and
  * ``target_cache`` spools target pixels into the system temp directory, which
    a test run has no business littering.

Both are redirected into one temporary tree that is removed when the session
ends.
"""

import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "img"

# One disposable tree for everything the suite writes outside tmp_path.
_SESSION_TMP = Path(tempfile.mkdtemp(prefix="voronoi-tests-"))

os.environ["VORONOI_TARGET_CACHE"] = str(_SESSION_TMP / "targets")
os.environ["VORONOI_WEB_DATA"] = str(_SESSION_TMP / "web")
# Inherited values would change what the code under test does, so the suite runs
# from a known-clean environment regardless of the developer's shell.
for _name in ("VORONOI_EMBED_TARGET", "VORONOI_PROGRESS", "VORONOI_ALLOW_PRIVATE_URLS"):
    os.environ.pop(_name, None)


@pytest.fixture(scope="session", autouse=True)
def _clean_session_tmp():
    """Remove the session's scratch tree once the run finishes."""
    yield
    shutil.rmtree(_SESSION_TMP, ignore_errors=True)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Seed the RNGs the algorithm draws from, so failures are reproducible.

    Tests that need a specific genome seed explicitly; this only guarantees that
    a test which does not care still behaves the same way on every run.
    """
    random.seed(20260811)
    np.random.seed(20260811)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


def make_target(width: int = 48, height: int = 36) -> Image.Image:
    """Build a small synthetic target with structure the seeding code can find.

    A flat image would collapse to a single region and hide bugs in the
    palette/region path, so this has a horizontal gradient, two solid blocks and
    a hard edge between them.

    :param width: Target width in pixels.
    :param height: Target height in pixels.
    """
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    pixels[:, :, 0] = gradient[None, :]
    pixels[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    pixels[: height // 2, : width // 2] = (200, 30, 40)
    pixels[height // 2 :, width // 2 :] = (20, 90, 220)
    return Image.fromarray(pixels, mode="RGB").convert("RGBA")


@pytest.fixture(scope="session")
def target() -> Image.Image:
    """A small synthetic target image, shared by the whole session."""
    return make_target()


@pytest.fixture(scope="session")
def photo_target() -> Image.Image:
    """A real photograph, scaled down, for tests that want natural statistics."""
    path = SAMPLES_DIR / "car.jpeg"
    if not path.is_file():
        pytest.skip(f"sample image {path} is missing")
    image = Image.open(path).convert("RGBA")
    image.thumbnail((96, 96), Image.LANCZOS)
    return image


@pytest.fixture
def target_png(tmp_path: Path, target: Image.Image) -> Path:
    """The synthetic target written to disk as a PNG."""
    path = tmp_path / "target.png"
    target.convert("RGB").save(path, "PNG")
    return path


def build_painting(target_image: Image.Image, num_points: int = 60, seed: int = 0):
    """Build a painting with a reproducible random genome.

    A few points are pushed out of bounds, onto the exact edge, and onto a
    duplicate coordinate, so clipping and duplicate-seed handling are exercised
    by every test that uses this helper.

    :param target_image: The image the painting evolves towards.
    :param num_points: How many points the genome carries.
    :param seed: Seed for the genome's randomness.
    """
    from voronoi_painting import VoronoiPainting

    random.seed(seed)
    painting = VoronoiPainting(num_points, target_image, background_color=(128, 128, 128))
    if num_points >= 4:
        painting.points[0].coordinates = (-25, -40)
        painting.points[1].coordinates = (10**6, 10**6)
        painting.points[2].coordinates = painting.points[3].coordinates
    return painting


@pytest.fixture
def painting_factory(target: Image.Image):
    """Factory building reproducible paintings against the session target."""

    def factory(num_points: int = 60, seed: int = 0, target_image: Image.Image = None):
        return build_painting(target_image or target, num_points=num_points, seed=seed)

    return factory


@pytest.fixture
def isolated_target_cache(tmp_path: Path, monkeypatch):
    """Run a test against an empty target registry and a private spool directory.

    ``target_cache`` keeps process-global state on purpose, so a test that pokes
    at it has to put the module back the way it found it - otherwise paintings
    built by earlier tests would lose their targets.
    """
    import target_cache

    saved = (
        dict(target_cache._TARGETS),
        set(target_cache._SPOOLED),
        set(target_cache._SPOOL_ATTEMPTED),
    )
    spool_dir = tmp_path / "spool"
    monkeypatch.setenv("VORONOI_TARGET_CACHE", str(spool_dir))
    target_cache._TARGETS.clear()
    target_cache._SPOOLED.clear()
    target_cache._SPOOL_ATTEMPTED.clear()
    try:
        yield spool_dir
    finally:
        target_cache._TARGETS.clear()
        target_cache._TARGETS.update(saved[0])
        target_cache._SPOOLED.clear()
        target_cache._SPOOLED.update(saved[1])
        target_cache._SPOOL_ATTEMPTED.clear()
        target_cache._SPOOL_ATTEMPTED.update(saved[2])


def tiny_run_params(**overrides):
    """``RunParams`` small enough that a full run finishes in a second or two.

    :param overrides: Fields to change from the fast defaults.
    """
    from webapp.params import RunParams

    defaults = dict(
        mode="single",
        generations=2,
        population_size=6,
        num_points=8,
        max_dimension=64,  # the model's floor; anything smaller is rejected
        render_scale=1,
        workers=1,
        preview_every=1,
        initial_colors=8,
        final_colors=4,
        tile_rows=1,
        tile_cols=1,
        tile_overlap=4,
    )
    defaults.update(overrides)
    return RunParams(**defaults)

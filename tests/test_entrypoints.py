"""Import-level smoke tests for the scripts a user actually runs.

These catch the failure mode unit tests miss entirely: a module that no longer
imports, or a ``__main__`` block that executes work on import. The evolution
drivers kick off multi-hour runs at import time if their guard is ever lost, so
they are imported in a subprocess with a short timeout rather than in-process.
"""

import subprocess
import sys
import textwrap

import pytest

MODULES = [
    "voronoi_painting",
    "target_cache",
    "evolve_voronoi",
    "evolve_tiled",
    "benchmark",
    "run_web",
    "webapp.app",
    "webapp.runner",
    "webapp.params",
    "webapp.imaging",
]


def run_python(repo_root, code: str, timeout: int = 120):
    """Run ``code`` in a fresh interpreter rooted at the repository."""
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(repo_root)!r})
        """
    ) + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(repo_root),
    )


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_doing_work(repo_root, module):
    result = run_python(repo_root, f"import {module}")
    assert result.returncode == 0, result.stderr


def test_run_web_help_does_not_start_a_server(repo_root):
    result = subprocess.run(
        [sys.executable, "run_web.py", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(repo_root),
    )

    assert result.returncode == 0, result.stderr
    assert "--host" in result.stdout
    assert "--port" in result.stdout


def test_benchmark_help_lists_its_options(repo_root):
    result = subprocess.run(
        [sys.executable, "benchmark.py", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(repo_root),
    )

    assert result.returncode == 0, result.stderr
    assert "--image" in result.stdout


def test_the_bundled_samples_are_readable(repo_root):
    """The web UI and both drivers point at these paths."""
    from PIL import Image

    samples = sorted((repo_root / "img").glob("*.*"))
    assert samples

    for path in samples:
        with Image.open(path) as image:
            image.verify()

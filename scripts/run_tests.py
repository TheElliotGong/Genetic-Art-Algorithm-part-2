"""Run the test suite the same way on every platform.

    python scripts/run_tests.py              # the whole suite
    python scripts/run_tests.py --fast       # skip the real-evolution tests
    python scripts/run_tests.py --coverage   # with a coverage report
    python scripts/run_tests.py -- -k render # anything after -- goes to pytest

This exists so the CI workflow, the git hook and a developer at a terminal all
invoke pytest identically, and so a missing dev dependency produces one clear
message instead of a traceback.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_pytest_is_installed() -> None:
    """Fail with an actionable message rather than an ImportError."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        print(
            "pytest is not installed in this interpreter.\n"
            f"  {Path(sys.executable).name} -m pip install -r requirements-dev.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip tests marked slow (the ones that run a real evolution loop).",
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Also produce a coverage report."
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed straight through to pytest, after a bare --.",
    )
    args = parser.parse_args()

    ensure_pytest_is_installed()

    command = [sys.executable, "-m", "pytest"]
    if args.fast:
        command += ["-m", "not slow"]
    if args.coverage:
        command += ["--cov", "--cov-report=term-missing"]
    # argparse.REMAINDER keeps the separating "--", which pytest would treat as
    # a file name.
    command += [argument for argument in args.pytest_args if argument != "--"]

    print(" ".join(command))
    return subprocess.call(command, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

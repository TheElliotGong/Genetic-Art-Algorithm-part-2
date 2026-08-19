"""Install this repository's git hooks.

    python scripts/install_hooks.py            # install
    python scripts/install_hooks.py --list     # show what is installed
    python scripts/install_hooks.py --uninstall

Hooks live in ``scripts/hooks/`` so they are version controlled; git only ever
looks in ``.git/hooks``, which is not, so they have to be copied in. Existing
hooks that this script did not write are left alone unless ``--force`` is given.
"""

import argparse
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path(__file__).resolve().parent / "hooks"

# Written into every installed hook so the installer can recognise its own work.
MARKER = "# managed by scripts/install_hooks.py"


def hooks_dir() -> Path:
    """Where git expects hooks for this checkout (honours core.hooksPath)."""
    try:
        configured = subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        print("This does not look like a git checkout.", file=sys.stderr)
        raise SystemExit(2)

    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def install(force: bool) -> int:
    destination_dir = hooks_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)
    installed = 0

    for source in sorted(SOURCE_DIR.iterdir()):
        if source.is_dir():
            continue
        destination = destination_dir / source.name

        if destination.exists() and not force:
            existing = destination.read_text(encoding="utf-8", errors="replace")
            if MARKER not in existing:
                print(
                    f"skipped {destination}: a hook is already installed there. "
                    "Re-run with --force to replace it."
                )
                continue

        body = source.read_text(encoding="utf-8")
        lines = body.splitlines()
        # Keep the shebang first, then the marker.
        if lines and lines[0].startswith("#!"):
            body = "\n".join([lines[0], MARKER, *lines[1:]]) + "\n"
        else:
            body = MARKER + "\n" + body

        destination.write_text(body, encoding="utf-8", newline="\n")
        destination.chmod(destination.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)
        print(f"installed {destination}")
        installed += 1

    if installed:
        print("\nHooks installed. Bypass one push with 'git push --no-verify'.")
    return 0


def uninstall() -> int:
    destination_dir = hooks_dir()
    for source in sorted(SOURCE_DIR.iterdir()):
        destination = destination_dir / source.name
        if not destination.is_file():
            continue
        if MARKER not in destination.read_text(encoding="utf-8", errors="replace"):
            print(f"skipped {destination}: not installed by this script.")
            continue
        destination.unlink()
        print(f"removed {destination}")
    return 0


def show() -> int:
    destination_dir = hooks_dir()
    print(f"git hooks directory: {destination_dir}")
    for source in sorted(SOURCE_DIR.iterdir()):
        destination = destination_dir / source.name
        if not destination.is_file():
            state = "not installed"
        elif MARKER in destination.read_text(encoding="utf-8", errors="replace"):
            state = "installed"
        else:
            state = "present, but not installed by this script"
        print(f"  {source.name}: {state}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Overwrite foreign hooks.")
    parser.add_argument("--uninstall", action="store_true", help="Remove the hooks.")
    parser.add_argument("--list", action="store_true", help="Show hook status.")
    args = parser.parse_args()

    if not SOURCE_DIR.is_dir():
        print(f"No hooks to install: {SOURCE_DIR} is missing.", file=sys.stderr)
        return 2
    if args.list:
        return show()
    if args.uninstall:
        return uninstall()
    return install(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())

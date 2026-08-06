"""
Process-global registry for target images.

Every ``VoronoiPainting`` needs the target image to score itself, but the target
is identical for every individual in a population and never changes during a
run. Storing it on each painting means that:

  * every ``deepcopy`` in the evolution operators duplicates the full image, and
  * every individual shipped to a worker process carries a full copy of the
    image (``evol`` serializes whole individuals on each ``evaluate``).

For a 250 individual population that is hundreds of megabytes of serialization
per generation, which dwarfs the actual rendering work.

This module keeps the pixels in one place per process. A painting only stores a
short content key; the pixels are looked up here. Because the key is derived
from the image content, two paintings built from the same image share one entry,
and a worker process rehydrates the image once and reuses it for every
individual it ever receives.

Worker processes are populated in one of two ways:

  * on fork-based platforms (Linux/macOS) they simply inherit this module's
    state, and
  * on spawn-based platforms (Windows) they load the pixels from a small
    content-addressed spool file written to the system temp directory.

Set ``VORONOI_TARGET_CACHE`` to control where spool files are written, or set
``VORONOI_EMBED_TARGET=1`` to disable spooling and embed the pixels in every
serialized painting (slower, but fully self-contained - useful if you want
population checkpoints to stay loadable on another machine).
"""

import hashlib
import os
import pickle
import tempfile

import numpy as np
from PIL import Image

# Maps content key -> _TargetEntry for this process.
_TARGETS = {}

# Content keys known to have a readable spool file on disk.
_SPOOLED = set()

# Content keys we have already tried (and failed) to spool, so that a read-only
# temp directory costs one failed write per run rather than one per evaluation.
_SPOOL_ATTEMPTED = set()

# Attribute used to memoize the content key on a PIL image, so that repeated
# registrations of the same image object never re-hash the pixel data.
_KEY_ATTRIBUTE = "_voronoi_target_key"


def _embed_targets() -> bool:
    """Whether to embed pixels in serialized paintings instead of spooling."""
    return os.environ.get("VORONOI_EMBED_TARGET", "").strip() not in ("", "0")


def _spool_dir() -> str:
    """Directory used for content-addressed spool files."""
    return os.environ.get(
        "VORONOI_TARGET_CACHE", os.path.join(tempfile.gettempdir(), "voronoi-targets")
    )


class _TargetEntry:
    """A registered target image together with the arrays derived from it.

    :param image: The PIL image used as the evolution target.
    """

    __slots__ = ("image", "rgb", "payload")

    def __init__(self, image: Image.Image):
        self.image = image
        # Scoring always compares against RGB, so convert once per process
        # instead of once per painting.
        self.rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        self.payload = None

    def as_payload(self) -> dict:
        """Return a compact, picklable representation of the raw pixels."""
        if self.payload is None:
            self.payload = {
                "mode": self.image.mode,
                "size": self.image.size,
                "bytes": self.image.tobytes(),
            }
        return self.payload


def _entry_from_payload(payload: dict) -> _TargetEntry:
    """Rebuild a registry entry from the raw pixel payload."""
    image = Image.frombytes(payload["mode"], payload["size"], payload["bytes"])
    return _TargetEntry(image)


def content_key(image: Image.Image) -> str:
    """Return a stable content key for ``image``, memoized on the image itself.

    :param image: The PIL image to key.
    """
    key = getattr(image, _KEY_ATTRIBUTE, None)
    if key is not None:
        return key

    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"{image.mode}|{image.size[0]}x{image.size[1]}|".encode())
    digest.update(image.tobytes())
    key = digest.hexdigest()

    try:
        setattr(image, _KEY_ATTRIBUTE, key)
    except AttributeError:  # pragma: no cover - PIL images accept attributes
        pass

    return key


def _spool_path(key: str) -> str:
    return os.path.join(_spool_dir(), f"{key}.pkl")


def _write_spool(key: str, entry: _TargetEntry) -> bool:
    """Write the pixels for ``key`` to disk so spawned workers can load them.

    :param key: Content key of the target.
    :param entry: Registry entry holding the pixels.
    :return: True if a readable spool file exists afterwards.
    """
    if _embed_targets():
        return False

    path = _spool_path(key)
    if os.path.exists(path):
        _SPOOLED.add(key)
        return True

    try:
        os.makedirs(_spool_dir(), exist_ok=True)
        # Write to a process-unique temporary name and rename into place, so a
        # partially written file is never visible to a concurrent reader.
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "wb") as handle:
            pickle.dump(entry.as_payload(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    except OSError:
        # A read-only or full temp directory is not fatal: paintings fall back
        # to embedding the pixels in their serialized state.
        return False

    _SPOOLED.add(key)
    return True


def _read_spool(key: str):
    """Load a spooled target, or return None when it is not on disk."""
    path = _spool_path(key)
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None

    _SPOOLED.add(key)
    return _entry_from_payload(payload)


def register(image: Image.Image) -> str:
    """Register ``image`` in this process and return its content key.

    Registering the same image twice is cheap: the key is memoized on the image
    object and the registry is keyed by content.

    :param image: The PIL image to register as an evolution target.
    """
    key = content_key(image)

    entry = _TARGETS.get(key)
    if entry is None:
        entry = _TargetEntry(image)
        _TARGETS[key] = entry

    if key not in _SPOOLED and key not in _SPOOL_ATTEMPTED:
        _SPOOL_ATTEMPTED.add(key)
        _write_spool(key, entry)

    return key


def is_portable(key: str) -> bool:
    """Whether ``key`` can be resolved by another process without help."""
    return key in _SPOOLED


def payload_for(key: str):
    """Return the raw pixel payload for ``key``, or None if it is unknown."""
    entry = _TARGETS.get(key)
    return None if entry is None else entry.as_payload()


def resolve(key: str, payload=None) -> _TargetEntry:
    """Return the registry entry for ``key``, loading it if necessary.

    :param key: Content key of the target.
    :param payload: Optional raw pixel payload to rebuild from, used when the
        target was serialized with its pixels embedded.
    """
    entry = _TARGETS.get(key)
    if entry is not None:
        return entry

    if payload is not None:
        entry = _entry_from_payload(payload)
    else:
        entry = _read_spool(key)

    if entry is None:
        raise KeyError(
            f"Target image {key} is not available in this process and no spool "
            f"file was found in {_spool_dir()}. If you are loading a checkpoint "
            f"written on another machine (or after the temp directory was "
            f"cleared), re-run with VORONOI_EMBED_TARGET=1 so that target "
            f"pixels are stored inside the pickle."
        )

    _TARGETS[key] = entry
    return entry


def image_for(key: str) -> Image.Image:
    """Return the PIL image registered under ``key``."""
    return resolve(key).image


def rgb_for(key: str) -> np.ndarray:
    """Return the cached RGB pixel array registered under ``key``."""
    return resolve(key).rgb


def rgb_of(image: Image.Image) -> np.ndarray:
    """Return the cached RGB array for ``image``, registering it if needed.

    This is on the scoring path and runs once per individual per generation, so
    the already-registered case is kept to a memoized attribute read and a dict
    lookup.

    :param image: A PIL image, typically the evolution target.
    """
    key = getattr(image, _KEY_ATTRIBUTE, None)
    if key is not None:
        entry = _TARGETS.get(key)
        if entry is not None:
            return entry.rgb
    return resolve(register(image)).rgb

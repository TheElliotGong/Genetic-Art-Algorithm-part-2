"""Tests for the process-global target image registry.

The registry is what keeps target pixels off the wire when individuals are
shipped to worker processes, so the properties that matter are: the same image
maps to one entry, another process can rehydrate it, and a target that cannot be
rehydrated fails loudly instead of silently scoring against the wrong pixels.
"""

import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image

import target_cache
from tests.conftest import build_painting, make_target

pytestmark = pytest.mark.usefixtures("isolated_target_cache")


def test_same_content_gets_one_entry():
    first = make_target(24, 18)
    second = make_target(24, 18)

    key_a = target_cache.register(first)
    key_b = target_cache.register(second)

    assert key_a == key_b
    assert len(target_cache._TARGETS) == 1


def test_different_content_gets_different_keys():
    assert target_cache.content_key(make_target(24, 18)) != target_cache.content_key(
        make_target(18, 24)
    )
    assert target_cache.content_key(
        Image.new("RGB", (8, 8), (1, 2, 3))
    ) != target_cache.content_key(Image.new("RGB", (8, 8), (1, 2, 4)))


def test_content_key_is_memoized_on_the_image():
    image = make_target(16, 16)
    key = target_cache.content_key(image)
    assert getattr(image, target_cache._KEY_ATTRIBUTE) == key
    assert target_cache.content_key(image) is key


def test_rgb_arrays_are_cached_and_shared():
    image = make_target(16, 16)
    first = target_cache.rgb_of(image)
    second = target_cache.rgb_of(image)

    assert first is second
    assert first.dtype == np.uint8
    assert first.shape == (16, 16, 3)
    assert np.array_equal(first, np.array(image.convert("RGB")))


def test_image_for_and_rgb_for_resolve_a_registered_key():
    image = make_target(20, 12)
    key = target_cache.register(image)

    assert target_cache.image_for(key) is image
    assert np.array_equal(target_cache.rgb_for(key), np.array(image.convert("RGB")))


def test_registering_writes_a_spool_file(isolated_target_cache):
    key = target_cache.register(make_target(16, 16))

    assert target_cache.is_portable(key)
    assert (isolated_target_cache / f"{key}.pkl").is_file()


def test_an_existing_spool_file_is_reused(isolated_target_cache):
    """A second process registering the same target must not rewrite the file."""
    image = make_target(16, 16)
    key = target_cache.register(image)
    path = isolated_target_cache / f"{key}.pkl"
    written_at = path.stat().st_mtime_ns

    target_cache._SPOOLED.clear()
    target_cache._SPOOL_ATTEMPTED.clear()
    target_cache.register(image)

    assert path.stat().st_mtime_ns == written_at
    assert target_cache.is_portable(key)


def test_a_cleared_registry_reloads_from_the_spool_file():
    image = make_target(16, 16)
    key = target_cache.register(image)

    # Simulate a fresh process that has the spool file but no in-memory entry.
    target_cache._TARGETS.clear()
    restored = target_cache.resolve(key)

    assert np.array_equal(restored.rgb, np.array(image.convert("RGB")))


def test_a_missing_target_raises_a_helpful_error():
    with pytest.raises(KeyError, match="VORONOI_EMBED_TARGET"):
        target_cache.resolve("0" * 32)


def test_payload_round_trip_rebuilds_the_pixels():
    image = make_target(16, 16)
    key = target_cache.register(image)
    payload = target_cache.payload_for(key)

    target_cache._TARGETS.clear()
    target_cache._SPOOLED.clear()
    restored = target_cache.resolve(key, payload=payload)

    assert np.array_equal(restored.rgb, np.array(image.convert("RGB")))


def test_payload_for_an_unknown_key_is_none():
    assert target_cache.payload_for("f" * 32) is None


def test_an_unwritable_spool_directory_is_not_fatal(monkeypatch, tmp_path):
    """A read-only temp directory must degrade to embedding, not crash."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    monkeypatch.setenv("VORONOI_TARGET_CACHE", str(blocker / "spool"))

    key = target_cache.register(make_target(16, 16))

    assert not target_cache.is_portable(key)
    assert target_cache.payload_for(key) is not None


def test_paintings_embed_their_target_when_spooling_is_off(monkeypatch, target):
    """With no spool file to fall back on, serialization must stay lossless."""
    monkeypatch.setenv("VORONOI_EMBED_TARGET", "1")
    painting = build_painting(target, 12, seed=1)

    assert painting.__getstate__()["target_payload"] is not None
    blob = pickle.dumps(painting, protocol=pickle.HIGHEST_PROTOCOL)

    # Nothing in this process remembers the target any more, so the pickle is
    # the only source of the pixels.
    target_cache._TARGETS.clear()
    restored = pickle.loads(blob)

    assert np.array_equal(
        np.array(restored.target_image.convert("RGB")), np.array(target.convert("RGB"))
    )


def test_a_serialized_painting_carries_no_pixels(monkeypatch, photo_target):
    """The whole point of the registry: individuals stay small on the wire."""
    painting = build_painting(photo_target, 250, seed=2)
    spooled = pickle.dumps(painting, protocol=pickle.HIGHEST_PROTOCOL)

    pixels = photo_target.tobytes()
    assert pixels not in spooled

    monkeypatch.setenv("VORONOI_EMBED_TARGET", "1")
    target_cache._SPOOLED.clear()
    embedded = pickle.dumps(painting, protocol=pickle.HIGHEST_PROTOCOL)

    assert pixels in embedded
    assert len(spooled) < len(embedded) - len(pixels) / 2


def test_another_process_can_rehydrate_a_pickled_painting(tmp_path, target, repo_root):
    """The spawn path: a fresh interpreter must resolve the target from disk.

    Windows starts worker processes with ``spawn``, so this is exactly what
    ``evol`` does to every individual on every generation. The child inherits
    ``VORONOI_TARGET_CACHE`` and has nothing but the spool file to work from.
    """
    painting = build_painting(target, 20, seed=3)
    blob = tmp_path / "painting.pkl"
    blob.write_bytes(pickle.dumps(painting, protocol=pickle.HIGHEST_PROTOCOL))
    expected = painting.image_diff(target)

    script = textwrap.dedent(
        f"""
        import pickle
        import sys

        sys.path.insert(0, {str(repo_root)!r})

        with open({str(blob)!r}, "rb") as handle:
            painting = pickle.load(handle)
        print(painting.image_diff(painting.target_image))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    assert float(result.stdout.strip()) == pytest.approx(expected)

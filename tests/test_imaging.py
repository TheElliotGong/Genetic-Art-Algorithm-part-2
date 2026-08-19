"""Tests for target-image loading, scaling and the URL fetch guard.

Nothing here touches the network: the only host names used resolve locally, and
the one test that exercises a successful download stubs ``urlopen``.
"""

import io
import socket

import pytest
from PIL import Image

from webapp import imaging
from webapp.imaging import (
    ImageLoadError,
    decode_image,
    fit_within,
    load_image_from_url,
    to_png_bytes,
)


def png_bytes(size=(20, 15), color=(10, 20, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


# --- Decoding --------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP"])
def test_common_formats_decode_to_rgba(fmt):
    buffer = io.BytesIO()
    Image.new("RGB", (24, 18), (200, 100, 50)).save(buffer, format=fmt)

    image = decode_image(buffer.getvalue())

    assert image.mode == "RGBA"
    assert image.size == (24, 18)


def test_greyscale_and_palette_images_are_converted():
    for mode in ("L", "P"):
        buffer = io.BytesIO()
        Image.new(mode, (10, 10)).save(buffer, format="PNG")
        assert decode_image(buffer.getvalue()).mode == "RGBA"


def test_empty_input_is_rejected():
    with pytest.raises(ImageLoadError, match="empty"):
        decode_image(b"")


def test_garbage_input_is_rejected():
    with pytest.raises(ImageLoadError, match="could not be read"):
        decode_image(b"this is definitely not a PNG")


def test_oversized_input_is_rejected(monkeypatch):
    monkeypatch.setattr(imaging, "MAX_IMAGE_BYTES", 128)
    with pytest.raises(ImageLoadError, match="larger than"):
        decode_image(png_bytes((64, 64)))


def test_exif_orientation_is_applied():
    """A phone photo tagged as rotated must come back upright."""
    buffer = io.BytesIO()
    image = Image.new("RGB", (30, 10), (255, 0, 0))
    exif = image.getexif()
    exif[274] = 6  # rotate 90 degrees
    image.save(buffer, format="JPEG", exif=exif)

    assert decode_image(buffer.getvalue()).size == (10, 30)


# --- Scaling ---------------------------------------------------------------


def test_fit_within_scales_the_longest_edge_down():
    image = Image.new("RGB", (800, 400))
    assert fit_within(image, 200).size == (200, 100)

    image = Image.new("RGB", (400, 800))
    assert fit_within(image, 200).size == (100, 200)


def test_fit_within_leaves_small_images_alone():
    image = Image.new("RGB", (100, 50))
    assert fit_within(image, 400) is image


def test_fit_within_never_produces_a_zero_dimension():
    image = Image.new("RGB", (1000, 3))
    assert fit_within(image, 64).size == (64, 1)


def test_to_png_bytes_round_trips():
    image = Image.new("RGBA", (12, 9), (1, 2, 3, 255))
    decoded = Image.open(io.BytesIO(to_png_bytes(image)))

    assert decoded.format == "PNG"
    assert decoded.size == (12, 9)
    assert decoded.convert("RGB").getpixel((0, 0)) == (1, 2, 3)


# --- URL guard -------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/a.png", "gopher://example.com"],
)
def test_non_http_schemes_are_refused(url):
    with pytest.raises(ImageLoadError, match="http"):
        load_image_from_url(url)


def test_urls_without_a_host_are_refused():
    with pytest.raises(ImageLoadError, match="no host"):
        load_image_from_url("http:///just-a-path.png")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/image.png",
        "http://localhost:8000/image.png",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.5/image.png",
        "http://192.168.1.1/image.png",
    ],
)
def test_private_addresses_are_blocked_by_default(url):
    """The server fetches whatever it is handed, so SSRF targets must be refused."""
    with pytest.raises(ImageLoadError, match="private address"):
        load_image_from_url(url)


def test_unresolvable_hosts_report_cleanly(monkeypatch):
    def boom(*args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(imaging.socket, "getaddrinfo", boom)
    with pytest.raises(ImageLoadError, match="Could not resolve"):
        load_image_from_url("http://nonexistent.invalid/a.png")


class FakeResponse:
    """Minimal stand-in for the object ``urlopen`` returns."""

    def __init__(self, data: bytes, headers=None):
        self._data = data
        self.headers = headers or {}

    def read(self, amount=None):
        return self._data if amount is None else self._data[:amount]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_a_successful_download_is_decoded(monkeypatch):
    monkeypatch.setenv("VORONOI_ALLOW_PRIVATE_URLS", "1")
    payload = png_bytes((40, 20), (7, 8, 9))
    monkeypatch.setattr(
        imaging.urllib.request,
        "urlopen",
        lambda request, timeout=None: FakeResponse(payload),
    )

    image = load_image_from_url("http://127.0.0.1/image.png")

    assert image.size == (40, 20)
    assert image.mode == "RGBA"


def test_a_declared_oversize_content_length_is_refused(monkeypatch):
    monkeypatch.setenv("VORONOI_ALLOW_PRIVATE_URLS", "1")
    monkeypatch.setattr(
        imaging.urllib.request,
        "urlopen",
        lambda request, timeout=None: FakeResponse(
            b"", {"Content-Length": str(imaging.MAX_IMAGE_BYTES + 1)}
        ),
    )

    with pytest.raises(ImageLoadError, match="larger than"):
        load_image_from_url("http://127.0.0.1/huge.png")


def test_an_undeclared_oversize_body_is_refused(monkeypatch):
    """No Content-Length header, so the cap has to be enforced on what arrives."""
    monkeypatch.setenv("VORONOI_ALLOW_PRIVATE_URLS", "1")
    monkeypatch.setattr(imaging, "MAX_IMAGE_BYTES", 64)
    monkeypatch.setattr(
        imaging.urllib.request,
        "urlopen",
        lambda request, timeout=None: FakeResponse(b"x" * 512),
    )

    with pytest.raises(ImageLoadError, match="larger than"):
        load_image_from_url("http://127.0.0.1/huge.png")


def test_http_errors_are_reported(monkeypatch):
    monkeypatch.setenv("VORONOI_ALLOW_PRIVATE_URLS", "1")

    def raise_http_error(request, timeout=None):
        raise imaging.urllib.error.HTTPError(
            "http://127.0.0.1/missing.png", 404, "Not Found", {}, None
        )

    monkeypatch.setattr(imaging.urllib.request, "urlopen", raise_http_error)

    with pytest.raises(ImageLoadError, match="HTTP 404"):
        load_image_from_url("http://127.0.0.1/missing.png")


def test_transport_errors_are_reported(monkeypatch):
    monkeypatch.setenv("VORONOI_ALLOW_PRIVATE_URLS", "1")

    def raise_url_error(request, timeout=None):
        raise imaging.urllib.error.URLError("connection refused")

    monkeypatch.setattr(imaging.urllib.request, "urlopen", raise_url_error)

    with pytest.raises(ImageLoadError, match="Could not fetch"):
        load_image_from_url("http://127.0.0.1/image.png")


def test_the_private_address_guard_can_be_lifted(monkeypatch):
    monkeypatch.setenv("VORONOI_ALLOW_PRIVATE_URLS", "1")
    # No exception means the guard let it through; the fetch itself is stubbed.
    imaging._check_url_is_fetchable("http://127.0.0.1/image.png")

"""Loading and preparing target images supplied through the web interface."""

import io
import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, ImageOps, UnidentifiedImageError

# Images are decoded fully in memory, so the transfer is capped well below what
# the algorithm could sensibly consume anyway.
MAX_IMAGE_BYTES = 25 * 1024 * 1024

# Downloads are a user-initiated action in a UI, not a background job, so a
# short timeout is better than a stalled request.
DOWNLOAD_TIMEOUT_SECONDS = 15

# Pillow refuses very large images by default to avoid decompression bombs. The
# limit here is deliberately generous but still finite.
Image.MAX_IMAGE_PIXELS = 80_000_000


class ImageLoadError(Exception):
    """Raised when a user-supplied image cannot be fetched or decoded."""


def _private_urls_allowed() -> bool:
    """Whether fetching URLs that resolve to private addresses is permitted.

    Blocked by default: the server fetches whatever URL it is handed, so on a
    shared or exposed deployment an unrestricted fetch turns the app into a
    probe for hosts that are only reachable from the server itself. Set
    ``VORONOI_ALLOW_PRIVATE_URLS=1`` when running against a LAN image host.
    """
    return os.environ.get("VORONOI_ALLOW_PRIVATE_URLS", "").strip() not in ("", "0")


def _check_url_is_fetchable(url: str) -> None:
    """Validate the scheme and resolved address of ``url``.

    :param url: The URL the user asked the server to fetch.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ImageLoadError("Only http:// and https:// image URLs are supported.")
    if not parsed.hostname:
        raise ImageLoadError("That URL has no host to fetch from.")

    if _private_urls_allowed():
        return

    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as error:
        raise ImageLoadError(f"Could not resolve {parsed.hostname}.") from error

    for family, _, _, _, sockaddr in addresses:
        address = ipaddress.ip_address(sockaddr[0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
        ):
            raise ImageLoadError(
                f"{parsed.hostname} resolves to a private address. Set "
                "VORONOI_ALLOW_PRIVATE_URLS=1 to allow this."
            )


def load_image_from_url(url: str) -> Image.Image:
    """Download ``url`` and decode it as an image.

    :param url: An http(s) URL pointing at an image file.
    """
    url = url.strip()
    _check_url_is_fetchable(url)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "voronoi-genetic-art/1.0", "Accept": "image/*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            declared_length = response.headers.get("Content-Length")
            if declared_length and int(declared_length) > MAX_IMAGE_BYTES:
                raise ImageLoadError(
                    f"That image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
                )
            # Read one byte past the cap so an oversized body without a
            # Content-Length header is still caught.
            data = response.read(MAX_IMAGE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise ImageLoadError(f"The server returned HTTP {error.code} for that URL.") from error
    except (urllib.error.URLError, socket.timeout, OSError) as error:
        raise ImageLoadError(f"Could not fetch that URL: {error}") from error

    if len(data) > MAX_IMAGE_BYTES:
        raise ImageLoadError(
            f"That image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )

    return decode_image(data)


def decode_image(data: bytes) -> Image.Image:
    """Decode raw bytes into an RGBA image with EXIF rotation applied.

    :param data: The raw bytes of an image file.
    """
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageLoadError(
            f"That image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )
    if not data:
        raise ImageLoadError("That file is empty.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageLoadError("That file could not be read as an image.") from error

    # Phone photos carry their orientation in EXIF; without this the target is
    # evolved sideways.
    image = ImageOps.exif_transpose(image)
    return image.convert("RGBA")


def fit_within(image: Image.Image, max_dimension: int) -> Image.Image:
    """Scale ``image`` down so its longest edge is at most ``max_dimension``.

    Images already inside the bound are returned unchanged - the algorithm's
    cost scales with pixel count, so upscaling would only waste time.

    :param image: The image to bound.
    :param max_dimension: The longest edge allowed, in pixels.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_dimension:
        return image

    ratio = max_dimension / longest
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    return image.resize(size, Image.LANCZOS)


def to_png_bytes(image: Image.Image) -> bytes:
    """Encode an image as PNG bytes."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()

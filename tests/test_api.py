"""End-to-end tests for the HTTP surface.

``webapp.app`` builds its data directory and job manager at import time, so the
suite's conftest points ``VORONOI_WEB_DATA`` at a temporary tree before this
module is imported.
"""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import tiny_run_params
from webapp.app import app

TERMINAL = ("done", "error", "cancelled")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def png_bytes(size=(80, 60), color=(120, 40, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def upload_png(client, size=(80, 60)) -> str:
    response = client.post(
        "/api/uploads",
        files={"file": ("target.png", png_bytes(size), "image/png")},
    )
    assert response.status_code == 200, response.text
    return response.json()["image_id"]


def run_to_completion(client, image_id, timeout=180.0, **overrides) -> dict:
    """Start a tiny run and poll until it reaches a terminal state."""
    body = tiny_run_params(**overrides).model_dump()
    body["image_id"] = image_id
    created = client.post("/api/jobs", json=body)
    assert created.status_code == 200, created.text
    job_id = created.json()["id"]

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        if snapshot["status"] in TERMINAL:
            return snapshot
        time.sleep(0.05)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


# --- Static surface --------------------------------------------------------


def test_index_serves_the_single_page_app(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


def test_favicon_is_served(client):
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"


def test_static_assets_are_mounted(client):
    for path in ("/static/app.js", "/static/styles.css"):
        assert client.get(path).status_code == 200


def test_openapi_document_is_generated(client):
    document = client.get("/openapi.json").json()
    assert "/api/jobs" in document["paths"]


# --- Schema and samples ----------------------------------------------------

def test_schema_describes_every_parameter(client):
    fields = client.get("/api/schema").json()["fields"]

    from webapp.params import RunParams

    assert set(fields) == set(RunParams.model_fields)
    assert fields["generations"]["ge"] == 1
    assert fields["generations"]["le"] == 20000
    assert fields["generations"]["default"] == RunParams().generations
    assert all(field["description"] for field in fields.values())


def test_schema_is_json_serialisable(client):
    """The front-end reads this straight off the wire."""
    json.dumps(client.get("/api/schema").json())


def test_samples_are_listed(client):
    samples = client.get("/api/samples").json()["samples"]
    names = {sample["name"] for sample in samples}

    assert names
    assert "car.jpeg" in names
    assert all(name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")) for name in names)


def test_a_missing_samples_directory_is_not_an_error(client, monkeypatch, tmp_path):
    """A deployment without the bundled images still has to serve the page."""
    from webapp import app as app_module

    monkeypatch.setattr(app_module, "SAMPLES_DIR", tmp_path / "gone")
    assert client.get("/api/samples").json() == {"samples": []}


def test_an_undecodable_sample_is_a_client_error(client, monkeypatch):
    from webapp import app as app_module
    from webapp.imaging import ImageLoadError

    def boom(_data):
        raise ImageLoadError("that file could not be read as an image")

    monkeypatch.setattr(app_module, "decode_image", boom)

    response = client.post("/api/uploads/sample", json={"name": "car.jpeg"})
    assert response.status_code == 400
    assert "image" in response.json()["detail"]


def test_a_sample_image_can_be_fetched(client):
    response = client.get("/api/samples/car.jpeg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


@pytest.mark.parametrize(
    "name", ["nope.png", "..%2FREADME.md", "..%5CREADME.md", "%2Fetc%2Fpasswd"]
)
def test_unknown_or_escaping_sample_names_are_refused(client, name):
    assert client.get(f"/api/samples/{name}").status_code == 404


# --- Uploads ---------------------------------------------------------------


def test_uploading_a_file_returns_a_descriptor(client):
    response = client.post(
        "/api/uploads",
        files={"file": ("photo.png", png_bytes((40, 25)), "image/png")},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["width"] == 40
    assert body["height"] == 25
    assert body["label"] == "photo.png"
    assert body["url"] == f"/api/uploads/{body['image_id']}.png"


def test_an_uploaded_image_can_be_fetched_back(client):
    image_id = upload_png(client, (30, 20))
    response = client.get(f"/api/uploads/{image_id}.png")

    assert response.status_code == 200
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (30, 20)


def test_a_supplied_label_wins(client):
    response = client.post(
        "/api/uploads",
        files={"file": ("photo.png", png_bytes(), "image/png")},
        data={"label": "my picture"},
    )
    assert response.json()["label"] == "my picture"


def test_uploading_a_non_image_is_a_client_error(client):
    response = client.post(
        "/api/uploads", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"]


def test_uploading_an_empty_file_is_a_client_error(client):
    response = client.post(
        "/api/uploads", files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_a_sample_can_be_promoted_to_a_target(client):
    response = client.post("/api/uploads/sample", json={"name": "car.jpeg"})
    body = response.json()

    assert response.status_code == 200
    assert body["label"] == "car.jpeg"
    assert client.get(body["url"]).status_code == 200


def test_an_unknown_sample_cannot_be_promoted(client):
    assert client.post("/api/uploads/sample", json={"name": "ghost.png"}).status_code == 404


def test_url_uploads_report_a_bad_url(client):
    response = client.post("/api/uploads/url", json={"url": "http://127.0.0.1/x.png"})
    assert response.status_code == 400
    assert "private address" in response.json()["detail"]


def test_url_uploads_fetch_and_store(client, monkeypatch):
    from tests.test_imaging import FakeResponse
    from webapp import imaging

    monkeypatch.setenv("VORONOI_ALLOW_PRIVATE_URLS", "1")
    payload = png_bytes((22, 11))
    monkeypatch.setattr(
        imaging.urllib.request,
        "urlopen",
        lambda request, timeout=None: FakeResponse(payload),
    )

    response = client.post("/api/uploads/url", json={"url": "http://127.0.0.1/x.png"})

    assert response.status_code == 200
    assert (response.json()["width"], response.json()["height"]) == (22, 11)


@pytest.mark.parametrize("image_id", ["../secret", "not-hex!", "0" * 12])
def test_unknown_upload_ids_are_refused(client, image_id):
    assert client.get(f"/api/uploads/{image_id}.png").status_code == 404


def test_a_missing_url_field_fails_validation(client):
    assert client.post("/api/uploads/url", json={}).status_code == 422


# --- Jobs ------------------------------------------------------------------


def test_creating_a_job_requires_a_known_image(client):
    body = tiny_run_params().model_dump()
    body["image_id"] = "0" * 12
    assert client.post("/api/jobs", json=body).status_code == 404


def test_creating_a_job_validates_the_parameters(client):
    image_id = upload_png(client)
    body = tiny_run_params().model_dump()
    body["image_id"] = image_id
    body["generations"] = 0

    response = client.post("/api/jobs", json=body)
    assert response.status_code == 422


def test_unknown_job_ids_are_404(client):
    for path in ("", "/preview.png", "/target.png", "/result.png"):
        assert client.get(f"/api/jobs/deadbeef{path}").status_code == 404
    assert client.post("/api/jobs/deadbeef/cancel").status_code == 404
    assert client.delete("/api/jobs/deadbeef").status_code == 404


def test_a_fresh_job_has_no_artifacts_yet(client):
    """Occupy the worker, then check a queued job reports nothing to download."""
    image_id = upload_png(client)
    body = tiny_run_params(generations=4000).model_dump()
    body["image_id"] = image_id
    blocker = client.post("/api/jobs", json=body).json()

    body = tiny_run_params().model_dump()
    body["image_id"] = image_id
    queued = client.post("/api/jobs", json=body).json()

    try:
        assert queued["status"] == "queued"
        assert client.get(f"/api/jobs/{queued['id']}/result.png").status_code == 404
        assert client.get(f"/api/jobs/{queued['id']}/preview.png").status_code == 404
        assert client.get(f"/api/jobs/{queued['id']}/target.png").status_code == 404
    finally:
        client.post(f"/api/jobs/{blocker['id']}/cancel")
        client.post(f"/api/jobs/{queued['id']}/cancel")


def test_cancelling_reports_the_new_state(client):
    image_id = upload_png(client)
    body = tiny_run_params(generations=4000).model_dump()
    body["image_id"] = image_id
    job = client.post("/api/jobs", json=body).json()

    response = client.post(f"/api/jobs/{job['id']}/cancel")

    assert response.status_code == 200
    assert response.json()["cancelled"] is True
    assert response.json()["job"]["id"] == job["id"]


@pytest.mark.slow
def test_a_run_completes_and_exposes_its_images(client):
    image_id = upload_png(client, (90, 70))
    snapshot = run_to_completion(client, image_id, generations=3)

    assert snapshot["status"] == "done", snapshot["error"]
    assert snapshot["progress"] == 1.0
    assert snapshot["has_result"] is True

    job_id = snapshot["id"]
    for path in ("preview.png", "target.png", "result.png"):
        response = client.get(f"/api/jobs/{job_id}/{path}")
        assert response.status_code == 200, path
        assert response.headers["content-type"] == "image/png"
        Image.open(io.BytesIO(response.content)).verify()


@pytest.mark.slow
def test_a_finished_run_can_be_downloaded_and_deleted(client):
    image_id = upload_png(client)
    snapshot = run_to_completion(client, image_id, generations=2)
    job_id = snapshot["id"]

    download = client.get(f"/api/jobs/{job_id}/result.png?download=true")
    assert download.status_code == 200
    assert f'filename="voronoi-{job_id}.png"' in download.headers["content-disposition"]

    assert client.get("/api/jobs").json()["jobs"]
    assert client.delete(f"/api/jobs/{job_id}").json() == {"deleted": True}
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


@pytest.mark.slow
def test_the_event_stream_ends_on_a_terminal_state(client):
    image_id = upload_png(client)
    snapshot = run_to_completion(client, image_id, generations=2)

    with client.stream("GET", f"/api/jobs/{snapshot['id']}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert frames
    assert frames[-1]["status"] in TERMINAL


@pytest.mark.slow
def test_a_tiled_run_completes_through_the_api(client):
    image_id = upload_png(client, (90, 70))
    snapshot = run_to_completion(
        client, image_id, mode="tiled", tile_rows=2, tile_cols=2, generations=2
    )

    assert snapshot["status"] == "done", snapshot["error"]
    assert snapshot["tile_count"] == 4
    assert snapshot["mode"] == "tiled"

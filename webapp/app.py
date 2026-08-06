"""FastAPI application exposing the genetic art algorithm to a browser.

The surface is small on purpose:

  * ``/api/uploads*`` accept a target image (file, URL, or bundled sample) and
    hand back an id,
  * ``/api/jobs`` starts a run against one of those ids with a set of
    hyperparameters, and
  * ``/api/jobs/{id}/events`` streams progress until the run finishes.

Run it with ``python run_web.py`` (or ``uvicorn webapp.app:app``).
"""

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from .imaging import ImageLoadError, decode_image, load_image_from_url
from .params import RunParams
from .runner import JobManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLES_DIR = PROJECT_ROOT / "img"

# Everything the app writes lives under one directory so it is trivial to clear
# or to point at a volume when deployed.
DATA_DIR = Path(os.environ.get("VORONOI_WEB_DATA", PROJECT_ROOT / "runs"))
UPLOADS_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"

# How often the SSE stream pushes a progress frame.
EVENT_INTERVAL_SECONDS = 0.5

app = FastAPI(title="Voronoi Genetic Art", version="1.0.0")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
jobs = JobManager(JOBS_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class UrlUploadRequest(BaseModel):
    url: str = Field(min_length=1)


class SampleUploadRequest(BaseModel):
    name: str = Field(min_length=1)


class JobRequest(RunParams):
    """A run request: the hyperparameters plus the target image to run them on."""

    image_id: str = Field(min_length=1)


def _store_upload(image: Image.Image, label: str) -> dict:
    """Persist a decoded target image and return its descriptor.

    :param image: The decoded RGBA image.
    :param label: A display name for the source (filename, URL, or sample name).
    """
    image_id = uuid.uuid4().hex[:12]
    path = UPLOADS_DIR / f"{image_id}.png"
    image.save(path, "PNG")
    return {
        "image_id": image_id,
        "label": label,
        "width": image.width,
        "height": image.height,
        "url": f"/api/uploads/{image_id}.png",
    }


def _upload_path(image_id: str) -> Path:
    """Resolve an upload id to a path, rejecting anything that escapes the directory."""
    # ``image_id`` comes straight off the wire, so it is validated as a plain
    # hex token rather than trusted as a path component.
    if not image_id.isalnum():
        raise HTTPException(status_code=404, detail="Unknown image id.")
    path = UPLOADS_DIR / f"{image_id}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown image id.")
    return path


def _require_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the single-page interface."""
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Browsers ask for this path regardless of the declared SVG icon."""
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/schema")
def schema() -> dict:
    """Field metadata (bounds, defaults, help text) for the hyperparameter form."""
    fields = {}
    for name, field in RunParams.model_fields.items():
        metadata = {"default": RunParams().model_dump()[name], "description": field.description}
        for constraint in field.metadata:
            for attribute in ("ge", "le"):
                if hasattr(constraint, attribute):
                    metadata[attribute] = getattr(constraint, attribute)
        fields[name] = metadata
    return {"fields": fields}


@app.get("/api/samples")
def samples() -> dict:
    """List the target images bundled with the repository."""
    if not SAMPLES_DIR.is_dir():
        return {"samples": []}
    names = sorted(
        path.name
        for path in SAMPLES_DIR.iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    )
    return {"samples": [{"name": name} for name in names]}


@app.get("/api/samples/{name}")
def sample_image(name: str) -> FileResponse:
    """Serve a bundled sample image for the picker thumbnails."""
    path = (SAMPLES_DIR / name).resolve()
    if SAMPLES_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown sample.")
    return FileResponse(path)


@app.post("/api/uploads")
async def upload_file(file: UploadFile = File(...), label: str = Form(default=None)) -> dict:
    """Accept a target image uploaded from the user's machine."""
    data = await file.read()
    try:
        image = decode_image(data)
    except ImageLoadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _store_upload(image, label or file.filename or "uploaded image")


@app.post("/api/uploads/url")
def upload_url(request: UrlUploadRequest) -> dict:
    """Fetch a target image from a URL the user supplied."""
    try:
        image = load_image_from_url(request.url)
    except ImageLoadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _store_upload(image, request.url)


@app.post("/api/uploads/sample")
def upload_sample(request: SampleUploadRequest) -> dict:
    """Use one of the bundled sample images as the target."""
    path = (SAMPLES_DIR / request.name).resolve()
    if SAMPLES_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown sample.")
    try:
        image = decode_image(path.read_bytes())
    except ImageLoadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _store_upload(image, request.name)


@app.get("/api/uploads/{image_id}.png")
def upload_image(image_id: str) -> FileResponse:
    """Serve a stored target image."""
    return FileResponse(_upload_path(image_id), media_type="image/png")


@app.post("/api/jobs")
def create_job(request: JobRequest) -> dict:
    """Queue an evolution run."""
    source = _upload_path(request.image_id)
    params = RunParams(**request.model_dump(exclude={"image_id"}))
    job = jobs.submit(params, source)
    return jobs.snapshot(job)


@app.get("/api/jobs")
def list_jobs() -> dict:
    """List recent runs, newest first."""
    return {"jobs": jobs.list_jobs()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Current state of one run."""
    return jobs.snapshot(_require_job(job_id))


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Ask a queued or running job to stop."""
    job = _require_job(job_id)
    cancelled = jobs.cancel(job_id)
    return {"cancelled": cancelled, "job": jobs.snapshot(job)}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Stream progress snapshots as server-sent events until the run ends."""
    job = _require_job(job_id)

    async def stream():
        terminal_sent = False
        while not terminal_sent:
            snapshot = jobs.snapshot(job)
            terminal_sent = snapshot["status"] in ("done", "error", "cancelled")
            yield f"data: {json.dumps(snapshot)}\n\n"
            if terminal_sent:
                break
            await asyncio.sleep(EVENT_INTERVAL_SECONDS)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}/preview.png")
def job_preview(job_id: str) -> Response:
    """Latest live preview frame for a run."""
    job = _require_job(job_id)
    if job.preview_png is None:
        raise HTTPException(status_code=404, detail="No preview yet.")
    return Response(
        content=job.preview_png,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/jobs/{job_id}/target.png")
def job_target(job_id: str) -> FileResponse:
    """The scaled target image the run is evolving towards."""
    job = _require_job(job_id)
    if job.target_path is None or not job.target_path.is_file():
        raise HTTPException(status_code=404, detail="Target not ready yet.")
    return FileResponse(job.target_path, media_type="image/png")


@app.get("/api/jobs/{job_id}/result.png")
def job_result(job_id: str, download: bool = False) -> FileResponse:
    """The final rendered image for a finished run."""
    job = _require_job(job_id)
    if job.result_path is None or not job.result_path.is_file():
        raise HTTPException(status_code=404, detail="This run has no result yet.")
    disposition = f'attachment; filename="voronoi-{job_id}.png"' if download else None
    headers = {"Content-Disposition": disposition} if disposition else None
    return FileResponse(job.result_path, media_type="image/png", headers=headers)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> JSONResponse:
    """Cancel a run and delete everything it wrote to disk."""
    job = _require_job(job_id)
    jobs.remove(job_id)
    shutil.rmtree(job.run_dir, ignore_errors=True)
    return JSONResponse({"deleted": True})

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import FastAPI, HTTPException, Path as P, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import AfterValidator, BaseModel

from ckviz import datasets as ds
from ckviz.pipeline import check_stages, iter_run, parse
from ckviz.stages import PLUGINS, STAGES

app = FastAPI(title="ckviz")
# DNS-rebinding guard; add hostnames via CKVIZ_ALLOWED_HOSTS when serving under a real name.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=os.environ.get(
    "CKVIZ_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(","))


@app.middleware("http")
async def csrf_header(request: Request, call_next):
    """Mutating API calls must carry X-CKVIZ: 1. Browsers can't add it cross-site without a CORS preflight,
    which this app never grants, so forms/fetches from other origins are refused."""
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path.startswith("/api/") \
            and request.headers.get("x-ckviz") != "1":
        return JSONResponse({"detail": "missing X-CKVIZ header"}, status_code=403)
    return await call_next(request)
PIPELINES = Path(os.environ.get("CKVIZ_PIPELINES", ds.CONFIG.parent / "pipelines"))  # user-saved
EXAMPLES = Path(__file__).parents[2] / "examples"  # read-only reproductions of ConvoKit notebooks
Name = P(pattern=r"^[\w-]+$")


class RunRequest(BaseModel):
    dataset: str
    stages: Annotated[list[dict], AfterValidator(check_stages)]


@app.get("/api/stages")
def stages():
    return {name: {**s.config.model_json_schema(), "x-module": s.fn.__module__,
                   "x-description": " ".join((s.fn.__doc__ or "").split())} for name, s in STAGES.items()}


@app.get("/api/plugins")
def plugins():
    return PLUGINS


@app.get("/api/datasets")
def datasets():
    kind = lambda d: next((k for k in ("dsn", "zip_url", "zip", "convokit") if k in d), "dir")
    return {name: {"kind": kind(d), "snapshots": [p.name for p in ds.snapshots(name)]}
            for name, d in ds.datasets().items()}


@app.post("/api/datasets/{name}/snapshot")
def snapshot(name: str = Name):
    if not {"dsn", "zip_url"} & set(ds.datasets().get(name, {})):
        raise HTTPException(404, f"no refreshable (dsn / zip_url) dataset {name!r}")
    try:
        return {"snapshot": ds.snapshot(name).name}
    except ValueError as e:  # bad download, wrong hash, unsafe zip
        raise HTTPException(400, str(e))


def fields(frames) -> dict:
    meta_keys = lambda df: [c.removeprefix("meta.") for c in df.columns if c.startswith("meta.")]
    return {"utterance": list(frames.utts.columns), "speaker": meta_keys(frames.speakers),
            "conversation": meta_keys(frames.convos)}


# ponytail: one worker thread, jobs kept in memory; move to a real queue when runs must survive restarts.
JOBS: dict[str, dict] = {}
WORKER = ThreadPoolExecutor(max_workers=1)


def work(job: dict, req: RunRequest) -> None:
    try:
        job["status"] = "loading"
        key, frames = ds.load(req.dataset)
        job.update(status="running", fields=fields(frames))
        for step in iter_run(req.stages, frames, key):
            job["steps"].append(step)
        job["status"] = "done"
    except Exception as e:
        job.update(status="error", error=f"{type(e).__name__}: {e}")


@app.post("/api/runs")
def start_run(req: RunRequest):
    config = ds.datasets().get(req.dataset)
    if config is None:
        raise HTTPException(404, f"no dataset {req.dataset!r}")
    try:
        ds.check_paths(req.dataset, config)
    except ValueError as e:
        raise HTTPException(400, str(e))
    run_id = uuid.uuid4().hex[:12]
    JOBS[run_id] = job = {"id": run_id, "status": "queued", "total": len(req.stages), "fields": None, "steps": []}
    WORKER.submit(work, job, req)
    return {"id": run_id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in JOBS:
        raise HTTPException(404, run_id)
    return JOBS[run_id]


@app.get("/api/pipelines")
def pipelines():
    return sorted({p.stem for d in (PIPELINES, EXAMPLES) for p in d.glob("*.yaml")})


@app.get("/api/pipelines/{name}")
def get_pipeline(name: str = Name):
    for path in (PIPELINES / f"{name}.yaml", EXAMPLES / f"{name}.yaml"):
        if path.exists():
            return parse(path.read_text())
    raise HTTPException(404, name)


@app.put("/api/pipelines/{name}")
def put_pipeline(body: dict, name: str = Name):
    try:
        parse(yaml.safe_dump(body))
    except ValueError as e:
        raise HTTPException(400, str(e))
    PIPELINES.mkdir(parents=True, exist_ok=True)
    (PIPELINES / f"{name}.yaml").write_text(yaml.safe_dump(body, sort_keys=False))
    return {"saved": name}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")

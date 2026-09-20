from __future__ import annotations

import asyncio
import hmac
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import SUPPORTED_LANGUAGES, settings
from app.jobs import jobs
from app.runtime import runtime


class GenerateRequest(BaseModel):
    text: str = Field(min_length=1)
    language: Literal["hi", "mr", "gu", "bn", "arb"]
    mode: Literal["clone", "auto", "design"] = "clone"
    ref_audio_b64: str | None = None
    ref_text: str | None = None
    instruct: str | None = None
    num_step: int = Field(default=32, ge=8, le=64)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


async def require_token(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="missing bearer token")
    supplied = authorization[len(prefix) :]
    if not hmac.compare_digest(supplied, settings.api_token):
        raise HTTPException(status_code=403, detail="invalid bearer token")


async def _initialize_runtime() -> None:
    try:
        await asyncio.to_thread(runtime.initialize)
    except Exception:
        return


@asynccontextmanager
async def lifespan(_: FastAPI):
    await jobs.start()
    load_task = asyncio.create_task(_initialize_runtime(), name="omnivoice-model-loader")
    yield
    if not load_task.done():
        load_task.cancel()


app = FastAPI(
    title="OmniVoice GPU Runtime",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    return {
        "service": "voice-studio-omnivoice-runtime",
        "version": "0.1.0",
        "languages": SUPPORTED_LANGUAGES,
    }


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "alive",
        "ready": runtime.ready,
        "loading": runtime.loading,
        "uptime_seconds": time.time() - runtime.started_at,
    }


@app.get("/readyz")
def readyz() -> Response:
    payload = runtime.diagnostics()
    if runtime.ready:
        return JSONResponse(status_code=200, content=payload)
    return JSONResponse(status_code=503, content=payload)


@app.get("/runtimez")
def runtimez(_: None = Depends(require_token)) -> dict:
    return runtime.diagnostics()


@app.get("/smokez")
def smokez() -> Response:
    if runtime.last_smoke and runtime.last_smoke.get("status") == "passed":
        return JSONResponse(status_code=200, content=runtime.last_smoke)
    return JSONResponse(
        status_code=503,
        content={"status": "not-passed", "ready": runtime.ready, "error": runtime.error},
    )


@app.post("/smokez")
async def rerun_smoke(_: None = Depends(require_token)) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.error or "runtime not ready")
    return await asyncio.to_thread(runtime.run_smoke)


@app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: GenerateRequest, _: None = Depends(require_token)
) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.error or "runtime not ready")
    try:
        job = await jobs.submit(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"job_id": job.id, "state": job.state}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, _: None = Depends(require_token)) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.public()


@app.get("/v1/jobs/{job_id}/audio")
def get_job_audio(job_id: str, _: None = Depends(require_token)) -> FileResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state != "completed" or not job.output_path:
        raise HTTPException(status_code=409, detail=f"job state is {job.state}")
    return FileResponse(job.output_path, media_type="audio/wav", filename=f"{job_id}.wav")

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.audio_validation import validate_reference_wav
from app.config import SUPPORTED_LANGUAGES, settings
from app.jobs import jobs
from app.runtime import runtime


class GenerateRequest(BaseModel):
    text: str = Field(min_length=1)
    language: Literal["hi", "mr", "gu", "bn", "arb"]
    mode: Literal["clone", "auto", "design"] = "clone"
    voice_id: str | None = None
    ref_audio_b64: str | None = None
    ref_text: str | None = None
    instruct: str | None = None
    num_step: int = Field(default=32, ge=8, le=64)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class VoiceProfileCreateRequest(BaseModel):
    voice_id: str = Field(min_length=1, max_length=128)
    ref_audio_b64: str = Field(min_length=1)
    ref_text: str = Field(min_length=1)
    replace: bool = False


class VoiceProfileImportRequest(BaseModel):
    voice_id: str = Field(min_length=1, max_length=128)
    profile_b64: str = Field(min_length=1)
    replace: bool = False


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


@app.get("/v1/voices")
def list_voices(_: None = Depends(require_token)) -> dict:
    return {"voices": runtime.list_voice_profiles()}


@app.post("/v1/voices")
async def create_voice(
    request: VoiceProfileCreateRequest, _: None = Depends(require_token)
) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.error or "runtime not ready")

    try:
        raw = base64.b64decode(request.ref_audio_b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid base64 reference audio: {exc}") from exc
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="reference WAV exceeds 20 MiB")

    temp_path = settings.voices_dir / f".incoming-{uuid.uuid4().hex}.wav"
    temp_path.write_bytes(raw)
    try:
        validation = validate_reference_wav(temp_path)
        profile = await asyncio.to_thread(
            runtime.create_voice_profile,
            voice_id=request.voice_id,
            ref_audio=temp_path,
            ref_text=request.ref_text,
            replace=request.replace,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return {"status": "created", "profile": profile, "reference_validation": validation}


@app.post("/v1/voices/import")
async def import_voice(
    request: VoiceProfileImportRequest, _: None = Depends(require_token)
) -> dict:
    try:
        raw = base64.b64decode(request.profile_b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid base64 voice profile: {exc}") from exc
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="voice profile exceeds 20 MiB")

    temp_path = settings.voices_dir / f".incoming-{uuid.uuid4().hex}.pt"
    temp_path.write_bytes(raw)
    try:
        profile = await asyncio.to_thread(
            runtime.import_voice_profile,
            voice_id=request.voice_id,
            source_path=temp_path,
            replace=request.replace,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid voice profile: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return {"status": "imported", "profile": profile}


@app.get("/v1/voices/{voice_id}/export")
def export_voice(voice_id: str, _: None = Depends(require_token)) -> dict:
    try:
        raw = runtime.export_voice_profile(voice_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "voice_id": voice_id,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "profile_b64": base64.b64encode(raw).decode("ascii"),
    }


@app.delete("/v1/voices/{voice_id}")
def delete_voice(voice_id: str, _: None = Depends(require_token)) -> dict:
    try:
        deleted = runtime.delete_voice_profile(voice_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"voice profile not found: {voice_id}")
    return {"status": "deleted", "voice_id": voice_id}


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

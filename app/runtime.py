from __future__ import annotations

import importlib.metadata
import platform
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.audio_validation import validate_wav
from app.config import SUPPORTED_LANGUAGES, settings
from app.model_manifest import verify_model_manifest
from app.text_splitter import split_text


class RuntimeNotReady(RuntimeError):
    pass


class OmniRuntime:
    def __init__(self) -> None:
        self.model: Any = None
        self.torch: Any = None
        self.ready = False
        self.loading = False
        self.error: str | None = None
        self.started_at = time.time()
        self.loaded_at: float | None = None
        self.load_seconds: float | None = None
        self.manifest_report: dict | None = None
        self.last_smoke: dict | None = None
        self._inference_lock = threading.Lock()
        self._state_lock = threading.Lock()

    def initialize(self) -> None:
        with self._state_lock:
            if self.loading or self.ready:
                return
            self.loading = True
            self.error = None
        started = time.perf_counter()
        try:
            self.manifest_report = verify_model_manifest(
                settings.model_dir, settings.manifest_path
            )

            import torch
            import torchaudio
            from omnivoice import OmniVoice

            self.torch = torch
            package_version = importlib.metadata.version("omnivoice")
            torchaudio_version = importlib.metadata.version("torchaudio")
            transformers_version = importlib.metadata.version("transformers")
            if package_version != settings.expected_package_version:
                raise RuntimeError(
                    f"omnivoice version mismatch: {package_version} != "
                    f"{settings.expected_package_version}"
                )
            if torchaudio_version != settings.expected_torchaudio_version:
                raise RuntimeError(
                    f"torchaudio version mismatch: {torchaudio_version} != "
                    f"{settings.expected_torchaudio_version}"
                )
            if transformers_version != settings.expected_transformers_version:
                raise RuntimeError(
                    f"transformers version mismatch: {transformers_version} != "
                    f"{settings.expected_transformers_version}"
                )
            if not torch.__version__.startswith(settings.expected_torch_prefix):
                raise RuntimeError(
                    f"torch version mismatch: {torch.__version__} does not start with "
                    f"{settings.expected_torch_prefix}"
                )
            if not str(torch.version.cuda or "").startswith(settings.expected_cuda_prefix):
                raise RuntimeError(
                    f"torch CUDA mismatch: {torch.version.cuda} does not start with "
                    f"{settings.expected_cuda_prefix}"
                )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")

            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            if total_gb < settings.min_vram_gb:
                raise RuntimeError(
                    f"insufficient VRAM: {total_gb:.2f} GiB < {settings.min_vram_gb:.2f} GiB"
                )

            self.model = OmniVoice.from_pretrained(
                str(settings.model_dir),
                device_map="cuda:0",
                dtype=torch.float16,
                load_asr=False,
            )
            expected_sr = int(self.manifest_report["expected_sampling_rate"])
            actual_sr = int(self.model.sampling_rate)
            if actual_sr != expected_sr:
                raise RuntimeError(
                    f"model sampling rate mismatch: {actual_sr} != {expected_sr}"
                )

            self.loaded_at = time.time()
            self.load_seconds = time.perf_counter() - started

            if settings.startup_smoke:
                smoke_path = settings.jobs_dir / "_startup_smoke.wav"
                self.last_smoke = self.generate(
                    text="This is the OmniVoice runtime startup smoke test.",
                    output_path=smoke_path,
                    mode="auto",
                    language=None,
                    num_step=16,
                    speed=1.0,
                )
                self.last_smoke["kind"] = "startup-real-audio"

            with self._state_lock:
                self.ready = True
                self.loading = False
        except Exception as exc:
            with self._state_lock:
                self.ready = False
                self.loading = False
                self.error = f"{type(exc).__name__}: {exc}"
            raise

    def _require_ready_model(self) -> None:
        if self.model is None or self.torch is None:
            raise RuntimeNotReady(self.error or "model has not loaded")

    def diagnostics(self) -> dict:
        base = {
            "ready": self.ready,
            "loading": self.loading,
            "error": self.error,
            "process_uptime_seconds": time.time() - self.started_at,
            "loaded_at": self.loaded_at,
            "load_seconds": self.load_seconds,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "model_manifest": self.manifest_report,
            "last_smoke": self.last_smoke,
        }
        if self.torch is None:
            return base
        torch = self.torch
        gpu: dict[str, Any] = {
            "cuda_available": torch.cuda.is_available(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
        try:
            gpu["omnivoice"] = importlib.metadata.version("omnivoice")
            gpu["torchaudio"] = importlib.metadata.version("torchaudio")
            gpu["transformers"] = importlib.metadata.version("transformers")
            gpu["fastapi"] = importlib.metadata.version("fastapi")
            gpu["uvicorn"] = importlib.metadata.version("uvicorn")
        except Exception:
            pass
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            gpu.update(
                {
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_vram_gib": total_bytes / (1024**3),
                    "free_vram_gib": free_bytes / (1024**3),
                    "allocated_vram_gib": torch.cuda.memory_allocated(0) / (1024**3),
                    "reserved_vram_gib": torch.cuda.memory_reserved(0) / (1024**3),
                }
            )
        base["gpu"] = gpu
        return base

    def generate(
        self,
        *,
        text: str,
        output_path: Path,
        mode: str,
        language: str | None,
        ref_audio: Path | None = None,
        ref_text: str | None = None,
        instruct: str | None = None,
        num_step: int = 32,
        speed: float = 1.0,
    ) -> dict:
        self._require_ready_model()
        if language is not None and language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported qualification language: {language}")
        if not text.strip():
            raise ValueError("text is empty")
        if len(text) > settings.max_text_chars:
            raise ValueError(
                f"text exceeds maximum of {settings.max_text_chars} characters"
            )
        if mode not in {"clone", "auto", "design"}:
            raise ValueError(f"unsupported mode: {mode}")
        if mode == "clone" and (ref_audio is None or not ref_text):
            raise ValueError("clone mode requires ref_audio and ref_text")
        if mode == "design" and not instruct:
            raise ValueError("design mode requires instruct")

        chunks = split_text(text, max_chars=settings.chunk_chars)
        spoken_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        if not spoken_chunks:
            raise ValueError("text contains no speakable content")

        started = time.perf_counter()
        torch = self.torch
        pieces: list[np.ndarray] = []

        with self._inference_lock:
            clone_prompt = None
            if mode == "clone":
                clone_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=str(ref_audio), ref_text=ref_text
                )

            for chunk in spoken_chunks:
                kwargs: dict[str, Any] = {
                    "text": chunk,
                    "language": language,
                    "num_step": int(num_step),
                    "speed": float(speed),
                }
                if clone_prompt is not None:
                    kwargs["voice_clone_prompt"] = clone_prompt
                elif mode == "design":
                    kwargs["instruct"] = instruct

                audio = self.model.generate(**kwargs)
                if not audio:
                    raise RuntimeError("model returned no audio")
                array = np.asarray(audio[0], dtype=np.float32).reshape(-1)
                if array.size == 0:
                    raise RuntimeError("model returned an empty audio array")
                pieces.append(array)

            if len(pieces) > 1:
                gap = np.zeros(int(self.model.sampling_rate * 0.08), dtype=np.float32)
                merged: list[np.ndarray] = []
                for index, piece in enumerate(pieces):
                    if index:
                        merged.append(gap)
                    merged.append(piece)
                final_audio = np.concatenate(merged)
            else:
                final_audio = pieces[0]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, final_audio, int(self.model.sampling_rate), subtype="PCM_16")
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        synthesis_seconds = time.perf_counter() - started
        metrics = validate_wav(output_path, expected_sample_rate=int(self.model.sampling_rate))
        rtf = synthesis_seconds / metrics.duration_seconds
        return {
            "status": "passed",
            "mode": mode,
            "language": language,
            "input_characters": len(text),
            "split_chunks": len(chunks),
            "spoken_chunks": len(spoken_chunks),
            "text_coverage_exact": "".join(chunks) == text,
            "num_step": int(num_step),
            "speed": float(speed),
            "synthesis_seconds": synthesis_seconds,
            "rtf": rtf,
            "audio": metrics.to_dict(),
        }

    def run_smoke(self) -> dict:
        if self.model is None:
            raise RuntimeNotReady(self.error or "model has not loaded")
        path = settings.jobs_dir / f"_smoke_{int(time.time())}.wav"
        result = self.generate(
            text="This is a real audio smoke test for OmniVoice.",
            output_path=path,
            mode="auto",
            language=None,
            num_step=16,
            speed=1.0,
        )
        result["kind"] = "manual-real-audio"
        self.last_smoke = result
        return result


runtime = OmniRuntime()

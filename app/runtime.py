from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import threading
import time
import uuid
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
        self.instance_id = uuid.uuid4().hex
        self.loaded_at: float | None = None
        self.load_seconds: float | None = None
        self.manifest_report: dict | None = None
        self.last_smoke: dict | None = None
        self._inference_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._voice_cache: dict[str, Any] = {}
        self._design_previews: dict[str, dict[str, Any]] = {}

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
            if not torchaudio_version.startswith(settings.expected_torchaudio_version):
                raise RuntimeError(
                    f"torchaudio version mismatch: {torchaudio_version} does not start with "
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
            "instance_id": self.instance_id,
            "voice_profiles": len(list(settings.voices_dir.glob("*.pt"))),
            "design_previews": len(list(settings.design_previews_dir.glob("*.json"))),
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

    @staticmethod
    def _validate_voice_id(voice_id: str) -> str:
        normalized = str(voice_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", normalized):
            raise ValueError(
                "voice_id must be 1-128 characters using letters, numbers, _ or -"
            )
        return normalized

    def _voice_path(self, voice_id: str) -> Path:
        normalized = self._validate_voice_id(voice_id)
        return settings.voices_dir / f"{normalized}.pt"

    def _voice_meta_path(self, voice_id: str) -> Path:
        normalized = self._validate_voice_id(voice_id)
        return settings.voices_dir / f"{normalized}.json"

    @staticmethod
    def _validate_preview_id(preview_id: str) -> str:
        normalized = str(preview_id or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{32}", normalized):
            raise ValueError("preview_id must be a 32-character hexadecimal id")
        return normalized

    def _design_preview_meta_path(self, preview_id: str) -> Path:
        normalized = self._validate_preview_id(preview_id)
        return settings.design_previews_dir / f"{normalized}.json"

    def _design_preview_audio_path(self, preview_id: str) -> Path:
        normalized = self._validate_preview_id(preview_id)
        return settings.design_previews_dir / f"{normalized}.wav"

    def _persist_design_preview(self, preview: dict[str, Any]) -> None:
        preview_id = self._validate_preview_id(str(preview["preview_id"]))
        path = self._design_preview_meta_path(preview_id)
        tmp_path = path.with_suffix(".json.tmp")
        public = {k: v for k, v in preview.items() if k != "audio_path"}
        tmp_path.write_text(
            json.dumps(public, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _write_voice_metadata(self, voice_id: str, metadata: dict[str, Any]) -> None:
        path = self._voice_meta_path(voice_id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def voice_profile_info(self, voice_id: str) -> dict:
        normalized = self._validate_voice_id(voice_id)
        profile_path = self._voice_path(normalized)
        if not profile_path.is_file():
            raise FileNotFoundError(f"voice profile not found: {normalized}")
        raw = profile_path.read_bytes()
        info = {
            "voice_id": normalized,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "cached": normalized in self._voice_cache,
        }
        meta_path = self._voice_meta_path(normalized)
        if meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(metadata, dict):
                    info.update(metadata)
            except Exception:
                info["metadata_error"] = "invalid metadata sidecar"
        return info

    def voice_profile_exists(self, voice_id: str) -> bool:
        return self._voice_path(voice_id).is_file()

    def list_voice_profiles(self) -> list[dict]:
        profiles: list[dict] = []
        for path in sorted(settings.voices_dir.glob("*.pt")):
            profiles.append(self.voice_profile_info(path.stem))
        return profiles

    def load_voice_profile(self, voice_id: str) -> Any:
        normalized = self._validate_voice_id(voice_id)
        with self._state_lock:
            cached = self._voice_cache.get(normalized)
        if cached is not None:
            return cached

        path = self._voice_path(normalized)
        if not path.is_file():
            raise FileNotFoundError(f"voice profile not found: {normalized}")

        from omnivoice import VoiceClonePrompt

        prompt = VoiceClonePrompt.load(str(path), map_location="cpu")
        with self._state_lock:
            self._voice_cache[normalized] = prompt
        return prompt

    def create_voice_profile(
        self,
        *,
        voice_id: str,
        ref_audio: Path,
        ref_text: str,
        replace: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        self._require_ready_model()
        normalized = self._validate_voice_id(voice_id)
        transcript = str(ref_text or "").strip()
        if not transcript:
            raise ValueError("ref_text is required because runtime ASR is disabled")

        path = self._voice_path(normalized)
        if path.exists() and not replace:
            raise FileExistsError(f"voice profile already exists: {normalized}")
        tmp_path = path.with_suffix(".pt.tmp")
        with self._inference_lock:
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(ref_audio), ref_text=transcript
            )
            prompt.save(str(tmp_path))
            tmp_path.replace(path)

        with self._state_lock:
            self._voice_cache[normalized] = prompt

        now = time.time()
        extra = dict(metadata or {})
        for key in (
            "voice_id",
            "bytes",
            "sha256",
            "cached",
            "profile_type",
            "ref_text",
            "ref_text_characters",
            "created_at",
            "updated_at",
        ):
            extra.pop(key, None)
        profile_metadata = {
            **extra,
            "profile_type": "cloned",
            "ref_text": transcript,
            "ref_text_characters": len(transcript),
            "created_at": now,
            "updated_at": now,
        }
        self._write_voice_metadata(normalized, profile_metadata)
        return self.voice_profile_info(normalized)

    def create_design_preview(
        self,
        *,
        instruct: str,
        sample_text: str,
        language: str | None,
        num_step: int = 32,
        speed: float = 1.0,
    ) -> dict:
        self._require_ready_model()
        design_prompt = str(instruct or "").strip()
        seed_text = str(sample_text or "").strip()
        if not design_prompt:
            raise ValueError("instruct is required")
        if not seed_text:
            raise ValueError("sample_text is required")
        if language is not None and language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported qualification language: {language}")

        preview_id = uuid.uuid4().hex
        preview_path = self._design_preview_audio_path(preview_id)
        started = time.perf_counter()

        with self._inference_lock:
            audio = self.model.generate(
                text=seed_text,
                language=language,
                instruct=design_prompt,
                num_step=int(num_step),
                speed=float(speed),
            )
            if not audio:
                raise RuntimeError("model returned no designed preview audio")
            array = np.asarray(audio[0], dtype=np.float32).reshape(-1)
            if array.size == 0:
                raise RuntimeError("model returned an empty designed preview audio array")
            sf.write(
                preview_path,
                array,
                int(self.model.sampling_rate),
                subtype="PCM_16",
            )
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()

        synthesis_seconds = time.perf_counter() - started
        duration_seconds = array.size / float(self.model.sampling_rate)
        preview = {
            "preview_id": preview_id,
            "instruct": design_prompt,
            "sample_text": seed_text,
            "language": language,
            "num_step": int(num_step),
            "speed": float(speed),
            "duration_seconds": duration_seconds,
            "synthesis_seconds": synthesis_seconds,
            "rtf": synthesis_seconds / duration_seconds if duration_seconds > 0 else None,
            "created_at": time.time(),
            "audio_path": str(preview_path),
        }
        self._persist_design_preview(preview)
        with self._state_lock:
            self._design_previews[preview_id] = preview
        return {k: v for k, v in preview.items() if k != "audio_path"}

    def get_design_preview(self, preview_id: str) -> dict:
        normalized = self._validate_preview_id(preview_id)
        with self._state_lock:
            preview = self._design_previews.get(normalized)
        if preview is None:
            meta_path = self._design_preview_meta_path(normalized)
            if not meta_path.is_file():
                raise FileNotFoundError(f"design preview not found: {normalized}")
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"invalid design preview metadata: {normalized}") from exc
            if not isinstance(loaded, dict):
                raise RuntimeError(f"invalid design preview metadata: {normalized}")
            preview = dict(loaded)
            preview["preview_id"] = normalized
            preview["audio_path"] = str(self._design_preview_audio_path(normalized))
            with self._state_lock:
                self._design_previews[normalized] = preview
        path = Path(preview["audio_path"])
        if not path.is_file():
            raise FileNotFoundError(f"design preview audio not found: {normalized}")
        return dict(preview)

    def approve_design_preview(
        self,
        *,
        preview_id: str,
        voice_id: str,
        replace: bool = False,
    ) -> dict:
        preview = self.get_design_preview(preview_id)
        profile = self.create_voice_profile(
            voice_id=voice_id,
            ref_audio=Path(preview["audio_path"]),
            ref_text=preview["sample_text"],
            replace=replace,
        )
        current = self.voice_profile_info(voice_id)
        created_at = current.get("created_at", time.time())
        self._write_voice_metadata(
            voice_id,
            {
                "profile_type": "designed",
                "design_prompt": preview["instruct"],
                "sample_text": preview["sample_text"],
                "language": preview["language"],
                "design_preview_id": preview_id,
                "created_at": created_at,
                "updated_at": time.time(),
            },
        )
        return self.voice_profile_info(voice_id)

    def delete_design_preview(self, preview_id: str) -> bool:
        normalized = self._validate_preview_id(preview_id)
        with self._state_lock:
            preview = self._design_previews.pop(normalized, None)
        audio_path = self._design_preview_audio_path(normalized)
        meta_path = self._design_preview_meta_path(normalized)
        existed = bool(preview is not None or audio_path.is_file() or meta_path.is_file())
        audio_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return existed

    def import_voice_profile(
        self,
        *,
        voice_id: str,
        source_path: Path,
        replace: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        normalized = self._validate_voice_id(voice_id)
        from omnivoice import VoiceClonePrompt

        prompt = VoiceClonePrompt.load(str(source_path), map_location="cpu")
        destination = self._voice_path(normalized)
        if destination.exists() and not replace:
            raise FileExistsError(f"voice profile already exists: {normalized}")
        tmp_path = destination.with_suffix(".pt.tmp")
        tmp_path.write_bytes(source_path.read_bytes())
        tmp_path.replace(destination)
        with self._state_lock:
            self._voice_cache[normalized] = prompt
        if metadata:
            safe_metadata = dict(metadata)
            safe_metadata.pop("voice_id", None)
            safe_metadata.pop("bytes", None)
            safe_metadata.pop("cached", None)
            safe_metadata["updated_at"] = time.time()
            safe_metadata.setdefault("created_at", safe_metadata["updated_at"])
            self._write_voice_metadata(normalized, safe_metadata)
        return self.voice_profile_info(normalized)

    def export_voice_profile(self, voice_id: str) -> bytes:
        path = self._voice_path(voice_id)
        if not path.is_file():
            raise FileNotFoundError(f"voice profile not found: {voice_id}")
        return path.read_bytes()

    def delete_voice_profile(self, voice_id: str) -> bool:
        normalized = self._validate_voice_id(voice_id)
        path = self._voice_path(normalized)
        existed = path.is_file()
        if existed:
            path.unlink()
        self._voice_meta_path(normalized).unlink(missing_ok=True)
        with self._state_lock:
            self._voice_cache.pop(normalized, None)
        return existed

    def generate(
        self,
        *,
        text: str,
        output_path: Path,
        mode: str,
        language: str | None,
        ref_audio: Path | None = None,
        ref_text: str | None = None,
        voice_id: str | None = None,
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
        if mode == "clone" and not voice_id and (ref_audio is None or not ref_text):
            raise ValueError("clone mode requires voice_id or ref_audio + ref_text")
        if mode == "design" and not instruct:
            raise ValueError("design mode requires instruct")

        chunks = split_text(text, max_chars=settings.chunk_chars)
        spoken_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        if not spoken_chunks:
            raise ValueError("text contains no speakable content")

        clone_prompt = None
        if mode == "clone" and voice_id:
            clone_prompt = self.load_voice_profile(voice_id)

        started = time.perf_counter()
        torch = self.torch
        pieces: list[np.ndarray] = []

        with self._inference_lock:
            if mode == "clone" and clone_prompt is None:
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
            "voice_id": voice_id if mode == "clone" else None,
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

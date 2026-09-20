from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "arb": "Arabic",
}


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(os.getenv("OMNIVOICE_MODEL_DIR", "/opt/models/omnivoice"))
    manifest_path: Path = Path(os.getenv("OMNIVOICE_MANIFEST", "/app/MODEL_MANIFEST.json"))
    jobs_dir: Path = Path(os.getenv("OMNIVOICE_JOBS_DIR", "/tmp/omnivoice/jobs"))
    voices_dir: Path = Path(os.getenv("OMNIVOICE_VOICES_DIR", "/tmp/omnivoice/voices"))
    api_token: str = os.getenv("OMNIVOICE_API_TOKEN", "")
    min_vram_gb: float = float(os.getenv("OMNIVOICE_MIN_VRAM_GB", "20"))
    expected_package_version: str = os.getenv("OMNIVOICE_EXPECTED_VERSION", "0.2.1")
    expected_torch_prefix: str = os.getenv("OMNIVOICE_EXPECTED_TORCH", "2.8.0")
    expected_torchaudio_version: str = os.getenv("OMNIVOICE_EXPECTED_TORCHAUDIO", "2.8.0")
    expected_transformers_version: str = os.getenv("OMNIVOICE_EXPECTED_TRANSFORMERS", "5.3.0")
    expected_cuda_prefix: str = os.getenv("OMNIVOICE_EXPECTED_CUDA", "12.8")
    startup_smoke: bool = os.getenv("OMNIVOICE_STARTUP_SMOKE", "1") == "1"
    max_text_chars: int = int(os.getenv("OMNIVOICE_MAX_TEXT_CHARS", "12000"))
    chunk_chars: int = int(os.getenv("OMNIVOICE_CHUNK_CHARS", "420"))


settings = Settings()
settings.jobs_dir.mkdir(parents=True, exist_ok=True)
settings.voices_dir.mkdir(parents=True, exist_ok=True)

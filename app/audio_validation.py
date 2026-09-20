from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioValidationError(RuntimeError):
    pass


@dataclass
class AudioMetrics:
    path: str
    sha256: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    peak: float
    rms: float
    clipping_fraction: float
    file_size: int

    def to_dict(self) -> dict:
        return asdict(self)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file() or path.stat().st_size < 128:
        raise AudioValidationError("audio file is missing or too small")
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioValidationError(f"soundfile could not decode audio: {exc}") from exc
    if data.shape[0] <= 0 or data.shape[1] <= 0:
        raise AudioValidationError("audio contains no frames")
    if not np.isfinite(data).all():
        raise AudioValidationError("audio contains NaN or infinite samples")
    return data, int(sample_rate)


def validate_wav(path: str | Path, expected_sample_rate: int = 24000) -> AudioMetrics:
    p = Path(path)
    data, sample_rate = _read_audio(p)
    if sample_rate != expected_sample_rate:
        raise AudioValidationError(
            f"unexpected sample rate {sample_rate}; expected {expected_sample_rate}"
        )

    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))
    duration = float(data.shape[0] / sample_rate)
    clipping_fraction = float(np.mean(np.abs(data) >= 0.999))

    if duration < 0.20:
        raise AudioValidationError(f"audio duration too short: {duration:.3f}s")
    if peak < 0.005 or rms < 0.0001:
        raise AudioValidationError(
            f"audio appears silent: peak={peak:.6f} rms={rms:.6f}"
        )
    if peak > 2.0:
        raise AudioValidationError(f"audio has pathological peak level: {peak:.4f}")
    if clipping_fraction > 0.10:
        raise AudioValidationError(
            f"audio appears heavily clipped: clipping_fraction={clipping_fraction:.4f}"
        )

    return AudioMetrics(
        path=str(p),
        sha256=file_sha256(p),
        sample_rate=sample_rate,
        channels=int(data.shape[1]),
        frames=int(data.shape[0]),
        duration_seconds=duration,
        peak=peak,
        rms=rms,
        clipping_fraction=clipping_fraction,
        file_size=p.stat().st_size,
    )


def validate_reference_wav(path: str | Path) -> dict:
    p = Path(path)
    data, sample_rate = _read_audio(p)
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))
    duration = float(data.shape[0] / sample_rate)
    if duration < 1.0 or duration > 30.0:
        raise AudioValidationError(
            f"reference audio must be 1-30 seconds; got {duration:.3f}s"
        )
    if peak < 0.005 or rms < 0.0001:
        raise AudioValidationError("reference audio appears silent")
    return {
        "sample_rate": sample_rate,
        "channels": int(data.shape[1]),
        "frames": int(data.shape[0]),
        "duration_seconds": duration,
        "peak": peak,
        "rms": rms,
        "sha256": file_sha256(p),
    }

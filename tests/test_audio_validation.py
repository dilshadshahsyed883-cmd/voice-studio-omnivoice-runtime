from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio_validation import AudioValidationError, validate_reference_wav, validate_wav


def test_valid_non_silent_wav(tmp_path: Path):
    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "valid.wav"
    sf.write(path, audio, sr, subtype="PCM_16")
    metrics = validate_wav(path)
    assert metrics.sample_rate == 24000
    assert metrics.duration_seconds >= 0.99
    assert metrics.rms > 0.01


def test_silent_wav_fails(tmp_path: Path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(24000, dtype=np.float32), 24000, subtype="PCM_16")
    with pytest.raises(AudioValidationError):
        validate_wav(path)


def test_reference_accepts_non_24k(tmp_path: Path):
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    audio = 0.1 * np.sin(2 * np.pi * 220 * t)
    path = tmp_path / "ref.wav"
    sf.write(path, audio, sr, subtype="PCM_16")
    report = validate_reference_wav(path)
    assert report["sample_rate"] == 16000


def test_wrong_output_sample_rate_fails(tmp_path: Path):
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "wrong-rate.wav"
    sf.write(path, audio, sr, subtype="PCM_16")
    with pytest.raises(AudioValidationError):
        validate_wav(path)


def test_corrupt_audio_fails(tmp_path: Path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"RIFF-not-a-real-wave-file" * 10)
    with pytest.raises(AudioValidationError):
        validate_wav(path)

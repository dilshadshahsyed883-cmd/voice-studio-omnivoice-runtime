from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import app.runtime as runtime_module
from app.runtime import OmniRuntime


@pytest.mark.parametrize(
    "voice_id",
    [
        "umar",
        "voice_001",
        "Hindi-Story-Voice",
        "A1",
    ],
)
def test_validate_voice_id_accepts_safe_ids(voice_id: str) -> None:
    assert OmniRuntime._validate_voice_id(voice_id) == voice_id


@pytest.mark.parametrize(
    "voice_id",
    [
        "",
        "has space",
        "../escape",
        "slash/name",
        "x" * 129,
    ],
)
def test_validate_voice_id_rejects_unsafe_ids(voice_id: str) -> None:
    with pytest.raises(ValueError):
        OmniRuntime._validate_voice_id(voice_id)


def test_validate_voice_id_trims_outer_whitespace() -> None:
    assert OmniRuntime._validate_voice_id("  voice-001  ") == "voice-001"


def test_voice_profile_path_and_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "settings",
        SimpleNamespace(voices_dir=tmp_path),
    )
    runtime = OmniRuntime()

    path = runtime._voice_path("voice_001")
    assert path == tmp_path / "voice_001.pt"
    assert runtime.voice_profile_exists("voice_001") is False

    path.write_bytes(b"profile")
    assert runtime.voice_profile_exists("voice_001") is True


def test_voice_profile_info_exposes_hash_and_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "settings",
        SimpleNamespace(voices_dir=tmp_path),
    )
    runtime = OmniRuntime()
    payload = b"portable-profile-bytes"
    (tmp_path / "voice_001.pt").write_bytes(payload)
    (tmp_path / "voice_001.json").write_text(
        json.dumps({"profile_type": "cloned", "source_profile_id": "vps3-1"}),
        encoding="utf-8",
    )

    info = runtime.voice_profile_info("voice_001")

    assert info["sha256"] == hashlib.sha256(payload).hexdigest()
    assert info["profile_type"] == "cloned"
    assert info["source_profile_id"] == "vps3-1"


def test_design_preview_metadata_survives_runtime_recreation(tmp_path, monkeypatch) -> None:
    previews = tmp_path / "previews"
    voices = tmp_path / "voices"
    previews.mkdir()
    voices.mkdir()
    monkeypatch.setattr(
        runtime_module,
        "settings",
        SimpleNamespace(voices_dir=voices, design_previews_dir=previews),
    )

    preview_id = "a" * 32
    audio_path = previews / f"{preview_id}.wav"
    audio_path.write_bytes(b"RIFF" + b"x" * 256)
    (previews / f"{preview_id}.json").write_text(
        json.dumps(
            {
                "preview_id": preview_id,
                "instruct": "warm documentary narrator",
                "sample_text": "Sample line",
                "language": "hi",
                "created_at": 1.0,
            }
        ),
        encoding="utf-8",
    )

    runtime = OmniRuntime()
    preview = runtime.get_design_preview(preview_id)

    assert preview["preview_id"] == preview_id
    assert preview["instruct"] == "warm documentary narrator"
    assert preview["audio_path"] == str(audio_path)
    assert runtime.delete_design_preview(preview_id) is True
    assert not audio_path.exists()
    assert not (previews / f"{preview_id}.json").exists()

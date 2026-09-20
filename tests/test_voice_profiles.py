from __future__ import annotations

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

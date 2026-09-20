import json
from pathlib import Path


def test_model_manifest_is_pinned():
    manifest = json.loads(Path("MODEL_MANIFEST.json").read_text())
    assert manifest["model_id"] == "k2-fsa/OmniVoice"
    assert manifest["revision"] == "18db15024ce4b7e15638be6ef0e283d99d282f39"
    assert manifest["expected_sampling_rate"] == 24000
    assert len(manifest["critical_files"]["model.safetensors"]["sha256"]) == 64

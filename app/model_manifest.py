from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_manifest(model_dir: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results: dict[str, dict] = {}
    for relative, spec in manifest["critical_files"].items():
        path = model_dir / relative
        if not path.is_file():
            raise RuntimeError(f"missing critical model file: {relative}")
        size = path.stat().st_size
        if "size" in spec and size != int(spec["size"]):
            raise RuntimeError(
                f"size mismatch for {relative}: {size} != {int(spec['size'])}"
            )
        actual_sha = sha256(path)
        if actual_sha != spec["sha256"]:
            raise RuntimeError(
                f"sha256 mismatch for {relative}: {actual_sha} != {spec['sha256']}"
            )
        results[relative] = {"size": size, "sha256": actual_sha}
    return {
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "expected_sampling_rate": manifest["expected_sampling_rate"],
        "verified": True,
        "files": results,
    }

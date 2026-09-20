#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=manifest["model_id"],
        revision=manifest["revision"],
        local_dir=str(output),
    )
    print(f"downloaded {manifest['model_id']}@{manifest['revision']} to {output}")


if __name__ == "__main__":
    main()

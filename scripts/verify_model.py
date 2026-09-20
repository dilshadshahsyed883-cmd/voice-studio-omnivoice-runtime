#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_manifest import verify_model_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    report = verify_model_manifest(Path(args.model_dir), Path(args.manifest))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

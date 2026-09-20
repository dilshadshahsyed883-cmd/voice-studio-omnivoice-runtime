#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SAMPLES = {
    "hi": "यह आवाज़ निर्माण प्रणाली का परीक्षण है।",
    "mr": "ही आवाज निर्मिती प्रणालीची चाचणी आहे.",
    "gu": "આ અવાજ બનાવવાની સિસ્ટમની કસોટી છે.",
    "bn": "এটি কণ্ঠস্বর তৈরির ব্যবস্থার একটি পরীক্ষা।",
    "arb": "هذا اختبار لنظام توليد الصوت.",
}


def request_json(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_job(base_url: str, token: str, job_id: str, timeout: int = 600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = request_json("GET", f"{base_url}/v1/jobs/{job_id}", token)
        if record["state"] in {"completed", "failed"}:
            return record
        time.sleep(2)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--mode", choices=["auto", "clone"], default="auto")
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--stress-runs", type=int, default=10)
    parser.add_argument("--report", default="qualification-report.json")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    report: dict = {
        "started_at": time.time(),
        "base_url": base,
        "mode": args.mode,
        "checks": {},
        "languages": {},
        "stress": [],
    }

    report["checks"]["healthz"] = request_json("GET", f"{base}/healthz", args.token)
    report["checks"]["readyz"] = request_json("GET", f"{base}/readyz", args.token)
    report["checks"]["smokez"] = request_json("GET", f"{base}/smokez", args.token)
    report["checks"]["runtime_before"] = request_json("GET", f"{base}/runtimez", args.token)

    ref_b64 = None
    if args.mode == "clone":
        if not args.ref_audio or not args.ref_text:
            raise SystemExit("clone qualification requires --ref-audio and --ref-text")
        ref_b64 = base64.b64encode(Path(args.ref_audio).read_bytes()).decode("ascii")

    for language, text in SAMPLES.items():
        payload = {
            "text": text,
            "language": language,
            "mode": args.mode,
            "num_step": 32,
            "speed": 1.0,
        }
        if args.mode == "clone":
            payload["ref_audio_b64"] = ref_b64
            payload["ref_text"] = args.ref_text
        created = request_json("POST", f"{base}/v1/jobs", args.token, payload)
        record = wait_job(base, args.token, created["job_id"])
        report["languages"][language] = record
        if record["state"] != "completed":
            raise SystemExit(f"language {language} failed: {record.get('error')}")

    for index in range(args.stress_runs):
        payload = {
            "text": SAMPLES["hi"],
            "language": "hi",
            "mode": args.mode,
            "num_step": 16,
            "speed": 1.0,
        }
        if args.mode == "clone":
            payload["ref_audio_b64"] = ref_b64
            payload["ref_text"] = args.ref_text
        before = request_json("GET", f"{base}/runtimez", args.token)
        created = request_json("POST", f"{base}/v1/jobs", args.token, payload)
        record = wait_job(base, args.token, created["job_id"])
        after = request_json("GET", f"{base}/runtimez", args.token)
        report["stress"].append(
            {"run": index + 1, "job": record, "gpu_before": before.get("gpu"), "gpu_after": after.get("gpu")}
        )
        if record["state"] != "completed":
            raise SystemExit(f"stress run {index + 1} failed")

    report["checks"]["runtime_after"] = request_json("GET", f"{base}/runtimez", args.token)
    report["finished_at"] = time.time()
    report["overall"] = "PASS"
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": "PASS", "report": args.report}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc

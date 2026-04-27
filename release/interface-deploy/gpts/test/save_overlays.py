"""
Test-only helper: upload an image to /api/detect, poll status if needed,
and download the overlay/original images into a local folder.

Usage:
  python save_overlays.py --image /path/to/img.png --out ./test_outputs --server http://localhost:5000

This does NOT modify backend code; it just calls the running backend.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test uploader for /api/detect")
    p.add_argument("--server", default="http://localhost:5000", help="Backend server url (default: http://localhost:5000)")
    p.add_argument("--image", required=True, help="Path to image/DICOM to upload")
    p.add_argument("--out", default="test_outputs", help="Output folder to save images")
    p.add_argument("--poll-interval", type=float, default=1.2, help="Polling interval for async jobs (seconds)")
    return p.parse_args()


def _safe_json(text: str) -> Dict[str, Any]:
    try:
        return {} if not text else requests.utils.json.loads(text)
    except Exception:
        return {}


def upload(server: str, image_path: Path) -> Dict[str, Any]:
    url = f"{server.rstrip('/')}/api/detect"
    files = {"image": open(image_path, "rb")}
    resp = requests.post(url, files=files, timeout=60)
    data = _safe_json(resp.text)
    if not resp.ok or not data.get("success"):
        raise RuntimeError(f"upload failed: HTTP {resp.status_code}, msg={data.get('message')}")
    return data


def poll_status(server: str, status_url: str, interval: float) -> Dict[str, Any]:
    full_url = status_url if status_url.startswith("http") else f"{server.rstrip('/')}{status_url}"
    while True:
        r = requests.get(full_url, timeout=30)
        d = _safe_json(r.text)
        if d.get("status") == "done" and d.get("result"):
            return d["result"]
        if d.get("status") == "failed":
            raise RuntimeError(f"job failed: {d.get('error')}")
        time.sleep(interval)


def download(url: str, out_path: Path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    print(f"saved {out_path}")


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[upload] {image_path}")
    data = upload(args.server, image_path)

    result: Optional[Dict[str, Any]] = data.get("result")
    status_url = data.get("status_url") or data.get("statusUrl") or (data.get("job_id") and f"/api/detect/status/{data['job_id']}")

    if result is None:
        if not status_url:
            raise RuntimeError("No result and no status_url from server")
        print(f"[poll] {status_url}")
        result = poll_status(args.server, status_url, args.poll_interval)

    # 현재 백엔드가 단일 overlay만 내려주므로 같은 이미지를 복사 저장
    overlay_url = result.get("overlay_url")
    image_url = result.get("image_url")

    def full(u: Optional[str]) -> Optional[str]:
        if not u:
            return None
        return u if u.startswith("http") else f"{args.server.rstrip('/')}{u}"

    overlay_full = full(overlay_url)
    image_full = full(image_url)

    # 저장
    mapping = {
        "all.png": overlay_full,
        "teeth.png": image_full or overlay_full,
        "caries_peri.png": overlay_full,
        "other.png": overlay_full,
    }
    for name, url in mapping.items():
        if url:
            download(url, out_dir / name)
        else:
            print(f"skip {name}: no url")

    # det_counts 기록
    counts_path = out_dir / "det_counts.txt"
    counts = result.get("det_counts", {})
    counts_str = "\n".join(f"{k}: {v}" for k, v in counts.items())
    counts_path.write_text(counts_str or "no det_counts")
    print(f"det_counts saved to {counts_path}")


if __name__ == "__main__":
    main()

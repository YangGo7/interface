"""
Test-only helper: run multiple YOLO models directly and save separate overlays.
This is used by /api/test_split_detect and is isolated under backend/test/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
from ultralytics import YOLO


def _run_one(img_path: Path, weights: Optional[Path], out_dir: Path, name: str) -> Optional[Path]:
    if weights is None or not weights.exists():
        return None
    model = YOLO(str(weights))
    results = model.predict(str(img_path), verbose=False)
    if not results:
        return None
    arr = results[0].plot()  # numpy array (BGR)
    save_path = out_dir / f"{name}.png"
    cv2.imwrite(str(save_path), arr)
    return save_path


def run_split_models(
    image_path: Path,
    case_dir: Path,
    model_all: Optional[Path] = None,
    model_teeth: Optional[Path] = None,
    model_caries: Optional[Path] = None,
    model_other: Optional[Path] = None,
    model_extra: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Run up to 4 models on the same image and save overlays.
    Returns a dict of relative file paths (relative to case_dir.parent).
    """
    out_dir = case_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "all": _run_one(image_path, model_all, out_dir, "all"),
        "teeth": _run_one(image_path, model_teeth, out_dir, "teeth"),
        "caries_peri": _run_one(image_path, model_caries, out_dir, "caries_peri"),
        "other": _run_one(image_path, model_other, out_dir, "other"),
        "extra": _run_one(image_path, model_extra, out_dir, "extra"),
    }

    # Build relative paths (e.g., /temp/<case>/all.png)
    rel = {}
    for k, p in mapping.items():
        if p:
            rel[k] = f"/temp/{case_dir.name}/{p.name}"
    return rel

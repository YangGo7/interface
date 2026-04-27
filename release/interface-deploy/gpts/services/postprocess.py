import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.sample_axis_service import compute_from_gray_with_mask


def polygon_to_mask(segmentation_mask) -> Optional[np.ndarray]:
    """Convert polygon segmentation to binary mask (uint8) using mask size."""
    if not segmentation_mask or not segmentation_mask.counts:
        return None
    try:
        height, width = segmentation_mask.size
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(segmentation_mask.counts, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None
        cv2.fillPoly(mask, [pts], 255)
        return mask
    except Exception as exc:
        print(f"polygon_to_mask failed: {exc}")
        return None


def compute_diameter_for_label(
    detections,
    target_labels=None,
    target_names=None,
    label_map=None,
    image_path: Optional[str] = None,
    pixel_to_mm: float = 1.0,
):
    """Compute axis/diameter for detections with the target label or name using sample-axis logic."""
    target_labels = {str(t) for t in (target_labels or {"implant"})}
    target_names = {str(t).lower() for t in (target_names or set())}
    label_map = label_map or {}
    gray = None
    if image_path and Path(image_path).exists():
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    metrics = []
    for det in detections:
        lbl = str(det.label)
        name = label_map.get(det.label, label_map.get(lbl, ""))
        name_l = str(name).lower()
        if lbl not in target_labels and name_l not in target_names:
            continue
        print(f"[diameter] det_id={det.id} label={det.label} conf={det.confidence:.3f}")
        mask = polygon_to_mask(getattr(det, "segmentation_mask", None))
        if mask is None:
            print(f"[diameter] det_id={det.id} skip: no polygon mask")
            continue
        if gray is None:
            print(f"[diameter] det_id={det.id} skip: no gray image loaded")
            continue
        result = compute_from_gray_with_mask(gray, mask, n_samples=40)
        if not result:
            print(f"[diameter] det_id={det.id} skip: compute_from_gray_with_mask returned None")
            continue
        axis_center, axis_vec, max_d, p1, p2, mids = result
        if axis_center is None or axis_vec is None:
            print(f"[diameter] det_id={det.id} skip: axis_center/axis_vec is None")
            continue

        diameter_mm = float(max_d) * float(pixel_to_mm)
        diameter_pi = diameter_mm / math.pi if diameter_mm else 0.0
        diameter_mm_bucket = bucket_diameter_mm(diameter_mm)
        print(
            f"[diameter] det_id={det.id} diameter_mm={diameter_mm:.3f} bucket={diameter_mm_bucket:.1f} "
            f"diameter_pi={diameter_pi:.3f} max_d={max_d} mids={len(mids) if mids else 0}"
        )

        metrics.append(
            {
                "detection_id": det.id,
                "label": str(det.label),
                "axis_center": [float(axis_center[0]), float(axis_center[1])],
                "axis_vector": [float(axis_vec[0]), float(axis_vec[1])],
                "diameter_mm": diameter_mm_bucket,
                "diameter_mm_bucket": diameter_mm_bucket,
                "diameter_pi": diameter_pi,
                "diameter_endpoints": (
                    [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]]
                    if p1 is not None and p2 is not None
                    else None
                ),
                "sample_count": len(mids) if mids else 0,
            }
        )
    return metrics


def build_overlay_lines(diameter_metrics, img_width: int, img_height: int, detection_lookup=None):
    """Build overlay lines (axis and diameter) for rendering on frontend."""
    if not diameter_metrics:
        return []

    lines = []
    for metric in diameter_metrics:
        axis_center = metric.get("axis_center")
        axis_vec = metric.get("axis_vector")
        diameter_endpoints = metric.get("diameter_endpoints")
        det_id = metric.get("detection_id")
        mm_bucket = metric.get("diameter_mm_bucket")
        mm_raw = metric.get("diameter_mm")

        # [Filtering] Skip if detection center is in the outer 5% margin
        if axis_center:
            cx, cy = axis_center
            margin_x = img_width * 0.05
            margin_y = img_height * 0.05
            if not (margin_x <= cx <= img_width - margin_x and margin_y <= cy <= img_height - margin_y):
                 continue
        axis_length = None
        if detection_lookup and det_id in detection_lookup:
            bbox_h = detection_lookup[det_id].bounding_box.height
            axis_length = bbox_h * 1.5
        if not axis_length:
            axis_length = max(img_width, img_height) * 0.6

        if axis_center and axis_vec:
            cx, cy = axis_center
            vx, vy = axis_vec
            half = axis_length / 2.0
            p1 = [cx - vx * half, cy - vy * half]
            p2 = [cx + vx * half, cy + vy * half]
            lines.append(
                {
                    "type": "axis",
                    "label": metric.get("label"),
                    "detection_id": det_id,
                    "points": [p1, p2],
                    "length": float(axis_length),
                }
            )

        if diameter_endpoints:
            bucketed_mm = mm_bucket if mm_bucket is not None else bucket_diameter_mm(mm_raw if mm_raw is not None else 0.0)
            lines.append(
                {
                    "type": "diameter",
                    "label": metric.get("label"),
                    "detection_id": det_id,
                    "points": diameter_endpoints,
                    "length_mm": float(bucketed_mm),
                }
            )

    return lines


def deduplicate_detections_by_label(detections):
    """
    Keep only one detection per FDI label.
    Prefer detections that have a segmentation_mask; if multiple have masks, keep the highest-confidence one.
    If none have masks, keep the highest-confidence overall.
    Allow duplicates for specific FDI labels (e.g., implant, crown, endo, bridge).
    """
    allow_fdi = { "crown", "bridge", "endo","implant"}

    passthrough = [det for det in detections if str(det.label) in allow_fdi]
    to_dedupe = [det for det in detections if str(det.label) not in allow_fdi]

    best_by_label = {}
    for det in to_dedupe:
        key = str(det.label)
        has_mask = bool(getattr(det, "segmentation_mask", None))
        if key not in best_by_label:
            best_by_label[key] = det
            continue
        existing = best_by_label[key]
        existing_has_mask = bool(getattr(existing, "segmentation_mask", None))

        if has_mask and not existing_has_mask:
            best_by_label[key] = det
            continue
        if has_mask == existing_has_mask:
            if det.confidence > existing.confidence:
                best_by_label[key] = det
            continue
    deduped = sorted(best_by_label.values(), key=lambda d: d.id)
    combined = passthrough + deduped
    combined.sort(key=lambda d: d.id)
    return combined


def bucket_diameter_mm(value: float) -> float:
    """Bucket a mm value to the nearest 0.5 step."""
    if value is None:
        return 0.0
    return round(float(value) * 2.0) / 2.0

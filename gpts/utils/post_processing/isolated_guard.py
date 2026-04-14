from __future__ import annotations

from typing import List, Sequence

import numpy as np


ISOLATED_GUARD_HIGH_CONF = 0.25
ISOLATED_GUARD_LOW_CONF = 0.20
ISOLATED_GUARD_IOU = 0.30


def box_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate IoU for two boxes in [x1, y1, x2, y2] format."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0.0 else 0.0


def build_isolated_guard_keep_indices(
    boxes,
    confidences,
    high_conf: float = ISOLATED_GUARD_HIGH_CONF,
    low_conf: float = ISOLATED_GUARD_LOW_CONF,
    iou_guard: float = ISOLATED_GUARD_IOU,
) -> List[int]:
    """Keep confident boxes and only isolated low-tier boxes."""
    boxes_np = np.asarray(boxes, dtype=np.float32)
    confs_np = np.asarray(confidences, dtype=np.float32)

    if boxes_np.size == 0 or confs_np.size == 0:
        return []

    high_indices = [idx for idx, conf in enumerate(confs_np) if conf >= high_conf]
    low_indices = [
        idx for idx, conf in enumerate(confs_np) if low_conf <= conf < high_conf
    ]

    keep_indices = set(high_indices)
    high_boxes = boxes_np[high_indices] if high_indices else np.empty((0, 4))

    for low_idx in low_indices:
        low_box = boxes_np[low_idx]
        is_isolated = True

        for high_box in high_boxes:
            if box_iou_xyxy(low_box, high_box) > iou_guard:
                is_isolated = False
                break

        if is_isolated:
            keep_indices.add(low_idx)

    return sorted(keep_indices)


def filter_results_with_isolated_guard(
    result,
    high_conf: float = ISOLATED_GUARD_HIGH_CONF,
    low_conf: float = ISOLATED_GUARD_LOW_CONF,
    iou_guard: float = ISOLATED_GUARD_IOU,
):
    """Filter an Ultralytics Results object using the isolated guard logic."""
    if result.boxes is None or len(result.boxes) == 0:
        return result

    keep_indices = build_isolated_guard_keep_indices(
        boxes=result.boxes.xyxy.cpu().numpy(),
        confidences=result.boxes.conf.cpu().numpy(),
        high_conf=high_conf,
        low_conf=low_conf,
        iou_guard=iou_guard,
    )

    return result[keep_indices]

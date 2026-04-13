"""
Enhanced PBL axis comparison:
- Default PBL axis (get_principal_axis)
- PCA axis (row midpoints → farthest pair)
Select better axis per tooth using simple heuristics and overlay results.

Usage:
  python backend/test/pbl_axis_enhanced.py ^
    --image <img> ^
    --seg backend/weights/yolo11_seg_ver1_800_1024px.pt ^
    --cej backend/weights/cej.pt ^
    --bone backend/weights/bonelevel.pt ^
    --out backend/test/pbl_axis_enhanced.png
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# add repo root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.pano_calc_utils import (
    get_principal_axis,
    get_line_length,
    get_squeeze_points,
)


def load_seg(seg_w: Path, img_path: Path):
    model = YOLO(str(seg_w))
    res = model([str(img_path)], verbose=False)
    if not res:
        return None
    return res[0]


def load_mask_from_model(weights: Path, img_path: Path) -> Optional[np.ndarray]:
    model = YOLO(str(weights))
    res = model([str(img_path)], verbose=False)
    if not res or res[0].masks is None:
        return None
    img = cv2.imread(str(img_path))
    mask = np.zeros_like(img)
    for poly in res[0].masks.xy:
        cnt = get_squeeze_points(poly, copy.deepcopy(img), 0, cv2.CHAIN_APPROX_NONE)
        cv2.fillPoly(mask, cnt, (255, 255, 255))
    return mask


def pca_axis_from_mask(mask_roi: np.ndarray, n_samples: int = 40) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    ys, xs = np.where(mask_roi > 0)
    if xs.size < 2:
        return None
    y_min, y_max = ys.min(), ys.max()
    ys_to_probe = np.linspace(y_min, y_max, n_samples).astype(int) if y_max != y_min else np.array([y_min])
    mids = []
    for y in ys_to_probe:
        xs_row = xs[ys == y]
        if xs_row.size == 0:
            continue
        x1, x2 = xs_row.min(), xs_row.max()
        mids.append(((x1 + x2) / 2.0, float(y)))
    if len(mids) < 2:
        return None
    pts = np.array(mids, dtype=np.float32)
    # 전체 중점 집합에서 가장 먼 두 점을 축 끝점으로 선택
    max_d = -1.0
    p1 = pts[0]
    p2 = pts[1]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_d:
                max_d = d
                p1, p2 = pts[i], pts[j]
    return (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1])))


def axis_angle_deg(p1, p2) -> float:
    if p1 is None or p2 is None:
        return 0.0
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def compute_metrics_for_axis(
    axis_pts: Tuple[Tuple[int, int], Tuple[int, int]],
    bone_roi: np.ndarray,
    cej_roi: np.ndarray,
) -> Optional[Dict]:
    p1, p2 = axis_pts
    if p1 is None or p2 is None:
        return None
    length_axis = np.linalg.norm(np.array(p1) - np.array(p2))
    if length_axis < 5:  # 너무 짧으면 스킵
        return None
    axis_mask = np.zeros(bone_roi.shape[:2], dtype=np.uint8)
    cv2.line(axis_mask, p1, p2, 255, 1)
    bl_mask = cv2.cvtColor(bone_roi, cv2.COLOR_BGR2GRAY) if len(bone_roi.shape) == 3 else bone_roi
    cej_mask = cv2.cvtColor(cej_roi, cv2.COLOR_BGR2GRAY) if len(cej_roi.shape) == 3 else cej_roi
    L_bl = get_line_length(cv2.bitwise_and(axis_mask, bl_mask))
    L_cej = get_line_length(cv2.bitwise_and(axis_mask, cej_mask))
    periodontal_to_root = length_axis - L_bl
    cej_to_root = length_axis - L_cej
    if cej_to_root <= 0:
        return None
    pbl_ratio = periodontal_to_root / cej_to_root
    return {
        "len_axis": length_axis,
        "L_bl": L_bl,
        "L_cej": L_cej,
        "pbl_percent": float(pbl_ratio * 100),
    }


def main():
    ap = argparse.ArgumentParser(description="Enhanced PBL axis comparison (per tooth)")
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--seg", required=True, help="tooth segmentation weight path")
    ap.add_argument("--cej", required=True, help="cej weight path")
    ap.add_argument("--bone", required=True, help="bonelevel weight path")
    ap.add_argument("--out", default="backend/test/pbl_axis_enhanced.png", help="output image path")
    args = ap.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    seg_res = load_seg(Path(args.seg), img_path)
    bone_mask = load_mask_from_model(Path(args.bone), img_path)
    cej_mask = load_mask_from_model(Path(args.cej), img_path)
    if bone_mask is None or cej_mask is None:
        raise RuntimeError("Failed to get CEJ/Bone masks")

    overlay = img.copy()
    logs = []

    if seg_res is None or seg_res.masks is None:
        raise RuntimeError("No tooth segmentation found")

    left_th = -10
    for idx, coords in enumerate(seg_res.masks.xy):
        keyval = seg_res.names[int(seg_res.boxes.cls[idx])]
        if len(keyval) > 1 and keyval[1] == "8":
            continue
        key_lower = str(keyval).lower()
        if any(skip in key_lower for skip in ("implant", "bridge", "crown")):
            continue
        res2 = get_squeeze_points(coords, copy.deepcopy(img), 1)
        cnt = res2[0]
        x, y, w, h = cv2.boundingRect(cnt)
        if len(keyval) > 1 and keyval[1] == "8" and h <= w:
            continue

        # ROI 추출
        tooth_roi = np.zeros_like(img)
        cv2.fillPoly(tooth_roi, res2, (255, 255, 255))
        tooth_roi = tooth_roi[y : y + h, x + left_th : x + w]
        bone_roi = bone_mask[y : y + h, x + left_th : x + w]
        cej_roi = cej_mask[y : y + h, x + left_th : x + w]

        # 축 후보
        axis_pbl = get_principal_axis(tooth_roi, keyval)
        axis_pca = pca_axis_from_mask(cv2.cvtColor(tooth_roi, cv2.COLOR_BGR2GRAY) if len(tooth_roi.shape) == 3 else tooth_roi)

        # 메트릭 계산
        met_pbl = compute_metrics_for_axis(axis_pbl, bone_roi, cej_roi)
        met_pca = compute_metrics_for_axis(axis_pca, bone_roi, cej_roi) if axis_pca else None

        # 선택 로직
        chosen = "pbl"
        chosen_axis = axis_pbl
        chosen_met = met_pbl
        alt_axis = axis_pca
        alt_met = met_pca
        ang_diff = None

        if met_pbl is None and met_pca is not None:
            chosen, chosen_axis, chosen_met = "pca", axis_pca, met_pca
        elif met_pbl is not None and met_pca is not None:
            ang_diff = abs(axis_angle_deg(*axis_pbl) - axis_angle_deg(*axis_pca))
            # PCA가 CEJ 겹침이 더 길고, 각도 차이가 크면 교체
            if ang_diff > 20 and met_pca["L_cej"] > met_pbl["L_cej"] * 1.1:
                chosen, chosen_axis, chosen_met = "pca", axis_pca, met_pca

        # 그리기
        color = (0, 255, 0) if chosen == "pbl" else (255, 0, 0)
        if chosen_axis and chosen_axis[0] and chosen_axis[1]:
            cv2.line(
                overlay[y : y + h, x + left_th : x + w],
                chosen_axis[0],
                chosen_axis[1],
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(overlay, (x + left_th + chosen_axis[0][0], y + chosen_axis[0][1]), 3, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x + left_th + chosen_axis[1][0], y + chosen_axis[1][1]), 3, color, -1, cv2.LINE_AA)

        if chosen_met:
            logs.append(f"{keyval}: {chosen} {chosen_met['pbl_percent']:.1f}% (Lcej={chosen_met['L_cej']:.1f})"
                        + (f" ang_diff={ang_diff:.1f}" if ang_diff is not None else ""))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    print(f"[saved] {out_path}")
    for line in logs:
        print(line)


if __name__ == "__main__":
    main()

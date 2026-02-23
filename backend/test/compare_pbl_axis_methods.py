"""
Compare per-tooth PBL using two axis strategies on the same CEJ/Bonelevel masks:
- pbl_axis: existing get_principal_axis (root/crown bands)
- pca_axis: row-midpoint sampling + PCA

Outputs a side-by-side overlay (left=pbl_axis, right=pca_axis) with axes drawn.

Usage:
  python backend/test/compare_pbl_axis_methods.py ^
    --image <img> ^
    --seg backend/weights/yolo11_seg_ver1_800_1024px.pt ^
    --cej backend/weights/cej.pt ^
    --bone backend/weights/bonelevel.pt ^
    --out backend/test/pbl_axis_compare.png
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

# add repo root for backend imports
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
    # 가장 먼 두 점을 선택 (다근치에서 루트별로 나뉘지 않고 전체 형태 최장거리 사용)
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


def compute_pbl_for_axis(
    img: np.ndarray,
    seg_res,
    bone_mask: np.ndarray,
    cej_mask: np.ndarray,
    axis_mode: str = "pbl",
    left_th: int = -10,
) -> Tuple[np.ndarray, Dict[str, float]]:
    overlay = np.zeros_like(img)
    pbl_percent: Dict[str, float] = {}
    if seg_res is None or seg_res.masks is None:
        return overlay, pbl_percent

    for idx, coords in enumerate(seg_res.masks.xy):
        try:
            keyval = seg_res.names[int(seg_res.boxes.cls[idx])]
            # 8번 치아 제외
            if len(keyval) > 1 and keyval[1] == "8":
                continue
            key_lower = str(keyval).lower()
            # 보철/질병 클래스 제외
            if any(skip in key_lower for skip in ("implant", "bridge", "crown")):
                continue

            res2 = get_squeeze_points(coords, copy.deepcopy(img), 1)
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, res2, (255, 255, 255))
            cnt = res2[0]
            x, y, w, h = cv2.boundingRect(cnt)
            if len(keyval) > 1 and keyval[1] == "8" and h <= w:
                continue

            # ROI 추출
            tooth_roi = mask[y : y + h, x + left_th : x + w]
            bone_roi = bone_mask[y : y + h, x + left_th : x + w]
            cej_roi = cej_mask[y : y + h, x + left_th : x + w]

            # 축 계산
            if axis_mode == "pbl":
                pt1, pt2 = get_principal_axis(tooth_roi, keyval)
            else:
                pts = pca_axis_from_mask(cv2.cvtColor(tooth_roi, cv2.COLOR_BGR2GRAY) if len(tooth_roi.shape) == 3 else tooth_roi)
                if pts is None:
                    continue
                pt1, pt2 = pts

            # ROI 내 축 그리기
            tooth_axis_roi = np.zeros(tooth_roi.shape[:2], dtype=np.uint8)
            cv2.line(tooth_axis_roi, pt1, pt2, 255, 1)

            periodontal_roi = cv2.bitwise_and(
                tooth_axis_roi,
                cv2.cvtColor(bone_roi, cv2.COLOR_BGR2GRAY) if len(bone_roi.shape) == 3 else bone_roi,
            )
            periodontal_to_root_val = np.linalg.norm(np.array(pt1) - np.array(pt2)) - get_line_length(periodontal_roi)

            cej_overlap_roi = cv2.bitwise_and(
                tooth_axis_roi,
                cv2.cvtColor(cej_roi, cv2.COLOR_BGR2GRAY) if len(cej_roi.shape) == 3 else cej_roi,
            )
            cej_to_root_val = np.linalg.norm(np.array(pt1) - np.array(pt2)) - get_line_length(cej_overlap_roi)
            if cej_to_root_val <= 0:
                continue

            pbl_ratio = periodontal_to_root_val / cej_to_root_val
            pbl_percent[keyval] = float(pbl_ratio * 100)

            # 전역 오버레이에 축 그리기 (좌표 보정)
            cv2.line(
                overlay[y : y + h, x + left_th : x + w],
                pt1,
                pt2,
                (0, 255, 255) if axis_mode == "pbl" else (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            continue

    return overlay, pbl_percent


def main():
    ap = argparse.ArgumentParser(description="Compare PBL per tooth with two axis strategies (pbl vs pca)")
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--seg", required=True, help="tooth segmentation weight path")
    ap.add_argument("--cej", required=True, help="cej weight path")
    ap.add_argument("--bone", required=True, help="bonelevel weight path")
    ap.add_argument("--out", default="backend/test/pbl_axis_compare.png", help="output image path")
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

    overlay_pbl, pbl_vals = compute_pbl_for_axis(img, seg_res, bone_mask, cej_mask, axis_mode="pbl")
    overlay_pca, pca_vals = compute_pbl_for_axis(img, seg_res, bone_mask, cej_mask, axis_mode="pca")

    left = cv2.add(img, overlay_pbl)
    right = cv2.add(img, overlay_pca)
    concat = np.concatenate([left, right], axis=1)
    cv2.putText(concat, "PBL axis", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(concat, "PCA axis", (img.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), concat)
    print(f"[saved] {out_path}")
    # 간단한 값 비교 출력
    common_keys = set(pbl_vals.keys()) & set(pca_vals.keys())
    for k in sorted(common_keys):
        print(f"Tooth {k}: PBL-axis {pbl_vals[k]:.1f}% | PCA-axis {pca_vals[k]:.1f}%")


if __name__ == "__main__":
    main()

"""
Compare two axis estimation methods on the same CEJ/Bonelevel masks:
- minAreaRect long-side midpoints (current implant-style)
- PCA on row midpoints (centerline sampling)

Outputs side-by-side overlays for quick visual comparison.

Usage:
  python backend/test/compare_axis_cej_bone.py --image img.png ^
    --cej backend/weights/cej.pt --bone backend/weights/bonelevel.pt ^
    --out ./axis_compare.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def load_masks(weights: Path, image_path: Path) -> List[np.ndarray]:
    model = YOLO(str(weights))
    res = model.predict(str(image_path), verbose=False)
    if not res or res[0].masks is None:
        return []
    masks = []
    for poly in res[0].masks.xy:
        pts = np.array(poly, dtype=np.int32)
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)
        h = max(1, y_max - y_min + 1)
        w = max(1, x_max - x_min + 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = pts - np.array([x_min, y_min])
        cv2.fillPoly(mask, [shifted], 255)
        masks.append((mask, (x_min, y_min)))
    return masks


def axis_minarearect(mask: np.ndarray, origin: Tuple[int, int]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (None, None)
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    def dist(p1, p2):
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    d01 = dist(box[0], box[1])
    d12 = dist(box[1], box[2])
    if d01 > d12:
        m1 = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
        m2 = ((box[2][0] + box[3][0]) / 2, (box[2][1] + box[3][1]) / 2)
    else:
        m1 = ((box[1][0] + box[2][0]) / 2, (box[1][1] + box[2][1]) / 2)
        m2 = ((box[3][0] + box[0][0]) / 2, (box[3][1] + box[0][1]) / 2)
    ox, oy = origin
    return (m1[0] + ox, m1[1] + oy), (m2[0] + ox, m2[1] + oy)


def pca_axis(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if pts.shape[0] < 2:
        return np.array([0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov(pts - mean, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return mean.astype(np.float32), axis.astype(np.float32)


def sample_axis_pca(mask: np.ndarray, origin: Tuple[int, int], n_samples: int = 40) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if xs.size < 2:
        return (None, None)

    y_min, y_max = ys.min(), ys.max()
    ys_to_probe = np.linspace(y_min, y_max, n_samples).astype(int) if y_max != y_min else np.array([y_min])
    mids = []
    for y in ys_to_probe:
        xs_row = xs[ys == y]
        if xs_row.size == 0:
            continue
        x1, x2 = xs_row.min(), xs_row.max()
        mids.append(((x1 + x2) / 2.0, float(y)))
    pts = np.array(mids, dtype=np.float32)
    c_pca, axis_vec = pca_axis(pts)
    proj = (pts - c_pca) @ axis_vec
    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))
    p1 = pts[i_min]
    p2 = pts[i_max]
    ox, oy = origin
    return (p1[0] + ox, p1[1] + oy), (p2[0] + ox, p2[1] + oy)


def draw_axis(img: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float], color: Tuple[int, int, int]) -> np.ndarray:
    if p1 is None or p2 is None:
        return img
    p1_i = (int(round(p1[0])), int(round(p1[1])))
    p2_i = (int(round(p2[0])), int(round(p2[1])))
    cv2.line(img, p1_i, p2_i, color, 2, cv2.LINE_AA)
    cv2.circle(img, p1_i, 4, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2_i, 4, color, -1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Compare axis estimators (minAreaRect vs PCA sampling) on CEJ/Bone masks")
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--cej", required=True, help="cej weight path")
    ap.add_argument("--bone", required=True, help="bonelevel weight path")
    ap.add_argument("--out", default="axis_compare.png", help="output image path")
    args = ap.parse_args()

    img_path = Path(args.image)
    out_path = Path(args.out)
    cej_w = Path(args.cej)
    bone_w = Path(args.bone)

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    bone_masks = load_masks(bone_w, img_path)
    if not bone_masks:
        raise RuntimeError("No bonelevel masks detected")
    # 첫 번째 bonelevel 마스크 기준 비교
    bone_mask, origin = bone_masks[0]

    axis_min = axis_minarearect(bone_mask, origin)
    axis_pca = sample_axis_pca(bone_mask, origin, n_samples=40)

    # 시각화: 좌=MinAreaRect, 우=PCA
    left = img.copy()
    right = img.copy()
    draw_axis(left, *axis_min, (0, 255, 255))     # yellow
    draw_axis(right, *axis_pca, (255, 0, 0))      # blue

    concat = np.concatenate([left, right], axis=1)
    cv2.putText(concat, "minAreaRect", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(concat, "PCA sampling", (img.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), concat)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

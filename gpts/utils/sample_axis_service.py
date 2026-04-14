import math
from typing import List, Tuple

import cv2
import numpy as np


def pca_axis(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return center and principal axis (unit vector) for 2D points."""
    if pts.shape[0] < 2:
        return np.array([0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)
    mean = pts.mean(axis=0)
    cov = np.cov(pts - mean, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return mean.astype(np.float32), axis.astype(np.float32)


def sample_centerline_points(mask: np.ndarray, axis_center, axis_vec, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Row-wise midpoints inside mask; axis arguments are unused (kept for API parity)."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return []
    y_min, y_max = ys.min(), ys.max()
    if n_samples <= 1 or y_max == y_min:
        ys_to_probe = np.unique(ys)
    else:
        ys_to_probe = np.linspace(y_min, y_max, n_samples).astype(int)
    rows = []
    for y in ys_to_probe:
        xs_row = xs[ys == y]
        if xs_row.size == 0:
            continue
        x1 = xs_row.min()
        x2 = xs_row.max()
        mid = np.array([(x1 + x2) / 2.0, y], dtype=np.float32)
        rows.append((mid, np.array([x1, y], dtype=np.float32), np.array([x2, y], dtype=np.float32)))
    return rows


def filter_outliers(midpoints, center, axis, tol_ratio=0.15):
    """Remove points that deviate too far in x or radial distance."""
    if not midpoints:
        return [], [], 0
    xs = [m[0][0] for m in midpoints]
    ys = [m[0][1] for m in midpoints]
    cx, cy = center if center is not None else (np.mean(xs), np.mean(ys))
    rng_x = max(xs) - min(xs)
    x_thr = rng_x * (0.5 + tol_ratio)
    dists = [math.hypot(mx - cx, my - cy) for mx, my in zip(xs, ys)]
    mean_dist = float(np.mean(dists)) if dists else 0.0
    dist_thr = mean_dist * (1.0 + tol_ratio) if mean_dist > 0 else x_thr
    keep, out = [], []
    for entry, dx, dy, dist in zip(midpoints, [abs(x - cx) for x in xs], [abs(y - cy) for y in ys], dists):
        if dx <= x_thr and dist <= dist_thr:
            keep.append(entry)
        else:
            out.append(entry)
    return keep, out, max(x_thr, dist_thr)


def fit_axis_pca_points(midpoints):
    if len(midpoints) < 2:
        return None, None
    pts = np.array([[m[0][0], m[0][1]] for m in midpoints], dtype=np.float32)
    return pca_axis(pts)


def max_horizontal_run_pos(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Find the longest horizontal run of foreground; returns length and coords in mask frame."""
    max_len, best_x1, best_x2, best_y = 0, None, None, None
    for y in range(mask.shape[0]):
        row = mask[y, :]
        x_on = np.where(row > 0)[0]
        if x_on.size == 0:
            continue
        run_len = x_on.max() - x_on.min() + 1
        if run_len > max_len:
            max_len = run_len
            best_x1, best_x2, best_y = int(x_on.min()), int(x_on.max()), y
    return max_len, best_x1, best_x2, best_y


def max_width_perp_to_axis(mask: np.ndarray, center: np.ndarray, axis: np.ndarray):
    """Fallback width perpendicular to given axis."""
    if center is None or axis is None or np.linalg.norm(axis) < 1e-6:
        return 0, None, None
    perp = np.array([-axis[1], axis[0]], dtype=np.float32)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0, None, None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    proj = (pts - center) @ perp
    idx_min = int(np.argmin(proj))
    idx_max = int(np.argmax(proj))
    w = float(proj[idx_max] - proj[idx_min])
    return w, (pts[idx_min][0], pts[idx_min][1]), (pts[idx_max][0], pts[idx_max][1])


def compute_sample_axis(geom_mask: np.ndarray, width_mask: np.ndarray, n_samples: int = 40) -> Tuple:
    """Compute sample-based center axis and diameter inside a binary mask."""
    ys_all, xs_all = np.where(geom_mask > 0)
    if xs_all.size < 2:
        return None, None, 0, None, None, []
    pts_xy = np.stack([xs_all, ys_all], axis=1).astype(np.float32)
    c_pca, axis_pca = pca_axis(pts_xy)
    mids_raw = sample_centerline_points(geom_mask, c_pca, axis_pca, n_samples=n_samples)
    if len(mids_raw) > 2:
        drop = max(1, int(round(len(mids_raw) * 0.10)))  # central 80%
        mids_raw = mids_raw[drop:-drop] if len(mids_raw) > 2 * drop else mids_raw
    keep_mids, out, _ = filter_outliers(mids_raw, c_pca, axis_pca, tol_ratio=0.15)
    if len(keep_mids) > 2:
        drop = max(1, int(round(len(keep_mids) * 0.10)))
        keep_mids = keep_mids[drop:-drop] if len(keep_mids) > 2 * drop else keep_mids
    axis_center, axis_vec = fit_axis_pca_points(keep_mids) if keep_mids else (None, None)
    if axis_center is None or axis_vec is None or np.linalg.norm(axis_vec) < 1e-6:
        axis_center, axis_vec = c_pca, axis_pca
    if axis_center is None or axis_vec is None or np.linalg.norm(axis_vec) < 1e-6:
        return None, None, 0, None, None, keep_mids
    angle_deg = math.degrees(math.atan2(axis_vec[0], axis_vec[1]))
    rot_mat_w = cv2.getRotationMatrix2D(
        (width_mask.shape[1] / 2.0, width_mask.shape[0] / 2.0), -angle_deg, 1.0
    )
    width_upright = cv2.warpAffine(
        width_mask, rot_mat_w, (width_mask.shape[1], width_mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0
    )
    max_d_upright, x1_u, x2_u, y_u = max_horizontal_run_pos(width_upright)
    if max_d_upright > 0 and x1_u is not None and x2_u is not None and y_u is not None:
        inv_rot_w = cv2.invertAffineTransform(rot_mat_w)
        p1_back = cv2.transform(np.array([[[x1_u, y_u]]], dtype=np.float32), inv_rot_w)[0, 0]
        p2_back = cv2.transform(np.array([[[x2_u, y_u]]], dtype=np.float32), inv_rot_w)[0, 0]
        return axis_center, axis_vec, max_d_upright, (p1_back[0], p1_back[1]), (p2_back[0], p2_back[1]), keep_mids
    max_d, max_p1, max_p2 = max_width_perp_to_axis(width_mask, axis_center, axis_vec)
    return axis_center, axis_vec, max_d, max_p1, max_p2, keep_mids


def draw_sample_overlay(canvas: np.ndarray, mid_records, axis_center, axis_vec, max_p1, max_p2, axis_color=(0, 0, 255), show_axis=True):
    """(Disabled) Demo/debug overlay. Left as no-op to avoid extra drawings."""
    return


# --------------------------------------------------------------------------
# Preprocess helpers (CLAHE + histogram equalization + threshold)
# --------------------------------------------------------------------------

def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to improve local contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def apply_hist_eq(gray: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization."""
    return cv2.equalizeHist(gray)


def binarize_gray(gray: np.ndarray, thr: int = 114, use_clahe: bool = True, use_heq: bool = True) -> np.ndarray:
    """CLAHE + HEQ (optional) then manual threshold -> binary uint8."""
    proc = apply_clahe(gray) if use_clahe else gray
    proc = apply_hist_eq(proc) if use_heq else proc
    _, binary = cv2.threshold(proc, thr, 255, cv2.THRESH_BINARY)
    return binary


# --------------------------------------------------------------------------
# Contour 기반 편의 함수 (백엔드에서 바로 사용)
# --------------------------------------------------------------------------

def largest_contour(binary: np.ndarray):
    """Return largest contour from a binary mask (None if empty)."""
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts[0]


def contour_to_masks(contour, shape):
    """Given a contour and target shape (h, w), return filled masks (geom/width are same here)."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask, mask.copy()


def compute_from_gray_with_mask(
    gray: np.ndarray,
    mask: np.ndarray,
    thr: int = 137,
    use_clahe: bool = True,
    use_heq: bool = True,
    n_samples: int = 40,
):
    """
    Gray + provided mask를 결합하여 계산.
    - gray를 CLAHE/HEQ 후 threshold로 이진화
    - 모델 마스크와 AND 연산으로 범위를 제한 (마스크의 x/y 범위 밖은 무시)
    - 결합된 바이너리로 컨투어→축/직경 계산
    """
    if gray.ndim != 2:
        raise ValueError("Expected grayscale image for compute_from_gray_with_mask")
    if mask.ndim != 2:
        raise ValueError("Expected binary mask (H,W) for compute_from_gray_with_mask")
    bin_gray = binarize_gray(gray, thr=thr, use_clahe=use_clahe, use_heq=use_heq)
    # 마스크 범위 밖 제거
    combined = cv2.bitwise_and(bin_gray, mask)
    cnt = largest_contour(combined)
    if cnt is None:
        return None
    geom_mask, width_mask = contour_to_masks(cnt, combined.shape[:2])
    return compute_sample_axis(geom_mask, width_mask, n_samples=n_samples)

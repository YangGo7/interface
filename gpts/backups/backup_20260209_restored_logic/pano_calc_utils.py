import copy
import math
import statistics as st
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Geometry helpers for axis selection
def _angle_deg(p1, p2) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = float(np.degrees(np.arctan2(dy, dx)))
    return abs(ang) % 180.0


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def pca_axis_from_mask(mask_roi: np.ndarray, n_samples: int = 40):
    """
    Row-wise midpoint sampling -> choose farthest pair as axis (handles multi-root without splitting).
    Returns ((x1,y1),(x2,y2)) in ROI coords or None.
    """
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
    max_d = -1.0
    p1, p2 = pts[0], pts[1]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_d:
                max_d = d
                p1, p2 = pts[i], pts[j]
    return (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1])))

# Paths will be set at runtime
BONELEVEL = ""
CEJ = ""


def set_weight_paths(bonelevel: Path, cej: Path) -> None:
    global BONELEVEL, CEJ
    BONELEVEL = str(bonelevel)
    CEJ = str(cej)


def get_squeeze_points(coordinates, img_path: np.ndarray, e: int = 0, opt=cv2.CHAIN_APPROX_NONE):
    if coordinates is None or len(coordinates) < 3:
        return []

    def erode(src, epoch=1):
        dst = copy.deepcopy(src)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for _ in range(epoch):
            dst = cv2.erode(dst, k)
        return dst

    img = img_path
    h, w = img.shape[:2]
    mask = np.zeros([h, w], dtype=np.uint8)
    linearr = []
    
    # Safely convert coordinates
    try:
        # Check if coordinates is already a numpy array of shape (N, 2)
        if isinstance(coordinates, np.ndarray) and coordinates.ndim == 2 and coordinates.shape[1] == 2:
             pts = coordinates.astype(np.int32)
             linearr = pts
        else:
            for xy in coordinates:
                x = int(xy[0])
                y = int(xy[1])
                linearr.append([x, y])
            linearr = np.array(linearr, dtype=np.int32)
            
        if len(linearr) > 0:
            mask = cv2.fillPoly(mask, [linearr], (255, 255, 255))
        
        post_mask = erode(mask, e)
        contour, _ = cv2.findContours(post_mask, cv2.RETR_EXTERNAL, opt)
        return contour
    except Exception as ex:
        # print(f"get_squeeze_points error: {ex}")
        return []


def contour_to_polyline(img: np.ndarray, coords, color=(0, 0, 255), thickness=2):
    cnts = get_squeeze_points(coords, img, 0, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        pts = c.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    return img


def _get_bb_from_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return [x, y, x + w, y + h]


def _calc_iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    xx1 = max(x1, a1)
    yy1 = max(y1, b1)
    xx2 = min(x2, a2)
    yy2 = min(y2, b2)
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def map_boxes_to_teeth(detect_boxes, seg_result):
    mapped = []
    if seg_result is None or seg_result.masks is None:
        return mapped
    tooth_boxes = []
    for idx, coords in enumerate(seg_result.masks.xy):
        cnts = get_squeeze_points(coords, copy.deepcopy(seg_result.orig_img), 0)
        bb = _get_bb_from_contour(cnts[0])
        label = seg_result.names[int(seg_result.boxes.cls[idx])]
        tooth_boxes.append((label, bb))
    for box, conf in detect_boxes:
        best_iou = 0
        best_label = None
        for label, tbb in tooth_boxes:
            iou = _calc_iou(box, tbb)
            if iou > best_iou:
                best_iou = iou
                best_label = label
        mapped.append((best_label if best_label is not None else "unknown", conf, box))
    return mapped


def get_principal_axis(cropped_img, num: str):
    def get_root_pos(img, th=5, opt="up"):
        """NumPy vectorized boundary finding (Optimization 1, 3)"""
        # Optimization 3: Slice region of interest (th lines)
        roi = img[:th, :] if opt == "up" else img[-th:, :]
        start_y = 0 if opt == "up" else len(img) - 1

        points = np.argwhere(roi > 0)
        if points.size == 0:
            return (int(img.shape[1] / 2), start_y)
        
        # Optimization 1: Get range using NumPy min/max
        most_left = np.min(points[:, 1])
        most_right = np.max(points[:, 1])
        return (int((most_left + most_right) / 2), start_y)

    def get_crown_pos(img, num, th=15, opt="up"):
        if len(num) > 1 and num[1] in ["6", "7"]:
            th = 30
        
        # Optimization 3: Slice ROI
        roi = img[-th:, :] if opt == "up" else img[:th, :]
        edge_y = len(img) if opt == "up" else 0

        points = np.argwhere(roi > 0)
        if points.size == 0:
            return (int(img.shape[1] / 2), edge_y)
        
        most_left = np.min(points[:, 1])
        most_right = np.max(points[:, 1])
        return (int((most_left + most_right) / 2), edge_y)

    # Pre-convert to grayscale (Optimization 2)
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) if len(cropped_img.shape) == 3 else cropped_img

    def check_multi_root(img, th=20, opt="up"):
        """Optimization 4: Smarter multi-root detection"""
        start = len(img) - th
        end = len(img)
        step = 1
        flag = False
        x_arr = []
        y_arr = []
        if opt == "up":
            start = th
            end = 0
            step = -1
        
        for col in range(start, end, step):
            crop = img[end:col, :] if opt == "up" else img[col:end, :]
            # Check for multiple components in small ROI
            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(crop)
            if cnt > 2:
                flag = True
                for i in range(1, cnt):
                    (x, y, w, h, area) = stats[i]
                    if opt == "up":
                        pt_r = get_root_pos(crop[y:y + h + 1, x:x + w + 1], th=5)
                        y_bias = 0
                    else:
                        pt_r = get_root_pos(crop[y:y + h, x:x + w], th=5, opt="lo")
                        y_bias = col
                    x_arr.append(pt_r[0] + x)
                    y_arr.append(pt_r[1] + y + y_bias)
                break
        
        if not x_arr:
            return (0, 0), False
        
        avg_x = int(np.mean(x_arr))
        avg_y = int(st.harmonic_mean(y_arr))
        return (avg_x, avg_y), flag

    f = False
    pt1 = (0, 0)
    if num and num[0] in ["1", "2"]:
        if len(num) > 1 and num[1] in ["6", "7"]:
            pt1, f = check_multi_root(gray_img, 55)
        if not f:
            pt1 = get_root_pos(gray_img, 3)
        pt2 = get_crown_pos(gray_img, num, 5)
    else:
        if len(num) > 1 and num[1] in ["6", "7"]:
            pt1, f = check_multi_root(gray_img, 55, "lo")
        if not f:
            pt1 = get_root_pos(gray_img, 3, "lo")
        pt2 = get_crown_pos(gray_img, num, 5, "lo")
    return pt1, pt2


def get_line_length(image):
    """NumPy vectorized line length calculation (Optimizations 1, 2)"""
    if image is None or image.size == 0:
        return 0.0
    
    # Check if image is already grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Optimization 1: Use np.nonzero to avoid Python loops
    points = np.argwhere(gray > 0)
    if points.size == 0:
        return 0.0
    
    # Distance between first and last non-zero points
    first, last = points[0], points[-1]
    return float(np.linalg.norm(first - last))


def bonelevel_postprocessing(img, coord):
    most_right = 0
    most_left = 9999
    th = 40
    for c in coord[0]:
        x = c[0][0]
        if x > most_right:
            most_right = x
        if x < most_left:
            most_left = x
    h, w, _ = img.shape
    mask = np.zeros([h, w], dtype=np.uint8)
    mask_post = np.zeros([h, w])
    most_left = int(most_left)
    most_right = int(most_right)
    most_left_list = [x for x in range(most_left, most_left + th)]
    most_right_list = [x for x in range(most_right, most_right - th, -1)]
    for c in coord[0]:
        x = round(c[0][0])
        y = round(c[0][1])
        if x in most_left_list or x in most_right_list:
            pass
        else:
            mask_post[y][x] = 255
    mask_post = mask_post.astype(np.uint8)
    mask_post = cv2.merge([mask_post, mask, mask])
    dst = copy.deepcopy(mask_post)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.dilate(dst, k)
    return dst


def get_bonelevel(img, seg_res, bonelv_model=None, cej_model=None, axis_only: bool = False):
    """
    Full bonelevel visualization + PBL% (adapted from demo/temp_utils.py).
    Returns (overlay, pbl_dict, cej_count, bonelevel_count).
    """
    if not BONELEVEL or not CEJ:
        return np.zeros_like(img), {}, 0, 0

    if bonelv_model is None:
        bonelv = YOLO(BONELEVEL)
        bonelv_result = bonelv(img)
    else:
        bonelv_result = bonelv_model(img)

    if len(bonelv_result) == 0 or bonelv_result[0].masks is None:
        return np.zeros_like(img), {}, 0, 0

    if cej_model is None:
        cej_m = YOLO(CEJ)
        cej_result = cej_m(img)
    else:
        cej_result = cej_model(img)

    if len(cej_result) == 0 or cej_result[0].masks is None:
        return np.zeros_like(img), {}, 0, 0

    overlay = np.zeros_like(img)  # draw lines/contours only
    bonelevel_dict = {}

    def _bbox_intersects(b1, b2):
        x1, y1, x2, y2 = b1
        a1, b1_, a2, b2_ = b2
        return not (x2 < a1 or a2 < x1 or y2 < b1_ or b2_ < y1)

    # Build masks and polylines
    bonelevel_mask = np.zeros_like(img)
    bl_cnt = get_squeeze_points(bonelv_result[0].masks.xy[0], copy.deepcopy(img), 0, cv2.CHAIN_APPROX_NONE)
    cv2.fillPoly(bonelevel_mask, bl_cnt, (255, 255, 255))
    bl_bbox = cv2.boundingRect(bl_cnt[0]) if bl_cnt else (0, 0, img.shape[1], img.shape[0])
    bl_bbox_xyxy = (bl_bbox[0], bl_bbox[1], bl_bbox[0] + bl_bbox[2], bl_bbox[1] + bl_bbox[3])
    # outline bonelevel (yellow), thicker
    if not axis_only:
        overlay = contour_to_polyline(overlay, bonelv_result[0].masks.xy[0], color=(0, 255, 255), thickness=2)

    cej_mask = np.zeros_like(img)
    for t in cej_result[0].masks.xy:
        cej_cnt = get_squeeze_points(t, copy.deepcopy(img), 0, cv2.CHAIN_APPROX_NONE)
        if not cej_cnt:
            continue
        bb = cv2.boundingRect(cej_cnt[0])
        cej_bb_xyxy = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
        if not _bbox_intersects(cej_bb_xyxy, bl_bbox_xyxy):
            continue  # skip CEJ far from bonelevel
        cv2.fillPoly(cej_mask, cej_cnt, (255, 255, 255))
        # outline CEJ (blue), thicker
        if not axis_only:
            overlay = contour_to_polyline(overlay, t, color=(255, 0, 0), thickness=2)

    # Per-tooth axis and PBL
    if seg_res.masks is None:
        return overlay, {}

    def _compute_metrics(axis_pts, bone_roi, cej_roi, min_len=10.0):
        if axis_pts is None or axis_pts[0] is None or axis_pts[1] is None:
            return None
        p1, p2 = axis_pts
        axis_len = np.linalg.norm(np.array(p1) - np.array(p2))
        if axis_len < min_len:
            return None
            
        bl_mask_roi = cv2.cvtColor(bone_roi, cv2.COLOR_BGR2GRAY) if len(bone_roi.shape) == 3 else bone_roi
        cej_mask_roi = cv2.cvtColor(cej_roi, cv2.COLOR_BGR2GRAY) if len(cej_roi.shape) == 3 else cej_roi
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bl_mask_roi = cv2.dilate(bl_mask_roi, k)
        cej_mask_roi = cv2.dilate(cej_mask_roi, k)
        
        # Determine points of intersection along the line p1->p2
        # Create a line mask
        h_r, w_r = bl_mask_roi.shape
        line_mask = np.zeros((h_r, w_r), dtype=np.uint8)
        cv2.line(line_mask, p1, p2, 255, 1)
        
        # Get points on the line
        # We can iterate or just use bitwise
        # For precise "shortest line", we want the segment of p1-p2 that falls between CEJ and Bone.
        
        # Find intersection pixels
        cej_inter = cv2.bitwise_and(line_mask, cej_mask_roi)
        # bone_inter = cv2.bitwise_and(line_mask, bl_mask_roi) # This is the bone support
        
        # P1 is usually Crown/CEJ side? Or Apex side? 
        # get_principal_axis usually returns directions.
        # Let's assume P1 is top, P2 is bottom (or vice versa).
        # We need to find the specific gap.
        
        # Find contours of intersection is overkill for line. Use np.where
        y_cej, x_cej = np.where(cej_inter > 0)
        
        # Logic: 
        # 1. Find point on line closest to CEJ (center of CEJ intersection) -> P_C
        # 2. Find point on line where Bone Starts (closest to CEJ) -> P_B
        
        if len(x_cej) == 0:
            # Fallback to length method if no intersection
            L_bl = get_line_length(cv2.bitwise_and(line_mask, bl_mask_roi))
            L_cej = get_line_length(cej_inter)
            pbl_ratio = (axis_len - L_bl) / (axis_len - L_cej) if (axis_len - L_cej) > 0 else 0
            
            return {
                "pbl_percent": float(pbl_ratio * 100),
                "pts": None # No drawing
            }
            
        # P_C: Average of CEJ intersection points
        pc_x, pc_y = int(np.mean(x_cej)), int(np.mean(y_cej))
        
        # Find Bone intersection
        bone_inter = cv2.bitwise_and(line_mask, bl_mask_roi)
        y_bone, x_bone = np.where(bone_inter > 0)
        
        if len(x_bone) == 0:
            # 100% loss?
            return { "pbl_percent": 100.0, "pts": None }
            
        # P_B: The bone pixel closest to P_C
        # Calculate distances from P_C to all bone pixels
        dists = (x_bone - pc_x)**2 + (y_bone - pc_y)**2
        min_idx = np.argmin(dists)
        pb_x, pb_y = x_bone[min_idx], y_bone[min_idx]
        
        # Measure Euclidean distance
        dist_loss = math.sqrt((pc_x - pb_x)**2 + (pc_y - pb_y)**2)
        
        # Total Root Length (estimate from P_C to Apex?)
        # Axis p1-p2 covers the whole tooth usually.
        # We can estimate root length as dist(P_C, P_Apex).
        # Which point is apex? The one furthest from P_C.
        d1 = (p1[0]-pc_x)**2 + (p1[1]-pc_y)**2
        d2 = (p2[0]-pc_x)**2 + (p2[1]-pc_y)**2
        p_apex = p2 if d2 > d1 else p1
        root_len = math.sqrt((pc_x - p_apex[0])**2 + (pc_y - p_apex[1])**2)
        
        if root_len < 1.0: root_len = 1.0
        ratio = dist_loss / root_len
        
        return {
            "pbl_percent": float(min(ratio * 100, 100.0)),
            "pts": ((pc_x, pc_y), (pb_x, pb_y))
        }

    for idx, coords in enumerate(seg_res.masks.xy):
        try:
            keyval = seg_res.names[int(seg_res.boxes.cls[idx])]
            if len(keyval) > 1 and keyval[1] == "8":
                continue
            # skip prosthetic/disease classes for PBL (implant/bridge/crown)
            key_lower = str(keyval).lower()
            if any(skip in key_lower for skip in ("implant", "bridge", "crown")):
                continue

            res2 = get_squeeze_points(coords, copy.deepcopy(img), 1)
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, res2, (255, 255, 255))
            cnt = res2[0]
            x, y, w, h = cv2.boundingRect(cnt)
            if len(keyval) > 1 and keyval[1] == "8" and h <= w:
                continue

            # ROI with adaptive left margin (proportional to width)
            left_th = -int(max(5, min(0.05 * w, 15)))
            roi_tooth = mask[y:y + h, x + left_th:x + w]
            roi_bone = bonelevel_mask[y:y + h, x + left_th:x + w]
            roi_cej = cej_mask[y:y + h, x + left_th:x + w]

            # quality gate: too few pixels -> skip
            if cv2.countNonZero(cv2.cvtColor(roi_tooth, cv2.COLOR_BGR2GRAY)) < 30:
                continue

            # candidate axes
            pt_pbl = get_principal_axis(roi_tooth, keyval)
            n_samples = int(min(40, max(10, h // 2)))
            pt_pca = pca_axis_from_mask(
                cv2.cvtColor(roi_tooth, cv2.COLOR_BGR2GRAY) if len(roi_tooth.shape) == 3 else roi_tooth,
                n_samples=n_samples,
            )

            met_pbl = _compute_metrics(pt_pbl, roi_bone, roi_cej)
            met_pca = _compute_metrics(pt_pca, roi_bone, roi_cej) if pt_pca else None

            chosen = pt_pbl
            chosen_met = met_pbl
            # Use same selection logic for metric comparison? 
            # Original used L_cej (length of intersection). New uses pts.
            # Simplified: prefer metric if available.
            if met_pca and met_pbl:
                 # Prefer PCA if angle is good.
                 # Just use PBL for now to match user request of "CEJ-Bone line".
                 pass
            elif met_pca and not met_pbl:
                 chosen, chosen_met = pt_pca, met_pca

            # Determine Level
            pct = chosen_met["pbl_percent"]
            lvl = 1
            if pct < 15: lvl = 1
            elif pct < 33: lvl = 2
            elif pct < 66: lvl = 3
            else: lvl = 4

            # store data
            global_cej = None
            if chosen_met.get("pts"):
                lc_x, lc_y = chosen_met["pts"][0]
                global_cej = (int(x + left_th + lc_x), int(y + lc_y))
                
            bonelevel_dict[keyval] = {
                "ratio": pct / 100.0,
                "percent": pct,
                "level": lvl,
                "cej_pt": global_cej
            }

            # Draw Visualization
            # Draw the connection line (Yellow/Red)
            if chosen_met.get("pts"):
                 p_c, p_b = chosen_met["pts"]
                 # Draw Line
                 cv2.line(overlay[y:y + h, x + left_th:x + w], p_c, p_b, (0, 255, 255), 2)
                 # Draw Ends
                 cv2.circle(overlay[y:y + h, x + left_th:x + w], p_c, 2, (0, 0, 255), -1) # CEJ point
                 cv2.circle(overlay[y:y + h, x + left_th:x + w], p_b, 2, (255, 0, 0), -1) # Bone point
            else:
                 # Fallback: draw axis
                 cv2.line(overlay[y:y + h, x + left_th:x + w], chosen[0], chosen[1], (0, 0, 255), 1)

        except Exception:
            continue

    cej_count = len(cej_result[0].masks.xy)
    bl_count = len(bonelv_result[0].masks.xy)
    return overlay, bonelevel_dict, cej_count, bl_count


def get_bonelevel_info(img, seg_res):
    """
    Compute PBL% per tooth using bonelevel.pt and cej.pt (ported from demo/temp_utils.py).
    """
    if not BONELEVEL or not CEJ:
        return {}
    bonelv = YOLO(BONELEVEL)
    bonelv_result = bonelv(img)
    if len(bonelv_result) == 0 or bonelv_result[0].masks is None:
        return {}
    cej_model = YOLO(CEJ)
    cej_result = cej_model(img)
    if len(cej_result) == 0 or cej_result[0].masks is None:
        return {}

    left_th = -10
    original_img = img.copy()
    bonelevel_dict = {}

    bonelevel_mask = np.zeros_like(original_img)
    if len(bonelv_result[0].masks.xy) > 0:
        bonelevel_mask_cnt = get_squeeze_points(bonelv_result[0].masks.xy[0], copy.deepcopy(original_img), 0, cv2.CHAIN_APPROX_NONE)
        if bonelevel_mask_cnt:
            cv2.fillPoly(bonelevel_mask, bonelevel_mask_cnt, (255, 255, 255))
    cej_mask = np.zeros_like(original_img)
    try:
        for t in cej_result[0].masks.xy:
            cej_mask_cnt = get_squeeze_points(t, copy.deepcopy(original_img), 0, cv2.CHAIN_APPROX_NONE)
            if cej_mask_cnt:
                cv2.fillPoly(cej_mask, cej_mask_cnt, (255, 255, 255))
                _ = bonelevel_postprocessing(copy.deepcopy(original_img), cej_mask_cnt)
    except Exception:
        return {}

    for idx, coords in enumerate(seg_res.masks.xy if seg_res.masks is not None else []):
        try:
            keyval = seg_res.names[int(seg_res.boxes.cls[idx])]
            if len(keyval) > 1 and keyval[1] == "8":
                continue
            key_lower = str(keyval).lower()
            if any(skip in key_lower for skip in ("implant", "bridge", "crown")):
                continue
            res2 = get_squeeze_points(coords, copy.deepcopy(original_img), 1)
            if not res2:
                continue
                
            mask = np.zeros_like(original_img)
            cv2.fillPoly(mask, res2, (255, 255, 255))
            cnt = res2[0]
            x, y, w, h = cv2.boundingRect(cnt)
            if len(keyval) > 1 and keyval[1] == "8" and h <= w:
                continue
            pt1, pt2 = get_principal_axis(mask[y:y + h, x + left_th:x + w], keyval)
            teeth_len = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            tooth_length_mask = np.zeros_like(original_img)
            cv2.line(tooth_length_mask[y:y + h, x + left_th:x + w], pt1, pt2, (255, 255, 255), 1)
            periodontal_to_root = cv2.bitwise_and(tooth_length_mask, bonelevel_mask)
            periodontal_to_root_val = teeth_len - get_line_length(periodontal_to_root[y:y + h, x + left_th:x + w])
            tooth_length_mask = np.zeros_like(original_img)
            cv2.line(tooth_length_mask[y:y + h, x + left_th:x + w], pt1, pt2, (255, 255, 255), 1)
            cej_to_root = cv2.bitwise_and(tooth_length_mask, cej_mask)
            cej_to_root_val = teeth_len - get_line_length(cej_to_root[y:y + h, x + left_th:x + w])
            if cej_to_root_val == 0:
                continue
            pbl = periodontal_to_root_val / cej_to_root_val
            bonelevel_dict[keyval] = pbl * 100
        except Exception:
            continue
    return bonelevel_dict


def calc_implant_metrics(contour, mm_per_px=0.1):
    """
    Calculate implant metrics (diameter, length, axis points) using MinAreaRect.
    Using MinAreaRect ensures we get the 'oriented' bounding box, respecting the implant's angle.
    
    Returns:
        diameter_mm (float)
        length_mm (float)
        (p1, p2) (tuple): Top and bottom center points of the implant axis (for visualization)
    """
    if contour is None or len(contour) == 0:
        return 0.0, 0.0, (None, None)

    # 1. Get Oriented Bounding Box
    # rect: ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    # 2. Determine Length vs Diameter
    # Implicit assumption: Implant is longer than it is wide.
    length_px = max(w, h)
    diameter_px = min(w, h)

    # 3. Calculate Axis Points (Top and Bottom centers)
    # Convert angle to radians
    # minAreaRect angle is in degrees.
    # We need to find the direction vector of the LONG side.
    
    # Box points
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Sort points to find the long side
    # minAreaRect points are not guaranteed order, but usually sequential.
    # Let's use the center and angle to project.
    
    # If h > w, angle corresponds to the side 'h' being 'vertical-ish' relative to the rotated frame?
    # Actually, it's easier to just compute the unit vector from the angle.
    # However, opencv angle definition varies by version. 
    # Reliability fallback: Use the box points.
    
    def dist(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    # Find the two points that make the long side
    # box has 4 points. 0-1, 1-2, 2-3, 3-0 are edges.
    d01 = dist(box[0], box[1])
    d12 = dist(box[1], box[2])
    
    if d01 > d12:
        # 0-1 and 2-3 are long sides. 
        # Midpoint of 0-1
        m1 = ((box[0][0]+box[1][0])/2, (box[0][1]+box[1][1])/2)
        # Midpoint of 2-3
        m2 = ((box[2][0]+box[3][0])/2, (box[2][1]+box[3][1])/2)
    else:
        # 1-2 and 3-0 are long sides.
        m1 = ((box[1][0]+box[2][0])/2, (box[1][1]+box[2][1])/2)
        m2 = ((box[3][0]+box[0][0])/2, (box[3][1]+box[0][1])/2)

    # 4. Convert to mm
    diameter_mm = diameter_px * mm_per_px
    length_mm = length_px * mm_per_px
    
    return diameter_mm, length_mm, (m1, m2)

import copy
import os
import time
import uuid
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import json
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# --- Compatibility Patch for Custom Model 'Segment26' & 'Proto26' ---
try:
    import ultralytics.nn.modules.head
    import ultralytics.nn.modules.block
    
    if not hasattr(ultralytics.nn.modules.head, 'Segment26'):
        # Alias Segment26 to the standard Segment class
        ultralytics.nn.modules.head.Segment26 = ultralytics.nn.modules.head.Segment
        
    if not hasattr(ultralytics.nn.modules.block, 'Proto26'):
        # Alias Proto26 to the standard Proto class
        ultralytics.nn.modules.block.Proto26 = ultralytics.nn.modules.block.Proto

except ImportError:
    pass
# ------------------------------------------------------


from services import pano_calc_utils as calc
from utils.sample_axis_service import compute_from_gray_with_mask, compute_sample_axis


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_dicom_meta(path: Path) -> Dict[str, Any]:
    """Safely extract a few useful DICOM fields (best effort)."""
    try:
        import pydicom
    except ImportError:
        return {}
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    except Exception:
        return {}

    def get_val(tag):
        try:
            v = ds.get(tag, None)
            if v is None:
                return None
            if hasattr(v, "value"):
                v = v.value
            # Convert multi-valued elements to list
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                return list(v)
            return str(v)
        except Exception:
            return None

    meta = {
        "PatientName": get_val("PatientName"),
        "PatientID": get_val("PatientID"),
        "PatientSex": get_val("PatientSex"),
        "PatientBirthDate": get_val("PatientBirthDate"),
        "StudyDate": get_val("StudyDate"),
        "StudyDescription": get_val("StudyDescription"),
        "SeriesDescription": get_val("SeriesDescription"),
        "Modality": get_val("Modality"),
        "PixelSpacing": get_val("PixelSpacing"),
        "ImageOrientationPatient": get_val("ImageOrientationPatient"),
        "ImagePositionPatient": get_val("ImagePositionPatient"),
        "SliceThickness": get_val("SliceThickness"),
        "Rows": get_val("Rows"),
        "Columns": get_val("Columns"),
        "Manufacturer": get_val("Manufacturer"),
    }
    # Drop empty entries
    return {k: v for k, v in meta.items() if v not in (None, "", [])}


def load_image_any(path: Path) -> np.ndarray:
    """
    Load common image formats or DICOM (if pydicom available).
    Returns BGR uint8 image, raises if cannot be read.
    """
    # Safe Image Load (Support Windows/Unicode paths)
    try:
        abs_path = os.path.abspath(str(path))
        stream = open(abs_path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        if img is not None:
             return img
    except Exception:
        pass

    if path.suffix.lower() in (".dcm", ".dicom"):
        try:
            import pydicom
        except ImportError:
            raise ValueError(f"Failed to load image: {path} (pydicom not installed for DICOM)")
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc is not None and ww is not None:
            try:
                if hasattr(wc, "__len__"):
                    wc = float(wc[0])
                else:
                    wc = float(wc)
                if hasattr(ww, "__len__"):
                    ww = float(ww[0])
                else:
                    ww = float(ww)
                min_w, max_w = wc - ww / 2.0, wc + ww / 2.0
                arr = np.clip(arr, min_w, max_w)
            except Exception:
                pass
        arr = arr - arr.min()
        maxv = arr.max()
        if maxv > 0:
            arr = arr / maxv * 255.0
        img = arr.astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    raise ValueError(f"Failed to load image: {path}")


class PanoPipeline:
    """
    Multi-model pipeline: tooth seg + caries + periapical + CEJ + Bonelevel
    Returns overlay PNG and rich detection data (tooth labels, PBL%, etc.).
    """

    def __init__(self, model_dir: Path, model_cfg: Dict[str, Dict[str, Any]], device: str = "cpu"):
        self.model_dir = model_dir
        self.model_cfg = model_cfg
        self.device = device
        self.models: Dict[str, YOLO] = {}
        # Preload all models to memory at startup
        self.preload_models()

    def preload_models(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        dev_name = torch.cuda.get_device_name(0) if self.device == "cuda" and torch.cuda.is_available() else "CPU"
        print(f"  [INIT] Preloading AI models onto {self.device.upper()} ({dev_name})...")
        for name in self.model_cfg.keys():
            try:
                self._load_model(name)
            except Exception as e:
                print(f"  [WARN] Failed to preload {name}: {e}")

    def _load_model(self, name: str) -> YOLO:
        if name in self.models:
            return self.models[name]
        info = self.model_cfg[name]
        path = Path(info["path"])
        if not path.is_absolute():
            path = self.model_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Model weight not found: {path}")
        model = YOLO(str(path))
        model.to(self.device)
        self.models[name] = model
        return model

    def _bbox_center_inside(self, bbox, inner_rect):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return inner_rect["x1"] <= cx <= inner_rect["x2"] and inner_rect["y1"] <= cy <= inner_rect["y2"]

    def _overlay_seg(self, base: np.ndarray, result, color_map: Dict[int, Tuple[int, int, int]], inner_rect: Dict[str, int], thickness=1):
        """
        Draw only contours for segmentation masks to avoid brightening the base image.
        Control line thickness via 'thickness' param.
        """
        if result.masks is None:
            return base
        img = base.copy()
        
        # 2025-01-27: Filter small noise masks
        min_area = 200.0
        
        for idx, m in enumerate(result.masks.xy):
            if len(m) == 0: continue
            pts = np.int32([m])
            
            # Check Area
            area = cv2.contourArea(pts)
            if area < min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(pts)
            if not self._bbox_center_inside((x, y, x + w, y + h), inner_rect):
                continue
            
            # Get color from map (allows different colors for different models)
            color = color_map.get(idx, (0, 255, 0))
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return img

    def _overlay_det(self, base: np.ndarray, result, names, color: Tuple[int, int, int], inner_rect: Dict[str, int], exclusion_boxes: List[List[float]] = None) -> Tuple[np.ndarray, List[Tuple[List[int], float, str]]]:
        img = base.copy()
        det_list = []
        if result.boxes is None:
            return img, det_list
        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Exclusion Check (e.g. Disease on Implant)
            if exclusion_boxes:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                is_excluded = False
                for eb in exclusion_boxes:
                    # Check if center of disease is inside exclusion box (Implant)
                    if eb[0] <= cx <= eb[2] and eb[1] <= cy <= eb[3]:
                        is_excluded = True
                        break
                if is_excluded:
                    continue

            if not self._bbox_center_inside((x1, y1, x2, y2), inner_rect):
                continue
            label = names[int(cls)]
            # Draw dashed rectangle (no solid line, no text)
            def draw_dashed_rect(img, p1, p2, color, thickness=1, gap=5):
                x1, y1 = p1
                x2, y2 = p2
                # Top
                for x in range(x1, x2, gap * 2):
                    cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
                # Bottom
                for x in range(x1, x2, gap * 2):
                    cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
                # Left
                for y in range(y1, y2, gap * 2):
                    cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
                # Right
                for y in range(y1, y2, gap * 2):
                    cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)
            
            draw_dashed_rect(img, (x1, y1), (x2, y2), color, 1, 5)
            # Text label removed per user request
            det_list.append(([int(x1), int(y1), int(x2), int(y2)], float(conf), label))
        return img, det_list

    def _get_calibration_scale(self, seg_res, tooth_objects):
        """
        Calculate mm per pixel using Molar heights (Default 22.5mm).
        Molars: 16, 17, 26, 27, 36, 37, 46, 47
        """
        molar_fdis = [16, 17, 26, 27, 36, 37, 46, 47]
        max_height = 0.0
        
        if seg_res.masks is None: return 0.05 # Fallback ~0.05 mm/px (assuming 500px ~ 25mm)

        for tooth in tooth_objects:
            lbl = tooth.get('tooth_label')
            if not lbl or not lbl.isdigit(): continue
            if int(lbl) not in molar_fdis: continue
            
            # Find corresponding mask index
            # tooth['id'] should correspond to seg_res index if derived from there
            # But currently tooth objects are re-created globally.
            # We need to rely on the box to find the mask, or better, store mask index in tooth object.
            # In _assign_objects, object_id is preserved.
            # Let's assume tooth['id'] is reliable index if it's an integer.
            
            try:
                idx = int(tooth.get('id'))
                if idx < len(seg_res.masks.xy):
                    # Height of the mask
                    # masks.xy is list of points.
                    pts = seg_res.masks.xy[idx]
                    if len(pts) > 0:
                        ys = pts[:, 1]
                        h = np.max(ys) - np.min(ys)
                        if h > max_height: max_height = h
            except: pass
            
        if max_height > 0:
            return 22.5 / max_height
        return 0.05 # Fallback

    def _measure_shortest_distance(self, p_start, target_mask, scale):
        """
        Measure SHORTEST Euclidean distance from p_start to Target Mask (Nerve/Sinus)
        p_start: [x, y]
        target_mask: Binary mask
        """
        # Find all non-zero points in target mask
        # Optimization: ROI around p_start?
        # Or just findNonZero (might be slow if mask is huge, but typically okay for 1000x500)
        
        # To speed up, we can use Distance Transform on the INVERSE of the mask?
        # dist_map = cv2.distanceTransform(cv2.bitwise_not(target_mask), cv2.DIST_L2, 5)
        # val = dist_map[int(p_start[1]), int(p_start[0])]
        # But we need the endpoint coordinates for visualization.
        
        y, x = np.where(target_mask > 0)
        if len(y) == 0: return None, None
        
        # Points: (N, 2)
        pts = np.column_stack((x, y))
        
        # Calculate distances
        # p_start is (x, y)
        dists = np.linalg.norm(pts - np.array(p_start), axis=1)
        
        min_idx = np.argmin(dists)
        min_dist_px = dists[min_idx]
        target_pt = pts[min_idx] # [x, y]
        
        dist_mm = (min_dist_px * scale) - 2.0 # Safety Margin
        if dist_mm < 0: dist_mm = 0.0
        
        return dist_mm, [int(p_start[0]), int(p_start[1]), int(target_pt[0]), int(target_pt[1])]

    def _check_overlap(self, tooth_mask_pts, target_mask, shape):
        """Check if tooth polygon overlaps with target mask."""
        if len(tooth_mask_pts) == 0: return False
        
        # Draw tooth mask
        t_mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(t_mask, [np.array(tooth_mask_pts, dtype=np.int32)], 255)
        
        # Overlap
        # target_mask should be 0/1 or 0/255
        overlap = cv2.bitwise_and(t_mask, target_mask)
        return cv2.countNonZero(overlap) # Return pixel count

    def run(self, image_path: Path, out_dir: Path) -> Dict[str, Any]:
        _ensure_dir(out_dir)

        # If DICOM, save basic metadata for reference (best-effort)
        if image_path.suffix.lower() in (".dcm", ".dicom"):
            meta = extract_dicom_meta(image_path)
            if meta:
                try:
                    with open(out_dir / "dicom_meta.json", "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        img = load_image_any(image_path)
        base = img.copy()
        h, w = base.shape[:2]
        # 2026-01-27: Changed margin to 0.0 to show edge implants
        margin = 0.0
        inner_rect = {
            "x1": int(w * margin),
            "x2": int(w * (1 - margin)),
            "y1": int(h * margin),
            "y2": int(h * (1 - margin)),
        }

        # Load models
        seg_model = self._load_model("pano_seg")
        caries_model = self._load_model("caries")
        peri_model = self._load_model("periapical")
        iac_model = self._load_model("iac")
        self._load_model("cej")  # ensure weights exist; actual loading in calc.get_bonelevel
        self._load_model("bonelevel")

        # 1) Segmentation
        seg_cfg = self.model_cfg["pano_seg"]
        t0 = time.perf_counter()
        seg_res = seg_model.predict(
            source=img,
            imgsz=seg_cfg.get("default_imgsz", 1024),
            conf=seg_cfg.get("default_confidence", 0.25),
            iou=seg_cfg.get("default_iou", 0.7),
            retina_masks=True,
            verbose=False,
        )[0]
        print(f"  [PERF][{self.device.upper()}] pano_seg inference: {time.perf_counter()-t0:.3f}s")

        # --- HYBRID INFERENCE START ---
        # If 'pano_seg_aux' is loaded (YOLO11), use it to get correct labels (11~48)
        # because the main 'pano_seg' (YOLO26 Root) might only return class '0'.
        aux_boxes_data = []
        if "pano_seg_aux" in self.model_cfg:
            try:
                # Ensure model is loaded
                aux_model = self._load_model("pano_seg_aux")
                aux_cfg = self.model_cfg["pano_seg_aux"]
                
                t_aux = time.perf_counter()
                aux_res = aux_model.predict(
                    source=img,
                    imgsz=aux_cfg.get("default_imgsz", 1024),
                    conf=aux_cfg.get("default_confidence", 0.25),
                    iou=aux_cfg.get("default_iou", 0.7),
                    verbose=False
                )[0]
                
                if aux_res.boxes is not None:
                    aux_boxes = aux_res.boxes.xyxy.cpu().numpy()
                    aux_clss = aux_res.boxes.cls.cpu().numpy()
                    aux_names = aux_res.names
                    for i, box in enumerate(aux_boxes):
                        l = aux_names[int(aux_clss[i])]
                        aux_boxes_data.append((box, str(l)))
                
                print(f"  [PERF][{self.device.upper()}] pano_seg_aux (Labeling): {time.perf_counter()-t_aux:.3f}s")
            except Exception as e:
                print(f"  [WARN] Failed to run pano_seg_aux: {e}")

        # Helper to find best label from aux
        def _get_hybrid_label(target_box):
            if not aux_boxes_data: 
                return None
            
            x1, y1, x2, y2 = target_box
            best_iou = 0.0
            best_lbl = None
            
            area_target = (x2-x1)*(y2-y1)
            
            for (abox, albl) in aux_boxes_data:
                ax1, ay1, ax2, ay2 = abox
                
                xx1 = max(x1, ax1)
                yy1 = max(y1, ay1)
                xx2 = min(x2, ax2)
                yy2 = min(y2, ay2)
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                
                if inter > 0:
                    area_aux = (ax2-ax1)*(ay2-ay1)
                    union = area_target + area_aux - inter
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_lbl = albl
            
            # Threshold for matching
            if best_iou > 0.3:
                return best_lbl
            return None
        # --- HYBRID INFERENCE END ---

        # 1-1) 숫자 치아 세그먼트 미리 추출 (모든 매핑 및 공간 검증의 기준)
        tooth_only_boxes = []
        if seg_res.boxes is not None and seg_res.masks is not None:
            for idx, coords in enumerate(seg_res.masks.xy):
                label = seg_res.names[int(seg_res.boxes.cls[idx])]
                
                # Hybrid Override REMOVED (Native 26L has correct labels)
                if str(label).isdigit():
                    cnts = calc.get_squeeze_points(coords, copy.deepcopy(seg_res.orig_img), 0)
                    if cnts:
                        bb = calc._get_bb_from_contour(cnts[0])
                        tooth_only_boxes.append((label, bb))

        # 2) Caries detection
        car_cfg = self.model_cfg["caries"]
        t1 = time.perf_counter()
        # For CPU speed, reducing imgsz from 1024 to 640 is highly recommended.
        car_res = caries_model.predict(
            source=img,
            imgsz=car_cfg.get("default_imgsz", 640),
            conf=car_cfg.get("default_confidence", 0.2),
            iou=car_cfg.get("default_iou", 0.5),
            verbose=False,
        )[0]
        print(f"  [PERF][{self.device.upper()}] caries inference: {time.perf_counter()-t1:.3f}s")

        # 3) Periapical detection
        per_cfg = self.model_cfg["periapical"]
        t2 = time.perf_counter()
        per_res = peri_model.predict(
            source=img,
            imgsz=per_cfg.get("default_imgsz", 640),
            conf=per_cfg.get("default_confidence", 0.1),
            iou=per_cfg.get("default_iou", 0.5),
            verbose=False,
        )[0]
        print(f"  [PERF][{self.device.upper()}] periapical inference: {time.perf_counter()-t2:.3f}s")
        
        # 4) IAC (Inferior Alveolar Canal) - Multi-Preprocessing Ensemble
        iac_cfg = self.model_cfg["iac"]
        t_iac = time.perf_counter()
        
        # Candidate Preprocessing Methods
        # 1. Original
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # 3. Gamma Correction (Darker/Higher Contrast)
        
        best_iac_res = None
        best_score = -1.0
        
        # Prepare variants
        variants = [("original", img)]
        
        # CLAHE
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl_img = clahe.apply(gray)
            cl_img = cv2.cvtColor(cl_img, cv2.COLOR_GRAY2BGR)
            variants.append(("clahe", cl_img))
        except: pass
        
        # Gamma
        try:
            gamma = 1.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gam_img = cv2.LUT(img, table)
            variants.append(("gamma", gam_img))
        except: pass
        
        for name, v_img in variants:
            res = iac_model.predict(
                source=v_img,
                imgsz=iac_cfg.get("default_imgsz", 1024),
                conf=0.05, # Lowered from 0.15 to catch faint nerves
                iou=iac_cfg.get("default_iou", 0.5),
                retina_masks=True, # Improved mask resolution
                verbose=False,
            )[0]
            
            # Score metric: Max confidence of any detection
            # If no detection, score is 0
            if res.boxes is None or len(res.boxes) == 0:
                current_score = 0
            else:
                current_score = float(res.boxes.conf.max())
            
            print(f"  [IAC] {name} score: {current_score:.4f}")
            
            if current_score > best_score:
                best_score = current_score
                best_iac_res = res
        
        if best_iac_res is not None:
            iac_res = best_iac_res
        else:
             # Fallback to standard inference if something went wrong
             iac_res = iac_model.predict(img, verbose=False)[0]

        print(f"  [PERF][{self.device.upper()}] iac inference (ensemble): {time.perf_counter()-t_iac:.3f}s")

        # 0) 유효 치아 구역(Dental Zone) 정교화 계산
        tooth_x_centers = []
        tooth_y_coords = []
        if tooth_only_boxes:
            for _, bb in tooth_only_boxes:
                tooth_x_centers.append((bb[0] + bb[2]) / 2)
                tooth_y_coords.extend([float(bb[1]), float(bb[3])])
        
        # 치아들의 가로 중심점 중앙값 기준으로 유효 범위 설정 (이상치 제거 효과)
        if tooth_x_centers:
            med_x = np.median(tooth_x_centers)
            std_x = np.std(tooth_x_centers) if len(tooth_x_centers) > 1 else img.shape[1] * 0.2
            # 중앙값에서 일정 거리 이상 떨어진 치아는 무시 (턱 가장자리 artifacts 방지)
            # 2026-01-27: Relaxed from 2.5 to 5.0 to detect far-side implants (outer implants)
            valid_x = [x for x in tooth_x_centers if abs(x - med_x) < 5.0 * std_x]
            # Ensure at least 5% margin to 95% margin coverage
            min_valid = min(valid_x) if valid_x else img.shape[1] * 0.1
            max_valid = max(valid_x) if valid_x else img.shape[1] * 0.9
            
            bl_x1 = min(min_valid - img.shape[1] * 0.05, img.shape[1] * 0.02)
            bl_x2 = max(max_valid + img.shape[1] * 0.05, img.shape[1] * 0.98)
        else:
            bl_x1, bl_x2 = img.shape[1] * 0.1, img.shape[1] * 0.9

        # 공간적 유효성 검사 함수 (객체 타입별로 정밀화)
        def _is_spatial_valid(box, label_str=""):
            # 2026-01-27: USER REQUEST - Temporarily disable filter
            return True

            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            lbl = label_str.lower()
            
            # 1) 가로 범위: 유효 중심 구역에서 너무 먼 측면(턱 관절 등) 차단
            if cx < bl_x1 or cx > bl_x2: return False

            # 2) 세로 범위: 객체 종류에 따른 가변 임계값
            if tooth_y_coords:
                ty1, ty2 = min(tooth_y_coords), max(tooth_y_coords)
                
                # 임플란트/픽스처는 뼈 깊숙이 있으므로 깊은 곳까지 허용 (25% 마진)
                if "implant" in lbl or "fixture" in lbl:
                    margin = img.shape[0] * 0.25
                # 일반 보철물/충치는 치아 영역에 밀착되어야 함 (10% 마진)
                else:
                    margin = img.shape[0] * 0.10
                
                if cy < ty1 - margin or cy > ty2 + margin: return False
            
            # 3) 밀착도 검사: 보철물/충치인데 주변에 숫자 치아 세그먼트가 너무 없으면 허상으로 간주
            if any(k in lbl for k in ["crown", "bridge", "caries", "filling"]):
                near_tooth = False
                for _, tbb in tooth_only_boxes:
                    # 가로축으로 인접해 있는지 확인
                    if abs(cx - (tbb[0]+tbb[2])/2) < (tbb[2]-tbb[0]) * 1.5:
                        near_tooth = True
                        break
                if not near_tooth: return False

            return True

        def _infer_teeth_by_pos(box, img_w, img_h):
            """박스 위치 및 너비를 기반으로 치아 번호 리스트 반환 (브릿지 대응)"""
            bx1, by1, bx2, by2 = box
            is_upper = ((by1 + by2) / 2) < (img_h / 2)
            center_x = (bl_x1 + bl_x2) / 2
            # 한 쪽 사분면의 폭을 8.5 정도로 나누어 사랑니(8번) 공간 확보
            seg_w = ((bl_x2 - bl_x1) / 2) / 8.2
            
            matched = set()
            center_bx = (bx1 + bx2) / 2
            samples = [bx1, center_bx, bx2] if (bx2 - bx1) > seg_w else [center_bx]
            
            for sx in samples:
                safe_x = max(bl_x1, min(bl_x2, sx))
                is_right = safe_x < center_x
                if is_right:
                    dist_from_edge = max(0, safe_x - bl_x1)
                    idx = min(int(dist_from_edge / (seg_w + 1e-6)), 7)
                    t_num = 8 - idx
                else:
                    dist_from_center = max(0, safe_x - center_x)
                    idx = min(int(dist_from_center / (seg_w + 1e-6)), 7)
                    t_num = idx + 1
                q = 1 if is_upper and is_right else 2 if is_upper else 3 if not is_right else 4
                matched.add(f"{q}{t_num}")
            return list(matched)

        def _get_vertical_ref_teeth(box, tooth_only_boxes, target_is_upper):
            """수직(교합) 방향의 치아를 참고하여 번호 리스트 반환"""
            bx1, _, bx2, _ = box
            matched = set()
            for label, tbb in tooth_only_boxes:
                tx1, _, tx2, _ = tbb
                q = int(str(label)[0])
                if (q in (1, 2)) == target_is_upper: continue
                overlap = min(bx2, tx2) - max(bx1, tx1)
                if overlap > 0 and (overlap / (tx2 - tx1 + 1e-6)) >= 0.3:
                    new_q = 1 if q == 4 else 4 if q == 1 else 2 if q == 3 else 3
                    matched.add(f"{new_q}{str(label)[1]}")
            return list(matched)
        
        # --- 치열 영역 기반 이상치 필터링 (ROI) ---
        # 자연치들의 X, Y좌표 분포를 구해, 그 범위를 심하게 벗어나는(상악동, 하악골, 턱관절 등) 허상 제거
        min_y_arch, max_y_arch = 0, img.shape[0]
        min_x_arch, max_x_arch = 0, img.shape[1]
        
        if tooth_only_boxes:
            ys = []
            xs = []
            for _, tb in tooth_only_boxes:
                ys.append(tb[1]); ys.append(tb[3])
                xs.append(tb[0]); xs.append(tb[2])
            
            if ys and xs:
                # Vertical Margin (Tightened to 20%)
                min_y, max_y = min(ys), max(ys)
                margin_y = (max_y - min_y) * 0.2
                min_y_arch = max(0, min_y - margin_y)
                max_y_arch = min(img.shape[0], max_y + margin_y)
                
                # Horizontal Margin (15% is usually enough to cover wisdom teeth area but exclude ramus)
                min_x, max_x = min(xs), max(xs)
                margin_x = (max_x - min_x) * 0.15
                min_x_arch = max(0, min_x - margin_x)
                max_x_arch = min(img.shape[1], max_x + margin_x)


        def _infer_single_tooth_hybrid(box, target_is_upper):
            """Hybrid: IoU Match -> Contextual Neighbor Interpolation"""
            bx1, by1, bx2, by2 = box
            det_x = (bx1 + bx2) / 2
            
            # 1. IoU Best Match
            best_t, best_iou = None, 0
            for lt, tbb in tooth_only_boxes:
                q = int(str(lt)[0])
                if (q in (1, 2)) != target_is_upper: continue
                iou = calc._calc_iou(box, tbb)
                if iou > best_iou: best_iou, best_t = iou, lt
            
            if best_iou >= 0.35: return str(best_t)
            
            # 2. Vertical Ref
            v_refs = _get_vertical_ref_teeth(box, tooth_only_boxes, target_is_upper)
            if v_refs: return v_refs[0]
            
            # 3. Neighborhood Interpolation
            jaw_teeth = []
            for lt, tbb in tooth_only_boxes:
                q = int(str(lt)[0])
                if (q in (1, 2)) == target_is_upper:
                    jaw_teeth.append((int(lt), (tbb[0]+tbb[2])/2))
            
            if not jaw_teeth:
                inf = _infer_teeth_by_pos(box, img.shape[1], img.shape[0])
                return inf[0] if inf else None

            jaw_teeth.sort(key=lambda x: x[1])
            left_n, right_n = None, None
            for t, tx in jaw_teeth:
                if tx < det_x: left_n = (t, tx)
                elif tx > det_x:
                    if right_n is None: right_n = (t, tx)
            
            def fdi_to_idx(f):
                q, n = f // 10, f % 10
                if q in (1, 4): return 9 - n
                return 8 + n

            def idx_to_fdi(idx, is_up):
                idx = max(1, min(16, round(idx)))
                if is_up: return 10+(9-idx) if idx<=8 else 20+(idx-8)
                return 40+(9-idx) if idx<=8 else 30+(idx-8)

            if left_n and right_n:
                l_idx, r_idx = fdi_to_idx(left_n[0]), fdi_to_idx(right_n[0])
                if r_idx > l_idx:
                    ratio = (det_x - left_n[1]) / (right_n[1] - left_n[1])
                    est_idx = l_idx + ratio * (r_idx - l_idx)
                    target_idx = round(est_idx)
                    
                    # Ensure we don't snap to the existing anchors (natural teeth)
                    if target_idx <= l_idx: target_idx = l_idx + 1
                    if target_idx >= r_idx: target_idx = r_idx - 1
                    # If neighbors are adjacent (e.g. 33, 34) and we are squeezed in, defaulting to l_idx+1 might overlap r_idx. 
                    # But assumes gap exists. IF valid gap, this works.
                    
                    return str(idx_to_fdi(target_idx, target_is_upper))
            
            # 4. One-sided Extrapolation (Rule 3-1: Avg width slot logic)
            neighbor = left_n or right_n
            if neighbor:
                n_fdi, n_x = neighbor
                
                # Calculate global median width of *all* detected teeth for stable slot size
                # This aligns with user request "average tooth width"
                all_widths = [(bb[2] - bb[0]) for _, bb in tooth_only_boxes]
                avg_w = np.median(all_widths) if all_widths else 40.0
                
                # Also consider specific neighbor width but weight global average higher for stability
                # Or just use global average as requested.
                # User said: "average tooth width" -> "칸을잡고" (establish grid)
                
                dist_px = abs(det_x - n_x)
                
                # Rule 3-1: " 좌우에 1칸 혹은 2칸내에 자연치가 있다면 그번호의 다음 혹은 다다음 번호로 배치한다"
                # Slot distance = distance / avg_w
                # 70% check: If distance is within 0.7~1.3 of 1 slot -> 1 slot away.
                # If distance is within 1.7~2.3 of 1 slot -> 2 slots away.
                
                raw_slots = dist_px / avg_w
                # Rule 3-1: Enforce at least 1 slot away from the reference neighbor to avoiding collision
                slots_away = max(1, round(raw_slots))
                
                # Limit to 1 or 2 if close enough, but allow more if very far (safety)
                # But strictly per user rule "1 or 2 slots", implies we should prioritize those.
                
                n_idx = fdi_to_idx(n_fdi)
                direction = 1 if (left_n and det_x > n_x) or (right_n and det_x < n_x) else -1
                
                target_idx = n_idx + (slots_away * direction)
                return str(idx_to_fdi(target_idx, target_is_upper))

            # Fallback
            inf = _infer_teeth_by_pos(box, img.shape[1], img.shape[0])
            return inf[0] if inf else None

        seg_best_indices = {}
        if seg_res.boxes is not None:
            for idx in range(len(seg_res.boxes)):
                lbl = seg_res.names[int(seg_res.boxes.cls[idx])]
                conf = float(seg_res.boxes.conf[idx])
                box = seg_res.boxes.xyxy[idx].cpu().numpy()
                
                if not _is_spatial_valid(box, lbl): continue
                
                # '11', '12' 같은 숫자 라벨만 중복 제거 (가장 확실한 것 하나만)
                if str(lbl).isdigit():
                    if lbl not in seg_best_indices or conf > seg_res.boxes.conf[seg_best_indices[lbl]]:
                        seg_best_indices[lbl] = idx
                else:
                    # 임플란트, 크라운 등은 인스턴스 전체 유지
                    seg_best_indices[f"{lbl}_{idx}"] = idx
        
        valid_indices = list(seg_best_indices.values())
        seg_res_filtered = seg_res[valid_indices] if seg_res.boxes is not None else seg_res

        # Random contour colors
        # User Request: Single fixed neon color (Cyan) for teeth overlay
        # BGR (255, 255, 0)
        fixed_neon = (255, 255, 0)
        colors = {idx: fixed_neon for idx in range(len(seg_res_filtered.boxes) if seg_res_filtered.boxes is not None else 0)}
        
        # Pass thickness=2 for thicker neon lines (User Request)
        overlay = self._overlay_seg(base, seg_res_filtered, colors, inner_rect, thickness=2)

        # Prepare Exclusion Boxes (Implants) to prevent Caries/Periapical on Implants
        implant_exclusion_boxes = []
        if seg_res_filtered.boxes is not None:
            for idx, cls in enumerate(seg_res_filtered.boxes.cls):
                lbl = str(seg_res_filtered.names[int(cls)]).lower()
                if "implant" in lbl:
                    implant_exclusion_boxes.append(seg_res_filtered.boxes.xyxy[idx].cpu().numpy().tolist())

        overlay, car_list = self._overlay_det(overlay, car_res, caries_model.model.names, (0, 0, 255), inner_rect, exclusion_boxes=implant_exclusion_boxes)
        overlay, per_list = self._overlay_det(overlay, per_res, peri_model.model.names, (0, 200, 0), inner_rect, exclusion_boxes=implant_exclusion_boxes)
        
        # Overlay IAC (Magenta for nerve canal)
        # Class 0 usually. Wemap all classes to Magenta just in case.
        iac_colors = {i: (255, 0, 255) for i in range(10)} 
        overlay = self._overlay_seg(overlay, iac_res, iac_colors, inner_rect)

        # Caries, Periapical 리스트 필터링 (그리기 위한 리스트 추출 후 적용) - Re-extracting for list processing (redundant but safe)
        _, car_list_unfiltered = self._overlay_det(base, car_res, caries_model.model.names, (0, 0, 255), inner_rect, exclusion_boxes=implant_exclusion_boxes)
        _, per_list_unfiltered = self._overlay_det(base, per_res, peri_model.model.names, (0, 200, 0), inner_rect, exclusion_boxes=implant_exclusion_boxes)

        car_list = [d for d in car_list_unfiltered if _is_spatial_valid(d[0], str(d[2]))]
        per_list = [d for d in per_list_unfiltered if _is_spatial_valid(d[0], "periapical")]

        # 4) CEJ/Bonelevel
        t3 = time.perf_counter()
        bl_canvas = np.zeros_like(img)
        pbl_dict = {}
        cej_count = 0
        bl_count = 0
        try:
            bonelevel_model = self._load_model("bonelevel")
            cej_model = self._load_model("cej")
            axis_only = os.getenv("PBL_AXIS_ONLY", "1") == "1"
            bl_canvas, pbl_dict, cej_count, bl_count = calc.get_bonelevel(
                copy.deepcopy(img), 
                seg_res,
                bonelv_model=bonelevel_model,
                cej_model=cej_model,
                axis_only=axis_only
            )
            print(f"  [PERF] PBL (Bonelevel/CEJ) calculation: {time.perf_counter()-t3:.3f}s")
        except Exception as e:
            print(f"  [WARN] PBL (Bonelevel/CEJ) calculation failed: {e}")
        if os.getenv("SKIP_PBL_OVERLAY", "0") != "1":
            if inner_rect:
                bl_canvas[:inner_rect["y1"], :] = 0
                bl_canvas[inner_rect["y2"]:, :] = 0
                bl_canvas[:, :inner_rect["x1"]] = 0
                bl_canvas[:, inner_rect["x2"]:] = 0
            overlay = cv2.add(overlay, bl_canvas)
        else:
            # Keep counts but do not draw CEJ/Bone overlays
            bl_canvas = np.zeros_like(bl_canvas)




        # 4-1) Implant metrics (Diameter & Length)
        implant_metrics = {}

        # Pre-calculate pure implant metrics BEFORE mapping for sync
        raw_implant_data = {} # box_id -> metrics
        if seg_res.boxes is not None:
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Not needed if using contour directly
            ppm = float(os.getenv("DIAMETER_PIXEL_TO_MM", 0.1))

            for i, bbox in enumerate(seg_res.boxes.xyxy):
                lbl = str(seg_res.names[int(seg_res.boxes.cls[i])]).lower()
                
                # Check spatial validity (ROI) to skip floating artifacts
                # Need to convert tensor bbox to numpy/list for valid check if function expects specific type
                # _is_spatial_valid expects [x1, y1, x2, y2]
                bbox_np = bbox.cpu().numpy()
                if not _is_spatial_valid(bbox_np, lbl): continue

                if "implant" in lbl:
                    # Use the contour from masks directly
                    # seg_res.masks.xy[i] contains the polygon points
                    contour = seg_res.masks.xy[i]
                    contour = np.array(contour, dtype=np.float32) # Ensure numpy array

                    # Use the new robust calculation
                    d_mm, l_mm, (p1, p2) = calc.calc_implant_metrics(contour, mm_per_px=ppm)
                    
                    db, lb = round(d_mm * 2.0) / 2.0, round(l_mm * 2.0) / 2.0
                    # Standardize p1, p2 to tuples if they exist
                    p1_t = tuple(p1) if p1 is not None else None
                    p2_t = tuple(p2) if p2 is not None else None

                    # Store with index key
                    raw_implant_data[f"idx_{i}"] = (d_mm, l_mm, db, lb, p1_t, p2_t)

        # Build raw objects for downstream rule engine (non-destructive)
        teeth_objects = []
        if seg_res_filtered.boxes is not None:
            for idx, box in enumerate(seg_res_filtered.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(float).tolist()
                lbl = str(seg_res_filtered.names[int(seg_res_filtered.boxes.cls[idx])])
                conf = float(seg_res_filtered.boxes.conf[idx])
                obj_type = "natural" if lbl.isdigit() else ("implant" if "implant" in lbl.lower() else "unknown")
                teeth_objects.append({
                    "id": f"tooth_{idx}",
                    "box": [x1, y1, x2, y2],
                    "type": obj_type,
                    "tooth_label": lbl if lbl.isdigit() else None,
                    "confidence": conf,
                })

        caries_objects = []
        for i, (box, conf, lbl) in enumerate(car_list):
            if not box: 
                continue
            x1, y1, x2, y2 = box
            caries_objects.append({
                "id": f"caries_{i}",
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "label": str(lbl),
                "confidence": float(conf),
            })

        # 데이터 필터링 (오도토그램 매핑용)
        car_list = [d for d in car_list if _is_spatial_valid(d[0])]
        per_list = [d for d in per_list if _is_spatial_valid(d[0])]


        # 통합된 전역 최적화 매칭 결과 사용
        
        # 1. 모든 감지된 객체 수집 (Observations)
        #    - 자연치(Numerical Label), 임플란트, 크라운, 브릿지, 충치 등
        #    Observation: { 'type': str, 'box': [x1,y1,x2,y2], 'label_hint': str/None, 'conf': float, 'orig_idx': int }
        
        observations = []
        if seg_res.boxes is not None:
            for idx in range(len(seg_res.boxes)):
                lbl = str(seg_res.names[int(seg_res.boxes.cls[idx])]).lower()
                
                # Hybrid Override REMOVED
                
                box = seg_res.boxes.xyxy[idx].cpu().numpy().tolist()
                conf = float(seg_res.boxes.conf[idx])
                
                # ROI 필터링은 여기서도 적용 (쓰레기 데이터 배제)
                # bbox_np 필요 (위에서 box는 list로 변환함)
                if not _is_spatial_valid(seg_res.boxes.xyxy[idx].cpu().numpy(), lbl):
                    continue

                obj_type = "other"
                label_hint = None
                
                if str(lbl).isdigit():
                    obj_type = "natural"
                    label_hint = lbl # 모델이 예측한 번호
                elif "implant" in lbl:
                    obj_type = "implant"
                elif any(k in lbl for k in ["crown", "bridge", "prosthesis", "pontic"]):
                    obj_type = "crown"
                # Caries, Peri 등은 나중에 위치 기반으로 매핑 (이들은 치아 자체가 아니라 병변이므로 Slot 경쟁자가 아님)
                
                if obj_type in ["natural", "implant", "crown"]:
                    # [NEW] Extract Contour for Visualization
                    contour = []
                    if seg_res.masks is not None:
                        try:
                            # idx corresponds to seg_res index
                            contour = seg_res.masks.xy[idx].tolist()
                        except: pass

                    observations.append({
                        "id": idx,
                        "type": obj_type,
                        "box": box,
                        "cx": (box[0]+box[2])/2,
                        "cy": (box[1]+box[3])/2,
                        "label_hint": label_hint,
                        "conf": conf,
                        "contour": contour # Pass contour
                    })
        
        # --- 제안 2: 수직 객체 병합 (Vertical Merging) ---
        # "크라운은 임플란트 위에 올라올 수 있어" -> Crown과 Implant가 수직으로 겹치면 하나로 합쳐야 함.
        # 따로 두면 헝가리안 알고리즘이 서로 다른 슬롯(46, 47)으로 찢어놓음 (고스트 크라운 발생 원인).
        
        merged_obs = []
        skip_indices = set()
        
        # 1. Implant 기준으로 Crown 찾기 (Implant가 더 중요)
        # N^2지만 객체 수가 적어서(32개 내외) 괜찮음
        
        obs_implants = [o for o in observations if o['type'] == 'implant']
        obs_crowns = [o for o in observations if o['type'] == 'crown']
        
        # USER REQUEST: Disable Crown/Bridge Merging Logic
        # 간단한 매칭: Implant와 가장 x좌표 가깝고, Implant보다 위에 있는 Crown
        # for imp in obs_implants:
        #     if imp['id'] in skip_indices: continue
            
        #     best_crn = None
        #     min_dist = 999
            
        #     for crn in obs_crowns:
        #         if crn['id'] in skip_indices: continue
                
        #         # 수직 정렬 확인
        #         dist_x = abs(imp['cx'] - crn['cx'])
        #         # Crown은 Implant보다 위에 있어야 함 (Y좌표가 작아야 함) -> 엄격하지 않게 비슷해도 허용
        #         is_above = crn['cy'] < imp['cy'] + (imp['box'][3]-imp['box'][1])*0.5
                
        #         # X축 겹침(Overlap) 계산
        #         x_overlap = min(imp['box'][2], crn['box'][2]) - max(imp['box'][0], crn['box'][0])
                
        #         if dist_x < 35.0 and is_above:
        #             if dist_x < min_dist:
        #                 min_dist = dist_x
        #                 best_crn = crn
            
        #     if best_crn:
        #         # 병합!
        #         # 박스 Union (x1_min, y1_min, x2_max, y2_max)
        #         new_box = [
        #             min(imp['box'][0], best_crn['box'][0]),
        #             min(imp['box'][1], best_crn['box'][1]),
        #             max(imp['box'][2], best_crn['box'][2]),
        #             max(imp['box'][3], best_crn['box'][3])
        #         ]
        #         imp['box'] = new_box # Implant 정보를 업데이트 (Crown 흡수)
        #         imp['cx'] = (new_box[0]+new_box[2])/2
        #         imp['cy'] = (new_box[1]+new_box[3])/2
        #         imp['has_crown'] = True # 크라운 속성 승계
                
        #         # Crown은 삭제 처리
        #         skip_indices.add(best_crn['id'])
        
        # 1-2. Natural 기준으로 Crown 찾기 (자연치 위에 씌운 크라운)
        obs_naturals = [o for o in observations if o['type'] == 'natural']
        
        for nat in obs_naturals:
            # Natural은 이미 label_hint가 있으므로, Crown과 합칠 때 신중해야 함.
            if nat['id'] in skip_indices: continue
            
            best_crn = None
            min_dist = 999
            
            for crn in obs_crowns:
                if crn['id'] in skip_indices: continue
                # 수직 정렬 확인
                dist_x = abs(nat['cx'] - crn['cx'])
                is_above = crn['cy'] < nat['cy'] 
                
                # X축 겹침(Overlap) 계산
                x_overlap = min(nat['box'][2], crn['box'][2]) - max(nat['box'][0], crn['box'][0])
                
                if dist_x < 20.0 and x_overlap > 0 and is_above:
                    if dist_x < min_dist:
                        min_dist = dist_x
                        best_crn = crn
            
            if best_crn:
                # 병합! Natural이 Crown 흡수 (좌표 확장)
                new_box = [
                    min(nat['box'][0], best_crn['box'][0]),
                    min(nat['box'][1], best_crn['box'][1]),
                    max(nat['box'][2], best_crn['box'][2]),
                    max(nat['box'][3], best_crn['box'][3])
                ]
                nat['box'] = new_box
                nat['cx'] = (new_box[0]+new_box[2])/2
                nat['cy'] = (new_box[1]+new_box[3])/2
                nat['has_crown'] = True # 크라운 속성 승계
                skip_indices.add(best_crn['id'])
        
        # 1-3. 고아 크라운(Orphan Crown) 삭제
        # Implant나 Natural과 합쳐지지 않은 Crown은 '허공에 뜬 크라운(Pontic)'일 가능성이 높으므로 제거
        # (오도토그램상 Ghost 방지)
        for crn in obs_crowns:
            if crn['id'] not in skip_indices:
                # 합쳐지지 않고 남은 녀석 -> 삭제
                skip_indices.add(crn['id'])

        final_observations = [o for o in observations if o['id'] not in skip_indices]
        merged_obs = final_observations # 이름 변경 (규칙 엔진 입력용)

        # --- RULES ENGINE v1.1 Integration ---
        from services.pano_rules_engine import RulesEngine
        
        # 1. 실행
        engine = RulesEngine(img_width=w, img_height=h)
        rule_result = engine.run(merged_obs, caries_objects)
        
        # 2. 결과 포맷팅: teeth_objects
        final_teeth_objects = []
        assigned_ids = set()
        
        # (1) 배정된 객체들
        for fdi, slot in rule_result['slots'].items():
            if slot['object_id'] is not None:
                obj = slot['candidates'][0]
                assigned_ids.add(obj['id'])
                
                t_obj = {
                    "id": f"tooth_{fdi}", 
                    "box": obj['box'],
                    "type": obj['type'], # natural / implant / crown
                    "tooth_label": str(fdi),
                    "confidence": obj['conf'],
                    "status": slot['status'], # confirmed / ambiguous
                    "geometry": "box",
                    "contour": obj.get('contour', []) # Pass contour
                }
                final_teeth_objects.append(t_obj)
        
        # (2) 미배정 객체들
        for obj in merged_obs:
            if obj['id'] not in assigned_ids:
                final_teeth_objects.append({
                    "id": f"obj_{obj['id']}",
                    "box": obj['box'],
                    "type": obj['type'],
                    "tooth_label": None,
                    "confidence": obj['conf'],
                    "status": "unassigned",
                    "geometry": "box",
                    "contour": obj.get('contour', []) # Pass contour
                })

        # 3. 결과 포맷팅: caries_objects
        final_caries_objects = []
        c_assignment = rule_result['caries_assignment']
        for c_obj in caries_objects:
            res = c_assignment.get(c_obj['id'])
            new_c = c_obj.copy()
            if res:
                new_c['status'] = res['status']
                if res.get('assigned_to'):
                    new_c['assigned_tooth'] = res['assigned_to']
                    
                    # [NEW] Inject finding into final_teeth_objects for Report/Feedback
                    for t_obj in final_teeth_objects:
                        if t_obj['tooth_label'] == res['assigned_to']:
                            t_obj['caries'] = True
                            if 'findings' not in t_obj: t_obj['findings'] = []
                            t_obj['findings'].append({
                                'type': 'caries',
                                'box': new_c['box'],
                                'conf': new_c['confidence']
                            })
                            break
            else:
                new_c['status'] = 'unassigned'
            final_caries_objects.append(new_c)

        # 3-2. Periapical Objects Processing (Strict Intersection Assignment)
        periapical_objects = []
        for i, (box, conf, lbl) in enumerate(per_list):
            if not box: continue
            x1, y1, x2, y2 = box
            p_area = (x2 - x1) * (y2 - y1)
            
            best_tooth = None
            max_inter_area = 0.0
            
            for t_obj in final_teeth_objects:
                if not t_obj['box']: continue
                tx1, ty1, tx2, ty2 = t_obj['box']
                
                # Intersection
                ix1 = max(x1, tx1)
                iy1 = max(y1, ty1)
                ix2 = min(x2, tx2)
                iy2 = min(y2, ty2)
                
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area > max_inter_area:
                        max_inter_area = inter_area
                        best_tooth = t_obj
            
            # Strict Assignment: Must overlap
            # (Optional: check intersection over lesion area ratio, e.g. > 10%)
            assigned_to = None
            status = 'unassigned'
            
            if best_tooth and max_inter_area > 0:
                assigned_to = best_tooth['tooth_label']
                status = 'confirmed'
                
                # Add to tooth findings for FeedbackGenerator
                best_tooth['periapical'] = True
                if 'findings' not in best_tooth: best_tooth['findings'] = []
                best_tooth['findings'].append({
                    'type': 'periapical',
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'conf': float(conf)
                })

                p_obj = {
                    "id": f"peri_{i}",
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "label": "Periapical Lesion",
                    "confidence": float(conf),
                    "status": status, 
                    "assigned_tooth": assigned_to,
                    "geometry": "box"
                }
                periapical_objects.append(p_obj)
            else:
                # Discard orphan periapical lesions (per user request: "visualize ONLY when overlapping")
                # Do not append to periapical_objects list
                pass

        # --- Implant Safety Guide Logic ---
        # 1. Prepare Target Masks (Nerve / Sinus)
        nerve_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        sinus_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        if iac_res.masks is not None:
            # We assume classes might distinguish, or we use spatial heuristic
            # IAC model usually class 0: 'mandibular_canal'.
            # If user provided a model that detects sinus, we check names.
            # Fallback: Spatial Split (Upper Half = Sinus, Lower Half = Nerve) if class is ambiguous
            
            for idx, m in enumerate(iac_res.masks.xy):
                if len(m) == 0: continue
                pts = np.int32([m])
                cls = int(iac_res.boxes.cls[idx])
                name = iac_res.names[cls].lower()
                
                is_sinus = "sinus" in name
                is_nerve = "nerve" in name or "canal" in name or "iac" in name
                
                # Heuristic: If name is generic or unknown, use Y-position
                if not is_sinus and not is_nerve:
                    cy = np.mean(m[:, 1])
                    if cy < img.shape[0] / 2: # Upper
                        is_sinus = True
                    else:
                        is_nerve = True
                
                
                if is_sinus:
                    cv2.fillPoly(sinus_mask, [pts], 255)
                elif is_nerve:
                    cv2.fillPoly(nerve_mask, [pts], 255)

        # 2. Calibration
        mm_per_px = self._get_calibration_scale(seg_res, final_teeth_objects)
        print(f"  [CALIB] Scale: {mm_per_px:.4f} mm/px")

        # --- 3.5 Measure Tooth-Nerve Distance (User Request) ---
        # "Center of Tooth Mask" -> "Center of CEJ" -> Straight line to Nerve logic
        cej_centers = []
        cej_model = self.models.get("cej")
        if cej_model:
            cej_results = cej_model(img, verbose=False)
            if cej_results and cej_results[0].masks:
                for c_pts in cej_results[0].masks.xy:
                    if len(c_pts) > 0:
                        c_pts = np.array(c_pts)
                        cx = np.mean(c_pts[:, 0])
                        cy = np.mean(c_pts[:, 1])
                        cej_centers.append((cx, cy))
        
        # Process Distance for Lower Teeth (Q3, Q4: 31-38, 41-48)
        for tooth in final_teeth_objects:
            lbl = tooth.get('tooth_label')
            if not lbl: continue
            q = int(lbl[0])
            if q not in [3, 4]: continue # Lower Jaw Only
            
            box = tooth.get('box')
            if not box: continue
            
            # 1. Tooth Centroid
            # Use 'contour' if available for precision, else box center
            contour = tooth.get('contour')
            if contour is not None and len(contour) > 0:
                 m = np.array(contour)
                 tcx = np.mean(m[:, 0])
                 tcy = np.mean(m[:, 1])
            else:
                 tcx = (box[0] + box[2]) / 2
                 tcy = (box[1] + box[3]) / 2
                 
            # 2. Find Closest CEJ Center
            best_cej = None
            min_cej_dist = 9999
            for (cx, cy) in cej_centers:
                # Logic: CEJ must be horizontally close (within box width)
                # And vertically... close to top of box? 
                # Lower Tooth: CEJ is at top (low Y). Root is bottom.
                if box[0] < cx < box[2]:
                     dy = abs(cy - box[1]) # dist to top
                     if dy < (box[3]-box[1]): # reasonably close
                         d = abs(cx - tcx) + dy
                         if d < min_cej_dist:
                             min_cej_dist = d
                             best_cej = (cx, cy)
            
            if best_cej:
                tooth['cej_center'] = best_cej  # [NEW] Store for neighbor calculation
                
                # 3. Cast Ray from CEJ through Tooth Center to Nerve
                p_start = best_cej
                p_through = (tcx, tcy)
                
                dx = p_through[0] - p_start[0]
                dy = p_through[1] - p_start[1]
                norm = math.hypot(dx, dy)
                if norm < 1.0: continue
                
                dx /= norm
                dy /= norm
                
                # Ray trace
                max_step = 2000 # enough to cover image
                # Start from CEJ? Or measure form where? 
                # User: "Center of CEJ line... drop straight line to nerve line... measure length"
                # So Distance from CEJ Center to Nerve Line Intersection.
                
                ray_x, ray_y = p_start[0], p_start[1]
                found = False
                p_end = None
                
                # Use numpy for speed? No, just loop is fine for single ray
                h_img, w_img = nerve_mask.shape
                for _ in range(max_step):
                    ray_x += dx
                    ray_y += dy
                    ix, iy = int(ray_x), int(ray_y)
                    
                    if not (0 <= ix < w_img and 0 <= iy < h_img):
                        break
                        
                    if nerve_mask[iy, ix] > 0:
                        # Hit Nerve
                        found = True
                        p_end = (ix, iy)
                        break
                
                if found and p_end:
                    dist_px = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
                    dist_mm = dist_px * mm_per_px
                    
                    tooth['nerve_dist_mm'] = dist_mm
                    tooth['nerve_dist_line'] = [p_start, p_end]
                    
                    # Store in overlap_warning if not overlap?
                    # Or separate finding? 
                    # User: "Take length". Usually displayed as text or graphic.


        # 4. Odontogram Map 생성 (Frontend 호환용)
        # 'missing'은 11~48 중 present가 아닌 것들
        present_fdis = set()
        odontogram_map = {}
        
        # 배정된 치아 정보로 맵 구성
        for tooth in final_teeth_objects:
            label = tooth.get('tooth_label')
            if not label: continue
            present_fdis.add(int(label))
            
            if label not in odontogram_map: odontogram_map[label] = []
            
            t_type = tooth['type']
            if "implant" in t_type: odontogram_map[label].append("implant")
            if "crown" in t_type or "bridge" in t_type: odontogram_map[label].append("crown")
                
            # 병변 정보 추가 (Caries)
            has_caries = any(c.get('assigned_tooth') == label and c['status'] == 'confirmed' for c in final_caries_objects)
            if has_caries: odontogram_map[label].append("caries")
            
            # --- Check Overlap (Nerve/Sinus) ---
            # Need mask points for this tooth.
            mask_pts = tooth.get('contour', [])
            
            if len(mask_pts) > 0:
                # Determine target based on quadrant
                q = int(label[0])

                # [MODIFIED] Restrict to Nerve Overlap ONLY (Lower Jaw)
                if q in [1, 2]: continue # Skip Upper/Sinus
                
                target_mask = nerve_mask 
                
                # Check Overlap & Quantify
                overlap_px = self._check_overlap(mask_pts, target_mask, img.shape[:2])
                
                if overlap_px > 0:
                    area_mm2 = overlap_px * (mm_per_px ** 2)
                    
                    tooth['nerve_overlap'] = True 
                    # tooth['sinus_overlap'] = False # Implicitly false
                    
                    # Store Area
                    tooth['overlap_area_mm2'] = area_mm2
                    
                    # Store Area
                    tooth['overlap_area_mm2'] = area_mm2
                        
                    # Also append to odontogram map for frontend
                    odontogram_map[label].append("overlap_warning")

        # [MOVED UP] Logic Update: Use Global Arch Curve to stabilize Y-centers
        # Split into Upper and Lower Centers
        upper_centers = []
        lower_centers = []
        
        viz_points = [] # For debug visualization
        
        for tooth in final_teeth_objects:
             try:
                 # Handle '16.0', 16, '16'
                 lbl_raw = tooth.get('tooth_label', '')
                 if not lbl_raw: continue
                 fdi = int(float(lbl_raw))
                 q = fdi // 10
             except:
                 continue
             
             cx, cy = None, None
             
             # Priority 1: Contour Centroid
             if tooth.get('contour') and len(tooth['contour']) >= 3:
                 try:
                     cnt = np.array(tooth['contour'], dtype=np.int32)
                     M = cv2.moments(cnt)
                     if M['m00'] != 0:
                         cx = int(M['m10'] / M['m00'])
                         cy = int(M['m01'] / M['m00'])
                 except: pass
             
             # Priority 2: Box Center
             if cx is None and tooth.get('box'):
                 bx1, by1, bx2, by2 = tooth['box']
                 cx, cy = (bx1+bx2)/2, (by1+by2)/2
             
             if cx is not None and cy is not None:
                 if q in [1, 2]: # Upper
                     upper_centers.append([cx, cy])
                 elif q in [3, 4]: # Lower
                     lower_centers.append([cx, cy])
                 viz_points.append((int(cx), int(cy)))
        
        # [NEW] Symmetric Fitting ('Decalcomania' - User Request)
        def fit_symmetric(centers, mid_x):
            if not centers: return None
            vc = np.array(centers)
            # Create Mirror Points across Midline
            mirrored = np.column_stack((2*mid_x - vc[:,0], vc[:,1]))
            # Combine Original + Mirrored
            combined = np.vstack((vc, mirrored))
            try:
                # Fit 2nd degree polynomial
                return np.poly1d(np.polyfit(combined[:,0], combined[:,1], 2))
            except: return None

        # Calculate Midlines (using incisors for accuracy, fallback to image center)
        mid_x_u, mid_x_l = w/2.0, w/2.0
        
        # Upper Midline (11, 21)
        u_incs = [t for t in final_teeth_objects if str(t.get('tooth_label')) in ['11', '21']]
        if u_incs: 
             ctrs = [(t['box'][0]+t['box'][2])/2 for t in u_incs]
             mid_x_u = float(np.mean(ctrs))
        
        # Lower Midline (31, 41)
        l_incs = [t for t in final_teeth_objects if str(t.get('tooth_label')) in ['31', '41']]
        if l_incs: 
             ctrs = [(t['box'][0]+t['box'][2])/2 for t in l_incs]
             mid_x_l = float(np.mean(ctrs))
        
        print(f"[ARCH DEBUG] Midlines: Upper={mid_x_u:.1f}, Lower={mid_x_l:.1f}")

        # Fit Curves (Relaxed requirement: 2 points * 2 sides = 4 points total -> Enough for 2nd degree)
        y_poly_fn_upper = fit_symmetric(upper_centers, mid_x_u) if len(upper_centers) >= 2 else None
        y_poly_fn_lower = fit_symmetric(lower_centers, mid_x_l) if len(lower_centers) >= 2 else None
            
        print(f"[ARCH DEBUG] Upper Centers: {len(upper_centers)}, Lower Centers: {len(lower_centers)}")
        print(f"[ARCH DEBUG] Upper Curve: {'YES' if y_poly_fn_upper else 'NO'}, Lower Curve: {'YES' if y_poly_fn_lower else 'NO'}")
        print(f"[ARCH DEBUG] Viz Points: {len(viz_points)}")

        if len(viz_points) > 1:
            # Sort by X to find mean spacing
            xs = sorted([p[0] for p in viz_points])
            diffs = [xs[i+1]-xs[i] for i in range(len(xs)-1)]
            # Filter huge jumps (e.g. midline or gaps)
            valid_diffs = [d for d in diffs if 20 < d < 150]
            if valid_diffs:
                avg_tooth_width = np.median(valid_diffs)
        
        print(f"[ARCH DEBUG] Avg Tooth Width: {avg_tooth_width:.1f}")

        # Missing 처리
        # Exclude wisdom teeth (18, 28, 38, 48) per user request
        all_fdis = [
            17,16,15,14,13,12,11, 21,22,23,24,25,26,27,
            47,46,45,44,43,42,41, 31,32,33,34,35,36,37
        ]
        missing_teeth_list = []
        
        # Helper to find tooth object by label
        def get_tooth_by_label(lbl):
            for t in final_teeth_objects:
                if t.get('tooth_label') == str(lbl): return t
            return None

        for fdi in all_fdis:
            slabel = str(fdi)
            if fdi not in present_fdis:
                if slabel not in odontogram_map: odontogram_map[slabel] = []
                odontogram_map[slabel].append("missing")
                
                # Create Report Object
                mt = {'tooth_label': fdi, 'type': 'missing'}
                
                # [NEW] Measure Bone Height for Missing Site (Virtual CEJ)
                q = fdi // 10
                if q in [3, 4]: # Lower Jaw
                     # Find Neighbors (Left/Right in Arch)
                     n = fdi % 10
                     neighbors = []
                     
                     # Simple logic: Check fdi-1 and fdi+1
                     # Handle midline cross (31 <-> 41)
                     # Candidates to check
                     candidates = []
                     if n > 1: candidates.append(q*10 + (n-1))
                     else:
                         # Midline
                         if q == 3: candidates.append(41) # 31 -> 41
                         elif q == 4: candidates.append(31) # 41 -> 31
                         
                     if n < 8: candidates.append(q*10 + (n+1))
                     
                     # Robust Neighbor Logic
                     # We need Gap Center (X) and Gap Level (Y)
                     
                     gap_cx = None
                     gap_cy = None
                     
                     # 1. Calculate Gap Center X using Boxes
                     # Re-fetch neighbor objects including those without CEJ for geometry
                     t_n1 = get_tooth_by_label(candidates[0]) if candidates else None
                     t_n2 = get_tooth_by_label(candidates[1]) if len(candidates) > 1 else None
                     
                     valid_boxes = [t for t in [t_n1, t_n2] if t and t.get('box')]
                     
                     print(f"[MISSING DEBUG] Tooth {fdi} (Q{q}): Candidates={candidates}, ValidBoxes={len(valid_boxes)}")
                     
                     if len(valid_boxes) >= 2:
                         # Between 2 boxes
                         b1 = valid_boxes[0]['box']
                         b2 = valid_boxes[1]['box']
                         # Sort by X
                         if b1[0] > b2[0]: b1, b2 = b2, b1
                         gap_cx = (b1[2] + b2[0]) / 2
                         print(f"[GAP DEBUG] Tooth {fdi} (2-neighbor): b1_right={b1[2]:.1f}, b2_left={b2[0]:.1f}, gap_cx={gap_cx:.1f}")
                     elif len(valid_boxes) == 1:
                         # Extrapolate from 1 box
                         b = valid_boxes[0]['box']
                         w = b[2] - b[0]
                         # Direction?
                         # distinct Q3 vs Q4 logic
                         # if Q3 (3x): 31..38. X increases Left. (Patient Right).
                         # Image: Right Side. 38 is Rightmost. 31 is Leftmost (towards midline).
                         # 3x: X increases as n increases? No.
                         
                         ref_lbl = int(valid_boxes[0]['tooth_label'])
                         is_left_neighbor = False # Left in image?
                         
                         # Determine relative position
                         # Simple heuristic: If ref < fdi in Q3 (31 < 32), ref is closer to midline.
                         # 31 is Image Left of 32? (Standard Pano).
                         # 41 is Image Right of 42?
                         
                         # Let's just use box center check if possible? No reference.
                         # Assume standard width gap.
                         
                         # If neighbors[0] (fdi-1):
                         # Q3: 35(missing). 34(ref). 34 is closer to midline (Left in image). So Gap is Right of 34.
                         # gap_cx = b[2] + w/2?
                         
                         # Q4: 45(missing). 44(ref). 44 is closer to midline (Right in image). So Gap is Left of 44.
                         # gap_cx = b[0] - w/2?
                         
                         dir_factor = 0
                         if q == 3: # Right Side of Image. 31(L)..38(R)
                              if ref_lbl < fdi: dir_factor = 1 # 34->35 (Right)
                              else: dir_factor = -1
                         elif q == 4: # Left Side of Image. 48(L)..41(R)
                              if ref_lbl < fdi: dir_factor = -1 # 44->45 (Left)
                              else: dir_factor = 1
                              
                         gap_cx = ((b[0]+b[2])/2) + (dir_factor * w * 1.1)
                         print(f"[GAP DEBUG] Tooth {fdi}: box_center={(b[0]+b[2])/2:.1f}, dir_factor={dir_factor}, w={w:.1f}, gap_cx={gap_cx:.1f}")
                         # [FIX Q1/Q2]
                         if q in [1, 2] and dir_factor == 0:
                             if q == 1: df_fix = -1 if ref_lbl < fdi else 1
                             elif q == 2: df_fix = 1 if ref_lbl < fdi else -1
                             gap_cx = ((b[0]+b[2])/2) + (df_fix * w * 1.1)
                             print(f"[GAP DEBUG] Tooth {fdi} Q1/Q2 FIX: df_fix={df_fix}, new gap_cx={gap_cx:.1f}")

                     # 2. Calculate Gap Level Y using CEJ
                     valid_cejs = []
                     valid_axes = []
                     tooth_axes_viz = []  # For debug visualization
                     valid_centroids = []  # Added missing initialization
                     
                     for t in [t_n1, t_n2]:
                         if not t: continue
                         cej = t.get('cej_center')
                         if cej: valid_cejs.append(cej)
                         
                         # Calculate Tooth Axis from Contour PCA
                         tooth_axis = None
                         if t.get('contour'):
                             try:
                                 pts = np.array(t['contour'], dtype=np.float32)
                                 if pts.ndim == 3: pts = pts.reshape(-1, 2)
                                 
                                 # PCA
                                 mean = np.mean(pts, axis=0)
                                 centered = pts - mean
                                 cov = np.cov(centered, rowvar=False)
                                 evals, evecs = np.linalg.eigh(cov)
                                 
                                 # Major axis (Eigenvector with largest eigenvalue)
                                 major_axis = evecs[:, np.argmax(evals)]
                                 major_axis = major_axis / np.linalg.norm(major_axis)
                                 
                                 # Ensure pointing down (Y+)
                                 if major_axis[1] < 0: major_axis = -major_axis
                                 
                                 tooth_axis = (float(major_axis[0]), float(major_axis[1]))
                                 valid_axes.append(tooth_axis)
                                 
                                 # For Visual Debug
                                 tooth_axes_viz.append({
                                     'label': t.get('tooth_label', '?'),
                                     'center': mean,
                                     'vec': major_axis
                                 })
                             except Exception as e:
                                 print(f"[AXIS ERROR] PCA failed for {t.get('tooth_label')}: {e}")
                         
                         if not tooth_axis:
                             valid_axes.append((0, 1))
                
                     # Determine Gap CEJ Y and Vector
                     axis_vec = (0, 1) # Default Vertical
                     
                     # 1. Calc Axis Vector (Mean of neighbors)
                     if valid_axes:
                         avg_vx = sum(v[0] for v in valid_axes) / len(valid_axes)
                         avg_vy = sum(v[1] for v in valid_axes) / len(valid_axes)
                         n = math.hypot(avg_vx, avg_vy)
                         if n > 0: axis_vec = (avg_vx/n, avg_vy/n)
                     
                     # 3. Robust Gap Center Logic (Mix Centroids & Box Centers)
                     # We need 2 points to find a midpoint. Use Centroid if available, else Box Center.
                     
                     points = []
                     
                     # Check t_prev (n1)
                     p1 = None
                     if t_n1 and t_n1.get('tooth_label') in [t.get('label') for t in tooth_axes_viz]:
                         # Find matching centroid (inefficient but safe)
                         for ax in tooth_axes_viz:
                             if ax['label'] == t_n1.get('tooth_label'):
                                 p1 = (ax['center'][0], ax['center'][1])
                                 break
                     if p1 is None and t_n1 and t_n1.get('box'):
                         b = t_n1['box']
                         p1 = ((b[0]+b[2])/2, (b[1]+b[3])/2) # Box Center fallback
                         
                     # Check t_next (n2)
                     p2 = None
                     if t_n2 and t_n2.get('tooth_label') in [t.get('label') for t in tooth_axes_viz]:
                         for ax in tooth_axes_viz:
                             if ax['label'] == t_n2.get('tooth_label'):
                                 p2 = (ax['center'][0], ax['center'][1])
                                 break
                     if p2 is None and t_n2 and t_n2.get('box'):
                         b = t_n2['box']
                         p2 = ((b[0]+b[2])/2, (b[1]+b[3])/2) # Box Center fallback
                     
                     if p1 and p2:
                         # Use these points for Gap Center
                         old_cx = gap_cx
                         gap_cx = (p1[0] + p2[0]) / 2
                         gap_cy = (p1[1] + p2[1]) / 2
                         print(f"[GAP DEBUG] Force Centroid/Box Midpoint: p1={p1}, p2={p2}")
                         print(f"[GAP DEBUG] Gap X: {old_cx:.1f} -> {gap_cx:.1f}, Gap Y: {gap_cy:.1f}")
                     
                     elif valid_centroids:
                         # Only 1 side found (rare if box exists), use partial
                         gap_cy = valid_centroids[0][1] 
                         print(f"[GAP DEBUG] Single Centroid Fallback: gap_cy={gap_cy:.1f}")

                     # Fallback if NO centroids (use CEJ or Box - legacy) OR Gap X missing
                     if gap_cy is None or gap_cx is None:
                          # [NEW] Global Curve Fallback (For Free-end / Multiple Missing)
                          target_poly = None
                          if q in [1, 2] and 'y_poly_fn_upper' in locals() and y_poly_fn_upper:
                              target_poly = y_poly_fn_upper
                          elif q in [3, 4] and 'y_poly_fn_lower' in locals() and y_poly_fn_lower:
                              target_poly = y_poly_fn_lower
                          
                          if target_poly:
                              # Find nearest Anchor in same Quadrant
                              best_anchor = None
                              min_dist_slots = 999
                              
                              for t_obj in final_teeth_objects:
                                  try:
                                      t_lbl = int(float(t_obj.get('tooth_label')))
                                      if t_lbl // 10 == q:
                                          dist = abs(fdi - t_lbl)
                                          if dist < min_dist_slots:
                                              min_dist_slots = dist
                                              best_anchor = t_obj
                                  except: pass
                              
                              if best_anchor:
                                  anchor_lbl = int(float(best_anchor.get('tooth_label')))
                                  slots_diff = fdi - anchor_lbl # e.g. 37 - 35 = +2
                                  
                                  dx_step = 0
                                  if q in [2, 3]: # Patient Left / Image Right
                                      # FDI increase -> X increase
                                      dx_step = slots_diff * avg_tooth_width
                                  else: # Patient Right / Image Left (Q1, Q4)
                                      # FDI increase -> X decrease
                                      dx_step = -slots_diff * avg_tooth_width
                                      
                                  # Get Anchor CX (Box center fallback)
                                  acx = 0
                                  if best_anchor.get('box'):
                                      bx = best_anchor['box']
                                      acx = (bx[0]+bx[2])/2
                                  
                                  gap_cx = acx + dx_step
                                  gap_cy = target_poly(gap_cx)
                                  
                                  print(f"[GAP DEBUG] Global Curve Fallback for {fdi}: Anchor={anchor_lbl}, Slots={slots_diff}, Gap=({gap_cx:.1f}, {gap_cy:.1f})")
                                  
                                  # Calculate Normal Axis
                                  deriv_fn = np.polyder(target_poly)
                                  m = deriv_fn(gap_cx)
                                  
                                  if q in [1, 2]: # Upper
                                      axis_vec = (-m, -1.0)
                                  else: # Lower
                                      axis_vec = (-m, 1.0)
                                      
                                  # Normalize
                                  norm = math.hypot(axis_vec[0], axis_vec[1])
                                  if norm > 0:
                                      axis_vec = (axis_vec[0]/norm, axis_vec[1]/norm)

                     if gap_cy is None:
                          if valid_cejs:
                              ys = [c[1] for c in valid_cejs]
                              gap_cy = sum(ys)/len(ys)
                          elif valid_boxes:
                              # ... existing legacy ...
                              cej_estimates = []
                              for vb in valid_boxes:
                                  box = vb['box']
                                  estimated_cej_y = box[1] + (box[3] - box[1]) * 0.7
                                  cej_estimates.append(estimated_cej_y)
                              gap_cy = sum(cej_estimates) / len(cej_estimates)
                              print(f"[GAP Y DEBUG] Fallback to estimated CEJ: {gap_cy:.1f}")

                     # 3. Raycast along Axis Vector
                     cx_val = gap_cx if gap_cx is not None else 0.0
                     cy_val = gap_cy if gap_cy is not None else 0.0
                     print(f"[RAYCAST DEBUG] Tooth {fdi}: gap_cx={cx_val:.1f}, gap_cy={cy_val:.1f}, axis_vec=({axis_vec[0]:.3f}, {axis_vec[1]:.3f})")
                     if gap_cx is not None and gap_cy is not None:
                         cx, cy = gap_cx, gap_cy
                         vx, vy = axis_vec
                         
                         p_nerve = None
                         
                         # Ray tracing with float steps
                         ray_x, ray_y = cx, cy
                         h_img, w_img = nerve_mask.shape
                         max_step = 2000
                         
                         found_nerve = False
                         for _ in range(max_step):
                             ray_x += vx
                             ray_y += vy
                             ix, iy = int(ray_x), int(ray_y)
                             
                             if not (0 <= ix < w_img and 0 <= iy < h_img):
                                 break
                                 
                             if nerve_mask[iy, ix] > 0:
                                 p_nerve = (ix, iy)
                                 found_nerve = True
                                 break
                             
                         if found_nerve and p_nerve:
                             dist_px = math.hypot(p_nerve[0]-cx, p_nerve[1]-cy)
                             
                             # Safety Margin: 2mm (approx 40px at 0.05mm/px)
                             pixels_per_mm = 1.0 / mm_per_px if mm_per_px > 0 else 20.0
                             safety_margin_px = 2.0 * pixels_per_mm
                             
                             safe_dist_px = max(0, dist_px - safety_margin_px)
                             dist_mm = safe_dist_px * mm_per_px
                             
                             # Calculate safe endpoint (retract from nerve)
                             # p_nerve is (ix, iy). vector is (vx, vy).
                             # We want to move back by safety_margin_px along (vx, vy)
                             # BUT wait, (vx, vy) is the direction FROM gap TO nerve.
                             # So we subtract (vx, vy) * safety_margin_px from p_nerve? No.
                             # safe_end = start + axis * safe_dist_px
                             
                             safe_end_x = cx + vx * safe_dist_px
                             safe_end_y = cy + vy * safe_dist_px
                             
                             mt['implant_guide'] = {
                                 'dist_mm': dist_mm,
                                 # Original nerve point vs Safe point? 
                                 # User wants "Safety Margin deducted", so show the SAFE line.
                                 'line_coords': [cx, cy, safe_end_x, safe_end_y], 
                                 'type': 'vertical',
                                 'safety_margin_mm': 2.0
                             }
                             print(f"[IMPLANT DEBUG] Tooth {fdi}: Raw Dist={dist_px*mm_per_px:.1f}mm, Safe Dist={dist_mm:.1f}mm (Margin 2mm)")
                
                
                # --- Implant Guide Logic (Merged) ---
                # Check neighbors for Box calculation
                t_prev = get_tooth_by_label(fdi - 1)
                t_next = get_tooth_by_label(fdi + 1)
                
                # Use CEJ-based guide if computed, else None
                implant_guide = mt.get('implant_guide')

                # Calculate estimated box for visualization
                box = None
                
                ref_box = None
                if t_prev and t_prev.get('box'): ref_box = t_prev['box']
                elif t_next and t_next.get('box'): ref_box = t_next['box']
                
                if t_prev and t_next and t_prev.get('box') and t_next.get('box'):
                    p_box = t_prev['box']
                    n_box = t_next['box']
                    
                    x_coords = [p_box[0], p_box[2], n_box[0], n_box[2]]
                    y_coords = [p_box[1], p_box[3], n_box[1], n_box[3]]
                    
                    # 1.2x Expansion requested in User Prompt for CROP SIZE
                    # "크롭 크기는 1.2 배" -> Logic handled in report_v2 or here?
                    # report_v2 handles margin. But here we define the crop box.
                    # If we set tight box here, report adds margin.
                    # User probably means expand the box itself to include surrounding?
                    # The current logic "Union of Neighbors" already is wide.
                    # Let's stick to Union box here, and ensure report adds margin.
                    
                    x1 = min(x_coords)
                    y1 = min(y_coords)
                    x2 = max(x_coords)
                    y2 = max(y_coords)
                    
                    box = [max(0, x1), max(0, y1), x2, y2]
                
                elif ref_box:
                    # Single neighbor extrapolation
                    w = ref_box[2] - ref_box[0]
                    h = ref_box[3] - ref_box[1]
                    ref_cx = (ref_box[0] + ref_box[2]) / 2
                    ref_cy = (ref_box[1] + ref_box[3]) / 2
                    q = fdi // 10
                    
                    dir_x = 0
                    if q in [1, 4]: 
                         if t_prev: dir_x = -1 
                         else: dir_x = 1 
                    else: 
                         if t_prev: dir_x = 1 
                         else: dir_x = -1
                         
                    miss_cx = ref_cx + (dir_x * w * 1.1)
                    other_cx = miss_cx + (dir_x * w * 1.1)
                    all_cxs = [ref_cx, miss_cx, other_cx]
                    min_cx = min(all_cxs)
                    max_cx = max(all_cxs)
                    x1 = min_cx - w/2
                    x2 = max_cx + w/2
                    y1 = ref_cy - h/2 
                    y2 = ref_cy + h/2
                    box = [max(0, x1), max(0, y1), x2, y2]

                missing_obj = {
                    "tooth_label": slabel,
                    "box": box,
                    "type": "missing",
                    "guide_status": locals().get('guide_status', 'Unknown')
                }
                if implant_guide:
                    missing_obj['implant_guide'] = implant_guide
                
                # ==== VISUAL DEBUG: Save annotated image                # --- Visual Debug (Save Image) ---
                try:
                    # Create debug image with context
                    h, w = img.shape[:2]
                    
                    debug_img = img.copy()
                    
                    # [NEW] Draw CEJ and Bone Level Contours (Global)
                    if 'cej_res' in locals() and cej_res and cej_res[0].masks:
                         for seg in cej_res[0].masks.xy:
                             pts = np.array(seg, np.int32).reshape((-1, 1, 2))
                             cv2.polylines(debug_img, [pts], False, (0, 165, 255), 2) # Orange
                    
                    # Bone Level Contour (Re-enabled per user request)
                    if 'bone_res' in locals() and bone_res and bone_res[0].masks:
                         for seg in bone_res[0].masks.xy:
                             pts = np.array(seg, np.int32).reshape((-1, 1, 2))
                             cv2.polylines(debug_img, [pts], False, (0, 255, 0), 2) # Green (to distinguish from Yellow Curve)

                    # [NEW] Draw BOTH Arch Curves (Upper & Lower)
                    curve_color = (0, 255, 255) # Yellow
                    
                    for poly_name, label in [('y_poly_fn_upper', 'Upper Arch'), ('y_poly_fn_lower', 'Lower Arch')]:
                        if poly_name in locals() and locals()[poly_name]:
                            try:
                                poly_fn = locals()[poly_name]
                                curve_x = np.linspace(0, w, num=100)
                                curve_y = poly_fn(curve_x)
                                curve_pts = np.column_stack((curve_x, curve_y)).astype(np.int32)
                                cv2.polylines(debug_img, [curve_pts], False, curve_color, 2)
                            except: pass

                    # [NEW] Draw Arch Curve Points (Red)
                    if 'viz_points' in locals() and viz_points:
                        for vp in viz_points:
                             cv2.circle(debug_img, vp, 4, (0, 0, 255), -1) # Red

                    # Draw Neighbors
                    if t_prev and t_prev.get('box'):
                        b = t_prev['box']
                        cv2.rectangle(debug_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                        cv2.putText(debug_img, f"Prev {t_prev.get('tooth_label', '?')}", 
                                   (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    if t_next and t_next.get('box'):
                        b = t_next['box']
                        cv2.rectangle(debug_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                        cv2.putText(debug_img, f"Next {t_next.get('tooth_label', '?')}", 
                                   (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Draw Tooth Axes (Cyan)
                    if 'tooth_axes_viz' in locals():
                        for av in tooth_axes_viz:
                            c = av['center']
                            v = av['vec']
                            p1 = c - v * 40
                            p2 = c + v * 40
                            cv2.arrowedLine(debug_img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2) # Cyan
                            cv2.putText(debug_img, "Axis", (int(p1[0]), int(p1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # Draw gap center (red circle)
                    if gap_cx is not None and gap_cy is not None:
                        cv2.circle(debug_img, (int(gap_cx), int(gap_cy)), 8, (0, 0, 255), -1)
                        cv2.putText(debug_img, f"Gap({gap_cx:.0f},{gap_cy:.0f})", 
                                   (int(gap_cx)+15, int(gap_cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Draw Vertical Reference Axis (White Dashed)
                        vx_start = (int(gap_cx), 0)
                        vx_end = (int(gap_cx), h)
                        cv2.line(debug_img, vx_start, vx_end, (200, 200, 200), 1, cv2.LINE_AA)
                        cv2.putText(debug_img, "Y-Axis", (int(gap_cx)+5, int(gap_cy)-120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    # Draw axis vector (Mean Angle)
                    if axis_vec and gap_cx is not None and gap_cy is not None:
                        arrow_len = 100
                        end_x = int(gap_cx + axis_vec[0] * arrow_len)
                        end_y = int(gap_cy + axis_vec[1] * arrow_len)
                        cv2.arrowedLine(debug_img, (int(gap_cx), int(gap_cy)), (end_x, end_y), (255, 0, 0), 4, tipLength=0.3) # Blue
                        cv2.putText(debug_img, f"Mean Axis", 
                                   (end_x+10, end_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Draw final measurement line (magenta) - only if implant_guide exists
                    if implant_guide and implant_guide.get('line_coords'):
                        x1_line, y1_line, x2_line, y2_line = implant_guide['line_coords']
                        cv2.line(debug_img, (int(x1_line), int(y1_line)), (int(x2_line), int(y2_line)), (255, 0, 255), 3)
                        cv2.circle(debug_img, (int(x1_line), int(y1_line)), 5, (0, 255, 255), -1)  # start (yellow)
                        cv2.circle(debug_img, (int(x2_line), int(y2_line)), 5, (255, 0, 255), -1)  # end (magenta)
                        dist_mm = implant_guide.get('dist_mm', 0)
                        cv2.putText(debug_img, f"Dist: {dist_mm:.1f}mm", 
                                   (int((x1_line+x2_line)/2), int((y1_line+y2_line)/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    else:
                        # No implant guide - mark as "RAYCAST FAILED"
                        if gap_cx is not None and gap_cy is not None:
                            cv2.putText(debug_img, "RAYCAST FAILED", 
                                       (int(gap_cx), int(gap_cy)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                    # Crop around gap for better visibility
                    cx_int = int(gap_cx) if gap_cx else w//2
                    cy_int = int(gap_cy) if gap_cy else h//2
                    pad = 300
                    x1 = max(0, cx_int - pad)
                    x2 = min(w, cx_int + pad)
                    y1 = max(0, cy_int - pad)
                    y2 = min(h, cy_int + pad)
                    
                    debug_crop = debug_img[y1:y2, x1:x2]
                    
                    debug_path = out_dir / f"debug_missing_{fdi}.png"
                    cv2.imwrite(str(debug_path), debug_crop)
                    print(f"[VISUAL DEBUG] Saved: {debug_path}")
                    
                except Exception as e:
                    print(f"[VISUAL DEBUG] Failed to save image for tooth {fdi}: {e}")
                    
                if box: missing_teeth_list.append(missing_obj)
                else: missing_teeth_list.append(missing_obj)

        # 4.5 Measure Safety Distance for Present Teeth (User Request)
        # Ensure bonelevel_dict is available
        bonelevel_dict = locals().get("pbl_dict", {})

        for tooth in final_teeth_objects:
             # Only if overlap warning is active
             if tooth.get('nerve_overlap') or tooth.get('sinus_overlap'):
                 # Get Mask Points
                 mask_pts = tooth.get('contour', [])
                 # ID lookup removed as it was broken and contour is embedded

                 # If no mask points, fallback to box center
                 if len(mask_pts) > 0:
                     # Mask Centroid
                     M = cv2.moments(np.array(mask_pts, np.int32))
                     if M['m00'] != 0:
                         cx = int(M['m10'] / M['m00'])
                         cy = int(M['m01'] / M['m00'])
                     else: 
                         cx = int((tooth['box'][0]+tooth['box'][2])/2)
                         # Use Mask Centroid Y if available, but blend with Box Center
                         cy = int(M['m01'] / M['m00'])

                     # Blend with Curve Y if valid (Stabilization)
                     q = int(tooth.get('tooth_label', '0')[0])
                     target_poly = y_poly_fn_upper if q in [1, 2] else y_poly_fn_lower
                     
                     if target_poly is not None:
                         try:
                             curve_y = target_poly(cx)
                             # Weighted Blend: 60% Curve, 40% Mask/Box Center
                             cy = int(curve_y * 0.6 + cy * 0.4)
                         except: pass
                 else:
                     cx = int((tooth['box'][0]+tooth['box'][2])/2)
                     cy = int((tooth['box'][1]+tooth['box'][3])/2)
                     
                     q = int(tooth.get('tooth_label', '0')[0])
                     target_poly = y_poly_fn_upper if q in [1, 2] else y_poly_fn_lower
                     
                     if target_poly is not None:
                         # No mask, trust curve more
                         try:
                             cy = int(target_poly(cx) * 0.8 + cy * 0.2)
                         except: pass
                 
                 is_upper = q in [1, 2]
                 target_mask = sinus_mask if is_upper else nerve_mask
                 
                 dist, coords = self._measure_shortest_distance([cx, cy], target_mask, mm_per_px)
                 if dist is not None:
                     tooth['safety_guide'] = {
                         "dist_mm": dist,
                         "line_coords": coords,
                         "mode": 'upper' if is_upper else 'lower'
                     }

        # 5. Legacy/Extra fields
        # Implant Metrics (이미 위에서 계산된 raw_implant_data 사용)
        total_time = time.perf_counter() - t0
        print(f"  [PERF] Pipeline finished. Confirmed teeth: {len(present_fdis)}")

        # bonelevel_dict fallback (pbl_dict may exist from get_bonelevel)
        bonelevel_dict = locals().get("pbl_dict", {})

        # Draw Rules Engine Debug Info (Midline & Split) - DISABLED per user request
        debug_info = rule_result.get('debug', {})
        # X/Y axis overlay disabled
        # if 'overlay' in locals():
        #     # X Midline (Cyan)
        #     x_mid = debug_info.get('shared_x_mid')
        #     if x_mid is not None:
        #         cv2.line(overlay, (int(x_mid), 0), (int(x_mid), int(h)), (255, 255, 0), 2) 
        #         cv2.putText(overlay, f"X-Mid", (int(x_mid)+5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        #     
        #     # Y Split (Yellow) - Curve or Line
        #     y_coeffs = debug_info.get('curve_coeffs')
        #     y_split = debug_info.get('y_split')
        #     
        #     if y_coeffs:
        #         # Draw Curve
        #         poly_pts = []
        #         poly_fn = np.poly1d(y_coeffs)
        #         # Sample points every 10px
        #         for px in range(0, int(w), 10):
        #             py = int(poly_fn(px))
        #             poly_pts.append([px, py])
        #         pts_arr = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
        #         cv2.polylines(overlay, [pts_arr], False, (0, 255, 255), 2)
        #         
        #         # Label at start
        #         py_start = int(poly_fn(5))
        #         cv2.putText(overlay, f"Y-Curve", (5, py_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        #         
        #     elif y_split is not None:
        #         # Fallback to straight line
        #         cv2.line(overlay, (0, int(y_split)), (w, int(y_split)), (0, 255, 255), 2) 
        #         cv2.putText(overlay, f"Y-Split", (5, int(y_split)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Debug Grid Points (Ideal Slots vs Real Anchors)
            # if 'slots' in rule_result:
            #     for fdi, s in rule_result['slots'].items():
            #         # Ideal X (Green Cross)
            #         if 'ideal_x' in s:
            #             ix = int(s['ideal_x'])
            #             iy = int(h/4) if int(fdi)//10 in [1,2] else int(h*3/4) # Rough Y
            #             # Draw vertical line segment for grid
            #             cv2.line(overlay, (ix, iy-20), (ix, iy+20), (0, 255, 0), 1)
            #             cv2.putText(overlay, str(fdi), (ix-10, iy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw Final Tooth Numbers on Overlay (with simple collision avoidance)
        if 'overlay' in locals() and final_teeth_objects:
            overlay_copy = overlay.copy()
            
            # 1. Collect initial positions
            labels_to_draw = []
            for t_obj in final_teeth_objects:
                lbl = t_obj.get("tooth_label")
                if lbl and t_obj.get("box"):
                    x1, y1, x2, y2 = map(int, t_obj["box"])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    # Logic: Upper (1x, 2x) -> Bottom edge (y2)
                    #        Lower (3x, 4x) -> Top edge (y1)
                    draw_y = cy
                    try:
                        q = int(lbl) // 10
                        if q in [1, 2]: draw_y = y2
                        elif q in [3, 4]: draw_y = y1
                    except: pass
                    
                    # Store: [x, y, label, original_box_center_x, original_box_center_y]
                    labels_to_draw.append({
                        'x': cx, 'y': draw_y, 
                        'orig_x': cx, 'orig_y': draw_y,
                        'lbl': str(lbl)
                    })
            
            # 2. Iterative Repulsion (Simple Force-Directed)
            # Text size approx: 20px width, 20px height
            radius = 18 
            
            for _ in range(5): # 5 iterations
                for i in range(len(labels_to_draw)):
                    for j in range(i + 1, len(labels_to_draw)):
                        l1 = labels_to_draw[i]
                        l2 = labels_to_draw[j]
                        
                        dx = l1['x'] - l2['x']
                        dy = l1['y'] - l2['y']
                        dist_sq = dx*dx + dy*dy
                        
                        min_dist = radius * 2
                        if dist_sq < min_dist * min_dist:
                            # Push apart
                            dist = math.sqrt(dist_sq)
                            if dist < 0.1: dist = 0.1
                            
                            overlap = min_dist - dist
                            # Normalize vector
                            nx = dx / dist
                            ny = dy / dist
                            
                            # Move each by half overlap
                            shift = overlap * 0.5
                            l1['x'] += nx * shift
                            l1['y'] += ny * shift
                            l2['x'] -= nx * shift
                            l2['y'] -= ny * shift

            # 3. Draw
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            thickness = 2
            
            for item in labels_to_draw:
                x, y = int(item['x']), int(item['y'])
                ox, oy = item['orig_x'], item['orig_y']
                txt = item['lbl']
                
                # Draw Red Dot at Original Point (Interface Point)
                cv2.circle(overlay_copy, (ox, oy), 4, (0, 0, 255), -1)
                
                # If moved significantly, draw a connector line
                if (x-ox)**2 + (y-oy)**2 > 25: # > 5px move
                    cv2.line(overlay_copy, (ox, oy), (x, y-10), (200, 200, 200), 1)

                # Draw Text
                # Outline
                cv2.putText(overlay_copy, txt, (x - 10, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                # Face
                cv2.putText(overlay_copy, txt, (x - 10, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
            
            overlay = overlay_copy

        # 4. Draw Safety Guides & Implant Guides on Overlay (User Request: "Crop from Overlay")
        # Ensure these are baked into the final image
        if 'overlay' in locals():
             # 1. Missing Teeth Implant Guides
             if 'missing_teeth_list' in locals():
                 for m_obj in missing_teeth_list:
                     guide = m_obj.get('implant_guide')
                     if guide:
                         coords = guide.get('line_coords')
                         dist = guide.get('dist_mm', 0)
                         if coords:
                             x1, y1, x2, y2 = map(int, coords)
                             cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow
                             cv2.circle(overlay, (x1, y1), 4, (0, 0, 255), -1)      # Red Tip
                             cv2.circle(overlay, (x2, y2), 4, (0, 0, 255), -1)
                             
                             # Text
                             mx, my = (x1+x2)//2, (y1+y2)//2
                             cv2.putText(overlay, f"{dist:.1f}mm", (mx+5, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

             # 2. Present Teeth Safety Guides
             if 'final_teeth_objects' in locals():
                 for t_obj in final_teeth_objects:
                     guide = t_obj.get('safety_guide')
                     if guide:
                         coords = guide.get('line_coords')
                         dist = guide.get('dist_mm', 0)
                         if coords:
                             x1, y1, x2, y2 = map(int, coords)
                             cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow
                             cv2.circle(overlay, (x1, y1), 4, (0, 0, 255), -1)      # Red Tip
                             cv2.circle(overlay, (x2, y2), 4, (0, 0, 255), -1)

                             # Text
                             mx, my = (x1+x2)//2, (y1+y2)//2
                             cv2.putText(overlay, f"{dist:.1f}mm", (mx+5, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 5. Visualize Global Arch Curve & Tissue Structures
        if 'overlay' in locals():
             h, w = overlay.shape[:2]
             
             # Draw CEJ (Orange)
             if 'cej_res' in locals() and cej_res and cej_res[0].masks:
                 for seg in cej_res[0].masks.xy:
                     pts = np.array(seg, np.int32).reshape((-1, 1, 2))
                     cv2.polylines(overlay, [pts], False, (0, 165, 255), 2)
             
             # Draw Bone Level (Green)
             if 'bone_res' in locals() and bone_res and bone_res[0].masks:
                 for seg in bone_res[0].masks.xy:
                     pts = np.array(seg, np.int32).reshape((-1, 1, 2))
                     cv2.polylines(overlay, [pts], False, (0, 255, 0), 2)
             
             # Draw Arch Curves (Yellow)
             for poly_name, label in [('y_poly_fn_upper', 'Upper'), ('y_poly_fn_lower', 'Lower')]:
                 if poly_name in locals() and locals()[poly_name]:
                     try:
                         poly_fn = locals()[poly_name]
                         curve_x = np.linspace(0, w, num=100)
                         curve_y = poly_fn(curve_x)
                         curve_pts = np.column_stack((curve_x, curve_y)).astype(np.int32)
                         cv2.polylines(overlay, [curve_pts], False, (0, 255, 255), 2)
                     except: pass
             
             # Draw Arch Curve Points (Red)
             if 'viz_points' in locals() and viz_points:
                 for vp in viz_points:
                     cv2.circle(overlay, vp, 4, (0, 0, 255), -1)

        # Save Overlay Image
        out_file = out_dir / "overlay.png"
        if 'overlay' in locals():
            cv2.imwrite(str(out_file), overlay)
            
        # Save BoneLevel Visualization (Result of CEJ/BLV models)
        if 'bl_canvas' in locals() and hasattr(bl_canvas, 'shape'):
             bl_file = out_dir / "bl_viz.png"
             cv2.imwrite(str(bl_file), bl_canvas)

        # --- Prepare Contours for Visualization ---
        def get_contours(mask):
            # Pass copy to avoid modification
            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Convert to list of lists (JSON serializable)
            res = []
            for c in cnts:
                # c is (N, 1, 2) -> (N, 2)
                res.append(c.reshape(-1, 2).tolist())
            return res

        nerve_contours = get_contours(nerve_mask)
        sinus_contours = get_contours(sinus_mask)
        print(f"  [CONTOURS] Nerve: {len(nerve_contours)}, Sinus: {len(sinus_contours)}")

        return {
            "image_path": str(image_path),
            "overlay_path": str(out_dir / "overlay.png") if out_dir else "",
            "teeth": final_teeth_objects,
            "teeth_objects": final_teeth_objects, # Legacy
            "caries": final_caries_objects,
            "caries_objects": final_caries_objects, # Legacy
            "periapical": periapical_objects,
            "periapical_objects": periapical_objects, # Legacy
            "missing_teeth": missing_teeth_list,
            "implant_metrics": raw_implant_data if 'raw_implant_data' in locals() else {}, # Use raw metrics (referenced by box index or we need to map to id)
            "bonelevel": bonelevel_dict,
            "cej_count": cej_count,
            "bl_count": bl_count,
            "debug_rules": rule_result['debug'],
            "bl_canvas": bl_canvas.tolist() if hasattr(bl_canvas, 'tolist') else None,
            "odontogram_map": odontogram_map,
            # Visual Data
            "nerve_contours": nerve_contours,
            "sinus_contours": sinus_contours,
            # Legacy keys
            "detected_teeth": final_teeth_objects,
            "overlay_path": str(out_dir / "overlay.png") if out_dir else "", 
        }

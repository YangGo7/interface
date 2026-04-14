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
from services.visualizer import PanoVisualizer # [NEW] Refactoring
from services.tooth_logic import ToothLogic # [NEW] Refactoring


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
        self.visualizer = PanoVisualizer()
        self.tooth_logic = ToothLogic() # [NEW] Refactoring
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

    # [REFACTORED] Visualization methods moved to services/visualizer.py
    # _bbox_center_inside, _overlay_seg, _overlay_det removed.

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

    def _run_inference(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Executes all YOLO models and returns raw results.
        Refactored from run()
        """
        results = {}
        
        # Load models
        seg_model = self._load_model("pano_seg")
        caries_model = self._load_model("caries")
        peri_model = self._load_model("periapical")
        iac_model = self._load_model("iac")
        self._load_model("cej")
        self._load_model("bonelevel")

        # 1. Segmentation
        seg_cfg = self.model_cfg["pano_seg"]
        t0 = time.perf_counter()
        results['seg_res'] = seg_model.predict(
            source=img,
            imgsz=seg_cfg.get("default_imgsz", 1024),
            conf=seg_cfg.get("default_confidence", 0.25),
            iou=seg_cfg.get("default_iou", 0.7),
            retina_masks=True,
            verbose=False,
        )[0]
        results['seg_time'] = time.perf_counter() - t0
        print(f"  [PERF][{self.device.upper()}] pano_seg inference: {results['seg_time']:.3f}s")
        
        # 2. Aux Segmentation (Hybrid)
        results['aux_boxes_data'] = []
        if "pano_seg_aux" in self.model_cfg:
            try:
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
                        results['aux_boxes_data'].append((box, str(l)))
                
                print(f"  [PERF][{self.device.upper()}] pano_seg_aux: {time.perf_counter()-t_aux:.3f}s")
            except Exception as e:
                print(f"  [WARN] Failed to run pano_seg_aux: {e}")

        # 3. Caries
        car_cfg = self.model_cfg["caries"]
        t1 = time.perf_counter()
        results['car_res'] = caries_model.predict(
            source=img,
            imgsz=car_cfg.get("default_imgsz", 640),
            conf=car_cfg.get("default_confidence", 0.2),
            iou=car_cfg.get("default_iou", 0.5),
            verbose=False,
        )[0]
        results['car_time'] = time.perf_counter() - t1
        print(f"  [PERF][{self.device.upper()}] caries inference: {results['car_time']:.3f}s")

        # 4. Periapical
        per_cfg = self.model_cfg["periapical"]
        t2 = time.perf_counter()
        results['per_res'] = peri_model.predict(
            source=img,
            imgsz=per_cfg.get("default_imgsz", 640),
            conf=per_cfg.get("default_confidence", 0.1),
            iou=per_cfg.get("default_iou", 0.5),
            verbose=False,
        )[0]
        results['per_time'] = time.perf_counter() - t2
        print(f"  [PERF][{self.device.upper()}] periapical inference: {results['per_time']:.3f}s")

        # 5. IAC (Ensemble)
        iac_cfg = self.model_cfg["iac"]
        t_iac = time.perf_counter()
        
        # Preprocessing Variants
        variants = [("original", img)]
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl_img = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
            variants.append(("clahe", cl_img))
        except: pass
        
        try:
            gamma = 1.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gam_img = cv2.LUT(img, table)
            variants.append(("gamma", gam_img))
        except: pass
        
        best_iac_res = None
        best_score = -1.0
        
        for name, v_img in variants:
            res = iac_model.predict(
                source=v_img,
                imgsz=iac_cfg.get("default_imgsz", 1024),
                conf=0.05,
                iou=iac_cfg.get("default_iou", 0.5),
                retina_masks=True,
                verbose=False,
            )[0]
            
            score = float(res.boxes.conf.max()) if res.boxes is not None and len(res.boxes) > 0 else 0
            if score > best_score:
                best_score = score
                best_iac_res = res
        
        results['iac_res'] = best_iac_res if best_iac_res else iac_model.predict(img, verbose=False)[0]
        results['iac_time'] = time.perf_counter() - t_iac
        print(f"  [PERF][{self.device.upper()}] iac inference (ensemble): {results['iac_time']:.3f}s")
        
        return results

    def _process_detections(self, img: np.ndarray, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw inference results into structured business objects.
        Handles Rules Engine, conflict resolution, and measurements.
        """
        seg_res = results['seg_res']
        car_res = results['car_res']
        per_res = results['per_res']
        iac_res = results['iac_res']
        # aux_boxes_data is used internally if needed, but not heavily in post-processing 
        # unless hybrid logic is re-enabled. Here we use it if needed.
        
        h, w = img.shape[:2]
        
        # 1. Extract Tooth Boxes (Legacy Logic)
        tooth_only_boxes = []
        if seg_res.boxes is not None and seg_res.masks is not None:
            for idx, coords in enumerate(seg_res.masks.xy):
                label = seg_res.names[int(seg_res.boxes.cls[idx])]
                if str(label).isdigit():
                    cnts = calc.get_squeeze_points(coords, copy.deepcopy(seg_res.orig_img), 0)
                    if cnts:
                        bb = calc._get_bb_from_contour(cnts[0])
                        tooth_only_boxes.append((label, bb))

        # 2. Dental Zone & Spatial Logic (Delegated)
        zone_info = self.tooth_logic.calculate_dental_zone(tooth_only_boxes, img.shape)
        
        # Helper for spatial validation (Delegated)
        def _is_spatial_valid(box, label_str=""):
            return self.tooth_logic.is_spatial_valid(box, label_str, zone_info, img.shape)

        # 3. PBL (Bonelevel/CEJ)
        t3 = time.perf_counter()
        bl_canvas = np.zeros_like(img)
        pbl_dict = {}
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
            print(f"  [PERF] PBL calculation: {time.perf_counter()-t3:.3f}s")
        except Exception as e:
            print(f"  [WARN] PBL calculation failed: {e}")

        # 4. Implant Metrics
        implant_metrics = {}
        if seg_res.boxes is not None:
            ppm = float(os.getenv("DIAMETER_PIXEL_TO_MM", 0.1))
            for i, bbox in enumerate(seg_res.boxes.xyxy):
                lbl = str(seg_res.names[int(seg_res.boxes.cls[i])]).lower()
                bbox_np = bbox.cpu().numpy()
                if not _is_spatial_valid(bbox_np, lbl): continue

                if "implant" in lbl:
                    if seg_res.masks and len(seg_res.masks.xy) > i:
                        contour = np.array(seg_res.masks.xy[i], dtype=np.float32)
                        d_mm, l_mm, (p1, p2) = calc.calc_implant_metrics(contour, mm_per_px=ppm)
                        db, lb = round(d_mm * 2.0) / 2.0, round(l_mm * 2.0) / 2.0
                        p1_t = tuple(p1) if p1 is not None else None
                        p2_t = tuple(p2) if p2 is not None else None
                        implant_metrics[f"idx_{i}"] = (d_mm, l_mm, db, lb, p1_t, p2_t)

        # 5. Build Observations for Rules Engine
        observations = []
        if seg_res.boxes is not None:
            for idx in range(len(seg_res.boxes)):
                lbl = str(seg_res.names[int(seg_res.boxes.cls[idx])]).lower()
                box = seg_res.boxes.xyxy[idx].cpu().numpy().tolist()
                conf = float(seg_res.boxes.conf[idx])
                
                # Spatial Filter
                if not _is_spatial_valid(seg_res.boxes.xyxy[idx].cpu().numpy(), lbl):
                    continue

                obj_type = "other"
                label_hint = None
                if str(lbl).isdigit():
                    obj_type = "natural"
                    label_hint = lbl
                elif "implant" in lbl:
                    obj_type = "implant"
                elif any(k in lbl for k in ["crown", "bridge", "prosthesis", "pontic"]):
                    obj_type = "crown"
                
                if obj_type in ["natural", "implant", "crown"]:
                    contour = []
                    if seg_res.masks is not None:
                        try: contour = seg_res.masks.xy[idx].tolist()
                        except: pass
                    
                    observations.append({
                        "id": idx,
                        "type": obj_type,
                        "box": box,
                        "cx": (box[0]+box[2])/2,
                        "cy": (box[1]+box[3])/2,
                        "label_hint": label_hint,
                        "conf": conf,
                        "contour": contour
                    })

        # 6. Caries Objects (Pre-filtering)
        caries_objects = []
        if car_res.boxes is not None:
            for i, box in enumerate(car_res.boxes.xyxy):
                lbl = car_res.names[int(car_res.boxes.cls[i])]
                conf = float(car_res.boxes.conf[i])
                bbox_np = box.cpu().numpy()
                if not _is_spatial_valid(bbox_np, str(lbl)): continue
                
                caries_objects.append({
                    "id": f"caries_{i}",
                    "box": bbox_np.tolist(),
                    "label": str(lbl),
                    "confidence": conf,
                })

        # 7. Rules Engine Execution
        from services.pano_rules_engine import RulesEngine
        engine = RulesEngine(img_width=w, img_height=h)
        rule_result = engine.run(observations, caries_objects)

        # 8. Format Final Teeth Objects
        final_teeth_objects = []
        assigned_ids = set()
        
        # (1) Assigned
        for fdi, slot in rule_result['slots'].items():
            if slot['object_id'] is not None:
                obj = slot['candidates'][0]
                assigned_ids.add(obj['id'])
                t_obj = {
                    "id": f"tooth_{fdi}", 
                    "box": obj['box'],
                    "type": obj['type'],
                    "tooth_label": str(fdi),
                    "confidence": obj['conf'],
                    "status": slot['status'],
                    "geometry": "box",
                    "contour": obj.get('contour', [])
                }
                final_teeth_objects.append(t_obj)

        # (2) Unassigned
        for obj in observations:
            if obj['id'] not in assigned_ids:
                final_teeth_objects.append({
                    "id": f"obj_{obj['id']}",
                    "box": obj['box'],
                    "type": obj['type'],
                    "tooth_label": None,
                    "confidence": obj['conf'],
                    "status": "unassigned",
                    "geometry": "box",
                    "contour": obj.get('contour', [])
                })

        # 9. Format Final Caries Objects & Inject Findings
        final_caries_objects = []
        c_assignment = rule_result['caries_assignment']
        for c_obj in caries_objects:
            res = c_assignment.get(c_obj['id'])
            new_c = c_obj.copy()
            if res:
                new_c['status'] = res['status']
                if res.get('assigned_to'):
                    new_c['assigned_tooth'] = res['assigned_to']
                    # Inject finding
                    for t_obj in final_teeth_objects:
                        if t_obj['tooth_label'] == res['assigned_to']:
                            t_obj['caries'] = True
                            if 'findings' not in t_obj: t_obj['findings'] = []
                            t_obj['findings'].append({'type': 'caries', 'box': new_c['box'], 'conf': new_c['confidence']})
                            break
            else:
                new_c['status'] = 'unassigned'
            final_caries_objects.append(new_c)

        # 10. Periapical Processing (Strict Intersection)
        periapical_objects = []
        if per_res.boxes is not None:
            for i, box in enumerate(per_res.boxes.xyxy):
                lbl = per_res.names[int(per_res.boxes.cls[i])]
                conf = float(per_res.boxes.conf[i])
                bbox_np = box.cpu().numpy()
                if not _is_spatial_valid(bbox_np, "periapical"): continue
                
                x1, y1, x2, y2 = bbox_np.tolist()
                
                best_tooth = None
                max_inter_area = 0.0
                for t_obj in final_teeth_objects:
                    if not t_obj['box']: continue
                    tx1, ty1, tx2, ty2 = t_obj['box']
                    ix1, iy1 = max(x1, tx1), max(y1, ty1)
                    ix2, iy2 = min(x2, tx2), min(y2, ty2)
                    if ix2 > ix1 and iy2 > iy1:
                        area = (ix2 - ix1) * (iy2 - iy1)
                        if area > max_inter_area:
                            max_inter_area = area
                            best_tooth = t_obj
                
                if best_tooth and max_inter_area > 0:
                    best_tooth['periapical'] = True
                    if 'findings' not in best_tooth: best_tooth['findings'] = []
                    best_tooth['findings'].append({'type': 'periapical', 'box': [x1, y1, x2, y2], 'conf': conf})
                    
                    periapical_objects.append({
                        "id": f"peri_{i}",
                        "box": [x1, y1, x2, y2],
                        "label": "Periapical Lesion",
                        "confidence": conf,
                        "status": "confirmed",
                        "assigned_tooth": best_tooth['tooth_label'],
                        "geometry": "box"
                    })

        # 11. Prepare Masks (Nerve/Sinus) for Safety Logic
        nerve_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        sinus_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if iac_res.masks is not None:
            for idx, m in enumerate(iac_res.masks.xy):
                if len(m) == 0: continue
                pts = np.int32([m])
                cls = int(iac_res.boxes.cls[idx])
                name = iac_res.names[cls].lower()
                is_sinus = "sinus" in name
                is_nerve = "nerve" in name or "canal" in name or "iac" in name
                if not is_sinus and not is_nerve:
                    if np.mean(m[:, 1]) < h / 2: is_sinus = True
                    else: is_nerve = True
                if is_sinus: cv2.fillPoly(sinus_mask, [pts], 255)
                elif is_nerve: cv2.fillPoly(nerve_mask, [pts], 255)

        # Calibration
        mm_per_px = self._get_calibration_scale(seg_res, final_teeth_objects)

        # Odontogram Map Init & Nerve Overlap
        odontogram_map = {}
        present_fdis = set()
        for tooth in final_teeth_objects:
            lbl = tooth.get('tooth_label')
            if not lbl: continue
            present_fdis.add(int(lbl))
            if lbl not in odontogram_map: odontogram_map[lbl] = []
            t_type = tooth['type']
            if "implant" in t_type: odontogram_map[lbl].append("implant")
            if "crown" in t_type: odontogram_map[lbl].append("crown")
            if tooth.get('caries'): odontogram_map[lbl].append("caries")
            
            # Nerve overlap (Lower Jaw)
            mask_pts = tooth.get('contour', [])
            if len(mask_pts) > 0 and int(lbl[0]) in [3, 4]:
                overlap_px = self._check_overlap(mask_pts, nerve_mask, (h, w))
                if overlap_px > 0:
                    tooth['nerve_overlap'] = True
                    tooth['overlap_area_mm2'] = overlap_px * (mm_per_px**2)

        # Arch Curve Logic (Symmetric Fitting - Feature Parity)
        def fit_symmetric(centers, mid_x):
            if not centers: return None
            vc = np.array(centers)
            mirrored = np.column_stack((2*mid_x - vc[:,0], vc[:,1]))
            combined = np.vstack((vc, mirrored))
            try: return np.poly1d(np.polyfit(combined[:,0], combined[:,1], 2))
            except: return None

        mid_x_u, mid_x_l = w/2.0, w/2.0
        u_incs = [t for t in final_teeth_objects if str(t.get('tooth_label')) in ['11', '21']]
        if u_incs: mid_x_u = float(np.mean([(t['box'][0]+t['box'][2])/2 for t in u_incs]))
        l_incs = [t for t in final_teeth_objects if str(t.get('tooth_label')) in ['31', '41']]
        if l_incs: mid_x_l = float(np.mean([(t['box'][0]+t['box'][2])/2 for t in l_incs]))

        upper_centers = []
        lower_centers = []
        for t in final_teeth_objects:
            lbl = t.get('tooth_label')
            if not lbl: continue
            q = int(lbl[0])
            cx, cy = (t['box'][0]+t['box'][2])/2, (t['box'][1]+t['box'][3])/2
            if q in [1, 2]: upper_centers.append([cx, cy])
            elif q in [3, 4]: lower_centers.append([cx, cy])
        
        y_poly_fn_upper = fit_symmetric(upper_centers, mid_x_u) if len(upper_centers) >= 2 else None
        y_poly_fn_lower = fit_symmetric(lower_centers, mid_x_l) if len(lower_centers) >= 2 else None

        # Missing Tooth Detection (Delegated)
        tooth_logic_context = {
            'present_fdis': present_fdis,
            'final_teeth_objects': final_teeth_objects,
            'nerve_mask': nerve_mask,
            'sinus_mask': sinus_mask,
            'mm_per_px': mm_per_px,
            'y_poly_fn_upper': y_poly_fn_upper,
            'y_poly_fn_lower': y_poly_fn_lower,
            'pbl_dict': pbl_dict
        }
        
        missing_teeth_list, odo_updates, tooth_axes = self.tooth_logic.find_missing_teeth(tooth_logic_context)
        
        # [FIX] Calculate Nerve Safety for Present Teeth
        # Update context with new missing/odontogram data if needed, but mainly we need masks
        tooth_logic_context['final_teeth_objects'] = final_teeth_objects # Ensure updated list
        self.tooth_logic.calculate_nerve_safety(tooth_logic_context)

        # 12. Implant Metrics
        implant_metrics = []
        for t in final_teeth_objects:
            if "implant" in t['type'] or "fixture" in t['type']:
                try:
                    contour = t.get('contour', [])
                    if len(contour) > 0:
                         dia, length, axis = calc.calc_implant_metrics(np.array(contour).reshape(-1, 2), mm_per_px)
                         implant_metrics.append({
                             "id": t['id'],
                             "label": t['tooth_label'],
                             "diameter_mm": dia,
                             "length_mm": length,
                             "axis": axis
                         })
                         # Also store in tooth object for report
                         t['implant_meta'] = {"diameter": dia, "length": length}
                except Exception: pass
        
        for f, s in odo_updates.items():
            if f not in odontogram_map: odontogram_map[f] = []
            odontogram_map[f].append(s)

        return {
            "final_teeth_objects": final_teeth_objects,
            "final_caries_objects": final_caries_objects,
            "periapical_objects": periapical_objects,
            "missing_teeth_list": missing_teeth_list,
            "implant_metrics": implant_metrics,
            "pbl_dict": pbl_dict,
            "odontogram_map": odontogram_map,
            "mm_per_px": mm_per_px,
            "nerve_mask": nerve_mask, # For visualization
            "sinus_mask": sinus_mask,
            "bl_canvas": bl_canvas # For visualization overlay
        }

    def _draw_visuals(self, base: np.ndarray, inf_res: Dict[str, Any], proc_res: Dict[str, Any], inner_rect=None) -> np.ndarray:
        """
        Produce final visualization overlay.
        """
        overlay = base.copy()
        h, w = base.shape[:2]
        if inner_rect is None:
             inner_rect = {"x1": 0, "x2": w, "y1": 0, "y2": h}

        # 1. Segmentation
        # Fixed neon color (Cyan: 255, 255, 0 in BGR is (0, 255, 255) actually? 
        # OpenCV uses BGR. Yellow is (0, 255, 255). Cyan is (255, 255, 0).
        # User requested "Neon". Cyan/Yellow usually. PanoVisualizer has colors.
        # Let's map all indices to Cyan for teeth?
        seg_res = inf_res['seg_res']
        colors = {i: (255, 255, 0) for i in range(len(seg_res.boxes) if seg_res.boxes is not None else 0)} # Cyan
        overlay = self.visualizer.overlay_seg(overlay, seg_res, colors, inner_rect, thickness=2)

        # 2. Caries & Periapical
        # Need exclusion boxes (Implants) to avoid drawing boxes on implants?
        # Logic was: implant_exclusion_boxes.
        # We can re-derive quickly or just draw all.
        # Strict user request: "Prevent Caries/Periapical on Implants".
        # We assume proc_res['final_caries_objects'] handles filtering?
        # Actually _process_detections filtered list construction, but overlays often use raw model output.
        # Let's use the Processed Result lists for cleaner visualization?
        # But PanoVisualizer expects 'result' object (YOLO Result).
        
        # We will use raw overlay but pass exclusion.
        implant_boxes = []
        if seg_res.boxes is not None:
             for i, cls in enumerate(seg_res.boxes.cls):
                 lbl = seg_res.names[int(cls)].lower()
                 if "implant" in lbl: 
                     implant_boxes.append(seg_res.boxes.xyxy[i].cpu().numpy().tolist())

        caries_labels = inf_res['car_res'].names
        overlay, _ = self.visualizer.overlay_det(overlay, inf_res['car_res'], caries_labels, (0, 0, 255), inner_rect, exclusion_boxes=implant_boxes)
        
        peri_labels = inf_res['per_res'].names
        overlay, _ = self.visualizer.overlay_det(overlay, inf_res['per_res'], peri_labels, (0, 200, 0), inner_rect, exclusion_boxes=implant_boxes)
        
        # 3. IAC (Nerve & Sinus)
        # iac_colors = {i: (255, 0, 255) for i in range(10)}
        # overlay = self.visualizer.overlay_seg(overlay, inf_res['iac_res'], iac_colors, inner_rect)
        
        # Explicit Binary Mask Overlay
        nerve_mask = proc_res.get('nerve_mask')
        sinus_mask = proc_res.get('sinus_mask')
        
        # Nerve: Magenta (255, 0, 255)
        if nerve_mask is not None:
            overlay = self.visualizer.overlay_mask(overlay, nerve_mask, (255, 0, 255), alpha=0.3)
            
        # Sinus: Green (0, 255, 0) - Distinct from Nerve
        if sinus_mask is not None:
             overlay = self.visualizer.overlay_mask(overlay, sinus_mask, (0, 255, 0), alpha=0.3)

        # 4. PBL (Bonelevel) - Add Canvas
        if os.getenv("SKIP_PBL_OVERLAY", "0") != "1":
            bl_canvas = proc_res.get('bl_canvas')
            if bl_canvas is not None:
                overlay = cv2.add(overlay, bl_canvas)

        # 5. Tooth Labels
        teeth_objs = proc_res['final_teeth_objects']
        overlay = self.visualizer.draw_tooth_labels(overlay, teeth_objs)

        # 6. Safety Guides (Implant / Nerve Dist)
        # Collect objects with guides
        objects_with_guides = []
        # Missing teeth guides
        objects_with_guides.extend(proc_res.get('missing_teeth_list', []))
        
        # Present teeth nerve distance (stored as 'nerve_dist_line')
        # Adapt format for visualizer
        for t in teeth_objs:
             if t.get('nerve_dist_line'):
                 # Create a wrapper object that visualizer expects
                 objects_with_guides.append({
                     'safety_guide': {
                         'line_coords': t['nerve_dist_line'],
                         'dist_mm': t.get('nerve_dist_mm', 0)
                     }
                 })
        
        overlay = self.visualizer.draw_safety_guides(overlay, objects_with_guides)
        
        return overlay

    def run(self, image_path: Path, out_dir: Path) -> Dict[str, Any]:
        """
        Main pipeline orchestration.
        """
        _ensure_dir(out_dir)

        # 0. DICOM Metadata
        if image_path.suffix.lower() in (".dcm", ".dicom"):
            meta = extract_dicom_meta(image_path)
            if meta:
                try:
                    with open(out_dir / "dicom_meta.json", "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        t_start = time.perf_counter()
        img = load_image_any(image_path)
        base = img.copy()
        
        # 1. Inference
        inference_results = self._run_inference(img)
        
        # 2. Process Business Logic
        processed_results = self._process_detections(img, inference_results)
        
        # 3. Visualization
        h, w = img.shape[:2]
        inner_rect = {"x1": 0, "x2": w, "y1": 0, "y2": h} # Standard full view
        overlay = self._draw_visuals(base, inference_results, processed_results, inner_rect)
        
        # 4. Save Outputs
        cv2.imwrite(str(out_dir / "overlay.jpg"), overlay)
        if processed_results.get('bl_canvas') is not None:
             cv2.imwrite(str(out_dir / "bl_viz.png"), processed_results['bl_canvas'])
        
        # 5. Construct Response
        def to_relative(p):
            try: return str(Path(p).relative_to(Path.cwd())).replace('\\', '/')
            except: return str(p).replace('\\', '/')
            
        print(f"  [PERF] Pipeline finished in {time.perf_counter()-t_start:.3f}s")
        
        return {
            "image_path": to_relative(image_path),
            "overlay_path": to_relative(out_dir / "overlay.jpg"),
            "teeth": processed_results['final_teeth_objects'],
            "teeth_objects": processed_results['final_teeth_objects'], 
            "caries": processed_results['final_caries_objects'],
            "caries_objects": processed_results['final_caries_objects'],
            "periapical": processed_results['periapical_objects'],
            "periapical_objects": processed_results['periapical_objects'],
            "missing_teeth": processed_results['missing_teeth_list'],
            "implant_metrics": processed_results['implant_metrics'],
            "bonelevel": processed_results['pbl_dict'],
            "odontogram_map": processed_results['odontogram_map'],
            "mm_per_px": processed_results['mm_per_px'],
            "nerve_contours": self._get_contours(processed_results['nerve_mask']),
            "sinus_contours": self._get_contours(processed_results['sinus_mask'])
        }

    def _get_contours(self, mask):
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = []
        for c in cnts: res.append(c.reshape(-1, 2).tolist())
        return res


import os
import math
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

AUTO_WINDOW_THRESHOLD_RATIOS = (0.005, 0.12, 0.28, 0.45, 0.68, 0.88, 0.995)
AUTO_WINDOW_MAX_SAMPLES = 32768

def extract_dicom_meta(path: Path) -> Dict[str, Any]:
    """Safely extract a few useful DICOM fields (best effort)."""
    try:
        import pydicom
        from pydicom.multival import MultiValue
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
            # Keep PersonName and other scalar VRs as strings.
            # Only true DICOM multi-values should become lists.
            if isinstance(v, MultiValue):
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

def _sample_scalar_data(source: np.ndarray, sample_limit: int = AUTO_WINDOW_MAX_SAMPLES) -> np.ndarray:
    values = np.asarray(source, dtype=np.float32).reshape(-1)
    if values.size <= sample_limit:
        return values
    stride = max(1, values.size // sample_limit)
    return values[::stride]

def _percentile_from_sorted(sorted_values: np.ndarray, ratio: float) -> float:
    if sorted_values.size == 0:
        return 0.0
    clamped_ratio = float(np.clip(ratio, 0.0, 1.0))
    position = clamped_ratio * (sorted_values.size - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower = float(sorted_values[lower_index])
    upper = float(sorted_values[upper_index])
    if lower_index == upper_index:
        return lower
    weight = position - lower_index
    return lower + (upper - lower) * weight

def _estimate_auto_window(raw_pixels: np.ndarray, slope: float = 1.0, intercept: float = 0.0) -> Optional[Tuple[float, float]]:
    sampled = _sample_scalar_data(raw_pixels)
    if sampled.size < 16:
        return None

    sampled = sampled * slope + intercept
    sampled = sampled[np.isfinite(sampled)]
    if sampled.size < 16:
        return None

    sorted_values = np.sort(sampled)
    thresholds = np.asarray(
        [_percentile_from_sorted(sorted_values, ratio) for ratio in AUTO_WINDOW_THRESHOLD_RATIOS],
        dtype=np.float32,
    )

    soft_bucket = sampled[(sampled >= thresholds[1]) & (sampled <= thresholds[3])]
    bone_bucket = sampled[(sampled >= thresholds[4]) & (sampled <= thresholds[6])]
    soft_std = float(np.std(soft_bucket)) if soft_bucket.size > 1 else 0.0
    bone_std = float(np.std(bone_bucket)) if bone_bucket.size > 1 else 0.0

    expand_width = float((thresholds[3] - thresholds[1]) * 0.15 + soft_std)
    wnd_lower = float(thresholds[1] - expand_width)
    wnd_upper = float(min(thresholds[5] - bone_std * 0.75, thresholds[6]))
    if not np.isfinite(wnd_lower) or not np.isfinite(wnd_upper) or wnd_upper <= wnd_lower:
        return None

    global_low = float(thresholds[0])
    global_high = float(thresholds[6])
    safe_lower = float(np.clip(wnd_lower, global_low, global_high))
    safe_upper = float(np.clip(wnd_upper, global_low, global_high))
    if not np.isfinite(safe_lower) or not np.isfinite(safe_upper) or safe_upper <= safe_lower:
        return None

    width = max(1.0, safe_upper - safe_lower)
    level = safe_lower + width / 2.0
    return level, width

def _apply_window(arr: np.ndarray, level: float, width: float) -> np.ndarray:
    half_width = max(float(width), 1.0) / 2.0
    lower = float(level) - half_width
    upper = float(level) + half_width
    return np.clip(arr, lower, upper)

def _parse_dicom_window(ds) -> Optional[Tuple[float, float]]:
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is None or ww is None:
        return None
    try:
        if hasattr(wc, "__len__"):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if hasattr(ww, "__len__"):
            ww = float(ww[0])
        else:
            ww = float(ww)
    except Exception:
        return None
    if not np.isfinite(wc) or not np.isfinite(ww) or ww <= 0:
        return None
    return wc, ww

def load_image_any(path: Path, return_meta: bool = False, use_auto_window: bool = False):
    """
    Load common image formats or DICOM (if pydicom available).
    Returns BGR uint8 image, raises if cannot be read.
    If return_meta=True, returns (img, is_volume).
    """
    is_volume = False
    # Safe Image Load (Support Windows/Unicode paths)
    try:
        abs_path = os.path.abspath(str(path))
        stream = open(abs_path, "rb")
        bytes_data = bytearray(stream.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        if img is not None:
             return (img, False) if return_meta else img
    except Exception:
        pass

    if path.suffix.lower() in (".dcm", ".dicom"):
        try:
            import pydicom
        except ImportError:
            raise ValueError(f"Failed to load image: {path} (pydicom not installed for DICOM)")
        ds = pydicom.dcmread(str(path))
        _arr_orig = ds.pixel_array.astype(np.float32)
        
        # [NEW] Handle 3D Volumes: Take representative slice for 2D logic
        is_volume = False
        if _arr_orig.ndim == 3 and _arr_orig.shape[0] > 1 and _arr_orig.shape[2] != 3:
            is_volume = True
            mid_idx = _arr_orig.shape[0] // 2
            print(f"  [LOAD] 3D Volume detected ({_arr_orig.shape}). Using slice {mid_idx}.")
            arr = _arr_orig[mid_idx]
        else:
            arr = _arr_orig

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        window = _estimate_auto_window(_arr_orig[mid_idx] if is_volume else _arr_orig, slope=slope, intercept=intercept) if use_auto_window else None
        if window is not None:
            arr = _apply_window(arr, window[0], window[1])
        else:
            dicom_window = _parse_dicom_window(ds)
            if dicom_window is not None:
                arr = _apply_window(arr, dicom_window[0], dicom_window[1])

        arr = arr - arr.min()
        maxv = arr.max()
        if maxv > 0:
            arr = arr / maxv * 255.0
        img = arr.astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        return (img, is_volume) if return_meta else img

    raise ValueError(f"Failed to load image: {path}")

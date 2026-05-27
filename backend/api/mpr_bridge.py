import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request, send_file

try:
    import pydicom
except Exception:  # pragma: no cover - runtime response reports the missing dependency.
    pydicom = None

try:
    import numpy as np
except Exception:  # pragma: no cover - runtime response reports the missing dependency.
    np = None


mpr_bridge_api = Blueprint("mpr_bridge_api", __name__)

DEFAULT_LIBRARY_ROOT = Path(
    os.environ.get(
        "DENTAL_IMGDATA_LIBRARY",
        str(Path(__file__).resolve().parents[1] / "retrieved_dicom"),
    )
)
FALLBACK_LIBRARY_ROOT = Path(r"C:\interface\case")
SKIP_STACK_MODALITIES = {"SC", "PR", "SR", "KO", "DOC", "SEG", "RTSTRUCT", "RTPLAN"}
SKIP_SOP_CLASSES = {
    "1.2.840.10008.5.1.4.1.1.7",
    "1.2.840.10008.5.1.4.1.1.11.1",
}

_active_root: Optional[Path] = None
_focused_entry_id: Optional[str] = None
_focused_series_uid: Optional[str] = None
_library_root_override: Optional[Path] = None
_series_cache: Dict[str, List[Path]] = {}
_instances_cache: Dict[str, List[dict]] = {}
_instance_path_cache: Dict[Tuple[str, str], Path] = {}
_render_params_cache: Dict[str, dict] = {}

RENDER_THRESHOLD_BINS = 256
RENDER_HISTOGRAM_MAX_VALUES = 8_000_000


def _library_root() -> Path:
    if _library_root_override and _library_root_override.exists():
        return _library_root_override
    if DEFAULT_LIBRARY_ROOT.exists():
        return DEFAULT_LIBRARY_ROOT
    return FALLBACK_LIBRARY_ROOT


def _candidate_library_roots() -> List[Path]:
    roots: List[Path] = []
    for root in [DEFAULT_LIBRARY_ROOT, FALLBACK_LIBRARY_ROOT]:
        if not root.exists() or not root.is_dir():
            continue
        resolved = root.resolve()
        if all(existing.resolve() != resolved for existing in roots):
            roots.append(root)
    return roots or [_library_root()]


def _safe_str(value, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text:
        try:
            repaired = text.encode("latin-1").decode("cp949")
            if any("\uac00" <= char <= "\ud7a3" for char in repaired):
                text = repaired
        except Exception:
            pass
    return text or default


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value).strip().split("\\")[0]))
    except Exception:
        return None


def _parse_first_float(value, default: float) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).strip().split("\\")[0])
    except Exception:
        return default
    try:
        return int(str(value).strip().split("\\")[0])
    except Exception:
        return None


def _read_header(path: Path):
    if pydicom is None:
        return None
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def _read_dataset(path: Path):
    if pydicom is None:
        return None
    try:
        return pydicom.dcmread(str(path), force=True)
    except Exception:
        return None


def _is_dicom(path: Path) -> bool:
    ds = _read_header(path)
    return bool(ds is not None and getattr(ds, "SeriesInstanceUID", None))


def _has_dicom_in_tree(root: Path, max_file_checks: int = 200000) -> bool:
    checked = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        checked += 1
        if checked > max_file_checks:
            return False
        if _is_dicom(path):
            return True
    return False


def _has_dicom_in_immediate_files(root: Path) -> bool:
    for path in root.iterdir():
        if path.is_file() and not path.name.startswith(".") and _is_dicom(path):
            return True
    return False


def _peek_patient(root: Path) -> tuple[str, str]:
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        ds = _read_header(path)
        if ds is None or not getattr(ds, "SeriesInstanceUID", None):
            continue
        return (
            _safe_str(getattr(ds, "PatientName", ""), ""),
            _safe_str(getattr(ds, "PatientID", ""), ""),
        )
    return "", ""


def _find_entry_id_by_study_uid(root: Path, study_uid: str) -> str:
    expected_uid = str(study_uid or "").strip()
    if not expected_uid or not root.exists() or not root.is_dir():
        return ""

    resolved_root = root.resolve()
    direct_study_dir = root / expected_uid
    if direct_study_dir.exists() and direct_study_dir.is_dir():
        return expected_uid

    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        ds = _read_header(path)
        if ds is None or str(getattr(ds, "StudyInstanceUID", "") or "") != expected_uid:
            continue
        parent = path.parent.resolve()
        if resolved_root == parent or resolved_root in parent.parents:
            try:
                relative_parts = parent.relative_to(resolved_root).parts
            except ValueError:
                return ""
            return relative_parts[0] if relative_parts else "."
    return ""


def _relative_id(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _resolve_entry(root: Path, entry_id: str) -> Path:
    parts = [part for part in str(entry_id or "").replace("\\", "/").split("/") if part]
    if not parts or any(part in {".", ".."} or ".." in part for part in parts):
        raise ValueError("Invalid entryId.")
    resolved_root = root.resolve()
    target = (root.joinpath(*parts)).resolve()
    if resolved_root not in [target, *target.parents] or not target.is_dir():
        raise ValueError("Entry path not found or not under library root.")
    return target


def _series_payload(series: List[dict]) -> dict:
    return {
        "success": True,
        "count": len(series),
        "series": series,
    }


def _filter_series(series: List[dict], series_uid: Optional[str]) -> List[dict]:
    expected_uid = str(series_uid or "").strip()
    if not expected_uid:
        return series
    return [item for item in series if str(item.get("seriesInstanceUID") or "") == expected_uid]


def _active_root_matches_focus(study_uid: str, series_uid: str) -> bool:
    if not _active_root or not _active_root.exists():
        return False
    series = _filter_series(_scan_series(_active_root), series_uid or None)
    if not series:
        return False
    if str(series_uid or "").strip():
        return True
    expected_study_uid = str(study_uid or "").strip()
    if not expected_study_uid:
        return True
    return any(str(item.get("studyInstanceUID") or "") == expected_study_uid for item in series)


def _include_instance(ds) -> bool:
    modality = str(getattr(ds, "Modality", "") or "").strip().upper()
    sop_class = str(getattr(ds, "SOPClassUID", "") or "").strip()
    return modality not in SKIP_STACK_MODALITIES and sop_class not in SKIP_SOP_CLASSES


def _scan_series(root: Path) -> List[dict]:
    series: Dict[str, dict] = {}
    series_files: Dict[str, List[Path]] = {}
    instances: Dict[str, List[dict]] = {}
    instance_paths: Dict[Tuple[str, str], Path] = {}

    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        ds = _read_header(path)
        if ds is None or not getattr(ds, "SeriesInstanceUID", None):
            continue

        series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
        series_files.setdefault(series_uid, []).append(path)

        if _include_instance(ds):
            sop_uid = str(getattr(ds, "SOPInstanceUID", "") or "")
            if sop_uid:
                instance_number = _safe_int(getattr(ds, "InstanceNumber", 0)) or 0
                instances.setdefault(series_uid, []).append(
                    {
                        "sopInstanceUID": sop_uid,
                        "instanceNumber": instance_number,
                    }
                )
                instance_paths[(series_uid, sop_uid)] = path

        if series_uid not in series:
            series[series_uid] = {
                "seriesInstanceUID": series_uid,
                "studyInstanceUID": str(getattr(ds, "StudyInstanceUID", "") or ""),
                "patientID": _safe_str(getattr(ds, "PatientID", "Unknown"), "Unknown"),
                "patientName": _safe_str(getattr(ds, "PatientName", "Unknown"), "Unknown"),
                "patientSex": _safe_str(getattr(ds, "PatientSex", ""), ""),
                "patientAge": _safe_str(getattr(ds, "PatientAge", ""), ""),
                "studyDate": _safe_str(getattr(ds, "StudyDate", ""), ""),
                "modality": str(getattr(ds, "Modality", "Unknown") or "Unknown"),
                "seriesDescription": _safe_str(getattr(ds, "SeriesDescription", ""), ""),
                "gantryTilt": _safe_float(getattr(ds, "GantryDetectorTilt", None)),
                "tubeCurrent": _safe_float(getattr(ds, "XRayTubeCurrent", None)),
                "tubeVoltage": _safe_float(getattr(ds, "KVP", None)),
                "seriesNumber": _safe_int(getattr(ds, "SeriesNumber", None)),
                "instanceCount": 0,
            }
        series[series_uid]["instanceCount"] += 1

    for items in instances.values():
        items.sort(key=lambda item: item.get("instanceNumber", 0))

    global _series_cache, _instances_cache, _instance_path_cache
    _series_cache = series_files
    _instances_cache = instances
    _instance_path_cache = instance_paths
    _render_params_cache.clear()

    return sorted(
        series.values(),
        key=lambda item: (
            item.get("seriesNumber") if item.get("seriesNumber") is not None else 999999,
            item.get("seriesDescription") or "",
        ),
    )


def _series_instance_paths(series_uid: str) -> List[Path]:
    items = _instances_cache.get(series_uid)
    if items is None and _active_root and _active_root.exists():
        _scan_series(_active_root)
        items = _instances_cache.get(series_uid)
    if items:
        paths = []
        for item in items:
            sop_uid = str(item.get("sopInstanceUID") or "")
            path = _instance_path_cache.get((series_uid, sop_uid))
            if path and path.exists():
                paths.append(path)
        if paths:
            return paths
    return [path for path in _series_cache.get(series_uid, []) if path.exists()]


def _safe_scaled_range(min_value: float, max_value: float, slope: float, intercept: float) -> Tuple[float, float]:
    try:
        mn = float(min_value) * float(slope) + float(intercept)
        mx = float(max_value) * float(slope) + float(intercept)
    except Exception:
        return -1024.0, 3071.0
    if np is not None and (not np.isfinite(mn) or not np.isfinite(mx)):
        return -1024.0, 3071.0
    if mx < mn:
        mn, mx = mx, mn
    if mx == mn:
        return mn - 1024.0, mn + 3071.0
    return mn, mx


def _fallback_bone_thresholds(mn: float, mx: float) -> List[float]:
    span = max(float(mx) - float(mn), 1.0)
    if mn <= -300.0 <= 1000.0 <= mx:
        return [-300.0, 300.0, 1000.0]
    return [mn + span * 0.25, mn + span * 0.50, mn + span * 0.75]


def _multi_otsu_from_histogram(histogram: "np.ndarray") -> Optional[List[int]]:
    total = float(histogram.sum())
    if total <= 0:
        return None
    probability = histogram.astype(np.float64) / total
    indices = np.arange(len(histogram), dtype=np.float64)
    cumulative_weight = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * indices)

    def score(start: int, end: int) -> float:
        if end < start:
            return 0.0
        weight = cumulative_weight[end] - (cumulative_weight[start - 1] if start > 0 else 0.0)
        if weight <= 0:
            return 0.0
        mean = (cumulative_mean[end] - (cumulative_mean[start - 1] if start > 0 else 0.0)) / weight
        return float(weight * mean * mean)

    best_score = float("-inf")
    best = [1, 2, 3]
    bin_count = len(histogram)
    for t0 in range(1, bin_count - 3):
        for t1 in range(t0 + 1, bin_count - 2):
            base = score(0, t0) + score(t0 + 1, t1)
            for t2 in range(t1 + 1, bin_count - 1):
                current = base + score(t1 + 1, t2) + score(t2 + 1, bin_count - 1)
                if current > best_score:
                    best_score = current
                    best = [t0, t1, t2]
    return best


def _compute_render_params(series_uid: str) -> dict:
    if np is None:
        raise RuntimeError("numpy is not installed.")
    if pydicom is None:
        raise RuntimeError("pydicom is not installed.")
    if series_uid in _render_params_cache:
        return _render_params_cache[series_uid]

    paths = _series_instance_paths(series_uid)
    if not paths:
        raise FileNotFoundError(f"No DICOM files found for series: {series_uid}")

    datasets = []
    min_npv = float("inf")
    max_npv = float("-inf")
    total_values = 0

    for path in paths:
        ds = _read_dataset(path)
        if ds is None:
            continue
        try:
            pixels = np.asarray(ds.pixel_array)
        except Exception as exc:
            raise RuntimeError(f"Cannot decode DICOM pixels for render params: {path.name}") from exc
        if pixels.size == 0:
            continue
        datasets.append((ds, pixels))
        min_npv = min(min_npv, float(pixels.min()))
        max_npv = max(max_npv, float(pixels.max()))
        total_values += int(pixels.size)

    if not datasets or not np.isfinite(min_npv) or not np.isfinite(max_npv):
        raise RuntimeError("No scalar pixel data available for render params.")

    first_ds = datasets[0][0]
    default_intercept = 0.0 if min_npv < 0 else -1024.0
    slope = _parse_first_float(getattr(first_ds, "RescaleSlope", None), 1.0)
    intercept = _parse_first_float(getattr(first_ds, "RescaleIntercept", None), default_intercept)
    mn, mx = _safe_scaled_range(min_npv, max_npv, slope, intercept)
    raw_span = float(max_npv) - float(min_npv)
    if raw_span <= 0:
        thresholds = _fallback_bone_thresholds(mn, mx)
    else:
        histogram = np.zeros(RENDER_THRESHOLD_BINS, dtype=np.int64)
        sample_stride = max(1, total_values // RENDER_HISTOGRAM_MAX_VALUES)
        low_cutoff = min_npv + 0.005 * raw_span
        scale = (RENDER_THRESHOLD_BINS - 1) / raw_span

        for _, pixels in datasets:
            values = pixels.reshape(-1)
            if sample_stride > 1:
                values = values[::sample_stride]
            values = values.astype(np.float64, copy=False)
            values = np.where(values < low_cutoff, 0.0, values)
            bin_indexes = np.floor((values - min_npv) * scale).astype(np.int64)
            bin_indexes = np.clip(bin_indexes, 0, RENDER_THRESHOLD_BINS - 1)
            histogram += np.bincount(bin_indexes, minlength=RENDER_THRESHOLD_BINS)

        threshold_bins = _multi_otsu_from_histogram(histogram)
        if threshold_bins:
            thresholds = [
                float(t) * (raw_span / RENDER_THRESHOLD_BINS) * slope + intercept
                for t in threshold_bins
            ]
        else:
            thresholds = _fallback_bone_thresholds(mn, mx)

    thresholds = sorted(float(value) for value in thresholds)
    if len(thresholds) != 3 or not all(np.isfinite(value) for value in thresholds):
        thresholds = _fallback_bone_thresholds(mn, mx)
    t0, t1, t2 = thresholds
    if not (mn < t0 < t1 < t2 < mx):
        thresholds = _fallback_bone_thresholds(mn, mx)

    payload = {
        "success": True,
        "seriesUID": series_uid,
        "mn": mn,
        "mx": mx,
        "thresholds": thresholds,
        "t0": thresholds[0],
        "t1": thresholds[1],
        "t2": thresholds[2],
        "rescaleSlope": slope,
        "rescaleIntercept": intercept,
        "source": "backend-dicom",
    }
    _render_params_cache[series_uid] = payload
    return payload


def _build_imgdata_entries(root: Path) -> List[dict]:
    entries = []
    if root.exists() and root.is_dir():
        if _has_dicom_in_immediate_files(root):
            patient_name, patient_id = _peek_patient(root)
            entries.append(
                {
                    "id": ".",
                    "pathDisplay": root.name,
                    "patientName": patient_name,
                    "patientId": patient_id,
                }
            )
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not _has_dicom_in_tree(child):
                continue
            patient_name, patient_id = _peek_patient(child)
            entries.append(
                {
                    "id": _relative_id(root, child),
                    "pathDisplay": _relative_id(root, child).replace("/", " / "),
                    "patientName": patient_name,
                    "patientId": patient_id,
                }
            )
    return entries


@mpr_bridge_api.route("/imgdata-entries", methods=["GET"])
def imgdata_entries():
    global _focused_entry_id
    root = _library_root()
    if pydicom is None:
        return jsonify({"success": False, "detail": "pydicom is not installed.", "entries": []}), 500

    entries = _build_imgdata_entries(root)
    if _focused_entry_id:
        focused_entries = [entry for entry in entries if entry.get("id") == _focused_entry_id]
        if focused_entries:
            entries = focused_entries
    return jsonify(
        {
            "success": True,
            "libraryRoot": str(root),
            "libraryExists": root.exists() and root.is_dir(),
            "entries": entries,
        }
    )


@mpr_bridge_api.route("/mpr/focus-study", methods=["POST"])
def mpr_focus_study():
    global _active_root, _focused_entry_id, _focused_series_uid, _library_root_override
    root = _library_root()
    payload = request.get_json(silent=True) or {}
    entry_id = str(payload.get("entryId") or "").strip()
    study_uid = str(payload.get("studyInstanceUid") or "").strip()
    series_uid = str(payload.get("seriesInstanceUid") or payload.get("series_uid") or "").strip()

    if pydicom is None:
        return jsonify({"success": False, "detail": "pydicom is not installed."}), 500
    if not root.exists() or not root.is_dir():
        return jsonify({"success": False, "detail": "Dental IMGDATA library root does not exist."}), 404

    selected_root = root
    if entry_id:
        selected_root = root
    else:
        for candidate_root in _candidate_library_roots():
            candidate_entry_id = _find_entry_id_by_study_uid(candidate_root, study_uid)
            if candidate_entry_id:
                selected_root = candidate_root
                entry_id = candidate_entry_id
                break

    if not entry_id:
        if _active_root_matches_focus(study_uid, series_uid):
            series = _filter_series(_scan_series(_active_root), series_uid or None)
            _focused_series_uid = series_uid or _focused_series_uid
            return jsonify(
                {
                    **_series_payload(series),
                    "entryId": _focused_entry_id or "",
                    "activeRoot": str(_active_root),
                    "libraryRoot": str(_library_root()),
                    "message": "Using already focused Dental MPR case.",
                }
            )
        if _active_root and _active_root.exists():
            active_series = _filter_series(_scan_series(_active_root), _focused_series_uid)
            if active_series:
                print(
                    "[MPR focus-study] Requested study was not found; using active case instead.",
                    {
                        "requestedStudyInstanceUid": study_uid,
                        "requestedSeriesInstanceUid": series_uid,
                        "focusedSeriesUid": _focused_series_uid,
                        "activeRoot": str(_active_root),
                    },
                )
                return jsonify(
                    {
                        **_series_payload(active_series),
                        "entryId": _focused_entry_id or "",
                        "activeRoot": str(_active_root),
                        "libraryRoot": str(_library_root()),
                        "requestedStudyInstanceUid": study_uid,
                        "requestedSeriesInstanceUid": series_uid,
                        "message": "Requested study was not found; using already active Dental MPR case.",
                    }
                )
        print(
            "[MPR focus-study] Selected study was not found.",
            {
                "studyInstanceUid": study_uid,
                "seriesInstanceUid": series_uid,
                "searchedRoots": [str(root) for root in _candidate_library_roots()],
                "activeRoot": str(_active_root) if _active_root else "",
            },
        )
        return jsonify(
                {
                    "success": False,
                    "detail": "Selected study was not found in the Dental MPR library.",
                    "studyInstanceUid": study_uid,
                    "seriesInstanceUid": series_uid,
                    "searchedRoots": [str(root) for root in _candidate_library_roots()],
                    "activeRoot": str(_active_root) if _active_root else "",
                }
            )

    try:
        target = selected_root.resolve() if entry_id == "." else _resolve_entry(selected_root, entry_id)
    except ValueError as exc:
        return jsonify({"success": False, "detail": str(exc)}), 400

    _library_root_override = selected_root
    _active_root = target
    _focused_entry_id = entry_id
    _focused_series_uid = series_uid or None
    series = _scan_series(target)
    if not series:
        _active_root = None
        _focused_series_uid = None
        return jsonify({"success": False, "detail": "No DICOM series found for selected study."}), 422
    series = _filter_series(series, _focused_series_uid)
    if not series:
        _active_root = None
        _focused_entry_id = None
        _focused_series_uid = None
        return jsonify({"success": False, "detail": "Selected series was not found for selected study."}), 404

    return jsonify(
        {
            **_series_payload(series),
            "entryId": entry_id,
            "activeRoot": str(target),
            "libraryRoot": str(selected_root),
        }
    )


@mpr_bridge_api.route("/imgdata/select", methods=["POST"])
def imgdata_select():
    global _active_root, _focused_entry_id, _focused_series_uid
    root = _library_root()
    entry_id = (request.get_json(silent=True) or {}).get("entryId", "")
    entry_id = str(entry_id or "").strip()
    focused_series_uid = _focused_series_uid if entry_id == _focused_entry_id else None
    try:
        target = root.resolve() if entry_id == "." else _resolve_entry(root, entry_id)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    _active_root = target
    series = _filter_series(_scan_series(target), focused_series_uid)
    if not series:
        _active_root = None
        _focused_series_uid = None
        return jsonify({"detail": "No DICOM series found in selected folder."}), 422
    _focused_entry_id = None
    if not focused_series_uid:
        _focused_series_uid = None
    return jsonify({**_series_payload(series), "activeRoot": str(target)})


@mpr_bridge_api.route("/series", methods=["GET"])
def series():
    if not _active_root or not _active_root.exists():
        return jsonify({"success": True, "count": 0, "series": [], "message": "No case selected. Choose from DBM."})
    return jsonify(_series_payload(_filter_series(_scan_series(_active_root), _focused_series_uid)))


@mpr_bridge_api.route("/series/<series_uid>/instances", methods=["GET"])
def series_instances(series_uid: str):
    items = _instances_cache.get(series_uid)
    if items is None and _active_root and _active_root.exists():
        _scan_series(_active_root)
        items = _instances_cache.get(series_uid)
    if not items:
        return jsonify({"detail": f"No instances found for series: {series_uid}"}), 404
    return jsonify({"success": True, "seriesUID": series_uid, "count": len(items), "instances": items})


@mpr_bridge_api.route("/series/<series_uid>/render-params", methods=["GET"])
def series_render_params(series_uid: str):
    try:
        return jsonify(_compute_render_params(series_uid))
    except FileNotFoundError as exc:
        return jsonify({"success": False, "detail": str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({"success": False, "detail": str(exc)}), 422
    except Exception as exc:
        return jsonify({"success": False, "detail": f"Failed to calculate render params: {exc}"}), 500


@mpr_bridge_api.route("/dicom/<series_uid>/<sop_instance_uid>", methods=["GET"])
def dicom_file(series_uid: str, sop_instance_uid: str):
    path = _instance_path_cache.get((series_uid, sop_instance_uid))
    if path is None and _active_root and _active_root.exists():
        _scan_series(_active_root)
        path = _instance_path_cache.get((series_uid, sop_instance_uid))
    if path is None or not path.exists() or not path.is_file():
        return jsonify({"detail": f"DICOM file not found for instance: {sop_instance_uid}"}), 404
    return send_file(str(path), mimetype="application/dicom", download_name=path.name, max_age=86400)


@mpr_bridge_api.route("/health", methods=["GET"])
def mpr_health():
    root = _library_root()
    return jsonify(
        {
            "status": "ok",
            "imgdataLibrary": str(root),
            "library_exists": root.exists() and root.is_dir(),
            "activeCaseRoot": str(_active_root) if _active_root else "",
            "active_root_exists": bool(_active_root and _active_root.exists()),
        }
    )

from flask import Blueprint, jsonify, request, current_app
import json
import os
import uuid
from pathlib import Path
from services.image_loader import extract_dicom_meta
from services.file_validation import validate_browser_dicom_meta, validate_browser_raster

dicom_browser_api = Blueprint('dicom_browser_api', __name__)

# [CONFIG] Default DICOM Root Path
DICOM_ROOT = Path("C:/interface/case")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _normalize_relative_path(value: str) -> str:
    return str(value or "").replace("\\", "/").strip("/")


def _parent_directory(value: str) -> str:
    normalized = _normalize_relative_path(value)
    if "/" not in normalized:
        return ""
    return normalized.rsplit("/", 1)[0]


def _shared_prefix_score(left: str, right: str) -> int:
    left_parts = [part for part in _normalize_relative_path(left).split("/") if part]
    right_parts = [part for part in _normalize_relative_path(right).split("/") if part]
    score = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        score += 1
    return score


def _int_meta(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _load_image_sidecar_meta(path: Path) -> dict:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return {}

    json_path = path.with_suffix(".json")
    if not json_path.exists():
        return {}

    try:
        with json_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"Error parsing sidecar metadata {json_path}: {exc}")
        return {}


def _meta_value(meta: dict, *keys, default=""):
    for key in keys:
        value = meta.get(key)
        if value not in (None, "", []):
            return value
    return default


def _meta_str(meta: dict, *keys, default="") -> str:
    value = _meta_value(meta, *keys, default=default)
    if value in (None, "", []):
        return ""
    return str(value)


def _meta_number(meta: dict, *keys, default=0):
    value = _meta_value(meta, *keys, default=default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _meta_number_list(meta: dict, *keys, default=None):
    value = _meta_value(meta, *keys, default=default or [])
    if isinstance(value, list):
        parsed = []
        for item in value:
            try:
                parsed.append(float(item))
            except (TypeError, ValueError):
                pass
        return parsed or (default or [])
    if isinstance(value, str):
        parsed = []
        for item in value.replace(",", "\\").split("\\"):
            try:
                parsed.append(float(item.strip()))
            except (TypeError, ValueError):
                pass
        return parsed or (default or [])
    return default or []


def _meta_string_list(meta: dict, *keys):
    value = _meta_value(meta, *keys, default=[])
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", "\\").split("\\") if item.strip()]
    return []


def _build_sidecar_dicom_info(path: Path, meta: dict) -> dict:
    pixel_spacing = _meta_number_list(meta, "PixelSpacing", "pixelSpacing", default=[1.0, 1.0])
    row_spacing = pixel_spacing[0] if len(pixel_spacing) > 0 else 1.0
    column_spacing = pixel_spacing[1] if len(pixel_spacing) > 1 else row_spacing
    rows = int(_meta_number(meta, "Rows", "rows", default=0) or 0)
    columns = int(_meta_number(meta, "Columns", "columns", default=0) or 0)

    slice_thickness = _meta_number(meta, "SliceThickness", "sliceThickness", default=1)

    return {
        "fileName": path.name,
        "patientName": _meta_str(meta, "PatientName", "patientName"),
        "patientId": _meta_str(meta, "PatientID", "patientId"),
        "patientSex": _meta_str(meta, "PatientSex", "patientSex"),
        "patientBirthDate": _meta_str(meta, "PatientBirthDate", "patientBirthDate"),
        "studyDate": _meta_str(meta, "StudyDate", "studyDate"),
        "studyTime": _meta_str(meta, "StudyTime", "studyTime"),
        "studyDescription": _meta_str(meta, "StudyDescription", "studyDescription"),
        "studyInstanceUID": _meta_str(meta, "StudyInstanceUID", "studyInstanceUID"),
        "studyId": _meta_str(meta, "StudyID", "studyId"),
        "accessionNumber": _meta_str(meta, "AccessionNumber", "accessionNumber"),
        "seriesDescription": _meta_str(meta, "SeriesDescription", "seriesDescription"),
        "seriesInstanceUID": _meta_str(meta, "SeriesInstanceUID", "seriesInstanceUID"),
        "seriesNumber": int(_meta_number(meta, "SeriesNumber", "seriesNumber", default=0) or 0),
        "instanceNumber": int(_meta_number(meta, "InstanceNumber", "instanceNumber", default=0) or 0),
        "modality": _meta_str(meta, "Modality", "modality", default="IMG"),
        "institutionName": _meta_str(meta, "InstitutionName", "institutionName"),
        "manufacturer": _meta_str(meta, "Manufacturer", "manufacturer"),
        "bodyPartExamined": _meta_str(meta, "BodyPartExamined", "bodyPartExamined"),
        "sopClassUID": _meta_str(meta, "SOPClassUID", "sopClassUID"),
        "transferSyntaxUid": _meta_str(meta, "TransferSyntaxUID", "transferSyntaxUid"),
        "imageType": _meta_string_list(meta, "ImageType", "imageType"),
        "rows": rows,
        "columns": columns,
        "samplesPerPixel": int(_meta_number(meta, "SamplesPerPixel", "samplesPerPixel", default=1) or 1),
        "photometricInterpretation": _meta_str(meta, "PhotometricInterpretation", "photometricInterpretation", default="MONOCHROME2"),
        "bitsAllocated": int(_meta_number(meta, "BitsAllocated", "bitsAllocated", default=8) or 8),
        "bitsStored": int(_meta_number(meta, "BitsStored", "bitsStored", default=8) or 8),
        "highBit": int(_meta_number(meta, "HighBit", "highBit", default=7) or 7),
        "pixelRepresentation": int(_meta_number(meta, "PixelRepresentation", "pixelRepresentation", default=0) or 0),
        "windowCenter": _meta_number(meta, "WindowCenter", "windowCenter", default=128),
        "windowWidth": _meta_number(meta, "WindowWidth", "windowWidth", default=256),
        "numberOfFrames": int(_meta_number(meta, "NumberOfFrames", "numberOfFrames", default=1) or 1),
        "pixelSpacing": [row_spacing, column_spacing],
        "rowPixelSpacing": row_spacing,
        "columnPixelSpacing": column_spacing,
        "imageOrientationPatient": _meta_number_list(meta, "ImageOrientationPatient", "imageOrientationPatient", default=[1, 0, 0, 0, 1, 0]),
        "imagePositionPatient": _meta_number_list(meta, "ImagePositionPatient", "imagePositionPatient", default=[0, 0, 0]),
        "sliceThickness": slice_thickness,
        "spacingBetweenSlices": _meta_number(meta, "SpacingBetweenSlices", "spacingBetweenSlices", default=slice_thickness),
        "frameOfReferenceUID": _meta_str(meta, "FrameOfReferenceUID", "frameOfReferenceUID"),
    }


def _set_dicom_root(next_root: str):
    global DICOM_ROOT
    raw = (next_root or "").strip()
    if not raw:
        raise ValueError("Root path is required.")
    DICOM_ROOT = Path(os.path.abspath(os.path.expanduser(raw)))
    return DICOM_ROOT


@dicom_browser_api.route('/root-path', methods=['GET'])
def get_root_path():
    return jsonify({
        "success": True,
        "root_path": str(DICOM_ROOT),
        "root_exists": DICOM_ROOT.exists(),
    })


@dicom_browser_api.route('/root-path', methods=['POST'])
def update_root_path():
    payload = request.get_json(silent=True) or {}
    try:
        next_root = _set_dicom_root(payload.get("root_path", ""))
    except ValueError as exc:
        return jsonify({
            "success": False,
            "message": str(exc),
            "root_path": str(DICOM_ROOT),
            "root_exists": DICOM_ROOT.exists(),
        }), 400

    return jsonify({
        "success": True,
        "message": "Root path updated.",
        "root_path": str(next_root),
        "root_exists": next_root.exists(),
    })


@dicom_browser_api.route('/root-path/pick', methods=['POST'])
def pick_root_path():
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=str(DICOM_ROOT), mustexist=False)
        root.destroy()
    except Exception as exc:
        return jsonify({
            "success": False,
            "message": f"Failed to open folder dialog: {exc}",
            "root_path": str(DICOM_ROOT),
            "root_exists": DICOM_ROOT.exists(),
        }), 500

    if not selected:
        return jsonify({
            "success": False,
            "message": "Folder selection was cancelled.",
            "root_path": str(DICOM_ROOT),
            "root_exists": DICOM_ROOT.exists(),
        }), 400

    next_root = Path(os.path.abspath(os.path.expanduser(selected)))
    return jsonify({
        "success": True,
        "root_path": str(next_root),
        "root_exists": next_root.exists(),
    })

@dicom_browser_api.route('/studies', methods=['GET'])
def list_studies():
    """
    Returns a list of DICOM studies for the FolderLeaderVer2Page.
    Groups files by StudyInstanceUID and sets viewer mode based on modality.
    """
    root_exists = DICOM_ROOT.exists()
    study_map = {} # study_uid -> study_data
    images = []
    rejected_files = []
    
    if root_exists:
        # 1. Exhaustive recursive scan for all DICOM files
        all_dcm_files = []
        for f in DICOM_ROOT.rglob("*"):
            if f.is_file() and f.name.lower().endswith((".dcm", ".dicom")):
                all_dcm_files.append(f)
        
        for dcm_path in all_dcm_files:
            try:
                meta = extract_dicom_meta(dcm_path)
                validation = validate_browser_dicom_meta(dcm_path, str(meta.get("Modality", "") or ""))
                if not validation.ok:
                    rejected_files.append({
                        "path": dcm_path.relative_to(DICOM_ROOT).as_posix(),
                        "reason": validation.reason,
                    })
                    continue
                study_uid = meta.get("StudyInstanceUID")
                if not study_uid:
                    # Fallback unique key
                    study_uid = f"unassigned_{meta.get('PatientID', 'Unknown')}_{dcm_path.parent.name}"
                
                if study_uid not in study_map:
                    rel_p = dcm_path.relative_to(DICOM_ROOT).as_posix()
                    study_map[study_uid] = {
                        "id": study_uid.replace(".", "_"),
                        "label": meta.get("PatientName", dcm_path.name),
                        "description": meta.get("StudyDescription", "No Description"),
                        "patientId": meta.get("PatientID", "Unknown"),
                        "patientName": meta.get("PatientName", "Unknown Patient"),
                        "patientAge": meta.get("PatientAge", ""),
                        "patientSex": meta.get("PatientSex", ""),
                        "studyDate": meta.get("StudyDate", "-"),
                        "modalities": set(),
                        "totalFiles": 0,
                        "previewUrl": f"/api/dicom-server/preview?path={rel_p}",
                        "files": [],
                        "directories": set(),
                    }
                
                entry = study_map[study_uid]
                mod = meta.get("Modality", "OT")
                entry["totalFiles"] += 1
                entry["modalities"].add(mod)
                entry["files"].append({
                    "name": dcm_path.name,
                    "relativePath": dcm_path.relative_to(DICOM_ROOT).as_posix(),
                    "downloadUrl": f"/api/dicom-server/download?path={dcm_path.relative_to(DICOM_ROOT).as_posix()}"
                })
                entry["directories"].add(_parent_directory(dcm_path.relative_to(DICOM_ROOT).as_posix()))

            except Exception as e:
                print(f"Error parsing {dcm_path}: {e}")
                continue

        # 2. Format for JSON with smart viewer eligibility
        formatted_studies = []
        for uid, data in study_map.items():
            mod_list = list(data["modalities"])
            primary_mod = mod_list[0] if mod_list else "OT"
            
            # Smart logic: if it's CT/MR or multiple files, treat as Volume (Grid View)
            # If it's single file and PX/DX/CR, treat as 2D (Single View)
            is_volume = (primary_mod.upper() in ["CT", "MR", "CBCT"]) or (data["totalFiles"] > 1)
            
            formatted_studies.append({
                "id": data["id"],
                "label": data["label"],
                "description": data["description"],
                "patientId": data["patientId"],
                "patientName": data["patientName"],
                "patientAge": data["patientAge"],
                "patientSex": data["patientSex"],
                "studyDate": data["studyDate"],
                "modalities": mod_list,
                "totalFiles": data["totalFiles"],
                "totalSeries": 1,
                "previewUrl": data["previewUrl"],
                "series": [
                    {
                        "id": data["id"] + "_s1",
                        "studyId": data["id"],
                        "label": "Series 1",
                        "description": f"Study with {data['totalFiles']} files",
                        "modality": primary_mod,
                        "orientation": "Unknown",
                        "sliceCount": data["totalFiles"],
                        "spacingLabel": "",
                        "compression": "None",
                        "isCompressed": False,
                        "volumeEligible": is_volume,
                        "files": data["files"]
                    }
                ]
            })

        for img_path in DICOM_ROOT.rglob("*"):
            if not img_path.is_file() or not img_path.name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            validation = validate_browser_raster(img_path)
            if not validation.ok:
                rejected_files.append({
                    "path": img_path.relative_to(DICOM_ROOT).as_posix(),
                    "reason": validation.reason,
                })
                continue
            rel_path = img_path.relative_to(DICOM_ROOT).as_posix()
            image_dir = _parent_directory(rel_path)
            sidecar_meta = _load_image_sidecar_meta(img_path)
            sidecar_dicom_info = _build_sidecar_dicom_info(img_path, sidecar_meta) if sidecar_meta else None
            best_match = None
            best_score = -1

            for data in study_map.values():
                for directory in data.get("directories", set()):
                    score = _shared_prefix_score(image_dir, directory)
                    if image_dir and directory and (image_dir.startswith(directory) or directory.startswith(image_dir)):
                        score += 100
                    if img_path.parent.name and directory.lower().find(img_path.parent.name.lower()) >= 0:
                        score += 10
                    if score > best_score:
                        best_score = score
                        best_match = data

            linked_patient_id = best_match.get("patientId", "") if best_match and best_score > 0 else ""
            linked_patient_name = best_match.get("patientName", "") if best_match and best_score > 0 else ""
            linked_patient_age = best_match.get("patientAge", "") if best_match and best_score > 0 else ""
            linked_patient_sex = best_match.get("patientSex", "") if best_match and best_score > 0 else ""
            linked_study_date = best_match.get("studyDate", "") if best_match and best_score > 0 else ""
            linked_modalities = list(best_match.get("modalities", [])) if best_match and best_score > 0 else []
            linked_description = best_match.get("description", "") if best_match and best_score > 0 else ""
            sidecar_modality = _meta_str(sidecar_meta, "Modality", "modality")
            sidecar_description = _meta_str(sidecar_meta, "StudyDescription", "studyDescription", "SeriesDescription", "seriesDescription")
            sidecar_rows = _int_meta(_meta_value(sidecar_meta, "Rows", "rows", default=0))
            sidecar_columns = _int_meta(_meta_value(sidecar_meta, "Columns", "columns", default=0))

            images.append({
                "name": img_path.name,
                "relativePath": rel_path,
                "downloadUrl": f"/api/dicom-server/file?path={rel_path}",
                "folderLabel": img_path.parent.name,
                "width": sidecar_columns,
                "height": sidecar_rows,
                "format": img_path.suffix.lstrip(".").upper(),
                "size": img_path.stat().st_size,
                "linkedStudyId": best_match["id"] if best_match and best_score > 0 else None,
                "patientId": _meta_str(sidecar_meta, "PatientID", "patientId") or linked_patient_id,
                "patientName": _meta_str(sidecar_meta, "PatientName", "patientName") or linked_patient_name,
                "patientAge": _meta_str(sidecar_meta, "PatientAge", "patientAge") or linked_patient_age,
                "patientSex": _meta_str(sidecar_meta, "PatientSex", "patientSex") or linked_patient_sex,
                "studyDate": _meta_str(sidecar_meta, "StudyDate", "studyDate") or linked_study_date,
                "modalities": [sidecar_modality] if sidecar_modality else linked_modalities,
                "description": sidecar_description or linked_description,
                "hasSidecarJson": bool(sidecar_meta),
                "dicomInfo": sidecar_dicom_info,
            })

        response = jsonify({
            "success": True,
            "root_path": str(DICOM_ROOT),
            "root_exists": root_exists,
            "studies": formatted_studies,
            "images": images,
            "rejected_files": rejected_files,
        })
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    response = jsonify({
        "success": False,
        "message": "Root directory not found",
        "root_path": str(DICOM_ROOT),
        "root_exists": False,
        "studies": [],
        "images": [],
        "rejected_files": [],
    })
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@dicom_browser_api.route('/preview', methods=['GET'])
def get_preview():
    """Generates a JPEG preview from a DICOM file."""
    from flask import send_file
    import io
    from services.image_loader import load_image_any
    import cv2

    rel_path = request.args.get('path', '')
    full_path = DICOM_ROOT / rel_path
    
    if not full_path.exists():
        return "Not Found", 404
        
    try:
        # Load using our shared logic (handles windowing)
        img = load_image_any(full_path, use_auto_window=True)
        
        # Resize if too large for preview
        h, w = img.shape[:2]
        if w > 1024:
            new_w = 1024
            new_h = int(h * (1024 / w))
            img = cv2.resize(img, (new_w, new_h))
            
        _, buffer = cv2.imencode('.jpg', img)
        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')
    except Exception as e:
        return f"Error: {str(e)}", 500

@dicom_browser_api.route('/download', methods=['GET'])
def download_dicom():
    from flask import send_from_directory
    rel_path = request.args.get('path', '')
    # Ensure it's treated as a safe download
    return send_from_directory(str(DICOM_ROOT), rel_path, as_attachment=True)

@dicom_browser_api.route('/file', methods=['GET'])
def serve_file():
    from flask import send_from_directory
    rel_path = request.args.get('path', '')
    return send_from_directory(str(DICOM_ROOT), rel_path, as_attachment=False)

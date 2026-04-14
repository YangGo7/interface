from flask import Blueprint, jsonify, request, current_app
import os
import uuid
from pathlib import Path
from services.image_loader import extract_dicom_meta

dicom_browser_api = Blueprint('dicom_browser_api', __name__)

# [CONFIG] Default DICOM Root Path
DICOM_ROOT = Path("C:/interface/case")

@dicom_browser_api.route('/studies', methods=['GET'])
def list_studies():
    """
    Returns a list of DICOM studies for the FolderLeaderVer2Page.
    Groups files by StudyInstanceUID and sets viewer mode based on modality.
    """
    root_exists = DICOM_ROOT.exists()
    study_map = {} # study_uid -> study_data
    
    if root_exists:
        # 1. Exhaustive recursive scan for all DICOM files
        all_dcm_files = []
        for f in DICOM_ROOT.rglob("*"):
            if f.is_file() and f.name.lower().endswith((".dcm", ".dicom")):
                all_dcm_files.append(f)
        
        for dcm_path in all_dcm_files:
            try:
                meta = extract_dicom_meta(dcm_path)
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
                        "files": []
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

        return jsonify({
            "success": True,
            "root_path": str(DICOM_ROOT),
            "root_exists": root_exists,
            "studies": formatted_studies
        })

    return jsonify({
        "success": False,
        "message": "Root directory not found",
        "root_path": str(DICOM_ROOT),
        "root_exists": False,
        "studies": []
    })

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

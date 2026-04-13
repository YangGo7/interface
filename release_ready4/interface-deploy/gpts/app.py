"""
Flask Application - Object Detection API (YOLO-based)
"""

import os
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import math
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import torch
from dotenv import load_dotenv

from config import config, Config
from models.yolo_detector import YOLODetector
from utils.post_processing import ObjectCropper, MissingToothFinder
# from utils.report import ReportGenerator
from services.postprocess import (
    compute_diameter_for_label,
    build_overlay_lines,
    deduplicate_detections_by_label,
)
from services.pano_inference import PanoPipeline, load_image_any
from services.pano_jobs import JobManager
from services.detect_jobs import DetectJobManager
from services import pano_calc_utils as calc_utils
# test utilities (isolated)
from test.split_helper import run_split_models

# Resolve backend directory for relative paths
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
FRONTEND_DIR = BASE_DIR.parent / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
FRONTEND_STATIC_DIR = FRONTEND_DIST_DIR if FRONTEND_DIST_DIR.exists() else FRONTEND_DIR

# Windows may map modern frontend assets to text/plain; register them explicitly.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/json", ".map")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("image/x-icon", ".ico")

app = Flask(
    __name__,
    static_folder=str(FRONTEND_STATIC_DIR),
    static_url_path=""
)

# Environment configuration
env = os.environ.get("FLASK_ENV", "development")
app.config.from_object(config.get(env, config["default"]))

# Global upload limit (synced with config)
app.config['MAX_CONTENT_LENGTH'] = app.config.get('MAX_IMAGE_SIZE', 50 * 1024 * 1024)

# CORS configuration
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": app.config["CORS_ORIGINS"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    },
)

# --- Log Injection Defense ---
class LogInjectionFilter(logging.Formatter):
    """Filter to prevent log injection by escaping CRLF characters."""
    def format(self, record):
        message = super().format(record)
        return message.replace('\n', '\\n').replace('\r', '\\r')

# Apply secure formatter to Flask's logger and Werkzeug
_secure_handler = logging.StreamHandler()
_secure_handler.setFormatter(LogInjectionFilter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))

app.logger.handlers.clear()
app.logger.addHandler(_secure_handler)
app.logger.propagate = False

werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.handlers.clear()
werkzeug_logger.addHandler(_secure_handler)
# -----------------------------

# Resolve device: default to CPU unless explicitly set
_target_device = os.environ.get("PANO_DEVICE", "cpu").lower()
if _target_device == "gpu":
    _target_device = "cuda"
try:
    import torch
    if _target_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        _target_device = "cpu"
except Exception:
    _target_device = "cpu"

# Shared Pano pipeline (multi-model)
_shared_pipeline = PanoPipeline(
    model_dir=Path(app.config["BASE_DIR"]) / "weights",
    model_cfg=app.config.get("PANO_MODELS", {}),
    device=_target_device,
)
_detect_pipeline = _shared_pipeline

# Async job managers sharing the same pipeline instance to save memory
_pano_jobs = JobManager(
    pipeline=_shared_pipeline,
    max_workers=2,
    temp_root=Path(app.config["BASE_DIR"]) / "temp",
)
_detect_jobs = DetectJobManager(
    pipeline=_shared_pipeline,
    max_workers=2,
    temp_root=Path(app.config["BASE_DIR"]) / "temp",
)

# Configure calc utils paths
calc_utils.set_weight_paths(
    bonelevel=Path(app.config["BASE_DIR"]) / "weights" / app.config["PANO_MODELS"]["bonelevel"]["path"],
    cej=Path(app.config["BASE_DIR"]) / "weights" / app.config["PANO_MODELS"]["cej"]["path"],
)


# Register Blueprints
from api.routes_v2 import api_v2
from api.web_report import web_report_api
app.register_blueprint(api_v2, url_prefix='/api/v2')
app.register_blueprint(web_report_api, url_prefix='/api/web_report')
app.extensions["web_report_pipeline"] = _shared_pipeline


def _send_static_file(path: Path):
    guessed_mimetype, _ = mimetypes.guess_type(str(path))
    return send_file(path, mimetype=guessed_mimetype or "application/octet-stream")

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def index(path):
    """Serve the built frontend when available, otherwise fall back to index.html."""
    if path:
        requested_path = Path(app.static_folder) / path
        if requested_path.is_file():
            return _send_static_file(requested_path)
    return _send_static_file(Path(app.static_folder) / "index.html")


@app.route("/favicon.ico")
def favicon():
    favicon_path = Path(app.static_folder) / "favicon.ico"
    if favicon_path.is_file():
        return _send_static_file(favicon_path)
    return ("", 204)


@app.route("/temp/<path:filename>")
def serve_temp_files(filename):
    """Serve artifacts generated under backend/temp."""
    temp_dir = Path(app.config["BASE_DIR"]) / "temp"
    return send_from_directory(temp_dir, filename)

# [NEW] Root Redirect for Convenience
@app.route("/upload", methods=["GET"])
def upload_redirect():
    from flask import redirect
    return redirect("/api/v2/upload")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "API is working"})


@app.route("/api/models", methods=["GET"])
def get_models():
    """Expose available models and default selection."""
    models = [
        {
            "name": model_name,
            "description": model_info["description"],
            "size": model_info["size"],
            "path": model_info["path"],
        }
        for model_name, model_info in app.config["SUPPORTED_MODELS"].items()
    ]

    return jsonify(
        {
            "success": True,
            "models": models,
            "default_model": app.config["DEFAULT_MODEL"],
        }
    )


@app.route("/api/detect", methods=["POST"])
def detect_objects():
    """Run inference synchronously using the multi-model pipeline."""
    try:
        # A. Validate request
        if "image" not in request.files:
            return jsonify({"success": False, "message": "Missing image", "error_type": "NoImage"}), 400
        file = request.files["image"]
        if not file.filename:
            return jsonify({"success": False, "message": "Empty filename", "error_type": "EmptyFilename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "Unsupported file type", "error_type": "InvalidFileFormat"}), 400

        # Create case directory
        case_folder = f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        base_temp_dir = Path(app.config["BASE_DIR"]) / "temp"
        case_dir = base_temp_dir / case_folder
        case_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(file.filename).suffix or ".png"
        temp_path = case_dir / f"original{ext}"
        file.save(temp_path)

        # Run pipeline synchronously
        # We ignore custom 'model', 'conf', 'iou' params for now to ensure Full Pipeline consistency
        # as the frontend expects PBL/Caries/etc. which requires the full pipeline.
        result = _detect_pipeline.run(image_path=temp_path, out_dir=case_dir)

        # Post-process paths for URL access
        overlay_path = Path(result["overlay_path"])
        result["overlay_url"] = f"/temp/{case_folder}/{overlay_path.name}"
        heatmap_path = result.get("heatmap_overlay_path")
        if heatmap_path:
            result["heatmap_overlay_url"] = f"/temp/{case_folder}/{Path(heatmap_path).name}"
        result["image_url"] = f"/temp/{case_folder}/original{ext}"
        # Remove absolute path from response to avoid confusion/exposure
        del result["overlay_path"]
        if "heatmap_overlay_path" in result:
            del result["heatmap_overlay_path"]
        
        # Add original image URL
        result["image_url"] = f"/temp/{case_folder}/original{ext}"

        return jsonify({"success": True, "result": result}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False, 
            "message": f"Server error: {str(e)}", 
            "error_type": "InternalError"
        }), 500




# --- Pano async pipeline ---


def _parse_bool(val: str, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).lower() in {"1", "true", "yes", "on"}


@app.route("/api/pano", methods=["POST"])
def pano_submit():
    """Submit a pano image for async multi-model inference."""
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "message": "Missing image", "error_type": "NoImage"}), 400

        file = request.files["image"]
        if not file.filename:
            return jsonify({"success": False, "message": "Empty filename", "error_type": "EmptyFilename"}), 400

        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "Unsupported file type", "error_type": "InvalidFileFormat"}), 400

        

        case_folder = f"pano_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        base_temp_dir = Path(app.config["BASE_DIR"]) / "temp"
        case_dir = base_temp_dir / case_folder
        case_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(file.filename).suffix or ".png"
        temp_path = case_dir / f"original{ext}"
        file.save(temp_path)

        job_id = _pano_jobs.submit(image_path=temp_path, case_dir=case_dir)

        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "status": "queued",
                "status_url": f"/api/pano/status/{job_id}",
                "case_dir": f"/temp/{case_folder}/",
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": str(e), "error_type": "InternalError"}), 500


@app.route("/api/pano/status/<job_id>", methods=["GET"])
def pano_status(job_id):
    job = _pano_jobs.get(job_id)
    if not job:
        return jsonify({"success": False, "message": "Job not found", "error_type": "NotFound"}), 404

    resp = {
        "success": True,
        "job_id": job_id,
        "status": job["status"],
    }

    if job["status"] == "done" and job.get("result"):
        result = job["result"]
        case_dir = Path(job["case_dir"])
        overlay_path = Path(result["overlay_path"])
        overlay_url = f"/temp/{case_dir.name}/{overlay_path.name}"
        resp["result"] = {
            "overlay_url": overlay_url,
            "det_counts": result.get("det_counts", {}),
        }
    elif job["status"] == "failed":
        resp["error"] = job.get("error")

    return jsonify(resp)


# --- Detect async (multi-model, labels + PBL) ---

@app.route("/api/detect_async", methods=["POST"])
def detect_submit():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "message": "Missing image", "error_type": "NoImage"}), 400
        file = request.files["image"]
        if not file.filename:
            return jsonify({"success": False, "message": "Empty filename", "error_type": "EmptyFilename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "Unsupported file type", "error_type": "InvalidFileFormat"}), 400

        case_folder = f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        base_temp_dir = Path(app.config["BASE_DIR"]) / "temp"
        case_dir = base_temp_dir / case_folder
        case_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(file.filename).suffix or ".png"
        temp_path = case_dir / f"original{ext}"
        file.save(temp_path)

        # If DICOM, also save a PNG preview for frontend display
        preview_url = None
        if ext.lower() in [".dcm", ".dicom"]:
            try:
                img_any = load_image_any(temp_path)
                png_path = case_dir / "original.png"
                cv2.imwrite(str(png_path), img_any)
                preview_url = f"/temp/{case_folder}/original.png"
            except Exception as e:
                print(f"[WARN] Failed to save PNG preview for DICOM: {e}")
        else:
            preview_url = f"/temp/{case_folder}/original{ext}"

        job_id = _detect_jobs.submit(image_path=temp_path, case_dir=case_dir)

        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "status": "queued",
                "status_url": f"/api/detect/status/{job_id}",
                "case_dir": f"/temp/{case_folder}/",
                "preview_url": preview_url,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": str(e), "error_type": "InternalError"}), 500


@app.route("/api/detect/status/<job_id>", methods=["GET"])
def detect_status(job_id):
    job = _detect_jobs.get(job_id)
    if not job:
        return jsonify({"success": False, "message": "Job not found", "error_type": "NotFound"}), 404

    resp = {
        "success": True,
        "job_id": job_id,
        "status": job["status"],
    }
    case_dir = Path(job["case_dir"]) if job.get("case_dir") else None

    # Prefer converted PNG preview if exists, else first original.*
    image_url = None
    if case_dir and case_dir.exists():
        png_preview = case_dir / "original.png"
        if png_preview.exists():
            image_url = f"/temp/{case_dir.name}/{png_preview.name}"
        else:
            orig_file = next(case_dir.glob("original.*"), None)
            image_url = f"/temp/{case_dir.name}/{orig_file.name}" if orig_file else None

    if job["status"] == "done" and job.get("result"):
        result = job["result"]
        teeth = result.get("teeth") or result.get("teeth_objects") or []
        bonelevel = result.get("bonelevel") or {}

        def build_best_map(existing_map, teeth_objects, raw_items, finding_type):
            best_map = {}

            def upsert(tooth_id, conf=None, box=None):
                if not tooth_id:
                    return
                key = str(tooth_id)
                conf_val = float(conf or 0.0)
                current = best_map.get(key)
                if current is None or conf_val >= float(current.get("conf", 0.0)):
                    best_map[key] = {
                        "conf": conf_val,
                        "box": box or [],
                    }

            if isinstance(existing_map, dict):
                for tooth_id, det in existing_map.items():
                    if isinstance(det, dict):
                        upsert(
                            tooth_id,
                            det.get("conf", det.get("confidence", 0.0)),
                            det.get("box", []),
                        )

            for tooth in teeth_objects:
                tooth_id = tooth.get("tooth_label") or tooth.get("assigned_tooth")
                for finding in tooth.get("findings", []) or []:
                    if str(finding.get("type", "")).lower() == finding_type:
                        upsert(
                            tooth_id,
                            finding.get("conf", finding.get("confidence", 0.0)),
                            finding.get("box", []),
                        )

            if isinstance(raw_items, list):
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                    item_label = str(item.get("label", "")).lower()
                    if finding_type == "caries" and "caries" not in item_label and item.get("assigned_tooth") is None:
                        continue
                    if finding_type == "periapical" and "periapical" not in item_label and item.get("assigned_tooth") is None:
                        continue
                    upsert(
                        item.get("assigned_tooth") or item.get("tooth_label") or item.get("tooth"),
                        item.get("conf", item.get("confidence", 0.0)),
                        item.get("box", []),
                    )

            return best_map

        caries_best_map = build_best_map(
            result.get("caries_by_tooth_best") or result.get("caries_by_tooth"),
            teeth,
            result.get("caries", []),
            "caries",
        )
        periapical_best_map = build_best_map(
            result.get("periapical_by_tooth_best") or result.get("periapical_by_tooth"),
            teeth,
            result.get("periapical", []),
            "periapical",
        )

        pbl_map = result.get("pbl")
        if not pbl_map and isinstance(bonelevel, dict):
            pbl_map = {
                str(k): v.get("percent", 0)
                for k, v in bonelevel.items()
                if isinstance(v, dict)
            }

        pbl_level_map = result.get("pbl_level")
        if not pbl_level_map and isinstance(bonelevel, dict):
            pbl_level_map = {
                str(k): v.get("level", 0)
                for k, v in bonelevel.items()
                if isinstance(v, dict)
            }
        
        # Resolve paths from result directly (most reliable)
        image_path_rel = result.get("image_path", "").replace("\\", "/")
        overlay_path_rel = result.get("overlay_path", "").replace("\\", "/")
        heatmap_path_rel = result.get("heatmap_overlay_path", "").replace("\\", "/")
        
        # Ensure leading slash for URL usage
        if image_path_rel and not image_path_rel.startswith("/"):
            image_path_rel = "/" + image_path_rel
        if overlay_path_rel and not overlay_path_rel.startswith("/"):
            overlay_path_rel = "/" + overlay_path_rel
        if heatmap_path_rel and not heatmap_path_rel.startswith("/"):
            heatmap_path_rel = "/" + heatmap_path_rel
            
        # Fallback if missing in result (legacy check)
        if not image_path_rel and case_dir and case_dir.exists():
             png_preview = case_dir / "original.png"
             if png_preview.exists():
                 image_path_rel = f"/temp/{case_dir.name}/{png_preview.name}"
             else:
                 orig_file = next(case_dir.glob("original.*"), None)
                 image_path_rel = f"/temp/{case_dir.name}/{orig_file.name}" if orig_file else None

        if not heatmap_path_rel and case_dir and case_dir.exists():
             heatmap_file = case_dir / "heatmap_overlay.png"
             if heatmap_file.exists():
                 heatmap_path_rel = f"/temp/{case_dir.name}/{heatmap_file.name}"

        resp["result"] = {
            "overlay_url": overlay_path_rel, 
            "heatmap_overlay_url": heatmap_path_rel,
            "image_url": image_path_rel,
            "preview_url": result.get("preview_url"),
            "overlay_path": overlay_path_rel, # Alias for frontend
            "heatmap_overlay_path": heatmap_path_rel,
            "image_path": image_path_rel,     # Alias for frontend
            "is_volume": bool(result.get("is_volume", False)),
            "det_counts": result.get("det_counts", {}),
            "pbl": pbl_map or {},
            "pbl_level": pbl_level_map or {},
            "bonelevel": bonelevel,
            "teeth": teeth,
            "data": teeth,
            "odontogram_map": result.get("odontogram_map", {}),
            "caries_by_tooth": caries_best_map,
            "periapical_by_tooth": periapical_best_map,
            "teeth_missing": result.get("teeth_missing", []),
             # Ensure raw lists are passed for summary counts
            "missing_teeth": result.get("missing_teeth", []),
            "caries": result.get("caries", []),
            "periapical": result.get("periapical", []),
            "ai_commentary": result.get("ai_commentary", ""),
            "caries_by_tooth_best": caries_best_map,
            "periapical_by_tooth_best": periapical_best_map,
            "implant_by_tooth": result.get("implant_by_tooth", {}),
            "implant_by_tooth_best": result.get("implant_by_tooth_best", {}),
            "crown_by_tooth": result.get("crown_by_tooth", {}),
            "crown_by_tooth_best": result.get("crown_by_tooth_best", {}),
            "filling_by_tooth": result.get("filling_by_tooth", {}),
            "filling_by_tooth_best": result.get("filling_by_tooth_best", {}),
            "implant_metrics": result.get("implant_metrics", {}),
            "mm_per_px": result.get("mm_per_px"),
            "nerve_contours": result.get("nerve_contours", []),
            "sinus_contours": result.get("sinus_contours", []),
        }
    elif job["status"] == "running" and image_url:
        resp["result"] = {"image_url": image_url}
    elif job["status"] == "failed":
        resp["error"] = job.get("error")

    return jsonify(resp)


@app.route("/api/test_split_detect", methods=["POST"])
def test_split_detect():
    """
    테스트 전용 엔드포인트: 업로드한 이미지를 여러 YOLO 가중치로 직접 추론해
    /temp/<case>/all.png, teeth.png, caries_peri.png, other.png 파일을 만들고
    해당 URL을 반환합니다. 기존 파이프라인과 독립적으로 동작합니다.
    """
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "message": "Missing image", "error_type": "NoImage"}), 400
        file = request.files["image"]
        if not file.filename:
            return jsonify({"success": False, "message": "Empty filename", "error_type": "EmptyFilename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "Unsupported file type", "error_type": "InvalidFileFormat"}), 400

        # 모델 가중치: 지정 없으면 테스트용 기본값 사용
        weights_dir = BASE_DIR / "weights"
        defaults = {
            "model_all": weights_dir / "yolo11_seg_ver1_800_1024px.pt",
            "model_teeth": weights_dir / "cej.pt",
            "model_caries": weights_dir / "caries_det.pt",
            "model_other": weights_dir / "periapical.pt",
            "model_extra": weights_dir / "bonelevel.pt",
        }
        model_all = request.form.get("model_all") or defaults["model_all"]
        model_teeth = request.form.get("model_teeth") or defaults["model_teeth"]
        model_caries = request.form.get("model_caries") or defaults["model_caries"]
        model_other = request.form.get("model_other") or defaults["model_other"]
        model_extra = request.form.get("model_extra") or defaults["model_extra"]

        case_folder = f"test_split_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        base_temp_dir = Path(app.config["BASE_DIR"]) / "temp"
        case_dir = base_temp_dir / case_folder
        case_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(file.filename).suffix or ".png"
        temp_path = case_dir / f"original{ext}"
        file.save(temp_path)

        rels = run_split_models(
            image_path=temp_path,
            case_dir=case_dir,
            model_all=Path(model_all) if model_all else None,
            model_teeth=Path(model_teeth) if model_teeth else None,
            model_caries=Path(model_caries) if model_caries else None,
            model_other=Path(model_other) if model_other else None,
            model_extra=Path(model_extra) if model_extra else None,
        )

        # 기본 overlay는 all 우선, 없으면 teeth/그 외
        overlay_url = rels.get("all") or rels.get("teeth") or rels.get("caries_peri") or rels.get("other")
        image_url = f"/temp/{case_dir.name}/{temp_path.name}"

        return jsonify(
            {
                "success": True,
                "result": {
                    "overlay_url": overlay_url,
                    "image_url": image_url,
                    "all_overlay_url": rels.get("all"),
                    "teeth_overlay_url": rels.get("teeth"),
                    "caries_peri_overlay_url": rels.get("caries_peri"),
                    "other_overlay_url": rels.get("other"),
                    "extra_overlay_url": rels.get("extra"),
                    "det_counts": {},  # 로컬 추론이므로 카운트는 비워둠
                },
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": str(e), "error_type": "InternalError"}), 500


@app.route('/temp/<path:filename>')
def serve_temp(filename):
    """Serve temporary files (images) from the temp directory."""
    temp_dir = Path(app.config["BASE_DIR"]) / "temp"
    return send_from_directory(str(temp_dir), filename)



# Helpers

def allowed_file(filename: str) -> bool:
    """Check whether the uploaded filename uses an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config[
        "ALLOWED_EXTENSIONS"
    ]


@app.errorhandler(404)
def not_found(error):
    return (
        jsonify(
            {
                "success": False,
                "message": "Endpoint not found",
                "error_type": "NotFound",
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(error):
    return (
        jsonify(
            {
                "success": False,
                "message": "Internal server error",
                "error_type": "InternalServerError",
            }
        ),
        500,
    )


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # Real-time log output

    print("=" * 50)
    print("Starting Object Detection API Server")
    print("=" * 50)
    print(f"Environment: {env}")
    print("Server: http://localhost:5000")
    print(f"Debug Mode: {app.config['DEBUG']}")
    print(f"Inference Device: {_target_device}")
    print(f"Frontend Static Dir: {app.static_folder}")
    print(f"Default Model: {app.config['DEFAULT_MODEL']}")
    print("=" * 50)
    
    # [NEW] Start background cleanup thread for temp and reports
    import threading
    import shutil
    import time
    
    def cleanup_old_files(folder_path, max_age_seconds=86400):
        try:
            now = time.time()
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    if now - os.path.getmtime(file_path) > max_age_seconds:
                        os.remove(file_path)
                elif os.path.isdir(file_path):
                    if now - os.path.getmtime(file_path) > max_age_seconds:
                        shutil.rmtree(file_path, ignore_errors=True)
        except Exception as e:
            print(f"  [Cleanup] Error cleaning {folder_path}: {e}")

    def cleanup_worker():
        temp_dir = BASE_DIR / "temp"
        reports_dir = BASE_DIR / "reports"
        while True:
            try:
                if temp_dir.exists(): cleanup_old_files(str(temp_dir), 86400)
                if reports_dir.exists(): cleanup_old_files(str(reports_dir), 86400)
            except Exception as e:
                print(f"  [Cleanup] Worker error: {e}")
            time.sleep(3600) # Check every 1 hour

    t = threading.Thread(target=cleanup_worker, daemon=True)
    t.start()
    print("  [INIT] Background folder cleanup thread started (24h retention).")

    app.run(host="0.0.0.0", port=5000, debug=app.config["DEBUG"])

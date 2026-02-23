"""
Flask Application - Object Detection API (YOLO-based)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import math
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch

from config import config, Config
from models.yolo_detector import YOLODetector
from utils.post_processing import ObjectCropper, MissingToothFinder
from utils.report import ReportGenerator
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

app = Flask(
    __name__,
    static_folder=str(BASE_DIR.parent / "frontend"),
    static_url_path=""
)

# Environment configuration
env = os.environ.get("FLASK_ENV", "development")
app.config.from_object(config.get(env, config["default"]))

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
app.register_blueprint(api_v2, url_prefix='/api/v2')

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def index(path):
    """Serve the frontend entry point."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/temp/<path:filename>")
def serve_temp_files(filename):
    """Serve artifacts generated under backend/temp."""
    temp_dir = Path(app.config["BASE_DIR"]) / "temp"
    return send_from_directory(temp_dir, filename)


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
        result["image_url"] = f"/temp/{case_folder}/original{ext}"
        # Remove absolute path from response to avoid confusion/exposure
        del result["overlay_path"]
        
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
        overlay_path = Path(result["overlay_path"])
        overlay_url = f"/temp/{case_dir.name}/{overlay_path.name}"

        resp["result"] = {
            "overlay_url": overlay_url,
            "image_url": image_url,
            "det_counts": result.get("det_counts", {}),
            "pbl": result.get("pbl", {}),
            "pbl_level": result.get("pbl_level", {}),
            "odontogram_map": result.get("odontogram_map", {}),
            "caries_by_tooth": result.get("caries_by_tooth", []),
            "periapical_by_tooth": result.get("periapical_by_tooth", []),
            "teeth_missing": result.get("teeth_missing", []),
            "caries_by_tooth_best": result.get("caries_by_tooth_best", {}),
            "periapical_by_tooth_best": result.get("periapical_by_tooth_best", {}),
            "implant_by_tooth": result.get("implant_by_tooth", {}),
            "implant_by_tooth_best": result.get("implant_by_tooth_best", {}),
            "crown_by_tooth": result.get("crown_by_tooth", {}),
            "crown_by_tooth_best": result.get("crown_by_tooth_best", {}),
            "filling_by_tooth": result.get("filling_by_tooth", {}),
            "filling_by_tooth_best": result.get("filling_by_tooth_best", {}),
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
    print("=" * 50)
    print("Starting Object Detection API Server")
    print("=" * 50)
    print(f"Environment: {env}")
    print("Server: http://localhost:5000")
    print(f"Debug Mode: {app.config['DEBUG']}")
    print(f"Default Model: {app.config['DEFAULT_MODEL']}")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5000, debug=app.config["DEBUG"])

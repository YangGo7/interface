import copy
import threading
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

from services.pano_inference import load_image_any
from services.report_dictation_service import ReportDictationService
from services.web_report_merge_service import WebReportMergeService
from services.web_report_report_service import WebReportReportService
from services.web_report_session_service import WebReportSessionService


web_report_api = Blueprint("web_report_api", __name__)

session_service = WebReportSessionService()
merge_service = WebReportMergeService()
report_service = WebReportReportService()
dictation_service = ReportDictationService()


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


def _session_root(session_id: str) -> Path:
    return Path(current_app.config["BASE_DIR"]) / "runs" / "web_report" / session_id


def _ensure_dirs(session_id: str) -> Dict[str, Path]:
    root = _session_root(session_id)
    paths = {
        "root": root,
        "source": root / "source",
        "inference": root / "inference",
        "reports": root / "reports",
        "final": root / "final",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _source_path_from_url(url: Optional[str]) -> Optional[Path]:
    if not url or not url.startswith("/temp/"):
        return None
    relative = url[len("/temp/") :]
    path = (Path(current_app.config["BASE_DIR"]) / "temp" / relative).resolve()
    return path if path.exists() else None


def _pick_best_map(items: list[Dict[str, Any]], finding_type: str) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for item in items:
        tooth_id = str(item.get("assigned_tooth") or item.get("tooth_label") or item.get("tooth") or "")
        if not tooth_id:
            continue
        label = str(item.get("label", "")).lower()
        if finding_type not in label and item.get("assigned_tooth") is None:
            continue
        conf = float(item.get("conf", item.get("confidence", 0)) or 0)
        current = best.get(tooth_id)
        if current is None or conf >= float(current.get("conf", 0)):
            best[tooth_id] = {"conf": conf, "box": item.get("box", [])}
    return best


def _build_ai_result(result_map: Dict[str, Any]) -> Dict[str, Any]:
    teeth = copy.deepcopy(result_map.get("teeth_objects") or result_map.get("teeth") or [])
    caries = copy.deepcopy(result_map.get("caries_objects") or result_map.get("caries") or [])
    periapical = copy.deepcopy(result_map.get("periapical_objects") or result_map.get("periapical") or [])
    missing_teeth_objects = copy.deepcopy(result_map.get("missing_teeth") or [])
    missing_teeth = [
        str(item.get("tooth_label") if isinstance(item, dict) else item)
        for item in missing_teeth_objects
    ]

    pbl = {}
    pbl_level = {}
    bonelevel = copy.deepcopy(result_map.get("bonelevel") or {})
    for tooth in teeth:
        label = str(tooth.get("tooth_label", ""))
        if not label:
            continue
        pbl[label] = float(tooth.get("bone_loss_pct", 0) or 0)
        pbl_level[label] = int(tooth.get("bone_loss_level", 0) or 0)
        bonelevel.setdefault(
            label,
            {"percent": pbl[label], "level": pbl_level[label]},
        )

    return {
        "teeth": teeth,
        "data": teeth,
        "caries": caries,
        "periapical": periapical,
        "missing_teeth": missing_teeth_objects,
        "teeth_missing": missing_teeth,
        "implant_metrics": copy.deepcopy(result_map.get("implant_metrics") or {}),
        "bonelevel": bonelevel,
        "odontogram_map": copy.deepcopy(result_map.get("odontogram_map") or {}),
        "mm_per_px": result_map.get("mm_per_px"),
        "nerve_contours": copy.deepcopy(result_map.get("nerve_contours") or []),
        "sinus_contours": copy.deepcopy(result_map.get("sinus_contours") or []),
        "pbl": pbl,
        "pbl_level": pbl_level,
        "caries_by_tooth": _pick_best_map(caries, "caries"),
        "caries_by_tooth_best": _pick_best_map(caries, "caries"),
        "periapical_by_tooth": _pick_best_map(periapical, "periapical"),
        "periapical_by_tooth_best": _pick_best_map(periapical, "periapical"),
        "det_counts": {
            "seg_teeth": len(teeth),
            "caries": len(caries),
            "periapical": len(periapical),
            "cej_masks": 0,
            "bonelevel_masks": len(bonelevel),
        },
    }


def _run_session_analysis(app, session_id: str, source_path: Path, language: str) -> None:
    with app.app_context():
        try:
            session_service.set_status(session_id, "processing")
            paths = _ensure_dirs(session_id)
            pipeline = current_app.extensions["web_report_pipeline"]
            result_map = pipeline.run(source_path, paths["inference"])

            overlay_path = paths["inference"] / "overlay.png"
            bl_viz_path = paths["inference"] / "bl_viz.png"
            preview_path = paths["source"] / "preview.png"
            if not preview_path.exists():
                if source_path.suffix.lower() in (".dcm", ".dicom"):
                    preview_img = load_image_any(source_path)
                    cv2.imwrite(str(preview_path), preview_img)
                else:
                    preview_path = source_path

            assets = {
                "source_path": str(source_path),
                "preview_path": str(preview_path) if preview_path.exists() else None,
                "overlay_path": str(overlay_path) if overlay_path.exists() else None,
                "bl_viz_path": str(bl_viz_path) if bl_viz_path.exists() else None,
                "inference_dir": str(paths["inference"]),
                "reports_dir": str(paths["reports"]),
                "final_dir": str(paths["final"]),
            }
            session_service.set_assets(session_id, assets)

            ai_result = _build_ai_result(result_map)
            session_service.save_ai_result(session_id, ai_result)

            session = session_service.get_session(session_id)
            effective_result = merge_service.build_effective_result(
                session_id,
                session["ai_result"] or {},
                session["assets"] or {},
                session["doctor_overrides"] or {},
            )
            report_info = report_service.generate_report(
                session_id=session_id,
                user_name=f"WebReport_{session_id[:8]}",
                image_path=source_path,
                overlay_path=overlay_path,
                bl_viz_path=bl_viz_path,
                effective_result=effective_result,
                output_dir=paths["reports"],
            )
            session_service.create_report_version(
                session_id=session_id,
                status="draft",
                html_path=report_info.get("html_path"),
                pdf_path=report_info.get("pdf_path"),
                snapshot=report_info.get("snapshot") or {},
            )
            session_service.set_status(session_id, "completed")
        except Exception as exc:
            current_app.logger.exception("web_report analysis failed for session %s", session_id)
            session_service.set_status(session_id, "failed", str(exc))


def _copy_asset(src: Optional[Path], dst: Path) -> Optional[Path]:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


@web_report_api.route("/session", methods=["POST"])
def create_session():
    payload = request.get_json(silent=True) or {}
    session_id = session_service.create_session(payload.get("language", "English"))
    return jsonify({"success": True, "session_id": session_id})


@web_report_api.route("/from-chart", methods=["POST"])
def create_from_chart():
    session_id = None
    try:
        payload = request.get_json(silent=True) or {}
        result = payload.get("result")
        if not isinstance(result, dict):
            return jsonify({"success": False, "error": "Missing chart result"}), 400

        source_url = payload.get("source_url") or result.get("image_url")
        overlay_url = payload.get("overlay_url") or result.get("overlay_url")
        source_path = _source_path_from_url(source_url)
        if source_path is None:
            return jsonify({"success": False, "error": "Could not resolve source image from chart result"}), 400

        session_id = session_service.create_session(payload.get("language", "English"))
        paths = _ensure_dirs(session_id)

        copied_source = _copy_asset(source_path, paths["source"] / source_path.name)
        preview_path = copied_source
        if copied_source and copied_source.suffix.lower() in (".dcm", ".dicom"):
            try:
                preview_img = load_image_any(copied_source)
                preview_path = paths["source"] / "preview.png"
                cv2.imwrite(str(preview_path), preview_img)
            except Exception:
                preview_path = copied_source

        overlay_path = _copy_asset(_source_path_from_url(overlay_url), paths["inference"] / "overlay.png")
        bl_viz_src = None
        if overlay_url and overlay_url.startswith("/temp/"):
            overlay_src = _source_path_from_url(overlay_url)
            if overlay_src is not None:
                candidate = overlay_src.parent / "bl_viz.png"
                if candidate.exists():
                    bl_viz_src = candidate
        bl_viz_path = _copy_asset(bl_viz_src, paths["inference"] / "bl_viz.png")

        ai_result = copy.deepcopy(result)
        ai_result.setdefault("teeth", ai_result.get("data", []))
        ai_result.setdefault("data", ai_result.get("teeth", []))
        ai_result.setdefault("missing_teeth", ai_result.get("teeth_missing", []))
        ai_result.setdefault("teeth_missing", ai_result.get("missing_teeth", []))

        session_service.set_assets(
            session_id,
            {
                "source_path": str(copied_source) if copied_source else None,
                "preview_path": str(preview_path) if preview_path else None,
                "overlay_path": str(overlay_path) if overlay_path else None,
                "bl_viz_path": str(bl_viz_path) if bl_viz_path else None,
                "inference_dir": str(paths["inference"]),
                "reports_dir": str(paths["reports"]),
                "final_dir": str(paths["final"]),
            },
        )
        session_service.save_ai_result(session_id, ai_result)

        session = session_service.get_session(session_id)
        effective_result = merge_service.build_effective_result(
            session_id,
            session["ai_result"] or {},
            session["assets"] or {},
            session["doctor_overrides"] or {},
        )
        report_info = report_service.generate_report(
            session_id=session_id,
            user_name=f"WebReport_{session_id[:8]}",
            image_path=copied_source,
            overlay_path=overlay_path,
            bl_viz_path=bl_viz_path,
            effective_result=effective_result,
            output_dir=paths["reports"],
        )
        session_service.create_report_version(
            session_id=session_id,
            status="draft",
            html_path=report_info.get("html_path"),
            pdf_path=report_info.get("pdf_path"),
            snapshot=report_info.get("snapshot") or {},
        )
        session_service.set_status(session_id, "completed")

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "report_url": f"/report/{session_id}",
                "html_url": f"/api/web_report/session/{session_id}/report",
            }
        )
    except Exception as exc:
        current_app.logger.exception("web_report from-chart failed")
        if session_id:
            session_service.set_status(session_id, "failed", str(exc))
        return jsonify({"success": False, "error": f"Failed to create chart report: {exc}"}), 500


@web_report_api.route("/session/<session_id>/upload", methods=["POST"])
def upload_session_file(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": "Invalid session"}), 404

    if "image" not in request.files:
        return jsonify({"success": False, "error": "Missing image"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400
    if not _allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported file type"}), 400

    paths = _ensure_dirs(session_id)
    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix or ".png"
    source_path = paths["source"] / f"original{suffix}"
    file.save(source_path)

    preview_path: Optional[Path] = None
    if suffix.lower() in (".dcm", ".dicom"):
        try:
            preview_img = load_image_any(source_path)
            preview_path = paths["source"] / "preview.png"
            cv2.imwrite(str(preview_path), preview_img)
        except Exception:
            preview_path = None
    else:
        preview_path = source_path

    session_service.set_assets(
        session_id,
        {
            "source_path": str(source_path),
            "preview_path": str(preview_path) if preview_path else None,
            "overlay_path": None,
            "bl_viz_path": None,
            "inference_dir": str(paths["inference"]),
            "reports_dir": str(paths["reports"]),
            "final_dir": str(paths["final"]),
        },
    )
    session_service.set_status(session_id, "queued")

    app = current_app._get_current_object()
    thread = threading.Thread(
        target=_run_session_analysis,
        args=(app, session_id, source_path, session.get("language", "English")),
        daemon=True,
    )
    thread.start()

    return jsonify({"success": True, "session_id": session_id, "status": "queued"})


@web_report_api.route("/session/<session_id>", methods=["GET"])
def get_session(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": "Invalid session"}), 404

    effective_result = None
    if session.get("ai_result"):
        effective_result = merge_service.build_effective_result(
            session_id,
            session["ai_result"],
            session.get("assets") or {},
            session.get("doctor_overrides") or {},
        )

    report = session.get("report") or {}
    assets = session.get("assets") or {}
    public_assets = {
        "source_url": merge_service._public_url(session_id, assets.get("source_path")),
        "preview_url": merge_service._public_url(session_id, assets.get("preview_path")),
        "overlay_url": merge_service._public_url(session_id, assets.get("overlay_path")),
        "bl_viz_url": merge_service._public_url(session_id, assets.get("bl_viz_path")),
    }

    response = {
        "success": True,
        "session": {
            "id": session["id"],
            "status": session["status"],
            "error": session["error"],
            "language": session["language"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "finalized_at": session["finalized_at"],
            "is_finalized": session["is_finalized"],
            "assets": public_assets,
            "ai_result": session.get("ai_result"),
            "doctor_overrides": session.get("doctor_overrides") or {},
            "effective_result": effective_result,
            "report": {
                **report,
                "page_url": f"/report/{session_id}" if report else None,
                "html_url": f"/api/web_report/session/{session_id}/report" if report else None,
                "pdf_url": f"/api/web_report/session/{session_id}/report/pdf" if report else None,
            },
        },
    }
    return jsonify(response)


@web_report_api.route("/session/<session_id>/overrides", methods=["PATCH"])
def patch_overrides(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": "Invalid session"}), 404
    if session.get("is_finalized"):
        return jsonify({"success": False, "error": "Finalized sessions are read-only"}), 409

    payload = request.get_json(silent=True) or {}
    current_overrides = session.get("doctor_overrides") or {"teeth": {}, "report_note": "", "attached_captures": []}
    merged_overrides = copy.deepcopy(current_overrides)
    merged_overrides.setdefault("teeth", {})
    merged_overrides.setdefault("attached_captures", [])
    for tooth_id in payload.get("reset_tooth_ids") or []:
        merged_overrides["teeth"].pop(str(tooth_id), None)
    for tooth_id, tooth_override in (payload.get("tooth_overrides") or {}).items():
        current_tooth = merged_overrides["teeth"].get(str(tooth_id), {})
        current_tooth.update(tooth_override or {})
        merged_overrides["teeth"][str(tooth_id)] = current_tooth
    if "report_note" in payload:
        merged_overrides["report_note"] = payload.get("report_note") or ""
    if "attached_captures" in payload:
        merged_overrides["attached_captures"] = payload.get("attached_captures") or []

    session_service.save_overrides(session_id, merged_overrides)
    refreshed = session_service.get_session(session_id)
    effective_result = merge_service.build_effective_result(
        session_id,
        refreshed["ai_result"] or {},
        refreshed["assets"] or {},
        refreshed["doctor_overrides"] or {},
    )
    return jsonify({"success": True, "doctor_overrides": refreshed["doctor_overrides"], "effective_result": effective_result})


@web_report_api.route("/session/<session_id>/dictation", methods=["POST"])
def transcribe_dictation(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": "Invalid session"}), 404
    if session.get("is_finalized"):
        return jsonify({"success": False, "error": "Finalized sessions are read-only"}), 409
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "Missing audio"}), 400

    audio = request.files["audio"]
    if not audio.filename:
        return jsonify({"success": False, "error": "Empty audio filename"}), 400

    paths = _ensure_dirs(session_id)
    suffix = Path(secure_filename(audio.filename)).suffix or ".webm"
    audio_path = paths["source"] / f"dictation{suffix}"
    audio.save(audio_path)

    mime_type = audio.mimetype or "audio/webm"
    try:
        payload = dictation_service.transcribe_and_summarize(
            audio_path=audio_path,
            mime_type=mime_type,
            language=session.get("language", "English"),
        )
        return jsonify({"success": True, **payload})
    except Exception as exc:
        current_app.logger.exception("dictation transcription failed for session %s", session_id)
        return jsonify({"success": False, "error": str(exc)}), 500


def _generate_report_version(session_id: str, version_status: str) -> Dict[str, Any]:
    session = session_service.get_session(session_id)
    if session is None:
        raise FileNotFoundError("Invalid session")
    assets = session.get("assets") or {}
    source_path = Path(assets["source_path"])
    overlay_path = Path(assets["overlay_path"]) if assets.get("overlay_path") else None
    bl_viz_path = Path(assets["bl_viz_path"]) if assets.get("bl_viz_path") else None
    output_dir = Path(assets["final_dir"] if version_status == "final" else assets["reports_dir"])

    effective_result = merge_service.build_effective_result(
        session_id,
        session["ai_result"] or {},
        assets,
        session["doctor_overrides"] or {},
    )
    report_info = report_service.generate_report(
        session_id=session_id,
        user_name=f"WebReport_{session_id[:8]}_{version_status}",
        image_path=source_path,
        overlay_path=overlay_path,
        bl_viz_path=bl_viz_path,
        effective_result=effective_result,
        output_dir=output_dir,
    )
    version = session_service.create_report_version(
        session_id=session_id,
        status=version_status,
        html_path=report_info.get("html_path"),
        pdf_path=report_info.get("pdf_path"),
        snapshot=report_info.get("snapshot") or {},
    )
    if version_status == "final":
        session_service.set_status(session_id, "finalized")
    return {"version": version, **report_info}


@web_report_api.route("/session/<session_id>/report/regenerate", methods=["POST"])
def regenerate_report(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": "Invalid session"}), 404
    if not session.get("ai_result"):
        return jsonify({"success": False, "error": "Analysis not ready"}), 409

    report_info = _generate_report_version(session_id, "draft")
    return jsonify(
        {
            "success": True,
            "version": report_info["version"],
            "report_url": f"/api/web_report/session/{session_id}/report",
            "page_url": f"/report/{session_id}",
        }
    )


@web_report_api.route("/session/<session_id>/report/finalize", methods=["POST"])
def finalize_report(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": "Invalid session"}), 404
    if not session.get("ai_result"):
        return jsonify({"success": False, "error": "Analysis not ready"}), 409

    report_info = _generate_report_version(session_id, "final")
    return jsonify(
        {
            "success": True,
            "version": report_info["version"],
            "report_url": f"/api/web_report/session/{session_id}/report",
            "pdf_url": f"/api/web_report/session/{session_id}/report/pdf",
        }
    )


@web_report_api.route("/session/<session_id>/report/versions", methods=["GET"])
def list_report_versions(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": "Invalid session"}), 404
    return jsonify({"success": True, "versions": session_service.list_report_versions(session_id)})


@web_report_api.route("/session/<session_id>/report", methods=["GET"])
def get_report_html(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": "Invalid session"}), 404
    report = session.get("report")
    if not report or not report.get("html_path"):
        return jsonify({"success": False, "error": "Report not ready"}), 404
    return send_file(report["html_path"], mimetype="text/html")


@web_report_api.route("/session/<session_id>/report/pdf", methods=["GET"])
def get_report_pdf(session_id: str):
    session = session_service.get_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": "Invalid session"}), 404
    report = session.get("report")
    if not report or not report.get("pdf_path"):
        return jsonify({"success": False, "error": "PDF not available"}), 404
    return send_file(report["pdf_path"], mimetype="application/pdf")


@web_report_api.route("/session/<session_id>/files/<path:relative_path>", methods=["GET"])
def get_session_file(session_id: str, relative_path: str):
    session_root = _session_root(session_id).resolve()
    requested = (session_root / relative_path).resolve()
    if session_root not in requested.parents and requested != session_root:
        return jsonify({"success": False, "error": "Invalid path"}), 400
    if not requested.exists() or not requested.is_file():
        return jsonify({"success": False, "error": "File not found"}), 404
    return send_file(requested)

from flask import Blueprint, request, jsonify, send_from_directory, current_app
import os
import time
import tempfile
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import threading # [NEW] For async inference
import json
import traceback
from PIL import Image # [NEW] For Gemini Vision

from services.image_loader import extract_dicom_meta
from services.upload_handler import parse_any_upload
from services.analysis_service import analysis_service
from services.result_builder import result_builder
from services.report_dictation_service import report_dictation_service
from config import Config

api_v2 = Blueprint('api_v2', __name__)

# Initialize Services
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "temp"
REPORTS_DIR = BASE_DIR / "reports"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Warm up models
analysis_service.initialize(
    model_dir=str(BASE_DIR / "weights"),
    model_cfg=Config.PANO_MODELS
)

# ---------------------------------------------------------------------------
# Rate Limiting (In-Memory)
# ---------------------------------------------------------------------------
# Simple token bucket / sliding window per IP
import collections
IP_REQUEST_LOGS = collections.defaultdict(list)
RATE_LIMIT_MINUTES = 1
RATE_LIMIT_REQUESTS = 10

def is_rate_limited(ip_address: str) -> bool:
    """Returns True if the IP has exceeded 10 requests in the last 1 minute."""
    now = time.time()
    # Clean up old timestamps for this IP
    cutoff = now - (RATE_LIMIT_MINUTES * 60)
    IP_REQUEST_LOGS[ip_address] = [ts for ts in IP_REQUEST_LOGS[ip_address] if ts > cutoff]
    
    if len(IP_REQUEST_LOGS[ip_address]) >= RATE_LIMIT_REQUESTS:
        return True
        
    # Log the new request
    IP_REQUEST_LOGS[ip_address].append(now)
    return False

# ---------------------------------------------------------------------------
# Security Helpers
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}

def is_allowed_file(filename, file_path=None):
    """
    1. Check extension whitelist.
    2. Optional: check Magic Bytes (MIME) if file_path is provided.
    """
    if '.' not in filename:
        return False, "No file extension found."
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Extension '{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}"
        
    # Magic bytes check (Payload validation)
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                header = f.read(132) # Read enough for DICOM
                
            if ext in ['jpg', 'jpeg']:
                if not header.startswith(b'\xff\xd8'):
                    return False, "Invalid JPEG Magic Bytes (Payload Spoofing Detected)."
            elif ext == 'png':
                if not header.startswith(b'\x89PNG\r\n\x1a\n'):
                    return False, "Invalid PNG Magic Bytes (Payload Spoofing Detected)."
            elif ext in ['dcm', 'dicom']:
                # DICOM magic bytes 'DICM' are at offset 128
                if len(header) < 132 or header[128:132] != b'DICM':
                    try:
                        import pydicom
                        pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                    except Exception:
                        return False, "Invalid DICOM file signature."
        except Exception as e:
            return False, f"Failed to read file signature: {e}"
            
    return True, "Valid"

# ---------------------------------------------------------------------------
# JSON-safe conversion helper
# ---------------------------------------------------------------------------
def to_jsonable(obj):
    """
    Recursively convert numpy / Path / set objects so Flask jsonify can handle them.
    Leaves basic python scalars untouched.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    return obj

@api_v2.route('/analyze', methods=['POST'])
def analyze_gpt():
    """
    GPTs Integration Endpoint.
    Input:
      - image: File
      - user_name: String (For report filename)
    Output:
      - JSON with findings
    """
    if is_rate_limited(request.remote_addr or "unknown_ip"):
        return jsonify({"error": "Too Many Requests. Please try again in a minute."}), 429

    # 1. Parse and Validate Upload
    upload = parse_any_upload(request, UPLOAD_FOLDER)
    if not upload.is_valid:
        return jsonify({"error": upload.error}), 400

    # 2. Skip AI for 3D Volume
    if upload.is_volume:
        return jsonify({
            "status": "volume_detected",
            "message": "3D CBCT volume detected. Analysis currently supports 2D panoramic images only.",
            "user_name": upload.user_name
        })

    # 3. Create Async Task
    import uuid
    task_id = str(uuid.uuid4())
    output_dir = UPLOAD_FOLDER / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_service.run_async_inference(
        task_id=task_id,
        image_path=Path(upload.save_path),
        output_dir=output_dir,
        user_name=upload.user_name,
        language=upload.language
    )

    return jsonify({
        "status": "success",
        "task_id": task_id,
        "message": "Analysis started. Please poll status.",
        "user_name": upload.user_name
    })

@api_v2.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    status = analysis_service.get_task_status(task_id)
    if status['status'] == 'completed':
        # Use result_builder to format for frontend
        formatted = result_builder.build_inference_response(
            task_id, 
            status['result'], 
            status['result'].get('user_name', 'Patient'),
            status['result'].get('language', 'English')
        )
        return jsonify(formatted)
    
    return jsonify(status)



@api_v2.route('/files/<path:filename>', methods=['GET'])
def serve_files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@api_v2.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)
# --- Session Management for External Uploads ---
# In-memory store (cleared on restart)
SESSIONS = {} 

@api_v2.route('/session', methods=['POST'])
def create_session():
    import uuid
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = { "status": "waiting" }
    
    # Construct Upload URL (Ngrok aware)
    base_url = request.host_url.rstrip('/')
    # If host ends with ngrok-free.app, force https
    if 'ngrok-free.app' in base_url:
        base_url = base_url.replace("http://", "https://")
        
    upload_url = f"{base_url}/api/v2/upload_page/{session_id}"
    
    return jsonify({
        "session_id": session_id,
        "upload_url": upload_url,
        "message": "Present this URL to the user."
    })

# [NEW] Convenience Route for Direct Browser Access
@api_v2.route('/upload', methods=['GET'])
def start_upload_session():
    import uuid
    from flask import redirect
    
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = { "status": "waiting" }
    
    # Redirect to the upload page
    return redirect(f"/api/v2/upload_page/{session_id}")

@api_v2.route('/upload_page/<session_id>', methods=['GET'])
def serve_upload_page(session_id):
    # [STATS] Log Visit
    stats_service.log_visit(session_id)

    if session_id not in SESSIONS:
        return "Invalid or Expired Session", 404
        
    session = SESSIONS[session_id]
    if session.get('status') == 'completed':
        return "<div style='text-align:center; padding:50px; font-family:sans-serif;'><h1 style='color:green;'>Upload & Analysis Complete!</h1><p>Please return to the Chat to see the result.</p></div>"
    if session.get('status') == 'processing':
        return "<h1>Processing...</h1>"

    # Serve Template
    template_path = os.path.join(current_app.root_path, 'templates', 'upload_page.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Inject action URL dynamically to avoid 404
    action_url = f"/api/v2/upload_process/{session_id}"
    # Replace action="" if it exists, or inject it into <form ...>
    if 'action=""' in html:
        html = html.replace('action=""', f'action="{action_url}"')
    elif '<form id="uploadForm"' in html:
        html = html.replace('<form id="uploadForm"', f'<form id="uploadForm" action="{action_url}"')
    
    return html

def run_async_inference(session_id, save_path, dir_inference, dir_crop, base_url):
    """
    Background worker for inference.
    Updates SESSIONS state when done.
    """
    try:
        print(f"[{session_id}] Starting Async Inference...")
        
        # 1. Run Inference
        result_map = inference_service.run(Path(save_path), Path(dir_inference))
        
        # 2. Extract Data
        teeth = result_map.get("teeth_objects", [])
        caries_list = result_map.get("caries_objects", [])
        periapical_list = result_map.get("periapical_objects", [])
        teeth_map = {t['tooth_label']: t for t in teeth if t.get('tooth_label')}
        
        missing_teeth = result_map.get("missing_teeth", [])
        
        # 3. Generate Structured Feedback
        # Re-using FeedbackGenerator logic (but adapted for strict JSON output?)
        # Actually, let's just generate the HTML report here to be ready.
        
        # Prepare Analysis Result for Report/Feedback
        # Need to convert list back to "teeth" map structure if expected
        analysis_result = {
            "teeth": teeth,
            "missing_teeth": missing_teeth,
            "caries": caries_list,
            "periapical": periapical_list,
            "nerve_contours": result_map.get("nerve_contours", []),
            "sinus_contours": result_map.get("sinus_contours", [])
        }
        
        print(f"[{session_id}] DEBUG: Inference Done. Result Keys: {analysis_result.keys()}")
        
        # Generate Feedback Text (Markdown)
        print(f"[{session_id}] DEBUG: Starting Feedback Generator...")
        orig_img = cv2.imread(save_path)
        
        # Generate HTML & PDF Report
        print(f"[{session_id}] DEBUG: Starting Report Generator...")
        report_gen = ReportGeneratorV3()
        overlay_path = Path(dir_inference) / "overlay.png"
        bl_viz_path = Path(dir_inference) / "bl_viz.png"
        
        # [MODIFIED] Use Overlay Image for Feedback
        print(f"[{session_id}] DEBUG: Loading Overlay for Feedback...")
        feedback_img = cv2.imread(str(overlay_path))
        if feedback_img is None: feedback_img = orig_img 

        print(f"[{session_id}] DEBUG: Calling fb_gen.generate_structured...")
        fb_gen = FeedbackGenerator(base_url=base_url.rstrip('/'), upload_folder=UPLOAD_FOLDER)
        _, findings_list, _ = fb_gen.generate_structured(
            user_name="User", 
            analysis_result=analysis_result,
            original_img=feedback_img, 
            iac_result=result_map.get('iac_raw_res'),
            base_url=base_url.rstrip('/'),
            crop_dir=dir_crop,
            crop_url_prefix=f"{base_url}api/v2/crop/{session_id}"
        )
        print(f"[{session_id}] DEBUG: Feedback Generated.")
        
        print(f"[{session_id}] DEBUG: Calling Gemini Per-Tooth Analysis...")
        # [NEW] Per-Tooth Gemini Analysis (replaces global AI commentary)
        per_tooth_analysis = {}
        try:
            problem_teeth = []
            for t in teeth:
                fdi = t.get('tooth_label', '')
                has_caries = t.get('caries', False)
                has_perio = t.get('periapical', False)
                bl_lvl = t.get('bone_loss_level', 0)
                bl_pct = t.get('bone_loss_pct', 0)
                nerve = t.get('nerve_overlap', False)
                sinus = t.get('sinus_overlap', False)
                
                send_bl_lvl = 0
                send_bl_pct = 0.0
                if bl_lvl > 0:
                    send_bl_lvl = int(bl_lvl)
                    send_bl_pct = round(float(bl_pct), 1)

                # Only include teeth with ACTUAL AI-detected pathology
                is_significant = has_caries or has_perio or nerve or sinus or bl_lvl >= 3
                if is_significant:
                    findings = []
                    if has_caries: findings.append('caries')
                    if has_perio: findings.append('periapical')
                    if bl_lvl >= 3: findings.append('bone_loss')
                    problem_teeth.append({
                        "tooth_number": str(fdi),
                        "findings": findings,
                        "bone_loss_level": send_bl_lvl,
                        "bone_loss_pct": send_bl_pct,
                        "nerve_overlap": bool(nerve),
                        "sinus_overlap": bool(sinus)
                    })
            
            if problem_teeth:
                session = SESSIONS.get(session_id, {})
                lang = session.get('language', 'English')
                
                # Prepare crops for problem teeth
                problem_images_map = {}
                try:
                    pil_main = Image.open(save_path)
                    for pt in problem_teeth:
                        lbl = pt.get('tooth_number')
                        # Find the box in the 'teeth' list
                        mapped_tooth = next((t for t in teeth if str(t.get('tooth_label', '')) == str(lbl)), None)
                        box = mapped_tooth.get('box') if mapped_tooth else None
                        if lbl and box:
                            w, h = pil_main.size
                            x1, y1, x2, y2 = map(int, box)
                            pad = 20
                            cx1 = max(0, x1 - pad)
                            cy1 = max(0, y1 - pad)
                            cx2 = min(w, x2 + pad)
                            cy2 = min(h, y2 + pad)
                            crop = pil_main.crop((cx1, cy1, cx2, cy2))
                            problem_images_map[str(lbl)] = crop
                except Exception as e:
                    print(f"[{session_id}] Failed to create crops for problem teeth: {e}")

                per_tooth_analysis = gemini_service.generate_tooth_analysis(problem_teeth, images_map=problem_images_map, language=lang)
                # pass
        except Exception as e:
            print(f"[{session_id}] Per-tooth Analysis Error: {e}")
            traceback.print_exc()
            per_tooth_analysis = {}
        print(f"[{session_id}] DEBUG: Per-Tooth Analysis Generated for {len(per_tooth_analysis)} teeth")

        session_root = Path(dir_inference).parent
        print(f"[{session_id}] DEBUG: Calling report_gen.generate to {session_root}...")
        report_path, report_filename, generated_pdf_filename = report_gen.generate(
            user_name="User",
            image_path=Path(save_path),
            analysis_result=analysis_result, 
            overlay_path=overlay_path,
            bl_viz_path=bl_viz_path,
            ai_commentary="",  # [REMOVED] No longer using global AI insight
            detailed_findings=findings_list,
            llm_analysis={},
            per_tooth_analysis=per_tooth_analysis,  # [NEW]
            output_dir=str(session_root), # [USER REQUEST] Save to temp session folder
            language=lang,
        )
        print(f"[{session_id}] DEBUG: Report Generated at {report_path}")
        
        # Normalize Base URL (Strip trailing slash)
        base_url_clean = base_url.rstrip('/')
        session_root_name = session_root.name
        
        report_pdf_url = None
        if generated_pdf_filename:
             report_pdf_url = f"{base_url_clean}/api/v2/files/{session_root_name}/{generated_pdf_filename}"
        
        # Serve report from the temp files endpoint
        report_url = f"{base_url_clean}/api/v2/files/{session_root_name}/{report_filename}"
        
        print(f"[{session_id}] DEBUG: Final Report URL: {report_url}")

        # Update Session
        SESSIONS[session_id]['status'] = 'completed'
        # Define session_root_name for overlay_url
        session_root_name = Path(dir_inference).parent.name
        overlay_url = f"{base_url_clean}/api/v2/files/{session_root_name}/inference/overlay.png"
        print(f"[{session_id}] DEBUG: Final Overlay URL: {overlay_url}")
        
        # Read Report Content for Direct Injection
        report_html_content = ""
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_html_content = f.read()
        except Exception as e:
            print(f"[{session_id}] Failed to read report content: {e}")

        # [FIX] Calculate total_teeth based on unique FDI numbers
        unique_teeth_fdi = set()
        for t in teeth:
            lbl = t.get('tooth_label')
            if lbl and str(lbl).isdigit():
                tn = int(lbl)
                if 11 <= tn <= 48:
                    unique_teeth_fdi.add(tn)
        
        # Build structured model findings for GPTs
        model_findings_async = {
            "total_teeth": len(unique_teeth_fdi),
            "missing_teeth": [str(mt.get('tooth_label','')) for mt in missing_teeth] if missing_teeth else [],
            "problem_teeth": [],
            "per_tooth_ai_analysis": per_tooth_analysis or {}
        }
        for t in teeth:
            fdi = str(t.get('tooth_label', ''))
            issues = []
            if t.get('caries'): issues.append('충치')
            if t.get('periapical'): issues.append('치근단 병소')
            bl_lvl = t.get('bone_loss_level', 0)
            bl_pct = t.get('bone_loss_pct', 0)
            if (t.get('caries') or t.get('periapical')) and bl_lvl >= 3: issues.append(f'치조골 소실 Level {bl_lvl} ({bl_pct:.0f}%)')
            if t.get('nerve_overlap'): issues.append('하악 신경관 근접')
            if t.get('sinus_overlap'): issues.append('상악동 근접')
            
            # [FILTER] Skip if the only issue is '상악동 근접'
            if len(issues) == 1 and issues[0] == '상악동 근접':
                pass # Skip
            elif issues:
                model_findings_async["problem_teeth"].append({
                    "tooth": fdi,
                    "issues": issues
                })
        
        # [MODIFIED] Use Gemini to generate Technical Output Text independent of JSON dump
        # Retrieve Language Preference (Default: English)
        session_config = SESSIONS.get(session_id, {}).get('config', {})
        target_language = session_config.get('language', 'English')
        
        findings_text_async = gemini_service.generate_technical_report(model_findings_async, language=target_language)

        analysis_result_json = to_jsonable(analysis_result)
        SESSIONS[session_id]['result'] = {
            "message": "Analysis Complete",
            "gpt_text": findings_text_async,
            "report_url": report_url,
            "report_pdf_url": report_pdf_url,
            "overlay_url": overlay_url,
            "report_html": report_html_content, # Legacy Support
            "analysis_result": analysis_result_json, # [NEW] Raw Data for React UI (JSON-safe)
            "ai_commentary": findings_text_async
        }
        print(f"[{session_id}] Inference Complete.")
        
        # [MODIFIED] Return HTML if requested (for Test Page Loop)
        # REMOVED request access here to fix "Working outside of request context"
        # Since this runs in a thread, we cannot access request.
        # The calling function (process_upload_session) should handle the response.

        # [STATS] Log Findings
        stats_service.log_inference(session_id, model_findings_async)

        # return jsonify({...}) # Also removed jsonify return as this function shouldn't return HTTP response directly
        return # End of worker
        
    except Exception as e:
        print(f"[{session_id}] Inference Failed: {e}")
        import traceback
        traceback.print_exc()
        SESSIONS[session_id]['status'] = 'failed'
        SESSIONS[session_id]['error'] = str(e)
        return

@api_v2.route('/upload_page/<session_id>', methods=['GET'])
def upload_page(session_id):
    if session_id not in SESSIONS:
        return "Invalid Session ID", 404

# [NEW] Convenience Route for Testing
@api_v2.route('/test', methods=['GET'])
def simple_test():
    """renders the test page directly with a fixed session ID 'test' (derived from URL)."""
    return render_template('test_page.html')

@api_v2.route('/test_page/<session_id>', methods=['GET'])
def test_page(session_id):
    # Specialized page for real-time testing (no auto-close, shows overlay)
    if session_id not in SESSIONS:
        return "Invalid Session ID", 404
    return render_template('test_page.html')

@api_v2.route('/upload_process/<session_id>', methods=['POST'])
def process_upload_session(session_id):
    # Rate Limiting Check
    client_ip = request.remote_addr or "unknown_ip"
    if is_rate_limited(client_ip):
        current_app.logger.warning(f"Rate limit exceeded for IP: {client_ip} on Web UI")
        return "Too Many Requests. Please try again in a minute.", 429

    # [MODIFIED] Auto-create session if it doesn't exist (Support client-side UUID)
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            'status': 'initialized', 
            'created_at': time.time(),
            'result': None
        }
        # return "Invalid Session", 404 # Removed 404 restriction
        
    # Debug logging
    current_app.logger.info(f"Request files: {request.files}")
    current_app.logger.info(f"Request form: {request.form}")
    current_app.logger.info(f"Content-Type: {request.content_type}")
    
    if 'file' not in request.files:
        current_app.logger.error("No 'file' in request.files")
        return "No file uploaded", 400
        
    file = request.files['file']
    if file.filename == '':
        current_app.logger.error("Empty filename")
        return "Empty file", 400

    try:
        SESSIONS[session_id]['status'] = 'processing'
        
        # Security: Secure filename & Whitelist
        original_filename = secure_filename(file.filename)
        is_valid, reason = is_allowed_file(original_filename)
        if not is_valid:
            current_app.logger.warning(f"Rejected Web UI upload: {reason}")
            return reason, 400
            
        timestamp = int(time.time())
        ext = original_filename.split('.')[-1].lower() if '.' in original_filename else 'jpg'
        
        # New Feature: Organized File Structure
        # Root: temp/{timestamp}_session_{sessionid}
        session_root_name = f"{timestamp}_session_{session_id}"
        session_root_path = os.path.join(UPLOAD_FOLDER, session_root_name)
        
        dir_original = os.path.join(session_root_path, "original")
        dir_inference = os.path.join(session_root_path, "inference")
        dir_crop = os.path.join(session_root_path, "crop")
        
        os.makedirs(dir_original, exist_ok=True)
        os.makedirs(dir_inference, exist_ok=True)
        os.makedirs(dir_crop, exist_ok=True)
        
        # Save Original Securely (Random UUID)
        import uuid
        random_filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(dir_original, random_filename)
        file.save(save_path)
        
        # Security: Magic Bytes / Payload Validation
        is_valid, reason = is_allowed_file(original_filename, save_path)
        if not is_valid:
            os.remove(save_path)
            current_app.logger.warning(f"[{save_path}] Payload Rejected from Web UI: {reason}")
            return reason, 400
        
        
        # 0. Pre-Flight Check: Is this an X-Ray / Grayscale Panorama?
        import numpy as np
        try:
            chk_img = load_image_any(Path(save_path))
        except Exception:
            return "File is not a valid image format.", 400
            
        b, g, r = cv2.split(chk_img)
        mean_diff = (np.mean(cv2.absdiff(b, g)) + np.mean(cv2.absdiff(g, r)) + np.mean(cv2.absdiff(r, b))) / 3.0
        
        if mean_diff > 15.0:
            current_app.logger.warning(f"[{save_path}] Rejected from Web UI: Image appears to be a color photo (color diff: {mean_diff:.1f}).")
            
            # [SECURITY] Completely remove the empty session folder structure immediately
            import shutil
            try:
                shutil.rmtree(session_root_path, ignore_errors=True)
            except Exception as e:
                current_app.logger.error(f"Failed to delete rejected session root {session_root_path}: {e}")
                
            return "The uploaded image does not appear to be a dental x-ray panorama. Please upload a valid grayscale x-ray.", 400
        
        # Threading for Async
        import threading
        
        # [MODIFIED] Check if we want synchronous execution for Test Page
        # We grab base_url here because we can't inside the thread (if async)
        # However, for synchronous, we can just run it.
        
        base_url = request.host_url # This line was already present before the change, keeping it here for context.
        if request.args.get('render_html'):
            # Run Synchronously
            run_async_inference(session_id, save_path, dir_inference, dir_crop, base_url)
            
            # Check result
            if SESSIONS[session_id]['status'] == 'completed':
                overlay_url = SESSIONS[session_id]['result']['overlay_url']
                report_url = SESSIONS[session_id]['result'].get('report_url') # Keep for download link
                report_html = SESSIONS[session_id]['result'].get('report_html', '')
                return render_template('test_result_view.html', 
                                     session_id=session_id, 
                                     overlay_url=overlay_url, 
                                     report_url=report_url,
                                     report_html=report_html)
            else:
                err = SESSIONS[session_id].get('error', 'Unknown Error')
                return f"<h1>Analysis Failed</h1><p>{err}</p>", 500
        else:
            # Run Async (Standard Flow)
            thread = threading.Thread(target=run_async_inference, args=(session_id, save_path, dir_inference, dir_crop, base_url))
            thread.start()
            
            return jsonify({
                "status": "processing",
                "message": "File uploaded, analysis started in background."
            })

    except Exception as e:
        current_app.logger.error(f"Error in upload process: {e}")
        return str(e), 500
        


@api_v2.route('/result/<session_id>', methods=['GET'])
def get_session_result(session_id):
    if session_id not in SESSIONS:
        return jsonify({"status": "not_found"}), 404
    
    # Make JSON safe
    safe_payload = to_jsonable(SESSIONS[session_id])

    # Trim very large fields by default (keep links lightweight)
    result = safe_payload.get("result")
    include_html = request.args.get("include_html") in ("1", "true", "yes")
    include_details = request.args.get("include_details") in ("1", "true", "yes")
    if result:
        # Only serve report_html when explicitly requested
        if not include_html:
            result.pop("report_html", None)
        # Text-only default: strip heavy/non-text payloads unless details explicitly requested
        if not include_details:
            for key in ("overlay_url", "report_url", "report_pdf_url", "analysis_result"):
                result.pop(key, None)

    return jsonify(safe_payload)

@api_v2.route('/generate_report/<session_id>', methods=['POST'])
def generate_session_report(session_id):
    if session_id not in SESSIONS:
        return jsonify({"error": "Invalid Session"}), 404
    
    session = SESSIONS[session_id]
    result = session.get('result', {})
    report_url = result.get('report_url')
    
    if report_url:
         return jsonify({
            "status": "success",
            "report_url": report_url,
            "message": "Report available."
        })
    else:
        # Fallback (if eager failed or not ready)
        return jsonify({"error": "Report not available yet or generation failed."}), 404

        return jsonify({"error": str(e)}), 500

@api_v2.route('/session/<session_id>', methods=['GET'])
def get_session_status(session_id):
    if session_id not in SESSIONS:
        # If not found but we are in a permissive mode, maybe return a "not found" status instead of 404?
        # But logically it should exist if they started upload.
        return jsonify({'status': 'not_found', 'error': 'Session not found'}), 404
        
    session_data = SESSIONS[session_id]
    
    return jsonify({
        'status': session_data.get('status', 'waiting'),
        'error': session_data.get('error'),
        'result': session_data.get('result', {}),
        'config': session_data.get('config', {'language': 'English'}) # [NEW]
    })

# [STATS] Dashboard Endpoints
@api_v2.route('/stats/data', methods=['GET'])
def get_stats_data():
    """Return aggregated stats for dashboard."""
    data = stats_service.get_dashboard_data()
    return jsonify(data)

@api_v2.route('/dashboard', methods=['GET'])
def serve_dashboard():
    """Serve the statistics dashboard page."""
    template_path = os.path.join(current_app.root_path, 'templates', 'dashboard.html')
    if not os.path.exists(template_path):
        return "Dashboard template not found.", 404
        
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

# [CONFIG] Language Settings
@api_v2.route('/session/<session_id>/config', methods=['POST'])
def update_session_config(session_id):
    """
    Update session configuration (e.g. Language).
    Input: { "language": "Korean" | "English" }
    """
    if session_id not in SESSIONS:
        return jsonify({"error": "Invalid Session"}), 404
        
    data = request.get_json()
    language = data.get('language', 'English')
    
    # Store in session
    if 'config' not in SESSIONS[session_id]:
        SESSIONS[session_id]['config'] = {}
    
    SESSIONS[session_id]['config']['language'] = language
    
    return jsonify({
        "status": "success",
        "message": f"Language set to {language}",
        "config": SESSIONS[session_id]['config']
    })
    # If completed, include the full result
    response = {
        'status': session_data.get('status', 'unknown'),
        'result': session_data.get('result')
    }
    
    # If failed, include error
    if 'error' in session_data:
        response['error'] = session_data['error']
        
    return jsonify(response)

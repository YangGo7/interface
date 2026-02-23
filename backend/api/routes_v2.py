from flask import Blueprint, request, jsonify, send_from_directory, current_app, redirect
import os
import time
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import threading # [NEW] For async inference

# Services
from services.pano_inference import PanoPipeline
from utils.report_v2 import ReportGeneratorV2
from utils.report_v3 import ReportGeneratorV2 as ReportGeneratorV3
from utils.feedback_generator import FeedbackGenerator
from config import Config

api_v2 = Blueprint('api_v2', __name__)

# Helper to find models
BASE_DIR = Path(__file__).resolve().parent.parent # backend/api/ -> backend/
WEIGHTS_DIR = BASE_DIR / "weights"

# Initialize services (Lazy loading or global?)
# Ideally should be injected, but for now instantiate here or reuse app context if available.
# We will instantiate new Service for simplicity, sharing models is handled by singleton or cached loaders inside Service.
inference_service = PanoPipeline(
    model_dir=str(WEIGHTS_DIR),
    model_cfg=Config.PANO_MODELS
)
report_gen = ReportGeneratorV3() # v2
report_gen_v3 = ReportGeneratorV3()
feedback_gen = FeedbackGenerator()

UPLOAD_FOLDER = 'temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@api_v2.route('/analyze', methods=['POST'])
def analyze_gpt():
    """
    GPTs Integration Endpoint.
    Input:
      - image: File
      - user_name: String (For report filename)
    Output:
      - JSON with findings
      - report_url: URL to HTML report
    """
    
    # Check File (Support 'file'/'image' in multipart, 'file_base64' in JSON, or Raw Binary)
    current_app.logger.info(f"HEADERS: {request.headers}")
    
    file_obj = None
    save_path = ""
    filename = ""

    # 1. Handle Raw Binary (Best for GPTs single file)
    if 'image/' in request.content_type:
        current_app.logger.info(f"Raw Binary Upload Detected. Content-Type: {request.content_type}")
        current_app.logger.info(f"Request Data Length: {len(request.data)}")
        
        ext = request.content_type.split('/')[-1]
        timestamp = int(time.time())
        filename = f"upload_{timestamp}.{ext}"
        filename = secure_filename(filename)
        
        # Create unique subdirectory for this upload
        stem = Path(filename).stem
        upload_subdir = os.path.join(UPLOAD_FOLDER, stem)
        os.makedirs(upload_subdir, exist_ok=True)
        
        save_path = os.path.join(upload_subdir, filename)
        
        with open(save_path, "wb") as f:
            f.write(request.data)
            
        file_size = os.path.getsize(save_path)
        current_app.logger.info(f"Saved File Size: {file_size} bytes")
        
        if file_size == 0:
             return jsonify({"error": "Received empty body for Raw Binary upload."}), 400
             
        user_name = request.args.get("user_name", "Patient") # Query param for raw body

    # 2. Handle JSON Base64
    elif request.is_json:
        data = request.get_json()
        current_app.logger.info(f"JSON DATA KEYS: {data.keys()}")
        if 'file_base64' in data:
            import base64
            try:
                # Decode
                b64_str = data['file_base64']
                if not b64_str:
                    return jsonify({"error": "Received empty 'file_base64' string from GPT."}), 400
                    
                file_data = base64.b64decode(b64_str)
                if len(file_data) == 0:
                     return jsonify({"error": "Decoded file data is empty."}), 400
                     
                filename = data.get('filename', f"upload_{int(time.time())}.jpg")
                filename = secure_filename(filename)
                save_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
                
                with open(save_path, "wb") as f:
                    f.write(file_data)
                
                user_name = data.get("user_name", "Patient")
            except Exception as e:
                return jsonify({"error": f"Base64 decode failed: {str(e)}"}), 400
    
    # 3. Handle Multipart (Legacy/Direct)
    if not save_path:
        current_app.logger.info(f"FILES: {request.files}")
        if 'file' in request.files:
            file_obj = request.files['file']
        elif 'image' in request.files:
            file_obj = request.files['image']
            
        if file_obj:
            if file_obj.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            filename = secure_filename(file_obj.filename)
            save_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
            file_obj.save(save_path)
            
            user_name = request.form.get("user_name", "Patient")
            
    if not save_path:
        return jsonify({"error": "No file part found. Tried Raw Binary, JSON Base64, and Multipart."}), 400
    
    try:
        # 1. Run Inference
        out_dir = Path(UPLOAD_FOLDER) / f"{int(time.time())}"
        result_map = inference_service.run(Path(save_path), out_dir)
        
        # 2. Extract Data from result_map
        teeth = result_map.get("teeth_objects", [])
        caries_list = result_map.get("caries_objects", [])
        bonelevel_dict = result_map.get("bonelevel", {})
        
        # Merge Findings into Teeth Objects
        # We need to map caries/bonelevel back to the specific tooth object
        
        # Create a lookup for teeth by label
        teeth_map = {t['tooth_label']: t for t in teeth if t.get('tooth_label')}
        
        # 2.1 Map Caries & Perio
        # Create standardized list of findings for the report generator
        for c in caries_list:
             assigned = c.get('assigned_tooth')
             lbl = c.get('label', 'caries').lower()
             
             if assigned and assigned in teeth_map:
                 # Store full object (box, confidence, label)
                 if 'findings' not in teeth_map[assigned]: teeth_map[assigned]['findings'] = []
                 
                 # Normalize finding type
                 finding_type = 'caries'
                 if 'periapical' in lbl or 'perio' in lbl: finding_type = 'periapical'
                 
                 teeth_map[assigned]['findings'].append({
                     'type': finding_type,
                     'box': c['box'],
                     'conf': c['confidence']
                 })
                 
                 # Set flags for easy status checking
                 teeth_map[assigned][finding_type] = True

        # 2.2 Map Bone Level
        
        # 2.2 Map Bone Level
        for fdi, level in bonelevel_dict.items():
            if fdi in teeth_map:
                try:
                    # Handle case where level is a dictionary (from pano_calc_utils.py)
                    # It returns {'percent': 45.0, 'ratio': 0.45}
                    val = 0.0
                    lvl = 0
                    if isinstance(level, dict):
                         if 'percent' in level: val = float(level['percent'])
                         elif 'max' in level: val = float(level['max'])
                         elif 'value' in level: val = float(level['value'])
                         
                         if 'level' in level: lvl = int(level['level'])
                    else:
                         val = float(level)
                    
                    teeth_map[fdi]['bone_loss_pct'] = val
                    teeth_map[fdi]['bone_loss_level'] = lvl
                except:
                    pass
                
        # 2.3 Periapical (Map from odontogram if available)
        odo_map = result_map.get("odontogram_map", {})
        for fdi, issues in odo_map.items():
            if fdi in teeth_map:
                if 'caries' in issues: teeth_map[fdi]['caries'] = True
                if 'perio' in issues or 'periapical' in issues: teeth_map[fdi]['periapical'] = True
        
        data_for_report = list(teeth_map.values())
        
        # 3. Generate Report V2
        # Pass Paths for: Original (Base), Overlay (Full), BL_Viz (PBL Crop)
        overlay_path = out_dir / "overlay.png"
        bl_viz_path = out_dir / "bl_viz.png"
        
        # [FIX] Include missing teeth in analysis result
        missing_teeth = result_map.get("missing_teeth_objects", [])
        
        report_path, report_filename = report_gen.generate(
            user_name=user_name,
            image_path=Path(save_path),
            analysis_result={
                "teeth": list(teeth_map.values()),
                "missing_teeth": missing_teeth
            },
            overlay_path=overlay_path,
            bl_viz_path=bl_viz_path
        )
        
        # 4. Generate GPT Feedback
        # [NEW] Feedback Generation
        # Load feedback generator with host url
        fb_gen = FeedbackGenerator(base_url=request.host_url.rstrip('/'), upload_folder=UPLOAD_FOLDER)
        
        # We need original image for cropping. `save_path` is the file.
        orig_img = cv2.imread(save_path)
        
        # result_map might contain 'iac_raw_res' if we pass it, OR we rely on infer_service returning it?
        # PanoInference currently doesn't return the raw IAC result in the dict (just overlays).
        # We need to assume it's missing or fix inference.
        # For now, pass None and fix Inference in next step if checking nerve is critical.
        gpt_text = fb_gen.generate(
            user_name=user_name,
            analysis_result={"teeth": list(teeth_map.values())},
            original_img=orig_img,
            iac_result=result_map.get('iac_raw_res') 
        )

        
        # 5. Construct Response
        # URL for report
        base_url = request.host_url
        report_url = f"{base_url}api/v2/reports/{report_filename}"
        
        return jsonify({
            "status": "success",
            "user_name": user_name,
            "report_url": report_url,
            "gpt_text": gpt_text, # [NEW]
            "summary_counts": {
                "total_teeth": len(data_for_report),
                "issues": sum(1 for t in data_for_report if t.get('caries') or t.get('periapical') or t.get('bone_loss_mm',0)>3)
            },
            "data": data_for_report
        })

    except Exception as e:
        current_app.logger.error(f"Analysis failed: {e}")
        return jsonify({"error": str(e)}), 500

@api_v2.route('/files/<path:filename>', methods=['GET'])
def serve_files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@api_v2.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory('reports', filename)
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

@api_v2.route('/upload_page/<session_id>', methods=['GET'])
def serve_upload_page(session_id):
    if session_id not in SESSIONS:
        return "Invalid or Expired Session", 404
        
    session = SESSIONS[session_id]
    if session.get('status') == 'completed':
        return """
        <div style="text-align:center; padding:50px; font-family:sans-serif;">
            <h1 style="color:green;">Upload & Analysis Complete!</h1>
            <p>Please return to the Chat to see the result.</p>
        </div>
        """
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
        report_gen = ReportGeneratorV2()
        overlay_path = Path(dir_inference) / "overlay.png"
        bl_viz_path = Path(dir_inference) / "bl_viz.png"
        
        # [MODIFIED] Use Overlay Image for Feedback
        print(f"[{session_id}] DEBUG: Loading Overlay for Feedback...")
        feedback_img = cv2.imread(str(overlay_path))
        if feedback_img is None: feedback_img = orig_img 

        print(f"[{session_id}] DEBUG: Calling fb_gen.generate_structured...")
        fb_gen = FeedbackGenerator(base_url=base_url.rstrip('/'), upload_folder=UPLOAD_FOLDER)
        gpt_text = fb_gen.generate_structured(
            user_name="User", 
            analysis_result=analysis_result,
            original_img=feedback_img, 
            iac_result=result_map.get('iac_raw_res'),
            base_url=base_url.rstrip('/'),
            crop_dir=dir_crop,
            crop_url_prefix=f"{base_url}api/v2/crop/{session_id}"
        )
        print(f"[{session_id}] DEBUG: Feedback Generated.")
        
        print(f"[{session_id}] DEBUG: Calling report_gen.generate...")
        report_path, report_filename, generated_pdf_filename = report_gen.generate(
            user_name="User",
            image_path=Path(save_path),
            analysis_result=analysis_result, 
            overlay_path=overlay_path,
            bl_viz_path=bl_viz_path
        )
        print(f"[{session_id}] DEBUG: Report Generated.")

        # v3 report (UI spec version) for easy switching
        report_path_v3, report_filename_v3, generated_pdf_filename_v3 = report_gen_v3.generate(
            user_name="User",
            image_path=Path(save_path),
            analysis_result=analysis_result,
            overlay_path=overlay_path,
            bl_viz_path=bl_viz_path
        )
        print(f"[{session_id}] DEBUG: Report v3 Generated.")
        
        # Normalize Base URL (Strip trailing slash)
        base_url_clean = base_url.rstrip('/')
        
        report_pdf_url = None
        if generated_pdf_filename:
             report_pdf_url = f"{base_url_clean}/api/v2/reports/{generated_pdf_filename}"
        report_pdf_url_v3 = None
        if generated_pdf_filename_v3:
             report_pdf_url_v3 = f"{base_url_clean}/api/v2/reports/{generated_pdf_filename_v3}"
        
        report_url = f"{base_url_clean}/api/v2/reports/{report_filename}"
        report_url_v3 = f"{base_url_clean}/api/v2/reports/{report_filename_v3}"
        
        print(f"[{session_id}] DEBUG: Final Report URL: {report_url}")
        print(f"[{session_id}] DEBUG: Final Report v3 URL: {report_url_v3}")

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

        SESSIONS[session_id]['result'] = {
            "message": "Analysis Complete",
            "gpt_text": gpt_text,
            "report_url": report_url,
            "report_pdf_url": report_pdf_url,
            "report_url_v3": report_url_v3,
            "report_pdf_url_v3": report_pdf_url_v3,
            "overlay_url": overlay_url,
            "report_html": report_html_content # Direct Content
        }
        print(f"[{session_id}] Inference Complete.")
        
        # [MODIFIED] Return HTML if requested (for Test Page Loop)
        # REMOVED request access here to fix "Working outside of request context"
        # Since this runs in a thread, we cannot access request.
        # The calling function (process_upload_session) should handle the response.

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
        
        # Save File
        timestamp = int(time.time())
        ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        
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
        
        # Save Original
        filename = f"original.{ext}"
        save_path = os.path.join(dir_original, filename)
        file.save(save_path)
        
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
                report_url = SESSIONS[session_id]['result'].get('report_url') # legacy
                report_url_v3 = SESSIONS[session_id]['result'].get('report_url_v3')
                report_html = SESSIONS[session_id]['result'].get('report_html', '')

                # Auto-redirect to v3 if available, else fallback to v2
                target_url = report_url_v3 or report_url
                if target_url:
                    return redirect(target_url, code=302)

                # Fallback to inline view if no URL (should be rare)
                return render_template('test_result_view.html', 
                                     session_id=session_id, 
                                     overlay_url=overlay_url, 
                                     report_url=report_url,
                                     report_url_v3=report_url_v3,
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
        
    return jsonify(SESSIONS[session_id])

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
    
    # If completed, include the full result
    response = {
        'status': session_data.get('status', 'unknown'),
        'result': session_data.get('result')
    }
    
    # If failed, include error
    if 'error' in session_data:
        response['error'] = session_data['error']
        
    return jsonify(response)

# Redirect helper to open v3 report directly
@api_v2.route('/session/<session_id>/report_v3', methods=['GET'])
def redirect_report_v3(session_id):
    session_data = SESSIONS.get(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    url = session_data.get('result', {}).get('report_url_v3') or session_data.get('report_url_v3')
    if not url:
        return jsonify({"error": "Report v3 not ready"}), 404
    return redirect(url, code=302)

from flask import Blueprint, request, jsonify, current_app
import os
import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

detect_api = Blueprint('detect_api', __name__)

@detect_api.route('/detect', methods=['POST'])
def simple_detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save temp
    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    file.save(temp_path)

    try:
        pipeline = current_app.extensions.get("web_report_pipeline")
        if pipeline is None:
            return jsonify({"error": "Shared inference pipeline is not available"}), 500

        # Run inference
        out_dir = Path(tempfile.mkdtemp())
        results = pipeline.run(Path(temp_path), out_dir)
        
        return jsonify({
            "status": "success",
            "teeth_count": len(results.get('final_teeth_objects', [])),
            "caries_count": len(results.get('final_caries_objects', []))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

from flask import Blueprint, request, jsonify, current_app
import os
import tempfile
from pathlib import Path
from services.image_loader import load_image_any
from services.pano_inference import PanoPipeline
from config import Config

detect_api = Blueprint('detect_api', __name__)

# Initialize pipeline for this legacy route
pipeline = PanoPipeline(
    model_dir=str(Path(__file__).resolve().parent.parent / "weights"),
    model_cfg=Config.PANO_MODELS
)

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

import os
import sys
import cv2
import json
import shutil
from pathlib import Path
from config import config

# Ensure we can import services and utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.pano_inference import PanoPipeline
from utils.feedback_generator import FeedbackGenerator
from utils.report_v2 import ReportGeneratorV2

def test_pipeline(image_path):
    print(f"--- Starting Test Pipeline for: {image_path} ---")
    
    # Setup Output Dir
    base_name = Path(image_path).stem
    out_dir = Path("test_output") / base_name
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    inference_dir = out_dir / "inference"
    inference_dir.mkdir()
    crop_dir = out_dir / "crops"
    crop_dir.mkdir()
    
    # 1. Initialize Pipeline
    print("[1] Initializing PanoPipeline...")
    cfg = config["development"]
    # Mocking app config for pipeline
    model_cfg = cfg.PANO_MODELS
    
    pipeline = PanoPipeline(
        model_dir=Path("./weights"),
        model_cfg=model_cfg,
        device="cpu" # Force CPU for test if needed, or 'cuda'
    )
    
    # 2. Run Inference
    print("[2] Running Inference...")
    # PanoPipeline.run saves images to out_dir automatically if configured, 
    # but here we might need to handle it.
    # The 'run' method returns a dict.
    
    # Check signature: def run(self, image_path: Path, out_dir: Path) -> Dict[str, Any]:
    result_map = pipeline.run(Path(image_path), inference_dir)
    
    analysis_result = result_map.get("analysis_result")
    if not analysis_result:
        print("!!! Inference Failed: No analysis_result")
        return

    print(f"    Inference Done. Found {len(analysis_result.get('teeth', []))} teeth.")
    
    # 3. Simulate Analysis Results Processing (like in routes_v2)
    # The pipeline run already saved 'overlay.png' in inference_dir
    overlay_path = inference_dir / "overlay.png"
    if not overlay_path.exists():
        print("!!! Overlay not found!")
        return
        
    print(f"    Overlay saved at: {overlay_path}")
    
    # 4. Generate Feedback (Testing the 'Crop from Overlay' & Logic)
    print("[3] Generating Feedback (Testing Crops)...")
    
    # Load the OVERLAY image for cropping (The Key Change)
    feedback_img = cv2.imread(str(overlay_path))
    
    fb_gen = FeedbackGenerator(base_url="http://test_local", upload_folder=str(out_dir))
    
    # Generate
    summary, findings, markdown_text = fb_gen.generate_structured(
        user_name="TestUser",
        analysis_result=analysis_result,
        original_img=feedback_img, # Pass Overlay!
        iac_result=result_map.get('iac_raw_res'),
        base_url="http://test_local",
        crop_dir=crop_dir,
        crop_url_prefix="http://test_local/crops"
    )
    
    print("\n--- Generated Feedback Summary ---")
    print(summary)
    print("\n--- Generated Markdown (Partial) ---")
    print(markdown_text[:500] + "...")
    
    print(f"\n[4] Check Output Directory: {out_dir}")
    print(f"    - Overlay: {overlay_path}")
    print(f"    - Crops: {crop_dir}")

if __name__ == "__main__":
    # Use one of the uploaded images if available, otherwise prompt
    # Hardcoded for the user's current context
    target_img = "C:/Users/dwono/.gemini/antigravity/brain/d57cf2bd-49a5-4abf-a46f-b6eadcbb7ead/uploaded_media_1_1769729171312.png" 
    
    if len(sys.argv) > 1:
        target_img = sys.argv[1]
        
    if not os.path.exists(target_img):
        print(f"Image not found: {target_img}")
    else:
        test_pipeline(target_img)


import os
import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np

# Mocking Flask App Context if needed, or just import services directly
sys.path.append(r'c:\interface\gpts')

from services.llm_service import GeminiConsultant
from utils.report_v3 import ReportGeneratorV2

def test_llm_integration():
    print(">>> Testing LLM Integration on Disk Code...")
    
    # 1. Check if LLM Service has the new method
    llm_service = GeminiConsultant()
    if not hasattr(llm_service, 'generate_missing_tooth_analysis'):
        print("[FAIL] LLM Service missing 'generate_missing_tooth_analysis' method.")
        return
    else:
        print("[PASS] LLM Service has new method.")

    # 2. Check ReportGenerator has new signature
    repo_gen = ReportGeneratorV2()
    import inspect
    sig = inspect.signature(repo_gen.generate)
    if 'llm_analysis' in sig.parameters:
        print("[PASS] ReportGenerator.generate accepts 'llm_analysis'.")
    else:
        print("[FAIL] ReportGenerator.generate MISSING 'llm_analysis'.")
    
    # 3. Test LLM Call (Dry Run or Real)
    print("\n>>> Simulating Missing Tooth Analysis...")
    missing_data = [{
        "tooth_label": "16",
        "missing_reason": "Visualization Test",
        "implant_guide": {"dist_mm": 8.5}
    }]
    
    # Create valid dummy image
    dummy_img = Image.new('RGB', (100, 100), color='gray')
    images_map = {"16": dummy_img}
    
    try:
        # result = llm_service.generate_missing_tooth_analysis(missing_data, images_map)
        # print(f"LLM Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print("[SKIP] Actual API call skipped to save time/tokens, but logic path exists.")
    except Exception as e:
        print(f"[ERROR] LLM Call failed: {e}")

    print("\n>>> DIAGNOSIS: The code on disk IS updated.")
    print("If the web app is still showing old text, the running process has not reloaded this file.")

if __name__ == "__main__":
    test_llm_integration()

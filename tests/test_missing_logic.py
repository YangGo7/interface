
import sys
import os
import cv2
import numpy as np

# Add backend to path
sys.path.append(r'c:\interface\backend')

from services.pano_inference import PanoPipeline

def test_pipeline(image_path):
    print(f"Testing Image: {image_path}")
    if not os.path.exists(image_path):
        print("Image not found!")
        return

    pipeline = PanoPipeline()
    img = cv2.imread(image_path)
    
    # Run Inference
    result = pipeline.process(img)
    
    # Check Structure
    print(f"Result Keys: {result.keys()}")
    
    # Findings usually combine existing + missing
    findings = result.get('findings', [])
    missing_list = result.get('missing_teeth', []) # If separate
    
    # Check Findings for Implant Guide
    missing_count = 0
    for t in findings:
        lbl = t.get('tooth_label')
        # Check if missing
        # In findings, missing might produce 'is_missing': True?
        # Or look for 'implant_guide' key.
        if t.get('implant_guide'):
            missing_count += 1
            print(f"MATCH: Tooth {lbl} has Implant Guide!")
            ig = t['implant_guide']
            print(f"  Dist: {ig.get('dist_mm')} mm")
            print(f"  Coords: {ig.get('line_coords')}")
            print(f"  Type: {ig.get('type')}")
            
            # Check for validity
            coords = ig.get('line_coords')
            if coords and len(coords) == 4:
                print("  -> Coords Valid.")
            else:
                print("  -> Coords INVALID/MISSING.")
        elif t.get('is_missing'):
             print(f"FAIL: Tooth {lbl} is MISSING but NO Implant Guide!")
             
    print(f"Total Missing with Guide: {missing_count}")

if __name__ == "__main__":
    test_img = r"C:\interface\tests\000094.jpg"
    test_pipeline(test_img)

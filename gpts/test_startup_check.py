import sys
from pathlib import Path

# Simulate running from gpts directory
sys.path.insert(0, str(Path(__file__).parent))

print(f"[TEST] Python Path: {sys.path[:3]}")

try:
    from app import app, BASE_DIR
    print(f"[TEST] Imported app successfully.")
    print(f"[TEST] BASE_DIR: {BASE_DIR}")
    print(f"[TEST] Model Dir: {app.config['PANO_MODELS']['pano_seg']['path']}")
    
    # Check if backend module is inadvertently imported
    if 'backend' in sys.modules:
        print(f"[WARN] 'backend' module found in sys.modules: {sys.modules['backend']}")
    else:
        print(f"[PASS] 'backend' module NOT loaded (Good).")
        
    print("[TEST] Configurations verify OK.")
except Exception as e:
    print(f"[FAIL] Import Error: {e}")
    sys.exit(1)

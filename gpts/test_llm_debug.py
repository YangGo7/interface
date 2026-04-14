import os
import sys
import PIL.Image

# Add current dir to path to find services
sys.path.append(os.getcwd())
try:
    from services.llm_service import GeminiConsultant
except ImportError as e:
    print(f"Import Error: {e}")
    # Try adjusting path if running from root
    sys.path.append(os.path.join(os.getcwd(), 'c:\\interface\\gpts'))
    from services.llm_service import GeminiConsultant

print("Initializing Gemini Service...")
try:
    service = GeminiConsultant()
except Exception as check_e:
    print(f"Init Exception: {check_e}")
    sys.exit(1)

import requests
import json

print(f"API Key present: {bool(service.api_key)}")
if service.api_key:
    masked = service.api_key[:4] + "*" * (len(service.api_key)-8) + service.api_key[-4:]
    print(f"Loaded Key: {masked}")

# Model is now selected dynamically during generation

print("\n--- Test 3: Raw HTTP REST API Check ---")
if not service.api_key:
    print("Skipping Raw Test (No Key)")
else:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={service.api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{"text": "Hello, are you working?"}]
        }]
    }
    try:
        print(f"Sending POST to {url.split('?')[0]}...")
        resp = requests.post(url, headers=headers, json=data)
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Raw Response: {resp.text[:100]}...")
            print("SUCCESS: API Key is valid and REST API works.")
        else:
            print(f"FAILED: {resp.text}")
    except Exception as e:
        print(f"Raw Request Failed: {e}")

print("\n--- Listing Available Models ---")
try:
    with open('models.txt', 'w') as f:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                f.write(f"{m.name}\n")
except Exception as e:
    print(f"List models failed: {e}")

print("\n--- Test 1 (Retry): Text Only ---")
try:
    res = service.generate_patient_summary("TestPatient", [], [], {})
    print(f"Result: {res[:100]}...")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Test 2: Multimodal (Image) with Fallback Logic ---")
img_path = 'test_debug.png'
# Create dummy image
img = PIL.Image.new('RGB', (100, 100), color = 'red')
img.save(img_path)

try:
    # This should trigger the internal loop
    res = service.generate_patient_summary("TestPatient", [], [], {}, image_path=img_path)
    print(f"Final Result: {res[:100]}...")
except Exception as e:
    print(f"Error: {e}")

import requests
import os
from dotenv import load_dotenv

# Load env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
api_key = os.getenv("GEMINI_API_KEY")

print(f"Checking Key: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

if not api_key:
    print("No API Key found.")
    exit()

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

try:
    print(f"Requesting: {url.split('?')[0]}")
    response = requests.get(url)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("--- Available Models ---")
        if 'models' in data:
            for m in data['models']:
                print(f"Name: {m['name']}")
                print(f"Supported Methods: {m.get('supportedGenerationMethods', [])}")
                print("-" * 20)
        else:
            print("No models found in response. (Service might be disabled?)")
            print(f"Full Response: {data}")
    else:
        print("Error Response:")
        print(response.text)

except Exception as e:
    print(f"Request Failed: {e}")

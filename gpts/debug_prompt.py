import sys
import os
import yaml
import json
import google.generativeai as genai
from pathlib import Path

# Load env
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    print(f"Loading env from {env_path}")
    with open(env_path, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ[k] = v

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found.")
    sys.exit(1)

genai.configure(api_key=API_KEY)

def debug_prompt():
    # 1. Load Prompt from YAML
    yaml_path = Path("prompts.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    
    template = prompts.get("technical_report_prompt")
    print(f">>> Usage Template Length: {len(template)}")
    print(f">>> Template Preview:\n{template[:200]}...")

    # 2. Mock Data
    mock_findings = {
        "total_teeth": 28,
        "missing_teeth": ["18", "28", "38", "48"],
        "problem_teeth": [
            {
                "tooth": "16",
                "issues": ["충치", "상악동 근접"]
            },
            {
                "tooth": "26",
                "issues": ["치근단 병소"]
            }
        ]
    }
    
    model_findings_json = json.dumps(mock_findings, ensure_ascii=False, indent=2)
    
    # 3. Construct Final Prompt
    full_prompt = template.format(
        model_findings_json=model_findings_json,
        total_count=mock_findings["total_teeth"],
        missing_list_str=str(mock_findings["missing_teeth"])
    )
    
    print("\n>>> Sending Prompt to Gemini...")
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content([full_prompt])
    
    print("\n>>> [GEMINI RESPONSE]")
    print(response.text)

if __name__ == "__main__":
    debug_prompt()

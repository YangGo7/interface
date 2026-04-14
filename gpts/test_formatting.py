import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from services.llm_service import GeminiConsultant

def test_technical_report_generation():
    print(">>> Initializing Gemini Consultant...")
    service = GeminiConsultant()
    
    # Mock Model Findings
    mock_findings = {
        "total_teeth": 28,
        "missing_teeth": ["18", "28", "38", "48"],
        "problem_teeth": [
            {
                "tooth": "16",
                "issues": ["충치", "상악동 근접"]
            },
            {
                "tooth": "46",
                "issues": ["치근단 병소", "하악 신경관 근접"]
            }
        ],
        "per_tooth_ai_analysis": {} # Not used in this prompt flow directly
    }

    print("\n>>> Generating Technical Report from Mock Data...")
    print(json.dumps(mock_findings, ensure_ascii=False, indent=2))
    print("-" * 50)
    
    report_text = service.generate_technical_report(mock_findings)
    
    print("\n>>> [RESULT] Generated Text:")
    print("-" * 50)
    print(report_text)
    print("-" * 50)
    
    # Simple Validations
    if "본 출력은 연구/시연용" in report_text:
        print("[PASS] Disclaimer found.")
    else:
        print("[FAIL] Disclaimer NOT found.")
        
    if "tooth_id=" in report_text:
        print("[PASS] tooth_id format found.")
    else:
        print("[FAIL] tooth_id format NOT found.")
        
    if "finding=" in report_text and "education=" in report_text:
        print("[PASS] finding/education fields found.")
    else:
        print("[FAIL] finding/education fields NOT found.")

if __name__ == "__main__":
    test_technical_report_generation()

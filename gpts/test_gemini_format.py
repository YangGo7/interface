import json
from services.llm_service import GeminiConsultant
from PIL import Image

def test_gemini():
    service = GeminiConsultant()
    # Dummy image
    img = Image.new('RGB', (100, 100), color = 'red')
    images_map = {"16": img}
    
    problem_teeth = [{
        "tooth_number": "16",
        "findings": ["caries", "periapical"],
        "bone_loss_level": 3,
        "bone_loss_pct": 50,
        "nerve_overlap": False,
        "sinus_overlap": True
    }]
    
    print("Testing Multimodal Problem Teeth Analysis...")
    result = service.generate_tooth_analysis(problem_teeth, images_map=images_map, language="Korean")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_gemini()

import os
import yaml
import google.generativeai as genai
from dotenv import load_dotenv
import PIL.Image
import json

# Load env explicitly
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

class GeminiConsultant:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.prompts = self._load_prompts()
        self.active_model = None
        
        if not self.api_key:
            print("[LLM] Warning: GEMINI_API_KEY not found in .env")
        else:
            try:
                genai.configure(api_key=self.api_key)
                print("[LLM] Gemini Service Configured.")
            except Exception as e:
                print(f"[LLM] Init failed: {e}")

    def _load_prompts(self):
        """Loads prompts from prompts.yaml"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts.yaml')
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[LLM] Warning: Could not load prompts.yaml ({e}). Using defaults.")
            return {}

    def generate_patient_summary(self, patient_name, missing_teeth, findings, summary_stats, image_path=None, language="English"):
        """
        Generates a personalized doctor's note. Tries multiple models (Multimodal -> Text-only).
        """
        if not self.api_key:
            return None

        # 1. Prepare Data
        missing_list = [str(t.get('tooth_label')) for t in missing_teeth] if missing_teeth else []
        caries_count = len(summary_stats.get('caries', []))
        perio_count = len(summary_stats.get('periapical', []))
        max_bl = 0
        for f in findings:
            max_bl = max(max_bl, f.get('bone_loss_level', 0))

        # [NEW] Calculate present teeth count (unique valid FDI numbers)
        present_teeth_set = set()
        if isinstance(findings, list):
            for f in findings:
                t = f.get('tooth_label') or f.get('tooth')
                if t and str(t).isdigit():
                    tn = int(t)
                    if 11 <= tn <= 48:
                        present_teeth_set.add(tn)
        
        # Fallback: if finding has no label but exists, it might be an issue. 
        # But for 'present teeth', we usually want 28 - missing. 
        # However, user asks to send 'total teeth count'. 
        # Usage: "detected teeth count" is likely what is meant.
        present_count = len(present_teeth_set)

        # Prepare Findings Detail String for the Prompt
        findings_details = []
        if isinstance(findings, list):
            for f in findings:
                # Expecting f to be dict like {'tooth': '16', 'issues': ['caries', ...]}
                t_num = f.get('tooth', '?')
                issues = f.get('issues', [])
                
                # [FILTER] Exclude if the ONLY issue is '상악동 근접'
                # If there are multiple issues (e.g., caries + sinus), we keep it.
                if len(issues) == 1 and issues[0] == '상악동 근접':
                    continue
                    
                if issues:
                    findings_details.append(f"- {t_num}번 치아: {', '.join(issues)}")
        
        findings_str = "\n".join(findings_details) if findings_details else "특이 소견 없음"

        context = {
            # "patient_name": patient_name, # Removed per user request
            "missing_list": ', '.join(missing_list) if missing_list else '없음',
            "max_bl": max_bl, # Simplified context as requested
            "findings_detail": findings_str,
            "caries_count": caries_count,
            "perio_count": perio_count
        }

        # 2. Build Prompt
        system_p = self.prompts.get('system_prompt', "당신은 치과의사입니다.").format(language=language)
        user_template = self.prompts.get('user_prompt_template', "{patient_name}님의 진단 결과입니다.")
        full_text_prompt = f"{system_p}\n\n{user_template.format(**context, language=language)}"
        
        # 3. Dynamic Model Discovery (Auto-Select)
        # Instead of guessing, we ask the API what is available.
        valid_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    valid_models.append(m.name)
        except Exception as e:
            print(f"[LLM] Model discovery failed: {e}")
            # Fallback hardcoded list
            valid_models = ['models/gemini-1.5-flash', 'models/gemini-pro']

        # Prioritize 'flash' models, then 'pro'
        valid_models.sort(key=lambda x: (not 'flash' in x, not 'pro' in x))
        
        print(f"[LLM] Discovered viable models: {valid_models}")
        
        # 4. Prepare Logic
        img_obj = None
        if image_path and os.path.exists(image_path):
            try:
                img_obj = PIL.Image.open(image_path)
            except: 
                pass

        # 5. Try Discovered Models
        for model_name in valid_models:
            # SDK sometimes wants 'gemini-pro', sometimes 'models/gemini-pro'.
            # We try both variations for robust connection.
            variations = [model_name, model_name.replace('models/', '')]
            
            for name in variations:
                try:
                    # Setup Inputs
                    inputs = [full_text_prompt]
                    # Only attach image if model likely supports it (vision models usually have 'vision' or 'flash' or '1.5')
                    # But for safety, we try. If it fails, we catch exception.
                    if img_obj:
                         inputs.append(img_obj)
                         print(f"[LLM] Trying {name} with Image...")
                    else:
                         print(f"[LLM] Trying {name} (Text Only)...")

                    model = genai.GenerativeModel(name)
                    response = model.generate_content(inputs)
                    
                    print(f"[LLM] Success with {name}!")
                    return response.text.strip()
                    
                except Exception as e:
                    print(f"[LLM] {name} failed: {e}")
                    # If image caused failure, retry without image (Text Fallback)
                    if img_obj:
                         try:
                             print(f"[LLM] Retrying {name} without image...")
                             model = genai.GenerativeModel(name)
                             response = model.generate_content([full_text_prompt])
                             print(f"[LLM] Success with {name} (Text Only)!")
                             return response.text.strip()
                         except:
                             pass
        
    
    def generate_missing_tooth_analysis(self, missing_teeth_data, images_map=None, language="English"):
        """
        Generates structured analysis for missing teeth.
        If images_map ({label: PIL.Image}) is provided, sends individual multimodal requests.
        Otherwise, sends a single batch text request.
        """
        if not self.api_key or not missing_teeth_data:
            return {}

        prompt_template = self.prompts.get('missing_tooth_prompt', "")
        if not prompt_template:
            return {}
            
        full_results = {}

        # Mode A: Multimodal (Individual Requests for precision)
        if images_map:
            print("[LLM] Running Per-Tooth Multimodal Analysis...")
            model_name = "models/gemini-2.0-flash" # Vision capable
            model = genai.GenerativeModel(model_name)
            
            for mt in missing_teeth_data:
                lbl = mt.get('tooth_label')
                # Prepare single item data
                guide = mt.get('implant_guide', {})
                dist = guide.get('dist_mm', 0) if guide else 0
                single_data = [{
                    "tooth_label": lbl,
                    "missing_reason": mt.get('missing_reason', 'Structure not found'),
                    "bone_height_mm": f"{dist:.1f}"
                }]
                
                # Construct Prompt
                prompt = prompt_template.format(missing_teeth_json=json.dumps(single_data, indent=2), language=language)
                inputs = [prompt]
                
                # Add Image if available
                img = images_map.get(str(lbl)) or images_map.get(lbl)
                if img:
                    inputs.append(img)
                    print(f"[LLM] Analyzing Tooth {lbl} with Image...")
                
                try:
                    response = model.generate_content(inputs)
                    text = response.text.strip()
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        text = text.split("```")[1].strip()
                    
                    # Merge result
                    result = json.loads(text)
                    full_results.update(result)
                except Exception as e:
                    print(f"[LLM] Analysis failed for {lbl}: {e}")
            
            return full_results

        # Mode B: Text Batch (Existing logic)
        input_summary = []
        for mt in missing_teeth_data:
            guide = mt.get('implant_guide', {})
            dist = guide.get('dist_mm', 0) if guide else 0
            input_summary.append({
                "tooth_label": mt.get('tooth_label'),
                "missing_reason": mt.get('missing_reason', 'Structure not found'),
                "bone_height_mm": f"{dist:.1f}"
            })
        
        full_prompt = prompt_template.format(missing_teeth_json=json.dumps(input_summary, indent=2), language=language)
        model_name = "models/gemini-2.0-flash"
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([full_prompt])
            text = response.text.strip()
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].strip()
            
            return json.loads(text)
        except Exception as e:
            print(f"[LLM] Batch Analysis Failed: {e}")
            return {}

    def generate_tooth_analysis(self, problem_teeth, images_map=None, language="English"):
        """
        Generates per-tooth analysis & recommendation for problem teeth.
        
        Args:
            problem_teeth: list of dicts, each with:
                - tooth_number (str): FDI number
                - findings (list): ['caries', 'periapical', ...]
                - bone_loss_level (int): 0-4
                - bone_loss_pct (float): 0-100
                - nerve_overlap (bool)
                - sinus_overlap (bool)
        
        Returns:
            dict: {fdi_str: {"analysis": "...", "recommendation": "..."}, ...}
        """
        if not self.api_key or not problem_teeth:
            return {}

        prompt_template = self.prompts.get('tooth_analysis_prompt', '')
        if not prompt_template:
            print("[LLM] tooth_analysis_prompt not found in prompts.yaml")
            return {}

        model_name = "models/gemini-2.0-flash"
        full_results = {}

        # Mode A: Multimodal (Individual Requests for precision)
        if images_map:
            print("[LLM] Running Per-Tooth Multimodal Analysis for Problem Teeth...")
            try:
                model = genai.GenerativeModel(model_name)
                for pt in problem_teeth:
                    lbl = pt.get('tooth_number')
                    # Construct Prompt for a single tooth
                    single_json = json.dumps([pt], ensure_ascii=False, indent=2)
                    prompt = prompt_template.format(teeth_json=single_json, language=language)
                    inputs = [prompt]
                    
                    # Add Image if available
                    img = images_map.get(str(lbl)) or images_map.get(lbl)
                    if img:
                        inputs.append(img)
                        print(f"[LLM] Analyzing Problem Tooth {lbl} with Image...")
                    else:
                        print(f"[LLM] Analyzing Problem Tooth {lbl} (Text Only)...")
                        
                    try:
                        response = model.generate_content(inputs)
                        text = response.text.strip()
                        if "```json" in text:
                            text = text.split("```json")[1].split("```")[0].strip()
                        elif "```" in text:
                            text = text.split("```")[1].strip()
                        
                        result = json.loads(text)
                        full_results.update(result)
                    except Exception as e:
                        print(f"[LLM] Analysis failed for {lbl}: {e}")
                
                return full_results
            except Exception as e:
                print(f"[LLM] Per-Tooth Multimodal Analysis Failed: {e}")
                return {}

        # Mode B: Text Batch (Existing logic)
        print("[LLM] Running Batch Text Analysis for Problem Teeth...")
        # Build JSON input
        teeth_json = json.dumps(problem_teeth, ensure_ascii=False, indent=2)
        full_prompt = prompt_template.format(teeth_json=teeth_json, language=language)

        try:
            model = genai.GenerativeModel(model_name)
            print(f"[LLM] Sending {len(problem_teeth)} problem teeth to Gemini for analysis...")
            response = model.generate_content([full_prompt])
            text = response.text.strip()

            # Clean JSON from possible code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].strip()

            result = json.loads(text)
            print(f"[LLM] Per-tooth analysis received for {len(result)} teeth.")
            return result

        except Exception as e:
            print(f"[LLM] Per-Tooth Analysis Failed: {e}")
            return {}

    def generate_technical_report(self, model_findings, language="English"):
        """
        Generates a strict 'Technical Output' text report using Gemini.
        Includes legal disclaimers, phrased findings, and medical education.
        """
        if not self.api_key or not model_findings:
            return "모델 분석 결과가 없거나 API 키가 설정되지 않았습니다."

        prompt_template = self.prompts.get('technical_report_prompt', '')
        if not prompt_template:
            return "프롬프트 템플릿(technical_report_prompt)을 찾을 수 없습니다."

        try:
            # Prepare context variables for the prompt
            total_count = model_findings.get("total_teeth", 0)
            missing_list = model_findings.get("missing_teeth", [])
            missing_list_str = str(missing_list)
            
            # The 'problem_teeth' list is part of model_findings
            model_findings_json = json.dumps(model_findings, ensure_ascii=False, indent=2)

            full_prompt = prompt_template.format(
                model_findings_json=model_findings_json,
                total_count=total_count,
                missing_list_str=missing_list_str,
                language=language # [NEW]
            )

            # Use flash model for speed
            model_name = "models/gemini-2.0-flash"
            model = genai.GenerativeModel(model_name)
            
            print(f"[LLM] Generating Technical Report for {len(model_findings.get('problem_teeth',[]))} items...")
            response = model.generate_content([full_prompt])
            text = response.text.strip()
            
            # Cleanup markdown if present (though prompt asks not to)
            if text.startswith("```"):
                lines = text.splitlines()
                if lines[0].startswith("```"): lines = lines[1:]
                if lines[-1].startswith("```"): lines = lines[:-1]
                text = "\n".join(lines).strip()

            return text

        except Exception as e:
            print(f"[LLM] Technical Report Generation Failed: {e}")
            return "AI 리포트 생성 중 오류가 발생했습니다.\n\n" + json.dumps(model_findings, ensure_ascii=False, indent=2)


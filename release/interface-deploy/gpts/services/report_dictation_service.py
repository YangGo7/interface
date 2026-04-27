from utils.report_v3 import ReportGeneratorV3
from utils.feedback_generator import FeedbackGenerator
from services.llm_service import GeminiConsultant

class ReportDictationService:
    def __init__(self):
        self.report_gen = ReportGeneratorV3()
        self.feedback_gen = FeedbackGenerator()
        self.gemini_service = GeminiConsultant()

    def generate_clinical_report(self, task_id: str, result: dict, user_name: str, language: str):
        """
        Generates HTML reports, patient letters, and AI-driven clinical findings.
        """
        try:
            # 1. Generate standard HTML report
            report_data = self.report_gen.generate(result, user_name, language)
            
            # 2. Generate Gemini Vision / LLM consultation if enabled
            consultation = self.gemini_service.consult(result, user_name, language)
            
            # 3. Patient Feedback
            feedback = self.feedback_gen.generate_friendly_text(result, user_name, language)

            return {
                "html_report": report_data.get('html'),
                "consultation": consultation,
                "patient_feedback": feedback
            }
        except Exception as e:
            print(f"[ERROR] Reporting failed: {e}")
            return None

# Singleton
report_dictation_service = ReportDictationService()

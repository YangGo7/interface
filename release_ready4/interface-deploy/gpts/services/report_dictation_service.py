import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)


class ReportDictationService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_candidates = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash",
        ]
        self._configured = False

    def _configure(self) -> None:
        if self._configured:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        genai.configure(api_key=self.api_key)
        self._configured = True

    def _wait_for_file_ready(self, uploaded_file: Any, timeout_seconds: int = 90) -> Any:
        deadline = time.time() + timeout_seconds
        file_ref = uploaded_file
        while time.time() < deadline:
            state = getattr(getattr(file_ref, "state", None), "name", "") or ""
            normalized = str(state).upper()
            if not normalized or normalized in {"ACTIVE", "SUCCEEDED", "READY"}:
                return file_ref
            if normalized in {"FAILED", "ERROR"}:
                raise RuntimeError(f"Gemini file processing failed with state {normalized}.")
            time.sleep(1.5)
            file_ref = genai.get_file(file_ref.name)
        return file_ref

    def _generate_text(self, parts: List[Any], temperature: float = 0.1) -> str:
        errors: List[str] = []
        for model_name in self.model_candidates:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={"temperature": temperature},
                )
                response = model.generate_content(parts)
                text = (response.text or "").strip()
                if text:
                    return text
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
        raise RuntimeError("Gemini generation failed. " + " | ".join(errors))

    def _clean_json_block(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json") :].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```") :].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned

    def _parse_summary_json(self, text: str) -> Dict[str, Any]:
        cleaned = self._clean_json_block(text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini summary JSON parsing failed: {exc}")

        soap = parsed.get("soap_note") or {}
        tooth_findings = parsed.get("tooth_findings") or []
        keywords = parsed.get("keywords") or []
        return {
            "soap_note": {
                "subjective": str(soap.get("subjective") or "").strip(),
                "objective": str(soap.get("objective") or "").strip(),
                "assessment": str(soap.get("assessment") or "").strip(),
                "plan": str(soap.get("plan") or "").strip(),
            },
            "tooth_findings": [
                {
                    "tooth": str(item.get("tooth") or "").strip(),
                    "keywords": [str(keyword).strip() for keyword in (item.get("keywords") or []) if str(keyword).strip()],
                }
                for item in tooth_findings
                if item.get("tooth")
            ],
            "keywords": [str(keyword).strip() for keyword in keywords if str(keyword).strip()],
        }

    def _build_report_note_text(self, soap_note: Dict[str, str]) -> str:
        sections = [
            ("S", soap_note.get("subjective", "")),
            ("O", soap_note.get("objective", "")),
            ("A", soap_note.get("assessment", "")),
            ("P", soap_note.get("plan", "")),
        ]
        return "\n".join(f"{label}: {content.strip()}" for label, content in sections if content and content.strip())

    def transcribe_and_summarize(self, audio_path: Path, mime_type: str, language: str = "English") -> Dict[str, Any]:
        self._configure()

        uploaded_file = genai.upload_file(
            path=str(audio_path),
            mime_type=mime_type or "audio/webm",
            display_name=audio_path.name,
        )
        uploaded_file = self._wait_for_file_ready(uploaded_file)

        try:
            transcript_prompt = (
                "You are a dental radiology dictation transcription assistant.\n"
                "Transcribe the audio into a clean clinical transcript.\n"
                "Rules:\n"
                "- Preserve FDI tooth numbers exactly as dictated.\n"
                "- Normalize Korean dental dictation into standard English dental and radiology terminology when appropriate.\n"
                "- Use precise English clinical terms such as periapical lesion, endodontic treatment, root canal treatment, implant, crown, filling, mobility, bone loss.\n"
                "- Do not invent findings that are not present in the audio.\n"
                "- Return only the transcript text.\n"
                f"- Output language: {language}, but keep dental terminology in English.\n"
            )
            transcript = self._generate_text([uploaded_file, transcript_prompt], temperature=0.0)

            summary_prompt = (
                "You are a dental radiology reporting assistant.\n"
                "Using only the transcript below, create a structured SOAP summary and tooth-level keyword extraction.\n"
                "Rules:\n"
                "- Use concise professional English.\n"
                "- Keep tooth numbers exactly as strings such as \"26\".\n"
                "- Preserve technical medical terminology in English.\n"
                "- For tooth findings, extract normalized keywords such as periapical lesion, endodontic treatment, root canal treatment, mobility, bone loss, implant, crown.\n"
                "- Do not add unsupported diagnoses.\n"
                "- Return valid JSON only, with this schema:\n"
                "{\n"
                "  \"soap_note\": {\n"
                "    \"subjective\": \"...\",\n"
                "    \"objective\": \"...\",\n"
                "    \"assessment\": \"...\",\n"
                "    \"plan\": \"...\"\n"
                "  },\n"
                "  \"keywords\": [\"...\"],\n"
                "  \"tooth_findings\": [\n"
                "    {\"tooth\": \"26\", \"keywords\": [\"periapical lesion\", \"endodontic treatment\", \"mobility\"]}\n"
                "  ]\n"
                "}\n\n"
                f"Transcript:\n{transcript}"
            )
            summary_text = self._generate_text([summary_prompt], temperature=0.1)
            parsed = self._parse_summary_json(summary_text)
            report_note_text = self._build_report_note_text(parsed["soap_note"])
            return {
                "transcript": transcript,
                "soap_note": parsed["soap_note"],
                "tooth_findings": parsed["tooth_findings"],
                "keywords": parsed["keywords"],
                "report_note_text": report_note_text,
            }
        finally:
            try:
                genai.delete_file(uploaded_file)
            except Exception:
                pass

from pathlib import Path
from typing import Any, Dict, Optional

from utils.web_report_generator import WebReportGenerator


class WebReportReportService:
    def __init__(self):
        self.generator = WebReportGenerator()

    def build_analysis_result(self, effective_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "teeth": effective_result.get("teeth", []),
            "missing_teeth": effective_result.get("missing_teeth", []),
            "caries": effective_result.get("caries", []),
            "periapical": effective_result.get("periapical", []),
            "nerve_contours": effective_result.get("nerve_contours", []),
            "sinus_contours": effective_result.get("sinus_contours", []),
            "report_note": effective_result.get("report_note", ""),
            "attached_captures": effective_result.get("attached_captures", []),
        }

    def generate_report(
        self,
        session_id: str,
        user_name: str,
        image_path: Path,
        overlay_path: Optional[Path],
        bl_viz_path: Optional[Path],
        effective_result: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        analysis_result = self.build_analysis_result(effective_result)
        html_path, html_filename, pdf_filename = self.generator.generate(
            user_name=user_name,
            image_path=image_path,
            analysis_result=analysis_result,
            overlay_path=overlay_path if overlay_path and overlay_path.exists() else None,
            bl_viz_path=bl_viz_path if bl_viz_path and bl_viz_path.exists() else None,
            output_dir=output_dir,
        )
        pdf_path = str(output_dir / pdf_filename) if pdf_filename else None
        return {
            "html_path": html_path,
            "html_filename": html_filename,
            "pdf_path": pdf_path,
            "pdf_filename": pdf_filename,
            "snapshot": effective_result,
        }

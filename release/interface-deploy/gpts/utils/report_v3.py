import base64
import datetime
from pathlib import Path

import cv2

from services.pano_inference import load_image_any
from utils.report_v3_algorithms import ReportV3AlgorithmsMixin
from utils.report_v3_viewer import ReportV3ViewerMixin


class ReportGeneratorV3(ReportV3AlgorithmsMixin, ReportV3ViewerMixin):
    def __init__(self, output_dir="c:/interface/gpts/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.odonto_dir = Path("c:/interface/gpts/imgs/Odonto")
        self.renew_teeth_dir = Path("c:/interface/frontend/public/imgs/teeth")
        self.warning_icon_path = Path("c:/interface/gpts/imgs/!/icon_red.png")
        self.info_icon_path = Path("c:/interface/gpts/imgs/!/icon_black.png")
        self.logo_path = Path("c:/interface/gpts/imgs/logo/Cyberme_logo.png")
        self.upper_tooth_sizes = {
            1: "23 79",
            2: "20 75",
            3: "22 82",
            4: "25 74",
            5: "25 67",
            6: "38 58",
            7: "36 54",
            8: "35 49",
        }
        self.lower_tooth_sizes = {
            1: "18 71",
            2: "17 70",
            3: "22 78",
            4: "25 75",
            5: "26 76",
            6: "39 68",
            7: "38 65",
            8: "41 63",
        }

    def _load_odonto_icon(self, jaw, position, status="triage-3"):
        """Load Renew odontogram tooth icon as base64. jaw='up'/'down', position=1-8"""
        arch = "U" if jaw == "up" else "L"
        size = self.upper_tooth_sizes.get(position, "") if arch == "U" else self.lower_tooth_sizes.get(position, "")
        img_path = None

        if self.renew_teeth_dir.exists():
            if status == "triage-1":
                img_path = self.renew_teeth_dir / "warning (ff0037)" / f"{arch}-{position} ({size})_2.png"
            elif status == "triage-2":
                img_path = self.renew_teeth_dir / "notice (fcff2a)" / f"{arch}-{position} ({size})_3.png"
            elif status == "implant":
                img_path = self.renew_teeth_dir / "implant (003dff)" / f"{arch}-{position} ({size})_1@4x.png"
            elif status == "missing":
                if arch == "U":
                    img_path = self.renew_teeth_dir / "missing (3f3f3f)" / f"U-{position}.png"
                else:
                    img_path = self.renew_teeth_dir / "missing (3f3f3f)" / f"L-{position} ({size})@4x.png"
            else:
                img_path = self.renew_teeth_dir / "health(ffffff)" / f"{arch}-{position} ({size})_4.png"

        if not img_path or not img_path.exists():
            prefix = "up" if jaw == "up" else "down"
            img_path = self.odonto_dir / f"{prefix}-{position}.png"

        if img_path.exists():
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return ""

    def _encode_image(self, img_array):
        """Encodes numpy image to base64 string"""
        if img_array is None:
            return ""
        success, buffer = cv2.imencode(".jpg", img_array)
        if not success:
            return ""
        return base64.b64encode(buffer).decode("utf-8")

    def generate(
        self,
        user_name,
        image_path,
        analysis_result,
        overlay_path=None,
        bl_viz_path=None,
        output_dir=None,
        ai_commentary=None,
        llm_analysis=None,
        detailed_findings=None,
        per_tooth_analysis=None,
        language="English",
    ):
        """
        Generates the full HTML report.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return None, None, None

        main_img = load_image_any(image_path)
        if main_img is None:
            raise ValueError(f"Image not found at {image_path}")
        crop_source_img = main_img.copy()

        bl_viz_img = None
        if bl_viz_path:
            bl_viz_path = Path(bl_viz_path)
            if bl_viz_path.exists():
                bl_viz_img = cv2.imread(str(bl_viz_path))

        if overlay_path:
            overlay_path = Path(overlay_path)
            if overlay_path.exists():
                overlay_img = cv2.imread(str(overlay_path))
                if overlay_img is not None:
                    crop_source_img = main_img.copy()

        findings = analysis_result.get("teeth", [])
        missing_teeth = analysis_result.get("missing_teeth", [])

        summary_data = {
            "caries": analysis_result.get("caries", []),
            "periapical": analysis_result.get("periapical", []),
        }

        date_str = datetime.datetime.now().strftime("%Y-%m-%d")

        html_content = self._generate_html(
            user_name,
            date_str,
            summary_data,
            findings,
            main_img,
            crop_source_img,
            bl_viz_img,
            missing_teeth,
            nerve_contours=analysis_result.get("nerve_contours"),
            sinus_contours=analysis_result.get("sinus_contours"),
            ai_commentary=ai_commentary,
            llm_analysis=llm_analysis,
            detailed_findings=detailed_findings,
            per_tooth_analysis=per_tooth_analysis,
            report_note=analysis_result.get("report_note"),
            attached_captures=analysis_result.get("attached_captures"),
            language=language,
        )

        safe_name = "".join([c for c in user_name if c.isalnum() or c in (" ", "_", "-")]).strip()
        html_filename = f"{safe_name}_Report.html"
        pdf_filename = f"{safe_name}_Report.pdf"

        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        html_path = target_dir / html_filename
        pdf_path = target_dir / pdf_filename

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # PDF generation disabled
        _ = pdf_path
        pdf_filename = None

        return str(html_path), html_filename, pdf_filename

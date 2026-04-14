import os
from flask import url_for
from pathlib import Path

class ResultBuilder:
    @staticmethod
    def build_inference_response(task_id: str, result: dict, patient_name: str, language: str):
        """
        Formats the raw AI results into the rich response needed by the frontend (ChartPage, etc.)
        """
        # Calculate summary counts
        teeth_objects = result.get('final_teeth_objects', [])
        summary_counts = {
            "teeth": len(teeth_objects),
            "caries": len(result.get('final_caries_objects', [])),
            "periapical": len(result.get('final_periapical_objects', [])),
            "implants": sum(1 for t in teeth_objects if t.get('status') == 'implant')
        }

        # Construct image URLs (Assuming static/output serving)
        # Using placeholder logic for paths - usually handled by Flask send_from_directory
        return {
            "task_id": task_id,
            "status": "success",
            "patient_name": patient_name,
            "language": language,
            "images": {
                "original": f"/outputs/{task_id}/original.jpg",
                "overlay": f"/outputs/{task_id}/overlay.png",
                "heatmap": f"/outputs/{task_id}/heatmap_overlay.png"
            },
            "data": {
                "teeth": teeth_objects,
                "caries": result.get('final_caries_objects', []),
                "periapical": result.get('final_periapical_objects', []),
                "bonelevel": result.get('bonelevel', {}),
                "odontogram_map": result.get('odontogram_map', {}),
                "diagnostics": result.get('diagnostics', {})
            },
            "summary_counts": summary_counts
        }

# Singleton
result_builder = ResultBuilder()

import threading
import uuid
import time
from pathlib import Path
from typing import Dict, Any
from services.pano_inference import PanoPipeline

class AnalysisService:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.inference_engine: PanoPipeline = None

    def initialize(self, model_dir: str, model_cfg: Dict):
        if not self.inference_engine:
            self.inference_engine = PanoPipeline(
                model_dir=model_dir,
                model_cfg=model_cfg
            )

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())

    def run_async_inference(self, task_id: str, image_path: Path, output_dir: Path, user_name: str, language: str):
        self.tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "start_time": time.time(),
            "result": None,
            "error": None
        }

        def worker():
            try:
                # 1. AI Inference
                result = self.inference_engine.run(
                    image_path=image_path,
                    out_dir=output_dir,
                    user_name=user_name,
                    language=language
                )
                
                # 2. Update Task
                self.tasks[task_id].update({
                    "status": "completed",
                    "progress": 100,
                    "result": result,
                    "end_time": time.time()
                })
            except Exception as e:
                self.tasks[task_id].update({
                    "status": "failed",
                    "error": str(e),
                    "end_time": time.time()
                })

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        return self.tasks.get(task_id, {"status": "not_found"})

# Singleton instance
analysis_service = AnalysisService()

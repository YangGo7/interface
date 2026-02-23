import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional

from services.pano_inference import PanoPipeline


class DetectJobManager:
    """Async job manager for /api/detect multi-model pipeline."""

    def __init__(self, pipeline: PanoPipeline, max_workers: int = 2, temp_root: Path = None):
        self.pipeline = pipeline
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.temp_root = temp_root

    def submit(self, image_path: Path, case_dir: Path) -> str:
        job_id = uuid.uuid4().hex
        with self.lock:
            self.jobs[job_id] = {"status": "queued", "result": None, "error": None, "case_dir": case_dir}
        self.executor.submit(self._run_job, job_id, image_path, case_dir)
        return job_id

    def _run_job(self, job_id: str, image_path: Path, case_dir: Path):
        with self.lock:
            self.jobs[job_id]["status"] = "running"
        try:
            result = self.pipeline.run(image_path=image_path, out_dir=case_dir)
            with self.lock:
                self.jobs[job_id]["status"] = "done"
                self.jobs[job_id]["result"] = result
        except Exception as e:
            with self.lock:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["error"] = str(e)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.jobs.get(job_id)

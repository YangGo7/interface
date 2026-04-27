import os
import cv2
import concurrent.futures
from typing import List, Dict, Any

from .core import BaseProcessor


class ObjectCropper(BaseProcessor):
    """Parallel cropping for detection results with optional mask support."""

    def run(self, save_dir: str, mode: str = "box") -> List[Dict[str, Any]]:
        os.makedirs(save_dir, exist_ok=True)

        upscale_factor = 2.0
        expand_ratio = 1.5
        use_mask = mode == "mask"

        results: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_det = {
                executor.submit(
                    self._process_single_item,
                    det,
                    i,
                    save_dir,
                    use_mask,
                    upscale_factor,
                    expand_ratio,
                ): i
                for i, det in enumerate(self.detections)
            }

            for future in concurrent.futures.as_completed(future_to_det):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as exc:
                    print(f"Crop processing failed for an item: {exc}")

        results.sort(key=lambda x: x["id"])
        return results

    def _process_single_item(
        self,
        det,
        index: int,
        save_dir: str,
        use_mask: bool,
        upscale_factor: float,
        expand_ratio: float,
    ) -> Dict[str, Any] | None:
        x1_org, y1_org, x2_org, y2_org, label, mask_coords = self.validate_box(det)

        w = x2_org - x1_org
        h = y2_org - y1_org
        if w <= 0 or h <= 0:
            return None

        center_y = y1_org + h / 2
        target_h = int(h * expand_ratio)

        y1_new = int(center_y - target_h / 2)
        y2_new = int(center_y + target_h / 2)
        x1_new, x2_new = x1_org, x2_org

        y1_roi = max(0, y1_new)
        y2_roi = min(self.height, y2_new)
        x1_roi = max(0, x1_new)
        x2_roi = min(self.width, x2_new)

        crop_img = self.image[y1_roi:y2_roi, x1_roi:x2_roi].copy()

        pad_top = abs(min(0, y1_new))
        pad_bottom = max(0, y2_new - self.height)
        pad_left = abs(min(0, x1_new))
        pad_right = max(0, x2_new - self.width)

        final_img = crop_img
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            border_color = (0, 0, 0, 0) if use_mask else (0, 0, 0)
            final_img = cv2.copyMakeBorder(
                crop_img,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=border_color,
            )

        ext = "jpg"
        if use_mask and mask_coords:
            ext = "png"

        if upscale_factor != 1.0:
            final_img = cv2.resize(
                final_img,
                dsize=None,
                fx=upscale_factor,
                fy=upscale_factor,
                interpolation=cv2.INTER_CUBIC,
            )

        filename = f"{label}_{int(det.confidence * 100)}_{index}.{ext}"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, final_img)

        return {
            "id": det.id,
            "label": label,
            "filename": filename,
            "path": f"/temp/crops/{filename}",
        }

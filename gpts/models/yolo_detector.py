"""
YOLO Detector Implementation for YOLOv8/YOLO11 segmentation models.
"""

import time
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import List, Optional
import random

from config import Config

from .base_detector import BaseDetector
from .schemas import (
    DetectionResponse,
    Detection,
    BoundingBox,
    SegmentationMask,
    ModelMetrics,
    ModelInfo,
    ImageInfo,
)
from utils.post_processing import (
    ISOLATED_GUARD_HIGH_CONF,
    ISOLATED_GUARD_IOU,
    ISOLATED_GUARD_LOW_CONF,
    apply_fdi_template_correction,
    filter_results_with_isolated_guard,
)


class YOLODetector(BaseDetector):
    """YOLO-based detector with optional FDI correction."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        use_fdi_correction: bool = False,
    ):
        super().__init__(model_path, confidence_threshold)
        self.device = device
        self.color_map: dict[int, str] = {}
        self.use_fdi_correction = use_fdi_correction

    def load_model(self):
        """Load a YOLO model from disk."""
        print(f"Loading YOLO model from {self.model_path}")
        try:
            ckpt = torch.load(self.model_path, map_location="cpu")

            if isinstance(ckpt, dict) and "model" not in ckpt and "ema" not in ckpt:
                print("Non-standard checkpoint format detected. Converting...")
                temp_path = str(
                    Path(self.model_path).parent
                    / f"temp_{Path(self.model_path).name}"
                )

                new_ckpt = {"model": ckpt}
                torch.save(new_ckpt, temp_path)
                print(f"Converted checkpoint saved to {temp_path}")

                self.model = YOLO(temp_path)
                self.model.to(self.device)
                print(f"Model loaded successfully on {self.device}")
            else:
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def predict(
        self,
        image_path: str,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        imgsz: int = 1280,
        retina_masks: bool = True,
        **kwargs,
    ) -> DetectionResponse:
        """Run inference on an image and return structured detections."""
        if not self.is_loaded():
            self.load_model()

        use_isolated_guard = kwargs.pop("use_isolated_guard", False)
        isolated_guard_high_conf = kwargs.pop(
            "isolated_guard_high_conf", ISOLATED_GUARD_HIGH_CONF
        )
        isolated_guard_low_conf = kwargs.pop(
            "isolated_guard_low_conf", ISOLATED_GUARD_LOW_CONF
        )
        isolated_guard_iou = kwargs.pop(
            "isolated_guard_iou", ISOLATED_GUARD_IOU
        )

        total_start = time.time()

        preprocess_start = time.time()
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img_height, img_width = image.shape[:2]
        image_format = Path(image_path).suffix[1:]
        preprocess_time = (time.time() - preprocess_start) * 1000

        inference_start = time.time()
        results = self.model(
            image_path,
            conf=min(self.confidence_threshold, isolated_guard_low_conf)
            if use_isolated_guard
            else self.confidence_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            retina_masks=retina_masks,
            verbose=False,
            classes=classes,
        )
        inference_time = (time.time() - inference_start) * 1000

        postprocess_start = time.time()
        parsed_result = results[0]
        if use_isolated_guard:
            parsed_result = filter_results_with_isolated_guard(
                parsed_result,
                high_conf=isolated_guard_high_conf,
                low_conf=isolated_guard_low_conf,
                iou_guard=isolated_guard_iou,
            )

        detections = self._parse_results(parsed_result, img_width, img_height)

        if self.use_fdi_correction and len(detections) > 0:
            print(
                f"Applying FDI template-based correction to {len(detections)} detections..."
            )
            detections = apply_fdi_template_correction(
                detections, confidence_threshold=0.7
            )
            print("FDI correction completed")

        postprocess_time = (time.time() - postprocess_start) * 1000
        total_time = (time.time() - total_start) * 1000

        response = DetectionResponse(
            success=True,
            message=f"Detected {len(detections)} object(s)",
            detections=detections,
            metrics=ModelMetrics(
                preprocessing_time_ms=round(preprocess_time, 2),
                inference_time_ms=round(inference_time, 2),
                postprocessing_time_ms=round(postprocess_time, 2),
                total_time_ms=round(total_time, 2),
            ),
            model_info=self.get_model_info(),
            image_info=ImageInfo(
                width=img_width, height=img_height, format=image_format
            ),
        )

        return response

    def _parse_results(self, result, img_width: int, img_height: int) -> List[Detection]:
        """Convert YOLO outputs into Detection objects."""
        detections: List[Detection] = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        has_masks = result.masks is not None

        for idx, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            bbox = BoundingBox(
                x=float(x1),
                y=float(y1),
                width=float(x2 - x1),
                height=float(y2 - y1),
            )

            fdi_number = Config.CLASS_ID_TO_FDI.get(cls_id, cls_id)
            label = str(fdi_number)

            color = self._get_color_for_class(cls_id)

            seg_mask = None
            if has_masks:
                seg_mask = self._extract_mask(
                    result.masks.data[idx], img_width, img_height
                )

            detection = Detection(
                id=idx,
                label=label,
                class_id=int(cls_id),
                confidence=float(conf),
                bounding_box=bbox,
                segmentation_mask=seg_mask,
                color=color,
            )

            detections.append(detection)

        return detections

    def _extract_mask(
        self, mask_tensor, img_width: int, img_height: int
    ) -> Optional[SegmentationMask]:
        """Convert a mask tensor into polygon coordinates."""
        try:
            mask_np = mask_tensor.cpu().numpy()

            mask_resized = cv2.resize(
                mask_np, (img_width, img_height), interpolation=cv2.INTER_NEAREST
            )

            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                c = max(contours, key=cv2.contourArea)

                if cv2.contourArea(c) < 1.0:
                    return None

                polygon = c.reshape(-1, 2).astype(int).tolist()

                return SegmentationMask(
                    format="polygon",
                    size=[img_height, img_width],
                    counts=polygon,
                )

            return None

        except Exception as e:
            print(f"Failed to encode mask: {e}")
            return None

    def _get_color_for_class(self, class_id: int) -> str:
        """Assign a consistent hex color per class id."""
        if class_id not in self.color_map:
            random.seed(class_id)
            r = random.randint(50, 255)
            g = random.randint(50, 255)
            b = random.randint(50, 255)
            self.color_map[class_id] = f"#{r:02x}{g:02x}{b:02x}"

        return self.color_map[class_id]

    def get_model_info(self) -> ModelInfo:
        """Return metadata for the loaded model."""
        if not self.is_loaded():
            self.load_model()

        model_name = Path(self.model_path).stem

        return ModelInfo(
            name=model_name,
            version="8.0.0",
            task="segment" if "seg" in model_name else "detect",
        )

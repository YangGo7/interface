"""
Models package
데이터 스키마와 모델 관련 코드
"""

from .schemas import (
    BoundingBox,
    SegmentationMask,
    Detection,
    ModelMetrics,
    ModelInfo,
    ImageInfo,
    DetectionResponse,
    ErrorResponse
)

from .base_detector import BaseDetector
from .yolo_detector import YOLODetector

__all__ = [
    "BoundingBox",
    "SegmentationMask",
    "Detection",
    "ModelMetrics",
    "ModelInfo",
    "ImageInfo",
    "DetectionResponse",
    "ErrorResponse",
    "BaseDetector",
    "YOLODetector"
]

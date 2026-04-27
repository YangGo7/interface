"""
Data schemas for the object detection API.
"""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float = Field(..., description="Top-left X coordinate in pixels")
    y: float = Field(..., description="Top-left Y coordinate in pixels")
    width: float = Field(..., description="Box width in pixels")
    height: float = Field(..., description="Box height in pixels")


class SegmentationMask(BaseModel):
    format: str = Field(default="rle", description="Mask format: rle, polygon, or base64")
    size: List[int] = Field(..., description="[height, width] of the image")
    counts: Any = Field(..., description="RLE payload or polygon coordinate list")


class Detection(BaseModel):
    id: int = Field(..., description="Detection index")
    label: str = Field(..., description="Human-readable label")
    class_id: int = Field(..., description="Raw model class id")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bounding_box: BoundingBox = Field(..., description="Bounding box")
    segmentation_mask: Optional[SegmentationMask] = Field(None, description="Optional segmentation mask")
    color: str = Field(..., description="Display color as a hex string")
    original_color: Optional[str] = Field(None, description="Original model color for comparisons")
    gt_color: Optional[str] = Field(None, description="Ground-truth comparison color")
    gt_iou: Optional[float] = Field(None, description="Ground-truth IoU")
    gt_label_match: Optional[bool] = Field(None, description="Whether the GT label matched")


class ModelMetrics(BaseModel):
    preprocessing_time_ms: float = Field(..., description="Preprocessing time in milliseconds")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    postprocessing_time_ms: float = Field(..., description="Postprocessing time in milliseconds")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")


class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    task: str = Field(default="segment", description="Task type such as detect or segment")


class ImageInfo(BaseModel):
    width: int = Field(..., description="Input image width in pixels")
    height: int = Field(..., description="Input image height in pixels")
    format: str = Field(..., description="Input image format")


class DetectionResponse(BaseModel):
    success: bool = Field(default=True, description="Whether detection succeeded")
    message: str = Field(default="Detection completed successfully", description="Status message")
    detections: List[Detection] = Field(..., description="Detected objects")
    metrics: ModelMetrics = Field(..., description="Runtime metrics")
    detector_info: ModelInfo = Field(..., description="Metadata for the loaded detector")
    image_info: ImageInfo = Field(..., description="Information about the input image")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response creation timestamp",
    )


class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error category")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

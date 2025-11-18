"""
Data schemas for object detection API
Pydantic을 사용한 데이터 검증 및 직렬화
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class BoundingBox(BaseModel):
    """
    바운딩 박스 좌표
    YOLO 출력: [x_center, y_center, width, height] -> [x, y, width, height]로 변환
    """
    x: float = Field(..., description="좌상단 X 좌표 (픽셀)")
    y: float = Field(..., description="좌상단 Y 좌표 (픽셀)")
    width: float = Field(..., description="박스 너비 (픽셀)")
    height: float = Field(..., description="박스 높이 (픽셀)")


class SegmentationMask(BaseModel):
    """
    세그멘테이션 마스크 - RLE/Polygon 형식
    - RLE: Run-Length Encoding (COCO format)
    - Polygon: [[x,y], [x,y], ...] 좌표 리스트
    """
    format: str = Field(default="rle", description="마스크 형식: rle, polygon, base64")
    size: List[int] = Field(..., description="[height, width] 이미지 크기")
    counts: Any = Field(..., description="RLE 문자열 or Polygon 좌표 리스트 [[x,y], ...]")


class Detection(BaseModel):
    """
    개별 탐지 객체 (하나의 객체 = 하나의 Detection)
    """
    id: int = Field(..., description="탐지 객체 고유 ID (0부터 시작)")
    label: str = Field(..., description="클래스 라벨 (예: 'person', 'car')")
    class_id: int = Field(..., description="클래스 ID (COCO 클래스 번호)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 점수 (0.0~1.0)")
    bounding_box: BoundingBox = Field(..., description="바운딩 박스 좌표")
    segmentation_mask: Optional[SegmentationMask] = Field(None, description="세그멘테이션 마스크 (있을 경우)")
    color: str = Field(..., description="시각화용 색상 (hex 코드, 예: '#FF5733')")


class ModelMetrics(BaseModel):
    """
    모델 성능 지표
    """
    preprocessing_time_ms: float = Field(..., description="이미지 전처리 시간 (밀리초)")
    inference_time_ms: float = Field(..., description="모델 추론 시간 (밀리초)")
    postprocessing_time_ms: float = Field(..., description="후처리 시간 (밀리초)")
    total_time_ms: float = Field(..., description="전체 처리 시간 (밀리초)")


class ModelInfo(BaseModel):
    """
    사용된 모델 정보
    """
    name: str = Field(..., description="모델 이름 (예: 'yolov8n-seg')")
    version: str = Field(..., description="모델 버전")
    task: str = Field(default="segment", description="작업 유형: detect, segment, pose 등")


class ImageInfo(BaseModel):
    """
    입력 이미지 정보
    """
    width: int = Field(..., description="원본 이미지 너비 (픽셀)")
    height: int = Field(..., description="원본 이미지 높이 (픽셀)")
    format: str = Field(..., description="이미지 포맷 (jpg, png 등)")


class DetectionResponse(BaseModel):
    """
    전체 탐지 결과 응답
    이 구조가 API의 최종 출력입니다!
    """
    success: bool = Field(default=True, description="요청 성공 여부")
    message: str = Field(default="Detection completed successfully", description="상태 메시지")
    detections: List[Detection] = Field(..., description="탐지된 객체 리스트")
    metrics: ModelMetrics = Field(..., description="성능 지표")
    model_info: ModelInfo = Field(..., description="사용된 모델 정보")
    image_info: ImageInfo = Field(..., description="입력 이미지 정보")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="응답 생성 시각")


class ErrorResponse(BaseModel):
    """
    에러 응답
    """
    success: bool = Field(default=False)
    message: str = Field(..., description="에러 메시지")
    error_type: str = Field(..., description="에러 타입")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

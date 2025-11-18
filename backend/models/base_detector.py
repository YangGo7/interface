"""
Base Detector Class
모든 탐지 모델의 기본 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from .schemas import DetectionResponse


class BaseDetector(ABC):
    """
    모든 객체 탐지 모델의 기본 클래스

    새로운 모델을 추가할 때:
    1. 이 클래스를 상속받기
    2. load_model(), predict(), get_model_info() 구현하기
    3. 출력은 항상 DetectionResponse 형식으로!
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        """
        Args:
            model_path: 모델 가중치 파일 경로
            confidence_threshold: 최소 신뢰도 (이보다 낮으면 무시)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None

    @abstractmethod
    def load_model(self):
        """
        모델을 메모리에 로드

        예:
            self.model = YOLO(self.model_path)
        """
        pass

    @abstractmethod
    def predict(self, image_path: str, **kwargs) -> DetectionResponse:
        """
        이미지에서 객체 탐지 수행

        Args:
            image_path: 입력 이미지 경로
            **kwargs: 추가 옵션 (iou_threshold 등)

        Returns:
            DetectionResponse: 표준화된 탐지 결과
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환

        Returns:
            dict: {"name": "yolov8n", "version": "8.0.0", ...}
        """
        pass

    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인"""
        return self.model is not None

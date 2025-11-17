"""
Configuration settings
환경 설정 (모델 경로, 포트 번호 등)
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).resolve().parent

class Config:
    """기본 설정"""

    # Flask 설정
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = True

    # CORS 설정 (프론트엔드 주소)
    CORS_ORIGINS = [
        "http://localhost:3000",  # React 기본 포트
        "http://localhost:5173",  # Vite 기본 포트
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]

    # 모델 설정
    MODEL_DIR = BASE_DIR / "weights"  # 모델 가중치 저장 디렉토리
    DEFAULT_MODEL = "yolov8n-seg.pt"  # 기본 모델 (nano - 가장 빠름)

    # 지원하는 모델 목록
    SUPPORTED_MODELS = {
        "yolov8n-seg": {
            "path": "yolov8n-seg.pt",
            "description": "YOLOv8 Nano Segmentation (가장 빠름)",
            "size": "small"
        },
        "yolov8s-seg": {
            "path": "yolov8s-seg.pt",
            "description": "YOLOv8 Small Segmentation (빠름)",
            "size": "small"
        },
        "yolov8m-seg": {
            "path": "yolov8m-seg.pt",
            "description": "YOLOv8 Medium Segmentation (중간)",
            "size": "medium"
        },
        "yolo11n-seg": {
            "path": "yolo11n-seg.pt",
            "description": "YOLO11 Nano Segmentation (최신)",
            "size": "small"
        },
        "yolo11s-seg": {
            "path": "yolo11s-seg.pt",
            "description": "YOLO11 Small Segmentation (최신)",
            "size": "small"
        }
    }

    # 이미지 설정
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

    # 추론 설정
    CONFIDENCE_THRESHOLD = 0.25  # 최소 신뢰도 (이보다 낮으면 무시)
    IOU_THRESHOLD = 0.45  # Non-Maximum Suppression 임계값
    MAX_DETECTIONS = 100  # 최대 탐지 객체 수


class DevelopmentConfig(Config):
    """개발 환경 설정"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """프로덕션 환경 설정"""
    DEBUG = False
    TESTING = False


# 환경에 따라 설정 선택
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

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

    # 경로 설정
    BASE_DIR = BASE_DIR  # 프로젝트 루트 디렉토리

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
    DEFAULT_MODEL = "yolo11_mask.pt"  # 기본 모델 (nano - 가장 빠름)

    # 지원하는 모델 목록
    SUPPORTED_MODELS = {
        "yolo11_mask.pt": {
            "path": "yolo11_mask.pt",
            "description": "YOLOv8 Nano Segmentation (데이터 불균형)",
            "size": "small"
        },
        "yolo11l-seg": {
            "path": "yolo11l-seg.pt",
            "description": "yolo11l-seg.pt",
            "size": "small"
        },
        "yolov8l-seg": {
            "path": "yolov8l-seg.pt",
            "description": "YOLOv8 Medium Segmentation (중간)",
            "size": "medium"
        },
        "수정중": {
            "path": "yolo11n-seg.pt",
            "description": "YOLO11 Nano Segmentation (최신)",
            "size": "small"
        },
        "fix": {
            "path": "yolo11s-seg.pt",
            "description": "YOLO11 Small Segmentation (최신)",
            "size": "small"
        }
    }

    # 이미지 설정
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

    # 추론 설정
    CONFIDENCE_THRESHOLD = 0.5  # 최소 신뢰도 (이보다 낮으면 무시)
    IOU_THRESHOLD = 0.8  # Non-Maximum Suppression 임계값
    MAX_DETECTIONS = 100  # 최대 탐지 객체 수

    # FDI (Fédération Dentaire Internationale) 치아 번호 매핑
    # YOLO class_id (0-31) -> FDI 번호 (11-48)
    CLASS_ID_TO_FDI = {
        # 우측 상악 (11-18)
        0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18,
        # 좌측 상악 (21-28)
        8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28,
        # 좌측 하악 (31-38)
        16: 31, 17: 32, 18: 33, 19: 34, 20: 35, 21: 36, 22: 37, 23: 38,
        # 우측 하악 (41-48)
        24: 41, 25: 42, 26: 43, 27: 44, 28: 45, 29: 46, 30: 47, 31: 48
    }


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

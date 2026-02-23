"""
Configuration settings (simplified to single YOLO segmentation model).
"""

import os
from pathlib import Path


# project root
BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base config."""

    # paths
    BASE_DIR = BASE_DIR
    MODEL_DIR = BASE_DIR / "weights"  # directory containing model weights
    DEFAULT_MODEL = "best"  # weights/best.pt
    DEFAULT_PANO_MODEL = "pano_seg"  # segmentation for tooth mask

    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = True

    # CORS
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # Supported models (single YOLO seg)
    SUPPORTED_MODELS = {
        "best": {
            "path": "yolo26l-1024-teeth_seg.pt",
            "description": "Tooth segmentation (YOLO seg)",
            "size": "small",
            "type": "yolo",
            "default_confidence": 0.25,
            "default_iou": 0.7,
            "default_imgsz": 1024,
        }
    }
    # Multi-model pano pipeline (seg + disease)
    PANO_MODELS = {
        "pano_seg": {
            "path": "best.pt",
            "type": "seg",
            "default_confidence": 0.25,
            "default_iou": 0.6,
            "default_imgsz": 1024,
            "description": "Tooth segmentation (YOLO26L)",
        },
        "caries": {
            "path": "caries_det.pt",
            "type": "det",
            "default_confidence": 0.05,
            "default_iou": 0.5,
            "default_imgsz": 1024,
            "description": "Dental caries detection",
        },
        "periapical": {
            "path": "periapical.pt",
            "type": "det",
            "default_confidence": 0.2,
            "default_iou": 0.5,
            "default_imgsz": 1024,
            "description": "Periapical lesion detection",
        },
        "cej": {
            "path": "cej.pt",
            "type": "seg",
            "default_confidence": 0.1,
            "default_iou": 0.5,
            "default_imgsz": 1024,
            "description": "CEJ segmentation",
        },
        "bonelevel": {
            "path": "bonelevel.pt",
            "type": "seg",
            "default_confidence": 0.25,
            "default_iou": 0.5,
            "default_imgsz": 1024,
            "description": "Bone level segmentation",
        },
        "iac": {
            # Nerve/Sinus segmentation (YOLO26m-seg) trained with labels: NERVE, SINUS
            "path": "yolo26m-ioc.pt",
            "type": "seg",
            "default_confidence": 0.25,
            "default_iou": 0.5,
            "default_imgsz": 1024,
            "description": "Inferior Alveolar Canal segmentation",
        },
    }

    # Upload settings
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp", "dcm"}

    # Defaults
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.8
    MAX_DETECTIONS = 100
    MAX_JOBS = 8  # pano async jobs queue limit

    # Pixel -> mm scale for diameter calculations
    # Set to 0.1 to reduce 50px -> 5.0mm (tweak if your image DPI differs)
    DIAMETER_PIXEL_TO_MM = 0.1

    # Label name mapping (FDI -> readable category)
    # Optional display name map (FDI -> text)
  
    # FDI mapping
    CLASS_ID_TO_FDI = {
        0: 11,
        1: 12,
        2: 13,
        3: 14,
        4: 15,
        5: 16,
        6: 17,
        7: 18,
        8: 21,
        9: 22,
        10: 23,
        11: 24,
        12: 25,
        13: 26,
        14: 27,
        15: 28,
        16: 31,
        17: 32,
        18: 33,
        19: 34,
        20: 35,
        21: 36,
        22: 37,
        23: 38,
        24: 41,
        25: 42,
        26: 43,
        27: 44,
        28: 45,
        29: 46,
        30: 47,
        31: 48,
        32: "implant",
        
        
    }
    # FDI -> readable names
    LABEL_NAME_MAP = {
       
    }



class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False


# environment mapping
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}

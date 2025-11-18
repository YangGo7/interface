YOLOv11 기반의 치아 탐지 및 분석 시스템입니다. Flask API를 통해 이미지를 입력받아 치아 객체를 탐지하고, 결손치(Missing Tooth)를 분석하며, 개별 치아 이미지를 고화질로 크롭(Crop)하고 종합 리포트를 생성합니다.

## ✨ 주요 기능 (Features)

* **AI 객체 탐지:** YOLOv11-seg 모델을 활용한 정밀한 치아 및 마스크 탐지
* **결손치 분석 (Missing Tooth Analysis):** FDI 표기법(11~48번) 기준으로 없는 치아 자동 식별
* **스마트 크롭 (Smart Cropping):**
    * 치근 확인을 위한 상하 확장(Padding) 및 고화질 업스케일링(Upscaling)
    * Box Crop 및 Mask Crop(배경 제거) 모드 지원
* **자동 리포트 생성:** 탐지 결과 시각화 및 분석 통계가 포함된 HTML 리포트 제공
* **RESTful API:** 프론트엔드 및 외부 시스템 연동을 위한 API 제공

## 🛠 기술 스택 (Tech Stack)

* **Backend:** Python, Flask
* **AI/ML:** PyTorch, Ultralytics YOLOv11
* **Image Processing:** OpenCV, NumPy
* **Frontend:** HTML, CSS, JavaScript (Vanilla) 추후  react 이동

디렉토리 구조

interface/                  # 프로젝트 루트 (Root)
├── backend/                # 백엔드 (Flask 서버)
│   ├── app.py              # [메인] Flask 실행 파일 (전체 통합 코드)
│   ├── config.py           # 환경 설정 (모델 경로, 포트 등)
│   ├── requirements.txt    # 의존성 라이브러리 목록
│   ├── models/             # AI 모델 관련 코드
│   │   ├── __init__.py
│   │   ├── base_detector.py
│   │   ├── yolo_detector.py
│   │   └── schemas.py
│   ├── utils/              # [생성 필요] 후처리 및 리포트 유틸리티
│   │   ├── __init__.py     # (빈 파일, 패키지 인식용)
│   │   ├── post_processing.py # ObjectCropper, MissingToothFinder
│   │   └── report.py       # ReportGenerator (HTML 리포트 생성)
│   └── weights/            # [중요] 모델 가중치 파일 저장소
│       ├── yolov8n-seg.pt  # (예시) 실제 사용하는 .pt 파일들
│       └── yolo11-seg.pt
│
└── frontend/               # 프론트엔드 (Web UI)
    ├── index.html          # 메인 HTML 페이지
    ├── style.css           # 스타일 시트
    └── script.js           # 프론트엔드 로직


# 실행 방법 

# 1. 의존성 설치
pip install -r backend/requirements.txt

# 2. 서버 실행 (모듈 경로 인식을 위해 -m 옵션 사용 권장, 혹은 직접 실행)
python backend/app.py
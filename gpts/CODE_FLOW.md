# GPTS 코드 흐름 (Code Flow)

**최종 업데이트**: 2026-02-10

이 문서는 `gpts` 서버의 요청 처리 흐름과 주요 모듈 간의 관계를 설명합니다.

---

## 1. 전체 아키텍처 개요

`gpts`는 **Flask 기반의 REST API 서버**로, 치과 파노라마 이미지를 받아 AI 분석을 수행하고 JSON 결과와 리포트를 반환합니다.

### **핵심 흐름**
1.  **클라이언트** (Frontend/Postman)가 이미지 업로드 (`POST /api/v2/detect`)
2.  **API 라우터** (`api/routes_v2.py`)가 요청 수신
3.  **Core Pipeline** (`services/pano_inference.py`)이 AI 추론 실행
4.  **Business Logic** (`services/tooth_logic.py`, `utils/post_processing`)이 결과 가공
5.  **Report Generator** (`utils/report_v3.py`)가 HTML 리포트 생성
6.  **Response**: JSON 결과 및 리포트 URL 반환

---

## 2. 상세 흐름 (Step-by-Step)

### **Step 1: 서버 시작 (Entry Point)**
-   **파일**: `app.py`
-   **역할**:
    -   Flask 앱 초기화 (`app = Flask(...)`)
    -   설정 로드 (`config.py`)
    -   AI 모델 파이프라인(`PanoPipeline`) **미리 로드 (Preload)**
    -   API 블루프린트 등록 (`/api/v2/...`)

### **Step 2: 요청 처리 (Request Handling)**
-   **파일**: `api/routes_v2.py`
-   **Endpoint**: `/detect`
-   **과정**:
    1.  이미지 파일 수신 및 저장 (`temp/`)
    2.  `inference_service.run(...)` 호출 (비동기 처리 가능)

### **Step 3: AI 추론 (Inference Pipeline)**
-   **파일**: `services/pano_inference.py`
-   **클래스**: `PanoPipeline`
-   **과정**:
    1.  **YOLO 모델 실행**:
        -   `pano_seg` (치아 분할)
        -   `caries`, `periapical` (질환 탐지)
        -   `cej`, `bonelevel`, `iac` (해부학 구조)
    2.  **후처리 (Post-processing)**:
        -   `tooth_logic.process_teeth(...)`: 치아 번호 할당 및 결손치 계산

### **Step 4: 비즈니스 로직 (Business Logic)**
-   **파일**: `services/tooth_logic.py`
    -   **`calculate_nerve_safety()`**: 임플란트 안전거리 계산 (Raycast)
    -   **`find_missing_teeth()`**: 결손치 공간 분석, 인접치 축(Axis) 계산
-   **파일**: `services/pano_calc_utils.py`
    -   순수 수학/기하학 연산 (거리, 각도, PCA 등)

### **Step 5: 결과 생성 (Reporting)**
-   **파일**: `utils/report_v3.py`
-   **역할**:
    -   분석된 데이터를 바탕으로 HTML 리포트 생성
    -   이미지 위에 오버레이(Overlay) 그리기 (`_draw_predictions`)
    -   `gpts/reports/` 폴더에 HTML 저장

---

## 3. 주요 디렉토리 구조

```
gpts/
├── app.py                # [Main] 서버 실행 진입점
├── config.py             # 설정 (모델 경로, 포트 등)
├── .env                  # 환경 변수 (API Key 등)
├── api/
│   └── routes_v2.py      # [Controller] API 엔드포인트 정의
├── services/             # [Service] 핵심 비즈니스 로직
│   ├── pano_inference.py # AI 파이프라인 (Orchestrator)
│   ├── tooth_logic.py    # 치아 관련 로직 (결손치, 신경관)
│   ├── pano_calc_utils.py# 계산 유틸리티
│   └── visualizer.py     # 시각화 도구
├── utils/                # [Util] 보조 모듈
│   ├── report_v3.py      # 리포트 생성기
│   ├── sample_axis.py    # 샘플 축 계산
│   └── post_processing/  # 기타 후처리 (Cropper 등)
├── weights/              # [Model] AI 모델 파일 (.pt)
├── reports/              # [Output] 생성된 리포트 저장소
└── temp/                 # [Temp] 업로드된 이미지 임시 저장
```

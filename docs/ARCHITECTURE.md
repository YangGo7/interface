# 프로젝트 문서 (요약)

## 목적
- 치과 영상(세그멘테이션 모델) 기반 FDI 치식 감지/표시
- 임플란트 직경/축 계산 및 오버레이
- 리포트/크롭 결과 제공

## 주요 구성
- **백엔드**: Flask (`backend/app.py`), YOLO 추론, 후처리, 직경/축 계산
- **프론트**: 정적 HTML/JS (`frontend/index.html`, `scripts/index_scripts.js`), 캔버스 오버레이 렌더링

## 핵심 설정 (`backend/config.py`)
- `CLASS_ID_TO_FDI`: YOLO class_id → FDI 숫자 매핑
- `LABEL_NAME_MAP`: FDI → 표시 이름(32: bridge, 33: crown, 34: endo, 35: implant)
- `DIAMETER_PIXEL_TO_MM`: 픽셀→mm 스케일(기본 0.1, 이미지 스케일에 따라 조정)
- 업로드 허용 확장자/사이즈, CORS, 기본 모델 경로 등

## 백엔드 동작 (`backend/app.py`)
- 엔드포인트:
  - `GET /api/health`, `GET /api/models`
  - `POST /api/detect`: 이미지 업로드 → YOLO 추론 → 후처리 → JSON 응답
- 주요 흐름:
  1) 입력 검증(확장자/파일 존재)
  2) YOLO 추론 → class_id→FDI 변환
  3) 중복 정리: FDI별 1개만 남기되 32/33/34/35는 중복 허용
  4) 임플란트(35/implant) 직경/축 계산:
     - 조건: segmentation_mask + 그레이 이미지 존재
     - `compute_from_gray_with_mask`(CLAHE/HEQ→threshold→컨투어→PCA)
     - 직경 = `max_d * DIAMETER_PIXEL_TO_MM` 후 0.5 스텝 버킷
     - 축 길이 = bbox 높이 × 1.5
  5) 응답: `detections`, `diameter_metrics`, `overlay_lines`, `analysis`, `crops`, `report_url` 등
- 디버그: `PRINT_FDI_DEBUG=1` 환경변수 시 class_id→FDI 매핑 로그 출력

## 유틸 (`backend/utils/sample_axis_service.py`)
- `compute_from_gray_with_mask`: 그레이+마스크 → CLAHE+HEQ → threshold → 최대 컨투어 → `compute_sample_axis`
- `compute_sample_axis`: PCA 축, 최대 폭, 직경 끝점 계산

## 프론트 (`frontend/scripts/index_scripts.js`)
- `/api/detect` 호출, 캔버스 오버레이 렌더링
- 직경 표시: `overlay_lines.length_mm` → 0.5 스텝 버킷 → `ø {값} mm`
- 축: 선만 표시

## 엣지/제한
- 직경/축: 마스크 또는 그레이 이미지 없으면 계산/표시 안 됨
- mm 스케일은 `DIAMETER_PIXEL_TO_MM` 의존; DICOM PixelSpacing 미사용(추가 필요 시 확장)
- CORS는 로컬 도메인 위주(운영 시 제한 필요)

## 성능/보안
- CLAHE/컨투어는 CPU 처리(임플란트 대상이라 부담 제한적)
- 업로드 확장자/사이즈 검증, 민감 로그 없음

## 사용 예시
1) 서버 실행: `PRINT_FDI_DEBUG=0 python backend/app.py`
2) 프론트에서 이미지 업로드 → `/api/detect`
3) 캔버스에서 마스크/라벨/직경/축 오버레이 확인 (직경은 0.5mm 스텝, 스케일은 `DIAMETER_PIXEL_TO_MM` 적용)

## 구조 정리 제안 (향후)
- `backend/routers/` (API 라우트 분리), `backend/services/` (직경/축, GT 비교), `backend/models/` (데이터 스키마), `backend/config/` (환경/매핑)
- 프론트: `frontend/scripts/` 내 API/렌더/상태 관리 모듈 분리, 공용 util 함수 파일화
- 긴 파일(`app.py`, `index_scripts.js`)을 기능별로 분할해 가독성 및 유지보수성 개선

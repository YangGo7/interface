# Overview

본 문서는 Panorama API/Frontend의 주요 구성과 코드 위치를 간단히 설명합니다.

## 구조 요약
- **Backend (Flask)**: `backend/app.py`
  - 메인 추론: `/api/detect` (비동기 상태 조회 `/api/detect/status/<job_id>` 사용 가능)
  - 테스트 전용 분리 추론: `/api/test_split_detect` (여러 모델을 분리 실행)
  - 서비스 로직: `backend/services/pano_inference.py`, 후처리 `backend/services/postprocess.py`
  - 테스트 헬퍼: `backend/test/run_split_infer.py`, `backend/test/split_helper.py`
- **Frontend (Vite/React)**: `frontend/src`
  - 메인 UI: `pages/ChartPage.tsx` (뷰어, 오도토그램, 사이드패널)
  - 테스트 UI: `pages/TestPage.tsx` (/test 라우트)
  - 헤더/공통 컴포넌트: `components/TopHeader.tsx`, `components/BottomTeethChart.tsx` 등

## 주요 데이터 흐름
1. 사용자가 이미지를 업로드 → `/api/detect` 또는 `/api/test_split_detect` 호출
2. 백엔드가 이미지 저장 후 모델 추론 → 결과(overlay/image URL, 각종 메트릭) 반환
3. 프런트는 URL을 뷰어/카드에 표시, 오도토그램/사이드패널에 치식·병소·임플란트 상태 반영

## 모델·가중치 참고
백엔드 기본 경로 예시(`backend/weights`):
- `yolo11_seg_ver1_800_1024px.pt` : 치식 세그/디텍션
- `caries_det.pt` : 충치
- `periapical.pt` : 치근단염/기타
- `cej.pt` : CEJ
- `bonelevel.pt` : Bone level

테스트 모드(run_split_infer/test_split_detect)에서는 위 가중치를 분리 실행하여 all/teeth/caries/peri/extra(cej/bonelevel) 이미지를 생성합니다.

## 참고 문서 (docs/)
- `API_MAIN.md` : /api/detect, /api/test_split_detect 등 메인/테스트 API 명세
- `FRONT_GUIDE.md` : ChartPage/TestPage, 뷰어/오도토그램 매핑 안내
- `TEST_INFER.md` : 테스트 전용 분리 추론(run_split_infer, /api/test_split_detect) 사용 가이드
- `PANO_PIPELINE.md`, `ARCHITECTURE.md` : 기존 파이프라인/아키텍처 상세
- `AXIS_DIAMETER.md` : 축/직경 관련 기존 참고 자료

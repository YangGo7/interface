# 치과 파노라마 AI 분석 시스템 서비스 문서

작성 기준: 현재 저장소 코드 기준  
기준일: 2026-04-17

## 1. 서비스 개요

### 1.1 목적

이 시스템은 치과 파노라마 이미지 또는 일부 DICOM 기반 치과 영상을 업로드하거나 로컬 폴더에서 선택했을 때, AI가 치아 세그멘테이션과 주요 병변 후보를 자동 분석하고, 이를 뷰어와 오돈토그램, 리포트 형태로 시각화하는 것을 목표로 합니다.

해결하려는 문제:

- 파노라마 이미지에서 치아 위치와 상태를 빠르게 파악
- 우식, 치근단 병소, bone level, missing tooth, implant planning 등의 후보를 한 화면에서 확인
- 결과를 리포트로 재생성하고 의사 수정 사항을 반영

현재 사용 대상:

- 내부 개발/테스트
- 데모 시연
- 의사 또는 운영자 검증용 프로토타입 환경

현재 코드상 별도 로그인/권한 분리 기능은 없습니다.

### 1.2 한 줄 설명

→ “치과 파노라마 이미지 또는 DICOM을 업로드하거나 로컬 폴더에서 선택하면 AI가 자동 분석 결과를 뷰어, 오돈토그램, 리포트로 시각화하는 시스템”

### 1.3 전체 흐름

기본 흐름:

1. 이미지 업로드 또는 로컬 폴더에서 DICOM/이미지 선택
2. 서버에 원본 저장 또는 임시 세션 생성
3. AI 분석 비동기 실행
4. 결과 JSON 및 overlay 이미지 생성
5. 프론트엔드에서 viewer/odontogram/report로 표시

리포트 흐름:

1. 차트 결과에서 웹 리포트 세션 생성
2. AI 결과와 doctor override를 병합
3. HTML/PDF 리포트 생성
4. UI에서 미리보기, 재생성, 최종 확정

## 2. 시스템 구조

### 2.1 구성 요소

Frontend:

- React 18
- TypeScript
- Vite
- 일부 DICOM 시각화용 CornerstoneJS

Backend:

- Flask
- Flask-CORS
- Python 기반 비동기 작업 관리자

AI 모델:

- YOLO 계열 다중 모델 파이프라인
- 치아 세그멘테이션
- 우식 탐지
- 치근단 병소 탐지
- CEJ 세그멘테이션
- Bone level 세그멘테이션
- IAC(하치조신경관) 세그멘테이션

PACS / 스토리지:

- 현재 Orthanc 같은 별도 PACS 연동은 없음
- 대신 `C:/interface/case` 폴더를 로컬 DICOM/이미지 저장소처럼 사용하는 구조

저장소 / DB:

- 웹 리포트 세션: SQLite (`gpts/data/web_report.db`)
- 통계성 방문/기타 정보: SQLite (`gpts/data/stats.db`)
- 추론 산출물: `gpts/temp`, `gpts/runs/web_report`

### 2.2 데이터 흐름

```text
[User]
  ↓
[Frontend: Folder Loader / Upload / Renew]
  ↓
[Backend API: Flask]
  ↓
[AI Pipeline: PanoPipeline + YOLO models]
  ↓
[Temp files / SQLite session DB]
  ↓
[Frontend viewer / odontogram / report]
```

세부 예시:

```text
[User selects DICOM or image]
  ↓
/api/detect_async or /api/dicom-server/*
  ↓
PanoPipeline.run(...)
  ↓
overlay.png / preview.png / result JSON
  ↓
/api/detect/status/<job_id>
  ↓
RenewPage 표시
```

리포트 예시:

```text
[RenewPage result]
  ↓
/api/web_report/from-chart
  ↓
web_report.db session 생성
  ↓
HTML/PDF report 생성
  ↓
/api/web_report/session/<session_id>/report
  ↓
Report preview 표시
```

## 3. 사용자 매뉴얼

### 3.1 로그인

현재 로그인 기능은 없습니다.

접근 URL 예시:

- 메인 화면: `/`
- Folder Loader Ver2: `/folder_leader_ver_2`
- 업로드 페이지: `/upload`
- 차트 화면: `/renew` 또는 `/chart`

계정 방식:

- 없음

### 3.2 이미지 업로드

지원 포맷:

- 직접 업로드/API 분석: `JPG`, `JPEG`, `PNG`, `BMP`, `WEBP`, `DICOM(.dcm)`
- Folder Loader Ver2 이미지 브라우징: `PNG`, `JPG`, `JPEG`, `BMP`, `TIF`, `TIFF`, `WEBP`

주의:

- 폴더 브라우저는 TIFF/TIF 목록 표시가 가능하지만, 백엔드 AI 업로드 허용 확장자는 현재 `jpg/jpeg/png/bmp/webp/dcm` 기준입니다.

업로드 방법 1: 업로드 화면

1. `/upload` 접속
2. 파일 선택
3. 서버로 업로드
4. AI 분석 요청
5. 결과 페이지 또는 리포트로 이동

업로드 방법 2: 메인 화면(Folder Loader Ver2)

1. `/` 접속
2. Study 또는 Image 목록에서 항목 선택
3. `Open` 또는 더블클릭
4. `Renew` 화면으로 이동
5. 선택한 파일 기준으로 분석 또는 viewer 로드

### 3.3 AI 분석 실행

동작 방식:

- 일반 이미지: 선택 후 자동 분석
- DICOM study: 선택 후 viewer 로드와 함께 분석 흐름 진입
- 일부 리포트 API는 비동기 분석 후 status polling 구조 사용

자동/수동 여부:

- 현재 주요 `Renew` 흐름은 자동 분석에 가깝습니다.
- 리포트는 별도 생성/재생성 버튼이 있습니다.

처리 시간:

- GPU 환경에서는 보통 수 초~수십 초
- CPU 환경에서는 더 길 수 있음
- 이미지 크기, 모델 수, bone level/CEJ/nerve 계산 여부에 따라 달라짐

### 3.4 결과 확인

Viewer:

- 중앙 파노라마 이미지 위에 contour / detection / nerve / bone 관련 overlay 표시

Odontogram:

- 하단 치아 차트에 상태를 시각화
- Healthy / Missing / Implant / Treatment Required / Urgent Priority 등 표시

Overlay 의미:

- 노란 contour: 일반 치아 contour
- 파란 contour: implant 상태 또는 implant detection
- 빨간 contour/tag: caries detection
- 주황 contour/tag: periapical detection
- 보라 contour: nerve 관련 contour

Confidence 기준:

- 모델별 기본 confidence는 서로 다름
- 예시:
  - pano segmentation: 0.25
  - caries: 0.05
  - periapical: 0.2
  - cej: 0.1
  - bonelevel: 0.25
  - iac: 0.25

프론트 표시 단계에서 일부 detection은 별도 후처리와 threshold를 더 거칠 수 있습니다.

### 3.5 리포트 기능

리포트 생성 방법:

1. `Renew` 화면에서 Report 기능 진입
2. 차트 결과 기반으로 `/api/web_report/from-chart` 호출
3. 세션 생성 및 HTML/PDF 리포트 생성
4. Report workspace 또는 report panel에서 미리보기 확인

리포트 포함 내용:

- 원본/preview 이미지
- overlay 결과
- AI 결과 요약
- 치아별 finding
- doctor override 기반 보정 결과
- HTML / PDF 리포트 파일

리포트 관련 주요 기능:

- `Regenerate Preview`: 실제 리포트 재생성
- `Open Full`: 전체 리포트 페이지 열기
- `Finalize`: 최종 확정 및 PDF 생성 흐름

## 4. UI 설명

### 4.1 메인 화면

현재 기본 메인 화면은 `FolderLeaderVer2Page`입니다.

구성:

- 상단: 헤더
- 중앙 좌측: 필터 / 검색 / study list
- 중앙 우측: 선택된 study detail / series / preview
- 이미지 섹션일 경우: 이미지 목록 표시

차트 화면(`Renew`) 구성:

- 좌측: 툴바(Studies / View / Measure / Output / Task)
- 중앙: 파노라마 viewer
- 하단: Dental Chart(오돈토그램)
- 우측 고정 분석 결과 패널은 현재 기본 구조가 아님
- 필요 시 Studies dock, Report panel, HUD, DICOM metadata overlay가 뜸

### 4.2 주요 버튼

메인 / Folder Loader:

- `Refresh`
- `Open`
- 테이블 행 더블클릭

Renew 화면:

- `Studies`
- `Report`
- `Flip`
- `Invert`
- `Magnifier`
- `Length / Bidirectional / Angle`
- `Annotation / Arrow / Ellipse / Rectangle / Circle / Freehand ROI / Spline ROI / Livewire Tool`
- `Capture`
- `Capture Save`
- `Overlay`
- `Heatmap`
- `FDI / Univ`

## 5. 입력 / 출력 정의

### 입력

입력 유형:

- 2D 치과 파노라마 이미지
- DICOM 파일
- 로컬 폴더 내 study/series

주요 입력 필드:

- 이미지 파일(`image`)
- 언어(`language`)
- 사용자명/리포트명(`user_name`)

### 출력

출력 유형:

- tooth segmentation contour
- detection bounding box / contour
- confidence score
- 치아별 상태 map
- missing / implant metric / bone level 정보
- HTML/PDF report

대표 출력 데이터:

- `teeth`
- `data`
- `caries`
- `periapical`
- `bonelevel`
- `implant_metrics`
- `teeth_missing`
- `caries_by_tooth_best`
- `periapical_by_tooth_best`

## 6. 제한사항

반드시 인지해야 할 제한:

- 일부 이미지에서 detection 누락 가능
- DICOM과 PNG/JPG 결과가 동일하지 않을 수 있음
- 폴더 브라우저에서 보이는 모든 이미지 포맷이 AI 분석 가능 포맷과 완전히 일치하지 않음
- 특정 브라우저 또는 ngrok 환경에서 아이콘/경로/CORS 문제가 발생할 수 있음
- 분석 시간은 하드웨어와 이미지 상태에 따라 크게 달라질 수 있음
- posterior tooth assignment(특히 implant/missing 혼재 구간) 후처리 규칙은 아직 튜닝 중
- Viewer 플립, 오돈토그램 플립, detection 라벨 플립은 최근 수정 사항이므로 케이스별 검증이 필요
- 리포트는 생성 후 `Refresh`가 아니라 `Regenerate Preview`를 눌러야 실제 재생성됨

## 7. 에러 대응

### 업로드 실패

확인할 항목:

- 파일 확장자 지원 여부
- 파일 크기
- 브라우저 콘솔의 CORS/host 오류
- 백엔드 Flask 서버 실행 여부

대표 원인:

- 지원하지 않는 포맷
- `localhost:5000` 또는 `VITE_API_BASE_URL` 오설정
- ngrok host/allowedHosts/CORS 문제

### 결과가 안 나옴

확인 순서:

1. `/api/detect_async` 요청이 실제 나가는지 확인
2. `/api/detect/status/<job_id>`가 polling 되는지 확인
3. 백엔드 로그에서 예외 발생 여부 확인
4. 모델 weights 파일 존재 여부 확인
5. 특정 포맷(BMP 등) 샘플 자체 문제인지 확인

### DICOM 목록이 안 보임

확인 순서:

1. `C:/interface/case` 폴더 존재 여부
2. 해당 폴더에 `.dcm` / `.dicom` 파일이 있는지
3. `/api/dicom-server/studies` 응답 확인

### 리포트가 갱신되지 않음

확인 순서:

1. 세션이 `completed` 상태인지 확인
2. `Regenerate Preview` 호출 여부 확인
3. `/api/web_report/session/<session_id>` polling 결과 확인
4. `web_report.db` 및 `runs/web_report/<session_id>` 산출물 확인

## 8. 개발 참고

### API 구조

현재 자주 쓰는 API:

- `GET /api/health`
- `POST /api/detect`
- `POST /api/detect_async`
- `GET /api/detect/status/<job_id>`
- `POST /api/pano`
- `GET /api/pano/status/<job_id>`
- `GET /api/dicom-server/studies`
- `GET /api/dicom-server/preview`
- `GET /api/dicom-server/download`
- `GET /api/dicom-server/file`
- `POST /api/v2/analyze`
- `GET /api/v2/status/<task_id>`
- `POST /api/web_report/from-chart`
- `GET /api/web_report/session/<session_id>`
- `PATCH /api/web_report/session/<session_id>/overrides`
- `POST /api/web_report/session/<session_id>/report/regenerate`
- `POST /api/web_report/session/<session_id>/report/finalize`

### 주요 로직

Frontend 핵심:

- 메인 화면: `frontend/src/pages/FolderLeaderVer2Page.tsx`
- 기본 차트 화면: `frontend/src/pages/RenewPage.tsx`
- 레거시 차트: `frontend/src/pages/ChartPage.tsx`
- 웹 리포트 패널: `frontend/src/components/WebReportDrawer.tsx`
- 리포트 workspace: `frontend/src/components/chart/RenewReportWorkspacePanel.tsx`

Backend 핵심:

- Flask 진입점: `gpts/app.py`
- 로컬 DICOM 브라우저: `gpts/api/dicom_server_browser.py`
- 웹 리포트 API: `gpts/api/web_report.py`
- 파노라마 추론 파이프라인: `gpts/services/pano_inference.py`
- 치아 후처리 규칙: `gpts/services/pano_rules_engine.py`
- missing/gap 로직: `gpts/services/tooth_logic.py`

### AI inference 흐름

대표 흐름:

1. 업로드 또는 파일 선택
2. `requestAsyncDetection(...)`
3. `/api/detect_async`
4. `DetectJobManager`
5. `PanoPipeline.run(...)`
6. 모델별 결과 병합
7. bone level, caries, periapical, implant, missing 후처리
8. overlay 및 결과 JSON 생성
9. 프론트가 status polling 후 표시

### overlay 렌더링 방식

프론트는 결과 JSON을 바탕으로:

- tooth polygon
- detection contour/bounds
- nerve contour
- hover HUD
- dental chart image slot

를 별도로 렌더링합니다.

즉 viewer의 contour와 odontogram은 같은 결과를 참조하지만, 각각 독립 렌더링이므로 상태 불일치가 발생하면 프론트 렌더 조건과 백엔드 후처리 결과를 동시에 봐야 합니다.

### 실행 방법

Frontend 실행:

```powershell
cd c:\interface\frontend
npm install
npm run dev
```

Backend 실행:

```powershell
cd c:\interface\gpts
python app.py
```

통합 접속:

- 개발: `http://localhost:5173`
- 백엔드 직서빙: `http://localhost:5000`

AI 서버 실행:

- 현재는 별도 독립 AI 서버 프로세스가 아니라, Flask 앱 내부에서 `PanoPipeline`이 모델을 로드하는 구조입니다.
- 즉 `python app.py` 실행이 곧 API + AI 서버 실행입니다.

배포용 빌드:

```powershell
cd c:\interface
.\build_release.ps1 -Zip
```

## 9. 향후 계획

현재 코드 흐름과 최근 수정 방향을 기준으로 한 우선 과제:

- posterior tooth assignment 정확도 개선
- implant / natural tooth serial rule 개선
- flip 상태의 viewer/odontogram/label 일관성 추가 검증
- ngrok 및 외부 접근 안정화
- 리포트 템플릿/오돈토그램 디자인 개선
- UI 패널 구조 추가 분리 및 정리
- 이미지 포맷 지원 범위와 실제 inference 지원 범위 일치화
- 배포 문서 및 설치 스크립트 정리 강화


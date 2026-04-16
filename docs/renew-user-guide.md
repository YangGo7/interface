# Renew / Report Usage Guide

이 문서는 현재 프로젝트의 `Renew` 차트 화면, 리포트 생성 흐름, DICOM 서버 링크 사용법을 다른 사람이 바로 따라 쓸 수 있게 정리한 운영 가이드입니다.

## 1. 주요 화면 링크

- `/upload`
  - 이미지 또는 DICOM 업로드 시작 화면
- `/folder_leader_ver_2`
  - 서버 폴더 기반 DICOM study 목록 화면
- `/chart`
  - 현재 기본 차트 화면. 실제로는 `RenewPage`를 사용함
- `/chart-legacy`
  - 레거시 차트 화면
- `/report/:sessionId`
  - 생성된 웹 리포트 단독 페이지

라우트 정의: [frontend/src/main.tsx](/abs/path/c:/interface/frontend/src/main.tsx:83)

## 2. 로컬 실행

### 백엔드

```bash
cd c:\interface\gpts
python app.py
```

기본 주소:

- `http://localhost:5000`

### 프론트 개발 서버

```bash
cd c:\interface\frontend
npm run dev
```

기본 주소:

- `http://localhost:5173`

## 3. ngrok 사용법

가장 권장하는 방식은 **백엔드 5000 포트만 ngrok으로 공개**하는 방식입니다.

### 권장 방식

1. 백엔드 실행
2. `ngrok http 5000`
3. 브라우저에서 `https://<your-ngrok>.ngrok-free.app` 또는 `https://<your-ngrok>.ngrok-free.dev` 접속

이 방식이 맞는 이유:

- Flask가 프론트 정적 파일과 `/api/*`를 같이 서빙함
- 프론트와 백엔드가 같은 origin을 쓰게 되어 CORS 문제가 줄어듦
- `/api/dicom-server/preview`
- `/api/dicom-server/download`

같은 링크도 그대로 동작함

### 참고

프론트가 개발 서버(`5173`)에서 열릴 때는 내부적으로 개발 편의를 위해 `localhost:5000`으로 fallback 할 수 있습니다.  
ngrok 환경에서는 기본적으로 현재 브라우저 origin을 우선 사용하도록 정리되어 있습니다.

관련 코드:

- [frontend/src/lib/folderLeaderApi.ts](/abs/path/c:/interface/frontend/src/lib/folderLeaderApi.ts:1)
- [frontend/src/features/upload/uploadApi.ts](/abs/path/c:/interface/frontend/src/features/upload/uploadApi.ts:1)
- [frontend/src/lib/webReportApi.ts](/abs/path/c:/interface/frontend/src/lib/webReportApi.ts:29)
- [gpts/config.py](/abs/path/c:/interface/gpts/config.py:27)

## 4. DICOM 서버 폴더

현재 서버 폴더 브라우저는 아래 폴더를 기준으로 DICOM을 읽습니다.

- `C:/interface/case`

정의 위치:

- [gpts/api/dicom_server_browser.py](/abs/path/c:/interface/gpts/api/dicom_server_browser.py:8)

### 제공 API

- `GET /api/dicom-server/studies`
  - study / series 목록 반환
- `GET /api/dicom-server/preview?path=...`
  - 썸네일용 JPEG 프리뷰 반환
- `GET /api/dicom-server/download?path=...`
  - 원본 DICOM 다운로드

관련 코드:

- [gpts/api/dicom_server_browser.py](/abs/path/c:/interface/gpts/api/dicom_server_browser.py:12)
- [frontend/src/lib/folderLeaderApi.ts](/abs/path/c:/interface/frontend/src/lib/folderLeaderApi.ts:113)

## 5. Renew 차트 화면 사용법

현재 기본 차트 화면은 `/chart`이며 실제 컴포넌트는 `RenewPage`입니다.

정의 위치:

- [frontend/src/pages/RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx:1)

### 좌측 레일

- `Studies`
  - DICOM study / series 선택 패널 열기
- `Report`
  - 리포트 작업영역 전환

### General > View

- `Mouse`
  - 일반 선택
- `Pan`
  - 이동
- `WL/WW`
  - 밝기/대비 조정
- `Invert`
  - 반전
- `Magnifier`
  - 확대경
- `Flip`
  - 좌우 반전

### Measure

#### Ruler 메뉴

- `Length`
  - 2점 길이 측정
- `Bidirectional`
  - 2점 기준 width / height 측정
- `Angle`
  - 3점 각도 측정

#### Draw 메뉴

- `Annotation`
  - 클릭 후 텍스트 입력
- `Arrow`
  - 2점 화살표
- `Ellipse`
  - 타원
- `Rectangle`
  - 사각형
- `Circle`
  - 원
- `Freehand ROI`
  - 자유 곡선 ROI
- `Spline ROI`
  - 곡선 ROI
- `Livewire Tool`
  - 클릭 기반 contour ROI

ROI 계열 종료:

- `double-click`
- 또는 `right-click`

#### 기타

- `Eraser`
  - 도형 클릭 시 해당 측정/주석 삭제
- `Delete all measure`
  - 전체 측정/주석 삭제
- `Reset`
  - 측정/주석 포함 화면 상태 초기화

관련 코드:

- [frontend/src/pages/RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx:3019)
- [frontend/src/pages/RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx:3709)
- [frontend/src/components/chart/RenewToolSubmenu.tsx](/abs/path/c:/interface/frontend/src/components/chart/RenewToolSubmenu.tsx:1)

### Output

- `Capture`
  - 현재 파노라마 화면을 캡처
  - 클립보드 복사
  - `Capture Box`에 임시 저장
- `Capture Save`
  - PNG 파일로 저장

### Capture Box

- 기본 상태는 접힘
- 탭 클릭 시 열기 / 닫기
- 최근 캡처 최대 8장 임시 보관
- 각 캡처 개별 삭제 가능
- `Clear`로 전체 삭제 가능

관련 코드:

- [frontend/src/components/chart/OutputCapturePanel.tsx](/abs/path/c:/interface/frontend/src/components/chart/OutputCapturePanel.tsx:18)
- [frontend/src/pages/RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx:2970)

### Task

- `Overlay`
  - AI contour / detection 오버레이 표시
- `Heatmap`
  - 위험도 히트맵 표시

### Dental Chart

- 하단 치아 차트에서 tooth hover HUD 제공
- 우상단 `FDI / Univ` 토글 지원

## 6. Report 사용법

리포트는 두 가지 진입점이 있습니다.

### 좌측 레일 Report

- 리포트 작업영역으로 전환
- session이 없으면 생성 후 진입

### 원형 Report 버튼

- 내부 레이어의 report panel 열기 / 닫기

관련 코드:

- [frontend/src/pages/RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx:3250)
- [frontend/src/pages/RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx:3273)

### 리포트 작업영역에서 가능한 것

- session 상태 확인
- 프리뷰 보기
- `Regenerate Preview`
  - 실제 HTML 리포트 재생성
- `Open Full`
  - 리포트 단독 페이지 열기

API:

- `POST /api/web_report/from-chart`
- `GET /api/web_report/session/:sessionId`
- `POST /api/web_report/session/:sessionId/report/regenerate`
- `POST /api/web_report/session/:sessionId/report/finalize`

프론트 API 래퍼:

- [frontend/src/lib/webReportApi.ts](/abs/path/c:/interface/frontend/src/lib/webReportApi.ts:94)
- [frontend/src/lib/webReportApi.ts](/abs/path/c:/interface/frontend/src/lib/webReportApi.ts:130)
- [frontend/src/lib/webReportApi.ts](/abs/path/c:/interface/frontend/src/lib/webReportApi.ts:137)

## 7. Study 선택 시 동작

`Studies`에서 다른 case / series를 선택하면:

1. 기존 분석 상태 초기화
2. 새 DICOM preview를 뷰어에 반영
3. 새 series 기준으로 분석 재요청
4. 결과 도착 후 overlay / chart / report 대상 갱신

## 8. 환자 이름 표시 관련 주의

DICOM `PatientName`는 백엔드에서 추출합니다.  
이전에는 `PersonName`이 잘못 list로 풀려서 `K,i,m,^,W,o,n...`처럼 보일 수 있었고, 현재는 문자열로 유지하도록 수정되었습니다.

관련 코드:

- [gpts/services/image_loader.py](/abs/path/c:/interface/gpts/services/image_loader.py:11)

## 9. 운영 팁

- ngrok에서는 프론트만 따로 공개하지 말고 가능하면 `5000` 전체를 공개하는 편이 안정적입니다.
- 서버 폴더 목록이 안 보이면 `C:/interface/case` 경로와 파일 확장자 `.dcm`, `.dicom` 여부를 먼저 확인합니다.
- 이미 만들어진 리포트 디자인 변경은 `Refresh`가 아니라 `Regenerate Preview`를 눌러야 반영됩니다.


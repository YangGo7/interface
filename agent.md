# Agent Guide

이 문서는 `c:\interface` 저장소에서 AI 에이전트가 작업할 때 따라야 할 기본 규칙이다.
코드를 수정하기 전에 이 파일을 먼저 읽고, 기존 문서와 현재 워크트리 변경을 확인한다.

## 프로젝트 개요

- 치과 영상 기반 진단/리포트/뷰어 프로젝트다.
- 주요 프론트엔드는 `frontend`의 React + Vite 앱이다.
- 주요 백엔드는 `gpts`의 Flask/Python API다.
- 배포 산출물은 `release/interface-deploy` 아래에 생성된다.
- 빌드/패키징은 루트의 `build_release.ps1`을 기준으로 한다.

## 주요 경로

- `frontend/src/pages`: 화면 단위 React 페이지.
- `frontend/src/components`: 공통 UI 컴포넌트.
- `frontend/src/viewer`: Cornerstone/DICOM 뷰어 관련 코드.
- `frontend/src/features`: 기능별 프론트엔드 모듈.
- `frontend/src/lib`: API 클라이언트, 설정, 공통 유틸리티.
- `gpts/app.py`: Flask 앱 진입점.
- `gpts/api`: 백엔드 API 라우트.
- `gpts/services`: 백엔드 서비스 로직.
- `gpts/utils`: 리포트, 모델 후처리 등 유틸리티.
- `docs/current-build`: 현재 빌드 기준 제품 흐름, 런타임 흐름, 배포 문서.
- `release/interface-deploy`: 배포 패키지 결과물.

## 먼저 읽을 문서

작업 범위에 따라 아래 문서를 먼저 확인한다.

- 전체 흐름: `docs/current-build/README.md`
- 제품/사용자 흐름: `docs/current-build/01_PRODUCT_AND_USER_FLOW.md`
- 런타임/데이터 흐름: `docs/current-build/02_RUNTIME_FLOW_AND_DATA.md`
- 빌드/배포: `docs/current-build/04_BUILD_RELEASE_AND_OPERATIONS.md`
- MPR/3D 렌더링: `docs/current-build/06_DENTAL_MPR_3D_RENDERING.md`
- MPR 툴박스 변경: `docs/current-build/07_MPR_TOOLBOX_CHANGELOG_2026-05-15.md`
- MPR 2D 도구/뷰포트: `docs/current-build/08_MPR_2D_TOOLS_AND_VIEWPORT_2026-05-18.md`

## 개발 명령

프론트엔드:

```powershell
cd frontend
npm.cmd run dev
npm.cmd run build
```

백엔드:

```powershell
cd gpts
python app.py
```

릴리스 빌드:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_release.ps1
powershell -ExecutionPolicy Bypass -File .\build_release.ps1 -Zip
```

## 작업 규칙

- 수정 전 `git status --short`로 사용자 변경을 확인한다.
- 사용자가 만든 변경을 되돌리지 않는다.
- 이미 수정된 파일을 건드릴 때는 현재 내용을 읽고 그 위에 맞춰 수정한다.
- 새 기능보다 기존 구조와 네이밍을 우선한다.
- 대규모 리팩터링은 사용자가 요청하지 않으면 피한다.
- 프론트엔드 변경은 실제 사용자 흐름 기준으로 확인한다.
- 백엔드 변경은 API 응답 형식과 기존 파일 저장 위치를 유지한다.
- 배포 폴더의 빌드 산출물은 의도한 릴리스 작업이 아니면 직접 수정하지 않는다.
- 대용량 데이터, 모델 가중치, ZIP 파일은 새로 커밋 대상에 넣지 않는다.
- 하네스, 테스트 UI, 운영 UI, 문서, 로그 메시지에 이모지나 장식용 아이콘을 쓰지 않는다.
- 변경이 있으면 코드만 고치지 말고 관련 문서도 함께 갱신한다.

## 문서화 기준

변경마다 영향 범위에 맞는 문서를 작성하거나 갱신한다.

- 사용자 흐름이 바뀌면 `docs/current-build/01_PRODUCT_AND_USER_FLOW.md`를 갱신한다.
- 런타임 데이터 흐름, API 연결, 저장 위치가 바뀌면 `docs/current-build/02_RUNTIME_FLOW_AND_DATA.md`를 갱신한다.
- UI/UX 의도나 화면 구조가 바뀌면 `docs/current-build/03_DESIGN_RATIONALE.md`를 갱신한다.
- 빌드, 배포, 실행 방법이 바뀌면 `docs/current-build/04_BUILD_RELEASE_AND_OPERATIONS.md`를 갱신한다.
- 구현 판단, 트레이드오프, 유지보수 기준이 바뀌면 `docs/current-build/05_IMPLEMENTATION_RATIONALE.md`를 갱신한다.
- MPR/3D 렌더링, 2D 도구, 툴박스가 바뀌면 해당 MPR 문서를 갱신한다.
- 작은 수정이라도 문서 변경이 불필요하다고 판단한 경우, 최종 답변에 그 이유를 적는다.
- 날짜가 들어가는 변경 로그는 실제 작업일 기준 `YYYY-MM-DD` 형식을 사용한다.

## 프론트엔드 기준

- React + TypeScript + Vite 기준으로 작업한다.
- `lucide-react` 아이콘이 이미 있으므로 버튼/툴바 아이콘은 우선 사용한다.
- Cornerstone 관련 변경은 `frontend/src/viewer`와 `frontend/src/features/mpr`의 기존 패턴을 따른다.
- 화면 라우팅은 `frontend/src/App.tsx`와 페이지 컴포넌트를 함께 확인한다.
- 설정/서버 주소 관련 변경은 `frontend/src/lib/appSettings.ts`와 관련 API 클라이언트를 확인한다.
- UI 텍스트가 버튼이나 패널 안에서 넘치지 않도록 모바일/데스크톱 폭을 고려한다.

## 백엔드 기준

- Flask 라우트는 `gpts/api`의 기존 라우트 스타일을 따른다.
- DICOM 검색/색인 관련 변경은 `gpts/api/dicom_index.py`, `gpts/services/dicom_index_service.py`, `gpts/services/dicom_discovery_jobs.py`를 함께 확인한다.
- 리포트 관련 변경은 `gpts/api/web_report.py`, `gpts/utils/report_v3_viewer.py`, 관련 service 파일을 함께 확인한다.
- 경로 처리는 문자열 연결보다 `pathlib` 또는 기존 헬퍼를 우선한다.
- 외부 모델/가중치 경로는 하드코딩하지 말고 기존 설정 구조를 확인한다.

## 검증 기준

작업 후 가능한 범위에서 아래를 실행한다.

```powershell
cd frontend
npm.cmd run build
```

백엔드만 바꾼 경우에는 최소한 해당 Python 파일의 import/문법 오류를 확인한다.
테스트 환경을 고려해 로컬에서 재현 가능한 테스트를 우선 실행한다.
외부 장비, GPU, 모델 가중치, 병원 DICOM 서버, 네트워크 접근이 필요한 테스트는 가능한 대체 검증을 수행하고 제한 사항을 기록한다.
하네스나 테스트 페이지가 있는 기능은 실제 테스트 경로로 실행해 사용자가 밟는 흐름을 확인한다.
환경 의존성이 커서 실행하지 못한 검증은 최종 답변에 명확히 남긴다.

## 커뮤니케이션 규칙

- 작업 결과는 변경 파일, 검증 결과, 남은 위험 순서로 짧게 보고한다.
- 실패한 명령은 숨기지 말고 원인과 다음 조치를 함께 적는다.
- 사용자가 한국어로 요청하면 한국어로 답한다.
- 추측한 내용은 사실처럼 말하지 말고 추정이라고 표시한다.
- 답변, 문서, 커밋 메시지에 이모지를 쓰지 않는다.

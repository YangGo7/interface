# Web Report Architecture

## 목적
- GPTs용 `v2` 리포트 세션 흐름과 별도로, 웹 제품 전용 리포트 세션 아키텍처를 정의한다.
- 웹에서는 파일 링크 중심이 아니라 `session_id` 중심으로 차트, 리포트, 의사 수정, 버전 관리를 일관되게 처리한다.
- AI 원본 결과와 의사 수정값을 분리해 추적 가능성과 재생성 안정성을 확보한다.

## 범위
- 업로드 이후 분석, 차트 진입, 의사 수정, draft 리포트 재생성, final PDF 확정까지의 흐름
- 백엔드 모듈 분리 전략
- 저장소 구조
- DB 스키마 초안
- API 초안
- 프론트 페이지 흐름

## 핵심 원칙
- `v2`는 유지한다. GPTs 연동 흐름은 건드리지 않는다.
- 웹 전용 기능은 `web_report` 네임스페이스로 분리한다.
- 외부 노출 URL은 파일 경로가 아니라 `session_id` 기반으로 고정한다.
- `ai_result`는 immutable snapshot으로 유지한다.
- 의사 수정은 `doctor_overrides`에만 저장한다.
- 화면과 리포트는 항상 `effective_result = ai_result + doctor_overrides`를 기준으로 렌더한다.
- 리포트는 draft와 final을 버전으로 관리한다.

## 네이밍 및 모듈 분리

### 백엔드
- `gpts/api/web_report.py`
- `gpts/services/web_report_session_service.py`
- `gpts/services/web_report_merge_service.py`
- `gpts/services/web_report_report_service.py`
- `gpts/utils/web_report_generator.py`
- `gpts/data/web_report.db`

### 프론트
- `frontend/src/pages/WebChartPage.tsx`
- `frontend/src/pages/WebReportPage.tsx`
- `frontend/src/lib/webReportApi.ts`

### 파일 저장 경로
- `gpts/runs/web_report/<session_id>/source/`
- `gpts/runs/web_report/<session_id>/inference/`
- `gpts/runs/web_report/<session_id>/reports/`
- `gpts/runs/web_report/<session_id>/final/`

## 세션 중심 데이터 구조

### source
- 원본 업로드 파일
- 미리보기 PNG 또는 thumbnail
- overlay PNG
- inference 산출물 경로

### ai_result
- AI 원본 분석 결과
- 최초 분석 후 저장
- 이후 수정 금지

### doctor_overrides
- 의사 수정값
- 치아별 finding override
- 치아별 note
- 리포트 전체 note

### effective_result
- `ai_result + doctor_overrides`
- 차트 표시값
- 리포트 생성 입력값

### report
- 현재 draft/final 상태
- HTML/PDF 경로
- version
- last_generated_at
- finalized_at

## 저장소 전략

### 권장 방식
- 메타데이터와 override는 SQLite 저장
- 대용량 산출물은 파일시스템 저장

### 이유
- 서버 재시작 후에도 세션 유지 가능
- 차트 재접속 및 리포트 재생성 가능
- 향후 검색, 통계, 변경 이력 조회에 유리

## SQLite 스키마 초안

### `web_report_sessions`
- `id TEXT PRIMARY KEY`
- `status TEXT NOT NULL`
- `language TEXT NOT NULL DEFAULT 'English'`
- `created_at TEXT NOT NULL`
- `updated_at TEXT NOT NULL`
- `finalized_at TEXT NULL`
- `is_finalized INTEGER NOT NULL DEFAULT 0`
- `current_report_version INTEGER NOT NULL DEFAULT 0`

### `web_report_assets`
- `session_id TEXT PRIMARY KEY`
- `source_path TEXT`
- `preview_path TEXT`
- `overlay_path TEXT`
- `inference_dir TEXT`
- `reports_dir TEXT`
- `final_dir TEXT`

### `web_report_ai_results`
- `session_id TEXT PRIMARY KEY`
- `result_json TEXT NOT NULL`
- `created_at TEXT NOT NULL`

### `web_report_doctor_overrides`
- `session_id TEXT PRIMARY KEY`
- `override_json TEXT NOT NULL`
- `updated_at TEXT NOT NULL`
- `updated_by TEXT NULL`

### `web_report_report_versions`
- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `session_id TEXT NOT NULL`
- `version INTEGER NOT NULL`
- `status TEXT NOT NULL`
- `html_path TEXT`
- `pdf_path TEXT`
- `snapshot_json TEXT NOT NULL`
- `created_at TEXT NOT NULL`

## JSON 예시

```json
{
  "session_id": "9f8c...",
  "status": "completed",
  "source": {
    "source_path": "runs/web_report/9f8c/source/original.dcm",
    "preview_path": "runs/web_report/9f8c/source/original.png",
    "overlay_path": "runs/web_report/9f8c/inference/overlay.png"
  },
  "ai_result": {
    "teeth": [],
    "missing_teeth": [],
    "caries": [],
    "periapical": []
  },
  "doctor_overrides": {
    "teeth": {
      "16": {
        "caries": false,
        "periapical": true,
        "bone_loss_level": 2,
        "note": "재평가 필요"
      }
    },
    "report_note": "우측 상악 우선 치료 권고"
  },
  "effective_result": {},
  "report": {
    "version": 3,
    "status": "draft",
    "html_path": "runs/web_report/9f8c/reports/report_v3.html",
    "pdf_path": "runs/web_report/9f8c/reports/report_v3.pdf"
  }
}
```

## 웹 플로우

### 1. 업로드
- 프론트가 `POST /api/web_report/session` 호출
- 서버가 `session_id` 생성
- 프론트가 `POST /api/web_report/session/:sessionId/upload` 호출

### 2. 분석
- 업로드 완료 후 서버가 비동기 분석 시작
- 분석 완료 시 `ai_result`, `source`, `report draft` 저장

### 3. 차트 진입
- 프론트는 `/chart/:sessionId`로 이동
- 차트는 `GET /api/web_report/session/:sessionId`로 세션 상태 조회
- 분석이 끝나면 `effective_result` 기준으로 차트를 렌더

### 4. 의사 수정
- 치아 라벨 수정 시 debounce autosave
- `PATCH /api/web_report/session/:sessionId/overrides`
- 저장 성공 후 화면은 최신 `effective_result` 재계산

### 5. draft 리포트
- 우측 상단에 `Report Draft` 버튼
- `Regenerate Report` 버튼으로 최신 override 반영
- 리포트 페이지는 `/report/:sessionId`
- 서버는 항상 해당 세션의 최신 draft 또는 final HTML 반환

### 6. 최종 확정
- `Finalize Report` 버튼 클릭
- 현재 `effective_result` snapshot을 final version으로 고정
- PDF를 final 문서로 생성

## 프론트 UX 요구사항
- `/chart/:sessionId`
- `/report/:sessionId`
- 차트 화면 상단에 세션 상태 표시
- 수정 중 표시와 autosave 상태 표시
- `Report Draft`
- `Regenerate Report`
- `Finalize Report`
- final 상태에서는 수정 잠금 또는 경고 표시

## API 초안

### `POST /api/web_report/session`
- 설명: 세션 생성
- 응답:

```json
{
  "success": true,
  "session_id": "..."
}
```

### `POST /api/web_report/session/:sessionId/upload`
- 설명: 파일 업로드 및 분석 시작
- 응답:

```json
{
  "success": true,
  "session_id": "...",
  "status": "processing"
}
```

### `GET /api/web_report/session/:sessionId`
- 설명: 세션 전체 상태 조회
- 응답:

```json
{
  "success": true,
  "session": {
    "status": "completed",
    "source": {},
    "ai_result": {},
    "doctor_overrides": {},
    "effective_result": {},
    "report": {}
  }
}
```

### `PATCH /api/web_report/session/:sessionId/overrides`
- 설명: 의사 수정 autosave
- 요청:

```json
{
  "tooth_overrides": {
    "16": {
      "caries": false,
      "periapical": true,
      "note": "재평가 필요"
    }
  },
  "report_note": "상악 우선 치료"
}
```

### `POST /api/web_report/session/:sessionId/report/regenerate`
- 설명: 현재 override 기준 draft 리포트 재생성

### `GET /api/web_report/session/:sessionId/report`
- 설명: 최신 HTML 리포트 반환

### `GET /api/web_report/session/:sessionId/report/pdf`
- 설명: 최신 PDF 반환

### `POST /api/web_report/session/:sessionId/report/finalize`
- 설명: 현재 effective_result를 final snapshot으로 고정하고 final PDF 생성

### `GET /api/web_report/session/:sessionId/report/versions`
- 설명: draft/final 버전 목록 조회

## effective_result 머지 규칙
- 기본값은 `ai_result`
- 동일 필드에 override가 있으면 override 우선
- 삭제 의도는 `null`이 아니라 명시적 boolean 또는 enum으로 표현
- 치아 note는 override에만 존재
- 리포트 summary는 merged result로 재계산

## 리포트 URL 정책

### 현재 문제
- 파일 경로 기반 URL은 버전 교체 시 불안정하다
- 프론트가 실제 파일 위치를 직접 알게 된다

### 목표
- 외부 노출 URL은 세션 기반으로 고정
- 실제 파일 경로는 서버 내부에서만 관리

### 예시
- 차트: `/chart/:sessionId`
- 리포트 페이지: `/report/:sessionId`
- HTML API: `/api/web_report/session/:sessionId/report`
- PDF API: `/api/web_report/session/:sessionId/report/pdf`

## `v2`와의 관계
- `routes_v2.py`는 그대로 유지
- GPTs 업로드/세션 흐름은 유지
- 웹 전용 기능은 `web_report.py`에서 별도 구현
- `report_v3.py`도 직접 재사용하지 말고 `web_report_generator.py`로 분기하는 것을 권장

## 구현 순서

### Phase 1
- `web_report` 라우트 파일 생성
- SQLite 초기화 모듈 생성
- 세션 생성/업로드/상태 조회 API 생성
- 세션 폴더 구조 생성

### Phase 2
- 분석 완료 후 `ai_result` 저장
- draft 리포트 생성
- 차트 페이지를 `session_id` 기반으로 연결

### Phase 3
- 의사 override autosave API
- `effective_result` merge service
- 차트와 리포트에 merged data 적용

### Phase 4
- `Regenerate Report`
- `Finalize Report`
- 버전 관리

## MVP 정의
- 세션 생성
- 업로드 및 분석
- 차트 페이지 진입
- 치아별 구조화 수정
- autosave
- draft 리포트 재생성
- final PDF 확정

## 비범위
- 다중 사용자 협업 잠금
- 서명 워크플로우
- 역할 기반 권한 관리
- 외부 EMR 연동
- 법적 전자서명

## 리스크
- 현재 메모리 세션 구조와 혼용 시 혼선 가능
- `report_v3.py`를 바로 공유하면 GPTs와 웹 요구사항이 다시 얽힐 수 있음
- autosave 충돌 방지를 위해 프론트 debounce와 서버 last-write 정책이 필요
- final 이후 수정 정책을 명확히 정해야 함

## 권장 결정
- `v2`는 유지
- 웹은 `web_report` 별도 구현
- DB는 SQLite 사용
- 리포트는 세션 URL로만 접근
- AI 원본과 의사 수정본을 분리 저장
- final PDF는 snapshot 기반으로 고정

## 다음 단계
- DB 스키마 SQL 작성
- `web_report.py` API skeleton 생성
- 프론트 `WebChartPage`와 `WebReportPage` 라우트 추가
- `web_report_generator.py` 초안 분리

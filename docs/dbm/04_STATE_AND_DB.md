# 상태 다이어그램과 DB 구조

이 문서는 화면 상태, 세션 상태, 그리고 DB 구조를 함께 보여줍니다.

## 1. 검사 찾기 화면 상태

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Loading: 첫 진입 / Search / Refresh
    Loading --> Ready: 목록 수신 성공
    Loading --> Error: API 실패
    Ready --> SelectingStudy: Study 선택
    Ready --> SelectingImage: Image 선택
    SelectingStudy --> Opening: Join 또는 더블클릭
    SelectingImage --> Opening: Open 또는 더블클릭
    Opening --> Ready: 실패
    Opening --> [*]: /renew 이동
    Error --> Loading: 재시도
```

## 2. 리포트 세션 상태

`web_report_sessions.status` 기준으로 보면 아래처럼 움직입니다.

```mermaid
stateDiagram-v2
    [*] --> waiting
    waiting --> processing: 분석 시작
    processing --> completed: 초안/리포트 생성 성공
    processing --> failed: 분석 또는 생성 실패
    completed --> completed: override 저장
    completed --> completed: 버전 재생성
    completed --> finalized: final report 저장
    failed --> processing: 재시도
```

주의:

- DB의 `status`는 `waiting`, `processing`, `completed`, `failed` 흐름을 관리합니다.
- `is_finalized`와 `current_report_version`은 별도 컬럼으로 최종 상태와 현재 버전을 관리합니다.

## 3. 캡처 상태

```mermaid
stateDiagram-v2
    [*] --> Empty
    Empty --> Captured: 캡처 생성
    Captured --> Selected: 보고서용 선택
    Selected --> Synced: override 동기화
    Selected --> Removed: 선택 해제 또는 삭제
    Synced --> Selected: 메모 수정
    Removed --> Captured: 다시 남아 있음
```

## 4. 리포트 버전 상태

```mermaid
stateDiagram-v2
    [*] --> DraftEditing
    DraftEditing --> DraftVersionSaved: regenerate report
    DraftVersionSaved --> DraftEditing: 추가 수정
    DraftVersionSaved --> Restored: rollback
    Restored --> DraftEditing: 다시 수정
    DraftVersionSaved --> FinalVersionSaved: finalize
    FinalVersionSaved --> [*]
```

## 5. DB ER 다이어그램

`web_report.db` 안의 대표 테이블 구조입니다.

```mermaid
erDiagram
    WEB_REPORT_SESSIONS ||--|| WEB_REPORT_ASSETS : has
    WEB_REPORT_SESSIONS ||--|| WEB_REPORT_AI_RESULTS : has
    WEB_REPORT_SESSIONS ||--|| WEB_REPORT_DOCTOR_OVERRIDES : has
    WEB_REPORT_SESSIONS ||--o{ WEB_REPORT_REPORT_VERSIONS : has

    WEB_REPORT_SESSIONS {
        text id PK
        text status
        text error
        text language
        text patient_name
        text created_at
        text updated_at
        text finalized_at
        int is_finalized
        int current_report_version
    }

    WEB_REPORT_ASSETS {
        text session_id PK
        text source_path
        text preview_path
        text overlay_path
        text bl_viz_path
        text inference_dir
        text reports_dir
        text final_dir
    }

    WEB_REPORT_AI_RESULTS {
        text session_id PK
        text result_json
        text created_at
    }

    WEB_REPORT_DOCTOR_OVERRIDES {
        text session_id PK
        text override_json
        text updated_at
        text updated_by
    }

    WEB_REPORT_REPORT_VERSIONS {
        int id PK
        text session_id FK
        int version
        text status
        text html_path
        text pdf_path
        text snapshot_json
        text created_at
    }
```

## 6. DB 저장 흐름

```mermaid
flowchart TD
    A[세션 생성] --> B[web_report_sessions INSERT]
    A --> C[doctor_overrides 기본값 INSERT]
    D[차트 결과 수신] --> E[web_report_ai_results UPSERT]
    F[파일 경로 결정] --> G[web_report_assets UPSERT]
    H[리포트 생성] --> I[web_report_report_versions INSERT]
    I --> J[current_report_version UPDATE]
    K[사용자 수정] --> L[web_report_doctor_overrides UPSERT]
    M[버전 복원] --> N[snapshot 조회]
    N --> O[override 재구성]
    O --> L
```

## 7. 스냅샷 복원 원리

버전 복원은 단순히 HTML 파일만 바꾸는 것이 아니라, `snapshot_json`을 읽어서 다시 `doctor_overrides`로 되돌리는 방식입니다.

```mermaid
flowchart LR
    A[선택한 version] --> B[report_versions.snapshot_json 조회]
    B --> C[build_overrides_from_snapshot]
    C --> D[save_overrides]
    D --> E[current_report_version 갱신]
    E --> F[화면 재렌더링]
```

관련 파일:

- [web_report_session_service.py](/abs/path/c:/interface/gpts/services/web_report_session_service.py)
- [web_report_merge_service.py](/abs/path/c:/interface/gpts/services/web_report_merge_service.py)
- [web_report.py](/abs/path/c:/interface/gpts/api/web_report.py)
- [WebReportPage.tsx](/abs/path/c:/interface/frontend/src/pages/WebReportPage.tsx)


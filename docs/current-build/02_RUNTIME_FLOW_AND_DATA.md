# 현재 빌드 런타임 흐름과 데이터 흐름

## 1. 프론트엔드에서 백엔드로 가는 전체 흐름

```mermaid
flowchart TD
    A[FolderLeaderVer2Page] --> B[folderLeaderApi.ts]
    B --> C[/api/dicom-server/studies]
    C --> D[dicom_server_browser.py]
    D --> E[(서버 DICOM 폴더)]
    E --> D
    D --> C
    C --> B
    B --> A

    A --> F[RenewPage]
    F --> G[webReportApi.ts]
    G --> H[/api/web_report/from-chart]
    H --> I[web_report.py]
    I --> J[WebReportSessionService]
    I --> K[WebReportMergeService]
    I --> L[WebReportReportService]
    J --> M[(web_report.db)]
    L --> N[(runs/web_report/session_id)]
```

## 2. 검사 선택 화면 런타임

### 2.1 서버 목록 로딩

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Page as FolderLeaderVer2Page
    participant Api as folderLeaderApi.ts
    participant Server as dicom_server_browser.py
    participant FS as 서버 폴더

    User->>Page: Search 또는 첫 진입
    Page->>Page: loadStudies()
    Page->>Api: fetchServerFolderIndex()
    Api->>Server: GET /api/dicom-server/studies
    Server->>FS: DICOM 파일과 이미지 파일 스캔
    FS-->>Server: 파일 목록
    Server-->>Api: studies + images JSON
    Api-->>Page: 응답 반환
    Page->>Page: 상태 저장과 화면 필터
```

### 2.2 `Upload Image` 흐름

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Page as FolderLeaderVer2Page
    participant Renew as RenewPage

    User->>Page: Upload Image 클릭
    Page->>Page: 파일 선택창 열기
    User->>Page: 이미지 선택
    Page->>Page: handleLocalFilePick(file)
    Page->>Renew: navigate('/renew', state)
    Renew->>Renew: 전달받은 파일 표시
```

### 2.3 `Upload Folder` 흐름

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Page as FolderLeaderVer2Page
    participant Builder as buildDicomFolderStudies
    participant Renew as RenewPage

    User->>Page: Upload Folder 클릭
    Page->>Page: 폴더 선택창 열기
    User->>Page: 폴더 선택
    Page->>Page: handleLocalFolderPick(files)
    Page->>Builder: buildDicomFolderStudies(files)
    Builder-->>Page: FolderStudy[]
    Page->>Renew: navigate('/renew', state)
    Renew->>Renew: 첫 series 표시
```

## 3. 뷰어 화면 런타임

### 3.1 뷰어 진입 후 작업 흐름

```mermaid
flowchart LR
    A[RenewPage 진입] --> B[현재 study 또는 image 로딩]
    B --> C[Studies 패널 구성]
    B --> D[캡처 패널 준비]
    B --> E[리포트 패널 준비]
    C --> F[다른 study/image 전환]
    D --> G[캡처 생성]
    E --> H[리포트 세션 생성]
```

### 3.2 캡처와 리포트 동기화

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Renew as RenewPage
    participant Api as webReportApi.ts
    participant Server as web_report.py
    participant DB as web_report.db

    User->>Renew: Capture 클릭
    Renew->>Renew: 캡처 데이터 생성
    Renew->>Renew: capturedOutputs 상태 저장
    Renew->>Renew: 캡처 박스 자동 열기
    User->>Renew: 캡처 선택
    Renew->>Api: patchWebReportOverrides(attached_captures)
    Api->>Server: PATCH /api/web_report/session/:id/overrides
    Server->>DB: doctor_overrides 저장
    Server-->>Api: effective_result 반환
    Api-->>Renew: 상태 반영
```

### 3.3 뷰어 안 `Studies` 전환

```mermaid
flowchart TD
    A[RenewPage] --> B[activeFolderStudies]
    A --> C[serverStudies]
    A --> D[serverImages]
    B --> E[combinedStudies]
    C --> E
    D --> E
    E --> F[사용자 선택]
    F --> G{image 인가}
    G -- 예 --> H[이미지 URL 로딩]
    G -- 아니오 --> I[materializeServerStudy]
    I --> J[series 상태 갱신]
    H --> K[뷰어 업데이트]
    J --> K
```

## 4. 리포트 런타임

### 4.1 차트에서 리포트 세션을 만들 때

```mermaid
sequenceDiagram
    participant Renew as RenewPage
    participant Api as webReportApi.ts
    participant Server as web_report.py
    participant Session as WebReportSessionService
    participant Merge as WebReportMergeService
    participant Report as WebReportReportService
    participant DB as web_report.db

    Renew->>Api: createWebReportFromChart(payload)
    Api->>Server: POST /api/web_report/from-chart
    Server->>Session: create_session()
    Server->>Session: set_assets()
    Server->>Session: save_ai_result()
    Server->>Merge: build_effective_result()
    Server->>Report: generate_report()
    Server->>Session: create_report_version()
    Session->>DB: 세션과 버전 저장
    Server-->>Api: session_id 반환
```

### 4.2 리포트 화면 편집 흐름

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Page as WebReportPage
    participant Api as webReportApi.ts
    participant Server as web_report.py
    participant DB as web_report.db

    User->>Page: 치아 정보 또는 note 수정
    Page->>Api: patchWebReportOverrides()
    Api->>Server: PATCH /overrides
    Server->>DB: override_json 저장
    Server-->>Api: doctor_overrides + effective_result
    Api-->>Page: 폼과 프리뷰 갱신
```

### 4.3 버전 복원 흐름

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Page as WebReportPage
    participant Api as webReportApi.ts
    participant Server as web_report.py
    participant Merge as WebReportMergeService
    participant Session as WebReportSessionService
    participant DB as web_report.db

    User->>Page: 과거 버전 선택 후 복원
    Page->>Api: rollbackWebReportVersion(sessionId, version)
    Api->>Server: POST /report/rollback
    Server->>Session: get_report_version()
    Session->>DB: snapshot_json 조회
    Server->>Merge: build_overrides_from_snapshot()
    Merge-->>Server: restored overrides
    Server->>Session: save_overrides()
    Server->>Session: current_report_version 갱신
    Session->>DB: 세션 갱신
    Server-->>Api: restored_version 반환
    Api-->>Page: 화면 재구성
```

## 5. 데이터 저장 위치

### 5.1 DB 저장

`web_report.db`는 아래 정보를 저장한다.

- 세션 기본 정보
- 자산 경로
- AI 결과
- 의사 override
- 리포트 버전 스냅샷

### 5.2 파일 저장

`runs/web_report/<session_id>`는 아래 구조로 저장한다.

- `source`
- `inference`
- `reports`
- `final`

## 6. 이 문서에서 확인해야 하는 코드

- [FolderLeaderVer2Page.tsx](/abs/path/c:/interface/frontend/src/pages/FolderLeaderVer2Page.tsx)
- [dicomFolderStudies.ts](/abs/path/c:/interface/frontend/src/features/upload/dicomFolderStudies.ts)
- [folderLeaderApi.ts](/abs/path/c:/interface/frontend/src/lib/folderLeaderApi.ts)
- [RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx)
- [WebReportPage.tsx](/abs/path/c:/interface/frontend/src/pages/WebReportPage.tsx)
- [webReportApi.ts](/abs/path/c:/interface/frontend/src/lib/webReportApi.ts)
- [dicom_server_browser.py](/abs/path/c:/interface/gpts/api/dicom_server_browser.py)
- [web_report.py](/abs/path/c:/interface/gpts/api/web_report.py)


# 전체 워크플로우

이 문서는 사용자가 프로그램에 들어와서 검사 선택, 뷰어 확인, 캡처, 리포트 저장까지 가는 전체 흐름을 보여줍니다.

## 1. 사용자 기준 전체 흐름

```mermaid
flowchart LR
    A[사용자] --> B[FolderLeaderVer2Page]
    B --> C[Upload Image]
    B --> D[Upload Folder]
    B --> E[Studies 검색/선택]
    C --> F[RenewPage]
    D --> F
    E --> F
    F --> G[검사 보기]
    F --> H[캡처]
    F --> I[리포트 시작]
    H --> I
    I --> J[WebReportPage]
    J --> K[버전 저장]
    J --> L[버전 복원]
    J --> M[최종 리포트]
```

## 2. 프론트엔드-백엔드 전체 흐름

```mermaid
flowchart TD
    U[사용자 입력] --> FE1[FolderLeaderVer2Page]
    FE1 --> API1[/api/dicom-server/studies]
    API1 --> FS[(서버 폴더)]
    FS --> API1
    API1 --> FE1

    FE1 --> FE2[RenewPage]
    FE2 --> API2[/api/web_report/from-chart]
    API2 --> SVC1[WebReportSessionService]
    API2 --> SVC2[WebReportMergeService]
    API2 --> SVC3[WebReportReportService]
    SVC1 --> DB[(web_report.db)]
    SVC3 --> RUNS[(runs/web_report/session_id)]

    FE2 --> FE3[WebReportPage]
    FE3 --> API3[/api/web_report/session/:id]
    API3 --> DB
    DB --> API3
    API3 --> FE3
```

## 3. 검사 탐색에서 리포트까지

```mermaid
flowchart TD
    A[검색 조건 입력] --> B[Search]
    B --> C[서버 폴더 스캔]
    C --> D[Study/Image 목록 생성]
    D --> E[목록에서 항목 선택]
    E --> F{무엇을 여는가?}
    F -- Image --> G[이미지 바로 열기]
    F -- Study --> H[Study materialize]
    G --> I[RenewPage 진입]
    H --> I
    I --> J[캡처/메모/검토]
    J --> K[리포트 세션 생성]
    K --> L[WebReportPage 편집]
```

## 4. 백엔드 리포트 처리 파이프라인

```mermaid
flowchart LR
    A[from-chart 요청] --> B[세션 생성]
    B --> C[소스/오버레이 자산 복사]
    C --> D[AI result 저장]
    D --> E[override merge]
    E --> F[HTML/PDF 생성]
    F --> G[report version 저장]
    G --> H[session status completed]
```

## 5. 물리 저장 위치 흐름

```mermaid
flowchart TD
    A[소스 이미지 또는 DICOM] --> B[temp 또는 source 폴더]
    B --> C[runs/web_report/session_id/source]
    C --> D[runs/web_report/session_id/inference]
    D --> E[runs/web_report/session_id/reports]
    E --> F[web_report.db에 메타데이터 저장]
```

관련 파일:

- [FolderLeaderVer2Page.tsx](/abs/path/c:/interface/frontend/src/pages/FolderLeaderVer2Page.tsx)
- [RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx)
- [WebReportPage.tsx](/abs/path/c:/interface/frontend/src/pages/WebReportPage.tsx)
- [dicom_server_browser.py](/abs/path/c:/interface/gpts/api/dicom_server_browser.py)
- [web_report.py](/abs/path/c:/interface/gpts/api/web_report.py)


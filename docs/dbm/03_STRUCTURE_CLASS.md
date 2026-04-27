# 구조도와 클래스/모듈 다이어그램

이 문서는 프론트엔드, 백엔드, 서비스 계층이 어떻게 나뉘는지 보여줍니다.

## 1. 상위 구조도

```mermaid
flowchart TD
    subgraph Frontend
        A[FolderLeaderVer2Page]
        B[RenewPage]
        C[WebReportPage]
        D[folderLeaderApi.ts]
        E[webReportApi.ts]
    end

    subgraph Backend
        F[dicom_server_browser.py]
        G[web_report.py]
        H[WebReportSessionService]
        I[WebReportMergeService]
        J[WebReportReportService]
    end

    subgraph Storage
        K[(web_report.db)]
        L[(서버 DICOM 폴더)]
        M[(runs/web_report)]
    end

    A --> D
    D --> F
    F --> L

    B --> E
    C --> E
    E --> G
    G --> H
    G --> I
    G --> J
    H --> K
    J --> M
```

## 2. 프론트엔드 모듈 클래스 다이어그램

```mermaid
classDiagram
    class FolderLeaderVer2Page {
        +loadStudies()
        +handleLocalFilePick(file)
        +handleLocalFolderPick(files)
        +openStudyEntry(study, seriesId)
        +openImage(image)
    }

    class RenewPage {
        +handleOpenStudies()
        +toggleReportCaptureSelection(captureId)
        +updateReportCaptureNote(captureId, note)
    }

    class WebReportPage {
        +saveDraft()
        +restoreVersion()
        +finalizeReport()
    }

    class folderLeaderApi {
        +fetchServerFolderIndex()
        +materializeServerStudy()
        +resolveServerAssetUrl()
    }

    class webReportApi {
        +createWebReportFromChart()
        +fetchWebReportSession()
        +patchWebReportOverrides()
        +rollbackWebReportVersion()
        +finalizeWebReport()
    }

    FolderLeaderVer2Page --> folderLeaderApi
    RenewPage --> webReportApi
    WebReportPage --> webReportApi
```

## 3. 백엔드 서비스 클래스 다이어그램

```mermaid
classDiagram
    class web_report_api {
        +create_session()
        +create_from_chart()
        +patch_overrides()
        +regenerate_report()
        +rollback_report()
        +finalize_report()
    }

    class WebReportSessionService {
        +create_session()
        +set_status()
        +set_assets()
        +save_ai_result()
        +save_overrides()
        +create_report_version()
        +get_session()
        +get_report_version()
        +list_report_versions()
    }

    class WebReportMergeService {
        +build_effective_result()
        +build_overrides_from_snapshot()
    }

    class WebReportReportService {
        +generate_report()
    }

    class dicom_browser_api {
        +list_studies()
        +get_root_path()
        +update_root_path()
        +pick_root_path()
    }

    web_report_api --> WebReportSessionService
    web_report_api --> WebReportMergeService
    web_report_api --> WebReportReportService
```

## 4. 리포트 생성 책임 분리

```mermaid
flowchart LR
    A[web_report.py] --> B[세션 생성/조회]
    A --> C[override merge]
    A --> D[HTML/PDF 생성]
    B --> E[WebReportSessionService]
    C --> F[WebReportMergeService]
    D --> G[WebReportReportService]
```

## 5. 파일별 역할 요약

| 파일 | 역할 |
| --- | --- |
| `FolderLeaderVer2Page.tsx` | 검색, 업로드, 목록 선택 |
| `folderLeaderApi.ts` | 서버 폴더 API 호출 |
| `RenewPage.tsx` | 뷰어, 캡처, 리포트 진입 |
| `webReportApi.ts` | 리포트 API 호출 |
| `WebReportPage.tsx` | 리포트 편집/저장/복원 |
| `dicom_server_browser.py` | 서버 폴더 스캔과 목록 생성 |
| `web_report.py` | 리포트 API 엔드포인트 |
| `WebReportSessionService` | SQLite 저장/조회 |
| `WebReportMergeService` | AI 결과와 의사 수정치 병합 |
| `WebReportReportService` | 문서 파일 생성 |


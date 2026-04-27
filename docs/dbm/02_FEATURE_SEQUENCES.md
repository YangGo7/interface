# 기능별 시퀀스 다이어그램

이 문서는 실제 기능이 어떤 순서로 호출되는지 보여줍니다.

## 1. `Studies` 검색

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Page as FolderLeaderVer2Page
    participant Api as folderLeaderApi.ts
    participant Server as dicom_server_browser.py
    participant FS as 서버 폴더

    User->>Page: Search 클릭
    Page->>Api: fetchServerFolderIndex()
    Api->>Server: GET /api/dicom-server/studies
    Server->>FS: DICOM/이미지 파일 스캔
    FS-->>Server: 파일 목록
    Server-->>Api: studies + images JSON
    Api-->>Page: 목록 반환
    Page->>Page: 상태 저장 + 화면 필터링
    Page-->>User: 결과 목록 표시
```

## 2. `Upload Image`

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Page as FolderLeaderVer2Page
    participant Renew as RenewPage

    User->>Page: Upload Image 클릭
    Page->>Page: 파일 선택창 열기
    User->>Page: 이미지 파일 선택
    Page->>Page: handleLocalFilePick(file)
    Page->>Renew: navigate('/renew', state)
    Renew->>Renew: 전달받은 파일 표시
```

## 3. `Upload Folder`

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
    Page->>Builder: DICOM 파일을 Study/Series로 묶기
    Builder-->>Page: FolderStudy[]
    Page->>Renew: navigate('/renew', state)
    Renew->>Renew: 첫 series 로딩
```

## 4. 서버 목록에서 `Study` 열기

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Page as FolderLeaderVer2Page
    participant Api as folderLeaderApi.ts
    participant Server as dicom_server_browser.py
    participant Renew as RenewPage

    User->>Page: Study 선택 후 Join
    Page->>Api: materializeServerStudy(study)
    Api->>Server: downloadUrl 별 파일 요청
    Server-->>Api: DICOM 파일 blob
    Api-->>Page: FolderStudy 반환
    Page->>Renew: navigate('/renew', state)
    Renew->>Renew: 선택된 series 표시
```

## 5. 뷰어에서 다른 `Study/Image`로 전환

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Renew as RenewPage
    participant Dock as RenewStudiesDock
    participant Api as folderLeaderApi.ts

    User->>Dock: 다른 항목 선택
    Dock->>Renew: onSelectSeries / onSelectImage
    Renew->>Renew: 현재 항목이 image 인지 study 인지 판별
    alt Image
        Renew->>Renew: image URL로 화면 교체
    else Study
        Renew->>Api: materializeServerStudy
        Api-->>Renew: FolderStudy 반환
        Renew->>Renew: series 상태 갱신
    end
    Renew-->>User: 새로운 대상 표시
```

## 6. 캡처 생성과 리포트 동기화

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Renew as RenewPage
    participant ReportApi as webReportApi.ts
    participant Server as web_report.py
    participant DB as web_report.db

    User->>Renew: Capture 클릭
    Renew->>Renew: 캡처 이미지 생성
    Renew->>Renew: capturedOutputs 저장
    Renew->>Renew: 캡처 박스 자동 열기
    User->>Renew: 캡처 선택
    Renew->>ReportApi: patchWebReportOverrides(attached_captures)
    ReportApi->>Server: PATCH /overrides
    Server->>DB: doctor_overrides 갱신
    Server-->>ReportApi: effective_result 반환
    ReportApi-->>Renew: 동기화 완료
```

## 7. 차트에서 리포트 세션 생성

```mermaid
sequenceDiagram
    participant User as 사용자
    participant Renew as RenewPage
    participant Api as webReportApi.ts
    participant Server as web_report.py
    participant Session as WebReportSessionService
    participant Report as WebReportReportService
    participant DB as web_report.db

    User->>Renew: Report 열기
    Renew->>Api: createWebReportFromChart(payload)
    Api->>Server: POST /api/web_report/from-chart
    Server->>Session: create_session()
    Server->>Session: save_ai_result()
    Server->>Report: generate_report()
    Server->>Session: create_report_version()
    Session->>DB: 세션/결과/버전 저장
    Server-->>Api: session_id + report_url
    Api-->>Renew: 생성 완료
```

## 8. 리포트 버전 복원

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
    Server->>Session: get_report_version(version)
    Session->>DB: snapshot 조회
    Server->>Merge: build_overrides_from_snapshot(...)
    Merge-->>Server: restored overrides
    Server->>Session: save_overrides()
    Server->>Session: set current_report_version
    Session->>DB: override + session 갱신
    Server-->>Api: restored_version + effective_result
    Api-->>Page: 복원된 화면 갱신
```


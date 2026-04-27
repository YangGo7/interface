# 현재 빌드 문서 모음

이 폴더는 현재 빌드를 기준으로 다시 정리한 문서 묶음이다.

문서 목표는 아래 3가지를 동시에 만족하는 데 있음.

- 처음 보는 사람이 화면 흐름을 이해하게 했음.
- 개발자가 내부 동작과 데이터 흐름을 추적하게 했음.
- 유지보수 담당자가 왜 코드를 이렇게 작성했는지 바로 읽게 했음.

읽는 순서는 아래와 같음.

1. [01_PRODUCT_AND_USER_FLOW.md](/abs/path/c:/interface/docs/current-build/01_PRODUCT_AND_USER_FLOW.md)
2. [02_RUNTIME_FLOW_AND_DATA.md](/abs/path/c:/interface/docs/current-build/02_RUNTIME_FLOW_AND_DATA.md)
3. [03_DESIGN_RATIONALE.md](/abs/path/c:/interface/docs/current-build/03_DESIGN_RATIONALE.md)
4. [04_BUILD_RELEASE_AND_OPERATIONS.md](/abs/path/c:/interface/docs/current-build/04_BUILD_RELEASE_AND_OPERATIONS.md)
5. [05_IMPLEMENTATION_RATIONALE.md](/abs/path/c:/interface/docs/current-build/05_IMPLEMENTATION_RATIONALE.md)
6. [release_dependency_troubleshooting_2026-04-24.md](/abs/path/c:/interface/docs/release_dependency_troubleshooting_2026-04-24.md)

이 문서 묶음이 답하는 질문은 아래와 같음.

- 이 프로그램은 무엇을 하는가
- 사용자는 어떤 순서로 쓰는가
- 프론트엔드와 백엔드는 어떻게 연결되는가
- 데이터는 어디에 저장되는가
- 왜 기능과 코드를 이 구조로 나눴는가
- 다른 PC에 어떻게 배포하는가

관련 핵심 파일은 아래와 같음.

- 메인 진입 화면: [FolderLeaderVer2Page.tsx](/abs/path/c:/interface/frontend/src/pages/FolderLeaderVer2Page.tsx)
- 뷰어 화면: [RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx)
- 리포트 화면: [WebReportPage.tsx](/abs/path/c:/interface/frontend/src/pages/WebReportPage.tsx)
- 서버 목록 API: [dicom_server_browser.py](/abs/path/c:/interface/gpts/api/dicom_server_browser.py)
- 리포트 API: [web_report.py](/abs/path/c:/interface/gpts/api/web_report.py)
- 리포트 세션 저장: [web_report_session_service.py](/abs/path/c:/interface/gpts/services/web_report_session_service.py)
- 릴리스 빌드 스크립트: [build_release.ps1](/abs/path/c:/interface/build_release.ps1)

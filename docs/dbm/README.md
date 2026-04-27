# DBM 다이어그램 문서 모음

이 폴더는 `DBM`을 이 프로젝트의 `DB + Backend + Module` 흐름으로 해석해서 정리한 다이어그램 문서 모음입니다.

목적:

- 전체 시스템이 어떻게 움직이는지 한눈에 보기
- 기능별 호출 순서 이해하기
- 프론트엔드와 백엔드 구조 파악하기
- 상태 변화와 DB 저장 구조 이해하기

문서 구성:

- [01_OVERALL_WORKFLOW.md](/abs/path/c:/interface/docs/dbm/01_OVERALL_WORKFLOW.md)
  - 전체 워크플로우와 화면 간 이동
- [02_FEATURE_SEQUENCES.md](/abs/path/c:/interface/docs/dbm/02_FEATURE_SEQUENCES.md)
  - 기능별 시퀀스 다이어그램
- [03_STRUCTURE_CLASS.md](/abs/path/c:/interface/docs/dbm/03_STRUCTURE_CLASS.md)
  - 구성도와 클래스/모듈 다이어그램
- [04_STATE_AND_DB.md](/abs/path/c:/interface/docs/dbm/04_STATE_AND_DB.md)
  - 상태 다이어그램과 DB 스키마 흐름

관련 핵심 파일:

- 메인 검사 선택: [FolderLeaderVer2Page.tsx](/abs/path/c:/interface/frontend/src/pages/FolderLeaderVer2Page.tsx)
- 뷰어: [RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx)
- 리포트: [WebReportPage.tsx](/abs/path/c:/interface/frontend/src/pages/WebReportPage.tsx)
- 서버 브라우저 API: [dicom_server_browser.py](/abs/path/c:/interface/gpts/api/dicom_server_browser.py)
- 리포트 API: [web_report.py](/abs/path/c:/interface/gpts/api/web_report.py)
- 리포트 세션 저장: [web_report_session_service.py](/abs/path/c:/interface/gpts/services/web_report_session_service.py)


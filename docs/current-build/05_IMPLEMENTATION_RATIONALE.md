# 현재 빌드 구현 이유

이 문서는 구조 설명이 아니라 구현 방식의 이유를 적은 문서다.

이 문서가 답하는 질문은 아래와 같음.

- 왜 일부 로직을 비동기로 작성했는가
- 왜 `useEffect`를 많이 썼는가
- 왜 polling을 썼는가
- 왜 `PATCH`로 저장했는가
- 왜 상태를 여러 개로 나눴는가
- 왜 특정 데이터는 즉시 불러오지 않고 나중에 materialize 했는가

## 1. 왜 `loadStudies`를 비동기로 작성했는가

`FolderLeaderVer2Page`의 `loadStudies`는 서버 폴더 스캔 결과를 받아와야 하므로 비동기로 작성했음.

이 함수가 비동기인 이유는 아래와 같음.

- 서버 폴더 스캔은 즉시 끝나는 연산이 아니다.
- 브라우저는 네트워크 응답을 기다리는 동안 화면을 멈추면 안 된다.
- 로딩, 새로고침, 실패 상태를 분리해서 다뤄야 한다.

그래서 현재 코드는 아래 구조를 썼음.

- `loading`
- `refreshing`
- `error`

이 구조 덕분에 첫 진입과 수동 새로고침을 다른 UI 상태로 보여준다.

관련 코드:

- [FolderLeaderVer2Page.tsx](/abs/path/c:/interface/frontend/src/pages/FolderLeaderVer2Page.tsx)
- [folderLeaderApi.ts](/abs/path/c:/interface/frontend/src/lib/folderLeaderApi.ts)

## 2. 왜 `buildDicomFolderStudies`도 비동기인가

로컬 DICOM 파일은 브라우저가 `File.arrayBuffer()`를 통해 읽어야 하기 때문에 비동기로 작성했음.

이 함수가 비동기인 이유는 아래와 같음.

- `File` 객체는 파일 내용을 바로 문자열처럼 들고 있지 않다.
- 각 파일을 읽어서 `dicomParser.parseDicom`에 넘겨야 한다.
- 여러 파일을 순서대로 읽고 study와 series로 다시 묶어야 한다.

즉 이 함수는 단순 계산 함수가 아니라 `파일 읽기 + DICOM 파싱 + 그룹핑`을 같이 수행한다.

관련 코드:

- [dicomFolderStudies.ts](/abs/path/c:/interface/frontend/src/features/upload/dicomFolderStudies.ts)

## 3. 왜 `materializeServerStudy`를 목록이 아니라 열 때 호출하는가

서버 목록 단계에서 모든 DICOM 파일을 전부 내려받지 않기 위해서다.

현재 목록 응답은 메타데이터 중심이다.

- patient 정보
- study 정보
- series 정보
- preview URL
- 개별 파일 download URL

뷰어는 실제 `File[]` 묶음이 필요하므로, 여는 순간에만 `materializeServerStudy`를 호출했음.

이렇게 구현한 이유는 아래와 같음.

- 목록 단계에서 모든 검사 파일을 미리 내려받으면 느려진다.
- 사용자가 열지 않는 검사까지 준비하는 것은 낭비다.
- 뷰어 입력 타입은 `FolderStudy`로 통일되어 있어 변환 지점이 필요하다.

관련 코드:

- [folderLeaderApi.ts](/abs/path/c:/interface/frontend/src/lib/folderLeaderApi.ts)
- [FolderLeaderVer2Page.tsx](/abs/path/c:/interface/frontend/src/pages/FolderLeaderVer2Page.tsx)
- [RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx)

## 4. 왜 `useEffect`로 초기 로딩을 거는가

React 함수형 컴포넌트는 렌더 함수 안에서 부수효과를 직접 실행하면 안 되기 때문이다.

현재 코드에서 초기 로딩은 아래처럼 처리했음.

- 첫 렌더 뒤 서버 목록 로딩
- 세션 로딩
- 주기적 갱신 시작
- 선택 상태 정리

이걸 `useEffect`에 둔 이유는 아래와 같음.

- 렌더와 네트워크 요청을 분리해야 한다.
- 의존성 배열로 언제 다시 실행할지 명확하게 제한할 수 있다.
- cleanup을 통해 interval을 정리할 수 있다.

관련 코드:

- [FolderLeaderVer2Page.tsx](/abs/path/c:/interface/frontend/src/pages/FolderLeaderVer2Page.tsx)
- [RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx)
- [WebReportPage.tsx](/abs/path/c:/interface/frontend/src/pages/WebReportPage.tsx)

## 5. 왜 일부 화면은 polling을 쓰는가

현재 리포트와 세션 화면은 백엔드 상태가 변할 수 있으므로 polling을 썼음.

예시는 아래와 같음.

- `WebReportPage`는 세션과 버전 목록을 주기적으로 다시 읽는다.
- `RenewReportWorkspacePanel`도 세션과 버전 상태를 반복 조회한다.

polling을 쓴 이유는 아래와 같음.

- 현재 구조에 WebSocket이 없다.
- 세션 상태와 버전은 서버에서 바뀌는 값이다.
- 사용자가 별도 새로고침 없이 최신 상태를 보게 해야 한다.

그래서 `setInterval` 기반 polling이 가장 단순하고 현재 구조에 맞는 선택이 됐음.

관련 코드:

- [WebReportPage.tsx](/abs/path/c:/interface/frontend/src/pages/WebReportPage.tsx)
- [RenewReportWorkspacePanel.tsx](/abs/path/c:/interface/frontend/src/components/chart/RenewReportWorkspacePanel.tsx)

## 6. 왜 리포트 수정은 `PATCH`를 쓰는가

리포트 수정은 전체 세션을 통째로 덮어쓰는 작업이 아니라 일부 override만 갱신하는 작업이기 때문이다.

현재 `patchWebReportOverrides`는 아래 항목만 부분적으로 바꾼다.

- `tooth_overrides`
- `report_note`
- `attached_captures`
- `reset_tooth_ids`

이걸 `PATCH`로 구현한 이유는 아래와 같음.

- AI 원본 결과는 유지해야 한다.
- 사용자가 바꾼 부분만 저장해야 한다.
- 변경 단위를 명확하게 제한해야 한다.

즉 현재 저장 모델은 `전체 문서 치환`이 아니라 `부분 수정 누적`이다.

관련 코드:

- [webReportApi.ts](/abs/path/c:/interface/frontend/src/lib/webReportApi.ts)
- [web_report.py](/abs/path/c:/interface/gpts/api/web_report.py)

## 7. 왜 `capturedOutputs`와 `attached_captures`를 분리했는가

뷰어에서 만든 모든 캡처를 자동으로 리포트에 넣지 않기 위해서다.

현재 상태 분리는 아래와 같음.

- `capturedOutputs`
  - 뷰어 안의 전체 캡처 목록
- `selectedReportCaptureIds`
  - 리포트에 보낼 캡처 선택 상태
- `attached_captures`
  - 백엔드 override에 저장되는 실제 리포트 캡처

이렇게 나눈 이유는 아래와 같음.

- 사용자가 캡처를 여러 장 만든 뒤 일부만 리포트에 넣게 하기 위해서다.
- 캡처 생성과 리포트 첨부를 분리하기 위해서다.
- 캡처 note와 치아 연결 정보를 리포트 저장 모델에 맞추기 위해서다.

관련 코드:

- [RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx)
- [RenewReportWorkspacePanel.tsx](/abs/path/c:/interface/frontend/src/components/chart/RenewReportWorkspacePanel.tsx)
- [web_report.py](/abs/path/c:/interface/gpts/api/web_report.py)

## 8. 왜 캡처 후 바로 캡처 박스를 열게 했는가

캡처를 만든 직후 사용자가 결과를 바로 확인하고 선택하게 하기 위해서다.

이렇게 구현한 이유는 아래와 같음.

- 캡처 성공 여부를 즉시 확인하게 했다.
- 사용자가 방금 만든 캡처를 다시 찾는 단계를 줄였다.
- 리포트 첨부 흐름을 끊기지 않게 했다.

즉 현재 동작은 `캡처 생성`과 `캡처 검토`를 연속 작업으로 묶은 구현이다.

관련 코드:

- [RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx)

## 9. 왜 리포트 세션은 생성 직후 버전까지 같이 만드는가

리포트 화면이 빈 세션만 보고 시작하지 않게 하기 위해서다.

현재 `createWebReportFromChart` 흐름은 아래를 바로 수행한다.

- 세션 생성
- assets 저장
- AI 결과 저장
- merge
- report 생성
- draft version 저장

이렇게 구현한 이유는 아래와 같음.

- 리포트 화면이 들어가자마자 미리보기를 보여줘야 한다.
- 첫 버전 기준점을 바로 남겨야 한다.
- 이후 복원과 재생성이 가능해진다.

관련 코드:

- [webReportApi.ts](/abs/path/c:/interface/frontend/src/lib/webReportApi.ts)
- [web_report.py](/abs/path/c:/interface/gpts/api/web_report.py)
- [web_report_session_service.py](/abs/path/c:/interface/gpts/services/web_report_session_service.py)

## 10. 왜 `current_report_version`을 따로 응답에 넣었는가

프론트가 현재 기준 버전을 명확히 알아야 하기 때문이다.

현재 리포트 UI는 아래 값을 써야 한다.

- 현재 프리뷰 URL
- 선택된 버전 표시
- 복원 후 현재 기준 버전
- draft와 final 상태 구분

그래서 세션 응답에 `current_report_version`을 직접 넣었음.

이 값이 없으면 프론트가 가장 최근 버전과 현재 선택 버전을 혼동하게 된다.

관련 코드:

- [web_report.py](/abs/path/c:/interface/gpts/api/web_report.py)
- [web_report_session_service.py](/abs/path/c:/interface/gpts/services/web_report_session_service.py)
- [WebReportPage.tsx](/abs/path/c:/interface/frontend/src/pages/WebReportPage.tsx)

## 11. 왜 버전 복원은 snapshot에서 override를 다시 만드는가

현재 편집 모델이 `AI 원본 + override` 조합이기 때문이다.

복원 시 HTML 파일만 바꾸면 아래 값이 맞지 않는다.

- 치아 체크 상태
- note
- bone loss 수정값
- included 상태
- attached capture

그래서 복원은 아래 순서로 구현했음.

1. 선택한 version의 `snapshot_json`을 읽는다.
2. `build_overrides_from_snapshot`를 호출한다.
3. override를 다시 저장한다.
4. `current_report_version`을 갱신한다.

즉 현재 구현은 `출력 파일 복구`가 아니라 `편집 상태 복구`다.

관련 코드:

- [web_report_merge_service.py](/abs/path/c:/interface/gpts/services/web_report_merge_service.py)
- [web_report.py](/abs/path/c:/interface/gpts/api/web_report.py)
- [WebReportPage.tsx](/abs/path/c:/interface/frontend/src/pages/WebReportPage.tsx)

## 12. 왜 릴리스 스크립트가 optional 파일을 경고만 내고 계속 가는가

배포 성공과 선택 파일 존재 여부를 분리했기 때문이다.

현재 `build_release.ps1`는 backend file 목록을 돌면서 아래처럼 처리한다.

- 있으면 복사한다.
- 없으면 warning을 출력한다.
- 전체 패키지 생성은 중단하지 않는다.

이렇게 한 이유는 아래와 같음.

- `.env.example` 같은 파일은 실행 필수 파일이 아니다.
- 선택 파일 하나 때문에 전체 릴리스가 실패하면 운영 흐름이 끊긴다.
- 배포 스크립트는 반복 실행에 강해야 한다.

관련 코드:

- [build_release.ps1](/abs/path/c:/interface/build_release.ps1)

## 13. 왜 API base 계산 함수를 별도로 뒀는가

개발 환경과 배포 환경의 주소 체계가 다르기 때문이다.

현재 프론트는 아래 상황을 구분한다.

- Vite 개발 서버
- 로컬 Flask 서버
- 같은 origin으로 붙는 배포 환경

그래서 `resolveDirectApiBase`와 보조 함수들을 분리했음.

이렇게 한 이유는 아래와 같음.

- 개발 서버에서 API 프록시가 HTML을 줄 때 직접 API base로 다시 붙기 위해서다.
- 배포 환경에서는 같은 origin을 그대로 쓰기 위해서다.
- API 경로 처리 규칙을 여러 파일에서 반복하지 않기 위해서다.

관련 코드:

- [folderLeaderApi.ts](/abs/path/c:/interface/frontend/src/lib/folderLeaderApi.ts)
- [webReportApi.ts](/abs/path/c:/interface/frontend/src/lib/webReportApi.ts)
- [RenewPage.tsx](/abs/path/c:/interface/frontend/src/pages/RenewPage.tsx)

## 14. 왜 이 문서가 따로 필요한가

설계 이유 문서만으로는 구현 선택의 기준점이 부족하기 때문이다.

이 문서는 아래 사람에게 필요하다.

- 비슷한 코드 패턴을 새 기능에도 그대로 적용해야 하는 개발자
- 비동기와 polling을 제거해도 되는지 판단해야 하는 리팩터링 담당자
- 상태 분리와 저장 방식을 바꾸기 전에 현재 이유를 확인해야 하는 유지보수 담당자


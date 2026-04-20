# Frontend 변경점 정리

작성일: 2026-04-20

## 범위

이번 변경은 로더 페이지의 이미지 선택 동작, 캡처 박스 미리보기 UX, 파노라마 측정 주석 입력 UX를 정리한 작업이다.

## 1. Loader 페이지 이미지 동작 변경

대상 파일: `frontend/src/pages/FolderLeaderVer2Page.tsx`

- 이미지 행에서 `single click` 시 해당 이미지를 선택하고 우측 패널에서 preview를 표시하도록 변경
- 이미지 행에서 `double click` 시 해당 이미지를 바로 `join/open` 하도록 변경
- `Studies` 탭 안에 섞여 있는 이미지 행에도 동일한 규칙 적용
- 기존 `Open` / `Join` 버튼 동작은 유지

## 2. Capture Box 미리보기 창 개선

대상 파일: `frontend/src/components/chart/OutputCapturePanel.tsx`

- 캡처 썸네일 클릭 시 큰 미리보기 창이 뜨도록 변경
- 같은 썸네일을 다시 클릭하면 미리보기 창이 닫히도록 변경
- 우측 상단 `X` 버튼으로 닫기 가능
- 미리보기 창을 드래그해서 이동 가능하도록 변경
- 스케일된 레이아웃에서도 마우스 포인터 기준으로 드래그 위치가 맞도록 좌표 계산 보정
- 캡처 제거 또는 패널 상태 변경 시 열린 미리보기 상태가 정리되도록 보완

## 3. Measure Annotation 입력 방식 변경

대상 파일: `frontend/src/pages/RenewPage.tsx`

- 기존 browser `alert/prompt` 기반 입력을 제거
- 클릭 시 파노 영역 위에 텍스트 입력 박스가 뜨도록 변경
- 입력 박스는 `OK` 시 저장, `Cancel` 시 취소
- 입력 박스와 저장된 annotation 텍스트 색상을 형광 초록 계열로 조정
- 반투명 검은 배경을 제거
- 입력 중 포커스가 반복 초기화되어 한 자리에서 글자가 덮어쓰이던 문제 수정
- 저장 후 텍스트가 우측 상단으로 튀는 현상을 줄이기 위해 overlay 기준 anchor 좌표를 함께 저장하도록 수정
- popup 내부 클릭이 상위 viewport click 처리로 전파되지 않도록 이벤트 전파 차단 추가

## 검증

- `frontend` 디렉터리에서 `npm.cmd run build` 실행
- 빌드 통과 확인


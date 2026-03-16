# Chart Features

## 대상 파일

- `frontend/src/pages/ChartPage.tsx`
- `frontend/src/components/BottomTeethChart.tsx`
- `gpts/utils/report_v3.py`

## 1. 메인 뷰어

### 기능

- 원본 이미지와 AI 오버레이를 전환해서 볼 수 있다.
- 확대/축소, 팬, 회전, 좌우 반전이 가능하다.
- `Window / Level` 방식으로 밝기와 대비를 조절할 수 있다.

### 사용 로직

- 뷰어 내부 상태를 `scale`, `zoom`, `offset`, `rotation`, `flipped`, `brightness`, `contrast`로 분리해서 관리한다.
- 이미지는 `img` 태그 하나를 기준으로 렌더하고, 시각 보정은 CSS `filter: brightness(...) contrast(...)`로 적용한다.
- `Window / Level`은 DICOM 전용 라이브러리를 붙이지 않고, 마우스 드래그를 밝기/대비 변화에 직접 매핑했다.

### 왜 이렇게 했는가

- 파노라마 이미지는 원본 해상도와 비율이 제각각이라, 좌표계를 하나로 유지하는 편이 중요하다.
- `img + filter` 조합은 구현 비용이 낮고 브라우저 호환성이 좋다.
- 현재 요구사항은 의료 PACS급 WL/WW 수치 정확도보다, 빠르게 보고 조정하는 인터랙션이 더 중요했다.

## 2. 드로잉 / 측정 도구

### 기능

- `Measure`: Length, Bidirectional, Angle
- `Annotate`: Annotation, Arrow, Ellipse, Rectangle, Circle, Freehand ROI, Spline ROI, Livewire Tool
- 도형 클릭 삭제
- 길이/면적/반경 등의 정보 라벨 표시

### 사용 로직

- 모든 도형은 공통 `shapes` 배열에 저장한다.
- 실제 렌더는 이미지 위의 SVG 오버레이에서 수행한다.
- 좌표는 화면 좌표가 아니라 이미지 좌표로 저장한다.
- 도형 스타일은 다음 원칙으로 맞췄다.
  - 화살표: 노란색 점선 1px
  - 일반 도형: 초록색 점선 1px
  - 텍스트: 배경 없이 표시

### 왜 이렇게 했는가

- 이미지 위에서 확대/축소/이동을 하더라도 도형이 같이 움직여야 하므로, 도형 좌표는 이미지 기준이어야 한다.
- SVG는 측정선, 텍스트, ROI를 벡터 기반으로 안정적으로 표시하기 쉽다.
- 툴을 추가하거나 스타일을 바꾸는 비용이 캔버스보다 낮다.

## 3. Livewire

### 기능

- 두 점 이상을 찍으면 경계에 가까운 경로를 따라가는 `livewire-like` 도구를 제공한다.

### 사용 로직

- 이미지 로드 시 edge map을 만든다.
- 현재 구현은 Sobel 기반 edge map을 사용한다.
- 사용자가 찍은 점 사이를 edge strength가 높은 방향으로 따라가도록 segment를 생성한다.

### 왜 이렇게 했는가

- 완전한 의료용 shortest-path livewire 알고리즘은 구현 비용이 크다.
- 현재 목적은 “완전 수동 freehand”보다 경계 추종성이 나은 도구를 제공하는 것이다.
- Sobel edge map 기반 접근은 성능과 구현 복잡도의 균형이 좋다.

## 4. 오버레이 좌표 정합

### 기능

- 도형이 이미지 밖 검은 배경으로 튀지 않도록 제한한다.
- 이미지 위에만 드로잉이 생성되게 한다.

### 사용 로직

- 클릭 좌표를 이미지 `getBoundingClientRect()` 기준으로 변환한다.
- 클릭 위치가 실제 이미지 rect 밖이면 `null` 처리해서 입력을 버린다.
- SVG 오버레이는 이미지와 동일한 박스 내부에만 둔다.

### 왜 이렇게 했는가

- 뷰어 배경까지 입력을 허용하면 큰 원/사각형이 엉뚱한 위치에 생긴다.
- 파노라마는 letterbox 검은 여백이 자주 생겨서, 이 구간 차단이 필수다.

## 5. 툴바 구성

### 기능

- 좌측 툴바는 2칸 페어 배치를 유지한다.
- 하단 컨트롤은
  - `Zoom In | Zoom Out`
  - `Reset | empty`
  - `Overlay on/off` 2칸 전체 사용

### 사용 로직

- CSS grid 대신 각 줄을 `flex` 2칸 행으로 고정했다.
- `Overlay on/off`만 단일 버튼이지만 전체 폭을 쓰게 설계했다.

### 왜 이렇게 했는가

- grid만으로는 실제 화면에서 한 줄 세로열처럼 무너지는 경우가 있었다.
- 행 단위 `flex`가 현재 레이아웃에서는 더 예측 가능했다.

## 6. 오돈토그램

### 기능

- `report_v3.py` 기준 triage 색 규칙을 따른다.
- missing, implant, finding 상태가 색과 타일 스타일에 반영된다.
- 타일 모양은 chamfered box 형태다.
- `20번대`, `30번대`는 치아 이미지를 좌우 대칭으로 반전한다.
- missing은 회색 배경 + 검은 치아, 일반 치아는 컬러 배경 + 흰 치아로 보인다.

### 사용 로직

- `BottomTeethChart.tsx`에서 치아별 상태를 기반으로 fill, border, dashed 여부를 계산한다.
- 타일은 `clipPath`로 모서리를 살짝 깎는다.
- 치아 이미지는 CSS `filter`와 `scaleX(-1)`로 색상/대칭을 처리한다.
- 강조 ring은 extraction, implant site, triage highlight를 함께 표시할 수 있게 분리했다.

### 왜 이렇게 했는가

- 오돈토그램은 “정확한 치아 위치 감각”과 “상태 식별 속도”가 중요하다.
- 대칭 반전이 없으면 상악/하악 좌우가 시각적으로 어색하다.
- `report_v3`와 규칙을 맞추지 않으면 PDF/HTML 리포트와 차트 화면이 서로 다르게 보이게 된다.

## 7. 범례와 표기

### 기능

- 범례는 triage 중심으로 단일 라인에 가깝게 배치한다.
- `Notation`은 우측 정렬, `Odontogram Reference`는 중앙 하단에 둔다.

### 사용 로직

- `DentalChartLegend()`에서 항목별 색과 간격을 개별 정의한다.
- 차트 상단은 범례, notation, reference를 서로 다른 역할로 분리해 배치했다.

### 왜 이렇게 했는가

- 이전 3탭 구조보다 단일 패널 + 범례 구성이 실제 사용 흐름에 맞았다.
- 차트는 “탭 이동”보다 “상태 읽기”가 더 중요한 화면이다.

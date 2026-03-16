# Status API Mapping

## 대상 파일

- `gpts/app.py`
- `frontend/src/pages/ChartPage.tsx`
- `frontend/src/components/RightPanel.tsx`

## 목적

프론트는 치아별 상태를 표시할 때 다음과 같은 정규화된 구조를 기대한다.

- `caries_by_tooth_best`
- `periapical_by_tooth_best`
- `implant_by_tooth_best`
- `crown_by_tooth_best`
- `filling_by_tooth_best`
- `pbl`
- `pbl_level`
- `teeth`

하지만 실제 파이프라인 결과는 케이스마다 구조가 다를 수 있다.

- raw list만 있는 경우
- `teeth[].findings`만 있는 경우
- 기존 map이 있지만 비어 있는 경우

그래서 `/api/detect/status/<job_id>` 단계에서 프론트 친화적인 형태로 다시 정규화한다.

## 1. `build_best_map`

### 기능

- 치아별 detection map을 재구성한다.
- 같은 치아에 여러 결과가 있으면 최고 confidence만 남긴다.

### 사용 로직

입력 소스 세 가지를 합친다.

1. 기존 `existing_map`
2. `teeth[].findings`
3. raw detection list (`caries`, `periapical`)

각 결과는 다음 형태로 정리한다.

```python
{
    "conf": float,
    "box": [...]
}
```

그리고 `conf`가 더 높은 항목만 유지한다.

### 왜 이렇게 했는가

- 프론트 우측 패널은 치아 하나당 대표 confidence 하나를 바로 보여줘야 한다.
- raw list만 내려주면 프론트에서 다시 그룹핑해야 하고, 화면별로 로직이 달라질 위험이 있다.
- 서버에서 정규화하면 `ChartPage`, `RightPanel`, 향후 리포트 생성기까지 같은 구조를 공유할 수 있다.

## 2. PBL fallback

### 기능

- `pbl`, `pbl_level`이 직접 없더라도 `bonelevel`에서 복구한다.

### 사용 로직

- `bonelevel[k]["percent"]` -> `pbl[k]`
- `bonelevel[k]["level"]` -> `pbl_level[k]`

### 왜 이렇게 했는가

- 일부 결과는 `bonelevel`만 있고 요약 map이 없을 수 있다.
- 프론트는 치아별 stage와 percent를 바로 쓰기 때문에 fallback이 필요하다.

## 3. 프론트에 내려주는 핵심 키

### 기능

다음 키를 안정적으로 내려준다.

- `overlay_url`, `image_url`
- `teeth`, `data`
- `odontogram_map`
- `caries_by_tooth`, `caries_by_tooth_best`
- `periapical_by_tooth`, `periapical_by_tooth_best`
- `missing_teeth`, `teeth_missing`
- `pbl`, `pbl_level`
- `implant_metrics`

### 왜 이렇게 했는가

- 기존 프론트는 버전별로 `image_path`, `image_url`, `overlay_path`, `overlay_url` 등을 혼용했다.
- alias를 같이 제공하면 이전 화면과 신규 화면을 동시에 안정적으로 지원할 수 있다.

## 4. 프론트에서 이 데이터를 어떻게 쓰는가

### `ChartPage.tsx`

- 치아 상태 맵 생성
- 오돈토그램 triage 계산
- 박스 오버레이 생성
- 요약 범례 구성

### `RightPanel.tsx`

- 선택한 치아의 caries/periapical confidence 표시
- implant / crown / filling 상태 표시
- PBL 값 표시

## 결론

`detect_status()`는 단순 상태 조회 API가 아니라, 프론트가 바로 렌더링할 수 있는 `view model builder` 역할도 같이 한다.

이 설계를 쓴 이유는 다음과 같다.

- 프론트가 얇아진다.
- 버전별 응답 차이를 한 곳에서 흡수할 수 있다.
- 차트, 우측 패널, 리포트 규칙을 같은 데이터 기준으로 맞추기 쉬워진다.

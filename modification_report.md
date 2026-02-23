# 코드 수정 내역 보고서

본 문서는 임플란트 직경 계산 방식 변경 및 오탐 방지를 위한 오버레이 마진(5%) 적용에 대한 수정 사항을 기술합니다.

---

## 1. 수정 함수: `backend/services/pano_inference.py`의 `PanoPipeline.run`

### 1-1. 수정 내역
1.  **임플란트 직경 계산 로직 변경**:
    -   기존: `compute_from_gray_with_mask` 사용. Grayscale 이미지의 명암(Intensity)을 참조하여 임플란트 영역을 계산.
    -   변경: `compute_sample_axis` 사용. AI가 예측한 **Segmentation Mask(Contour)** 의 형상만을 기준으로 직경을 계산하도록 변경.
2.  **Bone Level/CEJ 오버레이 마진 적용**:
    -   이미지 상하좌우 5% 영역에 그려진 Bone Level 및 CEJ 라인을 강제로 지워(0으로 설정), 가장자리 오탐 및 아티팩트가 오버레이에 표시되지 않도록 함.

### 1-2. 수정 코드 (발췌)

```python
# [변경 전]
# result = compute_from_gray_with_mask(gray, mask, n_samples=40)

# [변경 후] from utils.sample_axis_service import compute_sample_axis 추가 필요
# Contour(마스크) 형상만을 기준으로 축/직경 계산
result = compute_sample_axis(mask, mask, n_samples=40)
```

```python
# [추가된 마진 제거 로직]
bl_canvas, pbl_dict, cej_count, bl_count = calc.get_bonelevel(copy.deepcopy(img), seg_res)

# Clip bl_canvas to inner_rect to prevent edge artifacts
if inner_rect:
    # 5% 마진 영역(상/하/좌/우)을 0으로 지움
    bl_canvas[:inner_rect["y1"], :] = 0
    bl_canvas[inner_rect["y2"]:, :] = 0
    bl_canvas[:, :inner_rect["x1"]] = 0
    bl_canvas[:, inner_rect["x2"]:] = 0

overlay = cv2.add(overlay, bl_canvas)
```

### 2. 코드 설명
*   **`compute_sample_axis(mask, mask, ...)`**: 첫 번째 인자는 기하학적 형상(Geometry) 마스크, 두 번째는 폭(Width) 계산용 마스크입니다. 기존에는 첫 번째 인자에 `gray` 이미지를 넣어 명암 차이를 이용해 축을 보정했으나, 이를 `mask`로 대체함으로써 순수하게 AI가 예측한 영역의 모양만으로 중심축과 직경을 구하게 됩니다.
*   **`inner_rect` 마스킹**: `inner_rect`는 이미지 전체 크기의 5% 안쪽 영역을 정의한 사각형입니다. `bl_canvas` 배열 슬라이싱을 통해 이 사각형의 바깥쪽(가장자리) 픽셀 값을 모두 0(검은색/투명)으로 초기화하여, 가장자리에 잘못 그려진 라인을 시각적으로 제거합니다.

---

## 3. 수정 함수: `backend/services/postprocess.py`의 `build_overlay_lines`

### 3-1. 수정 내역
1.  **Frontend 오버레이 데이터 필터링**:
    -   백엔드 이미지뿐만 아니라, 프론트엔드로 전송되는 선(Line) 데이터에 대해서도 5% 마진 검사를 수행합니다.
    -   검출된 객체(임플란트 등)의 중심점이 이미지 가장자리 5% 영역에 위치할 경우, 해당 객체의 데이터(축, 직경 표시)를 결과 리스트에서 제외합니다.

### 3-2. 수정 코드 (발췌)

```python
# [추가된 필터링 로직]
# [Filtering] Skip if detection center is in the outer 5% margin
if axis_center:
    cx, cy = axis_center
    margin_x = img_width * 0.05
    margin_y = img_height * 0.05
    
    # 중심점이 안전 영역(90%) 안에 있는지 확인
    if not (margin_x <= cx <= img_width - margin_x and margin_y <= cy <= img_height - margin_y):
         continue
```

### 4. 코드 설명
*   **`margin_x`, `margin_y`**: 전체 이미지 폭과 높이의 5%에 해당하는 픽셀 수입니다.
*   **조건문 `if not (...)`**: 객체의 중심 좌표(`cx`, `cy`)가 `margin`보다 작거나(왼쪽/위쪽 끝), `width - margin`보다 클 경우(오른쪽/아래쪽 끝)를 감지합니다. 이 경우 `continue`를 실행하여 해당 객체의 시각화 데이터를 생성하지 않고 건너뜁니다. 이를 통해 프론트엔드 화면에서도 가장자리 오탐 객체가 표시되지 않게 됩니다.

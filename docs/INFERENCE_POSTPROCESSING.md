# Inference Postprocessing

## 대상 파일

- `gpts/models/yolo_detector.py`
- `gpts/utils/post_processing/isolated_guard.py`

## 1. Isolated Guard

### 기능

세그/디텍션 결과에서 낮은 confidence 박스 중, 충분히 강한 박스와 겹치는 것만 살리고 고립된 저신뢰 박스는 제거한다.

### 기본 임계값

```python
HIGH_CONF = 0.25
LOW_CONF = 0.20
IOU_GUARD = 0.30
```

### 사용 로직

1. 추론 자체는 `LOW_CONF`까지 받는다.
2. 결과를 두 그룹으로 나눈다.
   - high tier: `conf >= 0.25`
   - low tier: `0.20 <= conf < 0.25`
3. low tier 박스마다 high tier와 IoU를 비교한다.
4. 어떤 high tier와도 `IoU > 0.3`이 아니면 버린다.

즉 개념적으로는 다음과 같다.

```python
for low_pred in low_tier_preds:
    is_isolated = True
    for high_pred in high_tier_preds:
        if box_iou(low_pred, high_pred) > IOU_GUARD:
            is_isolated = False
            break
```

### 왜 이렇게 했는가

- confidence를 너무 높게 잡으면 놓치는 검출이 생긴다.
- confidence를 너무 낮게 잡으면 isolated noise가 늘어난다.
- 이 로직은 “낮은 conf도 일단 받고, 진짜 근거가 있는 것만 통과”시키는 절충안이다.

## 2. `yolo_detector.py` 통합 방식

### 기능

- detector에서 `use_isolated_guard=True`일 때만 guard를 적용한다.

### 사용 로직

- 추론 시 `conf`를 `min(self.confidence_threshold, LOW_CONF)`로 낮춘다.
- 결과를 받은 뒤 `filter_results_with_isolated_guard()`를 한 번 더 태운다.

### 왜 이렇게 했는가

- 모델 추론 코드는 그대로 두고, 후처리만 선택적으로 붙이는 구조가 유지보수에 유리하다.
- 실험 단계에서는 모델별로 guard on/off를 쉽게 비교할 수 있다.

## 3. 기대 효과

- 경계 근처에서 confidence가 약간 낮게 나온 후보는 살릴 수 있다.
- 완전히 뜬금없는 단독 저신뢰 박스는 줄일 수 있다.
- 단순히 confidence threshold 하나만 올리는 것보다 recall 손실이 덜하다.

## 4. 주의할 점

- 이 로직은 high tier 기준이 잘 잡혀 있어야 효과가 좋다.
- 데이터셋이 바뀌면 `0.25 / 0.20 / 0.30` 값은 다시 조정해야 할 수 있다.
- 클래스 특성에 따라 치아/병변별 최적값이 다를 수 있다.

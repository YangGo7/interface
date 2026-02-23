# 신경관 안전거리 측정 로직 (Nerve Safety Distance)

**최종 업데이트**: 2026-02-09

## 1. 개요

결손치(Missing Teeth) 부위에 임플란트 식립 시, **가용 골 높이(Available Bone Height)**를 측정하여 신경관(Inferior Alveolar Nerve)까지의 안전 거리를 계산합니다.

## 2. 측정 로직 흐름

```
[인접 치아 탐색] → [Gap Center 계산] → [축(Axis) 결정] → [Raycast] → [안전 마진 적용]
```

---

## 3. 상세 로직

### 3.1 인접 치아 탐색

```python
# FDI 번호 기준 양옆 치아 후보 결정
candidates = []
if n > 1: candidates.append(q*10 + (n-1))  # 앞 치아
if n < 8: candidates.append(q*10 + (n+1))  # 뒤 치아

t_n1 = get_tooth_by_label(candidates[0])
t_n2 = get_tooth_by_label(candidates[1])
```

---

### 3.2 Gap Center 계산

#### X 좌표 (수평 위치)
```python
if len(valid_boxes) >= 2:
    # 두 인접 치아 사이의 중간점
    b1, b2 = sorted boxes by X
    gap_cx = (b1[2] + b2[0]) / 2  # 왼쪽 박스 오른쪽 + 오른쪽 박스 왼쪽
elif len(valid_boxes) == 1:
    # 한쪽만 있으면 외삽(Extrapolate)
    gap_cx = box_center + (direction * width * 1.1)
```

#### Y 좌표 (수직 레벨)
```python
# 우선순위:
# 1. CEJ 중심점 (ML 탐지)
if valid_cejs:
    gap_cy = average(cej_y coordinates)

# 2. 인접 치아 박스 중심점 (Fallback)
elif valid_boxes:
    center_ys = [(box[1] + box[3]) / 2 for box in valid_boxes]
    gap_cy = average(center_ys)
```

---

### 3.3 축(Axis) 벡터 결정 - PCA 기반

**핵심 로직**: 인접 치아의 Contour에서 **주성분 분석(PCA)**으로 치아 기울기를 계산하고 평균합니다.

```python
valid_axes = []
for t in [t_n1, t_n2]:
    if t and t.get('contour'):
        pts = np.array(t['contour']).reshape(-1, 2)
        
        # PCA 수행
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.cov(centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        
        # 주축(Major Axis) = 가장 큰 고유값에 해당하는 고유벡터
        major_axis = evecs[:, np.argmax(evals)]
        major_axis = major_axis / np.linalg.norm(major_axis)
        
        # 방향 보정 (하악: Y+, 상악: Y-)
        if q in [3, 4] and major_axis[1] < 0: major_axis = -major_axis
        if q in [1, 2] and major_axis[1] > 0: major_axis = -major_axis
        
        valid_axes.append(major_axis)

# 인접 치아 축 평균
if valid_axes:
    avg_x = sum(v[0] for v in valid_axes) / len(valid_axes)
    avg_y = sum(v[1] for v in valid_axes) / len(valid_axes)
    axis_vec = normalize(avg_x, avg_y)
else:
    # Fallback: 순수 수직
    axis_vec = (0, 1) if Lower else (0, -1)
```

---

### 3.4 Raycast (신경관까지 거리 측정)

```python
ray_x, ray_y = gap_cx, gap_cy
for _ in range(max_step):
    ray_x += axis_vec[0]
    ray_y += axis_vec[1]
    
    if nerve_mask[int(ray_y), int(ray_x)] > 0:
        p_target = (ray_x, ray_y)
        break

dist_px = distance(gap_center, p_target)
```

---

### 3.5 안전 마진 적용 (2mm)

```python
pixels_per_mm = 1.0 / mm_per_px
safety_margin_px = 2.0 * pixels_per_mm

safe_dist_px = max(0, dist_px - safety_margin_px)
safe_dist_mm = safe_dist_px * mm_per_px

# 시각화 좌표
safe_end = gap_center + axis_vec * safe_dist_px    # Yellow Line
margin_end = safe_end + axis_vec * safety_margin_px # Cyan Line (2mm)
```

---

## 4. 시각화

| 요소 | 색상 | 설명 |
|------|------|------|
| Safe Distance Line | **Yellow** | 안전 거리 (사용 가능한 골 높이) |
| Safety Margin | **Cyan** | 2mm 안전 마진 구간 |
| Start Point | Yellow Dot | Gap Center (시작점) |
| End Point | Cyan Dot | Nerve 접촉점 |
| Distance Label | Yellow Text | "X.Xmm" 거리 표시 |
| Margin Label | Cyan Text | "2mm" 표시 |

---

## 5. 파일 참조

- **로직 구현**: `gpts/services/tooth_logic.py` → `find_missing_teeth()`
- **시각화**: `gpts/services/visualizer.py` → `draw_safety_guides()`
- **파이프라인 호출**: `gpts/services/pano_inference.py` → `_draw_visuals()`

---

## 6. 예시 다이어그램

```
     Gap Center (Yellow Dot)
          │
          │  ← axis_vec (PCA 기반 기울기)
          │
    Yellow Line (Safe Distance: 8.6mm)
          │
          ▼
    ─────────── Safe End
          │
     Cyan Line (2mm Safety Margin)
          │
          ▼
    ─────────── Nerve (Magenta Mask)
```

---

**Note**: 상악(Q1, Q2)의 경우 동일 로직이 Sinus(상악동)에 대해 적용됩니다.

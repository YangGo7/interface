# PBL 치아 축 vs 임플란트 축

치아(PBL) 축과 임플란트 축을 어떻게 계산하는지, 수식과 접근 방식, 유사점/차이점을 정리했습니다. 경로는 워크스페이스 기준입니다.

## 치아 축 (PBL) — `backend/services/pano_calc_utils.py:95-182`
- **입력**: 잘라낸 치아 마스크 `I(x, y)`, 치아 번호 `num`, 높이 `H`, 폭 `W`.
- **루트 점(root)**  
  - 루트 밴드 선택: 상악(번호 1*, 2*)은 위쪽 `th`줄, 하악(그 외)은 아래쪽 `th`줄.  
    `ROI_root = {(x, y) | I(x, y) > 0, y ∈ root band}`  
  - 비어 있으면 기본값 `(W/2, 0 또는 H-1)`.  
  - 비어 있지 않으면  
    `x_root = (min_x(ROI_root) + max_x(ROI_root)) / 2`  
    `y_root = 0`(상악) 또는 `H-1`(하악).
  - 다근치 보정(`check_multi_root`): 루트 밴드에서 `connectedComponentsWithStats` → 각 컴포넌트 중심 `(x_i, y_i)`를 모아  
    `x_root = mean(x_i)`, `y_root = harmonic_mean(y_i)`로 얕은 루트를 우선.
- **크라운 점(crown)**  
  - 반대쪽 밴드(대구치는 `th`를 더 넓게) 선택:  
    `ROI_crown = {(x, y) | I(x, y) > 0, y ∈ crown band}`  
  - 비어 있으면 기본값 `(W/2, H 또는 0)`.  
  - 비어 있지 않으면  
    `x_crown = (min_x(ROI_crown) + max_x(ROI_crown)) / 2`  
    `y_crown = H`(상악) 또는 `0`(하악).
- **축 벡터와 길이**  
  - `p_root = (x_root, y_root)`, `p_crown = (x_crown, y_crown)`  
  - `v_tooth = p_crown - p_root = (Δx, Δy)`  
  - `|v_tooth| = sqrt((Δx)^2 + (Δy)^2)`
- **PBL에서 사용**  
  - 로컬 ROI에 1픽셀 선을 그림.  
  - 겹침 길이: `L_bl = |line ∩ bonelevel_mask|`, `L_cej = |line ∩ cej_mask|`  
  - `pbl_ratio = (|v_tooth| - L_bl) / (|v_tooth| - L_cej)` (0 방어)  
  - `pbl_percent = pbl_ratio * 100`

## 임플란트 축 — `backend/services/pano_calc_utils.py:423-470`
- **입력**: 임플란트 컨투어 `C`.
- **회전 바운딩박스**  
  - `rect = cv2.minAreaRect(C)` → 중심 `(cx, cy)`, 변 `(w, h)`, 회전각 `θ`.  
  - 꼭짓점 `B = {b0, b1, b2, b3} = cv2.boxPoints(rect)`.
- **긴 변 선택**  
  - 변 길이: `d01 = ||b0 - b1||`, `d12 = ||b1 - b2||`  
  - `length_px = max(w, h)`, `diameter_px = min(w, h)`  
  - 긴 변에 해당하는 두 꼭짓점 쌍을 골라, 그 중점을 축 끝점 `p_top`, `p_bot`으로 사용.
- **축 벡터와 길이**  
  - `v_impl = p_bot - p_top`, `|v_impl| = length_px`  
  - 방향은 `θ`를 그대로 따름.
- **사용**  
  - 오버레이, 길이/직경 산출에 그대로 사용하며, 물체의 기울기를 보존.

## 임플란트 축 (PCA 샘플링 방식) — `C:/DentexSegAndDet-main/segmodel/util/sample_axis_service.py`
- **입력**: 이진 마스크 `mask` (컨투어 기반).
- **행 샘플링으로 중점 수집**  
  - 마스크 y범위 `[y_min, y_max]`를 따라 `n_samples`개 행을 균등 간격으로 선택.  
  - 각 행에서 좌우 에지 `x_min, x_max`를 잡고 중점 `m = ((x_min + x_max)/2, y)`를 저장.  
  - 점 집합 `M = {m_i}`를 10% 트리밍 후 이상치 제거(`filter_outliers`).
- **PCA로 축 추정**  
  - 초기축: 전체 픽셀 좌표 `P = {(x,y) | mask>0}`에 대해 공분산의 최대 고유벡터 `axis_pca`와 중심 `c_pca = mean(P)`.  
  - 최종축: 클린한 중점 집합 `M_keep`에 PCA 적용 → `axis_vec`, 중심 `axis_center` (없으면 `axis_pca`, `c_pca`로 대체).  
  - 길이/직경: 축에 직교하는 방향으로 마스크를 회전시켜 최대 가로 길이를 찾아 직경 좌표를 복원.
- **출력**: 축 방향 단위벡터 `axis_vec`, 중심 `axis_center`, 직경 점들, 샘플된 중점 기록(시각화용).

## 유사점
- 모두 끝점 두 개와 벡터 `v`, 크기 `|v|`를 반환한다.
- 모두 세그/컨투어 기하만으로 축을 만든다(외부 priors 없음).
- 시각화와 후속 계산(PBL 비율 vs 임플란트 길이/직경)에 동일 축을 재사용한다.

## 차이점
- **기준**: 치아는 상·하 밴드의 x 투영(min/max)으로 축을 잡고, 임플란트는 회전 최소사각형의 긴 변을 축으로 쓴다.
- **방향 제어**: 치아는 악궁(상/하)과 다근치 여부에 따라 ROI 방향·두께를 바꾼다. 임플란트는 `minAreaRect`의 회전각 `θ`를 그대로 따른다.
- **노이즈 민감도**: 치아는 crown/root 밴드가 끊기면 축이 흔들릴 수 있다. 임플란트는 컨투어가 깨끗하면 안정적이다.
- **길이 의미**: 치아 축 길이는 루트↔크라운 전장, 임플란트 축 길이는 회전 bbox의 긴 변(실제 길이 근사).
- **임플란트 방식 비교**:  
  - 현재 백엔드(`calc_implant_metrics`): `minAreaRect` 긴 변의 중점을 잇는 직선 → 단순·빠르고 바운딩박스 기울기를 그대로 사용.  
  - PCA 샘플링(`sample_axis_service`): 행별 중점을 모아 PCA 주성분을 축으로 선택 → 마스크 내부 형상에 더 민감하며 부분적으로 잘린 컨투어에서도 주성분을 추정 가능하지만, 샘플링·이상치 처리 비용이 추가됨.

## 접근 rationale
- 치아: 해부학적 상/하 방향을 유지하고, 다근치에서 루트 분리로 인한 편향을 줄이기 위해 투영+컴포넌트 평균을 사용.
- 임플란트: 경사·회전된 강체 모양에 대해 `minAreaRect`가 방향과 비율을 안정적으로 제공하므로 긴 변을 축으로 삼아 일관성 확보.
- PCA 임플란트: 마스크 내부 중심선에 더 잘 맞추기 위해 중점들의 주성분을 사용하며, bbox가 왜곡되거나 부분 가려짐이 있을 때 보완 수단이 된다.

## 관련 호출 위치
- 치아/PBL: `get_bonelevel` (`backend/services/pano_calc_utils.py:238-355`), 결과 평탄화는 `backend/services/pano_inference.py:975-995`.
- 임플란트: `calc_implant_metrics` (`backend/services/pano_calc_utils.py:423-470`), 매핑은 `backend/services/pano_inference.py` 내 임플란트 처리.

## PBL 축 코드(라인별 설명) — `backend/services/pano_calc_utils.py:95-182`
```
95: def get_principal_axis(cropped_img, num: str):
96:     def get_root_pos(img, th=5, opt="up"):
98:         roi = img[:th, :] if opt == "up" else img[-th:, :]        # 루트 밴드 슬라이스
100:        points = np.argwhere(roi > 0)                             # 밴드 내 포그라운드 픽셀
103:        if points.size == 0: return (int(img.shape[1]/2), 0 or H) # 없으면 중앙 x, 경계 y
106:        most_left = np.min(points[:, 1]); most_right = np.max(...)# x 최소/최대
109:        return (int((most_left + most_right) / 2), start_y)       # x 평균, y 경계

111:    def get_crown_pos(img, num, th=15, opt="up"):
112:        if len(num)>1 and num[1] in ["6","7"]: th = 30            # 대구치 두꺼운 밴드
116:        roi = img[-th:, :] if opt == "up" else img[:th, :]        # 크라운 밴드
119:        points = np.argwhere(roi > 0)
120:        if points.size == 0: return (int(img.shape[1]/2), edge_y) # 없으면 중앙 x, 반대 경계 y
123:        most_left = np.min(points[:, 1]); most_right = np.max(...)
125:        return (int((most_left + most_right)/2), edge_y)          # x 평균, y 반대 경계

130:    def check_multi_root(img, th=20, opt="up"):
143:        for col in range(start, end, step):                       # 루트 밴드에서 연결성 탐색
146:            cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(crop)
148:            if cnt > 2:                                           # 다근치 감지 시
150:                for i in range(1, cnt):
152:                    if opt == "up": pt_r = get_root_pos(...,"up") # 각 루트 중심 계산
155:                    else:           pt_r = get_root_pos(...,"lo")
158:                    x_arr.append(pt_r[0]); y_arr.append(pt_r[1]+y_bias)
164:        if not x_arr: return (0,0), False
164-166: avg_x = mean(x_arr); avg_y = harmonic_mean(y_arr); return (avg_x, avg_y), True

168:    f = False; pt1 = (0,0)
170:    if num startswith 1 or 2 (상악):
172:        if molar 6/7: pt1,f = check_multi_root(gray_img,55,"up")  # 다근 우선
174:        if not f: pt1 = get_root_pos(gray_img,3,"up")             # 루트 점
175:        pt2 = get_crown_pos(gray_img,num,5,"up")                  # 크라운 점
176:    else (하악):
178:        if molar 6/7: pt1,f = check_multi_root(gray_img,55,"lo")
180:        if not f: pt1 = get_root_pos(gray_img,3,"lo")
181:        pt2 = get_crown_pos(gray_img,num,5,"lo")
182:    return pt1, pt2                                               # 축 끝점(루트→크라운)
```
요약: 상/하악에 따라 밴드를 나누고, 루트/크라운 밴드의 x 최소·최대를 평균내어 각 끝점을 정합니다. 대구치는 연결성 검사로 다근을 우선 반영하고, 픽셀이 없을 때는 중앙 x와 경계 y를 기본값으로 사용합니다.

## PBL 계산 공식 흐름 — `backend/services/pano_calc_utils.py:238-355`
- 입력: CEJ/Bone YOLO 마스크(컨투어 → 채움), 치아 세그 결과.
- 치아별 ROI 자르기:
  1) 세그 폴리곤 bbox(`x,y,w,h`)를 구함.
  2) `left_th=-10`을 적용해 ROI를 살짝 좌측으로 확장: `x+left_th : x+w`.
  3) ROI 단위로 치아, CEJ, Bone 마스크를 슬라이스하여 계산 범위를 좁힘(속도↑, 노이즈↓).
- 축 구하기:
  - `get_principal_axis(ROI, tooth_label)`을 호출 → `(p_root, p_crown)`.
  - 축 벡터 길이 `L_axis = sqrt((Δx)^2 + (Δy)^2)`.
- 축과 마스크 겹침 길이:
  - 축을 ROI에 1px 선으로 그림 → `tooth_axis_roi`.
  - `L_bl = |tooth_axis_roi ∩ bonelevel_mask|` (축과 Bonelevel의 겹친 픽셀 길이)
  - `L_cej = |tooth_axis_roi ∩ cej_mask|` (축과 CEJ의 겹친 픽셀 길이)
  - `periodontal_to_root = L_axis - L_bl` (축 전체 길이에서 Bone 겹침을 뺀 길이)
  - `cej_to_root = L_axis - L_cej` (축 전체 길이에서 CEJ 겹침을 뺀 길이)
- PBL 계산:
  - `pbl_ratio = periodontal_to_root / cej_to_root`  
    (단, `cej_to_root <= 0`이면 해당 치아 스킵)
  - `pbl_percent = pbl_ratio * 100`
- 간단 수치 예시:
  - 축 길이 `L_axis = 20`
  - Bone 겹침 `L_bl = 8` → `periodontal_to_root = 12`
  - CEJ 겹침 `L_cej = 6` → `cej_to_root = 14`
  - `pbl_ratio = 12 / 14 ≈ 0.86`, `pbl_percent ≈ 86%`

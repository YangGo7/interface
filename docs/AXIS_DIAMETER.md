# Axis & Diameter Calculation (Implant)

## Overview
- 대상: FDI 35(implant) 감지에 대해 중심축과 직경을 계산.
- 입력: 원본 그레이스케일 이미지, 세그멘테이션 마스크(폴리곤).
- 출력: 축 벡터, 축 중심, 직경 픽셀/버킷(mm), 직경 선분 좌표.
- 주요 함수: `compute_from_gray_with_mask` → `compute_sample_axis` (`backend/utils/sample_axis_service.py`), 래퍼 `compute_diameter_for_label` (`backend/services/postprocess.py`).

## 처리 파이프라인
1) **마스크 생성**  
   - 폴리곤 → 바이너리 마스크(`polygon_to_mask`).

2) **그레이 전처리** (`compute_from_gray_with_mask`)  
   - CLAHE → 히스토그램 평활화 → 수동 임계값(thr)으로 이진화.
   - 이진화된 그레이와 마스크의 AND → 주 컨투어 선택.

3) **축 초기 추정 (PCA)** (`compute_sample_axis`)  
   - 마스크 유효 픽셀 좌표 집합 `P={p_i}`.
   - 평균 `μ = (1/N)∑p_i`, 공분산 `Σ = (1/N)∑(p_i-μ)(p_i-μ)^T`.
   - 첫 번째 고유벡터 `v_pca`를 초기 축, 중심 `c_pca=μ`로 사용.

4) **중심선 샘플링/정제**  
   - 축 직교선 여러 개 샘플 → 중간점 집합 `M` 생성(`sample_centerline_points`).  
   - 중앙 80%만 남기고(`drop=round(0.1*|M|)`), 축으로부터 외적 거리 기준 아웃라이어 제거(`filter_outliers`, 허용 비율 0.15).
   - 남은 점들에 PCA 재적용 → 축 보정(`fit_axis_pca_points`). 실패 시 초기 `v_pca` 사용.

5) **직경 계산**  
   - 축 방향 각도 `θ = atan2(v_x, v_y)`.
   - 마스크를 -θ 회전 → 수평 스캔으로 최대 연속 폭 `d_max`와 위치(`max_horizontal_run_pos`).  
   - 가능하면 역회전으로 직경 끝점 `(p1, p2)` 복원. 실패 시 축에 직교하는 폭 탐색(`max_width_perp_to_axis`).

6) **축 길이 확장/라인 생성** (`build_overlay_lines`)  
   - 축 길이: 바운딩박스 높이의 1.5배(없으면 이미지 크기 0.6배).  
   - 직경: (p1, p2) 그대로 전달.

7) **단위 변환 및 버킷팅**  
   - 픽셀 → mm: `diameter_mm = d_max * DIAMETER_PIXEL_TO_MM` (config 기본 0.1).  
   - 파이 지름: `diameter_pi = diameter_mm / π`.  
   - 버킷: `bucket_mm = round(diameter_mm * 2) / 2` (0.5 mm 스텝).

## 입력/출력 사양
- 입력: `gray`(H×W uint8), `mask`(H×W uint8, 0/255), `n_samples`(기본 40).  
- 출력: `(axis_center, axis_vec, max_d, p1, p2, mids)`  
  - `axis_center`: (x,y) float  
  - `axis_vec`: 단위 벡터 (vx, vy)  
  - `max_d`: 직경 픽셀 폭  
  - `p1,p2`: 직경 끝점 (x,y) 또는 `None`  
  - `mids`: 필터 후 중심점 리스트

## 예외/스킵 조건
- 마스크 픽셀 < 2 → 바로 `(None, None, 0, None, None, [])` 반환.
- 축 벡터 노름 < 1e-6 → 실패로 간주.
- 그레이/마스크 부재 시 직경 계산 스킵(`compute_diameter_for_label` 내부 로그 남김).

## 보안/성능 고려
- 입력은 업로드된 의료 이미지이므로 파일 경로 존재 여부만 확인; 추가 외부 접근 없음.
- 연산은 마스크 영역 한정(PCA/회전), O(N) 수준이며 한 객체당 수 ms~수십 ms 예상.

## 사용 예
```python
from services.postprocess import compute_diameter_for_label
metrics = compute_diameter_for_label(
    detections=result.detections,
    target_labels={"35"},               # FDI 35 (implant)
    target_names={"implant"},           # 이름 매칭
    label_map=label_map,                # config.LABEL_NAME_MAP
    image_path=str(temp_path),
    pixel_to_mm=app.config["DIAMETER_PIXEL_TO_MM"],
)
# metrics는 build_overlay_lines로 전달되어 프론트 오버레이 생성에 사용됨
```

## 한계/개선 포인트
- 절대 스케일은 `DIAMETER_PIXEL_TO_MM`에 의존 → DICOM 등 픽셀 스페이싱 사용한 자동 보정 필요.
- 저신뢰/잘못된 클래스 예측 시 라벨 혼동 발생 가능 → 모델 튜닝 또는 클래스별 후처리 추가 고려.

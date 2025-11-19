"""
인접 치아 간 거리 검증 모듈
"""

from typing import Dict, Any, List


def calculate_inter_tooth_distance(detections: List, img_width: int) -> Dict[str, Any]:
    """
    인접 치아 간 거리 검증

    작동 원리:
    1. 각 사분면 내에서 연속된 FDI 번호를 가진 치아 쌍 찾기 (예: #11과 #12)
    2. 두 치아의 bbox 중심 간 X축 거리 계산
    3. 거리를 이미지 너비로 정규화 (비율로 변환)
    4. 정상 범위(2~10%)를 벗어나면 이상으로 분류
    5. 평균 거리 및 이상 쌍 리스트 반환

    임계값 근거:
    - 최소 2%: 너무 가까우면 겹침 또는 오탐 가능성
    - 최대 10%: 너무 멀면 결손치 존재 또는 잘못된 라벨 가능성

    Args:
        detections: Detection 객체 리스트
        img_width: 이미지 너비 (픽셀)

    Returns:
        Dict: {
            'mean_distance_ratio': 평균 치아 간격 (이미지 너비 대비 비율),
            'abnormal_count': 이상 간격 쌍 개수,
            'total_pairs': 전체 인접 쌍 개수,
            'abnormal_pairs': 이상 간격 상세 리스트,
            'validation_rate': 정상 간격 비율 (0.0~1.0)
        }
    """
    # 사분면별로 치아 그룹화
    quadrants = {1: [], 2: [], 3: [], 4: []}

    for det in detections:
        try:
            fdi = int(det.label)
            quadrant = fdi // 10
            position = fdi % 10
            bbox_center_x = det.bounding_box.x + det.bounding_box.width / 2

            quadrants[quadrant].append({
                'fdi': fdi,
                'position': position,
                'x': bbox_center_x
            })
        except (ValueError, AttributeError):
            continue

    all_distances = []
    abnormal_distances = []

    # 정상 거리 범위 설정 (이미지 너비 대비 비율)
    # 파노라마 X-ray에서 인접 치아는 보통 전체 너비의 2~10% 거리
    min_distance_ratio = 0.02  # 2%: 이보다 가까우면 비정상
    max_distance_ratio = 0.10  # 10%: 이보다 멀면 비정상

    # 각 사분면별로 연속된 치아 쌍 검증
    for quadrant, teeth in quadrants.items():
        # FDI position 순으로 정렬 (치아 번호 순서)
        sorted_teeth = sorted(teeth, key=lambda t: t['position'])

        for i in range(len(sorted_teeth) - 1):
            tooth1 = sorted_teeth[i]
            tooth2 = sorted_teeth[i + 1]

            # 연속된 치아인지 확인 (position 차이가 1)
            # 예: #11(pos=1)과 #12(pos=2)는 연속, #11과 #13은 비연속
            if tooth2['position'] - tooth1['position'] == 1:
                # X축 거리 계산 (절대값)
                distance = abs(tooth2['x'] - tooth1['x'])
                distance_ratio = distance / img_width

                all_distances.append(distance_ratio)

                # 정상 범위 밖이면 이상 쌍으로 기록
                if distance_ratio < min_distance_ratio:
                    abnormal_distances.append({
                        'tooth1': tooth1['fdi'],
                        'tooth2': tooth2['fdi'],
                        'distance_ratio': distance_ratio,
                        'distance_px': distance,
                        'reason': 'too_close'  # 너무 가까움
                    })
                elif distance_ratio > max_distance_ratio:
                    abnormal_distances.append({
                        'tooth1': tooth1['fdi'],
                        'tooth2': tooth2['fdi'],
                        'distance_ratio': distance_ratio,
                        'distance_px': distance,
                        'reason': 'too_far'  # 너무 멀음
                    })

    # 평균 거리 계산
    mean_distance = sum(all_distances) / len(all_distances) if all_distances else 0.0

    # 검증 성공률 (정상 범위 내 쌍의 비율)
    validation_rate = 1 - (len(abnormal_distances) / len(all_distances)) if all_distances else 0.0

    return {
        'mean_distance_ratio': mean_distance,
        'abnormal_count': len(abnormal_distances),
        'total_pairs': len(all_distances),
        'abnormal_pairs': abnormal_distances,
        'validation_rate': validation_rate
    }

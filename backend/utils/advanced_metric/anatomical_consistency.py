"""
해부학적 일관성 점수 계산 모듈
"""

from typing import Dict, Any, List
from .bbox_iou import calculate_bbox_iou


def calculate_anatomical_consistency(detections: List) -> Dict[str, Any]:
    """
    해부학적 일관성 점수 계산

    작동 원리:
    1. 상악이 하악보다 위에 있는지 확인 (Y좌표 비교)
    2. 좌우 치아 개수 대칭성 평가
    3. 치아 bbox 겹침 여부 검사 (IoU 계산)
    4. 각 체크 결과를 종합하여 일관성 점수 산출

    검증 항목:
    - upper_above_lower: 상악(1x,2x) 평균 Y < 하악(3x,4x) 평균 Y
    - left_right_symmetry: 좌측 치아 수 / 우측 치아 수 비율
    - no_overlap: bbox IoU > 0.3인 쌍이 없어야 함

    Args:
        detections: Detection 객체 리스트

    Returns:
        Dict: {
            'anatomical_consistency_score': 종합 일관성 점수 (0.0~1.0),
            'checks': 개별 체크 통과 여부,
            'symmetry_ratio': 좌우 대칭 비율,
            'overlap_count': 겹치는 bbox 쌍 개수
        }
    """
    checks = {
        'upper_above_lower': True,
        'left_right_symmetry': True,
        'no_overlap': True
    }

    scores = []

    # 1. 상악/하악 위치 검증
    # 상악 치아 (FDI 1x, 2x): 이미지 상단에 위치해야 함
    upper_teeth = [d for d in detections if int(d.label) // 10 in [1, 2]]
    # 하악 치아 (FDI 3x, 4x): 이미지 하단에 위치해야 함
    lower_teeth = [d for d in detections if int(d.label) // 10 in [3, 4]]

    if upper_teeth and lower_teeth:
        # 각 그룹의 평균 Y좌표 계산 (Y축: 위=0, 아래=큰값)
        avg_upper_y = sum(d.bounding_box.y for d in upper_teeth) / len(upper_teeth)
        avg_lower_y = sum(d.bounding_box.y for d in lower_teeth) / len(lower_teeth)

        # 상악이 하악보다 Y값이 작아야 정상 (위쪽에 위치)
        checks['upper_above_lower'] = avg_upper_y < avg_lower_y
        scores.append(1.0 if checks['upper_above_lower'] else 0.0)

    # 2. 좌우 대칭성 검증
    # 좌측 치아 (FDI 2x, 3x): 환자 왼쪽
    left_count = len([d for d in detections if int(d.label) // 10 in [2, 3]])
    # 우측 치아 (FDI 1x, 4x): 환자 오른쪽
    right_count = len([d for d in detections if int(d.label) // 10 in [1, 4]])

    if max(left_count, right_count) > 0:
        # 대칭 비율 계산 (0.0~1.0, 1.0이 완전 대칭)
        symmetry_ratio = min(left_count, right_count) / max(left_count, right_count)

        # 80% 이상 대칭이면 정상으로 판단
        checks['left_right_symmetry'] = symmetry_ratio >= 0.8
        scores.append(symmetry_ratio)
    else:
        symmetry_ratio = 0.0
        checks['left_right_symmetry'] = False
        scores.append(0.0)

    # 3. 치아 겹침 검증
    # 정상적으로는 치아 bbox가 크게 겹치지 않아야 함
    overlap_count = 0
    overlap_threshold = 0.3  # IoU 30% 이상이면 이상으로 판단

    for i, det1 in enumerate(detections):
        for det2 in detections[i+1:]:
            # 두 bbox의 IoU 계산
            iou = calculate_bbox_iou(det1.bounding_box, det2.bounding_box)

            if iou > overlap_threshold:
                overlap_count += 1

    checks['no_overlap'] = overlap_count == 0
    # 겹침 개수에 따라 점수 감점 (겹침 1개당 -0.1)
    overlap_score = max(0.0, 1.0 - overlap_count * 0.1)
    scores.append(overlap_score)

    # 종합 일관성 점수 계산 (모든 체크의 평균)
    consistency_score = sum(scores) / len(scores) if scores else 0.0

    return {
        'anatomical_consistency_score': consistency_score,
        'checks': checks,
        'symmetry_ratio': symmetry_ratio,
        'overlap_count': overlap_count
    }

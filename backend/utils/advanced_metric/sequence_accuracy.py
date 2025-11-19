"""
치아 순서 정확도 계산 모듈
"""

from typing import Dict, Any, List


def calculate_sequence_accuracy(detections: List) -> Dict[str, Any]:
    """
    치아 순서 정확도 계산

    작동 원리:
    1. 탐지된 치아들을 사분면(Quadrant)별로 그룹화
    2. 각 사분면 내에서 치아들을 X좌표(bbox 중심) 기준으로 정렬
    3. 인접한 치아 쌍의 FDI 번호 순서가 X좌표 순서와 일치하는지 확인
    4. 정확한 쌍의 비율을 계산하여 순서 정확도 산출

    Args:
        detections: Detection 객체 리스트

    Returns:
        Dict: {
            'sequence_accuracy': 전체 순서 정확도 (0.0~1.0),
            'correct_pairs': 순서가 올바른 인접 쌍 개수,
            'total_pairs': 전체 인접 쌍 개수,
            'quadrant_details': 사분면별 상세 결과
        }
    """
    # 사분면별로 치아 그룹화 (1=UR, 2=UL, 3=LL, 4=LR)
    quadrants = {1: [], 2: [], 3: [], 4: []}

    for det in detections:
        try:
            fdi = int(det.label)
            quadrant = fdi // 10  # 11→1, 23→2, 35→3, 47→4
            position = fdi % 10   # 11→1, 23→3, 35→5

            # bbox 중심 X좌표 계산 (파노라마에서 좌우 위치)
            bbox_center_x = det.bounding_box.x + det.bounding_box.width / 2

            quadrants[quadrant].append({
                'fdi': fdi,
                'position': position,
                'x': bbox_center_x
            })
        except (ValueError, AttributeError):
            continue  # 유효하지 않은 라벨은 스킵

    total_pairs = 0
    correct_pairs = 0
    quadrant_details = {}

    # 각 사분면별로 순서 검증
    for quadrant, teeth in quadrants.items():
        if len(teeth) < 2:
            # 치아가 2개 미만이면 순서 비교 불가
            continue

        # X좌표 기준으로 정렬 (파노라마에서 왼쪽→오른쪽)
        sorted_by_x = sorted(teeth, key=lambda t: t['x'])

        quadrant_correct = 0
        quadrant_total = 0

        # 인접한 치아 쌍을 순차적으로 비교
        for i in range(len(sorted_by_x) - 1):
            tooth1 = sorted_by_x[i]
            tooth2 = sorted_by_x[i + 1]

            # FDI 순서가 X좌표 순서와 일치하는지 확인
            # 예: #11(pos=1)이 #12(pos=2)보다 왼쪽에 있어야 함
            # 파노라마 방향: 우측 사분면(1,4)과 좌측 사분면(2,3)의 증가 방향이 다름

            if quadrant in [1, 4]:  # 우측 사분면 (1x, 4x)
                # 환자의 우측: position이 작을수록 중심선에 가까움
                # 파노라마 좌측에 표시됨 (X좌표가 작음)
                expected_order = tooth1['position'] <= tooth2['position']
            else:  # 좌측 사분면 (2x, 3x)
                # 환자의 좌측: position이 작을수록 중심선에 가까움
                # 파노라마 우측에 표시됨 (X좌표가 큼)
                expected_order = tooth1['position'] >= tooth2['position']

            quadrant_total += 1
            total_pairs += 1

            if expected_order:
                quadrant_correct += 1
                correct_pairs += 1

        # 사분면별 결과 저장
        quadrant_details[quadrant] = {
            'correct': quadrant_correct,
            'total': quadrant_total,
            'accuracy': quadrant_correct / quadrant_total if quadrant_total > 0 else 0.0
        }

    # 전체 순서 정확도 계산
    accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0.0

    return {
        'sequence_accuracy': accuracy,
        'correct_pairs': correct_pairs,
        'total_pairs': total_pairs,
        'quadrant_details': quadrant_details
    }

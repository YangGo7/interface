"""
신뢰도 가중 정확도 계산 모듈
"""

from typing import Dict, Any, List


def calculate_confidence_weighted_accuracy(detections: List, gt_data: List, img_width: int, img_height: int) -> Dict[str, Any]:
    """
    신뢰도 가중 정확도 계산

    작동 원리:
    1. 각 detection에 대해 GT와 매칭하여 정확성 판단 (0 or 1)
    2. 정확성에 confidence를 곱하여 가중치 부여
    3. 모든 가중 정확성의 합을 전체 confidence 합으로 나눔

    의미:
    - 일반 Accuracy: 맞은 개수 / 전체 개수
    - Confidence-Weighted: 높은 신뢰도 예측을 더 중요하게 평가
    - 낮은 confidence 예측이 틀려도 점수에 미치는 영향 적음

    Args:
        detections: Detection 객체 리스트
        gt_data: Ground Truth 데이터 리스트
        img_width: 이미지 너비
        img_height: 이미지 높이

    Returns:
        Dict: {
            'confidence_weighted_accuracy': 가중 정확도 (0.0~1.0),
            'total_confidence': 전체 confidence 합
        }
    """
    if not gt_data or len(gt_data) == 0:
        return {
            'confidence_weighted_accuracy': 0.0,
            'total_confidence': 0.0
        }

    from utils.gt_comparison import find_best_gt_match

    weighted_correct = 0.0  # 신뢰도 가중 정확 점수
    total_confidence = 0.0  # 전체 신뢰도 합

    for det in detections:
        # GT와 매칭
        match_result = find_best_gt_match(det, gt_data, img_width, img_height)

        # 정확성 판단 (IoU ≥ 0.5 이고 라벨 일치하면 1, 아니면 0)
        is_correct = 0
        if match_result and match_result['iou'] >= 0.5 and match_result['label_match']:
            is_correct = 1

        # confidence로 가중치 부여
        # 예: confidence=0.9이고 정답이면 +0.9, 틀리면 +0.0
        weighted_correct += is_correct * det.confidence
        total_confidence += det.confidence

    # 가중 평균 계산
    weighted_accuracy = weighted_correct / total_confidence if total_confidence > 0 else 0.0

    return {
        'confidence_weighted_accuracy': weighted_accuracy,
        'total_confidence': total_confidence
    }

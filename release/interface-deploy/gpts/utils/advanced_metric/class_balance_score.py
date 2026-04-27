"""
클래스 균형도 점수 계산 모듈
"""

from typing import Dict, Any, List
from collections import defaultdict
import math


def calculate_class_balance_score(detections: List, gt_data: List) -> Dict[str, Any]:
    """
    클래스 균형도 점수 계산

    작동 원리:
    1. GT 데이터에서 각 클래스(치아)별로 실제 개수 집계
    2. detection에서 각 클래스별로 탐지된 개수 집계
    3. 클래스별 Recall 계산 (탐지 개수 / GT 개수)
    4. 모든 클래스 Recall의 표준편차 계산
    5. 표준편차가 낮을수록 균형적 (모든 치아를 골고루 잘 탐지)

    의미:
    - 높은 균형도: 모든 치아 종류를 비슷한 성능으로 탐지
    - 낮은 균형도: 특정 치아(예: 앞니)만 잘 탐지하고 어금니는 못 찾음

    Args:
        detections: Detection 객체 리스트
        gt_data: Ground Truth 데이터 리스트

    Returns:
        Dict: {
            'class_balance_score': 균형 점수 (0.0~1.0, 1.0이 완전 균형),
            'class_recall_std': 클래스별 Recall 표준편차,
            'mean_class_recall': 평균 클래스 Recall
        }
    """
    if not gt_data or len(gt_data) == 0:
        return {
            'class_balance_score': 0.0,
            'class_recall_std': 0.0,
            'mean_class_recall': 0.0
        }

    # GT를 클래스(FDI 번호)별로 그룹화
    gt_by_class = defaultdict(int)
    for gt in gt_data:
        gt_by_class[gt['label']] += 1

    # Detection을 클래스별로 그룹화
    det_by_class = defaultdict(int)
    for det in detections:
        det_by_class[det.label] += 1

    # 클래스별 Recall 계산
    class_recalls = []
    for class_label, gt_count in gt_by_class.items():
        det_count = det_by_class.get(class_label, 0)

        # Recall = 탐지 개수 / GT 개수 (단, 1.0 초과 방지)
        recall = min(det_count / gt_count, 1.0) if gt_count > 0 else 0.0
        class_recalls.append(recall)

    # 통계 계산
    if class_recalls:
        # 평균 Recall
        mean_recall = sum(class_recalls) / len(class_recalls)

        # 분산 계산: Σ(recall - mean)² / N
        variance = sum((r - mean_recall) ** 2 for r in class_recalls) / len(class_recalls)

        # 표준편차 계산
        std_dev = math.sqrt(variance)

        # 균형 스코어 계산 (1 - 정규화된 표준편차)
        # 표준편차는 최대 1.0 (모든 recall이 0 or 1로 극단적 분산)
        # 표준편차가 0이면 완벽한 균형 (점수 1.0)
        # 표준편차가 1이면 최악의 불균형 (점수 0.0)
        balance_score = max(0.0, 1.0 - std_dev)
    else:
        mean_recall = 0.0
        std_dev = 0.0
        balance_score = 0.0

    return {
        'class_balance_score': balance_score,
        'class_recall_std': std_dev,
        'mean_class_recall': mean_recall
    }

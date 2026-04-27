"""
Bounding Box IoU 계산 유틸리티
"""

def calculate_bbox_iou(bbox1, bbox2) -> float:
    """
    두 bounding box 간 IoU(Intersection over Union) 계산

    Args:
        bbox1, bbox2: {x, y, width, height} 형태의 bbox

    Returns:
        float: IoU 값 (0.0~1.0)
    """
    # bbox1의 경계
    x1_min = bbox1.x
    y1_min = bbox1.y
    x1_max = bbox1.x + bbox1.width
    y1_max = bbox1.y + bbox1.height

    # bbox2의 경계
    x2_min = bbox2.x
    y2_min = bbox2.y
    x2_max = bbox2.x + bbox2.width
    y2_max = bbox2.y + bbox2.height

    # 교집합 영역 계산
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 교집합 넓이
    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        intersection = 0.0

    # 각 bbox 넓이
    area1 = bbox1.width * bbox1.height
    area2 = bbox2.width * bbox2.height

    # 합집합 넓이 (A + B - 교집합)
    union = area1 + area2 - intersection

    # IoU 계산
    return intersection / union if union > 0 else 0.0

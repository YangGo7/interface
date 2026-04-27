"""
GT(Ground Truth) 비교 유틸리티
YOLO TXT 파일 파싱, IoU 계산, 매칭 및 색상 결정
"""

from typing import List, Dict, Any, Optional, Tuple
from config import Config


def parse_gt_file(file_path: str) -> List[Dict[str, Any]]:
    """
    YOLO 포맷의 GT TXT 파일 파싱
    - Segmentation 포맷: class_id x1 y1 x2 y2 x3 y3 ... (normalized polygon)
    - BBox 포맷: class_id x_center y_center width height (normalized)

    NOTE: class_id는 YOLO 원본 형식 (0-31) 또는 FDI 번호 (11-48) 모두 지원합니다.
          YOLO 형식일 경우 자동으로 FDI 번호로 변환합니다.

    Args:
        file_path: GT 파일 경로

    Returns:
        List of GT objects:
        [
            {
                "label": "11",  # FDI 치아 번호
                "polygon": [[x1, y1], [x2, y2], ...],  # segmentation인 경우
                "x_center": 0.5,  # bbox인 경우
                "y_center": 0.5,
                "width": 0.1,
                "height": 0.15
            },
            ...
        ]
    """
    gt_objects = []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # class_id를 FDI 번호로 변환
            raw_class_id = parts[0]
            try:
                class_id_int = int(raw_class_id)
                # 0-31 범위면 YOLO 형식 → FDI 변환
                if 0 <= class_id_int <= 31:
                    fdi_number = Config.CLASS_ID_TO_FDI.get(class_id_int, class_id_int)
                    label = str(fdi_number)
                else:
                    # 이미 FDI 형식 (11-48)
                    label = raw_class_id
            except ValueError:
                # 숫자가 아니면 그대로 사용
                label = raw_class_id

            # Segmentation 포맷 (polygon): class_id x1 y1 x2 y2 ...
            if len(parts) > 5:
                # polygon 좌표 추출
                coords = [float(p) for p in parts[1:]]

                # x, y 좌표를 쌍으로 묶기
                polygon = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        polygon.append([coords[i], coords[i+1]])

                if len(polygon) >= 3:  # 최소 3개 점 필요
                    gt_objects.append({
                        "label": label,  # FDI 번호
                        "polygon": polygon,  # normalized polygon coordinates
                        "type": "segmentation"
                    })

            # BBox 포맷: class_id x_center y_center width height
            elif len(parts) == 5:
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                gt_objects.append({
                    "label": label,  # FDI 번호
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "type": "bbox"
                })

    except Exception as e:
        print(f"⚠️ Failed to parse GT file: {e}")
        return []

    return gt_objects


def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    두 박스 간 IoU(Intersection over Union) 계산

    Args:
        box1: {"x": x, "y": y, "width": w, "height": h} (pixel 좌표)
        box2: {"x": x, "y": y, "width": w, "height": h} (pixel 좌표)

    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = box1["x"] + box1["width"]
    y1_max = box1["y"] + box1["height"]

    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = box2["x"] + box2["width"]
    y2_max = box2["y"] + box2["height"]

    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    intersection = inter_width * inter_height

    # Union
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def polygon_to_bbox(polygon: List[List[float]], img_width: int, img_height: int) -> Dict[str, float]:
    """
    Normalized polygon을 pixel bounding box로 변환

    Args:
        polygon: [[x, y], ...] normalized coordinates (0.0 ~ 1.0)
        img_width: 이미지 너비
        img_height: 이미지 높이

    Returns:
        {"x": x_min, "y": y_min, "width": w, "height": h} pixel 좌표
    """
    # Normalized → Pixel 좌표 변환
    pixel_coords = [[p[0] * img_width, p[1] * img_height] for p in polygon]

    # Bounding box 계산
    x_coords = [p[0] for p in pixel_coords]
    y_coords = [p[1] for p in pixel_coords]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min
    }


def find_best_gt_match(
    detection: Any,
    gt_data: List[Dict[str, Any]],
    img_width: int,
    img_height: int
) -> Optional[Dict[str, Any]]:
    """
    Detection과 가장 잘 매칭되는 GT 객체 찾기
    Segmentation 및 BBox 포맷 모두 지원

    Args:
        detection: Detection 객체
        gt_data: GT 객체 리스트
        img_width: 이미지 너비
        img_height: 이미지 높이

    Returns:
        {
            "gt_label": "11",
            "iou": 0.85,
            "label_match": True
        }
        또는 None (매칭 없음)
    """
    if not gt_data or len(gt_data) == 0:
        return None

    best_match = None
    best_iou = 0.0

    # Detection 박스 (pixel 좌표)
    det_box = {
        "x": detection.bounding_box.x,
        "y": detection.bounding_box.y,
        "width": detection.bounding_box.width,
        "height": detection.bounding_box.height
    }

    for gt in gt_data:
        # GT 타입에 따라 처리
        if gt.get("type") == "segmentation":
            # Polygon → BBox 변환
            gt_box = polygon_to_bbox(gt["polygon"], img_width, img_height)
        else:
            # BBox 포맷 (normalized → pixel)
            gt_box = {
                "x": (gt["x_center"] - gt["width"] / 2) * img_width,
                "y": (gt["y_center"] - gt["height"] / 2) * img_height,
                "width": gt["width"] * img_width,
                "height": gt["height"] * img_height
            }

        iou = calculate_iou(det_box, gt_box)

        if iou > best_iou:
            best_iou = iou
            best_match = {
                "gt_label": gt["label"],
                "iou": iou,
                # FDI 번호로 비교 (detection.label과 gt["label"] 모두 FDI 형식)
                "label_match": str(detection.label) == str(gt["label"])
            }

    return best_match


def get_color_by_match_quality(match_result: Optional[Dict[str, Any]]) -> str:
    """
    GT 매칭 품질에 따라 색상 결정

    Args:
        match_result: find_best_gt_match() 결과

    Returns:
        Hex 색상 코드:
        - #00FF00 (초록): IoU ≥ 0.8 & 라벨 일치
        - #FFFF00 (노랑): IoU < 0.8 또는 라벨 불일치
        - #FF0000 (빨강): 둘 다 틀림
    """
    if not match_result:
        return "#00FF00"  # GT 없으면 기본 초록

    iou = match_result["iou"]
    label_match = match_result["label_match"]

    if iou >= 0.8 and label_match:
        return "#00FF00"  # 초록 (정확)
    elif iou >= 0.8 or label_match:
        return "#FFFF00"  # 노랑 (부분 일치)
    else:
        return "#FF0000"  # 빨강 (불일치)

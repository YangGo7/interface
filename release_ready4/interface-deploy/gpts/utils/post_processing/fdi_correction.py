"""
FDI Template-Based Order Correction
치아 번호 순서 교정 후처리 로직
"""

from typing import List, Dict, Any, Optional
import numpy as np

# FDI Templates
UPPER_TEMPLATE = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_TEMPLATE = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


def get_fdi_from_class(cls_id: int) -> int:
    """
    Convert class ID (0-31) to FDI number

    Args:
        cls_id: YOLO class ID (0-31)

    Returns:
        FDI number (11-48 or 91 for supernumerary)
    """
    fdi_map = {}
    # Q1: Class 0-7 → FDI 11-18 (증가)
    for i in range(8):
        fdi_map[i] = 11 + i
    # Q2: Class 8-15 → FDI 21-28 (증가)
    for i in range(8):
        fdi_map[8 + i] = 21 + i
    # Q3: Class 16-23 → FDI 31-38 (증가)
    for i in range(8):
        fdi_map[16 + i] = 31 + i
    # Q4: Class 24-31 → FDI 41-48 (증가)
    for i in range(8):
        fdi_map[24 + i] = 41 + i

    # Special: Class 32 → FDI 91 (과잉치)
    fdi_map[32] = 91

    return fdi_map.get(cls_id, 0)


def get_class_from_fdi(fdi: int) -> int:
    """
    Convert FDI number to class ID (0-31)

    Args:
        fdi: FDI number (11-48 or 91)

    Returns:
        YOLO class ID (0-31 or 32 for supernumerary)
    """
    class_map = {}
    # Q1: FDI 11-18 → Class 0-7
    for i in range(8):
        class_map[11 + i] = i
    # Q2: FDI 21-28 → Class 8-15
    for i in range(8):
        class_map[21 + i] = 8 + i
    # Q3: FDI 31-38 → Class 16-23
    for i in range(8):
        class_map[31 + i] = 16 + i
    # Q4: FDI 41-48 → Class 24-31
    for i in range(8):
        class_map[41 + i] = 24 + i

    # Special: FDI 91 → Class 32
    class_map[91] = 32

    return class_map.get(fdi, 32)  # Default to 32 if unknown


def apply_fdi_template_correction(detections: List[Any],
                                  confidence_threshold: float = 0.7) -> List[Any]:
    """
    Apply FDI template-based order correction.
    Separates upper and lower arches and applies templates independently.
    Only applies correction to detections with confidence < threshold.

    Args:
        detections: List of Detection objects or [x1, y1, x2, y2, conf, cls_id] arrays
        confidence_threshold: Confidence threshold for applying correction (default: 0.7)

    Returns:
        corrected_detections: List with corrected labels/class IDs
    """
    if len(detections) == 0:
        return detections

    # Check if detections are objects or arrays
    is_object = hasattr(detections[0], 'bounding_box')

    # Separate high and low confidence detections
    high_conf_dets = []
    low_conf_dets = []

    for det in detections:
        if is_object:
            conf = det.confidence
            if conf >= confidence_threshold:
                high_conf_dets.append(det)
            else:
                low_conf_dets.append(det)
        else:
            # Array format: [x1, y1, x2, y2, conf, cls_id]
            conf = det[4]
            if conf >= confidence_threshold:
                high_conf_dets.append(det)
            else:
                low_conf_dets.append(det)

    # Only apply correction to low confidence detections
    if len(low_conf_dets) == 0:
        return detections

    # Convert low confidence detections to internal format
    teeth = []
    for det in low_conf_dets:
        if is_object:
            x1 = det.bounding_box.x
            y1 = det.bounding_box.y
            x2 = det.bounding_box.x + det.bounding_box.width
            y2 = det.bounding_box.y + det.bounding_box.height
            fdi = int(det.label)
            cls_id = det.class_id
        else:
            x1, y1, x2, y2, conf, cls_id = det
            fdi = get_fdi_from_class(int(cls_id))

        cx = (x1 + x2) / 2
        teeth.append({
            'det': det,
            'fdi': fdi,
            'cx': cx,
            'cls_id': int(cls_id),
            'is_object': is_object
        })

    # Separate upper and lower arches
    upper_teeth = [t for t in teeth if 11 <= t['fdi'] <= 28]
    lower_teeth = [t for t in teeth if 31 <= t['fdi'] <= 48]
    other_teeth = [t for t in teeth if t['fdi'] not in range(11, 29) and t['fdi'] not in range(31, 49)]

    corrected_low_conf = []

    # Process upper arch
    if len(upper_teeth) > 0:
        # Sort by X-coordinate (left to right)
        upper_teeth.sort(key=lambda t: t['cx'])

        # Get detected FDIs and sort according to template
        detected_upper_fdis = set(t['fdi'] for t in upper_teeth)
        upper_template_sequence = [fdi for fdi in UPPER_TEMPLATE if fdi in detected_upper_fdis]

        # Force-assign template FDI numbers
        for i, tooth in enumerate(upper_teeth):
            if i < len(upper_template_sequence):
                new_fdi = upper_template_sequence[i]
                new_cls_id = get_class_from_fdi(new_fdi)

                if tooth['is_object']:
                    # Update Detection object
                    det = tooth['det']
                    det.label = str(new_fdi)
                    det.class_id = new_cls_id
                    corrected_low_conf.append(det)
                else:
                    # Update array
                    det = tooth['det']
                    corrected_det = [det[0], det[1], det[2], det[3], det[4], float(new_cls_id)]
                    corrected_low_conf.append(corrected_det)
            else:
                corrected_low_conf.append(tooth['det'])

    # Process lower arch
    if len(lower_teeth) > 0:
        # Sort by X-coordinate (left to right)
        lower_teeth.sort(key=lambda t: t['cx'])

        # Get detected FDIs and sort according to template
        detected_lower_fdis = set(t['fdi'] for t in lower_teeth)
        lower_template_sequence = [fdi for fdi in LOWER_TEMPLATE if fdi in detected_lower_fdis]

        # Force-assign template FDI numbers
        for i, tooth in enumerate(lower_teeth):
            if i < len(lower_template_sequence):
                new_fdi = lower_template_sequence[i]
                new_cls_id = get_class_from_fdi(new_fdi)

                if tooth['is_object']:
                    # Update Detection object
                    det = tooth['det']
                    det.label = str(new_fdi)
                    det.class_id = new_cls_id
                    corrected_low_conf.append(det)
                else:
                    # Update array
                    det = tooth['det']
                    corrected_det = [det[0], det[1], det[2], det[3], det[4], float(new_cls_id)]
                    corrected_low_conf.append(corrected_det)
            else:
                corrected_low_conf.append(tooth['det'])

    # Keep other teeth (supernumerary, etc.) as is
    for tooth in other_teeth:
        corrected_low_conf.append(tooth['det'])

    # Merge high and low confidence detections
    return high_conf_dets + corrected_low_conf


def apply_spatial_ordering(detections: List[Any]) -> List[Any]:
    """
    Apply spatial ordering based on X-coordinates within each quadrant.
    This ensures teeth are ordered left-to-right correctly.

    Args:
        detections: List of Detection objects

    Returns:
        Spatially ordered detections
    """
    if len(detections) == 0:
        return detections

    # Group by quadrant
    quadrants = {
        'upper_right': [],  # 11-18
        'upper_left': [],   # 21-28
        'lower_left': [],   # 31-38
        'lower_right': [],  # 41-48
        'other': []
    }

    is_object = hasattr(detections[0], 'bounding_box')

    for det in detections:
        if is_object:
            fdi = int(det.label)
            cx = det.bounding_box.x + det.bounding_box.width / 2
        else:
            fdi = get_fdi_from_class(int(det[5]))
            cx = (det[0] + det[2]) / 2

        if 11 <= fdi <= 18:
            quadrants['upper_right'].append((cx, det))
        elif 21 <= fdi <= 28:
            quadrants['upper_left'].append((cx, det))
        elif 31 <= fdi <= 38:
            quadrants['lower_left'].append((cx, det))
        elif 41 <= fdi <= 48:
            quadrants['lower_right'].append((cx, det))
        else:
            quadrants['other'].append((cx, det))

    # Sort each quadrant by X-coordinate
    ordered_detections = []
    for quad_name, quad_dets in quadrants.items():
        if quad_name == 'other':
            # Don't sort 'other' category
            ordered_detections.extend([det for _, det in quad_dets])
        else:
            # Sort by X-coordinate
            sorted_dets = sorted(quad_dets, key=lambda x: x[0])
            ordered_detections.extend([det for _, det in sorted_dets])

    return ordered_detections

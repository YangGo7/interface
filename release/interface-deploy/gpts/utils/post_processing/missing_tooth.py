from typing import List, Dict, Any


def MissingToothFinder(detections: List[Any], tooth_system: str = "fdi") -> Dict[str, Any]:
    """Identify missing teeth from detection labels using the FDI system."""

    full_tooth_set = {
        11, 12, 13, 14, 15, 16, 17, 18,
        21, 22, 23, 24, 25, 26, 27, 28,
        31, 32, 33, 34, 35, 36, 37, 38,
        41, 42, 43, 44, 45, 46, 47, 48,
    }

    detected_set = set()
    for det in detections:
        try:
            if str(det.label).isdigit():
                detected_set.add(int(det.label))
        except ValueError:
            continue

    missing_set = full_tooth_set - detected_set

    detected_list = sorted(detected_set)
    missing_list = sorted(missing_set)

    return {
        "detected": detected_list,
        "missing": missing_list,
        "missing_count": len(missing_list),
        "status": "Missing Found" if missing_list else "Full Dentition",
    }
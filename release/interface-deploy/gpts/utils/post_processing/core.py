import cv2
from typing import List, Any, Tuple, Optional


class BaseProcessor:
    """Shared utilities for post-processing steps."""

    def __init__(self, image_path: str, detections: List[Any]):
        self.image_path = image_path
        self.detections = detections

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.height, self.width = self.image.shape[:2]

    def validate_box(self, det) -> Tuple[int, int, int, int, str, Optional[List]]:
        """
        Clamp a detection box to the image boundary and extract metadata.

        Returns:
            x1, y1, x2, y2: Clamped box coordinates
            label: Detection label (string)
            mask: Optional segmentation mask points
        """
        x, y = int(det.bounding_box.x), int(det.bounding_box.y)
        w, h = int(det.bounding_box.width), int(det.bounding_box.height)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + w)
        y2 = min(self.height, y + h)

        label = getattr(det, "label", "unknown")

        mask = None
        if det.segmentation_mask and det.segmentation_mask.counts:
            mask = det.segmentation_mask.counts

        return x1, y1, x2, y2, label, mask
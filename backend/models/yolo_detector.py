"""
YOLO Detector Implementation
YOLOv8/YOLO11 기반 객체 탐지 구현 (Polygon Mask 지원 추가)
"""

import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Optional
import random
from config import Config

from .base_detector import BaseDetector
from .schemas import (
    DetectionResponse,
    Detection,
    BoundingBox,
    SegmentationMask,
    ModelMetrics,
    ModelInfo,
    ImageInfo
)


class YOLODetector(BaseDetector):
    """
    YOLO 기반 객체 탐지기
    YOLOv8-seg, YOLO11-seg 지원
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.25, device: str = 'cpu'):
        """
        Args:
            model_path: YOLO 모델 파일 경로 (.pt 파일)
            confidence_threshold: 최소 신뢰도 (0.0 ~ 1.0)
            device: 'cpu' 또는 'cuda' (GPU)
        """
        super().__init__(model_path, confidence_threshold)
        self.device = device
        self.color_map = {}  # 클래스별 색상 캐싱

    def load_model(self):
        """YOLO 모델 로드"""
        print(f"📦 Loading YOLO model from {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def predict(self, image_path: str, iou_threshold: float = 0.45, classes: Optional[List[int]]= None, imgsz = 1280 , retina_masks : bool = True,**kwargs) -> DetectionResponse:
        """
        이미지에서 객체 탐지 수행

        Args:
            image_path: 입력 이미지 경로
            iou_threshold: NMS IoU 임계값
            **kwargs: 추가 옵션

        Returns:
            DetectionResponse: 탐지 결과
        """
        if not self.is_loaded():
            self.load_model()

        # ==================== 시간 측정 시작 ====================
        total_start = time.time()

        # 1. 전처리 (이미지 로드)
        preprocess_start = time.time()
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img_height, img_width = image.shape[:2]
        image_format = Path(image_path).suffix[1:]  # .jpg -> jpg
        preprocess_time = (time.time() - preprocess_start) * 1000

        # 2. 추론 (YOLO 실행)
        inference_start = time.time()
        results = self.model(
            image_path,
            conf=self.confidence_threshold,
            iou=iou_threshold,
            imgsz = imgsz,
            retina_masks = retina_masks , 
            verbose=False  # 로그 출력 억제
        )
        inference_time = (time.time() - inference_start) * 1000

        # 3. 후처리 (결과 파싱)
        postprocess_start = time.time()
        detections = self._parse_results(results[0], img_width, img_height)
        postprocess_time = (time.time() - postprocess_start) * 1000
        total_time = (time.time() - total_start) * 1000


        # ==================== 응답 생성 ====================
        response = DetectionResponse(
            success=True,
            message=f"Detected {len(detections)} object(s)",
            detections=detections,
            metrics=ModelMetrics(
                preprocessing_time_ms=round(preprocess_time, 2),
                inference_time_ms=round(inference_time, 2),
                postprocessing_time_ms=round(postprocess_time, 2),
                total_time_ms=round(total_time, 2)
            ),
            model_info=self.get_model_info(),
            image_info=ImageInfo(
                width=img_width,
                height=img_height,
                format=image_format
            )
        )

        return response

    def _parse_results(self, result, img_width: int, img_height: int) -> List[Detection]:
        """
        YOLO 결과를 Detection 객체 리스트로 변환
        class_id (0-31)를 FDI 치아 번호 (11-48)로 자동 변환합니다.

        Args:
            result: YOLO results[0]
            img_width: 이미지 너비
            img_height: 이미지 높이

        Returns:
            List[Detection]: 탐지 결과 리스트 (label은 FDI 형식)
        """
        detections = []

        # 탐지된 객체가 없으면 빈 리스트 반환
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # YOLO 출력 파싱
        boxes = result.boxes.xyxy.cpu().numpy()  # [[x1, y1, x2, y2], ...]
        confidences = result.boxes.conf.cpu().numpy()  # [conf1, conf2, ...]
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # [cls1, cls2, ...]

        # Segmentation mask (있을 경우)
        has_masks = result.masks is not None

        # 각 탐지 객체 처리
        for idx, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            # 바운딩 박스 좌표 변환: [x1, y1, x2, y2] -> [x, y, width, height]
            x1, y1, x2, y2 = box
            bbox = BoundingBox(
                x=float(x1),
                y=float(y1),
                width=float(x2 - x1),
                height=float(y2 - y1)
            )

            # FDI 치아 번호로 변환 (class_id 0-31 -> FDI 11-48)
            fdi_number = Config.CLASS_ID_TO_FDI.get(cls_id, cls_id)
            label = str(fdi_number)

            # 색상 할당 (클래스별로 고정 색상)
            color = self._get_color_for_class(cls_id)

            # Segmentation mask 처리
            seg_mask = None
            if has_masks:
                seg_mask = self._extract_mask(result.masks.data[idx], img_width, img_height)

            # Detection 객체 생성
            detection = Detection(
                id=idx,
                label=label,
                class_id=int(cls_id),
                confidence=float(conf),
                bounding_box=bbox,
                segmentation_mask=seg_mask,
                color=color
            )

            detections.append(detection)

        return detections

    def _extract_mask(self, mask_tensor, img_width: int, img_height: int) -> Optional[SegmentationMask]:
        """
        YOLO mask tensor를 Polygon(좌표 리스트) 형식으로 변환
        (프론트엔드에서 쉽게 그리기 위함)

        Args:
            mask_tensor: YOLO mask (torch.Tensor)
            img_width: 이미지 너비
            img_height: 이미지 높이

        Returns:
            SegmentationMask: Polygon 형식의 마스크
        """
        try:
            # Tensor -> NumPy 변환
            mask_np = mask_tensor.cpu().numpy()

            # 1. 원본 이미지 크기로 리사이즈
            mask_resized = cv2.resize(mask_np, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

            # 2. 이진화 (0.5 기준으로 0과 1로 나눔)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            # 3. 외곽선(Contour) 추출 -> 다각형 좌표 구하기
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 가장 큰 외곽선 선택 (작은 노이즈 제거)
                c = max(contours, key=cv2.contourArea)

                # 면적이 너무 작으면 무시
                if cv2.contourArea(c) < 1.0:
                    return None

                # [[x,y], [x,y]...] 형태의 리스트로 변환 (정수 좌표)
                polygon = c.reshape(-1, 2).astype(int).tolist()

                return SegmentationMask(
                    format="polygon",  # 형식을 'polygon'으로 변경
                    size=[img_height, img_width],
                    counts=polygon     # RLE 문자열 대신 좌표 리스트 저장
                )

            return None

        except Exception as e:
            print(f" Failed to encode mask: {e}")
            return None

    def _get_color_for_class(self, class_id: int) -> str:
        """
        클래스 ID에 대한 고정 색상 생성 (hex)

        Args:
            class_id: 클래스 ID

        Returns:
            str: Hex 색상 코드 (예: '#FF5733')
        """
        if class_id not in self.color_map:
            # 시드 고정으로 같은 클래스는 항상 같은 색상
            random.seed(class_id)
            r = random.randint(50, 255)  # 너무 어두운 색 제외
            g = random.randint(50, 255)
            b = random.randint(50, 255)
            self.color_map[class_id] = f"#{r:02x}{g:02x}{b:02x}"

        return self.color_map[class_id]

    def get_model_info(self) -> ModelInfo:
        """모델 정보 반환"""
        if not self.is_loaded():
            self.load_model()

        # YOLO 모델 정보 추출
        model_name = Path(self.model_path).stem  # yolov8n-seg.pt -> yolov8n-seg

        return ModelInfo(
            name=model_name,
            version="8.0.0",  # Ultralytics 버전
            task="segment" if "seg" in model_name else "detect"
        )

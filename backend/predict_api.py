import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.ops import box_iou

# DINO 관련 모듈 임포트
from models.dino.datasets import transforms as DT
from util.slconfig import SLConfig
from models.registry import MODULE_BUILD_FUNCS
from models.unet.utils import load_seunet

# --- FDI 변환 함수 ---
def to_fdi(universal_num: int) -> str:
    """Universal Numbering System (1-32)을 FDI (Quadrant-Tooth)로 변환"""
    if universal_num <= 0 or universal_num > 32:
        return str(universal_num)

    quadrant = (universal_num - 1) // 8 + 1
    tooth_in_quadrant = (universal_num - 1) % 8 + 1

    fdi_num = f"{quadrant}{tooth_in_quadrant}"
    return fdi_num

# --- DINO 관련 유틸리티 ---
def build_model_main(args):
    """DINO 모델 빌드 함수"""
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

class DinoDetectionPredictor:
    """DINO 모델 예측기"""
    def __init__(self, cfg_path, checkpoint_path, score_threshold=0.2, device='cuda'):
        # CUDA 사용 가능 여부 확인
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"[WARNING] CUDA not available, falling back to CPU")
            device = 'cpu'

        self.device = device
        args = SLConfig.fromfile(cfg_path)
        args.device = device
        args.modelname = 'dino'
        model, _, postprocessors = build_model_main(args)

        # ✅ 여기부터 수정
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                # DINO / DETR 계열 일반 포맷
                state_dict = checkpoint["model"]
            elif "model_state_dict" in checkpoint:
                # 네가 UNet / SEUNet에서 쓰던 포맷
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                # 일반 PyTorch 예제 포맷
                state_dict = checkpoint["state_dict"]
            else:
                # 예외적인 경우 – 어떤 키들이 있는지 한번 찍어보자
                print("[DINO] Unknown checkpoint keys:", checkpoint.keys())
                state_dict = checkpoint
        else:
            # 그냥 state_dict(OrderedDict)만 저장된 경우
            state_dict = checkpoint

        # 필요하면 'module.' 제거 (DDP로 학습한 가중치인 경우)
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        # ✅ 여기까지 수정

        model.to(device)
        model.eval()
        self.model = model
        self.postprocessors = postprocessors
        self.score_threshold = score_threshold
        self.transform = DT.Compose([
            DT.RandomResize([800], max_size=1333), DT.ToTensor(),
            DT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    @torch.no_grad()
    def predict(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        img_tensor, _ = self.transform(image_pil, None)
        img_tensor = img_tensor.to(self.device)
        outputs = self.model(img_tensor.unsqueeze(0))
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(self.device)
        results = self.postprocessors["bbox"](outputs, target_sizes)[0]
        mask = results["scores"] > self.score_threshold
        return {
            "boxes": results["boxes"][mask].cpu().numpy(),
            "labels": results["labels"][mask].cpu().numpy() + 1,
            "scores": results["scores"][mask].cpu().numpy()
        }

class SegmentationPredictor:
    """UNet Segmentation 모델 예측기"""
    def __init__(self, model, mean=0.458, std=0.173, cuda=True):
        self.model = model
        self.model.eval()
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.mean = mean
        self.std = std

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> torch.Tensor:
        origin_shape = image.shape
        image = Image.fromarray(image).resize((256, 256))
        image = F.to_tensor(image)
        image = F.normalize(image, [self.mean], [self.std])
        image = image.unsqueeze(0).to(self.device)
        predictions = self.model(image)
        predictions = predictions.squeeze(0)
        predictions = torch.argmax(predictions, dim=0, keepdim=True)
        predictions = F.resize(predictions, [origin_shape[0], origin_shape[1]], F.InterpolationMode.NEAREST)
        return predictions.squeeze(0)

def label_mask_to_bbox(mask: np.ndarray):
    """Segmentation Mask(H, W)를 Bounding Box (x2, y2)로 변환"""
    bbox_dict = {}
    for label in np.unique(mask):
        if label == 0: continue
        label_mask = (mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) == 0: continue
        bbox_xywh = cv2.boundingRect(contours[0])
        bbox_dict[label] = [
            bbox_xywh[0],
            bbox_xywh[1],
            bbox_xywh[0] + bbox_xywh[2],
            bbox_xywh[1] + bbox_xywh[3]
        ]
    return bbox_dict


class TeethDetectionModel:
    """통합 치아 탐지 모델 (DINO + UNet)"""

    def __init__(self, dino_config, dino_checkpoint, unet_checkpoint, unet_num_classes=33):
        """
        Args:
            dino_config: DINO 설정 파일 경로
            dino_checkpoint: DINO 체크포인트 경로
            unet_checkpoint: UNet 체크포인트 경로
            unet_num_classes: UNet 클래스 수
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[모델 로드] 디바이스: {self.device}")

        # DINO 모델 로드
        print("[모델 로드] DINO 모델 로딩 중...")
        self.dino_predictor = DinoDetectionPredictor(
            dino_config,
            dino_checkpoint,
            score_threshold=0.2,
            device=self.device
        )
        print("[모델 로드] DINO 모델 로드 완료")

        # UNet 모델 로드
        print("[모델 로드] UNet 모델 로딩 중...")
        unet_model = load_seunet(unet_checkpoint, unet_num_classes, cuda=(self.device=='cuda'))
        self.unet_predictor = SegmentationPredictor(unet_model, cuda=(self.device=='cuda'))
        print("[모델 로드] UNet 모델 로드 완료")

    def predict(self, image_path_or_array):
        """
        치아 탐지 예측 수행

        Args:
            image_path_or_array: 이미지 파일 경로 (str) 또는 numpy array

        Returns:
            dict: {
                'teeth': [
                    {
                        'fdi': str,
                        'universal': int,
                        'bbox': [x1, y1, x2, y2],
                        'center': [x, y],
                        'source': 'DINO' or 'UNET',
                        'confidence': float (DINO only)
                    },
                    ...
                ],
                'upper_arch': [fdi_numbers],
                'lower_arch': [fdi_numbers],
                'image_shape': [height, width]
            }
        """
        # 이미지 로드
        if isinstance(image_path_or_array, str):
            image_bgr = cv2.imread(image_path_or_array)
            if image_bgr is None:
                raise ValueError(f"이미지를 불러올 수 없습니다: {image_path_or_array}")
        else:
            image_bgr = image_path_or_array

        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # DINO 예측
        dino_results = self.dino_predictor.predict(image_bgr)

        # UNet 예측
        unet_mask = self.unet_predictor.predict(image_gray).cpu().numpy()
        unet_tooth_bboxes_dict = label_mask_to_bbox(unet_mask)

        # 치아 정보 통합
        detected_teeth = {}

        # DINO 결과 추가
        for box, label, score in zip(
            dino_results["boxes"],
            dino_results["labels"],
            dino_results["scores"]
        ):
            universal_num = int(label)
            x1, y1, x2, y2 = box
            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)
            fdi_num = to_fdi(universal_num)

            detected_teeth[universal_num] = {
                'fdi': fdi_num,
                'universal': universal_num,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'center': [center_x, center_y],
                'source': 'DINO',
                'confidence': float(score)
            }

        # UNet 결과 추가/업데이트
        for label, box in unet_tooth_bboxes_dict.items():
            universal_num = int(label)
            x1, y1, x2, y2 = box
            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)
            fdi_num = to_fdi(universal_num)

            # UNet 결과로 덮어쓰기 (UNet이 더 정확하다고 가정)
            detected_teeth[universal_num] = {
                'fdi': fdi_num,
                'universal': universal_num,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'center': [center_x, center_y],
                'source': 'UNET',
                'confidence': 1.0  # UNet은 confidence가 없으므로 1.0으로 설정
            }

        # 치아 리스트로 변환
        teeth_list = list(detected_teeth.values())

        # 상악/하악 분리
        upper_arch = sorted(
            [t['fdi'] for t in teeth_list if int(t['fdi'][0]) in [1, 2]],
            key=lambda x: (int(x[0]), int(x[1]))
        )
        lower_arch = sorted(
            [t['fdi'] for t in teeth_list if int(t['fdi'][0]) in [3, 4]],
            key=lambda x: (int(x[0]), int(x[1]))
        )

        return {
            'teeth': teeth_list,
            'upper_arch': upper_arch,
            'lower_arch': lower_arch,
            'image_shape': list(image_bgr.shape[:2]),
            'total_detected': len(teeth_list)
        }

"""
Grad-CAM proof-of-concept for YOLO detection weights used in this project.

This script is intentionally isolated from the main Flask/API flow.
It loads one detection model, runs a forward/backward pass, and saves only
verification artifacts under the output directory.

Examples:
  python gpts/test/run_gradcam_poc.py --image path/to/pano.png --preset caries
  python gpts/test/run_gradcam_poc.py --image path/to/pano.png --preset periapical --imgsz 1024
  python gpts/test/run_gradcam_poc.py --image path/to/pano.png --weights gpts/weights/caries_det.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.image_loader import load_image_any  # noqa: E402


PRESET_WEIGHTS = {
    "caries": ROOT / "weights" / "caries_det.pt",
    "periapical": ROOT / "weights" / "periapical.pt",
}


@dataclass
class LetterboxInfo:
    image: np.ndarray
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grad-CAM POC and save result artifacts only.")
    parser.add_argument("--image", required=True, help="Source image or DICOM path")
    parser.add_argument("--preset", choices=sorted(PRESET_WEIGHTS.keys()), default="caries", help="Project detection preset")
    parser.add_argument("--weights", help="Optional explicit detection weight path")
    parser.add_argument("--out", help="Optional output directory. Default: gpts/test/results/<timestamp>_<preset>")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference size for CAM forward")
    parser.add_argument("--conf", type=float, default=0.05, help="Detection confidence for saved detection snapshot")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--layer-index", type=int, help="Optional explicit model.model layer index for CAM hook")
    parser.add_argument("--target-class", type=int, help="Optional explicit class index for CAM target")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap blend amount")
    return parser.parse_args()


def resolve_weights(args: argparse.Namespace) -> Path:
    weights = Path(args.weights) if args.weights else PRESET_WEIGHTS[args.preset]
    if not weights.is_absolute():
        weights = (Path.cwd() / weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weight not found: {weights}")
    return weights


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.out:
        out_dir = Path(args.out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "test" / "results" / f"{timestamp}_{args.preset}"
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_image_any(path: Path) -> np.ndarray:
    img = load_image_any(path)
    if img is None or img.size == 0:
        raise RuntimeError(f"Failed to load image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def letterbox(image: np.ndarray, new_size: int = 1024, color: tuple[int, int, int] = (114, 114, 114)) -> LetterboxInfo:
    height, width = image.shape[:2]
    scale = min(float(new_size) / max(height, 1), float(new_size) / max(width, 1))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_size - resized_w
    pad_h = new_size - resized_h
    pad_left = int(round(pad_w / 2.0 - 0.1))
    pad_right = int(round(pad_w / 2.0 + 0.1))
    pad_top = int(round(pad_h / 2.0 - 0.1))
    pad_bottom = int(round(pad_h / 2.0 + 0.1))

    bordered = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return LetterboxInfo(
        image=bordered,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )


def build_input_tensor(image_bgr: np.ndarray, device: str, imgsz: int) -> tuple[torch.Tensor, LetterboxInfo]:
    lb = letterbox(image_bgr, new_size=imgsz)
    rgb = cv2.cvtColor(lb.image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    return tensor, lb


def select_target_layer(model: Any, explicit_index: int | None) -> tuple[int, torch.nn.Module]:
    layers = list(enumerate(model.model))
    if explicit_index is not None:
        return explicit_index, model.model[explicit_index]

    preferred_names = ("C2f", "C3", "C2", "SPPF", "Conv")
    for name in preferred_names:
        for idx, module in reversed(layers[:-1]):
            if module.__class__.__name__ == name:
                return idx, module
    fallback_idx = max(0, len(layers) - 2)
    return fallback_idx, model.model[fallback_idx]


class ActivationHook:
    def __init__(self, module: torch.nn.Module):
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.tensor_hook: Any = None
        self.forward_handle = module.register_forward_hook(self._save_activation)

    def _save_activation(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        self.activations = output
        self.gradients = None
        if self.tensor_hook is not None:
            self.tensor_hook.remove()
            self.tensor_hook = None
        if isinstance(output, torch.Tensor) and output.requires_grad:
            self.tensor_hook = output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def close(self) -> None:
        self.forward_handle.remove()
        if self.tensor_hook is not None:
            self.tensor_hook.remove()
            self.tensor_hook = None


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def box_iou(reference_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(reference_box[0], boxes[:, 0])
    y1 = np.maximum(reference_box[1], boxes[:, 1])
    x2 = np.minimum(reference_box[2], boxes[:, 2])
    y2 = np.minimum(reference_box[3], boxes[:, 3])

    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    intersection = inter_w * inter_h

    ref_area = max(0.0, float(reference_box[2] - reference_box[0])) * max(0.0, float(reference_box[3] - reference_box[1]))
    box_area = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None) * np.clip(boxes[:, 3] - boxes[:, 1], a_min=0.0, a_max=None)
    union = np.maximum(ref_area + box_area - intersection, 1e-8)
    return intersection / union


def map_box_to_letterbox(box_xyxy: np.ndarray, lb: LetterboxInfo) -> np.ndarray:
    mapped = box_xyxy.astype(np.float32).copy()
    mapped[[0, 2]] = mapped[[0, 2]] * lb.scale + lb.pad_left
    mapped[[1, 3]] = mapped[[1, 3]] * lb.scale + lb.pad_top
    return mapped


def select_best_detection_meta(result: Any, class_index: int | None) -> dict[str, Any] | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.zeros(len(boxes))
    classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)

    candidate_indices = np.where(classes == class_index)[0] if class_index is not None else np.arange(len(boxes))
    if candidate_indices.size == 0:
        candidate_indices = np.arange(len(boxes))
    best_idx = int(candidate_indices[np.argmax(confs[candidate_indices])])
    return {
        "best_detection_index": best_idx,
        "best_detection_confidence": float(confs[best_idx]),
        "best_detection_class": int(classes[best_idx]),
        "best_detection_box": [float(v) for v in boxes[best_idx].tolist()],
    }


def compute_gradcam(
    detection_model: Any,
    input_tensor: torch.Tensor,
    hook: ActivationHook,
    target_class: int | None,
    target_box_xyxy: np.ndarray | None,
    imgsz: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    input_tensor = input_tensor.requires_grad_(True)
    detection_model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        raw_output = detection_model(input_tensor)
        predictions = raw_output[0] if isinstance(raw_output, tuple) else raw_output
        output_aux = raw_output[1] if isinstance(raw_output, tuple) and len(raw_output) > 1 and isinstance(raw_output[1], dict) else {}
        feature_maps = output_aux.get("feats") if isinstance(output_aux, dict) else None
        score_logits = output_aux.get("scores") if isinstance(output_aux, dict) else None
        if isinstance(feature_maps, list):
            for feature_map in feature_maps:
                if isinstance(feature_map, torch.Tensor):
                    feature_map.retain_grad()

        if not isinstance(predictions, torch.Tensor) or predictions.ndim != 3:
            raise RuntimeError(f"Unsupported prediction output for CAM: {type(predictions)}")
        if predictions.shape[1] <= 4:
            raise RuntimeError(f"Prediction tensor does not expose class scores: {tuple(predictions.shape)}")

        class_scores = predictions[:, 4:, :]
        num_classes = int(class_scores.shape[1])
        num_candidates = int(class_scores.shape[2])
        prediction_boxes_xywh = predictions[0, :4, :].detach().cpu().transpose(0, 1).numpy()
        prediction_boxes_xyxy = xywh_to_xyxy(prediction_boxes_xywh)

        if target_box_xyxy is not None:
            class_index = int(target_class) if target_class is not None else 0
            if not (0 <= class_index < num_classes):
                raise ValueError(f"target_class must be within [0, {num_classes - 1}]")

            class_slice = class_scores[0, class_index]
            class_scores_np = class_slice.detach().cpu().numpy()
            ious = box_iou(target_box_xyxy.astype(np.float32), prediction_boxes_xyxy.astype(np.float32))

            ref_center = np.array(
                [
                    (target_box_xyxy[0] + target_box_xyxy[2]) * 0.5,
                    (target_box_xyxy[1] + target_box_xyxy[3]) * 0.5,
                ],
                dtype=np.float32,
            )
            pred_centers = np.stack(
                (
                    (prediction_boxes_xyxy[:, 0] + prediction_boxes_xyxy[:, 2]) * 0.5,
                    (prediction_boxes_xyxy[:, 1] + prediction_boxes_xyxy[:, 3]) * 0.5,
                ),
                axis=1,
            )
            center_dist = np.linalg.norm(pred_centers - ref_center[None, :], axis=1) / max(float(imgsz), 1.0)
            score_norm = class_scores_np / max(float(class_scores_np.max()), 1e-8)

            if float(ious.max()) >= 0.05:
                match_quality = 0.75 * ious + 0.25 * score_norm
                match_mode = "iou+score"
            else:
                proximity = 1.0 - np.clip(center_dist / np.sqrt(2.0), 0.0, 1.0)
                match_quality = 0.6 * proximity + 0.4 * score_norm
                match_mode = "center+score"

            det_index = int(np.argmax(match_quality))
            score = class_slice[det_index]
        else:
            flat_index = int(torch.argmax(class_scores[0]).item())
            class_index, det_index = np.unravel_index(flat_index, (num_classes, num_candidates))
            score = class_scores[0, class_index, det_index]
            match_mode = "global_max"
            ious = None
            center_dist = None

        score_for_backprop = score_logits[0, class_index, det_index] if isinstance(score_logits, torch.Tensor) else score

        score_for_backprop.backward()

    cam_source = "module_hook"
    target_feature_scale = None
    target_feature_shape = None
    target_feature_offset = None
    local_candidate_index = None
    selected_activations = hook.activations
    selected_gradients = hook.gradients

    if isinstance(feature_maps, list):
        feature_offset = 0
        for feature_scale, feature_map in enumerate(feature_maps):
            if not isinstance(feature_map, torch.Tensor) or feature_map.ndim != 4:
                continue
            feature_candidates = int(feature_map.shape[2] * feature_map.shape[3])
            if det_index < feature_offset + feature_candidates:
                selected_activations = feature_map
                selected_gradients = feature_map.grad
                cam_source = "detect_feats"
                target_feature_scale = feature_scale
                target_feature_shape = [int(v) for v in feature_map.shape]
                target_feature_offset = feature_offset
                local_candidate_index = int(det_index - feature_offset)
                break
            feature_offset += feature_candidates

    if selected_activations is None or selected_gradients is None:
        raise RuntimeError("CAM target did not capture activations/gradients")
    if selected_activations.ndim != 4 or selected_gradients.ndim != 4:
        raise RuntimeError(
            f"Expected 4D activation/gradient tensors, got {selected_activations.ndim} and {selected_gradients.ndim}"
        )

    weights = selected_gradients.mean(dim=(2, 3), keepdim=True)
    cam_logits = (weights * selected_activations).sum(dim=1, keepdim=True)
    cam_logits = F.interpolate(cam_logits, size=(imgsz, imgsz), mode="bilinear", align_corners=False)
    cam_logits_np = cam_logits[0, 0].detach().cpu().numpy()
    cam = np.maximum(cam_logits_np, 0.0)
    cam_mode = "gradcam_relu"
    if float(cam.max()) <= 1e-8:
        cam = np.abs(cam_logits_np)
        cam_mode = "gradcam_abs_fallback"
    cam -= cam.min()
    if cam.max() > 1e-8:
        cam /= cam.max()

    metadata = {
        "target_class": int(class_index),
        "target_detection_index": int(det_index),
        "target_score": float(score.detach().cpu().item()),
        "target_backprop_score": float(score_for_backprop.detach().cpu().item()),
        "num_classes": num_classes,
        "num_candidates": num_candidates,
        "target_mode": match_mode,
        "cam_source": cam_source,
        "cam_mode": cam_mode,
        "cam_logit_min": float(cam_logits_np.min()),
        "cam_logit_max": float(cam_logits_np.max()),
        "cam_nonzero_ratio": float((cam > 0).mean()),
        "cam_p95": float(np.percentile(cam, 95.0)),
        "cam_p99": float(np.percentile(cam, 99.0)),
        "matched_raw_box_xyxy": [float(v) for v in prediction_boxes_xyxy[det_index].tolist()],
    }
    if target_feature_scale is not None:
        metadata["target_feature_scale"] = int(target_feature_scale)
        metadata["target_feature_shape"] = target_feature_shape
        metadata["target_feature_offset"] = int(target_feature_offset)
        metadata["target_feature_local_index"] = int(local_candidate_index)
    if target_box_xyxy is not None and ious is not None and center_dist is not None:
        metadata["reference_box_xyxy"] = [float(v) for v in target_box_xyxy.tolist()]
        metadata["match_iou"] = float(ious[det_index])
        metadata["match_center_distance"] = float(center_dist[det_index])
    return cam, metadata


def unletterbox_cam(cam: np.ndarray, lb: LetterboxInfo, original_shape: tuple[int, int]) -> np.ndarray:
    h, w = cam.shape[:2]
    cropped = cam[
        lb.pad_top : max(lb.pad_top, h - lb.pad_bottom),
        lb.pad_left : max(lb.pad_left, w - lb.pad_right),
    ]
    original_h, original_w = original_shape
    if cropped.size == 0:
        return cv2.resize(cam, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)


def overlay_cam_on_image(image_bgr: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    heatmap = np.uint8(np.clip(cam, 0.0, 1.0) * 255.0)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0.0)


def build_focus_cam(cam: np.ndarray) -> np.ndarray:
    positive = np.clip(cam.astype(np.float32), 0.0, 1.0)
    if positive.max() <= 1e-8:
        return np.zeros_like(positive, dtype=np.float32)

    blurred = cv2.GaussianBlur(positive, (0, 0), sigmaX=9.0, sigmaY=9.0)
    if blurred.max() > 1e-8:
        blurred /= blurred.max()

    active = blurred[blurred > 0]
    if active.size == 0:
        return np.zeros_like(positive, dtype=np.float32)

    threshold = float(np.percentile(active, 97.5))
    focused = np.clip((blurred - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    focused = cv2.GaussianBlur(focused, (0, 0), sigmaX=5.0, sigmaY=5.0)
    if focused.max() > 1e-8:
        focused /= focused.max()
    return focused.astype(np.float32)


def overlay_focus_cam_on_image(image_bgr: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    focus = build_focus_cam(cam)
    heatmap = np.uint8(np.clip(focus, 0.0, 1.0) * 255.0)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    mask = np.clip(focus[..., None] * alpha, 0.0, 1.0)
    blended = image_bgr.astype(np.float32) * (1.0 - mask) + colored.astype(np.float32) * mask
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def draw_detection_boxes(image: np.ndarray, result: Any, names: Any) -> np.ndarray:
    boxed = image.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return boxed

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.zeros(len(boxes))
    classes = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)

    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        label = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            boxed,
            f"{label} {conf:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return boxed


def highlight_best_detection(image: np.ndarray, result: Any, class_index: int | None) -> tuple[np.ndarray, dict[str, Any] | None]:
    best_meta = select_best_detection_meta(result, class_index)
    if best_meta is None:
        return image.copy(), None

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    best_idx = int(best_meta["best_detection_index"])

    highlighted = image.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in boxes[best_idx].tolist()]
    cv2.rectangle(highlighted, (x1, y1), (x2, y2), (255, 255, 255), 3)
    cv2.putText(
        highlighted,
        f"best det {best_idx} conf {best_meta['best_detection_confidence']:.2f}",
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return highlighted, best_meta


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    weights_path = resolve_weights(args)
    out_dir = resolve_output_dir(args)

    original_bgr = read_image_any(image_path)
    cv2.imwrite(str(out_dir / "input_original.png"), original_bgr)

    yolo = YOLO(str(weights_path))
    detection_model = yolo.model
    detection_model.to(args.device)
    detection_model.eval()

    results = yolo.predict(
        source=original_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False,
        device=args.device,
    )
    detection_result = results[0]
    best_detection_meta = select_best_detection_meta(detection_result, args.target_class)

    layer_index, target_layer = select_target_layer(detection_model, args.layer_index)
    hook = ActivationHook(target_layer)

    try:
        input_tensor, letterbox_info = build_input_tensor(original_bgr, args.device, args.imgsz)
        target_box_xyxy = None
        target_box_class = args.target_class
        if best_detection_meta is not None:
            target_box_xyxy = map_box_to_letterbox(
                np.asarray(best_detection_meta["best_detection_box"], dtype=np.float32),
                letterbox_info,
            )
            target_box_class = int(best_detection_meta["best_detection_class"])
        cam_small, cam_meta = compute_gradcam(
            detection_model=detection_model,
            input_tensor=input_tensor,
            hook=hook,
            target_class=target_box_class,
            target_box_xyxy=target_box_xyxy,
            imgsz=args.imgsz,
        )
    finally:
        hook.close()

    cam_original = unletterbox_cam(cam_small, letterbox_info, original_bgr.shape[:2])
    cam_gray_path = out_dir / "cam_gray.png"
    cv2.imwrite(str(cam_gray_path), np.uint8(np.clip(cam_original, 0.0, 1.0) * 255.0))
    cam_focus = build_focus_cam(cam_original)
    cv2.imwrite(str(out_dir / "cam_gray_focus.png"), np.uint8(np.clip(cam_focus, 0.0, 1.0) * 255.0))
    detections_image = draw_detection_boxes(original_bgr, detection_result, detection_result.names)
    cv2.imwrite(str(out_dir / "detections.png"), detections_image)

    cam_overlay = overlay_cam_on_image(original_bgr, cam_original, alpha=args.alpha)
    cam_overlay_with_boxes = draw_detection_boxes(cam_overlay, detection_result, detection_result.names)
    cv2.imwrite(str(out_dir / "cam_overlay.png"), cam_overlay_with_boxes)
    cam_overlay_focus = overlay_focus_cam_on_image(original_bgr, cam_original, alpha=min(0.9, args.alpha + 0.2))
    cam_overlay_focus_with_boxes = draw_detection_boxes(cam_overlay_focus, detection_result, detection_result.names)
    cv2.imwrite(str(out_dir / "cam_overlay_focus.png"), cam_overlay_focus_with_boxes)

    highlighted, best_meta = highlight_best_detection(cam_overlay_with_boxes, detection_result, cam_meta.get("target_class"))
    cv2.imwrite(str(out_dir / "cam_overlay_best_detection.png"), highlighted)
    highlighted_focus, _ = highlight_best_detection(cam_overlay_focus_with_boxes, detection_result, cam_meta.get("target_class"))
    cv2.imwrite(str(out_dir / "cam_overlay_focus_best_detection.png"), highlighted_focus)

    metadata = {
        "image": str(image_path.resolve()),
        "weights": str(weights_path.resolve()),
        "preset": args.preset,
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "alpha": args.alpha,
        "target_layer_index": layer_index,
        "target_layer_name": target_layer.__class__.__name__,
        "letterbox": {
            "scale": letterbox_info.scale,
            "pad_left": letterbox_info.pad_left,
            "pad_top": letterbox_info.pad_top,
            "pad_right": letterbox_info.pad_right,
            "pad_bottom": letterbox_info.pad_bottom,
        },
        "cam": cam_meta,
        "best_detection": best_meta,
        "saved_files": [
            "input_original.png",
            "detections.png",
            "cam_gray.png",
            "cam_gray_focus.png",
            "cam_overlay.png",
            "cam_overlay_focus.png",
            "cam_overlay_best_detection.png",
            "cam_overlay_focus_best_detection.png",
            "meta.json",
        ],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(to_jsonable(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()

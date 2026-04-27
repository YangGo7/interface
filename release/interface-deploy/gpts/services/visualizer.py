
import cv2
import numpy as np
import math
from typing import Dict, List, Tuple

class PanoVisualizer:
    """
    Handles all image drawing and visualization tasks for Pano Analysis.
    Decouples 'Seeing' (UI) from 'Thinking' (Inference).
    """

    def __init__(self):
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'cyan': (255, 255, 0),
            'yellow': (0, 255, 255),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'orange': (0, 165, 255),
            'black': (0, 0, 0)
        }

    def _bbox_center_inside(self, bbox, inner_rect):
        if not inner_rect: return True
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return inner_rect["x1"] <= cx <= inner_rect["x2"] and inner_rect["y1"] <= cy <= inner_rect["y2"]

    def overlay_seg(self, base: np.ndarray, result, color_map: Dict[int, Tuple[int, int, int]], inner_rect: Dict[str, int], thickness=1, alpha=1.0):
        """
        Draw segmentation masks (contours only).
        """
        if result.masks is None:
            return base
        img = base.copy()
        overlay_layer = base.copy()
        min_area = 200.0
        
        for idx in range(len(result.masks.xy)):
            m = result.masks.xy[idx]
            if len(m) == 0: continue
            
            pts = np.int32([m])
            area = cv2.contourArea(pts)
            if area < min_area: continue
                
            x, y, w, h = cv2.boundingRect(pts)
            if not self._bbox_center_inside((x, y, x+w, y+h), inner_rect):
                continue
            
            color = color_map.get(idx, self.colors['green']) # Default Green
            cv2.polylines(overlay_layer, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            
        if alpha < 1.0:
            cv2.addWeighted(overlay_layer, alpha, img, 1 - alpha, 0, img)
            return img
        else:
            return overlay_layer

    def overlay_mask(self, base: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5):
        """
        Draw a binary mask overlay with transparency.
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return base
        
        img = base.copy()
        # Create colored mask
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0] = color
        
        # Blend
        dst = cv2.addWeighted(img, 1.0, colored_mask, alpha, 0)
        
        # Draw contour for sharpness
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dst, contours, -1, color, 2)
        
        return dst

    def _soften_mask(self, mask: np.ndarray, blur_size: int) -> np.ndarray:
        blur_size = max(3, int(blur_size) | 1)
        softened = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)
        if softened.max() > 0:
            softened /= softened.max()
        return softened

    def _add_blob(
        self,
        color_accum: np.ndarray,
        weight_accum: np.ndarray,
        center: Tuple[float, float],
        radius: int,
        strength: float,
        color: Tuple[int, int, int],
    ) -> None:
        h, w = weight_accum.shape
        radius = max(12, int(radius))
        cx, cy = int(center[0]), int(center[1])
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return

        yy, xx = np.mgrid[y1:y2, x1:x2]
        sigma = max(radius * 0.42, 8.0)
        gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        gaussian *= float(max(0.0, min(1.0, strength)))
        weight_accum[y1:y2, x1:x2] += gaussian
        color_accum[y1:y2, x1:x2] += gaussian[..., None] * np.array(color, dtype=np.float32)

    def _add_focus_blob(
        self,
        color_accum: np.ndarray,
        weight_accum: np.ndarray,
        center: Tuple[float, float],
        radius: int,
        strength: float,
        color: Tuple[int, int, int],
        sigma_scale: float = 0.26,
        percentile: float = 97.5,
        post_blur_sigma: float = 4.5,
    ):
        h, w = weight_accum.shape
        radius = max(14, int(radius))
        cx, cy = int(center[0]), int(center[1])
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return None

        yy, xx = np.mgrid[y1:y2, x1:x2]
        sigma = max(radius * float(sigma_scale), 3.5)
        gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        if gaussian.max() <= 1e-8:
            return None
        gaussian /= gaussian.max()

        active = gaussian[gaussian > 0]
        threshold = float(np.percentile(active, percentile)) if active.size else 1.0
        focused = np.clip((gaussian - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
        if float(focused.max()) <= 1e-8:
            focused = gaussian

        kernel_size = max(3, int(math.ceil(post_blur_sigma * 4.0)) | 1)
        focused = cv2.GaussianBlur(focused, (kernel_size, kernel_size), post_blur_sigma)
        if focused.max() > 1e-8:
            focused /= focused.max()

        focused *= float(max(0.0, min(1.0, strength)))
        weight_accum[y1:y2, x1:x2] += focused
        color_accum[y1:y2, x1:x2] += focused[..., None] * np.array(color, dtype=np.float32)
        return focused, x1, y1, x2, y2

    def _add_mask_glow(
        self,
        color_accum: np.ndarray,
        weight_accum: np.ndarray,
        mask: np.ndarray,
        strength: float,
        color: Tuple[int, int, int],
        blur_size: int,
    ) -> None:
        if mask is None or np.count_nonzero(mask) == 0:
            return
        softened = self._soften_mask(mask, blur_size) * float(max(0.0, min(1.0, strength)))
        weight_accum += softened
        color_accum += softened[..., None] * np.array(color, dtype=np.float32)

    def build_risk_overlay(
        self,
        base: np.ndarray,
        teeth_objects: List[dict],
        caries_objects: List[dict],
        periapical_objects: List[dict],
        pbl_dict: Dict[str, dict],
        nerve_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build a pseudo-heatmap from findings and bone-loss severity.
        This is a risk overlay, not a pixel-level probability map.
        """
        h, w = base.shape[:2]
        color_accum = np.zeros((h, w, 3), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)
        disease_weight_accum = np.zeros((h, w), dtype=np.float32)

        caries_color = (0, 140, 255)      # Orange-red in BGR
        periapical_color = (24, 24, 255)  # Red
        nerve_color = (255, 0, 255)       # Magenta

        for item in caries_objects or []:
            box = item.get("box") or []
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            radius = int(max(x2 - x1, y2 - y1) * 0.85)
            conf = float(item.get("confidence", item.get("conf", 0.55)) or 0.55)
            strength = 0.28 + min(max(conf, 0.0), 1.0) * 0.58
            focus_patch = self._add_focus_blob(
                color_accum,
                weight_accum,
                (cx, cy),
                radius,
                strength,
                caries_color,
                sigma_scale=0.22,
                percentile=96.8,
                post_blur_sigma=3.8,
            )
            if focus_patch is not None:
                focused, fx1, fy1, fx2, fy2 = focus_patch
                disease_weight_accum[fy1:fy2, fx1:fx2] += focused * 1.35

        for item in periapical_objects or []:
            box = item.get("box") or []
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            radius = int(max(x2 - x1, y2 - y1) * 1.08)
            conf = float(item.get("confidence", item.get("conf", 0.6)) or 0.6)
            strength = 0.34 + min(max(conf, 0.0), 1.0) * 0.62
            focus_patch = self._add_focus_blob(
                color_accum,
                weight_accum,
                (cx, cy),
                radius,
                strength,
                periapical_color,
                sigma_scale=0.25,
                percentile=96.0,
                post_blur_sigma=4.2,
            )
            if focus_patch is not None:
                focused, fx1, fy1, fx2, fy2 = focus_patch
                disease_weight_accum[fy1:fy2, fx1:fx2] += focused * 1.6

        # Nerve is rendered as a mask-based heat region, not a probabilistic output.
        # We soften the binary canal mask so the saved heatmap asset carries the nerve
        # signal directly instead of relying on a separate frontend-only glow layer.
        if nerve_mask is not None and np.count_nonzero(nerve_mask) > 0:
            self._add_mask_glow(
                color_accum,
                weight_accum,
                nerve_mask,
                strength=0.34,
                color=nerve_color,
                blur_size=35,
            )

        for tooth in teeth_objects or []:
            tooth_label = str(tooth.get("tooth_label") or "")
            if not tooth_label or tooth_label not in pbl_dict:
                continue
            pbl_info = pbl_dict.get(tooth_label) or {}
            percent = float(pbl_info.get("percent", tooth.get("bone_loss_pct", 0.0)) or 0.0)
            if percent < 10.0:
                continue

            severity = min(max((percent - 10.0) / 35.0, 0.0), 1.0)
            strength = 0.16 + severity * 0.48
            green = int(255 * (1.0 - severity))
            color = (0, green, 255)  # Yellow -> Red in BGR

            mask = np.zeros((h, w), dtype=np.uint8)
            contour = tooth.get("contour") or []
            if contour:
                try:
                    pts = np.array(contour, dtype=np.int32).reshape(-1, 2)
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], 255)
                except Exception:
                    pass

            if np.count_nonzero(mask) == 0:
                box = tooth.get("box") or []
                if len(box) == 4:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(w - 1, x2), min(h - 1, y2)), 255, -1)

            self._add_mask_glow(color_accum, weight_accum, mask, strength, color, blur_size=41)

        weight_accum = np.clip(weight_accum, 0.0, 1.0)
        disease_weight_accum = np.clip(disease_weight_accum, 0.0, 1.0)
        mixed_color = np.zeros_like(color_accum)
        valid = weight_accum > 1e-6
        mixed_color[valid] = color_accum[valid] / weight_accum[valid, None]
        mixed_color = np.clip(mixed_color * 1.12, 0, 255)

        alpha_base = weight_accum * 0.48
        alpha_disease = disease_weight_accum * 0.62
        alpha_map = np.clip(alpha_base + alpha_disease, 0.0, 0.82)[..., None]
        base_f = base.astype(np.float32)
        risk_overlay = base_f * (1.0 - alpha_map) + mixed_color * alpha_map
        return np.clip(risk_overlay, 0, 255).astype(np.uint8)

    def overlay_det(self, base: np.ndarray, result, names, color: Tuple[int, int, int], inner_rect: Dict[str, int], exclusion_boxes: List[List[float]] = None) -> Tuple[np.ndarray, List[Tuple[List[int], float, str]]]:
        """
        Draw detections (bounding boxes) with dashed styling.
        """
        img = base.copy()
        det_list = []
        if result.boxes is None:
            return img, det_list
            
        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Exclusion logic
            if exclusion_boxes:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                is_excluded = False
                for eb in exclusion_boxes:
                    if eb[0] <= cx <= eb[2] and eb[1] <= cy <= eb[3]:
                        is_excluded = True
                        break
                if is_excluded: continue

            if not self._bbox_center_inside((x1, y1, x2, y2), inner_rect):
                continue
            
            label = names[int(cls)]
            self._draw_dashed_rect(img, (x1, y1), (x2, y2), color, 1, 5)
            det_list.append(([int(x1), int(y1), int(x2), int(y2)], float(conf), label))
            
        return img, det_list

    def _draw_dashed_rect(self, img, p1, p2, color, thickness=1, gap=5):
        x1, y1 = p1
        x2, y2 = p2
        # Top & Bottom
        for x in range(x1, x2, gap * 2):
            cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, thickness)
        # Left & Right
        for y in range(y1, y2, gap * 2):
            cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, thickness)

    def draw_tooth_labels(self, overlay: np.ndarray, teeth_objects) -> np.ndarray:
        """
        Draw tooth numbers with force-directed layout (repulsion) to avoid collision.
        """
        overlay_copy = overlay.copy()
        h, w = overlay.shape[:2]
        
        # 1. Collect initial positions
        labels_to_draw = []
        for t_obj in teeth_objects:
            lbl = t_obj.get("tooth_label")
            if lbl and t_obj.get("box"):
                x1, y1, x2, y2 = map(int, t_obj["box"])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Default position logic
                draw_y = cy
                try:
                    q = int(lbl) // 10
                    if q in [1, 2]: draw_y = y2  # Upper teeth -> below box
                    elif q in [3, 4]: draw_y = y1 # Lower teeth -> above box
                except: pass
                
                labels_to_draw.append({
                    'x': float(cx), 'y': float(draw_y), 
                    'orig_x': cx, 'orig_y': draw_y,
                    'lbl': str(lbl)
                })
        
        # 2. Iterative Repulsion
        radius = 18 
        for _ in range(5):
            for i in range(len(labels_to_draw)):
                for j in range(i + 1, len(labels_to_draw)):
                    l1 = labels_to_draw[i]
                    l2 = labels_to_draw[j]
                    
                    dx = l1['x'] - l2['x']
                    dy = l1['y'] - l2['y']
                    dist_sq = dx*dx + dy*dy
                    
                    min_dist = radius * 2
                    if dist_sq < min_dist * min_dist:
                        dist = math.sqrt(dist_sq)
                        if dist < 0.1: dist = 0.1
                        
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        shift = overlap * 0.5
                        
                        l1['x'] += nx * shift
                        l1['y'] += ny * shift
                        l2['x'] -= nx * shift
                        l2['y'] -= ny * shift

        # 3. Draw
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        
        for item in labels_to_draw:
            x, y = int(item['x']), int(item['y'])
            ox, oy = int(item['orig_x']), int(item['orig_y'])
            txt = item['lbl']
            
            # Red Centroid Dot
            cv2.circle(overlay_copy, (ox, oy), 3, self.colors['red'], -1)
            
            # Text Outline + Face
            cv2.putText(overlay_copy, txt, (x-10, y+5), font, scale, self.colors['black'], thickness+1, cv2.LINE_AA)
            cv2.putText(overlay_copy, txt, (x-10, y+5), font, scale, self.colors['yellow'], thickness, cv2.LINE_AA)
            
        return overlay_copy

    def draw_arch_curve(self, overlay: np.ndarray, poly_fn, color: Tuple[int, int, int], thickness=2) -> np.ndarray:
        """Draws a polynomial arch curve across the image width."""
        if not poly_fn: return overlay
        h, w = overlay.shape[:2]
        pts = []
        for x in range(0, w, 10):
            y = int(poly_fn(x))
            if 0 <= y < h:
                pts.append([x, y])
        if len(pts) > 1:
            cv2.polylines(overlay, [np.array(pts)], False, color, thickness, cv2.LINE_AA)
        return overlay
        
        return overlay_copy

    def draw_safety_guides(self, overlay: np.ndarray, objects: List[Dict]) -> np.ndarray:
        """
        Draw Implant/Nerve safety measurement lines.
        """
        img = overlay.copy()
        for obj in objects:
            guide = obj.get('safety_guide') or obj.get('implant_guide')
            if guide:
                coords = guide.get('line_coords')
                margin_coords = guide.get('margin_line_coords')
                dist = guide.get('dist_mm', 0)
                
                if coords:
                    x1, y1, x2, y2 = map(int, coords)
                    # Safe Distance Line (Yellow)
                    cv2.line(img, (x1, y1), (x2, y2), self.colors['yellow'], 2)
                    cv2.circle(img, (x1, y1), 3, self.colors['yellow'], -1) # Start
                    
                    # Safe Distance Label
                    text_x, text_y = (x1+x2)//2, (y1+y2)//2
                    cv2.putText(img, f"{dist:.1f}mm", (text_x+5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['yellow'], 2)

                    if margin_coords:
                        mx1, my1, mx2, my2 = map(int, margin_coords)
                        # Safety Margin Line (Cyan)
                        cyan = (255, 255, 0) # BGR
                        cv2.line(img, (mx1, my1), (mx2, my2), cyan, 2)
                        cv2.circle(img, (mx2, my2), 4, cyan, -1) # End (On Nerve)
                        
                        # Margin Label (2mm)
                        margin_text_x = (mx1 + mx2) // 2
                        margin_text_y = (my1 + my2) // 2
                        cv2.putText(img, "2mm", (margin_text_x + 5, margin_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cyan, 1)
                    else:
                        # Fallback
                        cv2.circle(img, (x2, y2), 4, self.colors['red'], -1)

        return img

    def draw_arch_curves(self, overlay: np.ndarray, curves: List[Tuple], color=(0, 255, 255)) -> np.ndarray:
        """
        Draw polynomial curves for dental arches.
        curves: List of (x_points, y_points) or similar
        """
        # Logic to be implemented or adapted from existing
        # For now, simplistic implementation based on what was seen
        pass

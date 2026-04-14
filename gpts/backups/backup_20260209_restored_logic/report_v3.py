import os
import base64
import cv2
import numpy as np
import datetime
from pathlib import Path

class ReportGeneratorV3:
    def __init__(self, output_dir="c:/interface/backend/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.odonto_dir = Path("c:/interface/imgs/Odonto")
        self.warning_icon_path = Path("c:/interface/imgs/!/?袁⑹뵠??(1).png")
        self.info_icon_path = Path("c:/interface/imgs/!/?袁⑹뵠??(3).png")
        self.logo_path = Path("c:/interface/imgs/logo/Cyberme_logo.png")
    
    @staticmethod
    def get_universal_label(fdi_str):
        """Convert FDI to Universal Numbering System"""
        if not fdi_str or not fdi_str.isdigit(): return fdi_str
        
        fdi = int(fdi_str)
        q = fdi // 10
        idx = fdi % 10
        
        # Permanent
        if q == 1: return str(8 + idx - (2 * (idx - 1))) if False else str(19 - (idx + 10)) # logic error? 
        # Simpler map
        # 18->1, 11->8.  1-8: 8 -> (9-idx) ? No. 18: idx=8. 9-8=1. 11: idx=1. 9-1=8. Correct.
        if q == 1: return str(9 - idx)
        # 21->9, 28->16. 2-8: 9 -> (8+idx). 21: idx=1. 8+1=9. 28: idx=8. 8+8=16. Correct.
        if q == 2: return str(8 + idx)
        # 31->24, 38->17. 3-8: 17 -> (25-idx). 31: idx=1. 25-1=24. 38: idx=8. 25-8=17. Correct.
        if q == 3: return str(25 - idx) # Wait. LL is 31..38. Universal 24..17.
        # Yes. 31(Inc) -> 24. 38(Mol) -> 17.
        # 41->25, 48->32. 4-8: 25 -> (24+idx). 41: idx=1. 24+1=25. 48: idx=8. 24+8=32. Correct.
        if q == 4: return str(24 + idx)
        
        # Primary
        # 55->A, 51->E.
        # A=1, E=5. Map to char.
        # 55(idx=5)->A(0). 51(idx=1)->E(4).
        # index = 5 - idx. 0->A.
        # chars = "ABCDE"
        # 55: idx=5. 5-5=0 -> A. 51: idx=1. 5-1=4 -> E. Correct.
        if q == 5: return "ABCDE"[5 - idx]
        
        # 61->F, 65->J.
        # 61(idx=1)->F(5). 65(idx=5)->J(9).
        # index = idx - 1.
        # "FGHIJ"[idx-1]
        if q == 6: return "FGHIJ"[idx - 1]
        
        # 71->O, 75->K.
        # 71(idx=1)->O(14). 75(idx=5)->K(10).
        # "KLMNO". K->0. O->4.
        # 75: idx=5 -> K(0). 71: idx=1 -> O(4).
        # index = 5 - idx.
        # "KLMNO"[5-idx]
        if q == 7: return "KLMNO"[5 - idx] # Wait. O is 71. K is 75. 
        # Array: K(0), L(1), M(2), N(3), O(4).
        # 75: idx=5. 5-5=0 -> K. 71: idx=1. 5-1=4 -> O. Correct.
        
        # 81->P, 85->T.
        # 81(idx=1)->P(15). 85(idx=5)->T(19).
        # "PQRST"[idx-1]
        if q == 8: return "PQRST"[idx - 1]
        
        return fdi_str

    def _load_odonto_icon(self, jaw, position):
        """Load odontogram tooth icon as base64. jaw='up'/'down', position=1-8"""
        prefix = "up" if jaw == "up" else "down"
        img_path = self.odonto_dir / f"{prefix}-{position}.png"
        if img_path.exists():
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return ""

    def _encode_image(self, img_array):
        """Encodes numpy image to base64 string"""
        if img_array is None: return ""
        success, buffer = cv2.imencode('.jpg', img_array)
        if not success: return ""
        return base64.b64encode(buffer).decode('utf-8')

    def _crop_and_overlay(self, orig_img, box, findings, bl_viz=None, pbl_mm=0, implant_guide=None, nerve_contours=None, sinus_contours=None, safety_guide=None, tooth_contour=None, debug_text=None, nerve_dist_line=None, nerve_dist_mm=0, centerline=False, centerline_axis=None):
        """
        Crops tooth, draws specific findings only.
        If PBL > 3mm, overlays BL visualization.
        If implant_guide provided, draws measurement line.
        If contours provided, draws them.
        """
        h, w = orig_img.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        # [MODIFIED] Dynamic Margin (1.2x = 20% padding)
        w_box = x2 - x1
        h_box = y2 - y1
        margin_x = max(50, int(w_box * 0.2)) # Guarantee at least 50px
        margin_y = max(50, int(h_box * 0.2))
        
        target_contours = []
        if nerve_contours: target_contours.extend(nerve_contours)
        if sinus_contours: target_contours.extend(sinus_contours)
        
        expansion_y1 = y1
        expansion_y2 = y2
        
        if target_contours:
            # Check for contours overlapping with X-range of box (plus margin)
            bx1, bx2 = x1 - margin_x, x2 + margin_x
            
            relevant_pts = []
            for cnt in target_contours:
                c_arr = np.array(cnt, dtype=np.float32).reshape(-1, 2) # [FIX] Ensure N x 2 shape
                # Filter points within X range
                mask = (c_arr[:, 0] >= bx1) & (c_arr[:, 0] <= bx2)
                pts_in_x = c_arr[mask]
                if len(pts_in_x) > 0:
                    relevant_pts.extend(pts_in_x)
            
            print(f"[DEBUG Crop] Box: {box}, Margins: {margin_x}, {margin_y}")
            print(f"[DEBUG Crop] Target Contours: {len(target_contours)}, Relevant Pts: {len(relevant_pts)}")

            print(f"[DEBUG Crop] Box: {box}, Margins: {margin_x}, {margin_y}")
            print(f"[DEBUG Crop] Target Contours: {len(target_contours)}, Relevant Pts: {len(relevant_pts)}")

            if relevant_pts:
                # Find min/max Y of relevant contour points
                rel_arr = np.array(relevant_pts)
                min_cy = np.min(rel_arr[:, 1])
                max_cy = np.max(rel_arr[:, 1])
                
                print(f"[DEBUG Crop] Nerve Y Range: {min_cy} ~ {max_cy}")
                
                # Expand crop to include these Y + margin
                # Ensure we capture the nerve at the bottom
                expansion_y1 = min(y1, min_cy)
                expansion_y2 = max(y2, max_cy)
        
        # Apply expansion
        x1 = max(0, int(x1 - margin_x))
        y1 = max(0, int(expansion_y1 - margin_y))
        x2 = min(w, int(x2 + margin_x))
        y2 = min(h, int(expansion_y2 + margin_y))
        
        # Crop Base
        crop = orig_img[y1:y2, x1:x2].copy()
        
        if debug_text:
             cv2.putText(crop, str(debug_text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        crop_h, crop_w = crop.shape[:2]
        
        # If PBL Issue, blend BL Viz
        if pbl_mm > 3.0 and bl_viz is not None:
             bl_crop = bl_viz[y1:y2, x1:x2]
             if bl_crop.shape == crop.shape:
                 mask = cv2.cvtColor(bl_crop, cv2.COLOR_BGR2GRAY)
                 _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                 mask_inv = cv2.bitwise_not(mask)
                 
                 img_bg = cv2.bitwise_and(crop, crop, mask=mask_inv)
                 img_fg = cv2.bitwise_and(bl_crop, bl_crop, mask=mask)
                 crop = cv2.add(img_bg, img_fg)


        
        # Draw Contours (Overlay) - Shifted
        def draw_cnts(cnts, color):
            if not cnts: return
            shifted_cnts = []
            for c in cnts:
                # c is list of [x, y]
                c_arr = np.array(c, dtype=np.int32)
                # Shift
                c_arr[:, 0] -= x1
                c_arr[:, 1] -= y1
                shifted_cnts.append(c_arr.reshape(-1, 1, 2))
            cv2.polylines(crop, shifted_cnts, False, color, 1)

        if tooth_contour:
            shifted_tc = np.array(tooth_contour, dtype=np.int32) - [x1, y1]
            cv2.polylines(crop, [shifted_tc], True, (0, 255, 0), 1) # Green for Tooth Mask




        # Draw Nerve as Outline Only (no fill)
        if nerve_contours:
            shifted_cnts = []
            for c in nerve_contours:
                c_arr = np.array(c, dtype=np.int32)
                c_arr[:, 0] -= x1
                c_arr[:, 1] -= y1
                shifted_cnts.append(c_arr.reshape(-1, 1, 2))
            
            # Draw red outline only (thickness 2 for visibility)
            cv2.polylines(crop, shifted_cnts, False, (0, 0, 255), 2)

        # Draw centerline (for angle visualization)
        if centerline:
            # Compute axis
            ax, ay = None, None
            if centerline_axis:
                ax, ay = centerline_axis
            elif tooth_contour:
                try:
                    pts = np.array(tooth_contour, dtype=np.float32)
                    if pts.ndim == 3:
                        pts = pts.reshape(-1, 2)
                    pts = pts - pts.mean(axis=0)
                    cov = np.cov(pts, rowvar=False)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    vec = eigvecs[:, np.argmax(eigvals)]
                    norm = max(np.linalg.norm(vec), 1e-6)
                    ax, ay = float(vec[0]/norm), float(vec[1]/norm)
                except Exception:
                    ax = ay = None
            if ax is None or ay is None:
                ax, ay = 0.0, 1.0  # default vertical

            # Start point: midpoint of the shorter side of the box (prefer root-side)
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            if bw <= bh:
                # shorter side is width -> use midpoint of left-right edge (vertical midline)
                cx = (box[0] + box[2]) / 2 - x1
                cy = (box[1] + box[3]) / 2 - y1
            else:
                # shorter side is height -> use midpoint of top-bottom edge (horizontal midline)
                cx = (box[0] + box[2]) / 2 - x1
                cy = (box[1] + box[3]) / 2 - y1
            # if multiple roots branching, contour PCA already biases axis; origin keeps center of short side

            length = max(crop_h, crop_w) * 0.6
            p1 = (int(cx - ax*length/2), int(cy - ay*length/2))
            p2 = (int(cx + ax*length/2), int(cy + ay*length/2))
            cv2.line(crop, p1, p2, (0, 255, 255), 2)

        # Sinus overlay disabled per user request
        # Sinus overlay disabled per user request


        # Draw Guide Line (Implant Guide OR Safety Guide)
        guide_data = implant_guide if implant_guide else safety_guide
        
        if guide_data:
            # Line Coords: New format [x1, y1, x2, y2]
            coords = guide_data.get('line_coords')
            dist_mm = guide_data.get('dist_mm', 0.0)
            
            if coords and len(coords) == 4:
                gx1, gy1, gx2, gy2 = coords
                # Shift to crop
                dcx1 = int(gx1 - x1)
                dcy1 = int(gy1 - y1)
                dcx2 = int(gx2 - x1)
                dcy2 = int(gy2 - y1)
                
                # compute distance from coords if not provided
                dx_px = dcx2 - dcx1
                dy_px = dcy2 - dcy1
                dist_px = (dx_px**2 + dy_px**2) ** 0.5
                scale_available = False
                if not dist_mm or dist_mm <= 0:
                    mm_per_px = guide_data.get('mm_per_px')
                    px_per_mm = guide_data.get('px_per_mm')
                    if not mm_per_px and px_per_mm:
                        mm_per_px = 1.0 / max(px_per_mm, 1e-6)
                    if mm_per_px:
                        dist_mm = dist_px * mm_per_px
                        guide_data['dist_mm'] = dist_mm
                        scale_available = True
                    else:
                        dist_mm = None
                        guide_data['dist_mm'] = None
                        guide_data['dist_px'] = dist_px
                else:
                    scale_available = True
                
                # Guide line color (Pink/Magenta)
                line_color = (255, 0, 255)
                
                # Draw Line
                cv2.line(crop, (dcx1, dcy1), (dcx2, dcy2), line_color, 2)
                
                # Draw Endpoints: Red @ Start (Bone/CEJ), Yellow @ End (Nerve)
                cv2.circle(crop, (dcx1, dcy1), 4, (0, 0, 255), -1) 
                cv2.circle(crop, (dcx2, dcy2), 3, (255, 0, 255), -1)
                
                # Draw Dist Text (Midpoint)
                mid_x = int((dcx1 + dcx2) / 2)
                mid_y = int((dcy1 + dcy2) / 2)
                if scale_available and dist_mm is not None:
                    txt = f"{dist_mm:.1f}mm"
                else:
                    txt = f"{dist_px:.1f}px (no scale)"
                
                # Ensure text is inside
                mid_x = np.clip(mid_x, 10, crop_w - 50)
                mid_y = np.clip(mid_y, 20, crop_h - 10)
                
                # Text with Outline
                cv2.putText(crop, txt, (int(mid_x) + 5, int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(crop, txt, (int(mid_x) + 5, int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 1, cv2.LINE_AA)
            
            # Reference text removed: now displays actual computed/received distance above

        # Draw Nerve Distance Measurement (User Request: CEJ to Nerve)
        # Draw Nerve Distance Measurement (User Request: CEJ to Nerve)
        if nerve_dist_line:
            if len(nerve_dist_line) == 4:
                p1 = (nerve_dist_line[0], nerve_dist_line[1])
                p2 = (nerve_dist_line[2], nerve_dist_line[3])
            else:
                p1, p2 = nerve_dist_line
            # Shift to crop coordinates
            p1_crop = (int(p1[0] - x1), int(p1[1] - y1))
            p2_crop = (int(p2[0] - x1), int(p2[1] - y1))
            
            # Draw Yellow Line Line (0, 255, 255) - Thicker
            cv2.line(crop, p1_crop, p2_crop, (0, 255, 255), 2)
            
            # Draw Endpoints
            cv2.circle(crop, p1_crop, 3, (0, 0, 255), -1) # Red at CEJ
            cv2.circle(crop, p2_crop, 3, (0, 255, 255), -1) # Yellow at Nerve
            
            # Draw Distance Text
            mid_x = (p1_crop[0] + p2_crop[0]) // 2
            mid_y = (p1_crop[1] + p2_crop[1]) // 2
            label = f"{nerve_dist_mm:.1f}mm"
            
            # Text Outline for visibility
            cv2.putText(crop, label, (mid_x+5, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
            cv2.putText(crop, label, (mid_x+5, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Draw Specific Findings (Caries/Perio Boxes)
        for f in findings:
            fx1, fy1, fx2, fy2 = map(int, f['box'])
            # Shift to crop coords
            cx1, cy1 = fx1 - x1, fy1 - y1
            cx2, cy2 = fx2 - x1, fy2 - y1
            
            color = (0, 0, 255) if f['type'] == 'caries' else (0, 165, 255) # Red for Caries, Orange for Perio
            cv2.rectangle(crop, (cx1, cy1), (cx2, cy2), color, 2)
            cv2.putText(crop, f['type'].upper(), (cx1, cy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return self._encode_image(crop)

    def generate(self, user_name, image_path, analysis_result, overlay_path=None, bl_viz_path=None, output_dir=None):
        """
        Main entry point.
        """
        # Load Images
        orig_img = cv2.imread(str(image_path))
        if orig_img is None: raise ValueError("Image not found")
        
        overlay_img = None
        if overlay_path and overlay_path.exists():
            overlay_img = cv2.imread(str(overlay_path))
            
        bl_viz_img = None
    def _generate_html(self, user_name, date_str, summary_data, findings, main_img, crop_source_img, bl_viz_img=None, missing_teeth=None, nerve_contours=None, sinus_contours=None, ai_commentary=None, llm_analysis=None, detailed_findings=None):
        """Generates HTML content"""
        if missing_teeth is None: missing_teeth = []
        
        # Load warning icon
        warning_icon_b64 = ""
        if self.warning_icon_path.exists():
            with open(self.warning_icon_path, "rb") as f:
                warning_icon_b64 = base64.b64encode(f.read()).decode('utf-8')
                
        # Load info icon
        info_icon_b64 = ""
        if self.info_icon_path.exists():
            with open(self.info_icon_path, "rb") as f:
                info_icon_b64 = base64.b64encode(f.read()).decode('utf-8')

        # Load logo
        logo_b64 = ""
        if self.logo_path.exists():
            with open(self.logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # New Clean White Design (from report guidence)
        style = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Pretendard', 'Malgun Gothic', 'Apple SD Gothic Neo', 'Segoe UI', sans-serif; 
            max-width: 1100px; 
            margin: 0 auto; 
            padding: 32px 48px; 
            color: #333; 
            background: #fff;
        }
        h1 { font-size: 2.5rem; font-weight: 700; margin-bottom: 8px; }
        h2 { font-size: 1.75rem; font-weight: 700; margin-bottom: 24px; color: #000; }
        h3 { font-size: 1.5rem; font-weight: 700; margin-bottom: 12px; }
        .header { margin-bottom: 48px; }
        .header p { font-size: 1.25rem; color: #666; margin-bottom: 32px; }
        .user-info { display: flex; align-items: center; gap: 16px; }
        .user-avatar { 
            width: 64px; height: 64px; 
            border-radius: 50%; 
            background: #000; 
            display: flex; align-items: center; justify-content: center;
            color: #fff; font-size: 1.5rem; font-weight: 600;
        }
        .user-details span { display: block; }
        .user-details .label { font-weight: 600; display: inline; }
        .divider-thick { border-top: 4px solid #000; margin: 32px 0; }
        .divider { border-top: 1px solid #ddd; margin: 32px 0; }
        .pano-container { margin-bottom: 32px; position: relative; }
        .pano-img { 
            width: 100%; 
            aspect-ratio: 16/9; 
            object-fit: contain; 
            border-radius: 8px; 
            background: #f5f5f5;
        }
        .pano-overlay {
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            pointer-events: none;
        }
        .pano-hotspot {
            fill: transparent;
            stroke: transparent;
            cursor: pointer;
            pointer-events: all;
            transition: all 0.2s ease;
        }
        .pano-hotspot:hover {
            fill: rgba(220, 38, 38, 0.2);
            stroke: #dc2626;
            stroke-width: 2;
        }
        .pano-hotspot.missing:hover {
            fill: rgba(156, 163, 175, 0.2);
            stroke: #9ca3af;
        }
        .pano-hotspot.implant:hover {
            fill: rgba(37, 99, 235, 0.2);
            stroke: #2563eb;
        }
        .legend { 
            display: flex; 
            justify-content: center; 
            gap: 32px; 
            margin: 24px 0;
            flex-wrap: wrap;
        }
        .legend-item { display: flex; align-items: center; gap: 8px; font-weight: 600; }
        .legend-bar { width: 32px; height: 4px; }
        .legend-bar.healthy { background: #000; }
        .legend-bar.treatment { background: #dc2626; }
        .legend-bar.implant { background: #2563eb; }
        .legend-bar.missing { background: #d1d5db; }
        
        /* Odontogram Styles */
        .odontogram { 
            margin: 32px 0; 
            text-align: center;
            background: #fafafa;
            padding: 24px;
            border-radius: 12px;
        }
        .odontogram h3 { margin-bottom: 16px; font-size: 1.1rem; color: #374151; }
        .odonto-row { 
            display: flex; 
            justify-content: center; 
            gap: 4px; 
            margin: 8px 0;
        }
        .odonto-tooth { 
            width: 40px; 
            height: 60px; 
            position: relative;
            cursor: pointer;
            transition: transform 0.2s, background 0.2s;
            border-radius: 8px;
            padding: 4px;
            background: transparent;
        }
        .odonto-tooth:hover { transform: scale(1.1); background: rgba(0,0,0,0.05); }
        .odonto-tooth img { 
            width: 100%; 
            height: 100%; 
            object-fit: contain;
        }
        .odonto-tooth.flip img { transform: scaleX(-1); }
        
        /* Healthy - green background */
        .odonto-tooth.healthy { background: rgba(34, 197, 94, 0.15); }
        .odonto-tooth.healthy .odonto-label { color: #16a34a; font-weight: 700; }
        
        /* Treatment - red background */
        .odonto-tooth.treatment { background: rgba(220, 38, 38, 0.2); }
        .odonto-tooth.treatment .odonto-label { color: #dc2626; font-weight: 700; }
        
        /* Missing - gray dotted border */
        .odonto-tooth.missing { 
            background: rgba(156, 163, 175, 0.1); 
            border: 2px dashed #9ca3af;
        }
        .odonto-tooth.missing img { opacity: 0.3; }
        .odonto-tooth.missing .odonto-label { color: #9ca3af; }
        
        /* Implant - blue background */
        .odonto-tooth.implant { background: rgba(37, 99, 235, 0.2); }
        .odonto-tooth.implant .odonto-label { color: #2563eb; font-weight: 700; }

        /* Overlap - amber background */
        .odonto-tooth.overlap { background: rgba(217, 119, 6, 0.2); }
        .odonto-tooth.overlap .odonto-label { color: #d97706; font-weight: 700; }
        
        .odonto-label { 
            position: absolute; 
            bottom: -18px; 
            left: 50%; 
            transform: translateX(-50%);
            font-size: 10px;
            color: #666;
        }
        .odonto-quadrant { display: flex; gap: 2px; }
        .odonto-quadrant.flip { direction: ltr; }
        .odonto-separator { width: 20px; }
        
        .tooth-card { 
            border: 2px solid #e5e7eb; 
            border-radius: 16px; 
            padding: 32px; 
            margin-bottom: 24px;
            display: flex; 
            gap: 24px; 
            background: #fff;
        }
        .tooth-card.treatment .finding-tag { color: #dc2626; }
        .tooth-card.overlap .finding-tag { color: #d97706; }
        .tooth-card.missing .finding-tag { color: #9ca3af; }
        .tooth-card.implant .finding-tag { color: #2563eb; }
        .tooth-card.healthy .finding-tag { color: #22c55e; }
        .tooth-card.treatment { border-left: 5px solid #dc2626; }
        .tooth-card.overlap { border-left: 5px solid #d97706; }
        .tooth-card.missing { border-left: 5px solid #9ca3af; }
        .tooth-card.implant { border-left: 5px solid #2563eb; }
        .tooth-card.healthy { border-left: 5px solid #22c55e; }
        .crop-img { 
            width: 192px; 
            height: 256px; 
            object-fit: contain; 
            border-radius: 8px; 
            background: #f5f5f5;
            flex-shrink: 0;
        }
        .tooth-details { flex: 1; }
        .tooth-details h3 { font-size: 1.75rem; margin-bottom: 12px; }
        .finding-tag { 
            display: inline-flex; 
            align-items: center; 
            gap: 6px; 
            color: #dc2626; 
            font-weight: 600; 
            font-size: 1.25rem;
            margin-bottom: 16px;
        }
        .finding-tag svg, .finding-tag img { width: 30px; height: 30px; }
        .description { color: #6b7280; font-size: 0.9rem; margin-bottom: 24px; line-height: 1.6; }
        .treatments h4 { font-size: 1.25rem; font-weight: 700; margin-bottom: 12px; }
        .treatments ol { padding-left: 20px; color: #374151; }
        .treatments li { margin-bottom: 8px; }
        .badge { display:inline-block; padding:4px 8px; margin-right:6px; border-radius:6px; font-size:0.85rem; font-weight:600; color:#fff; }
        .badge.pbl { background:#6b21a8; }
        .badge.perio { background:#dc2626; }
        .badge.caries { background:#f97316; }
        .badge.overlap { background:#d97706; }
        .badge.treatment { background:#2563eb; }
        .badge.info { background:#6b7280; }
        .badge.implant { background:#0ea5e9; }
        .footer { 
            display: flex; 
            align-items: flex-start; 
            gap: 16px; 
            margin-top: 48px; 
            padding-top: 32px;
            border-top: 1px solid #ddd;
        }
        .footer-logo { color: #06b6d4; font-weight: 700; font-size: 1.5rem; }
        .footer-warning { font-size: 0.875rem; }
        .footer-warning strong { display: block; margin-bottom: 8px; }
        
        /* New Summary Box */
        .summary-box { 
            background: #f0f9ff; 
            padding: 24px; 
            border-radius: 12px; 
            margin-bottom: 32px; 
            border: 1px solid #bae6fd; 
        }
        .summary-title { margin-top:0; color:#0369a1; font-size: 1.4rem; margin-bottom: 16px; }
        .summary-grid { display: flex; gap: 40px; }
        .summary-item { display: flex; flex-direction: column; }
        .summary-val { font-size: 2rem; font-weight: 700; color: #0284c7; }
        .summary-label { font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
        
        .badge.success { background: #16a34a; } /* Green for safe implant */
        
        /* AI Insight Box */
        .ai-box {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid #86efac;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 32px;
            position: relative;
        }
        .ai-title {
            color: #15803d; 
            font-size: 1.2rem; 
            font-weight: 700; 
            margin-bottom: 12px; 
            display: flex; 
            align-items: center; 
            gap: 8px;
        }
        .ai-content {
            color: #14532d;
            font-size: 1.05rem;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        """

        user_initial = user_name[0].upper() if user_name else "U"

        pano_img_b64 = self._encode_image(main_img)
        # Draw centerlines for all teeth on pano_img
        # Draw aligned axes for adjacent teeth (green) and guide axis (pink) on pano
        try:
            pano_draw = main_img.copy()
            # draw guide axis if available (pink)
            for mt in (missing_teeth or []):
                if isinstance(mt, dict):
                    guide = mt.get('implant_guide')
                    box = mt.get('box')
                    if guide and guide.get('line_coords') and box:
                        gx1, gy1, gx2, gy2 = map(int, guide['line_coords'])
                        cv2.line(pano_draw, (gx1, gy1), (gx2, gy2), (255, 0, 255), 2)
            # draw adjacent tooth axes (green) for teeth with contours
            for t in findings:
                contour = t.get('contour')
                if not contour: continue
                try:
                    pts = np.array(contour, dtype=np.float32)
                    if pts.ndim == 3:
                        pts = pts.reshape(-1,2)
                    centroid = pts.mean(axis=0)
                    pts_c = pts - centroid
                    cov = np.cov(pts_c, rowvar=False)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    vec = eigvecs[:, np.argmax(eigvals)]
                    norm = max(np.linalg.norm(vec),1e-6)
                    ax, ay = float(vec[0]/norm), float(vec[1]/norm)
                    spread = np.max(np.linalg.norm(pts_c, axis=1))
                    length = max(spread*1.2, 40)
                    p1 = (int(centroid[0] - ax*length/2), int(centroid[1] - ay*length/2))
                    p2 = (int(centroid[0] + ax*length/2), int(centroid[1] + ay*length/2))
                    cv2.line(pano_draw, p1, p2, (0,255,0), 2)
                except Exception as e:
                    print(f"[PANO AXIS] failed tooth {t.get('tooth_label')}: {e}")
            pano_img_b64 = self._encode_image(pano_draw)
        except Exception as e:
            print(f"[PANO AXIS] draw failed: {e}")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Dental Analysis Report - {user_name}</title>
            <style>
{style}
            
            /* Toggle Button Style */
            .notation-toggle {{
                background: #fff;
                cursor: pointer;
                border: 1px solid #ddd;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.8rem;
                color: #666;
                display: inline-flex;
                align-items: center;
                gap: 5px;
                transition: all 0.2s;
            }}
            .notation-toggle:hover {{ background: #f9fafb; border-color: #bbb; }}
            
            /* Logic for switching display */
            body.use-universal .format-fdi {{ display: none !important; }}
            body.use-universal .format-univ {{ display: inline !important; }}
            
            body:not(.use-universal) .format-fdi {{ display: inline !important; }}
            body:not(.use-universal) .format-univ {{ display: none !important; }}
            </style>
            <script>
            window.toggleNotation = function() {{
                document.body.classList.toggle('use-universal');
                const btn = document.getElementById('btn-notation');
                const isUniv = document.body.classList.contains('use-universal');
                if(btn) btn.innerHTML = isUniv ? '🔄 Notation: <strong>Universal (1-32)</strong>' : '🔄 Notation: <strong>FDI (11-48)</strong>';
            }};
            </script>
        </head>
        <body>
            <div class="header">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <img src="data:image/png;base64,{logo_b64}" style="height:40px; margin-bottom:10px; display: block;">
                    <button id="btn-notation" class="notation-toggle" onclick="window.toggleNotation()">🔄 Notation: <strong>FDI (11-48)</strong></button>
                </div>
                <h1>Dental AI Analysis Report</h1>
                <p>Generated by Dental AI Assistant</p>
                
                <div class="user-info">
                    <div class="user-avatar">{user_initial}</div>
                    <div class="user-details">
                        <span><span class="label">Name</span> {user_name}</span>
                        <span><span class="label">Date</span> {date_str}</span>
                    </div>
                </div>
            </div>
            
            <div class="divider-thick"></div>

            <!-- EXECUTIVE SUMMARY -->
            <div class="summary-box">
                <h2 class="summary-title">Clinical Summary</h2>
                <div class="summary-grid">
                    <div class="summary-item">
                        <span class="summary-val">{len(missing_teeth)}</span>
                        <span class="summary-label">Missing Teeth</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-val">{len(summary_data.get('caries', []))}</span>
                        <span class="summary-label">Caries Detected</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-val">{len(summary_data.get('periapical', []))}</span>
                        <span class="summary-label">Apical Lesions</span>
                    </div>
                    <div class="summary-item" style="margin-left:auto; justify-content:center;">
                        <span class="badge {'treatment' if (missing_teeth or summary_data.get('caries')) else 'healthy'}" style="font-size:1.0rem; padding:8px 16px;">
                            {'Treatment Recommended' if (missing_teeth or summary_data.get('caries')) else 'Routine Checkup'}
                        </span>
                    </div>
                </div>
            </div>

            <!-- AI INSIGHT -->
            {f'''
            <div class="ai-box">
                <div class="ai-title">🤖 Dr. AI's Insight (종합 소견)</div>
                <div class="ai-content">{ai_commentary}</div>
            </div>
            ''' if ai_commentary else ''}

            <div class="pano-container" id="pano-wrapper">
                <h2>Full Panorama <span style="font-size:0.7em;color:#666;font-weight:normal;">(Click on highlighted areas)</span></h2>
                <div style="position:relative;display:inline-block;width:100%;">
                    <img class="pano-img" id="pano-img" src="data:image/jpeg;base64,{pano_img_b64}" alt="Full Panorama">
                    <svg class="pano-overlay" id="pano-svg" viewBox="0 0 1000 500" preserveAspectRatio="none">
                        <!-- Hotspots will be added dynamically -->
                    </svg>
                </div>
            </div>
            
            <div class="legend">
                <div class="legend-item"><div class="legend-bar healthy"></div>Healthy</div>
                <div class="legend-item"><div class="legend-bar treatment"></div><span style="color:#dc2626">Requires treatment</span></div>
                <div class="legend-item"><div class="legend-bar implant"></div><span style="color:#2563eb">Implant</span></div>
                <div class="legend-item"><div class="legend-bar missing"></div><span style="color:#9ca3af">Missing</span></div>
            </div>
            
            <!-- ODONTOGRAM_PLACEHOLDER -->
            
            <div class="divider"></div>
            
            <h2>Detailed Findings</h2>
        """
        
        # Build tooth status map
        tooth_status = {}  # FDI -> 'healthy'/'treatment'/'missing'/'implant'
        
        # Mark all detected teeth as healthy initially (or treatment if issues found)
        for tooth in findings:
            lbl = str(tooth.get('tooth_label', ''))
            if not lbl: continue
            
            status = 'healthy'
            
            # Check for issues in 'findings' list (mapped in routes_v2)
            tooth_findings = tooth.get('findings', [])
            has_issue = False
            for f in tooth_findings:
                ctype = f.get('type')
                if ctype in ('caries', 'periapical', 'perio'):
                    has_issue = True
                    break
            
            # Or PBL logic check
            pbl = tooth.get('pbl_mm', 0)
            if pbl > 3.0: has_issue = True
            
            if has_issue:
                status = 'treatment'
            elif tooth.get('nerve_overlap') or tooth.get('sinus_overlap'):
                status = 'overlap'
            
            tooth_status[lbl] = status
        
        # Fallback: Mark teeth with issues from summary_data (if populated)
        for c in summary_data.get('caries', []):
            lbl = str(c.get('tooth_label', ''))
            if lbl: tooth_status[lbl] = 'treatment'
        
        for p in summary_data.get('periapical', []):
            lbl = str(p.get('tooth_label', ''))
            if lbl: tooth_status[lbl] = 'treatment'
        
        # Mark missing teeth
        for mt in (missing_teeth or []):
            if isinstance(mt, dict):
                lbl = str(mt.get('tooth_label', ''))
                if lbl:
                    tooth_status[lbl] = 'missing'
        
        # Generate odontogram HTML with Toggle
        def build_odonto_row_toggle(jaw, quadrants):
            """
            Builds a row of teeth with support for both FDI and Universal numbering hidden/shown via JS.
            quadrants: list of tuples (q_num, list_of_positions, reverse_list, flip_image)
            Example: ('1', [8,7,6,5,4,3,2,1], False, False)
            """
            html_row = '<div class="odonto-row">'
            
            for qi, (q_prefix, positions, reverse, flip) in enumerate(quadrants):
                html_row += '<div class="odonto-quadrant">'
                
                # Determine traversal order for rendering
                pos_list = list(reversed(positions)) if reverse else positions
                
                for pos in pos_list:
                    # FDI definition
                    fdi_str = f"{q_prefix}{pos}"
                    
                    # Universal Definition
                    # We rely on self.get_universal_label()
                    univ_str = self.get_universal_label(fdi_str)
                    
                    status = tooth_status.get(fdi_str, '')
                    
                    # Image loading
                    icon_b64 = self._load_odonto_icon(jaw, pos)
                    flip_class = 'flip' if flip else ''
                    
                    if icon_b64:
                        html_row += f'''
                        <div class="odonto-tooth {status} {flip_class}" onclick="document.getElementById('tooth-{fdi_str}')?.scrollIntoView({{behavior:'smooth',block:'center'}})">
                            <img src="data:image/png;base64,{icon_b64}" alt="Tooth {fdi_str}">
                            <!-- Labels container -->
                            <span class="odonto-label num-fdi">{fdi_str}</span>
                            <span class="odonto-label num-univ" style="display:none; color:#4b5563;">{univ_str}</span>
                        </div>
                        '''
                html_row += '</div>'
                if qi == 0:
                    html_row += '<div class="odonto-separator"></div>'
            
            html_row += '</div>'
            return html_row

        # Upper jaw: Q1 (18-11) | Q2 (21-28)
        upper_row = build_odonto_row_toggle('up', [
            ('1', [8,7,6,5,4,3,2,1], False, False), 
            ('2', [1,2,3,4,5,6,7,8], False, True)
        ])
        
        # Lower jaw: Q4 (48-41) | Q3 (31-38)
        lower_row = build_odonto_row_toggle('down', [
            ('4', [8,7,6,5,4,3,2,1], False, False),
            ('3', [1,2,3,4,5,6,7,8], False, True)
        ])
        
        odontogram_html = f'''
        <div class="odontogram" style="position:relative;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h3 style="margin:0;">Dental Chart</h3>
                <!-- Toggle Buttons -->
                <div class="toggle-container" style="display:flex; gap:0;">
                    <button id="btn-fdi" class="toggle-btn active" onclick="setNumbering('fdi')">FDI</button>
                    <button id="btn-univ" class="toggle-btn" onclick="setNumbering('univ')">Universal</button>
                </div>
            </div>
            
            {upper_row}
            <div style="height:16px;border-bottom:2px solid #ddd;margin:0 auto;width:80%;"></div>
            {lower_row}
            
            <style>
                .toggle-btn {{
                    padding: 4px 10px;
                    border: 1px solid #d1d5db;
                    background: #f3f4f6;
                    cursor: pointer;
                    font-size: 0.8rem;
                    color: #4b5563;
                }}
                .toggle-btn:first-child {{ border-radius: 4px 0 0 4px; border-right:none; }}
                .toggle-btn:last-child {{ border-radius: 0 4px 4px 0; border-left:none; }}
                .toggle-btn.active {{
                    background: #2563eb;
                    color: white;
                    border-color: #2563eb;
                }}
            </style>
            
            <script>
                function setNumbering(type) {{
                    // Update Buttons
                    document.getElementById('btn-fdi').className = type === 'fdi' ? 'toggle-btn active' : 'toggle-btn';
                    document.getElementById('btn-univ').className = type === 'univ' ? 'toggle-btn active' : 'toggle-btn';
                    
                    // Toggle Labels on Odontogram
                    const fdiLabels = document.querySelectorAll('.num-fdi');
                    const univLabels = document.querySelectorAll('.num-univ');
                    
                    fdiLabels.forEach(el => el.style.display = (type === 'fdi') ? 'block' : 'none');
                    univLabels.forEach(el => el.style.display = (type === 'univ') ? 'block' : 'none');
                    
                    // Toggle Headers in Detailed Cards (using format-fdi/format-univ classes)
                    document.querySelectorAll('.format-fdi').forEach(el => el.style.display = (type === 'fdi') ? 'inline' : 'none');
                    document.querySelectorAll('.format-univ').forEach(el => el.style.display = (type === 'univ') ? 'inline' : 'none');
                }}
            </script>
        </div>
        '''
        
        html = html.replace('<!-- ODONTOGRAM_PLACEHOLDER -->', odontogram_html)
        
        # Generate SVG hotspots for problematic teeth
        img_h, img_w = main_img.shape[:2]
        svg_hotspots = []
        
        # Collect all teeth that have issues (will add to detail section)
        issue_teeth = set()
        for mt in (missing_teeth or []):
            if isinstance(mt, dict) and mt.get('tooth_label'):
                issue_teeth.add(str(mt['tooth_label']))
        
        for tooth in findings:
            lbl = tooth.get('tooth_label')
            box = tooth.get('box')
            if not lbl or not box:
                continue
            
            # Check if this tooth has issues
            has_caries = any(c.get('tooth_label') == lbl for c in summary_data.get('caries', []))
            has_perio = any(p.get('tooth_label') == lbl for p in summary_data.get('periapical', []))
            pbl_val = tooth.get('pbl_mm', 0)
            
            if has_caries or has_perio or pbl_val > 3:
                issue_teeth.add(str(lbl))
                # Convert box to percentage for SVG viewBox (1000x500)
                x1, y1, x2, y2 = box
                sx = (x1 / img_w) * 1000
                sy = (y1 / img_h) * 500
                sw = ((x2 - x1) / img_w) * 1000
                sh = ((y2 - y1) / img_h) * 500
                
                svg_hotspots.append(f'''<rect class="pano-hotspot" x="{sx:.1f}" y="{sy:.1f}" width="{sw:.1f}" height="{sh:.1f}" 
                    onclick="document.getElementById('tooth-{lbl}').scrollIntoView({{behavior:'smooth',block:'center'}})" 
                    data-tooth="{lbl}"><title>Tooth #{lbl} - Click to view details</title></rect>''')
        
        # Also add missing teeth to hotspots
        for mt in (missing_teeth or []):
            if isinstance(mt, dict):
                lbl = mt.get('tooth_label')
                box = mt.get('box')
                if lbl and box:
                    x1, y1, x2, y2 = box
                    sx = (x1 / img_w) * 1000
                    sy = (y1 / img_h) * 500
                    sw = ((x2 - x1) / img_w) * 1000
                    sh = ((y2 - y1) / img_h) * 500
                    
                    svg_hotspots.append(f'''<rect class="pano-hotspot missing" x="{sx:.1f}" y="{sy:.1f}" width="{sw:.1f}" height="{sh:.1f}" 
                        onclick="document.getElementById('tooth-{lbl}').scrollIntoView({{behavior:'smooth',block:'center'}})" 
                        data-tooth="{lbl}" style="stroke:#9ca3af;stroke-width:1;stroke-dasharray:5,3;"><title>Missing Tooth #{lbl}</title></rect>''')
        
        # Insert hotspots into HTML
        if svg_hotspots:
            hotspots_html = '\n'.join(svg_hotspots)
            html = html.replace('<!-- Hotspots will be added dynamically -->', hotspots_html)

        # Problematic Teeth Generation
        # Logic: PBL > 3mm OR Caries OR Periapical OR Bone Loss
        
        has_issues = False
        
        # 0. Missing Teeth Section
        if missing_teeth:
            has_issues = True
            
            # Build lookup for neighboring teeth by FDI
            teeth_by_fdi = {}
            for t in findings:
                fdi = t.get('tooth_label')
                if fdi and t.get('box'):
                    teeth_by_fdi[str(fdi)] = t
            
            def get_neighbors(fdi_str):
                """Find left and right neighbors based on FDI numbering."""
                try:
                    fdi = int(fdi_str)
                except:
                    return None, None
                q = fdi // 10
                n = fdi % 10
                # Same quadrant neighbors
                left_fdi = f"{q}{n-1}" if n > 1 else None
                right_fdi = f"{q}{n+1}" if n < 8 else None
                left_t = teeth_by_fdi.get(left_fdi)
                right_t = teeth_by_fdi.get(right_fdi)
                return left_t, right_t
            
            for mt in missing_teeth:
                # Handle both dict (new) and string (legacy/fallback)
                if isinstance(mt, dict):
                    lbl = mt.get('tooth_label', '?')
                    box = mt.get('box')
                    guide = mt.get('implant_guide')
                else:
                    lbl = str(mt)
                    box = None
                    guide = None
                
                # Molar-only distance inference (16-18, 26-28, 36-38, 46-48)
                def _is_molar(fdi_str):
                    try:
                        f = int(fdi_str)
                        return f in (16,17,18,26,27,28,36,37,38,46,47,48)
                    except:
                        return False
                is_molar = _is_molar(lbl)

                crop_html = ""
                implant_badge = ""
                implant_note = ""

                def _axis_from_contour(contour):
                    """Return principal axis unit vector (vx, vy) from contour points."""
                    try:
                        pts = np.array(contour, dtype=np.float32)
                        if pts.ndim == 3:
                            pts = pts.reshape(-1, 2)
                        pts = pts - pts.mean(axis=0)
                        cov = np.cov(pts, rowvar=False)
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        # principal axis = eigenvector with largest eigenvalue
                        vec = eigvecs[:, np.argmax(eigvals)]
                        # ensure not zero
                        norm = max(np.linalg.norm(vec), 1e-6)
                        return float(vec[0]/norm), float(vec[1]/norm)
                    except Exception as e:
                        print(f"[GUIDE AXIS] PCA failed for contour: {e}")
                        return None

                def _axis_from_adjacent_floor(l_box, r_box):
                    """Tangent from adjacent tooth bottoms (lower-edge midpoints)."""
                    try:
                        lx1, ly1, lx2, ly2 = map(int, l_box)
                        rx1, ry1, rx2, ry2 = map(int, r_box)
                        lp = ((lx1+lx2)/2.0, max(ly1, ly2))
                        rp = ((rx1+rx2)/2.0, max(ry1, ry2))
                        vx = rp[0]-lp[0]
                        vy = rp[1]-lp[1]
                        return vx, vy
                    except Exception as e:
                        print(f"[GUIDE AXIS] adjacent floor tangent failed: {e}")
                        return None

                def _axis_pca(contour):
                    """Return principal axis unit vector (vx, vy) from contour points."""
                    try:
                        pts = np.array(contour, dtype=np.float32)
                        if pts.ndim == 3:
                            pts = pts.reshape(-1, 2)
                        pts = pts - pts.mean(axis=0)
                        cov = np.cov(pts, rowvar=False)
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        vec = eigvecs[:, np.argmax(eigvals)]
                        norm = max(np.linalg.norm(vec), 1e-6)
                        return float(vec[0]/norm), float(vec[1]/norm)
                    except Exception as e:
                        print(f"[GUIDE AXIS] PCA failed: {e}")
                        return None
                # Generate Crop if box exists
                if box:
                    try:
                        # Find neighboring teeth for expanded crop
                        left_t, right_t = get_neighbors(lbl)
                        
                        # Get original box center
                        bx1, by1, bx2, by2 = map(int, box)
                        center_x = (bx1 + bx2) // 2
                        center_y = (by1 + by2) // 2
                        
                        # Use neighbor's dimensions as reference for proper sizing
                        if left_t and left_t.get('box'):
                            ref_box = left_t['box']
                            ref_w = int(ref_box[2] - ref_box[0])
                            ref_h = int(ref_box[3] - ref_box[1])
                        elif right_t and right_t.get('box'):
                            ref_box = right_t['box']
                            ref_w = int(ref_box[2] - ref_box[0])
                            ref_h = int(ref_box[3] - ref_box[1])
                        else:
                            # Fallback: use original box but cap dimensions
                            ref_w = min(bx2 - bx1, 200)
                            ref_h = min(by2 - by1, 300)
                        
                        # Create focused crop: 2.5 teeth wide (left tooth + gap + right extension)
                        crop_width = int(ref_w * 2.5)
                        crop_height = int(ref_h * 1.3)  # A bit taller for context
                        
                        # Calculate new box centered on missing tooth
                        new_bx1 = center_x - crop_width // 2
                        new_by1 = center_y - crop_height // 2
                        new_bx2 = center_x + crop_width // 2
                        new_by2 = center_y + crop_height // 2
                        
                        # If we have a left neighbor, shift to include it
                        if left_t and left_t.get('box'):
                            l_box = left_t['box']
                            new_bx1 = min(new_bx1, int(l_box[0]))
                            new_by1 = min(new_by1, int(l_box[1]))
                            new_by2 = max(new_by2, int(l_box[3]))
                        
                        # If we have a right neighbor, extend to include it
                        if right_t and right_t.get('box'):
                            r_box = right_t['box']
                            new_bx2 = max(new_bx2, int(r_box[2]))
                            new_by1 = min(new_by1, int(r_box[1]))
                            new_by2 = max(new_by2, int(r_box[3]))
                        
                        expanded_box = [new_bx1, new_by1, new_bx2, new_by2]
                        print(f"[CROP DEBUG] Tooth {lbl}: center=({center_x},{center_y}), ref_size=({ref_w}x{ref_h})")
                        print(f"[CROP DEBUG] expanded_box={expanded_box}")
                        
                        implant_guide = mt.get('implant_guide')
                        debug_msg = mt.get('guide_status')

                        center_axis = None
                        # Interpolate implant guide axis if missing (Axis-Interpolated, jaw-aware)
                        if implant_guide and isinstance(implant_guide, dict) and not implant_guide.get('line_coords') and is_molar:
                            try:
                                # Strategy: if both contours present, use shortest segment between them
                                left_contour = left_t.get('contour') if left_t else None
                                right_contour = right_t.get('contour') if right_t else None
                                if left_contour is not None and right_contour is not None:
                                    lc = np.array(left_contour, dtype=np.float32).reshape(-1,2)
                                    rc = np.array(right_contour, dtype=np.float32).reshape(-1,2)
                                    # compute pairwise distances
                                    dists = np.linalg.norm(lc[:,None,:] - rc[None,:,:], axis=2)
                                    idx = np.unravel_index(np.argmin(dists), dists.shape)
                                    p_left = lc[idx[0]]
                                    p_right = rc[idx[1]]
                                    vec = p_right - p_left
                                    norm = max(np.linalg.norm(vec), 1e-6)
                                    ax, ay = vec / norm
                                    center_axis = (float(ax), float(ay))
                                    # line from left point to right point
                                    implant_guide['line_coords'] = [float(p_left[0]), float(p_left[1]), float(p_right[0]), float(p_right[1])]
                                    try:
                                        quad = int(str(lbl)[0])
                                    except:
                                        quad = 0
                                    implant_guide['mode'] = implant_guide.get('mode') or ('upper' if quad in (1,2) else 'lower')
                                    implant_guide.setdefault('dist_mm', 0.0)
                                else:
                                    # 2) one contour -> that axis
                                    left_axis = _axis_pca(left_contour) if left_contour is not None else None
                                    right_axis = _axis_pca(right_contour) if right_contour is not None else None
                                    if left_axis or right_axis:
                                        ax, ay = (left_axis or right_axis)
                                        center_axis = (ax, ay)
                                        try:
                                            quad = int(str(lbl)[0])
                                        except:
                                            quad = 0
                                        if quad in (1, 2):
                                            ay = abs(ay)
                                        elif quad in (3, 4):
                                            ay = -abs(ay)
                                        center_axis = (ax, ay)
                                        axis_len = max(ref_h * 1.2, 80)
                                        gx1, gy1 = center_x, center_y
                                        gx2 = int(center_x + ax * axis_len)
                                        gy2 = int(center_y + ay * axis_len)
                                        implant_guide['line_coords'] = [gx1, gy1, gx2, gy2]
                                        implant_guide['mode'] = implant_guide.get('mode') or ('upper' if quad in (1,2) else 'lower')
                                        implant_guide.setdefault('dist_mm', 0.0)
                            except Exception as e:
                                print(f"[GUIDE AXIS] Interpolation failed for missing tooth {lbl}: {e}")

                        # Final fallback for molars: vertical axis through gap center
                        if is_molar and (not implant_guide or not implant_guide.get('line_coords')):
                            try:
                                gx1 = center_x
                                gy1 = center_y
                                # direction by jaw
                                quad = int(str(lbl)[0]) if str(lbl)[:1].isdigit() else 0
                                axis_len = max(ref_h * 1.2 if 'ref_h' in locals() else (by2 - by1), 100)
                                if quad in (1,2):  # upper
                                    ax, ay = center_axis if center_axis else (0,1)
                                    gx2, gy2 = int(center_x + ax*axis_len), int(center_y + abs(ay)*axis_len)
                                else:              # lower default
                                    ax, ay = center_axis if center_axis else (0,-1)
                                    gx2, gy2 = int(center_x + ax*axis_len), int(center_y - abs(ay)*axis_len)
                                center_axis = center_axis or (0, 1 if quad in (1,2) else -1)
                                if not implant_guide or not isinstance(implant_guide, dict):
                                    implant_guide = {}
                                implant_guide['line_coords'] = [gx1, gy1, gx2, gy2]
                                implant_guide['mode'] = implant_guide.get('mode') or ('upper' if quad in (1,2) else 'lower')
                            except Exception as e:
                                print(f"[GUIDE AXIS] Fallback axis failed for tooth {lbl}: {e}")

                        # Check if implant_guide is already provided by pano_inference (Prioritize this!)
                        if guide and guide.get('line_coords'):
                            print(f"[REPORT] Using pre-calculated implant guide for {lbl}: {guide['line_coords']}")
                            implant_guide = guide
                            # Ensure mode is set
                            if 'mode' not in implant_guide:
                                implant_guide['mode'] = 'upper' if str(lbl).startswith(('1','2')) else 'lower'
                        else:
                            # Fallback: Isosceles-based centerline using adjacent centers and nearest nerve point (all missing teeth)
                            try:
                                if left_t and right_t and left_t.get('box') and right_t.get('box'):
                                    lx1, ly1, lx2, ly2 = map(int, left_t['box'])
                                    rx1, ry1, rx2, ry2 = map(int, right_t['box'])
                                    lc = ((lx1+lx2)/2.0, (ly1+ly2)/2.0)
                                    rc = ((rx1+rx2)/2.0, (ry1+ry2)/2.0)
                                    base_mid = ((lc[0]+rc[0])/2.0, (lc[1]+rc[1])/2.0)
                                    nerve_pt = None
                                    if nerve_contours:
                                        pts = np.concatenate([np.array(c) for c in nerve_contours], axis=0)
                                        dists = ((pts[:,0]-base_mid[0])**2 + (pts[:,1]-base_mid[1])**2)
                                        idx = int(np.argmin(dists))
                                        nerve_pt = (float(pts[idx,0]), float(pts[idx,1]))
                                    if nerve_pt:
                                        ax = nerve_pt[0] - base_mid[0]
                                        ay = nerve_pt[1] - base_mid[1]
                                        norm = max((ax*ax + ay*ay) ** 0.5, 1e-6)
                                        center_axis = (ax/norm, ay/norm)
                                        if not implant_guide or not isinstance(implant_guide, dict):
                                            implant_guide = {}
                                        implant_guide['line_coords'] = [base_mid[0], base_mid[1], nerve_pt[0], nerve_pt[1]]
                                        implant_guide['mode'] = implant_guide.get('mode') or ('upper' if str(lbl).startswith(('1','2')) else 'lower')
                                        implant_guide.setdefault('dist_mm', 0.0)
                            except Exception as e:
                                print(f"[GUIDE AXIS] isosceles centerline failed for tooth {lbl}: {e}")

                        # Get tooth contours for adjacent teeth if nerve overlap
                        neighbor_contours = []
                        if left_t and left_t.get('contour'):
                            neighbor_contours.append(left_t['contour'])
                        if right_t and right_t.get('contour'):
                            neighbor_contours.append(right_t['contour'])
                        
                        # Pass contours to overlay on the crop
                        crop_b64 = self._crop_and_overlay(
                            crop_source_img, 
                            expanded_box, 
                            [], 
                            bl_viz_img, 
                            0, 
                            implant_guide=implant_guide if is_molar else None,
                            # Upper jaw (1x, 2x): no nerve. Lower jaw (3x, 4x): show nerve
                            nerve_contours=nerve_contours if is_molar and str(lbl).startswith(('3', '4')) else None,
                            sinus_contours=None,  # Already disabled
                            debug_text=debug_msg,
                            nerve_dist_line=mt.get('nerve_dist_line') if is_molar else None,
                            nerve_dist_mm=mt.get('nerve_dist_mm', guide.get('dist_mm', 0) if guide else 0) if is_molar else 0,
                            centerline=False,
                            centerline_axis=None
                        )

                        
                        crop_html = f"""
                            <img class="crop-img" src="data:image/jpeg;base64,{crop_b64}" alt="Missing Tooth #{lbl}">
                        """
                    except Exception as e:
                        print(f"Error cropping missing tooth {lbl}: {e}")

                # Sync updated guide back for badge logic (molars only)
                guide = implant_guide if (implant_guide and is_molar) else guide

                # Implant suitability logic (User Request: 9mm Threshold)
                if guide and isinstance(guide, dict):
                    dist = guide.get('dist_mm', 0)
                    mode = guide.get('mode', 'lower')
                    
                    if dist is None or dist <= 0:
                        implant_badge = '<span class="badge info">No Scale</span>'
                        implant_note = "Scale calibration needed for accurate measurement."
                    elif dist >= 7.0:
                        implant_badge = '<span class="badge success">Implant: Simple Placement Possible</span>'
                        implant_note = f"Available Safe Length <strong>{dist:.1f}mm</strong> (Total Bone ~{dist+2:.1f}mm).<br>Sufficient bone height for implant placement."
                    else:
                        # Less than 7mm (Total < 9mm) -> Bone Graft / Sinus Lift
                        implant_badge = '<span class="badge overlap">Bone Graft Option Required</span>'
                        if mode == 'upper':
                             # Upper Jaw -> Sinus Lift
                             implant_note = f"Available Safe Length <strong>{dist:.1f}mm</strong> (Total Bone ~{dist+2:.1f}mm).<br><strong>Sinus Bone Graft (Sinus Lift)</strong> indicated.<br>CBCT verification recommended."
                        else:
                             # Lower Jaw -> Vertical Argumentation
                             implant_note = f"Available Safe Length <strong>{dist:.1f}mm</strong> (Total Bone ~{dist+2:.1f}mm).<br><strong>Vertical Bone Graft / GBR</strong> indicated.<br>CBCT verification recommended."

                        # [LLM Injection] Use LLM analysis if available
                        llm_obs = "Absence of tooth structure detected."
                        llm_cons = "Long-term absence may affect alveolar bone and adjacent tooth position."
                        llm_sugg = "Evaluation of the edentulous area is recommended."
                        llm_space = implant_note

                        if llm_analysis and str(lbl) in llm_analysis:
                            la = llm_analysis[str(lbl)]
                            llm_obs = la.get('observation', llm_obs)
                            llm_cons = la.get('consideration', llm_cons)
                            llm_sugg = la.get('suggestion', llm_sugg)
                            llm_space = la.get('space_analysis', llm_space)

                        html += f"""
                        <div class="tooth-card missing" id="tooth-{lbl}">
                            {crop_html}
                            <div class="tooth-details">
                                <h3>
                                    <span class="format-fdi">Tooth #{lbl} <span style="font-size:0.8rem;color:#6b7280;font-weight:normal;">(Univ #{self.get_universal_label(str(lbl))})</span></span>
                                    <span class="format-univ" style="display:none">Tooth #{self.get_universal_label(str(lbl))} <span style="font-size:0.8rem;color:#6b7280;font-weight:normal;">(FDI #{lbl})</span></span>
                                </h3>
                                <div class="finding-tag">
                                    <img src="data:image/png;base64,{info_icon_b64}" style="width:30px;height:30px;opacity:0.5;margin-right:6px;">
                                    Missing Tooth
                                </div>
                                {implant_badge}
                                <p style="color:#374151; font-size:0.95rem; margin-bottom:8px;">
                                    <strong>Observation Reason:</strong> {mt.get('missing_reason', 'No tooth structure detected in the expected position; adjacent teeth present with edentulous gap.')}
                                </p>
                                <p class="description">
                                    <strong>Observation:</strong> {llm_obs}<br>
                                    <strong>Consideration:</strong> {llm_cons}<br>
                                    <strong>Suggestion:</strong> {llm_sugg}
                                </p>
                                {f'<p style="color:#2563eb; font-size:0.95rem; margin-bottom:8px;"><strong>Space Analysis:</strong> {llm_space}</p>' if llm_space else ''}
                                <div class="treatments">
                                     <h4>General Considerations</h4>
                                     <ol>
                                         <li>1. Implant Placement</li>
                                         <li>2. Fixed Partial Denture (Bridge)</li>
                                         <li>3. Removable Prosthesis</li>
                                     </ol>
                                    {f'<p style="margin-top:12px;color:#2563eb;font-weight:600;">➡ Available Bone Height: {mt["implant_guide"]["dist_mm"]:.1f} mm</p>' if isinstance(mt, dict) and mt.get('implant_guide') else ''}
                                </div>
                            </div>
                        </div>
                        """
        
        # Sort findings by FD1 (if available)
        try:
             # Sort by severity (if annotated) then tooth label
             def _sev_key(t):
                 return (
                     -int(t.get("severity_score", 0)),
                     int(t.get('tooth_label', 99)) if str(t.get('tooth_label','0')).isdigit() else 99
                 )
             findings.sort(key=_sev_key)
        except: pass

        # Build Detailed Findings Map
        detailed_map = {}
        if detailed_findings:
            for df in detailed_findings:
                if 'tooth' in df: detailed_map[str(df['tooth'])] = df

        for item in findings:
             # Check criteria
             is_problem = False
             badges = []
             desc = []
             severity_score = 0  # Higher = more critical
             
             treatments = []
             evidence = []  # [FIX] Initialize evidence list
             
             tooth_label = str(item.get("tooth_label", "Unknown"))
             
             # [Detail Injection] Check if we have pre-generated detailed findings (from FeedbackGenerator)
             detail_obj = detailed_map.get(tooth_label)
             
             # Synthesize Diagnosis
             diagnosis = []
             implications = []
             plan = []
             
             # Default
             title = "Check Required"
             card_class = "treatment"
             decision = "Monitor"  # Extraction / Treatment / Monitor

             pbl_mm = item.get("bone_loss_mm", 0)
             pbl_lvl = item.get("bone_loss_level", 0)
             bl_pct = item.get("bone_loss_pct", 0)
             
             has_overlap = item.get("nerve_overlap") or item.get("sinus_overlap")
             overlap_target = "Nerve" if item.get("nerve_overlap") else "Sinus"
             
             has_caries = item.get("caries")
             has_perio = item.get("periapical")
             
             hopeless = pbl_lvl >= 4 or bl_pct >= 60 or pbl_mm >= 6
             severe_bl = pbl_lvl >= 3 or bl_pct >= 40 or pbl_mm >= 4

             # 1. Critical Factors (Structure / Foundation)
             if severe_bl:
                 diagnosis.append(f"Alveolar bone loss detected (Level {bl_level}, {bl_pct:.0f}% / {pbl_mm:.1f}mm).")
                 implications.append("Periodontal support compromised; mobility and tooth loss risk elevated.")
                 if hopeless:
                      plan.append("Prognosis: Hopeless. Extraction indicated to control infection and plan definitive prosthesis.")
                      title = "Hopeless (Severe Bone Loss)"
                      badges.append('<span class="badge overlap">Hopeless</span>')
                      severity_score += 4
                 else:
                      plan.append("Prognosis: Guarded. Periodontal surgery (flap/graft) and maintenance required.")
                      title = "Periodontal Disease"
                      severity_score += 2
                 is_problem = True
                 badges.append(f'<span class="badge pbl">Bone Loss Lv.{bl_level}</span>')
                 evidence.append(f"Bone loss level {bl_level} ({bl_pct:.0f}%, {pbl_mm:.1f}mm)")

             # 2. Infection / Pathology
             if has_perio:
                 diagnosis.append("Periapical radiolucency observed at the root apex.")
                 implications.append("Consistent with chronic apical periodontitis or cyst formation.")
                 plan.append("Root Canal Treatment (RCT) or Apicoectomy is indicated.")
                 if not title.startswith("Severe"): title = "Periapical Lesion"
                 is_problem = True
                 badges.append('<span class="badge perio">Periapical Lesion</span>')
                 evidence.append("Periapical lesion at apex")
                 severity_score += 2

             if has_caries:
                 diagnosis.append("Radiolucency consistent with dental caries on the crown.")
                 if has_perio:
                      implications.append("Combined with apical pathology, suggesting pulpal necrosis.")
                 else:
                      implications.append("Risk of pulpal involvement if untreated.")
                 plan.append("Restorative treatment (Filling/Inlay/Crown) needed.")
                 if not is_problem: title = "Dental Caries"
                 is_problem = True
                 badges.append('<span class="badge caries">Caries</span>')
                 evidence.append("Caries detected")
                 severity_score += 1

             # 3. Surgical Risks (Anatomy)
             if has_overlap:
                 diagnosis.append(f"Root apices are superimposed on the {overlap_target}.")
                 implications.append(f"High risk of {overlap_target} injury during extraction/surgery.")
                 plan.append("CBCT verification is MANDATORY before any surgical intervention.")
                 title = "High Risk: Overlap"
                 card_class = "overlap"
                 is_problem = True
                 badges.append(f'<span class="badge overlap">Critical: {overlap_target}</span>')
                 evidence.append(f"Root overlaps {overlap_target}")
                 severity_score += 2

             # Nerve distance (if provided)
             nerve_dist_mm = item.get("nerve_dist_mm")
             if nerve_dist_mm:
                 evidence.append(f"Nerve distance: {nerve_dist_mm:.1f}mm")
                 if nerve_dist_mm < 2.0:
                     plan.append("Severe proximity (<2mm) to nerve. Extraction/implant must be deferred until CT planning.")
                     severity_score += 2
                     card_class = "overlap"
                     badges.append('<span class="badge overlap">Critical: Nerve Distance</span>')

             # Decision matrix (deterministic)
             if hopeless or (has_perio and severe_bl) or (has_overlap and severe_bl):
                 decision = "Extraction"
             elif has_overlap:
                 decision = "Extraction"
             elif severe_bl or has_perio or has_caries:
                 decision = "Treatment"
             else:
                 decision = "Monitor"
             
             decision_badge = {
                 "Extraction": '<span class="badge overlap">Decision: Extraction</span>',
                 "Treatment": '<span class="badge treatment">Decision: Treatment</span>',
                 "Monitor": '<span class="badge info">Decision: Monitor</span>'
             }[decision]
             badges.insert(0, decision_badge)
             
             # Formulate Narrative
             if is_problem:
                 # [Sync] Use detailed findings if available
                 if detail_obj:
                     evidence_txt = ", ".join(detail_obj.get('findings', []))
                     explanation_txt = detail_obj.get('explanation', '')
                     recommendation_txt = detail_obj.get('recommendation', '')
                     
                     full_desc = f"<strong>Clinical Findings:</strong><br>{evidence_txt}"
                     if explanation_txt:
                         full_desc += f"<br><br><strong>Analysis:</strong><br>{explanation_txt}"
                     if recommendation_txt:
                         full_desc += f"<br><br><strong>Recommendation:</strong><br>{recommendation_txt}"
                         # Also update Plan/Treatment logic if needed, but description is key for rendering
                         # treatments.append(recommendation_txt) 
                         
                     desc.append(full_desc)
                 else:
                     # Legacy Heuristic Logic
                     if not evidence:
                         evidence = diagnosis  # fallback to prior text if evidence list empty
                     full_desc = "<strong>Clinical Evidence:</strong><br>" + "<br>".join(evidence)
                     conclusion_txt = f"{decision}: " + (" ".join(implications) if implications else "Follow clinical judgement.")
                     full_desc += "<br><br><strong>Conclusion & Implication:</strong><br>" + conclusion_txt
                     
                     treatments = plan
                     desc.append(full_desc)

             if is_problem:
                 has_issues = True
                 # Fix: Use 'tooth_label' instead of 'label'
                 tooth_label = item.get("tooth_label", "Unknown")
                 box = item.get("box")
                 # propagate severity for sorting
                 item["severity_score"] = severity_score
                
                 crop_b64 = ""
                 if box:
                     # Gather specific findings for this tooth for visualization
                     specific_findings = item.get("findings", [])
                     # Use pct if available, else mm (legacy)
                     pbl_val = item.get("bone_loss_pct", item.get("bone_loss_mm", 0))
                     
                     # Safety Guide (if overlap detected)
                     safety_guide = item.get("safety_guide")
                     
                     # [VISUALIZATION LOGIC]
                     # Overlap Case: Show Nerve/Sinus AND Tooth Mask
                     # Disease Case: Show ONLY Disease Box (Clean)
                    
                     show_nc = []
                     show_sc = []
                     show_tc = None
                    
                     if item.get('nerve_overlap') or item.get('sinus_overlap'):
                         # Overlap Detected -> Show Context
                         show_nc = nerve_contours
                         show_sc = sinus_contours
                         show_tc = item.get('contour') # Make sure this exists in item
                     # Centerline axis from contour if available
                     center_axis = None
                     if item.get('contour'):
                         try:
                             pts = np.array(item['contour'], dtype=np.float32)
                             if pts.ndim == 3:
                                 pts = pts.reshape(-1, 2)
                             pts = pts - pts.mean(axis=0)
                             cov = np.cov(pts, rowvar=False)
                             eigvals, eigvecs = np.linalg.eigh(cov)
                             vec = eigvecs[:, np.argmax(eigvals)]
                             norm = max(np.linalg.norm(vec), 1e-6)
                             center_axis = (float(vec[0]/norm), float(vec[1]/norm))
                         except Exception:
                             center_axis = None
 
                     crop_b64 = self._crop_and_overlay(
                        crop_source_img, 
                        box, 
                        specific_findings, 
                        bl_viz_img, 
                        pbl_val,
                        nerve_contours=show_nc,
                        sinus_contours=show_sc,
                        safety_guide=safety_guide,
                        tooth_contour=show_tc,
                        nerve_dist_line=item.get("nerve_dist_line"),
                        nerve_dist_mm=item.get("nerve_dist_mm", 0),
                        centerline=False,
                        centerline_axis=None
                     )
                 
                 # Treatment HTML
                 treatment_html = f"<div style='margin-top:10px; padding:10px; background:#f0f8ff; border-radius:5px; font-size:0.95em; color:#2980b9;'>{'<br>'.join(treatments)}</div>" if treatments else ""
                 
                 # Check for Overlay Warning to add specific caution
                 caution_msg = ""
                 if item.get("nerve_overlap") or item.get("sinus_overlap"):
                     target = "Nerve" if item.get("nerve_overlap") else "Sinus"
                     caution_msg = f"""
                     <div style='margin-top:10px; padding:10px; background:#fffbeb; border-left: 4px solid #d97706; border-radius:4px; font-size:0.95em; color:#d97706;'>
                         <strong>Extraction Caution:</strong><br>
                         The tooth root overlaps with the {target}. 
                         <strong>Risk of {target} Injury.</strong><br>
                         {f'<strong>Est. Overlap Area:</strong> {item.get("overlap_area_mm2", 0):.1f} mm짼<br>' if item.get("overlap_area_mm2") else ''}
                         Precise planning and CT scan are strongly recommended before extraction.
                     </div>
                     """
                 
                 # Use Clinical Title
                 finding_title = title
                 
                 html += f"""
                 <div class="tooth-card {card_class}" id="tooth-{tooth_label}">
                     <img class="crop-img" src="data:image/jpeg;base64,{crop_b64}">
                     <div class="tooth-details">
                         <h3>
                              <span class="format-fdi">Tooth #{tooth_label} <span style="font-size:0.8rem;color:#6b7280;font-weight:normal;">(Univ #{self.get_universal_label(str(tooth_label))})</span></span>
                              <span class="format-univ" style="display:none">Tooth #{self.get_universal_label(str(tooth_label))} <span style="font-size:0.8rem;color:#6b7280;font-weight:normal;">(FDI #{tooth_label})</span></span>
                          </h3>
                         
                         <div class="finding-tag">
                            <img src="data:image/png;base64,{warning_icon_b64}" style="width:30px;height:30px;margin-right:6px;">
                            {finding_title}
                         </div>
                         
                         <div style="margin-bottom: 10px;">{''.join(badges)}</div>
                         <p>{' '.join(desc)}</p>
                         {treatment_html}
                         {caution_msg}
                         <p style="font-size:0.9em; color:#666; margin-top:5px;">
                            This area shows signs that may require treatment. Please check this tooth specifically.
                         </p>
                     </div>
                 </div>
                 """

        if not has_issues:
            html += """
            <div class="tooth-card">
                <div class="tooth-details">
                    <h3>Good News!</h3>
                    <p>No major issues (Cavities, significant bone loss, or inflammation) were detected by the AI in this scan.</p>
                </div>
            </div>
            """

        html += f"""
            <div class="footer">
                <div class="footer-logo"><img src="data:image/png;base64,{logo_b64}" style="height:35px;"></div>
                <div class="footer-warning" style="flex:1;">
                    <div style="display:flex;align-items:center;justify-content:center;gap:16px;margin-bottom:12px;">
                        <img src="data:image/png;base64,{warning_icon_b64}" style="height:60px; width:auto;">
                        <strong style="font-size:1.8rem; color:#dc2626;">WARNING</strong>
                    </div>
                    <div style="text-align: center;">
                        This report is generated by an Artificial Intelligence system and DOES NOT constitute a medical diagnosis. Errors may occur.<br>
                        The results are for informational purposes only. You must consult a licensed dentist for accurate diagnosis and treatment planning.
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html

    def generate(self, user_name, image_path, analysis_result, overlay_path=None, bl_viz_path=None, output_dir=None, ai_commentary=None, llm_analysis=None, detailed_findings=None):
        """
        Generates the full HTML report.
        """
        # Load Images
        if not image_path.exists():
            return None, None, None
            
        main_img = cv2.imread(str(image_path))
        if main_img is None: raise ValueError(f"Image not found at {image_path}")
        crop_source_img = main_img.copy() # For cropping
        
        # Load Visualization Images
        bl_viz_img = None
        if bl_viz_path and bl_viz_path.exists():
             bl_viz_img = cv2.imread(str(bl_viz_path))

        # Use Overlay as main if available, else Original
        if overlay_path and overlay_path.exists():
            overlay_img = cv2.imread(str(overlay_path))
            if overlay_img is not None:
                main_img = overlay_img
                crop_source_img = overlay_img # [FIX] Use overlay for crops to show masks (Nerve/Sinus)

        findings = analysis_result.get('teeth', [])
        missing_teeth = analysis_result.get('missing_teeth', [])
        
        # Summary Data (Caries, Perio, etc.)
        summary_data = {
            'caries': analysis_result.get('caries', []),
            'periapical': analysis_result.get('periapical', [])
        }
        
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Generate HTML
        html_content = self._generate_html(
            user_name, date_str, summary_data, findings, 
            main_img, crop_source_img, bl_viz_img, 
            missing_teeth, 
            nerve_contours=analysis_result.get('nerve_contours'),
            sinus_contours=analysis_result.get('sinus_contours'),
            ai_commentary=ai_commentary,
            llm_analysis=llm_analysis,
            detailed_findings=detailed_findings
        )
        
        # Save Files
        safe_name = "".join([c for c in user_name if c.isalnum() or c in (' ','_','-')]).strip()
        html_filename = f"{safe_name}_Report.html"
        pdf_filename = f"{safe_name}_Report.pdf"
        
        # Use custom output_dir if provided
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        html_path = target_dir / html_filename
        pdf_path = target_dir / pdf_filename
        
        # Save HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # PDF generation disabled
        pdf_filename = None
            
        return str(html_path), html_filename, pdf_filename







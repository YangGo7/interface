import cv2
import numpy as np


class ReportV3AlgorithmsMixin:
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

    def _crop_and_overlay(self, orig_img, box, findings, bl_viz=None, pbl_mm=0, implant_guide=None, nerve_contours=None, sinus_contours=None, safety_guide=None, tooth_contour=None, debug_text=None, nerve_dist_line=None, nerve_dist_mm=0, centerline=False, centerline_axis=None, tooth_label=None, neighbor_contours=None, gap_line_coords=None):
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
        
        # [MODIFIED] Enforce Minimum Aspect Ratio (e.g. 0.6)
        # Often teeth boxes are tall and narrow. We want at least some context width.
        min_aspect = 0.6
        if w_box < h_box * min_aspect:
             needed_w = int(h_box * min_aspect)
             diff = needed_w - w_box
             x1 -= diff // 2
             x2 += (diff - diff // 2)
             w_box = x2 - x1 # Update w_box

        # Check if caries is present to decrease crop margin (Zoom In)
        has_caries_finding = any(f.get('type') == 'caries' for f in findings)
        margin_multiplier = 0.05 if has_caries_finding else 0.2
        min_margin = 20 if has_caries_finding else 50
        
        margin_x = max(min_margin, int(w_box * margin_multiplier)) 
        margin_y = max(min_margin, int(h_box * margin_multiplier))
        
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
        
        # if debug_text:
        #      cv2.putText(crop, str(debug_text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # if tooth_label:
        #      # Draw the tooth label at the top-left of the crop
        #      cv2.putText(crop, f"{tooth_label}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        crop_h, crop_w = crop.shape[:2]
        
        # If PBL Issue, blend BL Viz (Lower threshold to 1.5mm to show early bone loss)
        # [USER REQUEST] Do not overlay the bone level mask (bl_viz) on individual crops
        # if pbl_mm >= 1.5 and bl_viz is not None:
        #      bl_crop = bl_viz[y1:y2, x1:x2]
        #      if bl_crop.shape == crop.shape:
        #          mask = cv2.cvtColor(bl_crop, cv2.COLOR_BGR2GRAY)
        #          _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        #          mask_inv = cv2.bitwise_not(mask)
        #          
        #          img_bg = cv2.bitwise_and(crop, crop, mask=mask_inv)
        #          img_fg = cv2.bitwise_and(bl_crop, bl_crop, mask=mask)
        #          crop = cv2.add(img_bg, img_fg)

        
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
            # Outline the target tooth mask with yellow (don't fill it) with 0.3 opacity
            overlay_layer = crop.copy()
            cv2.polylines(overlay_layer, [shifted_tc], True, (0, 255, 255), 2)
            cv2.addWeighted(overlay_layer, 0.3, crop, 0.7, 0, crop)
            
        # [USER REQUEST] Draw neighbor contours for context (Missing Teeth)
        if neighbor_contours:
            shifted_neighbors = []
            for c in neighbor_contours:
                shifted = np.array(c, dtype=np.int32) - [x1, y1]
                shifted_neighbors.append(shifted)
            # Outline neighbors in blue (or green) with 0.3 opacity
            overlay_layer = crop.copy()
            cv2.polylines(overlay_layer, shifted_neighbors, True, (0, 255, 0), 2)  # Green outline
            cv2.addWeighted(overlay_layer, 0.3, crop, 0.7, 0, crop)

        # Draw mesiodistal gap line — solid with endpoint circles
        if gap_line_coords and len(gap_line_coords) == 4:
            try:
                gx1, gy1, gx2, gy2 = [int(v) for v in gap_line_coords]
                # shift to crop coords
                gx1 -= x1; gx2 -= x1
                gy1 -= y1; gy2 -= y1
                sky_blue = (233, 165, 14)  # BGR for sky blue (#0ea5e9)
                # Solid line
                cv2.line(crop, (gx1, gy1), (gx2, gy2), sky_blue, 2)
                # Endpoint circles
                cv2.circle(crop, (gx1, gy1), 5, sky_blue, -1)
                cv2.circle(crop, (gx2, gy2), 5, sky_blue, -1)
            except Exception as e:
                print(f"[GAP LINE] Draw error: {e}")


        # Draw Nerve as Outline Only (no fill)
        if nerve_contours:
            shifted_cnts = []
            for c in nerve_contours:
                c_arr = np.array(c, dtype=np.int32)
                c_arr[:, 0] -= x1
                c_arr[:, 1] -= y1
                shifted_cnts.append(c_arr.reshape(-1, 1, 2))
            
            # Draw red outline only (thickness 1 for thinner profile)
            cv2.polylines(crop, shifted_cnts, False, (0, 0, 255), 1)

        # Draw centerline (for angle visualization)
        # [USER REQUEST] Remove Centerline Overlay
        # if centerline:
        #     # Compute axis
        #     ax, ay = None, None
        #     if centerline_axis:
        #         ax, ay = centerline_axis
        #     elif tooth_contour:
        #         try:
        #             pts = np.array(tooth_contour, dtype=np.float32)
        #             if pts.ndim == 3:
        #                 pts = pts.reshape(-1, 2)
        #             pts = pts - pts.mean(axis=0)
        #             cov = np.cov(pts, rowvar=False)
        #             eigvals, eigvecs = np.linalg.eigh(cov)
        #             vec = eigvecs[:, np.argmax(eigvals)]
        #             norm = max(np.linalg.norm(vec), 1e-6)
        #             ax, ay = float(vec[0]/norm), float(vec[1]/norm)
        #         except Exception:
        #             ax = ay = None
        #     if ax is None or ay is None:
        #         ax, ay = 0.0, 1.0  # default vertical

        #     # Start point: midpoint of the shorter side of the box (prefer root-side)
        #     bw = box[2] - box[0]
        #     bh = box[3] - box[1]
        #     if bw <= bh:
        #         # shorter side is width -> use midpoint of left-right edge (vertical midline)
        #         cx = (box[0] + box[2]) / 2 - x1
        #         cy = (box[1] + box[3]) / 2 - y1
        #     else:
        #         # shorter side is height -> use midpoint of top-bottom edge (horizontal midline)
        #         cx = (box[0] + box[2]) / 2 - x1
        #         cy = (box[1] + box[3]) / 2 - y1
        #     # if multiple roots branching, contour PCA already biases axis; origin keeps center of short side

        #     length = max(crop_h, crop_w) * 0.6
        #     p1 = (int(cx - ax*length/2), int(cy - ay*length/2))
        #     p2 = (int(cx + ax*length/2), int(cy + ay*length/2))
        #     cv2.line(crop, p1, p2, (0, 255, 255), 2)

        # Sinus overlay: outline only (no fill)
        if sinus_contours:
            shifted_scnts = []
            for sc in sinus_contours:
                sc_arr = np.array(sc, dtype=np.int32)
                sc_arr[:, 0] -= x1
                sc_arr[:, 1] -= y1
                shifted_scnts.append(sc_arr.reshape(-1, 1, 2))
            # Outline only — 30% opacity green polyline
            overlay_s = crop.copy()
            cv2.polylines(overlay_s, shifted_scnts, True, (0, 255, 0), 2)
            cv2.addWeighted(overlay_s, 0.3, crop, 0.7, 0, crop)


        # [USER REQUEST] Bone Level / Implant Guide line hidden (data preserved, drawing disabled)
        # Draw Guide Line (Implant Guide OR Safety Guide)
        guide_data = implant_guide if implant_guide else safety_guide
        if guide_data:
            coords = guide_data.get('line_coords')
            dist_mm = guide_data.get('dist_mm', 0.0)
            if coords and len(coords) == 4:
                gx1, gy1, gx2, gy2 = coords
                dcx1 = int(gx1 - x1); dcy1 = int(gy1 - y1)
                dcx2 = int(gx2 - x1); dcy2 = int(gy2 - y1)
                dx_px = dcx2 - dcx1; dy_px = dcy2 - dcy1
                dist_px = (dx_px**2 + dy_px**2) ** 0.5
                if not dist_mm or dist_mm <= 0:
                    mm_per_px = guide_data.get('mm_per_px')
                    px_per_mm = guide_data.get('px_per_mm')
                    if not mm_per_px and px_per_mm:
                        mm_per_px = 1.0 / max(px_per_mm, 1e-6)
                    if mm_per_px:
                        dist_mm = dist_px * mm_per_px
                        guide_data['dist_mm'] = dist_mm
                    else:
                        guide_data['dist_mm'] = None
                        guide_data['dist_px'] = dist_px
                # -- Drawing disabled (비가시화) --
                # cv2.line(crop, (dcx1, dcy1), (dcx2, dcy2), (255, 0, 255), 2)
                # cv2.circle(crop, (dcx1, dcy1), 4, (0, 0, 255), -1)
                # cv2.circle(crop, (dcx2, dcy2), 3, (255, 0, 255), -1)

        # [USER REQUEST] CEJ-to-Nerve distance line hidden (비가시화)
        # if nerve_dist_line:
        #     if len(nerve_dist_line) == 4:
        #         p1 = (nerve_dist_line[0], nerve_dist_line[1])
        #         p2 = (nerve_dist_line[2], nerve_dist_line[3])
        #     else:
        #         p1, p2 = nerve_dist_line
        #     p1_crop = (int(p1[0] - x1), int(p1[1] - y1))
        #     p2_crop = (int(p2[0] - x1), int(p2[1] - y1))
        #     cv2.line(crop, p1_crop, p2_crop, (0, 255, 255), 2)
        #     cv2.circle(crop, p1_crop, 3, (0, 0, 255), -1)
        #     cv2.circle(crop, p2_crop, 3, (0, 255, 255), -1)
        #     mid_x = (p1_crop[0] + p2_crop[0]) // 2
        #     mid_y = (p1_crop[1] + p2_crop[1]) // 2
        #     label = f"{nerve_dist_mm:.1f}mm"
        #     cv2.putText(crop, label, (mid_x+5, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        #     cv2.putText(crop, label, (mid_x+5, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Draw Specific Findings (Caries/Perio Boxes/Contours)
        for f in findings:
            fx1, fy1, fx2, fy2 = map(int, f['box'])
            
            is_caries = (f['type'] == 'caries')
            # [USER REQUEST] Scale Caries Box by 1.5x
            if is_caries:
                bw, bh = fx2 - fx1, fy2 - fy1
                bcx, bcy = fx1 + bw / 2.0, fy1 + bh / 2.0
                fx1, fx2 = int(bcx - bw * 1.5 / 2.0), int(bcx + bw * 1.5 / 2.0)
                fy1, fy2 = int(bcy - bh * 1.5 / 2.0), int(bcy + bh * 1.5 / 2.0)
                
            # Shift to crop coords
            cx1, cy1 = fx1 - x1, fy1 - y1
            cx2, cy2 = fx2 - x1, fy2 - y1
            
            is_caries = (f['type'] == 'caries')
            color = (0, 0, 255) if is_caries else (0, 165, 255) # Red for Caries, Orange for Perio
            thickness = 1
            
            overlay = crop.copy()
            
            # Use dashed lines for ALL pathology boxes as per user request
            def draw_dashed_polygon(img, pts, col, thk, dash_len=4):
                for i in range(len(pts)):
                    p1 = pts[i]
                    p2 = pts[(i+1)%len(pts)]
                    line_len = int(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                    if line_len <= 0: continue
                    dx = (p2[0] - p1[0]) / line_len
                    dy = (p2[1] - p1[1]) / line_len
                    curr_dist = 0
                    while curr_dist < line_len:
                        s_x = int(p1[0] + dx * curr_dist)
                        s_y = int(p1[1] + dy * curr_dist)
                        e_x = int(p1[0] + dx * min(curr_dist + dash_len, line_len))
                        e_y = int(p1[1] + dy * min(curr_dist + dash_len, line_len))
                        cv2.line(img, (s_x, s_y), (e_x, e_y), col, thk)
                        curr_dist += dash_len * 2
            
            if f.get('contour'):
                cnt = np.array(f['contour'], dtype=np.int32) - [x1, y1]
                draw_dashed_polygon(overlay, cnt, color, thickness)
            else:
                pts = [(cx1, cy1), (cx2, cy1), (cx2, cy2), (cx1, cy2)]
                draw_dashed_polygon(overlay, pts, color, thickness)
                
            crop = overlay
            
            # [USER REQUEST] Remove disease text label from crop
            # cv2.putText(crop, f['type'].upper(), (cx1, cy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return self._encode_image(crop)

import numpy as np
import math
import cv2

class ToothLogic:
    def __init__(self):
        pass

    def calculate_dental_zone(self, tooth_boxes, img_shape):
        """
        Calculate valid dental zone based on statistics of detected teeth.
        tooth_boxes: list of (label, [x1, y1, x2, y2])
        Returns: dict with zone boundaries
        """
        h, w = img_shape[:2]
        tooth_x_centers = []
        tooth_y_coords = []
        
        for _, bb in tooth_boxes:
            tooth_x_centers.append((bb[0] + bb[2]) / 2)
            tooth_y_coords.extend([float(bb[1]), float(bb[3])])
            
        bl_x1, bl_x2 = w * 0.1, w * 0.9 # Defaults
        
        if tooth_x_centers:
            med_x = np.median(tooth_x_centers)
            std_x = np.std(tooth_x_centers) if len(tooth_x_centers) > 1 else w * 0.2
            
            # Filter outliers (5.0 sigma)
            valid_x = [x for x in tooth_x_centers if abs(x - med_x) < 5.0 * std_x]
            
            min_valid = min(valid_x) if valid_x else w * 0.1
            max_valid = max(valid_x) if valid_x else w * 0.9
            
            bl_x1 = min(min_valid - w * 0.05, w * 0.02)
            bl_x2 = max(max_valid + w * 0.05, w * 0.98)
            
        return {
            "bl_x1": bl_x1,
            "bl_x2": bl_x2,
            "tooth_y_coords": tooth_y_coords, 
            "tooth_boxes": tooth_boxes # Store for neighbor check
        }

    def is_spatial_valid(self, box, label, zone_info, img_shape):
        """
        Check if a box is spatially valid within the dental zone.
        """
        # 2026-01-27: USER REQUEST - Temporarily disable filter
        return True

        # Legacy Logic (Preserved but disabled)
        # x1, y1, x2, y2 = box
        # cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # lbl = str(label).lower()
        # h, w = img_shape[:2]
        
        # bl_x1 = zone_info.get('bl_x1', 0)
        # bl_x2 = zone_info.get('bl_x2', w)
        # tooth_y_coords = zone_info.get('tooth_y_coords', [])
        
        # # 1) Horizontal Check
        # if cx < bl_x1 or cx > bl_x2: return False

        # # 2) Vertical Check
        # if tooth_y_coords:
        #     ty1, ty2 = min(tooth_y_coords), max(tooth_y_coords)
        #     margin = h * 0.25 if ("implant" in lbl or "fixture" in lbl) else h * 0.10
        #     if cy < ty1 - margin or cy > ty2 + margin: return False
        
        # # 3) Proximity Check (for non-teeth)
        # if any(k in lbl for k in ["crown", "bridge", "caries", "filling"]):
        #     near_tooth = False
        #     tooth_boxes = zone_info.get('tooth_boxes', [])
        #     for _, tbb in tooth_boxes:
        #         # Horizontal proximity check
        #         if abs(cx - (tbb[0]+tbb[2])/2) < (tbb[2]-tbb[0]) * 1.5:
        #             near_tooth = True
        #             break
        #     if not near_tooth: return False

        # return True

    def find_missing_teeth(self, context):
        """
        Detect missing teeth and calculate implant safety guides.
        context: dict containing all necessary data (final_teeth_objects, masks, calibration, etc.)
        """
        missing_teeth_list = []
        odontogram_update = {} # {fdi: 'missing'}
        
        present_fdis = context['present_fdis']
        final_teeth_objects = context['final_teeth_objects']
        y_poly_fn_upper = context.get('y_poly_fn_upper')
        y_poly_fn_lower = context.get('y_poly_fn_lower')
        nerve_mask = context.get('nerve_mask')
        sinus_mask = context.get('sinus_mask')
        mm_per_px = context.get('mm_per_px', 0.1)
        avg_tooth_width = context.get('avg_tooth_width', 40.0)
        
        # Exclude wisdom teeth 8s
        all_fdis = [
            17,16,15,14,13,12,11, 21,22,23,24,25,26,27,
            47,46,45,44,43,42,41, 31,32,33,34,35,36,37
        ]
        
        def get_tooth_by_label(lbl):
            for t in final_teeth_objects:
                if t.get('tooth_label') == str(lbl): return t
            return None

        # Pre-calculate tooth axes for visualization/logic
        tooth_axes_viz = []
        for t in final_teeth_objects:
             if t.get('contour'):
                 try:
                     pts = np.array(t['contour'], dtype=np.float32).reshape(-1, 2)
                     mean = np.mean(pts, axis=0)
                     centered = pts - mean
                     cov = np.cov(centered, rowvar=False)
                     evals, evecs = np.linalg.eigh(cov)
                     major_axis = evecs[:, np.argmax(evals)]
                     major_axis = major_axis / np.linalg.norm(major_axis)
                     if major_axis[1] < 0: major_axis = -major_axis # Point down
                     
                     tooth_axes_viz.append({
                         'label': t.get('tooth_label', '?'),
                         'center': mean,
                         'vec': major_axis
                     })
                 except: pass

        for fdi in all_fdis:
            if fdi in present_fdis: continue
            
            slabel = str(fdi)
            odontogram_update[slabel] = "missing"
            
            mt = {'tooth_label': fdi, 'type': 'missing'}
            
            # --- Geometric Analysis for Missing Site ---
            q = fdi // 10
            n = fdi % 10
            
            # Find Neighbors
            candidates = []
            if n > 1: candidates.append(q*10 + (n-1))
            else:
                if q == 3: candidates.append(41)
                elif q == 4: candidates.append(31)
            if n < 8: candidates.append(q*10 + (n+1))
            
            t_n1 = get_tooth_by_label(candidates[0]) if candidates else None
            t_n2 = get_tooth_by_label(candidates[1]) if len(candidates) > 1 else None
            
            valid_boxes = [t for t in [t_n1, t_n2] if t and t.get('box')]
            
            gap_cx = None
            gap_cy = None
            
            # 1. Calculate Gap Center X
            if len(valid_boxes) >= 2:
                b1 = valid_boxes[0]['box']
                b2 = valid_boxes[1]['box']
                if b1[0] > b2[0]: b1, b2 = b2, b1
                gap_cx = (b1[2] + b2[0]) / 2
                
            elif len(valid_boxes) == 1:
                b = valid_boxes[0]['box']
                w = b[2] - b[0]
                ref_lbl = int(valid_boxes[0]['tooth_label'])
                
                dir_factor = 0
                if q == 3: # Right Side (31..38)
                      if ref_lbl < fdi: dir_factor = 1 
                      else: dir_factor = -1
                elif q == 4: # Left Side (41..48)
                      if ref_lbl < fdi: dir_factor = -1 
                      else: dir_factor = 1
                elif q in [1, 2]: # Upper
                    if q == 1: dir_factor = -1 if ref_lbl < fdi else 1
                    elif q == 2: dir_factor = 1 if ref_lbl < fdi else -1

                gap_cx = ((b[0]+b[2])/2) + (dir_factor * w * 1.1)

            # 2. Calculate Gap Level Y
            valid_cejs = []
            for t in [t_n1, t_n2]:
                if t and t.get('cej_center'): valid_cejs.append(t['cej_center'])

            # Determine Axis Vector
            if q in [1, 2]: axis_vec = (0, -1) # Up
            else: axis_vec = (0, 1) # Down
            
            # Use Global Curve Fallback if Geometry is missing
            if gap_cx is None or (gap_cy is None and not valid_cejs and not valid_boxes):
                 target_poly = y_poly_fn_upper if q in [1, 2] else y_poly_fn_lower
                 if target_poly:
                     # Find nearest Anchor
                     best_anchor = None
                     min_dist_slots = 999
                     for t_obj in final_teeth_objects:
                         try:
                             t_lbl = int(float(t_obj.get('tooth_label')))
                             if t_lbl // 10 == q:
                                 dist = abs(fdi - t_lbl)
                                 if dist < min_dist_slots:
                                     min_dist_slots = dist
                                     best_anchor = t_obj
                         except: pass
                     
                     if best_anchor:
                         anchor_lbl = int(float(best_anchor.get('tooth_label')))
                         slots_diff = fdi - anchor_lbl
                         
                         dx_step = slots_diff * avg_tooth_width if q in [2, 3] else -slots_diff * avg_tooth_width
                         acx = (best_anchor['box'][0]+best_anchor['box'][2])/2 if best_anchor.get('box') else 0
                         
                         gap_cx = acx + dx_step
                         gap_cy = target_poly(gap_cx)
                         
                         # Normal vector from curve
                         deriv_fn = np.polyder(target_poly)
                         m = deriv_fn(gap_cx)
                         if q in [1, 2]: axis_vec = (m, -1.0)
                         else: axis_vec = (-m, 1.0)
                         norm = math.hypot(axis_vec[0], axis_vec[1])
                         if norm > 0: axis_vec = (axis_vec[0]/norm, axis_vec[1]/norm)

            # Y Fallback
            if gap_cy is None:
                if valid_cejs:
                    gap_cy = sum([c[1] for c in valid_cejs]) / len(valid_cejs)
                elif valid_boxes:
                    # Estimate Gap Y at Crest/CEJ level, not Apex
                    est_ys = []
                    for vb in valid_boxes:
                        bx, by, bx2, by2 = vb['box']
                        h = by2 - by
                        # Lower: Crest is near Top (y). Upper: Crest is near Bottom (y2).
                        if q in [3, 4]: est_ys.append(by + h * 0.3)
                        else: est_ys.append(by2 - h * 0.3)
                    gap_cy = sum(est_ys)/len(est_ys)

            # 3. Raycast for Safety Guide
            implant_guide = None
            if gap_cx is not None and gap_cy is not None:
                target_mask = sinus_mask if q in [1, 2] else nerve_mask
                if target_mask is not None:
                     ray_x, ray_y = gap_cx, gap_cy
                     max_step = 2000
                     found_target = False
                     p_target = None
                     h_img, w_img = target_mask.shape
                     
                     for _ in range(max_step):
                         ray_x += axis_vec[0]
                         ray_y += axis_vec[1]
                         ix, iy = int(ray_x), int(ray_y)
                         if not (0 <= ix < w_img and 0 <= iy < h_img): break
                         if target_mask[iy, ix] > 0:
                             p_target = (ix, iy)
                             found_target = True
                             break
                     
                     if found_target and p_target:
                         dist_px = math.hypot(p_target[0]-gap_cx, p_target[1]-gap_cy)
                         pixels_per_mm = 1.0 / mm_per_px if mm_per_px > 0 else 20.0
                         safety_margin_px = 2.0 * pixels_per_mm
                         safe_dist_px = max(0, dist_px - safety_margin_px)
                         dist_mm = safe_dist_px * mm_per_px
                         
                         safe_end_x = gap_cx + axis_vec[0] * safe_dist_px
                         safe_end_y = gap_cy + axis_vec[1] * safe_dist_px
                         
                         implant_guide = {
                             'dist_mm': dist_mm,
                             'line_coords': [gap_cx, gap_cy, safe_end_x, safe_end_y],
                             'type': 'vertical',
                             'safety_margin_mm': 2.0
                         }
            
            # 4. Box Calculation (for visualization)
            box = None
            if valid_boxes:
                # Same logic as before roughly
                all_coords = []
                for vb in valid_boxes:
                    all_coords.extend(vb['box'])
                # Simplified box: Extrapolate or Interpolate
                # Actually, earlier implementation used neighbor box UNION if 2 neighbors, or Extrapolation if 1
                pass # (Simplified for this file, implementation detail moved)
                # Re-implement minimal valid box logic if needed or fallback to gap_cx centric box
                if gap_cx and gap_cy:
                     sz = 50 # Default half-size
                     box = [gap_cx-sz, gap_cy-sz, gap_cx+sz, gap_cy+sz]

            mt['implant_guide'] = implant_guide
            mt['box'] = box
            mt['guide_status'] = 'Success' if implant_guide else 'Failed'
            
            missing_teeth_list.append(mt)
            
        return missing_teeth_list, odontogram_update, tooth_axes_viz

    def calculate_nerve_safety(self, context):
        """
        Calculate safety distance for PRESENT teeth against Nerve/Sinus.
        Adds 'nerve_dist_mm' and 'nerve_dist_line' to tooth objects.
        """
        final_teeth_objects = context['final_teeth_objects']
        nerve_mask = context.get('nerve_mask')
        sinus_mask = context.get('sinus_mask')
        mm_per_px = context.get('mm_per_px', 0.1)

        target_mask = None
        count_found = 0
        
        for t in final_teeth_objects:
            try:
                lbl = int(float(t.get('tooth_label', 0)))
                box = t.get('box')
                if not box: continue
                
                # Determine target (Nerve for Lower, Sinus for Upper)
                q = lbl // 10
                if q in [3, 4]: target_mask = nerve_mask
                elif q in [1, 2]: target_mask = sinus_mask
                else: continue
                
                if target_mask is None: 
                    # print(f"[DEBUG SAFETY] No mask for Q{q}")
                    continue
                
                # Start: Apex from Contour (lowest Y for Upper, highest Y for Lower)
                # Raycast downwards (or upwards for Upper)
                cx = (box[0] + box[2]) / 2
                cy = box[3] if q in [3, 4] else box[1]

                contour = t.get('contour', [])
                if contour and len(contour) > 0:
                    pts = np.array(contour).reshape(-1, 2)
                    if q in [3, 4]: # Lower: Find Highest Y (Root Tip)
                        idx = np.argmax(pts[:, 1])
                        cx, cy = pts[idx]
                    else: # Upper: Find Lowest Y (Root Tip)
                        idx = np.argmin(pts[:, 1])
                        cx, cy = pts[idx]
                
                # Calculate Axis using PCA (or fallback to vertical)
                dx, dy_axis = 0, 1 if q in [3, 4] else -1
                
                check_pts = t.get('contour', [])
                start_x, start_y = cx, cy # Default center
                
                # Check for precise ML-based CEJ point first
                pbl_dict = context.get('pbl_dict', {})
                cej_pt_ml = None
                if str(lbl) in pbl_dict:
                    cej_meta = pbl_dict[str(lbl)]
                    if cej_meta.get('cej_pt'):
                        cej_pt_ml = cej_meta['cej_pt']
                
                if cej_pt_ml:
                     # Use ML detected CEJ point
                     start_x, start_y = cej_pt_ml
                     
                     # Still compute axis for direction
                     if check_pts and len(check_pts) > 5:
                         pts_f = np.array(check_pts, dtype=np.float32).reshape(-1, 2)
                         mean, eigenvectors = cv2.PCACompute(pts_f, mean=None)
                         vec = eigenvectors[0]
                         base_dir = 1 if q in [3, 4] else -1
                         if vec[1] * base_dir < 0: vec = -vec
                         dx, dy_axis = vec[0], vec[1]
                
                elif check_pts and len(check_pts) > 5:
                    pts_f = np.array(check_pts, dtype=np.float32).reshape(-1, 2)
                    mean, eigenvectors = cv2.PCACompute(pts_f, mean=None)
                    vec = eigenvectors[0]
                    # Align with general vertical direction (Down for Lower, Up for Upper)
                    base_dir = 1 if q in [3, 4] else -1
                    if vec[1] * base_dir < 0:
                         vec = -vec
                    dx, dy_axis = vec[0], vec[1]
                    
                    # [Fallback]: User requests CEJ to Nerve distance.
                    # Estimate CEJ/Crest level as ~30-40% from the 'crown' end of the box
                    h_box = box[3] - box[1]
                    if q in [3, 4]: # Lower
                        # Apex is max Y. CEJ is approx box[1] + 0.3*h
                        start_y = box[1] + h_box * 0.35
                    else: # Upper
                        # Apex is min Y. CEJ is approx box[3] - 0.3*h
                        start_y = box[3] - h_box * 0.35
                    
                    if abs(dy_axis) > 0.01:
                        mean_pt = mean[0]
                        t_param = (start_y - mean_pt[1]) / dy_axis
                        start_x = mean_pt[0] + t_param * dx
                
                # Normalize direction
                mag = math.sqrt(dx*dx + dy_axis*dy_axis)
                if mag > 0: dx, dy_axis = dx/mag, dy_axis/mag
                
                # Trace
                dist_px = 0
                max_step = 1000
                found = False
                hit_x, hit_y = start_x, start_y
                h_img, w_img = target_mask.shape
                
                cx, cy = start_x, start_y # Update logic to start here
                
                for _ in range(max_step):
                    curr_x = int(cx + dx * dist_px)
                    curr_y = int(cy + dy_axis * dist_px)
                    if not (0 <= curr_x < w_img and 0 <= curr_y < h_img): break
                    if target_mask[curr_y, curr_x] > 0:
                        hit_x, hit_y = curr_x, curr_y
                        found = True
                        count_found += 1
                        break
                    dist_px += 1
                
                if found:
                    dist_mm = dist_px * mm_per_px
                    t['nerve_dist_mm'] = dist_mm
                    t['nerve_dist_line'] = [cx, cy, hit_x, hit_y]

                    # Overlap Logic: If dist < length of root? 
                    # Use simpler check: geometric overlap of mask
                    if dist_mm < 2.0: 
                         if q in [3, 4]: t['nerve_overlap'] = True
                         else: t['sinus_overlap'] = True

            except Exception as e:
                # print(f"Safety Calc Error {t.get('tooth_label')}: {e}")
                pass


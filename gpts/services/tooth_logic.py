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
        import sys
        sys.stdout.flush()  # Ensure previous buffered output is flushed
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
        image_width = context.get('image_width')
        if not image_width:
            for mask in (nerve_mask, sinus_mask):
                if mask is not None and hasattr(mask, 'shape') and len(mask.shape) >= 2:
                    image_width = int(mask.shape[1])
                    break
        if not image_width:
            x_candidates = []
            for tooth in final_teeth_objects:
                box = tooth.get('box')
                if box and len(box) >= 4:
                    x_candidates.append(float(box[2]))
            image_width = int(max(x_candidates) + avg_tooth_width) if x_candidates else 0
        
        # Include wisdom teeth and evaluate distal end sites first so 8s are
        # preferred when the terminal molar region is missing.
        all_fdis = [
            18,17,16,15,14,13,12,11, 28,27,26,25,24,23,22,21,
            48,47,46,45,44,43,42,41, 38,37,36,35,34,33,32,31
        ]
        
        def get_tooth_by_label(lbl):
            for t in final_teeth_objects:
                if t.get('tooth_label') == str(lbl): return t
            return None

        skip_fdis = set()
        terminal_pairs = [
            (1, 18, 17, 16),
            (2, 28, 27, 26),
            (3, 38, 37, 36),
            (4, 48, 47, 46),
        ]
        for quadrant, tooth8, tooth7, tooth6 in terminal_pairs:
            if tooth8 in present_fdis or tooth7 in present_fdis or tooth6 not in present_fdis:
                continue

            anchor = get_tooth_by_label(tooth6)
            anchor_box = anchor.get('box') if anchor else None
            if not anchor_box or len(anchor_box) < 4 or not image_width:
                continue

            anchor_width = max(float(anchor_box[2] - anchor_box[0]), avg_tooth_width, 1.0)
            if quadrant in (1, 4):
                terminal_space_px = max(float(anchor_box[0]), 0.0)
            else:
                terminal_space_px = max(float(image_width - anchor_box[2]), 0.0)

            estimated_terminal_slots = terminal_space_px / anchor_width
            if estimated_terminal_slots < 1.7:
                skip_fdis.add(tooth7)
                print(
                    f"[MISSING HEURISTIC] Q{quadrant} terminal ambiguity -> prefer {tooth8} over {tooth7} "
                    f"(slots={estimated_terminal_slots:.2f})",
                    flush=True,
                )

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
            if fdi in present_fdis or fdi in skip_fdis: continue
            
            slabel = str(fdi)
            odontogram_update[slabel] = "missing"
            
            mt = {'tooth_label': fdi, 'type': 'missing'}
            
            # --- Geometric Analysis for Missing Site ---
            q = fdi // 10
            n = fdi % 10
            
            # Find Neighbors: walk outward past consecutive missing teeth (up to 4 slots)
            def find_nearest_present(start_q, start_n, direction, max_steps=4):
                """Walk outward in direction (+1 or -1) to find nearest PRESENT tooth."""
                for step in range(1, max_steps + 1):
                    nn = start_n + direction * step
                    if nn < 1 or nn > 8:
                        # Handle midline crossing (Q3<->Q4 and Q1<->Q2)
                        if direction == -1 and start_n == 1:
                            cross_q = 4 if start_q == 3 else (3 if start_q == 4 else None)
                            if cross_q:
                                candidate = get_tooth_by_label(cross_q * 10 + 1)
                                if candidate: return candidate
                        break
                    candidate = get_tooth_by_label(start_q * 10 + nn)
                    if candidate: return candidate  # Found a present tooth
                return None

            # Determine which direction is "mesial" (toward midline) and "distal"
            # Q1: decreasing n = distal, increasing n = mesial -> midline at n=1
            # Q2: increasing n = distal, decreasing n = mesial -> midline at n=1
            # Q3: increasing n = distal, decreasing n = mesial -> midline at n=1
            # Q4: decreasing n = distal, increasing n = mesial -> midline at n=1
            t_n1 = find_nearest_present(q, n, -1)  # mesial direction
            t_n2 = find_nearest_present(q, n, +1)  # distal direction

            # Handle midline for mesial direction (Q3 n=1 -> cross to Q4 n=1)
            if t_n1 is None and n == 1:
                cross_q = 4 if q == 3 else (3 if q == 4 else None)
                if cross_q:
                    t_n1 = find_nearest_present(cross_q, 1, +1) or get_tooth_by_label(cross_q * 10 + 1)

            print(f"[GAP DEBUG] Tooth {fdi}: t_n1={t_n1.get('tooth_label') if t_n1 else None}, t_n2={t_n2.get('tooth_label') if t_n2 else None}")
            if fdi in (45, 46, 47, 48):
                lower_left_present = sorted(
                    [p for p in present_fdis if int(str(p)[0]) == 4],
                    key=lambda value: int(value),
                )
                print(
                    f"[GAP DEBUG] Q4 present_fdis before missing {fdi}: {lower_left_present}",
                    flush=True,
                )

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

            # 2. Calculate Gap Level Y & Collect Mask Outlines for Gap Dist
            valid_cejs = []
            valid_axes = []
            left_tooth_contours = []
            right_tooth_contours = []
            
            for i, t in enumerate([t_n1, t_n2]):
                if t and t.get('cej_center'): valid_cejs.append(t['cej_center'])
                
                # Calculate PCA axis from contour and collect contour for gap
                if t and t.get('contour'):
                    try:
                        pts = np.array(t['contour'], dtype=np.float32).reshape(-1, 2)
                        
                        # Store contour side for gap calculation
                        if i == 0: left_tooth_contours = pts
                        else: right_tooth_contours = pts
                        
                        if len(pts) > 5:
                            mean = np.mean(pts, axis=0)
                            centered = pts - mean
                            cov = np.cov(centered, rowvar=False)
                            evals, evecs = np.linalg.eigh(cov)
                            major_axis = evecs[:, np.argmax(evals)]
                            major_axis = major_axis / np.linalg.norm(major_axis)
                            # Ensure pointing toward root (Lower: Y+, Upper: Y-)
                            if q in [3, 4] and major_axis[1] < 0: major_axis = -major_axis
                            if q in [1, 2] and major_axis[1] > 0: major_axis = -major_axis
                            valid_axes.append((float(major_axis[0]), float(major_axis[1])))
                    except: pass

            # Determine Axis Vector from neighbors or fallback to vertical
            if valid_axes:
                avg_x = sum(v[0] for v in valid_axes) / len(valid_axes)
                avg_y = sum(v[1] for v in valid_axes) / len(valid_axes)
                norm = math.hypot(avg_x, avg_y)
                if norm > 0:
                    axis_vec = (avg_x / norm, avg_y / norm)
                else:
                    axis_vec = (0, 1) if q in [3, 4] else (0, -1)
            else:
                axis_vec = (0, 1) if q in [3, 4] else (0, -1)
            
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
                    # Use neighbor centroids (Box Center Y) for consistent level
                    center_ys = []
                    for vb in valid_boxes:
                        box = vb['box']
                        # Box Center Y
                        cy = (box[1] + box[3]) / 2
                        center_ys.append(cy)
                    gap_cy = sum(center_ys) / len(center_ys)

            # --- Calculate Mesiodistal Gap (Y-slice / contact-point method) ---
            gap_dist_mm = 0.0
            gap_line_pts = None  # [lx, ly, rx, ry] pixel coords for dotted line
            print(f"[GAP DEBUG] Tooth {fdi}: left_cnt={len(left_tooth_contours)}, right_cnt={len(right_tooth_contours)}, gap_cy={gap_cy}, gap_cx={gap_cx}", flush=True)
            if len(left_tooth_contours) > 0 and len(right_tooth_contours) > 0 and gap_cy is not None and gap_cx is not None:
                try:
                    is_lower = q in [3, 4]
                    lc_all = left_tooth_contours
                    rc_all = right_tooth_contours

                    # Ensure lc is LEFT (smaller centroid X) and rc is RIGHT
                    if np.mean(lc_all[:, 0]) > np.mean(rc_all[:, 0]):
                        lc_all, rc_all = rc_all, lc_all

                    # Step 1: filter to crown region only (above CEJ = gap_cy)
                    if is_lower:
                        lc_crown = lc_all[lc_all[:, 1] < gap_cy]
                        rc_crown = rc_all[rc_all[:, 1] < gap_cy]
                    else:
                        lc_crown = lc_all[lc_all[:, 1] > gap_cy]
                        rc_crown = rc_all[rc_all[:, 1] > gap_cy]

                    if len(lc_crown) > 0 and len(rc_crown) > 0:
                        # Keep only the GAP-FACING (inner) side of each crown using gap_cx
                        # lc (left tooth): only points with X LEFT of gap center → distal/inner face
                        # rc (right tooth): only points with X RIGHT of gap center → mesial/inner face
                        lc_inner = lc_crown[lc_crown[:, 0] <= gap_cx]
                        rc_inner = rc_crown[rc_crown[:, 0] >= gap_cx]
                        if len(lc_inner) == 0: lc_inner = lc_crown  # fallback
                        if len(rc_inner) == 0: rc_inner = rc_crown

                        # Y-slice scan on INNER (gap-facing) crown points only
                        if is_lower:
                            y_min = int(np.min(lc_inner[:, 1]))
                            y_max = int(gap_cy)
                        else:
                            y_min = int(gap_cy)
                            y_max = int(np.max(lc_inner[:, 1]))

                        scan_step = 2
                        band = max(4, (y_max - y_min) // 20)

                        best_gap_px = None
                        best_lx = best_ly = best_rx = best_ry = None

                        for y in range(y_min, y_max, scan_step):
                            lc_at_y = lc_inner[np.abs(lc_inner[:, 1] - y) <= band]
                            rc_at_y = rc_inner[np.abs(rc_inner[:, 1] - y) <= band]
                            if len(lc_at_y) == 0 or len(rc_at_y) == 0:
                                continue
                            lx = float(np.max(lc_at_y[:, 0]))  # right edge of left tooth
                            rx = float(np.min(rc_at_y[:, 0]))  # left edge of right tooth
                            if rx > lx:
                                h_gap = rx - lx
                                if best_gap_px is None or h_gap < best_gap_px:
                                    best_gap_px = h_gap
                                    best_lx, best_ly = lx, float(y)
                                    best_rx, best_ry = rx, float(y)

                        if best_gap_px is not None and best_gap_px > 0:
                            gap_dist_mm = best_gap_px * mm_per_px
                            gap_line_pts = [best_lx, best_ly, best_rx, best_ry]
                            print(f"[GAP] Tooth {fdi}: y={best_ly:.0f} lx={best_lx:.1f} rx={best_rx:.1f} gap={best_gap_px:.1f}px {gap_dist_mm:.2f}mm", flush=True)
                except Exception as e:
                    print(f"Error calculating mesiodistal gap for {fdi}: {e}")


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
                             'mesiodistal_gap_mm': gap_dist_mm,
                             'line_coords': [gap_cx, gap_cy, safe_end_x, safe_end_y],
                             'margin_line_coords': [safe_end_x, safe_end_y, p_target[0], p_target[1]], # Red
                             'type': 'vertical',
                             'safety_margin_mm': 2.0
                         }
                         # print(f"[IMPLANT DEBUG] Calc Safe Dist {dist_mm:.1f}mm, Margin {2.0}mm, Gap: {gap_dist_mm:.1f}mm")
            
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
            mt['mesiodistal_gap_mm'] = gap_dist_mm  # Always store at top level
            mt['gap_line_pts'] = list(gap_line_pts) if gap_line_pts else None  # [lx, ly, rx, ry]
            mt['box'] = box
            mt['guide_status'] = 'Success' if implant_guide else 'Failed'
            mt['t_n1_fdi'] = int(t_n1['tooth_label']) if t_n1 else None
            mt['t_n2_fdi'] = int(t_n2['tooth_label']) if t_n2 else None
            
            missing_teeth_list.append(mt)
        
        # --- Post-processing: divide gap equally for consecutive missing teeth sharing same neighbors ---
        # Group by (t_n1, t_n2) and count how many missing teeth share each group
        from collections import defaultdict
        group_map = defaultdict(list)
        for idx, mt in enumerate(missing_teeth_list):
            key = (mt.get('t_n1_fdi'), mt.get('t_n2_fdi'))
            if key[0] is not None and key[1] is not None:
                group_map[key].append(idx)

        for (n1, n2), idxs in group_map.items():
            if len(idxs) > 1:
                # Sort by FDI so we can assign proportional line positions
                idxs_sorted = sorted(idxs, key=lambda i: missing_teeth_list[i].get('tooth_label', 0))
                count = len(idxs_sorted)
                # Divide total gap equally
                for rank, idx in enumerate(idxs_sorted):
                    mt = missing_teeth_list[idx]
                    total_gap = mt.get('mesiodistal_gap_mm', 0)
                    if total_gap > 0:
                        mt['mesiodistal_gap_mm'] = round(total_gap / count, 2)
                        if mt.get('implant_guide'):
                            mt['implant_guide']['mesiodistal_gap_mm'] = mt['mesiodistal_gap_mm']
                    # Adjust gap_line_pts to show only this tooth's segment
                    pts = mt.get('gap_line_pts')
                    if pts and len(pts) == 4:
                        lx, ly, rx, ry = pts
                        seg_w = (rx - lx) / count
                        new_lx = lx + rank * seg_w
                        new_rx = lx + (rank + 1) * seg_w
                        mt['gap_line_pts'] = [new_lx, (ly+ry)/2, new_rx, (ly+ry)/2]
                    print(f"[GAP DIV] Tooth {mt.get('tooth_label')}: {total_gap:.1f}mm / {count} = {mt['mesiodistal_gap_mm']:.1f}mm", flush=True)

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
            # [USER REQUEST] Disable nerve calculation for present teeth
            # Only missing teeth (handled in find_missing_teeth) should show guides
            continue 

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
                
                # Direction
                dy = 1 if q in [3, 4] else -1
                
                # Trace
                dist_px = 0
                max_step = 1000
                found = False
                hit_y = cy
                
                h_img, w_img = target_mask.shape
                
                for _ in range(max_step):
                    curr_y = int(cy + dy * dist_px)
                    curr_x = int(cx)
                    
                    if not (0 <= curr_x < w_img and 0 <= curr_y < h_img): break
                    
                    if target_mask[curr_y, curr_x] > 0:
                        hit_y = curr_y
                        found = True
                        count_found += 1
                        break
                    dist_px += 1
                
                if found:
                    dist_mm = dist_px * mm_per_px
                    
                    # Safety Margin Visualization
                    pixels_per_mm = 1.0 / mm_per_px if mm_per_px > 0 else 20.0
                    safety_margin_px = 2.0 * pixels_per_mm
                    
                    safe_dist_px = max(0, dist_px - safety_margin_px)
                    safe_dist_mm = safe_dist_px * mm_per_px
                    
                    safe_end_y = cy + dy * safe_dist_px
                    hit_y_val = cy + dy * dist_px
                    
                    t['nerve_dist_mm'] = safe_dist_mm
                    t['nerve_dist_line'] = [cx, cy, cx, safe_end_y] # Safe (Yellow/Pink)
                    t['nerve_margin_line'] = [cx, safe_end_y, cx, hit_y_val] # Margin (Red)
                    
                    # Overlap Check
                    if dist_mm < 2.0: # Warning Threshold (Total Dist)
                         if q in [3, 4]: t['nerve_overlap'] = True
                         else: t['sinus_overlap'] = True

            except Exception as e:
                # print(f"Safety Calc Error {t.get('tooth_label')}: {e}")
                pass


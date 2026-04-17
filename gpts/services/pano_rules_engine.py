import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class RulesEngine:
    def __init__(
        self,
        img_width: int,
        img_height: int,
        k_implant: float = 1.5,
        anchor_confidence_threshold: float = 0.75,
        hard_anchor_confidence_threshold: float = 0.85,
    ):
        self.W = img_width
        self.H = img_height
        self.k_implant = k_implant
        self.anchor_confidence_threshold = anchor_confidence_threshold
        self.hard_anchor_confidence_threshold = hard_anchor_confidence_threshold
        self.debug_log = {}

    def run(self, objects: List[Dict[str, Any]], caries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point.
        objects: list of {'id', 'type', 'box', 'cx', 'cy', 'label_hint', ...}
        caries: list of {'id', 'box', 'confidence', ...}
        """
        # 1. Split Arches (Upper / Lower)
        upper_objs, lower_objs, y_split = self._split_arches(objects)
        self.debug_log['y_split'] = y_split

        # 2. Pre-calculate Metrics for both arches to Find Shared Midline
        #    (Many cases have good anchors on one arch but implants on the other)
        
        # A. Get Anchors
        up_anchors = self._get_valid_anchors(upper_objs, is_upper=True)
        low_anchors = self._get_valid_anchors(lower_objs, is_upper=False)
        
        # B. Calc Initial Metrics (Independent)
        up_metrics = self._calculate_metrics(up_anchors, is_upper=True)
        low_metrics = self._calculate_metrics(low_anchors, is_upper=False)
        
        # C. Decide Shared x_mid & dx Strategy
        # Criteria: Number of Incisor Anchors (11,21 vs 31,41) > Total Anchors
        up_incisors = len([a for a in up_anchors if str(a['label_hint'])[-1] == '1'])
        low_incisors = len([a for a in low_anchors if str(a['label_hint'])[-1] == '1'])
        
        final_x_mid = self.W / 2.0
        mid_source = "default"
        sync_strategy = "none" # none, use_upper, use_lower, avg
        
        if up_incisors > 0 and low_incisors > 0:
            # Both have incisors -> Average them
            final_x_mid = (up_metrics['x_mid'] + low_metrics['x_mid']) / 2.0
            mid_source = "avg_both"
            sync_strategy = "avg"
        elif low_incisors > 0:
            # Trust lower
            final_x_mid = low_metrics['x_mid']
            mid_source = "lower_incisors"
            sync_strategy = "use_lower"
        elif up_incisors > 0:
            # Trust upper
            final_x_mid = up_metrics['x_mid']
            mid_source = "upper_incisors"
            sync_strategy = "use_upper"
        else:
            # No incisors, rely on total anchor count/spread
            if len(low_anchors) > len(up_anchors):
                final_x_mid = low_metrics['x_mid']
                mid_source = "lower_majority"
                sync_strategy = "use_lower"
            elif len(up_anchors) > len(low_anchors):
                final_x_mid = up_metrics['x_mid']
                mid_source = "upper_majority"
                sync_strategy = "use_upper"
            else:
                 # Tie or both 0
                 if len(up_anchors) > 0:
                     final_x_mid = (up_metrics['x_mid'] + low_metrics['x_mid']) / 2.0
                     mid_source = "avg_anchors"
                     sync_strategy = "avg"
                 else:
                     final_x_mid = self.W / 2.0
                     # Keep strategy none
        
        self.debug_log['shared_x_mid'] = final_x_mid
        self.debug_log['mid_source'] = mid_source

        # Metric Synchronization (x_mid, dx, w_base)
        # Why? If one arch is reliable, its spacing (dx) is likely better than standard fallback.
        if sync_strategy == "use_lower":
            up_metrics['x_mid'] = final_x_mid
            # Only override dx if Upper was weak (fallback) or we want strict symmetry
            # Here we enforce symmetry for robustness in implant cases
            up_metrics['dx'] = low_metrics['dx']
            up_metrics['w_base'] = low_metrics['w_base']
        elif sync_strategy == "use_upper":
            low_metrics['x_mid'] = final_x_mid
            low_metrics['dx'] = up_metrics['dx']
            low_metrics['w_base'] = up_metrics['w_base']
        elif sync_strategy == "avg":
            # Sync Center
            up_metrics['x_mid'] = final_x_mid
            low_metrics['x_mid'] = final_x_mid
            # Sync dx? Maybe average them if they are close, but keep independent if distinct?
            # Safe bet: Average them to enforce uniform scale if both are valid
            avg_dx = (up_metrics['dx'] + low_metrics['dx']) / 2.0
            avg_w = (up_metrics['w_base'] + low_metrics['w_base']) / 2.0
            up_metrics['dx'] = avg_dx
            low_metrics['dx'] = avg_dx
            up_metrics['w_base'] = avg_w
            low_metrics['w_base'] = avg_w

        self.debug_log['upper_metrics'] = up_metrics
        self.debug_log['lower_metrics'] = low_metrics

        # 3. Process Arches with finalized metrics
        slots_upper = self._process_arch_with_metrics(upper_objs, up_metrics, is_upper=True)
        slots_lower = self._process_arch_with_metrics(lower_objs, low_metrics, is_upper=False)

        # Merge slots
        all_slots = {**slots_upper, **slots_lower}

        # 4. Assign Caries
        caries_assignment = self._assign_caries(all_slots, caries)

        return {
            "slots": all_slots,
            "caries_assignment": caries_assignment,
            "debug": self.debug_log
        }
    
    def _get_valid_anchors(self, objects, is_upper):
        anchors = [
            o for o in objects
            if o['type'] == 'natural'
            and o.get('label_hint')
            and float(o.get('conf', 0.0) or 0.0) >= self.anchor_confidence_threshold
        ]
        valid = []
        for anc in anchors:
            try:
                fdi = int(anc['label_hint'])
                q = fdi // 10
                if is_upper and q in [1, 2]: valid.append(anc)
                elif not is_upper and q in [3, 4]: valid.append(anc)
            except: continue
        return valid

    def _process_arch_with_metrics(self, objects: List[Dict[str, Any]], metrics: Dict[str, Any], is_upper: bool) -> Dict[str, Any]:
        """Process a single arch using pre-calculated metrics."""
        
        # C. Generate Slots
        slots_grid = self._generate_slots(metrics, is_upper)

        # D. Assign Objects
        assigned_slots = self._assign_objects(slots_grid, objects, metrics, is_upper)
        assigned_slots = self._prioritize_natural_labels(assigned_slots, objects, is_upper)
        assigned_slots = self._reflow_implant_segments(assigned_slots, is_upper)
        
        # E. Spatial Consistency Check (Swap/Ambiguous check)
        final_slots = self._enforce_spatial_order(assigned_slots, is_upper)
        
        return final_slots

    def _split_arches(self, objects: List[Dict[str, Any]]) -> Tuple[List, List, float]:
        """
        Split objects into upper and lower keys using a robust 'Occlusal Plane' curve.
        Strategy:
        1. Identify 'Interface Points' (Bite Line) from natural anchors.
           - Upper Tooth -> Bottom Edge (y2)
           - Lower Tooth -> Top Edge (y1)
        2. Detect and Remove Outliers (e.g. Mislabeled teeth whose Root Tips are picked).
        3. Fit a curve to the cleaned Bite Line.
        4. Split all objects relative to this curve.
        """
        anchors = [o for o in objects if o['type'] == 'natural' and o.get('label_hint')]
        
        interface_pts = []
        
        # 1. Collect Interface Points based on Labels
        if anchors:
            for a in anchors:
                try:
                    # If label exists, trust it initially to pick the Edge
                    fdi = int(a['label_hint'])
                    q = fdi // 10
                    if q in [1, 2]: # Upper
                        interface_pts.append((a['cx'], a['box'][3])) # y2 (Bottom)
                    elif q in [3, 4]: # Lower
                        interface_pts.append((a['cx'], a['box'][1])) # y1 (Top)
                except:
                    pass
        
        # Fallback: If no labels, use Naive Median to guess Upper/Lower
        if not interface_pts:
            if anchors:
                y_centers = [o['cy'] for o in anchors]
                naive_mid = float(np.median(y_centers))
                for a in anchors:
                    if a['cy'] < naive_mid: # Guess Upper
                        interface_pts.append((a['cx'], a['box'][3]))
                    else: # Guess Lower
                        interface_pts.append((a['cx'], a['box'][1]))
            else:
                # Absolute fallback
                all_ys = [o['cy'] for o in objects]
                scalar_y = float(np.median(all_ys)) if all_ys else self.H/2.0
                upper = [o for o in objects if o['cy'] < scalar_y]
                lower = [o for o in objects if o['cy'] >= scalar_y]
                return upper, lower, scalar_y

        # 2. Iterative Curve Fitting (Sigma Clipping)
        pts = np.array(interface_pts)
        X = pts[:, 0]
        Y = pts[:, 1]
        
        # Initial Fit
        deg = 2 if len(X) > 4 else 1
        mask = np.ones(len(X), dtype=bool)
        
        best_poly = None
        
        for _ in range(3): # 3 Iterations of clipping
            if np.sum(mask) < 2: break
            
            try:
                coeffs = np.polyfit(X[mask], Y[mask], deg)
                poly_fn = np.poly1d(coeffs)
                best_poly = poly_fn
                
                # Calc residuals
                preds = poly_fn(X)
                residuals = np.abs(Y - preds)
                
                # Threshold: 2.0 Sigma (or fixed px, e.g. 50px for panoramic)
                sigma = np.std(residuals[mask])
                threshold = max(30.0, 2.0 * sigma) 
                
                new_mask = residuals < threshold
                if np.array_equal(mask, new_mask): break
                mask = new_mask
            except:
                break
        
        # If fit failed completely, use mean
        if best_poly is None:
            scalar_y = float(np.mean(Y))
            poly_fn = lambda x: scalar_y
        else:
            poly_fn = best_poly

        # 3. Classify Objects
        upper = []
        lower = []
        
        for o in objects:
            cy = o['cy']
            cx = o['cx']
            boundary_y = poly_fn(cx)
            
            if cy < boundary_y:
                upper.append(o)
            else:
                lower.append(o)
                
        # Calculate mean y_split for logging
        y_split_scalar = float(np.mean(Y[mask])) if len(Y[mask]) > 0 else float(np.mean(Y))
        
        # Save Curve Coefficients for Visualization
        if best_poly is not None:
            self.debug_log['curve_coeffs'] = best_poly.coeffs.tolist()
        else:
            self.debug_log['curve_coeffs'] = None
        
        return upper, lower, y_split_scalar

    def _calculate_metrics(self, anchors: List[Dict[str, Any]], is_upper: bool) -> Dict[str, Any]:
        """Calculate statistical metrics (IQR based). Fallback if < 2 anchors."""
        if len(anchors) < 2:
            return self._get_fallback_metrics(is_upper)

        # 1. dx (Standard gap distance) first
        sorted_anchors = sorted(anchors, key=lambda x: x['cx'])
        deltas = []
        for i in range(len(sorted_anchors) - 1):
            pair_dist = sorted_anchors[i+1]['cx'] - sorted_anchors[i]['cx']
            try:
                fdi1 = int(sorted_anchors[i]['label_hint'])
                fdi2 = int(sorted_anchors[i+1]['label_hint'])
                step = abs(self._fdi_to_linear(fdi2) - self._fdi_to_linear(fdi1))
                if step > 0:
                    normalized_dx = pair_dist / step
                    deltas.append(normalized_dx)
            except:
                pass
        
        if deltas:
            # IQR Clipping for dx
            q1, q3 = np.percentile(deltas, [25, 75])
            iqr = q3 - q1
            valid_deltas = [d for d in deltas if (q1 - 1.5*iqr) <= d <= (q3 + 1.5*iqr)]
            dx = float(np.median(valid_deltas) if valid_deltas else np.median(deltas))
        else:
            dx = self.W / 14.0 # Fallback

        # 2. x_mid (Midline) - Project from Anchors
        # Instead of just mean(cx), we calculate "Where would x_mid be based on this anchor?"
        # x_mid = anchor_x - (offset * dx)
        x_mid_estimates = []
        
        for a in anchors:
            try:
                fdi = int(a['label_hint'])
                # Get ideal offset from mid
                offset = self._get_ideal_offset(fdi)
                
                # Reverse to find mid
                # ideal_x = x_mid + offset*dx
                # => x_mid = cx - offset*dx
                est = a['cx'] - (offset * dx)
                x_mid_estimates.append(est)
            except: pass
            
        if x_mid_estimates:
            # Robust mean (IQR clipping again? or simple mean)
            # Simple mean is fine as anchors are trusted
            x_mid = float(np.mean(x_mid_estimates))
        else:
             # Fallback
             x_mid = self.W / 2.0

        # 3. w_base (Tooth width)
        widths = [a['box'][2] - a['box'][0] for a in anchors]
        if widths:
            w_med = float(np.median(widths))
            w_p80 = float(np.percentile(widths, 80))
            tail = w_p80 / w_med if w_med > 0 else 1.0
            
            # Dynamic ratio based on tail
            if tail >= 1.25: r = 0.70
            elif tail <= 1.10: r = 0.80
            else: r = 0.70 + (0.80 - 0.70) * (1.25 - tail) / (0.15) 
            w_base = r * w_med
        else:
            w_base = dx * 0.8

        return {"x_mid": x_mid, "dx": dx, "w_base": w_base}

    def _get_ideal_offset(self, fdi):
        """Helper to get signed offset from x_mid (in units of dx)"""
        q, n = fdi // 10, fdi % 10
        # Right side (Image Left): 1x, 4x. n=1 is near mid. n=8 is far left.
        # offset should be negative.
        # n=1: offset -0.5. n=8: offset -7.5?
        
        if q in [1, 4]: 
            return -(n - 0.5)
        else:           
            return (n - 0.5)

    def _get_fallback_metrics(self, is_upper: bool) -> Dict[str, Any]:
        """Heuristic defaults for edentulous cases."""
        return {
            "x_mid": self.W / 2.0,
            "dx": self.W / 14.0, # Approx 14 teeth width
            "w_base": (self.W / 14.0) * 0.8
        }

    def _generate_slots(self, metrics: Dict[str, Any], is_upper: bool) -> Dict[str, Dict]:
        """Generate ideal 32-slot grid coords."""
        slots = {}
        x_mid = metrics['x_mid']
        dx = metrics['dx']
        
        target_qs = [1, 2] if is_upper else [4, 3] # Ordered Left-to-Right ideally?
        
        for q in target_qs:
            for n in range(1, 9):
                fdi = q * 10 + n
                offset = self._get_ideal_offset(fdi)
                ideal_x = x_mid + (offset * dx)
                slots[str(fdi)] = {
                    "fdi": str(fdi),
                    "ideal_x": ideal_x,
                    "status": "unassigned",
                    "object_id": None,
                    "candidates": [],
                    "hard_anchor": False,
                }
        return slots

    def _assign_objects(self, slots: Dict[str, Dict], objects: List[Dict[str, Any]], metrics: Dict[str, Any], is_upper: bool) -> Dict[str, Dict]:
        dx = metrics['dx']
        w_base = metrics['w_base']
        anchored_object_ids = set()
        
        # 1. Assign Anchors (Absolute Trust -> Conditional Trust)
        x_mid = metrics.get('x_mid', self.W/2)
        
        for obj in objects:
            is_high_conf_anchor = (
                obj['type'] == 'natural'
                and obj.get('label_hint')
                and float(obj.get('conf', 0.0) or 0.0) >= self.anchor_confidence_threshold
            )
            if is_high_conf_anchor:
                lbl = str(obj['label_hint'])
                if lbl in slots:
                    # Validate Side (Cross-Arch Check)
                    # Q1 (1x), Q4 (4x) should be Image Left (cx < x_mid)
                    # Q2 (2x), Q3 (3x) should be Image Right (cx > x_mid)
                    # Use a lenient buffer (e.g. 50px or 10% of W) to allow central incisors near midline.
                    
                    try:
                        q = int(lbl) // 10
                        is_left_side_label = (q in [1, 4]) # Image Left
                        is_right_side_label = (q in [2, 3]) # Image Right
                        
                        cx = obj['cx']
                        buffer = self.W * 0.1 # 10% buffer
                        
                        valid_side = True
                        if is_left_side_label and cx > (x_mid + buffer):
                            valid_side = False
                        elif is_right_side_label and cx < (x_mid - buffer):
                            valid_side = False
                            
                        if not valid_side:
                            # Skip this anchor assignment (treat as unassigned later)
                            continue
                    except:
                        pass

                    # Force assignment
                    slots[lbl]['status'] = 'confirmed'
                    slots[lbl]['object_id'] = obj['id']
                    slots[lbl]['candidates'].append(obj)
                    slots[lbl]['hard_anchor'] = float(obj.get('conf', 0.0) or 0.0) >= self.hard_anchor_confidence_threshold
                    anchored_object_ids.add(obj['id'])
                    # Update ideal_x to actual anchor x for better local accuracy?
                    # Yes, snap grid to anchor.
                    slots[lbl]['ideal_x'] = obj['cx']

        # 2. Assign Non-Anchors (Implants, etc.)
        unassigned_objs = [
            o for o in objects
            if o['id'] not in anchored_object_ids
            and (
                o['type'] != 'natural'
                or not o.get('label_hint')
                or float(o.get('conf', 0.0) or 0.0) < self.anchor_confidence_threshold
                or str(o.get('label_hint')) not in slots
            )
        ]
        
        # Sort objects by X position to process sequentially (optional but good for debugging)
        unassigned_objs.sort(key=lambda x: x['cx'])

        for obj in unassigned_objs:
            assigned_for_order = []
            for fdi, slot in slots.items():
                if slot['object_id'] is None or not slot['candidates']:
                    continue
                assigned_for_order.append({
                    'x': slot['candidates'][0]['cx'],
                    'order': self._fdi_to_linear_order(int(fdi), is_upper),
                })
            assigned_for_order.sort(key=lambda item: item['x'])

            min_allowed_order = None
            max_allowed_order = None
            if obj['type'] == 'natural':
                left_neighbor = next((item for item in reversed(assigned_for_order) if item['x'] < obj['cx']), None)
                right_neighbor = next((item for item in assigned_for_order if item['x'] > obj['cx']), None)
                if left_neighbor is not None:
                    min_allowed_order = left_neighbor['order'] + 1
                if right_neighbor is not None:
                    max_allowed_order = right_neighbor['order'] - 1
                if min_allowed_order is not None and max_allowed_order is not None and min_allowed_order > max_allowed_order:
                    min_allowed_order = None
                    max_allowed_order = None

            # Find best slot candidates
            best_slot = None
            min_dist = float('inf')
            second_dist = float('inf')
            
            # Simple distance to empty slots
            candidates = []
            eligible_free_slots = []
            
            for fdi, slot in slots.items():
                if slot['object_id'] is not None: continue # Already taken

                if obj['type'] == 'natural':
                    slot_order = self._fdi_to_linear_order(int(fdi), is_upper)
                    if min_allowed_order is not None and slot_order < min_allowed_order:
                        continue
                    if max_allowed_order is not None and slot_order > max_allowed_order:
                        continue
                    eligible_free_slots.append((fdi, slot_order))
                
                dist = abs(obj['cx'] - slot['ideal_x'])
                
                # Basic Threshold: 0.5 * dx + implicit tolerance?
                # Actually, standard grid approach.
                # Let's say max range is +/- 0.6 * dx
                threshold = dx * 0.6
                if obj['type'] == 'natural' and (min_allowed_order is not None or max_allowed_order is not None):
                    threshold = dx * 1.25
                
                if obj['type'] == 'implant':
                    threshold = dx * self.k_implant # Use Tunable k_implant
                
                if dist < threshold:
                    candidates.append((dist, fdi))
            
            # Sort candidates by distance
            candidates.sort(key=lambda x: x[0])
            
            # For natural teeth, left/right order wins over distance when only one
            # free slot remains between confirmed neighbors. This fixes distal-end
            # cases like 37 -> 38 where the object is visible but slightly off-grid.
            if obj['type'] == 'natural' and len(eligible_free_slots) == 1:
                forced_fdi = eligible_free_slots[0][0]
                slots[forced_fdi]['status'] = 'confirmed'
                slots[forced_fdi]['object_id'] = obj['id']
                slots[forced_fdi]['candidates'].append(obj)
                continue

            if candidates:
                best_dist, best_fdi = candidates[0]
                
                # Check for conflict? (Greedy: first come first served here?)
                # We sorted objects by X, so left-most takes left-most slot.
                # However, if two objects compete for same slot?
                # This simple loop might fail if order is mixed.
                # BUT, since we sorted objects by X and slots are spatial, it should map well.
                
                # Assign
                slots[best_fdi]['status'] = 'confirmed'
                slots[best_fdi]['object_id'] = obj['id']
                slots[best_fdi]['candidates'].append(obj)
        
        return slots

    def _prioritize_natural_labels(self, slots: Dict[str, Dict], objects: List[Dict[str, Any]], is_upper: bool) -> Dict[str, Dict]:
        natural_reclaim_threshold = 0.5

        def primary_candidate(slot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            candidates = slot.get('candidates') or []
            return candidates[0] if candidates else None

        def clear_slot(slot_id: str) -> None:
            slots[slot_id]['status'] = 'unassigned'
            slots[slot_id]['object_id'] = None
            slots[slot_id]['candidates'] = []
            slots[slot_id]['hard_anchor'] = False

        def find_slot_by_object_id(object_id: Any) -> Optional[str]:
            for slot_id, slot in slots.items():
                if slot.get('object_id') == object_id:
                    return slot_id
            return None

        displaced_implants: List[Tuple[Dict[str, Any], int]] = []
        natural_candidates = [
            obj for obj in objects
            if obj.get('type') == 'natural'
            and obj.get('label_hint')
            and float(obj.get('conf', 0.0) or 0.0) >= natural_reclaim_threshold
            and str(obj.get('label_hint')) in slots
        ]
        natural_candidates.sort(key=lambda obj: float(obj.get('conf', 0.0) or 0.0), reverse=True)

        for obj in natural_candidates:
            target_slot_id = str(obj['label_hint'])
            target_slot = slots[target_slot_id]
            target_candidate = primary_candidate(target_slot)

            if target_slot.get('object_id') == obj['id']:
                target_slot['hard_anchor'] = target_slot.get('hard_anchor', False) or (
                    float(obj.get('conf', 0.0) or 0.0) >= self.hard_anchor_confidence_threshold
                )
                continue

            if target_slot.get('hard_anchor'):
                continue

            current_slot_id = find_slot_by_object_id(obj['id'])
            if current_slot_id and current_slot_id != target_slot_id:
                clear_slot(current_slot_id)

            if target_slot.get('object_id') is not None and target_candidate and str(target_candidate.get('type', '')) == 'implant':
                displaced_implants.append((target_candidate, int(target_slot_id) // 10))

            target_slot['status'] = 'confirmed'
            target_slot['object_id'] = obj['id']
            target_slot['candidates'] = [obj]
            target_slot['hard_anchor'] = float(obj.get('conf', 0.0) or 0.0) >= self.hard_anchor_confidence_threshold
            target_slot['ideal_x'] = obj['cx']

            print(
                f"[RULE DEBUG] natural reclaim slot {target_slot_id}: obj_id={obj['id']} conf={float(obj.get('conf', 0.0) or 0.0):.3f}",
                flush=True,
            )

        for implant_obj, quadrant in displaced_implants:
            preferred_slots = self._quadrant_slot_ids(quadrant)
            free_slots = [slot_id for slot_id in preferred_slots if slots[slot_id].get('object_id') is None]
            if not free_slots:
                free_slots = [slot_id for slot_id, slot in slots.items() if slot.get('object_id') is None]
            if not free_slots:
                continue

            best_slot_id = min(
                free_slots,
                key=lambda slot_id: abs(float(implant_obj.get('cx', 0.0) or 0.0) - float(slots[slot_id]['ideal_x'])),
            )
            slots[best_slot_id]['status'] = 'confirmed'
            slots[best_slot_id]['object_id'] = implant_obj['id']
            slots[best_slot_id]['candidates'] = [implant_obj]
            slots[best_slot_id]['hard_anchor'] = False

            print(
                f"[RULE DEBUG] displaced implant reassigned -> slot {best_slot_id}: obj_id={implant_obj['id']}",
                flush=True,
            )

        return slots

    def _quadrant_slot_ids(self, quadrant: int) -> List[str]:
        if quadrant in [1, 4]:
            return [str(quadrant * 10 + n) for n in range(8, 0, -1)]
        return [str(quadrant * 10 + n) for n in range(1, 9)]

    def _reflow_implant_segments(self, slots: Dict[str, Dict], is_upper: bool) -> Dict[str, Dict]:
        quadrants = [1, 2] if is_upper else [4, 3]

        def primary_candidate(slot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            candidates = slot.get('candidates') or []
            return candidates[0] if candidates else None

        for quadrant in quadrants:
            slot_ids = self._quadrant_slot_ids(quadrant)
            segments: List[List[str]] = []
            current_segment: List[str] = []

            for slot_id in slot_ids:
                slot = slots[slot_id]
                candidate = primary_candidate(slot)
                candidate_type = str(candidate.get('type', '')) if candidate else ''
                is_assignable = (
                    slot.get('object_id') is None
                    or (candidate_type == 'implant' and not slot.get('hard_anchor'))
                )
                if is_assignable:
                    current_segment.append(slot_id)
                else:
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = []
            if current_segment:
                segments.append(current_segment)

            for segment in segments:
                implant_objects: List[Dict[str, Any]] = []
                current_implant_slots: List[str] = []

                for slot_id in segment:
                    slot = slots[slot_id]
                    candidate = primary_candidate(slot)
                    if slot.get('object_id') is not None and candidate and str(candidate.get('type', '')) == 'implant':
                        current_implant_slots.append(slot_id)
                        implant_objects.append(candidate)

                if len(implant_objects) <= 1:
                    continue

                implant_objects.sort(key=lambda item: float(item.get('cx', 0.0) or 0.0))
                window_size = len(implant_objects)
                windows = [segment[idx:idx + window_size] for idx in range(0, len(segment) - window_size + 1)]
                if not windows:
                    continue

                def score_window(window: List[str]) -> float:
                    return sum(
                        abs(float(obj.get('cx', 0.0) or 0.0) - float(slots[slot_id]['ideal_x']))
                        for obj, slot_id in zip(implant_objects, window)
                    )

                best_window = min(windows, key=score_window)
                if current_implant_slots == best_window:
                    continue

                print(
                    f"[RULE DEBUG] implant serial reflow q{quadrant}: {current_implant_slots} -> {best_window}",
                    flush=True,
                )

                for slot_id in current_implant_slots:
                    slots[slot_id]['status'] = 'unassigned'
                    slots[slot_id]['object_id'] = None
                    slots[slot_id]['candidates'] = []
                    slots[slot_id]['hard_anchor'] = False

                for obj, slot_id in zip(implant_objects, best_window):
                    slots[slot_id]['status'] = 'confirmed'
                    slots[slot_id]['object_id'] = obj['id']
                    slots[slot_id]['candidates'] = [obj]
                    slots[slot_id]['hard_anchor'] = False

        return slots

    def _enforce_spatial_order(self, slots: Dict[str, Dict], is_upper: bool) -> Dict[str, Dict]:
        """Check if assigned objects violate spatial order."""
        # Get assigned FDIs in sequence
        # Upper: 18 -> 11 -> 21 -> 28 (Left to Right on Image? No.)
        # Image Left is Patient Right (18). Image Right is Patient Left (28).
        # So X increases from 18 to 28.
        
        # 1. Collect all assigned slots
        assigned = []
        for fdi, slot in slots.items():
            if slot['object_id'] is not None:
                # get primary obj x
                obj = slot['candidates'][0]
                assigned.append({
                    'fdi': int(fdi),
                    'x': obj['cx'],
                    'slot': slot
                })
        
        if not assigned: return slots
        
        # Sort by actual X
        assigned.sort(key=lambda k: k['x'])
        
        # Check if FDI sequence is monotonic
        # Upper: 18..11, 21..28.
        # Linear map: 18->0 ... 28->15
        
        for i in range(len(assigned) - 1):
            curr = assigned[i]
            next_b = assigned[i+1]
            
            idx1 = self._fdi_to_linear_order(curr['fdi'], is_upper)
            idx2 = self._fdi_to_linear_order(next_b['fdi'], is_upper)
            
            # If physical Left (curr) has higher logical index (idx1) than physical Right (idx2),
            # it's a violation. (Indices should increase Left->Right).
            if idx1 > idx2:
                # If they are essentially adjacent (or close), let's SWAP them.
                # Heuristic: Swapping fixes the order locally.
                
                fdi1_str = str(curr['fdi'])
                fdi2_str = str(next_b['fdi'])
                if slots[fdi1_str].get('hard_anchor') or slots[fdi2_str].get('hard_anchor'):
                    continue
                
                # Swap object_id
                tmp_id = slots[fdi1_str]['object_id']
                slots[fdi1_str]['object_id'] = slots[fdi2_str]['object_id']
                slots[fdi2_str]['object_id'] = tmp_id
                
                # Swap candidates list
                tmp_cand = slots[fdi1_str]['candidates']
                slots[fdi1_str]['candidates'] = slots[fdi2_str]['candidates']
                slots[fdi2_str]['candidates'] = tmp_cand

                tmp_anchor = slots[fdi1_str].get('hard_anchor', False)
                slots[fdi1_str]['hard_anchor'] = slots[fdi2_str].get('hard_anchor', False)
                slots[fdi2_str]['hard_anchor'] = tmp_anchor
                
                # Mark as corrected (confirmed)
                slots[fdi1_str]['status'] = 'confirmed'
                slots[fdi2_str]['status'] = 'confirmed'
                
                # Update 'assigned' array for next iteration? 
                # Swapping fdi1 and fdi2 effectively fixes THIS pair.
                # But 'assigned' list has stale data (fdi mapping).
                # Actually, 'assigned' is sorted by X.
                # curr is Left Object. next_b is Right Object.
                # Originally: curr->47 (idx large), next_b->46 (idx small). Violation.
                # We want: Left Object -> 46. Right Object -> 47.
                # So we swap the CONTENTS of slot 47 and slot 46?
                # Wait. Slot 47 has Object A. Slot 46 has Object B.
                # We want Slot 47 to have Object B. Slot 46 to have Object A.
                # Yes.
        
        return slots

    def _fdi_to_linear_order(self, fdi: int, is_upper: bool) -> int:
        """Map FDI to 0..15 increasing X order on image."""
        q, n = fdi // 10, fdi % 10
        
        # Image Left (low X) -> Image Right (high X)
        # Upper: 18,17...11, 21...28
        # Lower: 48,47...41, 31...38
        
        if is_upper:
            if q == 1: return 8 - n # 18->0, 11->7
            else: return 7 + n      # 21->8, 28->15
        else:
            if q == 4: return 8 - n # 48->0, 41->7
            else: return 7 + n      # 31->8, 38->15

    def _assign_caries(self, slots: Dict[str, Dict], caries_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assign caries to slots based on intersection ratio >= 0.9."""
        assignment = {}
        
        if not caries_list: return assignment

        # For intersection, we need slot boxes. 
        # But slots only have X-center. We need to estimate Y-range.
        # We can look up the assigned object's box.
        
        for c in caries_list:
            c_box = c['box'] # x1,y1,x2,y2
            c_area = (c_box[2] - c_box[0]) * (c_box[3] - c_box[1])
            if c_area <= 0: continue
            
            best_ratio = 0.0
            best_slot = None
            
            # Check against all assigned slots (confirmed/ambiguous)
            for fdi, slot in slots.items():
                if slot['object_id'] is None: continue
                
                # Get object box
                # Note: candidates[0] is the primary object
                obj_box = slot['candidates'][0]['box']
                
                # Calc Intersection
                ix1 = max(c_box[0], obj_box[0])
                iy1 = max(c_box[1], obj_box[1])
                ix2 = min(c_box[2], obj_box[2])
                iy2 = min(c_box[3], obj_box[3])
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter_area = iw * ih
                
                ratio = inter_area / c_area
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_slot = fdi
            
            # Decision
            c_entry = {
                "id": c['id'],
                "label": c.get('label', 'caries'),
                "ratio": best_ratio,
                "status": "unassigned"
            }
            
            # Hard Gate: 0.9
            if best_ratio >= 0.9 and best_slot:
                c_entry['status'] = 'confirmed'
                c_entry['assigned_to'] = best_slot
            elif best_ratio >= 0.1 and best_slot:
                c_entry['status'] = 'ambiguous' # Overlap exists but weak
                c_entry['candidate'] = best_slot
            
            assignment[c['id']] = c_entry
            
        return assignment

    def _fdi_to_linear(self, fdi):
        """Helper to get linear index 1..16 from midline for one side?"""
        # This is strictly for calculating 'step' (how many slots apart)
        # 11->1, 12->2 ... 18->8
        # 21->1, ... 28->8
        return fdi % 10

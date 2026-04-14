import cv2
import numpy as np
from pathlib import Path
import math

class FeedbackGenerator:
    def __init__(self, base_url="http://localhost:5000", upload_folder="temp"):
        self.base_url = base_url
        self.upload_folder = Path(upload_folder)
        
    def _save_crop(self, img, box, name, base_url=None, save_dir=None, url_prefix=None, overlay_box=None, nerve_contours=None, sinus_contours=None, safety_guide=None) -> str:
        """Saves crop and returns relative URL path. Draws overlay_box if provided."""
        try:
            h, w = img.shape[:2]
            x1, y1, x2, y2 = map(int, box)
            margin = 40 # Increased to ~1.2x per request
            
            # --- Logic Update: 1.3~1.5x Bottom Padding for Nerve/Implant ---
            pad_bottom_ratio = 1.0
            if "nerve" in name.lower() or "missing" in name.lower() or "implant" in name.lower():
                 pad_bottom_ratio = 1.4 # User requested 1.3~1.5

            # Dynamic Expansion
            target_contours = []
            if nerve_contours: target_contours.extend(nerve_contours)
            if sinus_contours: target_contours.extend(sinus_contours)
            
            expansion_y1, expansion_y2 = y1, y2
            
            # Additional bottom padding calculation
            box_h = y2 - y1
            extra_pad = int(box_h * (pad_bottom_ratio - 1.0))
            expansion_y2 += extra_pad
            
            if target_contours:
                bx1, bx2 = x1 - margin, x2 + margin
                relevant_pts = []
                for cnt in target_contours:
                    c_arr = np.array(cnt)
                    mask = (c_arr[:, 0] >= bx1) & (c_arr[:, 0] <= bx2)
                    pts_in_x = c_arr[mask]
                    if len(pts_in_x) > 0:
                        relevant_pts.extend(pts_in_x)
                if relevant_pts:
                    rel_arr = np.array(relevant_pts)
                    min_cy = np.min(rel_arr[:, 1])
                    max_cy = np.max(rel_arr[:, 1])
                    expansion_y1 = min(expansion_y1, min_cy)
                    expansion_y2 = max(expansion_y2, max_cy)
            
            x1 = max(0, x1-margin)
            y1 = max(0, expansion_y1-margin)
            x2 = min(w, x2+margin)
            y2 = min(h, expansion_y2+margin)
            
            crop = img[y1:y2, x1:x2].copy() # Ensure copy for drawing
            if crop.size == 0: return ""
            crop_h, crop_w = crop.shape[:2]
            
            # Draw Overlay if present
            # [REMOVED] Redundant drawing (overlay_box, safety_guide, contours) 
            # because we are now cropping the Full Overlay Image which has these baked in.
            
            # Save
            if save_dir:
                crop_dir = Path(save_dir)
            else:
                crop_dir = self.upload_folder / "crops"
            
            crop_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{name}.jpg"
            save_path = crop_dir / filename
            cv2.imwrite(str(save_path), crop)
            
            # Return URL path (handled by static route)
            if url_prefix:
                return f"{url_prefix}/{filename}"
            
            url_base = base_url if base_url else self.base_url
            return f"{url_base}/api/v2/files/crops/{filename}"
        except Exception:
            return ""

    def _check_nerve_overlap(self, tooth_box, iac_mask):
        """Checks if tooth box overlaps with Inferior Alveolar Canal (IAC) mask"""
        if iac_mask is None: return False
        
        # Simple box overlap check on mask
        x1, y1, x2, y2 = map(int, tooth_box)
        h, w = iac_mask.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        roi = iac_mask[y1:y2, x1:x2]
        if np.count_nonzero(roi) > 10:
             return True
        return False
        
    def _check_sinus_risk(self, tooth_label, tooth_box, img_h):
        """
        Heuristic: Upper molars (16,17, 26,27) are close to Sinus.
        """
        try:
            fdi = int(tooth_label)
            # Upper Molars: 16, 17, 18, 26, 27, 28
            if fdi in [16, 17, 18, 26, 27, 28]:
                return True
        except:
            pass
        return False

    def generate_feedback(self, user_name, analysis_result, original_img, iac_result=None, base_url=None, crop_dir=None, crop_url_prefix=None):
        """
        Generates markdown text for GPTs.
        """
        teeth = analysis_result.get('teeth', [])
        
        # Prepare content
        warnings = []
        findings_text = []
        doctor_checks = []
        img_urls = []
        
        found_issues = False
        
        for t in teeth:
            fdi = t.get('tooth_label', 'Unknown')
            issues = []
            
            # 1. Caries
            if t.get('caries'):
                issues.append("Caries (Cavity)")
                doctor_checks.append(f"Tooth #{fdi}: Check depth of cavity. Is pulp involved?")
                found_issues = True
                
            # 2. Periapical
            if t.get('periapical'):
                issues.append("Periapical Lesion (Inflammation)")
                doctor_checks.append(f"Tooth #{fdi}: Requires vitality test. Root canal likely needed.")
                found_issues = True
                
            # 3. Bone Loss
            pbl_lvl = t.get('bone_loss_level', 0)
            if pbl_lvl >= 3:
                issues.append(f"Severe Bone Loss (Level {pbl_lvl})")
                doctor_checks.append(f"Tooth #{fdi}: Assess tooth mobility. Prognosis check.")
                found_issues = True
                
            # 4. Nerve/Sinus Overlap (Backend Flag)
            if t.get('nerve_overlap'):
                issues.append("Nerve Overlap (Critical)")
                doctor_checks.append(f"Tooth #{fdi}: **WARNING** Root tip overlaps with nerve canal. CT scan required.")
                found_issues = True
            
            if t.get('sinus_overlap'):
                # Only show sinus warning when combined with pathology
                has_pathology = t.get('caries') or t.get('periapical') or pbl_lvl >= 3
                if has_pathology:
                    issues.append("Sinus Overlap (Critical)")
                    doctor_checks.append(f"Tooth #{fdi}: **WARNING** Root tip overlaps with sinus floor. Check oro-antral risk.")
                    found_issues = True

            if issues:
                findings_to_highlight = []
                if 'findings' in t and t['findings']:
                     for f in t['findings']:
                         findings_to_highlight.append(f)
                
                generated_crops = set()
                local_img_mds = []
                
                for f in findings_to_highlight:
                    disease_name = f['type'].capitalize()
                    fname = f"{user_name}_{fdi}_{disease_name}"
                    
                    if fname in generated_crops: continue
                    
                    c_url = self._save_crop(
                        original_img, 
                        t['box'], 
                        fname, 
                        base_url=base_url,
                        save_dir=crop_dir,
                        url_prefix=crop_url_prefix,
                        overlay_box=f['box']
                    )
                    if c_url:
                        img_urls.append({"label": f"Tooth #{fdi} - {disease_name}", "url": c_url})
                        local_img_mds.append(f"![{disease_name}]({c_url})")
                        generated_crops.add(fname)
                
                if not generated_crops:
                     fallback_disease = "Issue"
                     if "Bone Loss" in str(issues): fallback_disease = "BoneLoss"
                     elif "Nerve" in str(issues): fallback_disease = "Nerve"
                     
                     fname = f"{user_name}_{fdi}_{fallback_disease}"
                     c_url = self._save_crop(
                        original_img, 
                        t['box'], 
                        fname, 
                        base_url=base_url,
                        save_dir=crop_dir,
                        url_prefix=crop_url_prefix
                     )
                     if c_url:
                         img_urls.append({"label": f"Tooth #{fdi}", "url": c_url})
                         local_img_mds.append(f"![Issue]({c_url})")

                md = f"### Tooth #{fdi}\n"
                
                if local_img_mds:
                    md += "\n" + "\n".join(local_img_mds) + "\n"
                
                md += f"**Findings**: {', '.join(issues)}\n"
                
                explanations = []
                for issue in issues:
                    if "Caries" in issue:
                        explanations.append("- **Caries (Cavity)**: Tooth decay caused by bacteria. If untreated, it can infection the pulp.")
                    elif "Periapical" in issue:
                        explanations.append("- **Periapical Lesion**: Inflammation at the root tip, assessing a chronic infection or nerve death.")
                    elif "Bone Loss" in issue:
                        explanations.append(f"- **{issue}**: Significant loss of jawbone support. Indicates periodontitis (gum disease).")
                    elif "Nerve" in issue:
                        explanations.append("- **Close to Nerve**: The root is very close to the main nerve canal. Surgical risks should be evaluated.")
                    elif "Sinus" in issue:
                        explanations.append("- **Close to Sinus**: The root tip is near the maxillary sinus floor. Caution needed during treatment.")
                
                if explanations:
                     md += "\n**Explanation**:\n" + "\n".join(explanations) + "\n"
                     
                findings_text.append(md)

        # Construct Final Text
        text = f"# Analysis for {user_name}\n\n"
        
        if not found_issues:
            text += "## Summary\nGood News! No major issues detected.\n"
        else:
            text += "## ⚠️ Critical Findings\n"
            text += "\n".join(findings_text)
            
            text += "\n## 🩺 Doctor Check List (Ask your Dentist)\n"
            for q in doctor_checks:
                text += f"- [ ] {q}\n"
                
            text += "\n## 💡 Recommendations\n"
            text += "Based on the findings, I recommend visiting a specialist.\n"
            text += "- For **Root Canals** (Nerve/Inflammation): Endodontist (보존과)\n"
            text += "- For **Gum Disease** (Bone Loss): Periodontist (치주과)\n"
            text += "- For **Extraction/Implants**: Oral Surgeon (구강외과)\n"
            
        return text, img_urls

    def generate_structured(self, user_name, analysis_result, original_img, iac_result=None, base_url=None, crop_dir=None, crop_url_prefix=None):
        """
        Generates structured JSON for frontend rendering.
        Returns: (summary_dict, findings_list, gpt_md)
        """
        import datetime
        
        teeth = analysis_result.get('teeth', [])
        nerve_contours = analysis_result.get('nerve_contours', [])
        sinus_contours = analysis_result.get('sinus_contours', [])
        
        # IAC Mask preparation
        iac_mask = None
        if iac_result is not None and iac_result.masks is not None:
             iac_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
             for m in iac_result.masks.xy:
                 pts = np.array(m, np.int32)
                 cv2.fillPoly(iac_mask, [pts], 255)
        
        findings_list = []
        total_findings = 0
        critical_count = 0
        
        for t in teeth:
            fdi = t.get('tooth_label', 'Unknown')
            issues = []
            severity = "info"
            
            # 1. Caries
            if t.get('caries'):
                issues.append("Caries (Cavity)")
                severity = "critical"
                
            # 2. Periapical Lesion
            if t.get('periapical'):
                issues.append("Periapical Lesion")
                severity = "critical"
                
            # 3. Bone Loss
            pbl_lvl = t.get('bone_loss_level', 0)
            if pbl_lvl >= 3:
                issues.append(f"Bone Loss (Level {pbl_lvl})")
                severity = "critical"
                
            # 4. Nerve/Sinus Overlap (Backend Flag)
            if t.get('nerve_overlap'):
                issues.append("Nerve Overlap (Critical)")
                if severity == "info": severity = "warn"
            
            if t.get('sinus_overlap'):
                # Only show sinus warning when combined with pathology
                has_pathology = t.get('caries') or t.get('periapical') or pbl_lvl >= 3
                if has_pathology:
                    issues.append("Sinus Overlap (Critical)")
                    if severity == "info": severity = "warn"
            
            # Legacy Overlap Fallback (just in case)
            is_lower_molar = False
            try:
                if 36 <= int(fdi) <= 38 or 46 <= int(fdi) <= 48: is_lower_molar = True
            except: pass
            
            if is_lower_molar and not t.get('nerve_overlap') and self._check_nerve_overlap(t['box'], iac_mask):
                issues.append("Close to Nerve (IAA)")
                if severity == "info": severity = "warn"

            if issues:
                total_findings += 1
                if severity == "critical":
                    critical_count += 1
                
                # Generate crops for each finding
                images = []
                findings_with_boxes = t.get('findings', [])
                generated = set()
                
                for f in findings_with_boxes:
                    disease_name = f['type'].capitalize()
                    fname = f"{user_name}_{fdi}_{disease_name}"
                    
                    if fname in generated: continue
                    
                    c_url = self._save_crop(
                        original_img, 
                        t['box'], 
                        fname, 
                        base_url=base_url,
                        save_dir=crop_dir,
                        url_prefix=crop_url_prefix,
                        overlay_box=f.get('box'),
                        nerve_contours=nerve_contours,
                        sinus_contours=sinus_contours,
                        safety_guide=t.get('safety_guide')
                    )
                    if c_url:
                        images.append({"label": f"Tooth #{fdi} - {disease_name}", "url": c_url})
                        generated.add(fname)
                
                # Fallback if no specific finding boxes
                if not images:
                     fallback = "Issue"
                     if "Caries" in str(issues): fallback = "Caries"
                     elif "Periapical" in str(issues): fallback = "Periapical"
                     elif "Bone Loss" in str(issues): fallback = "BoneLoss"
                     
                     fname = f"{user_name}_{fdi}_{fallback}"
                     c_url = self._save_crop(
                        original_img, 
                        t['box'], 
                        fname, 
                        base_url=base_url,
                        save_dir=crop_dir,
                        url_prefix=crop_url_prefix,
                        nerve_contours=nerve_contours,
                        sinus_contours=sinus_contours,
                        safety_guide=t.get('safety_guide')
                     )
                     if c_url:
                         images.append({"label": f"Tooth #{fdi}", "url": c_url})
                
                # Build explanation
                explanation = self._get_explanation(issues)
                recommendation = self._get_recommendation(issues)
                
                finding_obj = {
                    "tooth": str(fdi),
                    "severity": severity,
                    "title": f"Tooth #{fdi} – {', '.join([i.split('(')[0].strip() for i in issues])}",
                    "findings": issues,
                    "explanation": explanation,
                    "recommendation": recommendation,
                    "confidence": t.get('confidence', 0),
                    "images": images
                }
                findings_list.append(finding_obj)
        
        # Process Missing Teeth - Use pre-calculated data from inference
        missing_teeth = analysis_result.get('missing_teeth', [])
        
        for mt in missing_teeth:
            # handle dict findings
            if not isinstance(mt, dict): continue
            
            fdi = mt.get('tooth_label')
            box = mt.get('box')
            guide = mt.get('implant_guide')
            
            if not box: continue
            
            # Save Crop
            fname = f"{user_name}_{fdi}_missing"
            
            c_url = self._save_crop(
                original_img, 
                box, 
                fname, 
                base_url=base_url, 
                save_dir=crop_dir,
                url_prefix=crop_url_prefix,
                nerve_contours=nerve_contours,
                sinus_contours=sinus_contours
            )
            
            images = []
            if c_url:
                images.append({"label": f"Missing #{fdi}", "url": c_url})
            
            # Explanation
            explanation = "Tooth appears to be missing."
            recom = "Consult for Implant or Bridge."
            
            findings_details = ["Missing"]
            
            if guide:
                dist = guide.get('dist_mm', 0)
                mode = guide.get('mode', 'lower')
                
                findings_details.append(f"Implant Guide: {dist:.1f}mm")
                explanation += f" Vertical distance to Nerve/Sinus is approx {dist:.1f}mm."
                recom += " Check bone height availability."
                
                if dist < 6.0: 
                    if mode == 'upper':
                         findings_details.append("Warning: Sinus Proximity")
                         explanation += " **Bone height is low.** Sinus Lift procedure may be required."
                    else:
                         findings_details.append("Warning: Nerve Proximity")
                         explanation += " **Bone height is low.** Risk of nerve injury. Precise planning required."
            
            findings_list.append({
                "tooth": str(fdi),
                "severity": "Medium",
                "title": f"Tooth #{fdi} – Missing",
                "findings": findings_details,
                "explanation": explanation,
                "recommendation": recom,
                "confidence": 1.0,
                "images": images
            })
            total_findings += 1
            if guide: critical_count += 1 # consider guide presence as significant? Or only if low?

        
        # Summary
        if total_findings == 0:
            message = "Good news! No significant findings detected."
        elif critical_count > 0:
            message = f"⚠️ {critical_count}개 치아에서 중요 소견이 발견되었습니다. 치과 상담을 권장합니다."
        else:
            message = f"{total_findings}개 치아에서 소견이 발견되었습니다."
        
        summary = {
            "total_findings": total_findings,
            "critical_findings": critical_count,
            "message": message
        }
        
        # Generate gpt_markdown for direct Chat UI output
        gpt_md = f"## 🦷 분석 결과\n\n{message}\n\n"
        
        for f in findings_list:
            tooth_num = f.get('tooth', 'Unknown')
            gpt_md += f"---\n\n### {f['title']}\n\n"
            
            # Embed images first with preview label
            for img in f.get('images', []):
                gpt_md += f"![{tooth_num}번 치아 미리보기]({img['url']})\n\n"
            
            gpt_md += f"**{', '.join(f['findings'])}**\n\n"
            gpt_md += f"{f['explanation']}\n\n"
            gpt_md += f"✅ **권장:** {f['recommendation']}\n\n"
        
        return summary, findings_list, gpt_md
    
    def _get_explanation(self, issues):
        """Returns explanation text for the findings."""
        explanations = []
        for issue in issues:
            if "Caries" in issue:
                explanations.append("Suspected caries (cavity) caused by bacteria. If left untreated, it may progress to the nerve.")
            elif "Periapical" in issue:
                explanations.append("Apical lesion (inflammation at the root tip) is suspected. Root canal treatment may be required.")
            elif "Bone Loss" in issue:
                explanations.append("Alveolar bone loss is observed. This is a sign of periodontal disease (gum disease).")
            elif "Nerve" in issue:
                explanations.append("**Extraction Caution**: The tooth root overlaps with the mandibular nerve canal. Extreme caution is required due to the risk of nerve injury during extraction.")
            elif "Sinus" in issue:
                explanations.append("**Extraction Caution**: The tooth root overlaps with the maxillary sinus. Caution is required due to the risk of sinus perforation during extraction.")
        return " ".join(explanations)
    
    def _get_recommendation(self, issues):
        """Returns recommendation text."""
        if any("Caries" in i for i in issues):
            return "Resin or inlay treatment recommended"
        elif any("Periapical" in i for i in issues):
            return "Root canal treatment recommended"
        elif any("Bone Loss" in i for i in issues):
            return "Periodontology consultation recommended"
        elif any("Nerve" in i or "Sinus" in i for i in issues):
            return "CT scan and oral surgeon consultation required (Extraction Caution)"
        return "Regular checkups recommended"

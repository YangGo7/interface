# -*- coding: utf-8 -*-
from pathlib import Path
p = Path("backend/utils/report_v2.py")
text = p.read_text(errors="ignore")
old = """
                    except Exception as e:
                        print(f"Error cropping missing tooth {lbl}: {e}")`n                # Sync updated guide back for badge logic`n                guide = implant_guide if implant_guide else guide`n`n                # Implant suitability logic (deterministic thresholds)`n                if guide and isinstance(guide, dict):
                    dist = guide.get('dist_mm', 0)
                    mode = guide.get('mode', 'lower')
                    if dist >= 8.0:
                        implant_badge = '<span class=\"badge implant\">Implant: Feasible (??mm clearance)</span>'
                        implant_note = f"Clearance {dist:.1f}mm to {'sinus' if mode=='upper' else 'nerve'}: favorable for implant placement."
                    elif 6.0 <= dist < 8.0:
                        implant_badge = '<span class=\"badge overlap\">Implant: Borderline (6-8mm)</span>'
                        implant_note = f"Clearance {dist:.1f}mm: consider sinus lift/nerve safety planning before implant."
                    else:
                        implant_badge = '<span class=\"badge info\">Implant: Not recommended (<6mm)</span>'
                        implant_note = f"Clearance {dist:.1f}mm is insufficient; prefer grafting or alternative prosthesis."
"""
new = """
                    except Exception as e:
                        print(f"Error cropping missing tooth {lbl}: {e}")

                # Sync updated guide back for badge logic
                guide = implant_guide if implant_guide else guide

                # Implant suitability logic (deterministic thresholds)
                if guide and isinstance(guide, dict):
                    dist = guide.get('dist_mm', 0)
                    mode = guide.get('mode', 'lower')
                    if dist >= 8.0:
                        implant_badge = '<span class=\"badge implant\">Implant: Feasible (≥8mm clearance)</span>'
                        implant_note = f"Clearance {dist:.1f}mm to {'sinus' if mode=='upper' else 'nerve'}: favorable for implant placement."
                    elif 6.0 <= dist < 8.0:
                        implant_badge = '<span class=\"badge overlap\">Implant: Borderline (6-8mm)</span>'
                        implant_note = f"Clearance {dist:.1f}mm: consider sinus lift/nerve safety planning before implant."
                    else:
                        implant_badge = '<span class=\"badge info\">Implant: Not recommended (<6mm)</span>'
                        implant_note = f"Clearance {dist:.1f}mm is insufficient; prefer grafting or alternative prosthesis."
"""
if old not in text:
    raise SystemExit('pattern not found')
p.write_text(text.replace(old,new), encoding='utf-8')

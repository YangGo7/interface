# Panoramic Implant & Numbering Logic Update
**Date:** 2026-01-13
**Author:** Antigravity (AI Assistant)

## 1. Overview
This update addresses critical issues in the panoramic image analysis pipeline, specifically focusing on inaccurate tooth numbering, "Ghost Crown" artifacts, and the displacement of implants (e.g., drifting to wisdom tooth slots). A new **Global Assignment Optimization** approach has replaced the previous greedy mapping logic.

## 2. Key Problems Solved
*   **Numbering Collisions:** Implants and natural teeth conflicting for the same slot (e.g., 46 vs 47).
*   **Ghost Crowns:** Crown detections appearing as separate teeth (Pontics) or "Ghost" entities, causing valid implants to shift positions (e.g., 47 -> 48).
*   **Implant Separation:** Tilted implants where the Fixture and Crown were detected separately but failed to merge, creating duplicate teeth.
*   **Visualization Issues:** Lower teeth odontogram rendering upside down; missing "Layout On/Off" toggle.

## 3. Technical Implementation

### A. Global Assignment Optimization (Hungarian Algorithm)
*   **Replaced Greedy Logic:** Instead of iterating and assigning teeth one by one, we now collect all observations (Natural, Implant, Crown) and solve a **Global Cost Minimization** problem.
*   **Cost Matrix:**
    *   **Natural Teeth:** Heavy penalty for moving away from their `label_hint`. Natural teeth act as "Anchors".
    *   **Implants:** Flexible positioning based on distance to ideal slots.
    *   **Wisdom Tooth Penalty:** Added specific penalty (+200.0) for implants assigning to slots 18, 28, 38, 48 to prevent unnecessary drift to wisdom tooth positions.

### B. Vertical Object Merging (The "Ghost" Fix)
*   **Implant + Crown:**
    *   Before assignment, we merge vertically aligned `Implant` (Fixture) and `Crown` objects into a single `Implant` entity.
    *   **Relaxed Criteria:** Distance threshold increased to **35px** (from 10-20px) and removed strict X-overlap checks to correctly merge **tilted implants**.
    *   **Status Inheritance:** The merged Implant object retains `has_crown=True` property.
*   **Natural + Crown:**
    *   Similar merging logic for natural teeth capped with crowns.
*   **Orphan Filtering:**
    *   Any Crown object that fails to merge with a root (Implant or Natural) is treated as a **Pontic (Bridge)** or Artifact and is **removed** to prevent it from occupying a tooth slot.

### C. Bug Fix: Double Mapping
*   **Issue:** The legacy code loop for `Crown/Filling/Caries` was re-adding crowns that had already been processed (or filtered out) by the global solver.
*   **Fix:** Explicitly **skipped** `crown`, `bridge`, `pontic` labels in the legacy loop, ensuring only the Global Solver managers crowns.

## 4. UI / Visualization Updates

### A. Odontogram (BottomTeethChart.tsx)
*   **Lower Teeth Orientation:** Fixed rendering order so the **Crown (Yellow)** appears at the **top** segment for lower teeth (previously was bottom).
*   **Color Priority:** Adjusted style logic so `Crown (Yellow)` takes precedence over `Implant (Purple)` for the crown segment only.

### B. Layout Toggle
*   **New Feature:** Added "Layout On/Off" (Overlay Toggle) buttons.
    *   **Location 1:** Left Sidebar (Toolbox).
    *   **Location 2:** Central Control Bar (between Image and Odontogram).

## 5. Modified Files
1.  `backend/services/pano_inference.py`: Core logic changes (Global Assignment, Merging, Filtering).
2.  `frontend/src/components/BottomTeethChart.tsx`: Chart rendering fixes.
3.  `frontend/src/pages/ChartPage.tsx`: Layout toggle button implementation.

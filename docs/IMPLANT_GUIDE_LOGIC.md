# Implant Guide Calculation Logic

This document describes the algorithm used to predict implant placement positions and angles for missing teeth in panoramic X-rays.

## 1. Overview
The goal is to determine the optimal **position (Gap Center)**, **angle (Implant Axis)**, and **length (Safety Distance)** for a potential implant in an edentulous space.

## 2. Gap Center Calculation (X, Y)
The "Gap Center" represents the ideal entry point or top-center of the implant.

### Logic Strategy
We prioritize **Centroid-based** calculation to ensure the gap aligns with the "roots" of adjacent teeth, rather than just their visible crowns or bounding boxes (which can be tilted or irregular).

1.  **Centroid Midpoint (Primary)**:
    *   Extract the `contour` of the adjacent teeth (Previous and Next).
    *   Calculate the **Centroid** (center of mass) of each neighbor.
    *   **Gap X**: Average of the X-coordinates of the two centroids.
    *   **Gap Y**: Average of the Y-coordinates of the two centroids (Linear Interpolation).
    
2.  **Hybrid Fallback (Secondary)**:
    *   If one neighbor lacks a contour (no centroid), we use its **Bounding Box Center** as a proxy.
    *   Gap is calculated as the midpoint between the available Centroid and the proxy Box Center.

3.  **Box Gap (Tertiary Fallback)**:
    *   If no contours are available, we fall back to the empty space between the bounding boxes.
    *   `Gap X = (Box1.Right + Box2.Left) / 2`
    *   `Gap Y` is estimated using the CEJ level or 70% down the box height.

## 3. Implant Axis Determination (Angle)
The angle of the implant should mimic the natural roots of the neighboring teeth.

*   **Principal Component Analysis (PCA)**:
    *   For each neighbor with a contour, we apply PCA to find its **Principal Axis** (the direction of greatest variance, i.e., the root direction).
    *   **Mean Axis**: We average the unit vectors of the Previous and Next tooth axes.
    *   This ensures the implant is parallel to its neighbors, adapting to the curve of the jaw (e.g., tilted molars vs. straight incisors).

*   **Vertical Fallback**:
    *   If no neighbor contours exist, we default to a **Vertical Axis** (0, 1).

## 4. Raycast & Safety Margin
Once the Start Point (Gap Center) and Direction (Axis) are fixed, we determine the available bone height.

1.  **Nerve Detection**:
    *   A ray is cast from the Gap Center downwards along the Axis vector.
    *   It stops when it hits a **Nerve Segmentation Mask** or leaves the image.
    *   This intersection point defines the maximum physical distance.

2.  **Safety Margin (2mm)**:
    *   To ensure clinical safety, we subtract a **2mm buffer** from the measured distance.
    *   `Logic`: `Safe Distance = Raw Distance - (2mm in pixels)`
    *   The visual guide line is retracted by this amount.

## 5. Report Visualization
The calculated coordinates (`line_coords`) are passed directly to the Report Generator to ensure consistency between the analysis and the final PDF/HTML report. We explicitely prevent the report generator from recalculating/overwriting these values with simplified logic.

---
**File Reference**: `backend/services/pano_inference.py` (Functions: `inference_missing_teeth`, `run`)

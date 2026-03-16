# Cornerstone Native Tools Viewer

## Goal

Add a comparison viewer that stays as close as possible to Cornerstone's native stack viewport and tool flow.

This viewer is used to separate:

- file or DICOM parsing problems
- image loader and decode problems
- custom wrapper and layout problems
- StackViewport rendering problems

## Current Finding

For the local panoramic DICOM used in testing:

- local DICOM parsing succeeds
- raw preview rendering succeeds
- `dicomfile:` image creation succeeds
- Cornerstone native canvas probe renders correctly
- the custom `CornerstoneViewer` can still stay black

That means the main issue is not the source file itself. It is in the custom viewer path or in how the custom stack viewport is attached and managed.

## Viewers Added For Comparison

The sandbox page currently exposes four reference paths:

1. `Custom Cornerstone Wrapper`
2. `Minimal Cornerstone DICOM Viewer`
3. `Cornerstone Native Tools Viewer`
4. `Cornerstone Native Canvas Probe`

And one non-Cornerstone baseline:

1. `Original Viewer`

## Native Tools Viewer Design

`CornerstoneNativeToolsViewer` intentionally keeps the structure simple:

- native `RenderingEngine`
- `STACK` viewport
- Cornerstone ToolGroup
- Cornerstone tool instances only
- simple neutral toolbar
- CPU rendering forced for local validation

Supported sources:

- `dicomfile:` for local DICOM
- `web:` for PNG/JPG and similar browser images

## Why This Viewer Exists

The custom viewer mixes several concerns:

- custom header and tabs
- tool state UI
- multi-source switching
- engine lifecycle
- viewport lifecycle
- scheme switching
- CPU/GPU switching

That makes it harder to isolate the real failure point.

The native tools viewer is a control case:

- if this viewer works, Cornerstone stack rendering and tools are basically fine
- if this viewer fails, the problem is lower in the Cornerstone stack path

## Practical Usage

Use the pages like this:

- `/cornerstone_page`: debugging and side-by-side comparison
- `/test`: real upload flow with the same native tools viewer attached to the original file

## Files

- `src/viewer/CornerstoneNativeToolsViewer.tsx`
- `src/viewer/MinimalCornerstoneDicomViewer.tsx`
- `src/viewer/CornerstoneCanvasProbe.tsx`
- `src/pages/CornerstonePage.tsx`
- `src/pages/TestPage.tsx`

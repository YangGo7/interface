import * as cornerstone from '@cornerstonejs/core';
import {
    addTool,
    ToolGroupManager,
    PanTool,
    ZoomTool,
    WindowLevelTool,
    LengthTool,
    ArrowAnnotateTool,
    RectangleROITool,
    EllipticalROITool,
    TrackballRotateTool,
    StackScrollTool,
    EraserTool,
    Enums,
    annotation,
} from '@cornerstonejs/tools';

let toolsAdded = false;

export const TOOL_GROUP_ID = 'MY_TOOLBOX_TOOL_GROUP';
export const TOOL_GROUP_3D_ID = 'MY_TOOLBOX_3D_TOOL_GROUP';

export function addAndGroupTools() {
    if (toolsAdded) return;

    // Add tools globally to Cornerstone3D
    addTool(PanTool);
    addTool(ZoomTool);
    addTool(WindowLevelTool);
    addTool(LengthTool);
    addTool(ArrowAnnotateTool);
    addTool(RectangleROITool);
    addTool(EllipticalROITool);
    addTool(TrackballRotateTool);
    addTool(StackScrollTool);
    addTool(EraserTool);

    toolsAdded = true;
}

export function createOrGetToolGroup() {
    let toolGroup = ToolGroupManager.getToolGroup(TOOL_GROUP_ID);

    if (!toolGroup) {
        toolGroup = ToolGroupManager.createToolGroup(TOOL_GROUP_ID);
        if (toolGroup) {
            toolGroup.addTool(PanTool.toolName);
            toolGroup.addTool(ZoomTool.toolName);
            toolGroup.addTool(WindowLevelTool.toolName);
            toolGroup.addTool(LengthTool.toolName);
            toolGroup.addTool(ArrowAnnotateTool.toolName);
            toolGroup.addTool(RectangleROITool.toolName);
            toolGroup.addTool(EllipticalROITool.toolName);
            toolGroup.addTool(StackScrollTool.toolName);
            toolGroup.addTool(EraserTool.toolName);
            toolGroup.addTool(TrackballRotateTool.toolName);

            // Default bindings
            toolGroup.setToolActive(PanTool.toolName, {
                bindings: [{ mouseButton: Enums.MouseBindings.Primary }],
            });
            toolGroup.setToolActive(ZoomTool.toolName, {
                bindings: [{ mouseButton: Enums.MouseBindings.Wheel }],
            });
            toolGroup.setToolActive(WindowLevelTool.toolName, {
                bindings: [{ mouseButton: Enums.MouseBindings.Auxiliary }],
            });
        }
    }

    return toolGroup;
}

export function createOrGet3DToolGroup() {
    let toolGroup = ToolGroupManager.getToolGroup(TOOL_GROUP_3D_ID);

    if (!toolGroup) {
        toolGroup = ToolGroupManager.createToolGroup(TOOL_GROUP_3D_ID);
        if (toolGroup) {
            toolGroup.addTool(TrackballRotateTool.toolName);
            toolGroup.addTool(ZoomTool.toolName);
            toolGroup.addTool(PanTool.toolName);
            toolGroup.addTool(StackScrollTool.toolName);
            toolGroup.addTool(EraserTool.toolName);

            toolGroup.setToolActive(TrackballRotateTool.toolName, {
                bindings: [{ mouseButton: Enums.MouseBindings.Primary }],
            });
            toolGroup.setToolActive(ZoomTool.toolName, {
                bindings: [{ mouseButton: Enums.MouseBindings.Secondary }],
            });
            toolGroup.setToolActive(PanTool.toolName, {
                bindings: [{ mouseButton: Enums.MouseBindings.Auxiliary }],
            });
            toolGroup.setToolActive(StackScrollTool.toolName, {
                bindings: [{ mouseButton: Enums.MouseBindings.Wheel }],
            });
        }
    }

    return toolGroup;
}

export function setActiveTool(toolName: string) {
    const toolGroup = ToolGroupManager.getToolGroup(TOOL_GROUP_ID);
    const toolGroup3D = ToolGroupManager.getToolGroup(TOOL_GROUP_3D_ID);

    const groups = [toolGroup, toolGroup3D].filter(g => !!g);

    const toolsToReset = [
        PanTool.toolName,
        ZoomTool.toolName,
        WindowLevelTool.toolName,
        LengthTool.toolName,
        ArrowAnnotateTool.toolName,
        RectangleROITool.toolName,
        EllipticalROITool.toolName,
        TrackballRotateTool.toolName,
        EraserTool.toolName,
        StackScrollTool.toolName
    ];

    groups.forEach(group => {
        toolsToReset.forEach((tool) => {
            if (group.hasTool(tool)) {
                group.setToolPassive(tool);
            }
        });

        if (toolName === PanTool.toolName && group.hasTool(PanTool.toolName)) {
            group.setToolActive(PanTool.toolName, {
                bindings: [{ mouseButton: Enums.MouseBindings.Primary }],
            });
            if (group.hasTool(ZoomTool.toolName)) {
                group.setToolActive(ZoomTool.toolName, {
                    bindings: [{ mouseButton: Enums.MouseBindings.Wheel }],
                });
            }
            return;
        }

        group.setToolActive(toolName, {
            bindings: [{ mouseButton: Enums.MouseBindings.Primary }],
        });
    });
}

/**
 * Initialize global event listeners for tool interactions.
 */
export function initToolEventListeners() {
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Delete' || event.key === 'Backspace') {
            const annotationUIDs = annotation.selection.getAnnotationsSelected();
            if (annotationUIDs && annotationUIDs.length > 0) {
                annotationUIDs.forEach(uid => {
                    annotation.state.removeAnnotation(uid);
                });
                const engines = cornerstone.getRenderingEngines() || [];
                engines.forEach(engine => {
                    if (engine) engine.render();
                });
            }
        }
    });
}

/**
 * Remove all annotations/drawings from the viewer.
 */
export function clearAllAnnotations() {
    annotation.state.removeAllAnnotations();
    const engines = cornerstone.getRenderingEngines() || [];
    engines.forEach(engine => {
        if (engine) engine.render();
    });
}

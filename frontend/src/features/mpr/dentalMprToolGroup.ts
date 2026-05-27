import { BlendModes } from '@cornerstonejs/core/enums';
import {
  getRenderingEngine,
  type VolumeViewport,
} from '@cornerstonejs/core';
import { RENDERING_DEFAULTS } from '@cornerstonejs/core/constants';
import {
  createMprSlabPlane,
  type MprProjectionMode,
  type MprRenderingType,
  type MprSlabPlane,
} from './mpr3dControls';
import {
  addTool,
  CrosshairsTool,
  Enums as cornerstoneToolEnums,
  PanTool,
  StackScrollTool,
  ToolGroupManager,
  WindowLevelTool,
  ZoomTool,
  annotation,
  utilities as cornerstoneToolUtilities,
} from '@cornerstonejs/tools';
import type { Mpr2DTool } from './mpr3dControls';

export const DENTAL_MPR_TOOL_GROUP_ID = 'interface-dental-mpr-crosshairs-group';
const DENTAL_MPR_CROSSHAIR_COLOR = '#00d8d8';
const SLAB_MIN = RENDERING_DEFAULTS.MINIMUM_SLAB_THICKNESS;

type ToolGroup = NonNullable<ReturnType<typeof ToolGroupManager.getToolGroup>>;
type SlabSyncOptions = {
  getSlabPlane: (viewportId?: string) => MprSlabPlane;
  onSlabThicknessChange: (viewportId: string, thicknessMm: number) => void;
};

let stylePatched = false;
let dentalMprToolsRegistered = false;
let slabSyncOptions: SlabSyncOptions = {
  getSlabPlane: () => createMprSlabPlane('mpr', 0),
  onSlabThicknessChange: () => undefined,
};

function registerDentalMprTools() {
  if (dentalMprToolsRegistered) return;
  [CrosshairsTool, StackScrollTool, PanTool, ZoomTool, WindowLevelTool].forEach((ToolClass) => {
    addTool(ToolClass);
  });
  dentalMprToolsRegistered = true;
}

function patchCrosshairStyle() {
  if (stylePatched) return;
  const defaultStyles = annotation.config.style.getDefaultToolStyles();
  annotation.config.style.setDefaultToolStyles({
    ...defaultStyles,
    global: {
      ...defaultStyles.global,
      colorHighlightedActive: DENTAL_MPR_CROSSHAIR_COLOR,
      colorHighlighted: DENTAL_MPR_CROSSHAIR_COLOR,
    },
  });
  stylePatched = true;
}

function isMprViewportId(viewportId: string) {
  return viewportId === 'P0' || viewportId === 'P1' || viewportId === 'P2';
}

function crosshairsBlendForRenderingType(renderingType: MprRenderingType) {
  if (renderingType === 'MIP') return BlendModes.MAXIMUM_INTENSITY_BLEND;
  if (renderingType === 'minIP') return BlendModes.MINIMUM_INTENSITY_BLEND;
  return BlendModes.AVERAGE_INTENSITY_BLEND;
}

export function resolveDentalMprSlabBlendMode(slabPlane: MprSlabPlane) {
  if (slabPlane.renderingType === 'VR' || Math.abs(slabPlane.fe - slabPlane.fs) <= SLAB_MIN + 1e-4) {
    return BlendModes.COMPOSITE;
  }
  return crosshairsBlendForRenderingType(slabPlane.renderingType);
}

export function resolveDentalMprBlendMode(mode: MprProjectionMode, slabThicknessMm: number) {
  return resolveDentalMprSlabBlendMode(createMprSlabPlane(mode, slabThicknessMm));
}

export function configureDentalMprSlabSync(options: SlabSyncOptions) {
  slabSyncOptions = options;
}

export function applyDentalMprSlabBlends(renderingEngineId?: string) {
  const toolGroup = ToolGroupManager.getToolGroup(DENTAL_MPR_TOOL_GROUP_ID);
  if (!toolGroup) return;

  const infos = (toolGroup.getViewportsInfo?.() ?? toolGroup.viewportsInfo ?? []) as {
    viewportId: string;
    renderingEngineId?: string;
  }[];

  infos.forEach((info) => {
    if (!isMprViewportId(info.viewportId)) return;
    if (renderingEngineId && info.renderingEngineId !== renderingEngineId) return;

    const engine = getRenderingEngine(info.renderingEngineId ?? renderingEngineId ?? '');
    const viewport = engine?.getViewport(info.viewportId) as VolumeViewport | undefined;
    if (!viewport || typeof viewport.getSlabThickness !== 'function' || typeof viewport.setBlendMode !== 'function') {
      return;
    }

    const slabThickness = viewport.getSlabThickness();
    const currentSlabPlane = slabSyncOptions.getSlabPlane(info.viewportId);
    viewport.setBlendMode(
      resolveDentalMprSlabBlendMode({
        fs: -slabThickness / 2,
        fe: slabThickness / 2,
        thicknessMm: slabThickness,
        renderingType: currentSlabPlane.renderingType,
        renderPath: slabThickness > 0 ? 'svp' : 'single-plane',
      }),
      [],
      false
    );
    viewport.render();
  });
}

function installCrosshairsSlabSyncWrapper(toolGroup: ToolGroup) {
  const crosshairs = toolGroup.getToolInstance(CrosshairsTool.toolName) as
    | { setSlabThickness: (viewport: VolumeViewport, slabThickness: number) => void }
    | undefined;
  if (!crosshairs || typeof crosshairs.setSlabThickness !== 'function') return;

  const patchKey = '__interfaceDentalMprSlabSyncPatched';
  if ((crosshairs as unknown as Record<string, boolean>)[patchKey]) return;
  (crosshairs as unknown as Record<string, boolean>)[patchKey] = true;

  const originalSetSlabThickness = crosshairs.setSlabThickness.bind(crosshairs);
  crosshairs.setSlabThickness = (viewport: VolumeViewport, slabThickness: number) => {
    const viewportId = viewport.id;
    const slabPlane = slabSyncOptions.getSlabPlane(viewportId);
    toolGroup.setToolConfiguration(
      CrosshairsTool.toolName,
      { slabThicknessBlendMode: crosshairsBlendForRenderingType(slabPlane.renderingType) },
      false
    );

    originalSetSlabThickness(viewport, slabThickness);

    const nextSlabThickness =
      typeof viewport.getSlabThickness === 'function' ? viewport.getSlabThickness() : slabThickness;
    if (Number.isFinite(nextSlabThickness)) {
      slabSyncOptions.onSlabThicknessChange(viewportId, Number(nextSlabThickness.toFixed(2)));
    }
    applyDentalMprSlabBlends();
  };
}

export function createOrGetDentalMprToolGroup(): ToolGroup {
  registerDentalMprTools();
  patchCrosshairStyle();

  let toolGroup = ToolGroupManager.getToolGroup(DENTAL_MPR_TOOL_GROUP_ID);
  if (!toolGroup) {
    toolGroup = ToolGroupManager.createToolGroup(DENTAL_MPR_TOOL_GROUP_ID);
    if (!toolGroup) {
      throw new Error('Failed to create Dental MPR tool group.');
    }
 // renderAnnotation 의 경우 도형 변경 어려움 
    toolGroup.addTool(CrosshairsTool.toolName, {
      getReferenceLineColor: () => DENTAL_MPR_CROSSHAIR_COLOR,
      getReferenceLineControllable: () => true,
      getReferenceLineDraggableRotatable: () => true,
      getReferenceLineSlabThicknessControlsOn: () => true,
      slabThicknessBlendMode: BlendModes.AVERAGE_INTENSITY_BLEND,
      handleRadius: 4,
      centerPoint: {
        enabled: true,
        color: DENTAL_MPR_CROSSHAIR_COLOR,
        size: 2.25,
      },
    });
    toolGroup.addTool(StackScrollTool.toolName);
    toolGroup.addTool(PanTool.toolName);
    toolGroup.addTool(ZoomTool.toolName);
    toolGroup.addTool(WindowLevelTool.toolName);
  }

  installCrosshairsSlabSyncWrapper(toolGroup);
  return toolGroup;
}

export function activateDentalMprCrosshairs() {
  activateDentalMpr2DTool('crosshair');
}

export function pauseDentalMprCrosshairsForViewportAttach() {
  const toolGroup = createOrGetDentalMprToolGroup();
  if (!toolGroup.hasTool(CrosshairsTool.toolName)) return;
  try {
    toolGroup.setToolPassive(CrosshairsTool.toolName, { removeAllBindings: true });
  } catch {
    // Tool state is best-effort while Cornerstone viewports are being rebuilt.
  }
}

export function refreshDentalMprCrosshairs(renderingEngineId?: string, options: { reset?: boolean } = {}) {
  const toolGroup = createOrGetDentalMprToolGroup();
  const infos = (toolGroup.getViewportsInfo?.() ?? toolGroup.viewportsInfo ?? []) as {
    viewportId: string;
    renderingEngineId?: string;
  }[];
  const mprViewportInfos = infos.filter((info) => {
    if (!isMprViewportId(info.viewportId)) return false;
    if (renderingEngineId && info.renderingEngineId !== renderingEngineId) return false;
    const engine = getRenderingEngine(info.renderingEngineId ?? renderingEngineId ?? '');
    return Boolean(engine?.getViewport(info.viewportId));
  });
  if (mprViewportInfos.length < 2) return;

  const crosshairs = toolGroup.getToolInstance(CrosshairsTool.toolName) as
    | { computeToolCenter?: () => void; resetCrosshairs?: () => void }
    | undefined;
  if (options.reset) {
    crosshairs?.resetCrosshairs?.();
  } else {
    crosshairs?.computeToolCenter?.();
  }

  const viewportIdsToRender = mprViewportInfos.map((info) => info.viewportId);
  mprViewportInfos.forEach((info) => {
    const engine = getRenderingEngine(info.renderingEngineId ?? renderingEngineId ?? '');
    engine?.getViewport(info.viewportId)?.render();
  });

  if (viewportIdsToRender.length > 0) {
    cornerstoneToolUtilities.triggerAnnotationRenderForViewportIds(viewportIdsToRender);
  }
}

export function activateDentalMpr2DTool(tool: Mpr2DTool) {
  const toolGroup = createOrGetDentalMprToolGroup();
  [
    CrosshairsTool.toolName,
    StackScrollTool.toolName,
    PanTool.toolName,
    ZoomTool.toolName,
    WindowLevelTool.toolName,
  ].forEach((toolName) => {
    if (toolGroup.hasTool(toolName)) {
      toolGroup.setToolPassive(toolName, { removeAllBindings: true });
    }
  });

  if (tool === 'pan') {
    toolGroup.setToolActive(PanTool.toolName, {
      bindings: [{ mouseButton: cornerstoneToolEnums.MouseBindings.Primary }],
    });
    toolGroup.setToolActive(ZoomTool.toolName, {
      bindings: [{ mouseButton: cornerstoneToolEnums.MouseBindings.Wheel }],
    });
    return;
  }

  if (tool === 'zoom') {
    toolGroup.setToolActive(ZoomTool.toolName, {
      bindings: [
        { mouseButton: cornerstoneToolEnums.MouseBindings.Primary },
        { mouseButton: cornerstoneToolEnums.MouseBindings.Wheel },
      ],
    });
    return;
  }

  if (tool === 'wlww') {
    toolGroup.setToolPassive(CrosshairsTool.toolName, { removeAllBindings: true });
    toolGroup.setToolActive(StackScrollTool.toolName, {
      bindings: [{ mouseButton: cornerstoneToolEnums.MouseBindings.Wheel }],
    });
    toolGroup.setToolActive(WindowLevelTool.toolName, {
      bindings: [{ mouseButton: cornerstoneToolEnums.MouseBindings.Primary }],
    });
    return;
  }

  toolGroup.setToolActive(CrosshairsTool.toolName, {
    bindings: [{ mouseButton: cornerstoneToolEnums.MouseBindings.Primary }],
  });
  toolGroup.setToolActive(StackScrollTool.toolName, {
    bindings: [{ mouseButton: cornerstoneToolEnums.MouseBindings.Wheel }],
  });
}

export function removeDentalMprViewports(renderingEngineId: string) {
  const toolGroup = ToolGroupManager.getToolGroup(DENTAL_MPR_TOOL_GROUP_ID);
  if (!toolGroup) return;
  try {
    toolGroup.removeViewports(renderingEngineId);
  } catch {
    // The rendering engine may already be destroyed.
  }
}

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react';
import * as cornerstone from '@cornerstonejs/core';
import * as cornerstoneTools from '@cornerstonejs/tools';
import { BlendModes } from '@cornerstonejs/core/enums';
import {
  PanTool,
  TrackballRotateTool,
  ZoomTool,
} from '@cornerstonejs/tools';
import { initCornerstone } from '../../viewer/cornerstone/init';
import {
  addAndGroupTools,
  createOrGet3DToolGroup,
} from '../../viewer/cornerstone/tools';
import {
  activateDentalMpr2DTool,
  configureDentalMprSlabSync,
  createOrGetDentalMprToolGroup,
  pauseDentalMprCrosshairsForViewportAttach,
  refreshDentalMprCrosshairs,
  removeDentalMprViewports,
  resolveDentalMprSlabBlendMode,
} from './dentalMprToolGroup';
import type { ViewerSource } from '../../viewer/CornerstoneViewer';
import {
  createMprSlabPlane,
  DEFAULT_MPR_SLAB_THICKNESS_BY_AXIS,
  formatMprFilterSelection,
  hasMprFilterSelection,
  MPR_DEFAULT_SLAB_THICKNESS_MM,
  MPR_FILTER_OPTIONS,
  normalizeMprFilterSelection,
  normalizeMprSlabThicknessByAxis,
  type Mpr3DPreset,
  type Mpr2DTool,
  type Mpr3DControlState,
  type MprFilterSelection,
  type MprFilterType,
  type MprProjectionMode,
  type MprSlabAxis,
  type MprSlabPlane,
  type MprSlabThicknessByAxis,
  type MprViewportControlState,
  type MprWindowState,
} from './mpr3dControls';
import './DentalMprWorkspace.css';

type MprPanelId = 'P0' | 'P1' | 'P2' | 'P3';
type MprPlane = 'coronal' | 'sagittal' | 'axial' | 'volume3d';

export type DentalMprSeriesInfo = {
  seriesInstanceUID: string;
  patientID?: string;
  patientName?: string;
  patientSex?: string;
  patientAge?: string;
  studyDate?: string;
  seriesDescription?: string;
  seriesNumber?: number | null;
  gantryTilt?: number | null;
  tubeCurrent?: number | null;
  tubeVoltage?: number | null;
};

type MprPanelConfig = {
  id: MprPanelId;
  label: string;
  plane: MprPlane;
  orientation?: cornerstone.Enums.OrientationAxis;
};

type SliceInfo = {
  index: number;
  count: number;
};

type ViewportOverlayInfo = SliceInfo & {
  windowLevel: number | null;
  windowWidth: number | null;
  zoom: number | null;
  pixelSizeMm: number | null;
};

type MprHuOverlayInfo = {
  value: number;
  x: number;
  y: number;
};

type MprVoiRange = {
  lower: number;
  upper: number;
};

type NumericScalarArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export type VtkRenderParameters = {
  mn: number;
  mx: number;
  t0: number;
  t1: number;
  t2: number;
};

type VtkRenderParameterSource = {
  scalarData: NumericScalarArray;
  slope?: number | null;
  intercept?: number | null;
};

type DentalMprWorkspaceProps = {
  source: ViewerSource;
  seriesInfo: DentalMprSeriesInfo;
  height: number;
  viewportControlState?: MprViewportControlState;
  active2DTool?: Mpr2DTool;
  showHuOverlay?: boolean;
  control3DState?: Mpr3DControlState;
  vtkRenderParameters?: VtkRenderParameters | null;
  onViewportControlStateChange?: (state: MprViewportControlState) => void;
};

const PANELS: MprPanelConfig[] = [
  { id: 'P0', label: 'Coronal', plane: 'coronal', orientation: cornerstone.Enums.OrientationAxis.CORONAL },
  { id: 'P1', label: 'Sagittal', plane: 'sagittal', orientation: cornerstone.Enums.OrientationAxis.SAGITTAL },
  { id: 'P2', label: 'Axial', plane: 'axial', orientation: cornerstone.Enums.OrientationAxis.AXIAL },
  { id: 'P3', label: '3D', plane: 'volume3d', orientation: cornerstone.Enums.OrientationAxis.CORONAL },
];

const MPR_PANEL_SLAB_AXIS: Partial<Record<MprPanelId, MprSlabAxis>> = {
  P0: 'y',
  P1: 'x',
  P2: 'z',
};

const VOLUME_PRESETS: Record<string, any> = {
  'dental-bone': {
    name: 'Dental Bone',
    gradientOpacity: '4 0 1 255 1',
    specularPower: '10',
    scalarOpacity: '10 -3024 0 150 0 320 0.05 641.385 0.72 3071 0.74',
    specular: '0.2',
    shade: '1',
    ambient: '0.1',
    colorTransfer: '20 -3024 0 0 0 150 0 0 0 320 0.62 0.54 0.35 641.385 0.91 0.82 0.56 3071 1 1 1',
    diffuse: '0.9',
    interpolation: '1',
  },
  'dental-surface': {
    name: 'Dental Surface',
    gradientOpacity: '4 0 1 255 1',
    specularPower: '1',
    scalarOpacity: '10 -2048 0 120 0 260 0.08 780 1 3661 1',
    specular: '0',
    shade: '0',
    ambient: '0.2',
    colorTransfer: '20 -2048 0 0 0 120 0 0 0 260 0.18 0.18 0.18 780 1 1 1 3661 1 1 1',
    diffuse: '1',
    interpolation: '1',
  },
  'dental-soft-tissue': {
    name: 'Dental Soft Tissue',
    gradientOpacity: '4 0 1 255 1',
    specularPower: '1',
    scalarOpacity: '10 -2048 0 80 0 140 0.05 260 0.75 3661 0.9',
    specular: '0',
    shade: '0',
    ambient: '0.2',
    colorTransfer: '20 -2048 0 0 0 80 0 0 0 140 0.44 0.34 0.33 260 0.86 0.71 0.66 3661 1 0.92 0.88',
    diffuse: '1',
    interpolation: '1',
  },
  'dental-mip': {
    name: 'Dental MIP',
    gradientOpacity: '4 0 1 255 1',
    specularPower: '10',
    scalarOpacity: '8 -3024 0 180 0 480 1 3071 1',
    specular: '0.2',
    shade: '1',
    ambient: '0.1',
    colorTransfer: '16 -3024 0 0 0 180 0 0 0 480 1 1 1 3071 1 1 1',
    diffuse: '0.9',
    interpolation: '1',
  },
  'vtk-mip': {
    name: 'VTK MIP',
    gradientOpacity: '4 0 1 255 1',
    specularPower: '10',
    scalarOpacity: '4 0 0 2048 1',
    specular: '0.2',
    shade: '1',
    ambient: '0.1',
    colorTransfer: '8 0 1 1 1 255 1 1 1',
    diffuse: '0.9',
    interpolation: '1',
  },
};

const resolveProjectionLabel = (mode: MprProjectionMode) =>
  mode === 'minip' ? 'minIP' : mode === 'vr' ? 'VR' : mode.toUpperCase();

const PROJECTION_OPTIONS: { value: MprProjectionMode; label: string }[] = [
  { value: 'mpr', label: 'MPR' },
  { value: 'mip', label: 'MIP' },
  { value: 'minip', label: 'minIP' },
  { value: 'vr', label: 'VR' },
];

const DEFAULT_3D_CONTROL_STATE: Mpr3DControlState = {
  tool: 'rotate',
  preset: 'dental-surface',
  resetToken: 0,
};

const MPR_3D_STILL_IMAGE_SAMPLE_DISTANCE = 1.25;
const MPR_3D_STILL_MAXIMUM_SAMPLES_PER_RAY = 4096;
const MPR_3D_INTERACTIVE_SAMPLE_DISTANCE_MULTIPLIER = 2.5;
const MPR_3D_INTERACTIVE_IMAGE_SAMPLE_DISTANCE = 1.75;
const MPR_3D_INTERACTIVE_MAXIMUM_SAMPLES_PER_RAY = 2048;
const MPR_3D_QUALITY_RESTORE_DELAY_MS = 180;
const MPR_3D_MIN_PARALLEL_SCALE = 1;
const MPR_3D_MAX_PARALLEL_SCALE = 10000;
const VTK_RENDER_THRESHOLD_BINS = 256;
const VTK_RENDER_MAX_THRESHOLD_SAMPLES = 262144;
const VTK_RENDER_SOFT_TISSUE_WIDTH = 80;
const VTK_RENDER_DEFAULT_PARAMETERS: VtkRenderParameters = {
  mn: -1024,
  mx: 3071,
  t0: 120,
  t1: 320,
  t2: 641,
};
const MPR_3D_CORONAL_CAMERA = {
  viewPlaneNormal: [0, -1, 0],
  viewUp: [0, 0, 1],
};

const vtkRenderParameterCache = new WeakMap<object, VtkRenderParameters | null>();
const vtkFallbackRenderParameterCache = new Map<string, VtkRenderParameters | null>();

type MprFilterViewportProperties = {
  interpolationType: cornerstone.Enums.InterpolationType;
  smoothing: number;
  sharpening: number;
};

type MprFilterKernel = {
  filterType: string;
  implementation: string;
  divisor: number;
  kernel: readonly [number, number, number, number, number, number, number, number, number];
};

type ResolvedMprFilter = MprFilterViewportProperties & {
  kernel?: MprFilterKernel;
};

const MPR_SHARPEN_KERNELS: Record<'1x' | '1.5x' | '2x', MprFilterKernel> = {
  '1x': {
    filterType: 'FILTER_TYPE_IMAGE_ENHANCEMENT',
    implementation: 'ImageEnhancementFilter.cpp:17',
    divisor: 3,
    kernel: [-1, 0, -1, 0, 7, 0, -1, 0, -1],
  },
  '1.5x': {
    filterType: 'FILTER_TYPE_MILD_SHARPEN',
    implementation: 'MildSharpenFilter.cpp:18',
    divisor: 2,
    kernel: [-1, 0, -1, 0, 6, 0, -1, 0, -1],
  },
  '2x': {
    filterType: 'FILTER_TYPE_SHARPEN',
    implementation: 'SharpenFilter.cpp:18',
    divisor: 1,
    kernel: [-1, 0, -1, 0, 5, 0, -1, 0, -1],
  },
};

function resolveLegacyFilterType(filterType: MprFilterType | '1' | '1.5' | '2'): MprFilterType {
  if (filterType === '1') return '1x';
  if (filterType === '1.5') return '1.5x';
  if (filterType === '2') return '2x';
  return filterType;
}

function applyMprImageEnhancementFilter(): ResolvedMprFilter {
  return {
    interpolationType: cornerstone.Enums.InterpolationType.LINEAR,
    smoothing: 0,
    sharpening: 0.45,
    kernel: MPR_SHARPEN_KERNELS['1x'],
  };
}

function applyMprMildSharpenFilter(): ResolvedMprFilter {
  return {
    interpolationType: cornerstone.Enums.InterpolationType.LINEAR,
    smoothing: 0,
    sharpening: 0.7,
    kernel: MPR_SHARPEN_KERNELS['1.5x'],
  };
}

function applyMprSharpenKernelFilter(): ResolvedMprFilter {
  return {
    interpolationType: cornerstone.Enums.InterpolationType.LINEAR,
    smoothing: 0,
    sharpening: 1,
    kernel: MPR_SHARPEN_KERNELS['2x'],
  };
}

function applyMprBlurFilter(): MprFilterViewportProperties {
  return {
    interpolationType: cornerstone.Enums.InterpolationType.LINEAR,
    smoothing: 0.6,
    sharpening: 0,
  };
}

function applyMprSharpenFilter(): MprFilterViewportProperties {
  return {
    interpolationType: cornerstone.Enums.InterpolationType.LINEAR,
    smoothing: 0,
    sharpening: 0.8,
  };
}

function applyMprGaussianFilter(): MprFilterViewportProperties {
  return {
    interpolationType: cornerstone.Enums.InterpolationType.LINEAR,
    smoothing: 0.9,
    sharpening: 0,
  };
}

function applyMprAverageFilter(): MprFilterViewportProperties {
  return {
    interpolationType: cornerstone.Enums.InterpolationType.LINEAR,
    smoothing: 0.35,
    sharpening: 0,
  };
}

function applyMprMedianFilter(): MprFilterViewportProperties {
  return {
    interpolationType: cornerstone.Enums.InterpolationType.LINEAR,
    smoothing: 0.45,
    sharpening: 0.2,
  };
}

const MPR_FILTER_APPLIERS: Record<MprFilterType, () => ResolvedMprFilter> = {
  '1x': applyMprImageEnhancementFilter,
  '1.5x': applyMprMildSharpenFilter,
  '2x': applyMprSharpenKernelFilter,
  Blur: applyMprBlurFilter,
  Sharpen: applyMprSharpenFilter,
  Gaussian: applyMprGaussianFilter,
  Average: applyMprAverageFilter,
  Median: applyMprMedianFilter,
};

function resolveSingleMprFilterProperties(filterType: MprFilterType): ResolvedMprFilter {
  return MPR_FILTER_APPLIERS[resolveLegacyFilterType(filterType)]?.() ?? applyMprSharpenFilter();
}

function resolveMprFilterProperties(
  filterEnabled: boolean,
  filterType: MprFilterType | MprFilterSelection | null
): ResolvedMprFilter {
  const selection = normalizeMprFilterSelection(filterType);
  const selectedFilters = [selection.first, selection.second].filter(Boolean) as MprFilterType[];
  if (!filterEnabled || selectedFilters.length === 0) {
    return {
      interpolationType: cornerstone.Enums.InterpolationType.LINEAR,
      smoothing: 0,
      sharpening: 0,
    };
  }

  return selectedFilters.reduce<ResolvedMprFilter>(
    (acc, filter) => {
      const next = resolveSingleMprFilterProperties(filter);
      return {
        interpolationType:
          next.interpolationType === cornerstone.Enums.InterpolationType.LINEAR
            ? cornerstone.Enums.InterpolationType.LINEAR
            : acc.interpolationType,
        smoothing: Math.min(1, acc.smoothing + next.smoothing),
        sharpening: Math.min(1.5, acc.sharpening + next.sharpening),
        kernel: next.kernel ?? acc.kernel,
      };
    },
    {
      interpolationType: cornerstone.Enums.InterpolationType.LINEAR,
      smoothing: 0,
      sharpening: 0,
    }
  );
}

function applySlabPlaneToViewport(viewport: any, slabPlane: MprSlabPlane) {
  if (slabPlane.renderPath !== 'svp') {
    viewport.resetSlabThickness?.();
  } else {
    viewport.setSlabThickness?.(Math.max(Math.abs(slabPlane.fe - slabPlane.fs), 0.05), []);
  }
  viewport.setBlendMode?.(resolveDentalMprSlabBlendMode(slabPlane), [], false);
}

function apply3DMapperBlendMode(viewport: any, blendMode: BlendModes) {
  const actors = typeof viewport?.getActors === 'function' ? viewport.getActors() : [];
  actors.forEach((entry: any) => {
    const mapper = entry?.actor?.getMapper?.();
    if (typeof mapper?.setBlendMode === 'function') {
      mapper.setBlendMode(blendMode);
    }
    entry.blendMode = blendMode;
  });
}

function isNumericScalarArray(value: unknown): value is NumericScalarArray {
  return (
    value instanceof Int8Array ||
    value instanceof Uint8Array ||
    value instanceof Uint8ClampedArray ||
    value instanceof Int16Array ||
    value instanceof Uint16Array ||
    value instanceof Int32Array ||
    value instanceof Uint32Array ||
    value instanceof Float32Array ||
    value instanceof Float64Array
  );
}

function safelyReadScalarData(read: () => unknown): NumericScalarArray | null {
  try {
    const value = read();
    return isNumericScalarArray(value) ? value : null;
  } catch {
    return null;
  }
}

function getVolumeScalarData(volume: any): NumericScalarArray | null {
  const candidates = [
    safelyReadScalarData(() => volume?.voxelManager?.getScalarData?.()),
    safelyReadScalarData(() => volume?.getScalarData?.()),
    safelyReadScalarData(() => volume?.scalarData),
    safelyReadScalarData(() => volume?.imageData?.getPointData?.().getScalars?.().getData?.()),
    safelyReadScalarData(() => volume?.imageData?.getPointData?.().getScalars?.().getData?.().getData?.()),
  ];
  return candidates.find(isNumericScalarArray) ?? null;
}

function getVolumeRenderParameterSource(volume: any): VtkRenderParameterSource | null {
  const scalarData = getVolumeScalarData(volume);
  return scalarData ? { scalarData } : null;
}

function getImageRenderParameterSource(image: any): VtkRenderParameterSource | null {
  const scalarData = safelyReadScalarData(() => image?.getPixelData?.());
  if (!scalarData) return null;
  return {
    scalarData,
    slope: Number.isFinite(Number(image?.slope)) ? Number(image.slope) : 1,
    intercept: Number.isFinite(Number(image?.intercept)) ? Number(image.intercept) : 0,
  };
}

function formatTransferNumber(value: number) {
  if (!Number.isFinite(value)) return '0';
  const fixed = Number(value.toFixed(6));
  return Object.is(fixed, -0) ? '0' : String(fixed);
}

function formatColorTransfer(points: Array<[number, number, number, number]>) {
  return `${points.length * 4} ${points.flat().map(formatTransferNumber).join(' ')}`;
}

function formatScalarOpacity(points: Array<[number, number]>) {
  return `${points.length * 2} ${points.flat().map(formatTransferNumber).join(' ')}`;
}

function calculateMultiOtsuThresholds(values: number[], binCount = VTK_RENDER_THRESHOLD_BINS): [number, number, number] | null {
  if (values.length < 16) return null;
  let min = Infinity;
  let max = -Infinity;
  values.forEach((value) => {
    if (!Number.isFinite(value)) return;
    min = Math.min(min, value);
    max = Math.max(max, value);
  });
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null;

  const histogram = new Array(binCount).fill(0);
  const scale = (binCount - 1) / (max - min);
  values.forEach((value) => {
    if (!Number.isFinite(value)) return;
    const index = Math.max(0, Math.min(binCount - 1, Math.floor((value - min) * scale)));
    histogram[index] += 1;
  });

  const total = histogram.reduce((sum, count) => sum + count, 0);
  if (total <= 0) return null;

  const probability = histogram.map((count) => count / total);
  const cumulativeWeight = new Array(binCount).fill(0);
  const cumulativeMean = new Array(binCount).fill(0);
  probability.forEach((weight, index) => {
    cumulativeWeight[index] = weight + (cumulativeWeight[index - 1] ?? 0);
    cumulativeMean[index] = weight * index + (cumulativeMean[index - 1] ?? 0);
  });

  const classScore = (start: number, end: number) => {
    if (end < start) return 0;
    const weight = cumulativeWeight[end] - (cumulativeWeight[start - 1] ?? 0);
    if (weight <= 0) return 0;
    const mean = (cumulativeMean[end] - (cumulativeMean[start - 1] ?? 0)) / weight;
    return weight * mean * mean;
  };

  let bestScore = -Infinity;
  let best: [number, number, number] = [1, 2, 3];
  for (let t0 = 1; t0 < binCount - 3; t0 += 1) {
    for (let t1 = t0 + 1; t1 < binCount - 2; t1 += 1) {
      for (let t2 = t1 + 1; t2 < binCount - 1; t2 += 1) {
        const score =
          classScore(0, t0) +
          classScore(t0 + 1, t1) +
          classScore(t1 + 1, t2) +
          classScore(t2 + 1, binCount - 1);
        if (score > bestScore) {
          bestScore = score;
          best = [t0, t1, t2];
        }
      }
    }
  }

  const binWidth = (max - min) / (binCount - 1);
  return best.map((index) => min + index * binWidth) as [number, number, number];
}

function estimateVtkRenderParameters(source: VtkRenderParameterSource | null): VtkRenderParameters | null {
  const scalarData = source?.scalarData;
  if (!scalarData || scalarData.length < 16) {
    return null;
  }

  const slope = Number.isFinite(Number(source?.slope)) ? Number(source?.slope) : 1;
  const intercept = Number.isFinite(Number(source?.intercept)) ? Number(source?.intercept) : 0;

  const stride = Math.max(1, Math.floor(scalarData.length / VTK_RENDER_MAX_THRESHOLD_SAMPLES));
  const sampled: number[] = [];
  let mn = Infinity;
  let mx = -Infinity;
  for (let index = 0; index < scalarData.length; index += stride) {
    const value = Number(scalarData[index]) * slope + intercept;
    if (!Number.isFinite(value)) continue;
    sampled.push(value);
    mn = Math.min(mn, value);
    mx = Math.max(mx, value);
  }

  if (!Number.isFinite(mn) || !Number.isFinite(mx) || mx <= mn || sampled.length < 16) {
    return null;
  }

  const lowCutoff = mn + 0.005 * (mx - mn);
  const thresholdInput = sampled.map((value) => (value < lowCutoff ? 0 : value));
  const thresholds = calculateMultiOtsuThresholds(thresholdInput);
  if (!thresholds) {
    return null;
  }

  return {
    mn,
    mx,
    t0: thresholds[0],
    t1: thresholds[1],
    t2: thresholds[2],
  };
}

function getVtkRenderParametersFromVolume(volume: any): VtkRenderParameters | null {
  if (!volume || typeof volume !== 'object') return null;
  if (vtkRenderParameterCache.has(volume)) {
    return vtkRenderParameterCache.get(volume) ?? null;
  }
  const params = estimateVtkRenderParameters(getVolumeRenderParameterSource(volume));
  vtkRenderParameterCache.set(volume, params);
  return params;
}

function getVtkRenderParametersFromImage(imageId: string): VtkRenderParameters | null {
  if (!imageId) return null;
  if (vtkFallbackRenderParameterCache.has(imageId)) {
    return vtkFallbackRenderParameterCache.get(imageId) ?? null;
  }
  const image = cornerstone.cache.getImage?.(imageId);
  const params = estimateVtkRenderParameters(getImageRenderParameterSource(image));
  vtkFallbackRenderParameterCache.set(imageId, params);
  return params;
}

function cacheVtkRenderParametersFromImage(imageId: string, image: any) {
  if (!imageId) return null;
  if (vtkFallbackRenderParameterCache.has(imageId)) {
    return vtkFallbackRenderParameterCache.get(imageId) ?? null;
  }
  const params = estimateVtkRenderParameters(getImageRenderParameterSource(image));
  vtkFallbackRenderParameterCache.set(imageId, params);
  return params;
}

function createVtkBonePreset(params: VtkRenderParameters, includeSoftTissue: boolean) {
  const { mn, mx, t0, t1, t2 } = params;
  const colorTransfer = formatColorTransfer([
    [mn, 0.3, 0.3, 1.0],
    [t0, 0.95, 0.95, 0.85],
    [(t0 + t2) / 2, 0.75, 0.4, 0.35],
    [t2, 0.95, 0.84, 0.19],
    [mx, 0.78, 0.78, 0.92],
  ]);
  const opacityPoints: Array<[number, number]> = [[mn, 0]];
  if (includeSoftTissue) {
    opacityPoints.push(
      [t0, 0],
      [t0 + VTK_RENDER_SOFT_TISSUE_WIDTH / 2, 0.5],
      [t0 + VTK_RENDER_SOFT_TISSUE_WIDTH, 0]
    );
  }
  opacityPoints.push([t1, 0], [t2, 0.5], [mx, 0.75]);

  return {
    name: includeSoftTissue ? 'VTK Bone1' : 'VTK Bone2',
    gradientOpacity: '4 0 1 255 1',
    specularPower: '1',
    scalarOpacity: formatScalarOpacity(opacityPoints),
    specular: '0',
    shade: '1',
    ambient: '0.2',
    colorTransfer,
    diffuse: '1',
    interpolation: '1',
    scalarOpacityUnitDistance: '0.8919',
  };
}

function resolve3DVolumePreset(
  viewport: any,
  presetKey: Mpr3DPreset,
  vtkRenderParameters: VtkRenderParameters | null = null
) {
  if (presetKey === 'vtk-bone1' || presetKey === 'vtk-bone2') {
    const volumeId = typeof viewport?.getVolumeId === 'function' ? viewport.getVolumeId() : undefined;
    let volume = null;
    try {
      volume = volumeId ? (cornerstone.cache as any)?.getVolume?.(volumeId) : null;
    } catch {
      volume = null;
    }
    const imageId =
      (typeof viewport?.getCurrentImageId === 'function' ? viewport.getCurrentImageId() : '') ||
      (Array.isArray(volume?.imageIds) ? volume.imageIds[0] : '');
    const params = vtkRenderParameters || getVtkRenderParametersFromVolume(volume) || getVtkRenderParametersFromImage(imageId);
    if (!params) {
      console.warn(`[DentalMPR] ${presetKey} is using default VTK render parameters; no Dental preset fallback is used.`);
    }
    return createVtkBonePreset(params || VTK_RENDER_DEFAULT_PARAMETERS, presetKey === 'vtk-bone1');
  }
  return VOLUME_PRESETS[presetKey] || VOLUME_PRESETS['dental-bone'];
}

function get3DStillSampleDistanceMultiplier(presetKey: Mpr3DPreset) {
  return presetKey === 'dental-surface' || presetKey === 'vtk-bone1' ? 0.7 : 1;
}

function apply3DMapperRenderResolution(viewport: any, isInteractive: boolean) {
  const actors = typeof viewport?.getActors === 'function' ? viewport.getActors() : [];
  actors.forEach((entry: any) => {
    const mapper = entry?.actor?.getMapper?.();
    mapper?.setImageSampleDistance?.(
      isInteractive ? MPR_3D_INTERACTIVE_IMAGE_SAMPLE_DISTANCE : MPR_3D_STILL_IMAGE_SAMPLE_DISTANCE
    );
    mapper?.setMaximumSamplesPerRay?.(
      isInteractive ? MPR_3D_INTERACTIVE_MAXIMUM_SAMPLES_PER_RAY : MPR_3D_STILL_MAXIMUM_SAMPLES_PER_RAY
    );
  });
}

function apply3DRenderQuality(viewport: any, presetKey: Mpr3DPreset, isInteractive: boolean) {
  viewport?.setSampleDistanceMultiplier?.(
    isInteractive ? MPR_3D_INTERACTIVE_SAMPLE_DISTANCE_MULTIPLIER : get3DStillSampleDistanceMultiplier(presetKey)
  );
  apply3DMapperRenderResolution(viewport, isInteractive);
}

function apply3DVolumePreset(
  viewport: any,
  presetKey: Mpr3DPreset,
  isInteractive = false,
  vtkRenderParameters: VtkRenderParameters | null = null
) {
  const preset = resolve3DVolumePreset(viewport, presetKey, vtkRenderParameters);
  const volumeId = typeof viewport?.getVolumeId === 'function' ? viewport.getVolumeId() : undefined;
  viewport?.setProperties?.({ preset }, volumeId);
  apply3DMapperBlendMode(
    viewport,
    presetKey === 'dental-mip' || presetKey === 'vtk-mip' ? BlendModes.MAXIMUM_INTENSITY_BLEND : BlendModes.COMPOSITE
  );
  apply3DRenderQuality(viewport, presetKey, isInteractive);
}

function apply3DCoronalCamera(viewport: any, storeAsInitialCamera = false) {
  if (!viewport) return;
  viewport.setCamera?.(MPR_3D_CORONAL_CAMERA);
  viewport.resetCamera?.({
    resetPan: true,
    resetZoom: true,
    resetToCenter: true,
  });
  viewport.setCamera?.(MPR_3D_CORONAL_CAMERA, storeAsInitialCamera);
}

const subtract3 = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add3 = (a: number[], b: number[]) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale3 = (v: number[], s: number) => [v[0] * s, v[1] * s, v[2] * s];
const dot3 = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: number[], b: number[]) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalize3 = (v: number[]) => {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
};
const rotateAroundAxis = (value: number[], axisInput: number[], angle: number) => {
  const axis = normalize3(axisInput);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const term1 = scale3(value, cos);
  const term2 = scale3(cross3(axis, value), sin);
  const term3 = scale3(axis, dot3(axis, value) * (1 - cos));
  return add3(add3(term1, term2), term3);
};

function attachViewportToToolGroup(toolGroup: any, viewportId: string, renderingEngineId: string) {
  if (!toolGroup) return;
  const infos = (toolGroup.getViewportsInfo?.() ?? toolGroup.viewportsInfo ?? []) as {
    viewportId: string;
    renderingEngineId?: string;
  }[];
  infos
    .filter((info) => info.viewportId === viewportId && info.renderingEngineId !== renderingEngineId)
    .forEach((info) => {
      try {
        toolGroup.removeViewports(info.renderingEngineId ?? renderingEngineId, viewportId);
      } catch {
        // ToolGroup cleanup is best-effort because the previous rendering engine may already be destroyed.
      }
    });

  const hasCurrentViewport = ((toolGroup.getViewportsInfo?.() ?? toolGroup.viewportsInfo ?? []) as {
    viewportId: string;
    renderingEngineId?: string;
  }[]).some((info) => info.viewportId === viewportId && info.renderingEngineId === renderingEngineId);

  if (!hasCurrentViewport) {
    toolGroup.addViewport(viewportId, renderingEngineId);
  }
}

function formatSeriesIndex(series: DentalMprSeriesInfo) {
  if (series.seriesNumber != null && Number.isFinite(Number(series.seriesNumber))) {
    return String(series.seriesNumber);
  }
  return '1';
}

function getPanelSlabThickness(panelId: MprPanelId, values: MprSlabThicknessByAxis, fallbackThicknessMm: number) {
  const axis = MPR_PANEL_SLAB_AXIS[panelId];
  return axis ? values[axis] : fallbackThicknessMm;
}

function updatePanelSlabThickness(
  panelId: MprPanelId,
  values: MprSlabThicknessByAxis,
  thicknessMm: number
): MprSlabThicknessByAxis {
  const axis = MPR_PANEL_SLAB_AXIS[panelId];
  if (!axis) return values;
  return {
    ...values,
    [axis]: Math.max(0, Number(thicknessMm) || 0),
  };
}

function formatViewportNumber(value: number | null, digits = 0) {
  if (value == null || !Number.isFinite(value)) return '-';
  return digits > 0 ? value.toFixed(digits) : String(Math.round(value));
}

function readFirstNumber(value: unknown): number | null {
  const candidate = Array.isArray(value) ? value[0] : value;
  if (typeof candidate === 'number') {
    return Number.isFinite(candidate) ? candidate : null;
  }
  if (typeof candidate === 'string') {
    const parsed = Number.parseFloat(candidate.split('\\')[0]);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function createMprWindowState(windowCenter: unknown, windowWidth: unknown): MprWindowState | null {
  const center = readFirstNumber(windowCenter);
  const width = readFirstNumber(windowWidth);
  if (center == null || width == null || width <= 0) return null;
  return {
    windowLevel: Math.trunc(center + 0.5),
    windowWidth: Math.max(2, Math.trunc(width + 0.5)),
  };
}

function createVoiRangeFromDisplayWindow(windowState: MprWindowState | null): MprVoiRange | null {
  if (!windowState) return null;
  const nWndLevel = Math.trunc(windowState.windowLevel);
  const nWndWidth = Math.max(2, Math.trunc(windowState.windowWidth));
  if (!Number.isFinite(nWndLevel) || !Number.isFinite(nWndWidth) || nWndWidth <= 0) return null;
  const minv = Math.trunc(nWndLevel - nWndWidth / 2);
  const maxv = Math.trunc(nWndLevel + nWndWidth / 2 + 1);
  if (!Number.isFinite(minv) || !Number.isFinite(maxv) || maxv <= minv) return null;
  return { lower: minv, upper: maxv };
}

function createMprWindowStateFromVoiRange(voiRange: unknown): MprWindowState | null {
  const lower = Number((voiRange as { lower?: unknown } | null)?.lower);
  const upper = Number((voiRange as { upper?: unknown } | null)?.upper);
  if (!Number.isFinite(lower) || !Number.isFinite(upper) || upper <= lower) return null;
  return {
    windowLevel: Math.trunc((lower + upper) / 2 + 0.5),
    windowWidth: Math.max(2, Math.trunc(upper - lower + 0.5)),
  };
}

function resolveMprWindowFromImage(image: any, imageId: string): MprWindowState | null {
  const voiLutModule = cornerstone.metaData.get('voiLutModule', imageId) as
    | { windowCenter?: unknown; windowWidth?: unknown }
    | undefined;
  return (
    createMprWindowState(voiLutModule?.windowCenter, voiLutModule?.windowWidth) ||
    createMprWindowState(image?.windowCenter, image?.windowWidth)
  );
}

function getViewportOverlayInfo(viewport: any): ViewportOverlayInfo {
  const index = typeof viewport?.getSliceIndex === 'function' ? Number(viewport.getSliceIndex()) : 0;
  const count = typeof viewport?.getNumberOfSlices === 'function' ? Number(viewport.getNumberOfSlices()) : 0;
  const properties = typeof viewport?.getProperties === 'function' ? viewport.getProperties() : {};
  const voiRange = properties?.voiRange;
  const lower = Number(voiRange?.lower);
  const upper = Number(voiRange?.upper);
  const hasVoi = Number.isFinite(lower) && Number.isFinite(upper) && upper !== lower;
  const camera = typeof viewport?.getCamera === 'function' ? viewport.getCamera() : null;
  const parallelScale = Number(camera?.parallelScale);
  const elementHeight = Number(viewport?.element?.clientHeight);
  const imageData = typeof viewport?.getImageData === 'function' ? viewport.getImageData() : null;
  const spacing = Array.isArray(imageData?.spacing)
    ? imageData.spacing
    : typeof imageData?.imageData?.getSpacing === 'function'
      ? imageData.imageData.getSpacing()
      : typeof imageData?.getSpacing === 'function'
        ? imageData.getSpacing()
        : null;
  const rowSpacing = Array.isArray(spacing) ? Number(spacing[1] ?? spacing[0]) : NaN;
  const pixelSizeMm =
    Number.isFinite(parallelScale) && parallelScale > 0 && Number.isFinite(elementHeight) && elementHeight > 0
      ? (parallelScale * 2) / elementHeight
      : null;
  const zoom =
    pixelSizeMm != null && Number.isFinite(rowSpacing) && rowSpacing > 0
      ? rowSpacing / pixelSizeMm
      : null;

  return {
    index: Number.isFinite(index) ? index : 0,
    count: Number.isFinite(count) ? count : 0,
    windowLevel: hasVoi ? (lower + upper) / 2 : null,
    windowWidth: hasVoi ? upper - lower : null,
    zoom,
    pixelSizeMm,
  };
}

function MprViewportPanel({
  panel,
  hostRef,
  loading,
  error,
  isMaximized,
  onToggleMaximize,
  seriesInfo,
  sliceInfo,
  projectionMode,
  slabThicknessMm,
  filterEnabled,
  filterType,
  windowState,
  onWindowChange,
  onWindowReset,
  onProjectionModeChange,
  onSlabThicknessChange,
  onFilterChange,
  on3DPointerDown,
  on3DPointerMove,
  on3DPointerUp,
  on3DWheel,
  showHuOverlay,
  huOverlayInfo,
  on2DPointerMove,
  on2DPointerLeave,
}: {
  panel: MprPanelConfig;
  hostRef: (el: HTMLDivElement | null) => void;
  loading: boolean;
  error: string | null;
  isMaximized: boolean;
  onToggleMaximize: (id: MprPanelId) => void;
  seriesInfo: DentalMprSeriesInfo;
  sliceInfo?: ViewportOverlayInfo;
  projectionMode: MprProjectionMode;
  slabThicknessMm: number;
  filterEnabled: boolean;
  filterType: MprFilterType | MprFilterSelection | null;
  windowState: MprWindowState | null;
  onWindowChange: (state: MprWindowState) => void;
  onWindowReset: () => void;
  onProjectionModeChange: (mode: MprProjectionMode) => void;
  onSlabThicknessChange: (thicknessMm: number) => void;
  onFilterChange: (filterType: MprFilterSelection) => void;
  on3DPointerDown?: (event: ReactPointerEvent<HTMLDivElement>) => void;
  on3DPointerMove?: (event: ReactPointerEvent<HTMLDivElement>) => void;
  on3DPointerUp?: (event: ReactPointerEvent<HTMLDivElement>) => void;
  on3DWheel?: (event: ReactWheelEvent<HTMLDivElement>) => void;
  showHuOverlay: boolean;
  huOverlayInfo?: MprHuOverlayInfo | null;
  on2DPointerMove?: (panelId: MprPanelId, event: ReactPointerEvent<HTMLDivElement>) => void;
  on2DPointerLeave?: (panelId: MprPanelId) => void;
}) {
  const is3D = panel.plane === 'volume3d';
  const patientSexAge = [seriesInfo.patientSex, seriesInfo.patientAge].filter(Boolean).join(' ');
  const renderMenuRef = useRef<HTMLDivElement | null>(null);
  const [openMenu, setOpenMenu] = useState<'mode' | 'thickness' | 'filter' | null>(null);
  const [openWindowMenu, setOpenWindowMenu] = useState(false);
  const [thicknessDraft, setThicknessDraft] = useState(String(slabThicknessMm));
  const [windowDraft, setWindowDraft] = useState<MprWindowState | null>(windowState);
  const filterSelection = normalizeMprFilterSelection(filterType);
  const filterLabel = formatMprFilterSelection(filterType);

  useEffect(() => {
    setThicknessDraft(String(slabThicknessMm));
  }, [slabThicknessMm]);

  useEffect(() => {
    setWindowDraft(windowState);
  }, [windowState?.windowLevel, windowState?.windowWidth]);

  useEffect(() => {
    if (!openMenu) return;
    const handleClose = (event: MouseEvent) => {
      const root = renderMenuRef.current;
      if (root && root.contains(event.target as Node)) return;
      setOpenMenu(null);
    };
    document.addEventListener('mousedown', handleClose);
    return () => document.removeEventListener('mousedown', handleClose);
  }, [openMenu]);

  useEffect(() => {
    if (!openWindowMenu) return;
    const handleClose = (event: MouseEvent) => {
      const target = event.target as Node;
      if (target && (target as Element).closest?.('.mpr-window-control')) return;
      setOpenWindowMenu(false);
    };
    document.addEventListener('mousedown', handleClose);
    return () => document.removeEventListener('mousedown', handleClose);
  }, [openWindowMenu]);

  const stopOverlayMenuEvent = (event: ReactMouseEvent) => {
    event.stopPropagation();
  };

  const applyThicknessDraft = () => {
    const next = Math.max(0, Number(thicknessDraft) || 0);
    onSlabThicknessChange(next);
    setOpenMenu(null);
  };

  const updateWindowDraft = (patch: Partial<MprWindowState>) => {
    const current = windowDraft || windowState;
    if (!current) return;
    const next = {
      windowLevel: Math.trunc(Number(patch.windowLevel ?? current.windowLevel) || 0),
      windowWidth: Math.max(2, Math.trunc(Number(patch.windowWidth ?? current.windowWidth) || 2)),
    };
    setWindowDraft(next);
    onWindowChange(next);
  };
  return (
    <div className="mpr-viewport-panel">
      <div className="mpr-viewport-header">
        <div className="mpr-viewport-title">
          <span className="mpr-viewport-marker">::</span>
          <span>{panel.label}</span>
        </div>
        <div className="mpr-viewport-actions">
          <button className="mpr-viewport-action" type="button" aria-label="Layout">▦</button>
          <button className="mpr-viewport-action" type="button" aria-label="Pop out">↗</button>
          <button className="mpr-viewport-action is-wide" type="button" onClick={() => onToggleMaximize(panel.id)}>
            {isMaximized ? '축소' : '확대'}
          </button>
        </div>
      </div>
      <div className="mpr-viewport-content">
        <div
          ref={hostRef}
          className={`mpr-canvas-host${is3D ? ' is-3d' : ''}`}
          onPointerDown={is3D ? on3DPointerDown : undefined}
          onPointerMove={is3D ? on3DPointerMove : undefined}
          onPointerUp={is3D ? on3DPointerUp : undefined}
          onPointerCancel={is3D ? on3DPointerUp : undefined}
          onPointerMoveCapture={!is3D && showHuOverlay ? (event) => on2DPointerMove?.(panel.id, event) : undefined}
          onPointerLeave={!is3D && showHuOverlay ? () => on2DPointerLeave?.(panel.id) : undefined}
          onWheel={is3D ? on3DWheel : undefined}
        />
        <div className="mpr-viewport-overlay">
          {!is3D && showHuOverlay && huOverlayInfo ? (
            <div
              className="mpr-hu-overlay"
              style={{
                left: `${huOverlayInfo.x}px`,
                top: `${huOverlayInfo.y}px`,
              }}
            >
              HU: {Math.round(huOverlayInfo.value)}
            </div>
          ) : null}
          <div className="mpr-overlay-left-top">
            <div>ID: {seriesInfo.patientID || '-'}</div>
            <div>{seriesInfo.patientName || 'Anonymous'}{patientSexAge ? ` [${patientSexAge}]` : ''}</div>
            {seriesInfo.studyDate && <div>{seriesInfo.studyDate}</div>}
            <div>Series: {formatSeriesIndex(seriesInfo)}</div>
          </div>
          <div className="mpr-overlay-right-top">
            <div className="mpr-slice-index">
              {is3D ? 'Vol' : sliceInfo?.count ? `${sliceInfo.index + 1}/${sliceInfo.count}` : '-'}
            </div>
            <div>{is3D ? '3D' : '2D'}</div>
            {is3D ? (
              <div>VRT</div>
            ) : (
              <div
                ref={renderMenuRef}
                className="mpr-render-menu"
                onPointerDown={stopOverlayMenuEvent}
                onPointerMove={stopOverlayMenuEvent}
                onPointerUp={stopOverlayMenuEvent}
                onMouseDown={stopOverlayMenuEvent}
                onMouseMove={stopOverlayMenuEvent}
                onMouseUp={stopOverlayMenuEvent}
                onClick={stopOverlayMenuEvent}
                onWheel={stopOverlayMenuEvent}
              >
                <button
                  type="button"
                  className="mpr-render-line"
                  aria-haspopup="menu"
                  aria-expanded={openMenu === 'mode'}
                  onClick={() => setOpenMenu((current) => (current === 'mode' ? null : 'mode'))}
                >
                  {resolveProjectionLabel(projectionMode)}
                </button>
                <button
                  type="button"
                  className="mpr-render-line"
                  onClick={() => setOpenMenu((current) => (current === 'thickness' ? null : 'thickness'))}
                >
                  TH: {slabThicknessMm} mm
                </button>
                <button
                  type="button"
                  className="mpr-render-line"
                  onClick={() => setOpenMenu((current) => (current === 'filter' ? null : 'filter'))}
                >
                  Filter {filterEnabled && filterLabel ? filterLabel : 'off'}
                </button>
                {openMenu === 'mode' ? (
                  <div className="mpr-popup mpr-mode-popup" role="menu">
                    {PROJECTION_OPTIONS.map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        className={`mpr-mode-item${option.value === projectionMode ? ' is-selected' : ''}`}
                        onClick={() => {
                          onProjectionModeChange(option.value);
                          setOpenMenu(null);
                        }}
                      >
                        <span>{option.value === projectionMode ? 'v' : ''}</span>
                        {option.label}
                      </button>
                    ))}
                    <div className="mpr-popup-separator" />
                    <button type="button" className="mpr-mode-item" onClick={() => setOpenMenu(null)}>
                      Apply to All
                    </button>
                  </div>
                ) : null}
                {openMenu === 'thickness' ? (
                  <div className="mpr-popup mpr-thickness-popup" role="dialog" aria-label="Thickness">
                    <div className="mpr-popup-title">Thickness [mm]</div>
                    <div className="mpr-thickness-row">
                      <input
                        type="range"
                        min="0"
                        max="50"
                        step="0.5"
                        value={Math.max(0, Number(thicknessDraft) || 0)}
                        onChange={(event) => setThicknessDraft(event.target.value)}
                        disabled={projectionMode === 'vr'}
                      />
                      <input
                        className="mpr-thickness-value"
                        type="number"
                        min="0"
                        step="0.5"
                        value={thicknessDraft}
                        onChange={(event) => setThicknessDraft(event.target.value)}
                        disabled={projectionMode === 'vr'}
                      />
                    </div>
                    <div className="mpr-popup-actions">
                      <button type="button" onClick={applyThicknessDraft}>
                        OK
                      </button>
                      <button type="button" onClick={() => setOpenMenu(null)}>
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : null}
                {openMenu === 'filter' ? (
                  <div className="mpr-popup mpr-filter-popup" role="dialog" aria-label="Select filters">
                    <div className="mpr-popup-title">Select Filter(s)</div>
                    <div className="mpr-filter-table">
                      <div className="mpr-filter-head">Filter type</div>
                      <div className="mpr-filter-head">1st</div>
                      <div className="mpr-filter-head">2nd</div>
                      {MPR_FILTER_OPTIONS.map((filter) => {
                        const isFirst = filterSelection.first === filter;
                        const isSecond = filterSelection.second === filter;
                        return (
                          <div
                            key={filter}
                            className={`mpr-filter-row${isFirst || isSecond ? ' is-selected' : ''}`}
                          >
                            <span>{filter}</span>
                            <button
                              type="button"
                              className={isFirst ? 'is-selected' : ''}
                              onClick={() =>
                                onFilterChange({
                                  ...filterSelection,
                                  first: isFirst ? null : filter,
                                })
                              }
                            >
                              {isFirst ? 'O' : ''}
                            </button>
                            <button
                              type="button"
                              className={isSecond ? 'is-selected' : ''}
                              onClick={() =>
                                onFilterChange({
                                  ...filterSelection,
                                  second: isSecond ? null : filter,
                                })
                              }
                            >
                              {isSecond ? 'O' : ''}
                            </button>
                          </div>
                        );
                      })}
                    </div>
                    <div className="mpr-popup-actions">
                      <button type="button" onClick={() => setOpenMenu(null)}>
                        OK
                      </button>
                      <button type="button" onClick={() => setOpenMenu(null)}>
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : null}
              </div>
            )}
            {!is3D && (
              <div className="mpr-overlay-menu-spacer" aria-hidden="true" />
            )}
          </div>
          <div className="mpr-overlay-left-bottom">
            <div>GT: {seriesInfo.gantryTilt ?? '-'}</div>
            <div>{seriesInfo.tubeCurrent ?? '-'} [mA]</div>
            <div>{seriesInfo.tubeVoltage ?? '-'} [kVp]</div>
            <div
              className="mpr-window-control"
              onPointerDown={stopOverlayMenuEvent}
              onPointerMove={stopOverlayMenuEvent}
              onPointerUp={stopOverlayMenuEvent}
              onMouseDown={stopOverlayMenuEvent}
              onMouseMove={stopOverlayMenuEvent}
              onMouseUp={stopOverlayMenuEvent}
              onClick={stopOverlayMenuEvent}
              onWheel={stopOverlayMenuEvent}
            >
              <button
                type="button"
                className="mpr-window-line"
                onClick={() => setOpenWindowMenu((current) => !current)}
              >
                WL:{formatViewportNumber(sliceInfo?.windowLevel ?? null)} WW:{formatViewportNumber(sliceInfo?.windowWidth ?? null)}
              </button>
              {openWindowMenu && windowDraft ? (
                <div className="mpr-popup mpr-window-popup" role="dialog" aria-label="Window level and width">
                  <div className="mpr-popup-title">Window</div>
                  <label className="mpr-window-row">
                    <span>WL</span>
                    <input
                      type="range"
                      min={Math.trunc(windowDraft.windowLevel - windowDraft.windowWidth)}
                      max={Math.trunc(windowDraft.windowLevel + windowDraft.windowWidth)}
                      step="1"
                      value={windowDraft.windowLevel}
                      onChange={(event) => updateWindowDraft({ windowLevel: Number(event.target.value) })}
                    />
                    <input
                      className="mpr-window-value"
                      type="number"
                      step="1"
                      value={windowDraft.windowLevel}
                      onChange={(event) => updateWindowDraft({ windowLevel: Number(event.target.value) })}
                    />
                  </label>
                  <label className="mpr-window-row">
                    <span>WW</span>
                    <input
                      type="range"
                      min="2"
                      max={Math.max(2, Math.trunc(windowDraft.windowWidth * 2))}
                      step="1"
                      value={windowDraft.windowWidth}
                      onChange={(event) => updateWindowDraft({ windowWidth: Number(event.target.value) })}
                    />
                    <input
                      className="mpr-window-value"
                      type="number"
                      min="2"
                      step="1"
                      value={windowDraft.windowWidth}
                      onChange={(event) => updateWindowDraft({ windowWidth: Number(event.target.value) })}
                    />
                  </label>
                  <div className="mpr-popup-actions">
                    <button type="button" onClick={() => setOpenWindowMenu(false)}>
                      OK
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        onWindowReset();
                        setOpenWindowMenu(false);
                      }}
                    >
                      Reset
                    </button>
                  </div>
                </div>
              ) : null}
            </div>
            <div className="mpr-zoom-line">
              Zoom: x{formatViewportNumber(sliceInfo?.zoom ?? null, 1)} ({formatViewportNumber(sliceInfo?.pixelSizeMm ?? null, 3)} mm)
            </div>
          </div>
          <div className="mpr-orientation-cube" aria-hidden="true">
            <span />
            <span>A</span>
            <span />
            <span>R</span>
            <span />
            <span>L</span>
            <span />
            <span>F</span>
            <span />
          </div>
        </div>
        {loading && <div className="mpr-status">Loading {panel.label}...</div>}
        {error && <div className="mpr-status is-error">{error}</div>}
      </div>
    </div>
  );
}

export function DentalMprWorkspace({
  source,
  seriesInfo,
  height,
  viewportControlState,
  active2DTool = 'crosshair',
  showHuOverlay = true,
  control3DState,
  vtkRenderParameters = null,
  onViewportControlStateChange,
}: DentalMprWorkspaceProps) {
  const viewportRefs = useRef<Record<MprPanelId, HTMLDivElement | null>>({
    P0: null,
    P1: null,
    P2: null,
    P3: null,
  });
  const renderingEngineIdRef = useRef(`dental-mpr-engine-${Math.random().toString(36).slice(2)}`);
  const volumeIdRef = useRef(`streaming-volume:dental-mpr-${Math.random().toString(36).slice(2)}`);
  const originalMprWindowRef = useRef<MprWindowState | null>(null);
  const mprWindowRef = useRef<MprWindowState | null>(null);
  const renderingEngineRef = useRef<cornerstone.RenderingEngine | null>(null);
  const drag3DRef = useRef<{ pointerId: number; x: number; y: number } | null>(null);
  const restore3DQualityTimerRef = useRef<number | null>(null);
  const hasAppliedInitial3DCameraRef = useRef(false);
  const last3DResetTokenRef = useRef<number | null>(null);
  const vtkRenderParametersRef = useRef<VtkRenderParameters | null>(null);
  vtkRenderParametersRef.current = vtkRenderParameters ?? vtkRenderParametersRef.current;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sliceInfo, setSliceInfo] = useState<Record<MprPanelId, ViewportOverlayInfo>>(
    {} as Record<MprPanelId, ViewportOverlayInfo>
  );
  const [huOverlayInfo, setHuOverlayInfo] = useState<Partial<Record<MprPanelId, MprHuOverlayInfo>>>({});
  const [mprWindow, setMprWindow] = useState<MprWindowState | null>(null);
  const [maximizedPanel, setMaximizedPanel] = useState<MprPanelId | null>(null);
  const [localViewportControlState, setLocalViewportControlState] = useState<MprViewportControlState>({
    projectionMode: 'mpr',
    slabThicknessMm: 0,
    slabThicknessByAxis: DEFAULT_MPR_SLAB_THICKNESS_BY_AXIS,
    filterEnabled: false,
    filterType: null,
    windowLevel: null,
    windowWidth: null,
  });

  const imageIds = useMemo(() => source.imageIds || [], [source.imageIds]);
  const effectiveViewportControlState = viewportControlState ?? localViewportControlState;
  const effective3DControlState = control3DState ?? DEFAULT_3D_CONTROL_STATE;
  const projectionMode = effectiveViewportControlState.projectionMode;
  const slabThicknessMm = effectiveViewportControlState.slabThicknessMm;
  const slabThicknessByAxis = normalizeMprSlabThicknessByAxis(
    effectiveViewportControlState.slabThicknessByAxis,
    slabThicknessMm
  );
  const filterEnabled = Boolean(
    effectiveViewportControlState.filterEnabled && hasMprFilterSelection(effectiveViewportControlState.filterType)
  );
  const filterType = effectiveViewportControlState.filterType ?? null;
  const controlledMprWindow =
    typeof effectiveViewportControlState.windowLevel === 'number' &&
    typeof effectiveViewportControlState.windowWidth === 'number' &&
    effectiveViewportControlState.windowWidth > 0
      ? {
          windowLevel: effectiveViewportControlState.windowLevel,
          windowWidth: effectiveViewportControlState.windowWidth,
        }
      : null;
  const activeMprWindow = controlledMprWindow ?? mprWindow;
  const active2DToolRef = useRef<Mpr2DTool>(active2DTool);
  const viewportControlStateRef = useRef(effectiveViewportControlState);
  const onViewportControlStateChangeRef = useRef(onViewportControlStateChange);

  const setActiveMprWindow = (windowState: MprWindowState | null) => {
    mprWindowRef.current = windowState;
    setMprWindow(windowState);
  };

  useEffect(() => {
    if (!controlledMprWindow) return;
    mprWindowRef.current = controlledMprWindow;
    setMprWindow(controlledMprWindow);
    const renderingEngine = renderingEngineRef.current;
    if (!renderingEngine) return;
    const voiRange = createVoiRangeFromDisplayWindow(controlledMprWindow);
    if (!voiRange) return;
    PANELS.forEach((panel) => {
      if (panel.plane === 'volume3d') return;
      const viewport = renderingEngine.getViewport(panel.id) as any;
      viewport?.setProperties?.({ voiRange });
      viewport?.render?.();
    });
    refreshSliceInfo();
  }, [controlledMprWindow?.windowLevel, controlledMprWindow?.windowWidth]);

  useEffect(() => {
    viewportControlStateRef.current = effectiveViewportControlState;
    onViewportControlStateChangeRef.current = onViewportControlStateChange;
    active2DToolRef.current = active2DTool;
  }, [effectiveViewportControlState, onViewportControlStateChange, active2DTool]);

  const updateViewportControlState = (patch: Partial<MprViewportControlState>) => {
    const next = {
      ...viewportControlStateRef.current,
      ...patch,
    };
    viewportControlStateRef.current = next;
    setLocalViewportControlState(next);
    onViewportControlStateChangeRef.current?.(next);
  };

  useEffect(() => {
    configureDentalMprSlabSync({
      getSlabPlane: (viewportId = 'P0') => {
        const current = viewportControlStateRef.current;
        const currentByAxis = normalizeMprSlabThicknessByAxis(current.slabThicknessByAxis, current.slabThicknessMm);
        const thickness = getPanelSlabThickness(viewportId as MprPanelId, currentByAxis, current.slabThicknessMm);
        return createMprSlabPlane(current.projectionMode, thickness);
      },
      onSlabThicknessChange: (viewportId, thicknessMm) => {
        const current = viewportControlStateRef.current;
        const currentByAxis = normalizeMprSlabThicknessByAxis(current.slabThicknessByAxis, current.slabThicknessMm);
        const nextByAxis = updatePanelSlabThickness(viewportId as MprPanelId, currentByAxis, thicknessMm);
        const nextThickness = getPanelSlabThickness(viewportId as MprPanelId, nextByAxis, thicknessMm);
        if (
          current.projectionMode === 'vr' ||
          (Math.abs(current.slabThicknessMm - nextThickness) < 0.01 &&
            Object.keys(nextByAxis).every(
              (axis) =>
                Math.abs(
                  nextByAxis[axis as MprSlabAxis] - currentByAxis[axis as MprSlabAxis]
                ) < 0.01
            ))
        ) {
          return;
        }

        const next = {
          ...current,
          slabThicknessMm: nextThickness,
          slabThicknessByAxis: nextByAxis,
        };
        viewportControlStateRef.current = next;
        setLocalViewportControlState(next);
        onViewportControlStateChangeRef.current?.(next);
      },
    });
  }, []);

  const refreshSliceInfo = () => {
    const renderingEngine = renderingEngineRef.current;
    if (!renderingEngine) return;
    const next: Partial<Record<MprPanelId, ViewportOverlayInfo>> = {};
    PANELS.forEach((panel) => {
      if (panel.plane === 'volume3d') return;
      const viewport = renderingEngine.getViewport(panel.id) as any;
      if (!viewport) return;
      next[panel.id] = getViewportOverlayInfo(viewport);
    });
    setSliceInfo(next as Record<MprPanelId, ViewportOverlayInfo>);
  };

  const applyViewportControls = () => {
    const renderingEngine = renderingEngineRef.current;
    if (!renderingEngine) return;

    PANELS.forEach((panel) => {
      if (panel.plane === 'volume3d') return;
      const viewport = renderingEngine.getViewport(panel.id) as any;
      if (!viewport) return;
      const panelSlabThicknessMm = getPanelSlabThickness(panel.id, slabThicknessByAxis, slabThicknessMm);
      const slabPlane = createMprSlabPlane(projectionMode, panelSlabThicknessMm);
      const filterProperties = resolveMprFilterProperties(filterEnabled, filterType);
      const voiRange = createVoiRangeFromDisplayWindow(mprWindowRef.current);
      const viewportProperties: Record<string, unknown> = {
        interpolationType: filterProperties.interpolationType,
        smoothing: filterProperties.smoothing,
        sharpening: filterProperties.sharpening,
      };
      if (voiRange) {
        viewportProperties.voiRange = voiRange;
      }
      viewport.setProperties?.(viewportProperties);
      applySlabPlaneToViewport(viewport, slabPlane);
      viewport.render?.();
    });
    refreshSliceInfo();
  };

  const apply3DControls = () => {
    const renderingEngine = renderingEngineRef.current;
    const toolGroup = createOrGet3DToolGroup();
    if (!renderingEngine || !toolGroup) return;

    const primary = cornerstoneTools.Enums.MouseBindings.Primary;
    const wheel = cornerstoneTools.Enums.MouseBindings.Wheel;
    const active3DTool = effective3DControlState.tool === 'rotate' ? 'rotate' : 'panzoom';
    [TrackballRotateTool.toolName, PanTool.toolName, ZoomTool.toolName].forEach((toolName) => {
      if (toolGroup.hasTool(toolName)) {
        toolGroup.setToolPassive(toolName, { removeAllBindings: true });
      }
    });
    if (active3DTool === 'panzoom') {
      toolGroup.setToolActive(PanTool.toolName, { bindings: [{ mouseButton: primary }] });
      toolGroup.setToolActive(ZoomTool.toolName, { bindings: [{ mouseButton: wheel }] });
    } else {
      toolGroup.setToolActive(TrackballRotateTool.toolName, { bindings: [{ mouseButton: primary }] });
      toolGroup.setToolActive(ZoomTool.toolName, { bindings: [{ mouseButton: wheel }] });
    }

    const viewport = renderingEngine.getViewport('P3') as any;
    apply3DVolumePreset(viewport, effective3DControlState.preset, false, vtkRenderParametersRef.current);
    const shouldApplyDefaultCamera =
      !hasAppliedInitial3DCameraRef.current ||
      effective3DControlState.resetToken !== last3DResetTokenRef.current;
    if (shouldApplyDefaultCamera) {
      apply3DCoronalCamera(viewport, !hasAppliedInitial3DCameraRef.current);
      hasAppliedInitial3DCameraRef.current = true;
      last3DResetTokenRef.current = effective3DControlState.resetToken;
    }
    viewport?.render?.();
  };

  const resizeMprViewportsAfterLayout = (options: { resetCameraForResize?: boolean } = {}) => {
    const renderingEngine = renderingEngineRef.current;
    if (!renderingEngine) return false;

    const allElementsReady = PANELS.every((panel) => {
      const element = viewportRefs.current[panel.id];
      return element?.isConnected && element.clientWidth > 2 && element.clientHeight > 2;
    });
    if (!allElementsReady) return false;

    renderingEngine.resize(true, true);
    PANELS.forEach((panel) => {
      const viewport = renderingEngine.getViewport(panel.id) as any;
      if (!viewport) return;
      if (options.resetCameraForResize) {
        viewport.resetCameraForResize?.();
      }
      viewport.render?.();
    });
    refreshSliceInfo();
    if (active2DToolRef.current === 'crosshair') {
      refreshDentalMprCrosshairs(renderingEngineIdRef.current);
    }
    return true;
  };

  const scheduleMprResizeAfterLayout = (options: { resetCameraForResize?: boolean } = {}) => {
    let firstFrame = 0;
    let secondFrame = 0;
    let retryFrame = 0;

    firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => {
        if (!resizeMprViewportsAfterLayout(options)) {
          retryFrame = window.requestAnimationFrame(() => {
            resizeMprViewportsAfterLayout(options);
          });
        }
      });
    });

    return () => {
      window.cancelAnimationFrame(firstFrame);
      window.cancelAnimationFrame(secondFrame);
      window.cancelAnimationFrame(retryFrame);
    };
  };

  const set3DInteractiveQuality = (isInteractive: boolean) => {
    if (restore3DQualityTimerRef.current) {
      window.clearTimeout(restore3DQualityTimerRef.current);
      restore3DQualityTimerRef.current = null;
    }
    const viewport = renderingEngineRef.current?.getViewport('P3') as any;
    if (!viewport) return;
    apply3DRenderQuality(viewport, effective3DControlState.preset, isInteractive);
    viewport.render?.();
  };

  const schedule3DQualityRestore = () => {
    if (restore3DQualityTimerRef.current) {
      window.clearTimeout(restore3DQualityTimerRef.current);
    }
    restore3DQualityTimerRef.current = window.setTimeout(() => {
      restore3DQualityTimerRef.current = null;
      set3DInteractiveQuality(false);
    }, MPR_3D_QUALITY_RESTORE_DELAY_MS);
  };

  const rotate3DViewport = (deltaX: number, deltaY: number) => {
    const viewport = renderingEngineRef.current?.getViewport('P3') as any;
    if (!viewport) return;

    const camera = viewport.getCamera?.();
    const position = camera?.position as number[] | undefined;
    const focalPoint = camera?.focalPoint as number[] | undefined;
    const viewUp = camera?.viewUp as number[] | undefined;
    if (!position || !focalPoint || !viewUp) return;

    const yaw = -deltaX * 0.008;
    const pitch = -deltaY * 0.008;
    let offset = subtract3(position, focalPoint);
    let nextViewUp = normalize3([...viewUp]);

    offset = rotateAroundAxis(offset, nextViewUp, yaw);
    const right = normalize3(cross3(nextViewUp, offset));
    offset = rotateAroundAxis(offset, right, pitch);
    nextViewUp = rotateAroundAxis(nextViewUp, right, pitch);

    viewport.setCamera?.({
      position: add3(focalPoint, offset),
      focalPoint,
      viewUp: nextViewUp,
    });
    viewport.render?.();
  };

  const zoom3DViewport = (deltaY: number) => {
    const viewport = renderingEngineRef.current?.getViewport('P3') as any;
    if (!viewport) return;
    const camera = viewport.getCamera?.();
    const parallelScale = Number(camera?.parallelScale);
    if (!Number.isFinite(parallelScale) || parallelScale <= 0) return;

    const factor = deltaY > 0 ? 1.08 : 0.92;
    const nextParallelScale = Math.min(
      MPR_3D_MAX_PARALLEL_SCALE,
      Math.max(MPR_3D_MIN_PARALLEL_SCALE, parallelScale * factor)
    );
    viewport.setCamera?.({
      position: camera.position,
      focalPoint: camera.focalPoint,
      viewUp: camera.viewUp,
      parallelScale: nextParallelScale,
    });
    viewport.render?.();
  };

  const pan3DViewport = (deltaX: number, deltaY: number) => {
    const viewport = renderingEngineRef.current?.getViewport('P3') as any;
    if (!viewport) return;
    const camera = viewport.getCamera?.();
    const position = camera?.position as number[] | undefined;
    const focalPoint = camera?.focalPoint as number[] | undefined;
    const viewUp = camera?.viewUp as number[] | undefined;
    if (!position || !focalPoint || !viewUp) return;

    const element = viewportRefs.current.P3;
    const rect = element?.getBoundingClientRect();
    const width = Math.max(1, rect?.width || 1);
    const height = Math.max(1, rect?.height || 1);
    const direction = normalize3(subtract3(focalPoint, position));
    const up = normalize3([...viewUp]);
    const right = normalize3(cross3(direction, up));
    const distance = Math.hypot(...subtract3(position, focalPoint));
    const parallelScale = Number(camera.parallelScale) || 0;
    const worldPerPixel = parallelScale > 0 ? (parallelScale * 2) / height : (distance * 2) / Math.max(width, height);
    const translation = add3(
      scale3(right, -deltaX * worldPerPixel),
      scale3(up, deltaY * worldPerPixel)
    );

    viewport.setCamera?.({
      position: add3(position, translation),
      focalPoint: add3(focalPoint, translation),
      viewUp,
    });
    viewport.render?.();
  };

  const handle3DPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    set3DInteractiveQuality(true);
    drag3DRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const handle3DPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = drag3DRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    const deltaX = event.clientX - drag.x;
    const deltaY = event.clientY - drag.y;
    drag3DRef.current = { ...drag, x: event.clientX, y: event.clientY };
    const active3DTool = effective3DControlState.tool === 'rotate' ? 'rotate' : 'panzoom';
    if (active3DTool === 'panzoom') {
      pan3DViewport(deltaX, deltaY);
    } else {
      rotate3DViewport(deltaX, deltaY);
    }
  };

  const handle3DPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = drag3DRef.current;
    if (drag?.pointerId === event.pointerId) {
      event.preventDefault();
      event.stopPropagation();
      drag3DRef.current = null;
      event.currentTarget.releasePointerCapture?.(event.pointerId);
      schedule3DQualityRestore();
    }
  };

  const handle3DWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    set3DInteractiveQuality(true);
    zoom3DViewport(event.deltaY);
    schedule3DQualityRestore();
  };

  const handle2DPointerMove = (panelId: MprPanelId, event: ReactPointerEvent<HTMLDivElement>) => {
    if (!showHuOverlay) return;
    const viewport = renderingEngineRef.current?.getViewport(panelId) as any;
    if (!viewport || typeof viewport.canvasToWorld !== 'function') return;

    const hostRect = event.currentTarget.getBoundingClientRect();
    const contentRect = event.currentTarget.parentElement?.getBoundingClientRect() ?? hostRect;
    const canvasPoint = [event.clientX - hostRect.left, event.clientY - hostRect.top];
    const worldPoint = viewport.canvasToWorld(canvasPoint);
    const intensity =
      typeof viewport.getIntensityFromWorld === 'function'
        ? Number(viewport.getIntensityFromWorld(worldPoint))
        : Number.NaN;

    if (!Number.isFinite(intensity)) {
      setHuOverlayInfo((current) => {
        if (!current[panelId]) return current;
        const next = { ...current };
        delete next[panelId];
        return next;
      });
      return;
    }

    const nextInfo = {
      value: intensity,
      x: event.clientX - contentRect.left + 12,
      y: event.clientY - contentRect.top + 12,
    };

    setHuOverlayInfo((current) => ({
      ...current,
      [panelId]: nextInfo,
    }));
  };

  const handle2DPointerLeave = (panelId: MprPanelId) => {
    setHuOverlayInfo((current) => {
      if (!current[panelId]) return current;
      const next = { ...current };
      delete next[panelId];
      return next;
    });
  };

  useEffect(() => {
    if (showHuOverlay) return;
    setHuOverlayInfo({});
  }, [showHuOverlay]);

  useEffect(() => {
    let cancelled = false;
    async function setup() {
      if (!imageIds.length) {
        setError('No DICOM instances are available for Dental MPR.');
        setLoading(false);
        return;
      }
      setLoading(true);
      setError(null);
      hasAppliedInitial3DCameraRef.current = false;
      last3DResetTokenRef.current = null;
      try {
        await initCornerstone();
        addAndGroupTools();

        const renderingEngineId = renderingEngineIdRef.current;
        try {
          renderingEngineRef.current?.destroy();
        } catch {
          // ignore
        }
        const renderingEngine = new cornerstone.RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;

        if (PANELS.some((panel) => !viewportRefs.current[panel.id])) {
          throw new Error('Dental MPR viewport frame is not ready.');
        }

        const viewportInputs = PANELS.map((panel) => ({
          viewportId: panel.id,
          type:
            panel.plane === 'volume3d'
              ? cornerstone.Enums.ViewportType.VOLUME_3D
              : cornerstone.Enums.ViewportType.ORTHOGRAPHIC,
          element: viewportRefs.current[panel.id]!,
          defaultOptions: {
            orientation: panel.orientation,
            background: (panel.plane === 'volume3d' ? [0.05, 0.05, 0.08] : [0, 0, 0]) as cornerstone.Types.Point3,
          },
        }));

        
        renderingEngine.setViewports(viewportInputs);
        renderingEngine.resize(true, false);

        const toolGroup = createOrGetDentalMprToolGroup();
        const toolGroup3D = createOrGet3DToolGroup();
        pauseDentalMprCrosshairsForViewportAttach();
        PANELS.forEach((panel) => {
          if (panel.plane === 'volume3d') {
            attachViewportToToolGroup(toolGroup3D, panel.id, renderingEngineId);
          } else {
            attachViewportToToolGroup(toolGroup, panel.id, renderingEngineId);
          }
        });
        activateDentalMpr2DTool(active2DToolRef.current);

        const firstImage = await cornerstone.imageLoader.loadAndCacheImage(imageIds[0]);
        vtkRenderParametersRef.current =
          vtkRenderParametersRef.current ?? cacheVtkRenderParametersFromImage(imageIds[0], firstImage);
        const originalWindow = resolveMprWindowFromImage(firstImage, imageIds[0]);
        originalMprWindowRef.current = originalWindow;
        setActiveMprWindow(originalWindow);
        if (originalWindow && !controlledMprWindow) {
          updateViewportControlState({
            windowLevel: originalWindow.windowLevel,
            windowWidth: originalWindow.windowWidth,
          });
        }
        const volume = await cornerstone.volumeLoader.createAndCacheVolume(volumeIdRef.current, { imageIds });
        await volume.load();
        if (cancelled) return;
        await cornerstone.setVolumesForViewports(
          renderingEngine,
          [{ volumeId: volumeIdRef.current }],
          PANELS.map((panel) => panel.id)
        );
        if (cancelled) return;

        renderingEngine.resize(true, false);
        apply3DControls();
        PANELS.forEach((panel) => {
          const viewport = renderingEngine.getViewport(panel.id) as any;
          const voiRange = createVoiRangeFromDisplayWindow(mprWindowRef.current);
          if (panel.plane !== 'volume3d' && voiRange) {
            viewport?.setProperties?.({ voiRange });
          }
          viewport?.resetCamera?.();
          viewport?.render?.();
        });
        refreshDentalMprCrosshairs(renderingEngineId, { reset: true });
        applyViewportControls();
        setLoading(false);
      } catch (nextError) {
        if (cancelled) return;
        setError(nextError instanceof Error ? nextError.message : 'Failed to initialize Dental MPR.');
        setLoading(false);
      }
    }

    void setup();
    const onResize = () => {
      scheduleMprResizeAfterLayout();
    };
    window.addEventListener('resize', onResize);
    return () => {
      cancelled = true;
      window.removeEventListener('resize', onResize);
      if (restore3DQualityTimerRef.current) {
        window.clearTimeout(restore3DQualityTimerRef.current);
        restore3DQualityTimerRef.current = null;
      }
      try {
        removeDentalMprViewports(renderingEngineIdRef.current);
        renderingEngineRef.current?.destroy();
        cornerstone.cache.removeVolumeLoadObject(volumeIdRef.current);
      } catch {
        // ignore
      }
      renderingEngineRef.current = null;
    };
  }, [source.id, imageIds.join('|')]);

  useEffect(() => {
    applyViewportControls();
  }, [
    projectionMode,
    slabThicknessMm,
    slabThicknessByAxis.x,
    slabThicknessByAxis.y,
    slabThicknessByAxis.z,
    filterEnabled,
    filterType,
  ]);

  useEffect(() => {
    activateDentalMpr2DTool(active2DTool);
    if (active2DTool === 'crosshair') {
      window.requestAnimationFrame(() => refreshDentalMprCrosshairs(renderingEngineIdRef.current));
    }
  }, [active2DTool]);

  useEffect(() => {
    apply3DControls();
  }, [effective3DControlState.tool, effective3DControlState.preset, effective3DControlState.resetToken]);

  useEffect(() => {
    return scheduleMprResizeAfterLayout({ resetCameraForResize: Boolean(maximizedPanel) });
  }, [height, maximizedPanel]);

  useEffect(() => {
    const observers = PANELS.map((panel) => {
      const element = viewportRefs.current[panel.id];
      if (!element) return null;
      let cancelScheduledResize: (() => void) | null = null;
      const observer = new ResizeObserver(() => {
        cancelScheduledResize?.();
        cancelScheduledResize = scheduleMprResizeAfterLayout();
      });
      observer.observe(element);
      return {
        disconnect: () => {
          cancelScheduledResize?.();
          observer.disconnect();
        },
      };
    });

    return () => {
      observers.forEach((observer) => observer?.disconnect());
    };
  }, [source.id, maximizedPanel]);

  useEffect(() => {
    const refreshOverlay = () => refreshSliceInfo();
    const handleVoiModified = (event: Event) => {
      refreshSliceInfo();
      if (active2DToolRef.current !== 'wlww') return;
      const viewportId = (event as CustomEvent<{ viewportId?: string }>).detail?.viewportId;
      if (viewportId && !['P0', 'P1', 'P2'].includes(viewportId)) return;
      syncMprWindowFromViewport(viewportId);
    };
    cornerstone.eventTarget.addEventListener(cornerstone.Enums.Events.VOLUME_VIEWPORT_SCROLL, refreshOverlay);
    cornerstone.eventTarget.addEventListener(cornerstone.Enums.Events.CAMERA_MODIFIED, refreshOverlay);
    cornerstone.eventTarget.addEventListener(cornerstone.Enums.Events.VOI_MODIFIED, handleVoiModified);
    cornerstone.eventTarget.addEventListener(cornerstone.Enums.Events.IMAGE_RENDERED, refreshOverlay);
    return () => {
      cornerstone.eventTarget.removeEventListener(cornerstone.Enums.Events.VOLUME_VIEWPORT_SCROLL, refreshOverlay);
      cornerstone.eventTarget.removeEventListener(cornerstone.Enums.Events.CAMERA_MODIFIED, refreshOverlay);
      cornerstone.eventTarget.removeEventListener(cornerstone.Enums.Events.VOI_MODIFIED, handleVoiModified);
      cornerstone.eventTarget.removeEventListener(cornerstone.Enums.Events.IMAGE_RENDERED, refreshOverlay);
    };
  }, []);

  const applyMprWindow = (windowState: MprWindowState | null) => {
    setActiveMprWindow(windowState);
    updateViewportControlState({
      windowLevel: windowState?.windowLevel ?? null,
      windowWidth: windowState?.windowWidth ?? null,
    });
    const renderingEngine = renderingEngineRef.current;
    if (!renderingEngine) return;
    const voiRange = createVoiRangeFromDisplayWindow(windowState);
    PANELS.forEach((panel) => {
      if (panel.plane === 'volume3d') return;
      const viewport = renderingEngine.getViewport(panel.id) as any;
      if (!viewport || !voiRange) return;
      viewport.setProperties?.({ voiRange });
      viewport.render?.();
    });
    refreshSliceInfo();
  };

  const syncMprWindowFromViewport = (viewportId?: string) => {
    const renderingEngine = renderingEngineRef.current;
    if (!renderingEngine) return;
    const sourceViewport =
      (viewportId ? renderingEngine.getViewport(viewportId) : null) ||
      renderingEngine.getViewport('P0') ||
      renderingEngine.getViewport('P1') ||
      renderingEngine.getViewport('P2');
    const nextWindow = createMprWindowStateFromVoiRange((sourceViewport as any)?.getProperties?.()?.voiRange);
    if (!nextWindow) return;
    const current = mprWindowRef.current;
    if (
      current &&
      Math.abs(current.windowLevel - nextWindow.windowLevel) < 1 &&
      Math.abs(current.windowWidth - nextWindow.windowWidth) < 1
    ) {
      return;
    }
    setActiveMprWindow(nextWindow);
    updateViewportControlState({
      windowLevel: nextWindow.windowLevel,
      windowWidth: nextWindow.windowWidth,
    });

    const voiRange = createVoiRangeFromDisplayWindow(nextWindow);
    if (!voiRange) return;
    PANELS.forEach((panel) => {
      if (panel.plane === 'volume3d' || panel.id === viewportId) return;
      const viewport = renderingEngine.getViewport(panel.id) as any;
      viewport?.setProperties?.({ voiRange });
      viewport?.render?.();
    });
    refreshSliceInfo();
  };

  const renderPanel = (panel: MprPanelConfig, isMaximized = false) => {
    const panelSlabThicknessMm = getPanelSlabThickness(panel.id, slabThicknessByAxis, slabThicknessMm);

    return (
      <MprViewportPanel
        key={panel.id}
        panel={panel}
        hostRef={(el) => {
          viewportRefs.current[panel.id] = el;
        }}
        loading={loading}
        error={error}
        isMaximized={isMaximized}
        onToggleMaximize={(id) => setMaximizedPanel((current) => (current === id ? null : id))}
        seriesInfo={seriesInfo}
        sliceInfo={sliceInfo[panel.id]}
        projectionMode={projectionMode}
        slabThicknessMm={panelSlabThicknessMm}
        filterEnabled={filterEnabled}
        filterType={filterType}
        windowState={activeMprWindow}
        onWindowChange={(state) => applyMprWindow(state)}
        onWindowReset={() => applyMprWindow(originalMprWindowRef.current)}
        onProjectionModeChange={(mode) => {
          const nextSlabThicknessByAxis =
            mode === 'vr'
              ? DEFAULT_MPR_SLAB_THICKNESS_BY_AXIS
              : (mode === 'mip' || mode === 'minip') &&
                  Object.values(slabThicknessByAxis).every((thickness) => thickness <= 0)
                ? {
                    x: MPR_DEFAULT_SLAB_THICKNESS_MM,
                    y: MPR_DEFAULT_SLAB_THICKNESS_MM,
                    z: MPR_DEFAULT_SLAB_THICKNESS_MM,
                  }
                : slabThicknessByAxis;

          updateViewportControlState({
            projectionMode: mode,
            slabThicknessMm:
              mode === 'vr'
                ? 0
                : getPanelSlabThickness(panel.id, nextSlabThicknessByAxis, panelSlabThicknessMm),
            slabThicknessByAxis: nextSlabThicknessByAxis,
          });
        }}
        onSlabThicknessChange={(thicknessMm) => {
          const nextSlabThicknessByAxis =
            projectionMode === 'vr'
              ? DEFAULT_MPR_SLAB_THICKNESS_BY_AXIS
              : updatePanelSlabThickness(panel.id, slabThicknessByAxis, thicknessMm);
          updateViewportControlState({
            slabThicknessMm:
              projectionMode === 'vr'
                ? 0
                : getPanelSlabThickness(panel.id, nextSlabThicknessByAxis, thicknessMm),
            slabThicknessByAxis: nextSlabThicknessByAxis,
          });
        }}
        onFilterChange={(selection) =>
          updateViewportControlState({
            filterEnabled: hasMprFilterSelection(selection),
            filterType: selection,
          })
        }
        showHuOverlay={showHuOverlay}
        huOverlayInfo={huOverlayInfo[panel.id] ?? null}
        on2DPointerMove={handle2DPointerMove}
        on2DPointerLeave={handle2DPointerLeave}
        on3DPointerDown={handle3DPointerDown}
        on3DPointerMove={handle3DPointerMove}
        on3DPointerUp={handle3DPointerUp}
        on3DWheel={handle3DWheel}
      />
    );
  };

  return (
    <div className={`dental-mpr-workspace${maximizedPanel ? ' is-maximized' : ''}`} style={{ height }}>
      {PANELS.map((panel) => {
        const isMain = maximizedPanel === panel.id;
        const sideIndex = maximizedPanel ? PANELS.filter((targetPanel) => targetPanel.id !== maximizedPanel).findIndex(
          (targetPanel) => targetPanel.id === panel.id
        ) : -1;
        return (
          <div
            key={panel.id}
            className={`dental-mpr-cell${isMain ? ' is-main' : maximizedPanel ? ' is-side' : ''}`}
            data-side-index={sideIndex >= 0 ? sideIndex : undefined}
          >
            {renderPanel(panel, isMain)}
          </div>
        );
      })}
    </div>
  );
}

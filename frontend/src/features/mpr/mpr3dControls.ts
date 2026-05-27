export type Mpr3DTool = 'rotate' | 'panzoom';
export type Mpr2DTool = 'crosshair' | 'pan' | 'zoom' | 'wlww';

export type Mpr3DPreset =
  | 'dental-surface'
  | 'dental-bone'
  | 'dental-soft-tissue'
  | 'dental-mip'
  | 'vtk-bone1'
  | 'vtk-bone2'
  | 'vtk-mip';

export type Mpr3DControlState = {
  tool: Mpr3DTool;
  preset: Mpr3DPreset;
  resetToken: number;
};

export const MPR_3D_PRESETS: Array<{ value: Mpr3DPreset; label: string }> = [
  { value: 'dental-surface', label: 'Dental Surface' },
  { value: 'dental-bone', label: 'Dental Bone' },
  { value: 'dental-soft-tissue', label: 'Soft Tissue' },
  { value: 'dental-mip', label: 'Dental MIP' },
  { value: 'vtk-bone1', label: 'VTK Bone1' },
  { value: 'vtk-bone2', label: 'VTK Bone2' },
  { value: 'vtk-mip', label: 'VTK MIP' },
];

export type MprProjectionMode = 'mpr' | 'mip' | 'minip' | 'vr';
export type MprRenderingType = 'MPR' | 'MIP' | 'minIP' | 'VR';
export type MprRenderPath = 'single-plane' | 'svp' | 'volume-rendering';
export type MprSlabAxis = 'x' | 'y' | 'z';
export type MprFilterType =
  | '1x'
  | '1.5x'
  | '2x'
  | 'Blur'
  | 'Sharpen'
  | 'Gaussian'
  | 'Average'
  | 'Median';

export type MprFilterSelection = {
  first: MprFilterType | null;
  second: MprFilterType | null;
};

export type MprSlabThicknessByAxis = Record<MprSlabAxis, number>;

export type MprWindowState = {
  windowLevel: number;
  windowWidth: number;
};

export type MprViewportControlState = {
  projectionMode: MprProjectionMode;
  slabThicknessMm: number;
  slabThicknessByAxis: MprSlabThicknessByAxis;
  filterEnabled: boolean;
  filterType: MprFilterType | MprFilterSelection | null;
  windowLevel?: number | null;
  windowWidth?: number | null;
};

export type MprSlabPlane = {
  fs: number;
  fe: number;
  thicknessMm: number;
  renderingType: MprRenderingType;
  renderPath: MprRenderPath;
};

export function resolveMprRenderingType(mode: MprProjectionMode): MprRenderingType {
  if (mode === 'mip') return 'MIP';
  if (mode === 'minip') return 'minIP';
  if (mode === 'vr') return 'VR';
  return 'MPR';
}

export function createMprSlabPlane(mode: MprProjectionMode, slabThicknessMm: number): MprSlabPlane {
  const renderingType = resolveMprRenderingType(mode);
  const thicknessMm = renderingType === 'VR' ? 0 : Math.max(0, Number(slabThicknessMm) || 0);
  const halfThickness = thicknessMm / 2;
  const renderPath =
    renderingType === 'VR'
      ? 'volume-rendering'
      : thicknessMm > 0
        ? 'svp'
        : 'single-plane';

  return {
    fs: -halfThickness,
    fe: halfThickness,
    thicknessMm,
    renderingType,
    renderPath,
  };
}

export const MPR_PROJECTION_OPTIONS: Array<{ value: MprProjectionMode; label: string }> = [
  { value: 'mpr', label: 'MPR' },
  { value: 'mip', label: 'MIP' },
  { value: 'minip', label: 'minIP' },
  { value: 'vr', label: 'VR' },
];

export const MPR_SLAB_THICKNESS_OPTIONS = [0, 5, 10, 20, 30] as const;
export const MPR_DEFAULT_SLAB_THICKNESS_MM = 30;
export const DEFAULT_MPR_SLAB_THICKNESS_BY_AXIS: MprSlabThicknessByAxis = {
  x: 0,
  y: 0,
  z: 0,
};

export function createMprSlabThicknessByAxis(thicknessMm: number): MprSlabThicknessByAxis {
  const value = Math.max(0, Number(thicknessMm) || 0);
  return {
    x: value,
    y: value,
    z: value,
  };
}

export function normalizeMprSlabThicknessByAxis(
  slabThicknessByAxis: Partial<MprSlabThicknessByAxis> | null | undefined,
  fallbackThicknessMm = 0
): MprSlabThicknessByAxis {
  const fallback = Math.max(0, Number(fallbackThicknessMm) || 0);
  return {
    x: Math.max(0, Number(slabThicknessByAxis?.x ?? fallback) || 0),
    y: Math.max(0, Number(slabThicknessByAxis?.y ?? fallback) || 0),
    z: Math.max(0, Number(slabThicknessByAxis?.z ?? fallback) || 0),
  };
}

export const MPR_FILTER_OPTIONS: MprFilterType[] = [
  '1x',
  '1.5x',
  '2x',
  'Blur',
  'Sharpen',
  'Gaussian',
  'Average',
  'Median',
];

export const EMPTY_MPR_FILTER_SELECTION: MprFilterSelection = {
  first: null,
  second: null,
};

export function normalizeMprFilterSelection(
  filterType: MprFilterType | MprFilterSelection | null | undefined
): MprFilterSelection {
  if (!filterType) return EMPTY_MPR_FILTER_SELECTION;
  if (typeof filterType === 'string') {
    return { first: filterType, second: null };
  }
  return {
    first: filterType.first ?? null,
    second: filterType.second ?? null,
  };
}

export function hasMprFilterSelection(filterType: MprFilterType | MprFilterSelection | null | undefined) {
  const selection = normalizeMprFilterSelection(filterType);
  return Boolean(selection.first || selection.second);
}

export function formatMprFilterSelection(filterType: MprFilterType | MprFilterSelection | null | undefined) {
  const selection = normalizeMprFilterSelection(filterType);
  return [selection.first, selection.second].filter(Boolean).join(',');
}

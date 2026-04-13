type NumericArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export type AutoWindowResult = {
  width: number;
  level: number;
  thresholds: number[];
  softTissueStdDev: number;
  boneStdDev: number;
};

const autoWindowCache = new Map<string, AutoWindowResult>();
const DEFAULT_THRESHOLD_RATIOS = [0.005, 0.12, 0.28, 0.45, 0.68, 0.88, 0.995];
const MAX_SORTED_SAMPLES = 32768;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const toFiniteNumber = (value: unknown, fallback: number) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

const sampleScalarData = (source: NumericArray, sampleLimit = MAX_SORTED_SAMPLES) => {
  if (source.length <= sampleLimit) {
    return Array.from(source, (value) => Number(value));
  }

  const stride = Math.max(1, Math.floor(source.length / sampleLimit));
  const sampled: number[] = [];
  for (let index = 0; index < source.length; index += stride) {
    sampled.push(Number(source[index]));
  }
  return sampled;
};

const percentileFromSorted = (sorted: number[], ratio: number) => {
  if (sorted.length === 0) return 0;
  const clampedRatio = clamp(ratio, 0, 1);
  const position = clampedRatio * (sorted.length - 1);
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  const lower = sorted[lowerIndex];
  const upper = sorted[upperIndex];
  if (lowerIndex === upperIndex) return lower;
  const weight = position - lowerIndex;
  return lower + (upper - lower) * weight;
};

const computeStdDev = (values: number[]) => {
  if (values.length <= 1) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => {
    const delta = value - mean;
    return sum + delta * delta;
  }, 0) / values.length;
  return Math.sqrt(Math.max(variance, 0));
};

export function calculateAutoWindow(
  thresholds: number[],
  softTissueStdDev: number,
  boneStdDev: number
): AutoWindowResult | null {
  if (!thresholds || thresholds.length < 7) {
    return null;
  }

  const expandWidth = (thresholds[3] - thresholds[1]) * 0.15 + softTissueStdDev;
  const wndLowerBound = Math.round(thresholds[1] - expandWidth);

  let wndUpperBound = Math.round(thresholds[5] - boneStdDev * 0.75);
  if (wndUpperBound > thresholds[6]) {
    wndUpperBound = Math.round(thresholds[6]);
  }

  const diff = wndUpperBound - wndLowerBound + 1;
  const width = Math.max(1, Math.round(diff + 0.5));
  const level = Math.round(wndLowerBound + Math.floor(diff / 2));

  return {
    width,
    level,
    thresholds,
    softTissueStdDev,
    boneStdDev,
  };
}

export function estimateAutoWindowFromPixelData(
  scalarData: NumericArray,
  slope = 1,
  intercept = 0
): AutoWindowResult | null {
  if (!scalarData || scalarData.length < 16) {
    return null;
  }

  const sampled = sampleScalarData(scalarData)
    .map((value) => value * slope + intercept)
    .filter((value) => Number.isFinite(value));

  if (sampled.length < 16) {
    return null;
  }

  const sorted = sampled.slice().sort((a, b) => a - b);
  const thresholds = DEFAULT_THRESHOLD_RATIOS.map((ratio) => percentileFromSorted(sorted, ratio));

  const softWindowStart = thresholds[1];
  const softWindowEnd = thresholds[3];
  const boneWindowStart = thresholds[4];
  const boneWindowEnd = thresholds[6];

  const softTissueBucket = sampled.filter((value) => value >= softWindowStart && value <= softWindowEnd);
  const boneBucket = sampled.filter((value) => value >= boneWindowStart && value <= boneWindowEnd);

  const softTissueStdDev = computeStdDev(softTissueBucket);
  const boneStdDev = computeStdDev(boneBucket);

  const autoWindow = calculateAutoWindow(thresholds, softTissueStdDev, boneStdDev);
  if (!autoWindow) {
    return null;
  }

  const globalLow = percentileFromSorted(sorted, 0.005);
  const globalHigh = percentileFromSorted(sorted, 0.995);
  const safeLower = clamp(autoWindow.level - autoWindow.width / 2, globalLow, globalHigh);
  const safeUpper = clamp(autoWindow.level + autoWindow.width / 2, globalLow, globalHigh);
  const safeWidth = Math.max(1, Math.round(safeUpper - safeLower));
  const safeLevel = Math.round(safeLower + safeWidth / 2);

  return {
    ...autoWindow,
    width: safeWidth,
    level: safeLevel,
    thresholds: thresholds.map((value) => toFiniteNumber(value, 0)),
    softTissueStdDev: toFiniteNumber(softTissueStdDev, 0),
    boneStdDev: toFiniteNumber(boneStdDev, 0),
  };
}

export function getAutoWindowForCornerstoneImage(image: {
  imageId?: string;
  slope?: number;
  intercept?: number;
  getPixelData?: () => unknown;
} | null | undefined) {
  if (!image) return null;

  const cacheKey = image.imageId || '';
  if (cacheKey && autoWindowCache.has(cacheKey)) {
    return autoWindowCache.get(cacheKey)!;
  }

  const pixelData = image.getPixelData?.();
  if (
    !(pixelData instanceof Int8Array) &&
    !(pixelData instanceof Uint8Array) &&
    !(pixelData instanceof Uint8ClampedArray) &&
    !(pixelData instanceof Int16Array) &&
    !(pixelData instanceof Uint16Array) &&
    !(pixelData instanceof Int32Array) &&
    !(pixelData instanceof Uint32Array) &&
    !(pixelData instanceof Float32Array) &&
    !(pixelData instanceof Float64Array)
  ) {
    return null;
  }

  const autoWindow = estimateAutoWindowFromPixelData(
    pixelData,
    toFiniteNumber(image.slope, 1),
    toFiniteNumber(image.intercept, 0)
  );

  if (autoWindow && cacheKey) {
    autoWindowCache.set(cacheKey, autoWindow);
  }

  return autoWindow;
}

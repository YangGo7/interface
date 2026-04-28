import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import dicomParser from 'dicom-parser';
import O3Logo from '../assets/O3_logo_only.png';
import { DentalChartLegendOverlay } from '../components/chart/DentalChartLegendOverlay';
import { OutputCapturePanel } from '../components/chart/OutputCapturePanel';
import { RenewReportWorkspacePanel } from '../components/chart/RenewReportWorkspacePanel';
import { RenewStudiesDock } from '../components/chart/RenewStudiesDock';
import { RenewToolSubmenu } from '../components/chart/RenewToolSubmenu';
import { ReportWorkspaceControls } from '../components/chart/ReportWorkspaceControls';
import { ToothHoverHud } from '../components/chart/ToothHoverHud';
import { WebReportDrawer } from '../components/WebReportDrawer';
import { readStoredNumberingSystem, writeStoredNumberingSystem } from '../lib/appSettings';
import { createWebReportFromChart, patchWebReportOverrides } from '../lib/webReportApi';
import {
  fetchServerFolderIndex,
  materializeServerStudy,
  resolveServerAssetUrl,
  type ServerFolderImage,
} from '../lib/folderLeaderApi';
import type { FolderStudy } from '../features/upload/dicomFolderStudies';
import { requestAsyncDetection } from '../features/upload/uploadApi';
import { clearAllAnnotations, setActiveTool as setCornerstoneActiveTool } from '../viewer/cornerstone/tools';
import { estimateAutoWindowFromPixelData } from '../viewer/cornerstone/autoWindow';
import { parseDicomMetadataFromDataSet, type DicomOverlayMetadata } from '../viewer/cornerstone/dicomMetadata';
import { inspectLocalDicomFile } from '../viewer/cornerstone/dicomDebug';
import { DicomMetadataOverlay } from '../viewer/DicomMetadataOverlay';

const DESIGN_WIDTH = 1920;
const DESIGN_HEIGHT = 1080;

const wp = (value: number) => `${value}px`;
const hp = (value: number) => `${value}px`;
const scalePx = (value: number) => `${value}px`;
const relativePercent = (value: number, total: number) => `${(value / total) * 100}%`;
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const buildSmoothPath = (pts: ImagePoint[], close = false) => {
  if (pts.length === 0) return '';
  if (pts.length === 1) return `M ${pts[0].x} ${pts[0].y}`;
  if (pts.length === 2) {
    const d = `M ${pts[0].x} ${pts[0].y} L ${pts[1].x} ${pts[1].y}`;
    return close ? `${d} Z` : d;
  }

  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 1; i < pts.length - 1; i += 1) {
    const midX = (pts[i].x + pts[i + 1].x) / 2;
    const midY = (pts[i].y + pts[i + 1].y) / 2;
    d += ` Q ${pts[i].x} ${pts[i].y} ${midX} ${midY}`;
  }

  const last = pts[pts.length - 1];
  d += ` T ${last.x} ${last.y}`;
  return close ? `${d} Z` : d;
};
const resolveDirectApiBase = () => {
  const configured = ((import.meta as any)?.env?.VITE_API_BASE_URL as string | undefined)?.trim();
  if (configured) return configured;
  if (typeof window === 'undefined') return 'http://localhost:5000';

  const { protocol, hostname, port, origin } = window.location;
  const isLocalDevHost = hostname === 'localhost' || hostname === '127.0.0.1';
  if (isLocalDevHost && (port === '3000' || port === '5173')) {
    return `${protocol}//${hostname}:5000`;
  }
  return origin;
};

const DIRECT_API_BASE = resolveDirectApiBase();

const assetPath = (relativePath: string) => encodeURI(`/imgs/${relativePath}`);
const headerMarkerIconRelativePath = '7 7.png';
const headerMarkerIcon = assetPath(headerMarkerIconRelativePath);
const PANO_LENS_SIZE = 240;
const PANO_LENS_ZOOM = 2.25;
const PANO_LENS_EDGE_PADDING = 10;
const AI_CONFIDENCE_THRESHOLD = 0.35;
const PANO_DEFAULT_BRIGHTNESS = 88;
const PANO_DEFAULT_CONTRAST = 100;
const heatmapOpacityPresets = [30, 50, 70];
const SERVER_IMAGE_SERIES_PREFIX = 'server-image:';

const loadImageFromUrl = (src: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const nextImage = new Image();
    nextImage.decoding = 'async';
    nextImage.onload = () => resolve(nextImage);
    nextImage.onerror = () => reject(new Error('Failed to load overlay image'));
    nextImage.src = src;
  });

const withDirectApiBase = (path: string) => {
  if (/^https?:\/\//i.test(path)) return path;
  return `${DIRECT_API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
};

const isDicomPath = (url?: string | null) => !!url && /\.(dcm|dicom)(?:$|[?#])/i.test(url);

const deriveDisplayWindowFromControls = (
  baseCenter: number,
  baseWidth: number,
  brightnessPercent: number,
  contrastPercent: number
) => {
  const safeBaseWidth = Math.max(1, baseWidth || 1);
  const safeContrast = Math.max(1, contrastPercent || 100);
  const windowWidth = Math.max(1, Math.round(safeBaseWidth * (100 / safeContrast)));
  const windowCenter = Math.round(baseCenter + ((brightnessPercent - 100) / 100) * safeBaseWidth);
  return { windowCenter, windowWidth };
};

const deriveControlsForWindow = (
  baseCenter: number,
  baseWidth: number,
  targetCenter: number,
  targetWidth: number
) => {
  const safeBaseWidth = Math.max(1, baseWidth || 1);
  const safeTargetWidth = Math.max(1, targetWidth || 1);
  const contrastPercent = Math.max(1, Math.min(300, Math.round((safeBaseWidth / safeTargetWidth) * 100)));
  const brightnessPercent = Math.max(0, Math.min(300, Math.round(100 + ((targetCenter - baseCenter) / safeBaseWidth) * 100)));
  return { brightnessPercent, contrastPercent };
};

const readJsonOrThrow = async <T,>(response: Response): Promise<T> => {
  const contentType = response.headers.get('content-type') || '';
  const raw = await response.text();
  const trimmed = raw.trim();

  if (!contentType.includes('application/json') && trimmed.startsWith('<')) {
    throw new Error(
      `Expected JSON but received HTML from ${response.url || 'request'}. Check the API route/proxy.`
    );
  }

  try {
    return (raw ? JSON.parse(raw) : {}) as T;
  } catch {
    throw new Error(
      `Failed to parse JSON from ${response.url || 'request'} (${contentType || 'unknown content-type'}).`
    );
  }
};

const reportButtonIcons = {
  inactive: assetPath('botton/report.png'),
  active: assetPath('botton/report click.png'),
};

const railIcons = {
  studies: assetPath('메인-클릭/좌측 bar 아이콘 (51 57)/자산 425@4x.png'),
  captures: assetPath('메인-비활성화 아이콘/좌측 bar 아이콘 (51 57)/자산 584.png'),
  report: assetPath('메인-비활성화 아이콘/report버튼 (94 94).png'),
};

const toolbarIcons = [
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 20@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 21@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 22@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 23@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 24@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 26@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 31@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 30@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 25@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 29@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 28@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 27@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 35@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 34@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 33@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 32@4x.png'),
];

const displayRailIcons = {
  studies: assetPath('mian_deactive/left_bar_deactive (51 57)/자산 427@4x.png'),
  captures: assetPath('mian_deactive/left_bar_deactive (51 57)/자산 584.png'),
};

const activeRailIcons = {
  studies: assetPath('main_active/left_bar_active (51 57)/자산 425@4x.png'),
  captures: assetPath('main_active/left_bar_active (51 57)/자산 587.png'),
};

const panoZoomIcons = {
  maximize: assetPath('botton/maxima642@4x.png'),
  minimize: assetPath('botton/minima.png'),
};

const displayToolbarIcons = [
  assetPath('mian_deactive/tools_deactive(36 36)/자산 20@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 21@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 22@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 23@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 24@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 26@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 31@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 30@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 25@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 29@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 28@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 27@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 35@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 34@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 33@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 32@4x.png'),
];

const activeToolbarIcons = [
  assetPath('main_active/tools_active (36 36)/자산 399@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 401@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 400@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 398@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 397@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 391@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 396@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 395@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 386@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 394@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 393@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 392@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 390@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 389@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 388@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 387@4x.png'),
];

const displayReportButtonIcons = {
  inactive: assetPath('botton/report.png'),
  active: assetPath('botton/report click.png'),
};

const legendItems = [
  { key: 'warning', label: 'Urgent Priority', color: '#FF0037', top: 834 },
  { key: 'requires', label: 'Treatment Required', color: '#FCFF2A', top: 867 },
  { key: 'implant', label: 'Implant', color: '#003DFF', top: 900 },
  { key: 'missing', label: 'Missing Tooth', color: '#3F3F3F', top: 933 },
  { key: 'healthy', label: 'Healthy Tooth', color: '#FFFFFF', top: 966 },
] as const;

const overlayPresetOptions = [
  { key: 'all', label: '전체', menuLabel: '1. all' },
  { key: 'sinus', label: 'sinus', menuLabel: '2. sinus' },
  { key: 'nerve', label: 'nerve', menuLabel: '3. nerve' },
  { key: 'tooth', label: 'tooth', menuLabel: '4. tooth' },
  { key: 'sinus-upper-tooth', label: 'sinus + upper tooth', menuLabel: '5. sinus + upper tooth' },
  { key: 'nerve-lower-tooth', label: 'nerve + lower tooth', menuLabel: '6. nerve + lower tooth' },
] as const;

const upperSizes = ['23 79', '20 75', '22 82', '25 74', '25 67', '38 58', '36 54', '35 49'];
const lowerSizes = ['18 71', '17 70', '22 78', '25 75', '26 76', '39 68', '38 65', '41 63'];
const upperLeftOrders = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const upperRightOrders = [1, 2, 3, 4, 5, 6, 7, 8] as const;
const lowerLeftOrders = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const lowerRightOrders = [1, 2, 3, 4, 5, 6, 7, 8] as const;
const upperTeeth = [18, 17, 16, 15, 14, 13, 12, 11] as const;
const upperRightTeeth = [21, 22, 23, 24, 25, 26, 27, 28] as const;
const lowerTeeth = [48, 47, 46, 45, 44, 43, 42, 41] as const;
const lowerRightTeeth = [31, 32, 33, 34, 35, 36, 37, 38] as const;
const odontogramTeeth = [
  ...upperTeeth,
  ...upperRightTeeth,
  ...lowerTeeth,
  ...lowerRightTeeth,
] as const;
type NumberingSystem = 'fdi' | 'univ';
const UNIVERSAL_TOOTH_MAP: Record<number, number> = {
  18: 1, 17: 2, 16: 3, 15: 4, 14: 5, 13: 6, 12: 7, 11: 8,
  21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
  38: 17, 37: 18, 36: 19, 35: 20, 34: 21, 33: 22, 32: 23, 31: 24,
  41: 25, 42: 26, 43: 27, 44: 28, 45: 29, 46: 30, 47: 31, 48: 32,
};

const formatToothNumber = (toothFdi: string | number, numberingSystem: NumberingSystem) => {
  const numeric = Number(toothFdi);
  if (numberingSystem !== 'univ' || !Number.isFinite(numeric)) return String(toothFdi);
  return String(UNIVERSAL_TOOTH_MAP[numeric] || numeric);
};

const formatDisplayedToothNumber = (
  toothFdi: string | number,
  numberingSystem: NumberingSystem,
  flipped = false
) => {
  const displayNumber = formatToothNumber(toothFdi, numberingSystem);
  return flipped ? displayNumber.split('').reverse().join('') : displayNumber;
};

const getDisplayedArchTeeth = (
  flipped: boolean,
  leftTeeth: readonly number[],
  rightTeeth: readonly number[]
) => {
  if (!flipped) {
    return {
      left: [...leftTeeth],
      right: [...rightTeeth],
    };
  }

  return {
    left: [...rightTeeth].reverse(),
    right: [...leftTeeth].reverse(),
  };
};

const outerToInnerOffsets = [390.5, 332.5, 274.5, 216.5, 166.5, 116.5, 66.5, 22.5] as const;
const innerToOuterOffsets = [22.5, 66.5, 116.5, 166.5, 216.5, 274.5, 332.5, 390.5] as const;
const upperBaseline = 935;
const lowerTop = 953;
const chartSectionTop = 804;
const chartHeaderHeight = 18;
const chartContentTop = chartSectionTop + chartHeaderHeight;
const chartContentHeight = 246;
const chartLegendLeft = 222;
const chartLegendTop = 844;
const chartLegendHeight = 151;
const RAIL_ICON_WIDTH = 61;
const RAIL_ICON_HEIGHT = 68;
const TOOL_ICON_SIZE = 43;

const buildLowerContourRuns = (contour: any[], lowerRatio = 0.25) => {
  if (!Array.isArray(contour) || contour.length < 3) return [];

  const points = contour
    .map((pt: any) =>
      Array.isArray(pt) && pt.length >= 2
        ? { x: Number(pt[0]), y: Number(pt[1]) }
        : null
    )
    .filter((pt): pt is { x: number; y: number } => pt !== null && Number.isFinite(pt.x) && Number.isFinite(pt.y));

  if (points.length < 3) return [];

  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  points.forEach((pt) => {
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  });

  const height = maxY - minY;
  if (height <= 0) return [];

  const cutoffY = maxY - height * lowerRatio;
  const runs: Array<Array<{ x: number; y: number }>> = [];
  let currentRun: Array<{ x: number; y: number }> = [];

  points.forEach((pt) => {
    if (pt.y >= cutoffY) {
      currentRun.push(pt);
      return;
    }

    if (currentRun.length >= 2) runs.push(currentRun);
    currentRun = [];
  });

  if (currentRun.length >= 2) runs.push(currentRun);

  if (runs.length > 1) {
    const first = runs[0];
    const last = runs[runs.length - 1];
    if (first[0] && last[last.length - 1] && points[0].y >= cutoffY && points[points.length - 1].y >= cutoffY) {
      runs[0] = [...last, ...first];
      runs.pop();
    }
  }

  return runs;
};

const warmPastelPalette = [
  { fill: 'rgba(255, 216, 194, 0.26)', stroke: 'rgba(255, 182, 151, 0.30)' },
  { fill: 'rgba(255, 226, 203, 0.26)', stroke: 'rgba(241, 174, 129, 0.30)' },
  { fill: 'rgba(255, 212, 209, 0.26)', stroke: 'rgba(234, 153, 150, 0.30)' },
  { fill: 'rgba(246, 224, 198, 0.26)', stroke: 'rgba(221, 171, 118, 0.30)' },
  { fill: 'rgba(255, 229, 214, 0.26)', stroke: 'rgba(232, 163, 121, 0.30)' },
  { fill: 'rgba(255, 239, 221, 0.26)', stroke: 'rgba(227, 180, 126, 0.30)' },
] as const;

type ImagePoint = { x: number; y: number };

type ToothGeometry = {
  fdi: string;
  contour: ImagePoint[];
  centroid: ImagePoint;
  bounds: { x1: number; y1: number; x2: number; y2: number };
};

type ToothCondition = 'healthy' | 'requires' | 'warning' | 'implant' | 'missing';
type OverlayPreset = typeof overlayPresetOptions[number]['key'];

type NormalizedDetection = {
  id: string;
  type: string;
  label: string;
  confidence: number;
  toothFdi: string | null;
  contour: ImagePoint[] | null;
  bounds: { x1: number; y1: number; x2: number; y2: number };
  center: ImagePoint;
  source: any;
};

type ToothHoverAnchor = {
  toothFdi: string;
  x: number;
  y: number;
};

type ToothHoverPanelData = {
  toothFdi: string;
  title: string;
  kind: 'finding' | 'implant' | 'planning';
  status: string;
  pblPct?: number | null;
  level?: number | string | null;
  cariesProb?: number | null;
  periapicalProb?: number | null;
  primaryLabel?: string | null;
  primaryProb?: number | null;
  diameterMm?: number | null;
  lengthMm?: number | null;
  gapMm?: number | null;
  centerToNerveMm?: number | null;
};

const toImagePoint = (value: any): ImagePoint | null => {
  if (Array.isArray(value) && value.length >= 2) {
    const x = Number(value[0]);
    const y = Number(value[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) return { x, y };
  }

  if (value && typeof value === 'object') {
    const x = Number(value.x ?? value.cx ?? value.left);
    const y = Number(value.y ?? value.cy ?? value.top);
    if (Number.isFinite(x) && Number.isFinite(y)) return { x, y };
  }

  return null;
};

const normalizeContourPoints = (value: any): ImagePoint[] => {
  if (!Array.isArray(value)) return [];
  return value.map(toImagePoint).filter((point): point is ImagePoint => point !== null);
};

const boundsFromPoints = (points: ImagePoint[]) => {
  if (!points.length) return null;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    x1: Math.min(...xs),
    y1: Math.min(...ys),
    x2: Math.max(...xs),
    y2: Math.max(...ys),
  };
};

const centroidFromPoints = (points: ImagePoint[]) => {
  if (!points.length) return { x: 0, y: 0 };
  const total = points.reduce(
    (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }),
    { x: 0, y: 0 }
  );
  return { x: total.x / points.length, y: total.y / points.length };
};

const centerFromBounds = (bounds: { x1: number; y1: number; x2: number; y2: number }) => ({
  x: (bounds.x1 + bounds.x2) / 2,
  y: (bounds.y1 + bounds.y2) / 2,
});

const estimateImplantMetricsFromContour = (contour: any, mmPerPx: number) => {
  if (!Array.isArray(contour) || contour.length < 3 || !Number.isFinite(mmPerPx) || mmPerPx <= 0) {
    return { diameterMm: null, lengthMm: null };
  }

  const points = contour
    .map((pt: any) => Array.isArray(pt) && pt.length >= 2 ? [Number(pt[0]), Number(pt[1])] : null)
    .filter((pt: number[] | null): pt is number[] => {
      if (!pt) return false;
      return Number.isFinite(pt[0]) && Number.isFinite(pt[1]);
    });

  if (points.length < 3) {
    return { diameterMm: null, lengthMm: null };
  }

  const meanX = points.reduce((sum, [x]) => sum + x, 0) / points.length;
  const meanY = points.reduce((sum, [, y]) => sum + y, 0) / points.length;
  let xx = 0;
  let xy = 0;
  let yy = 0;
  points.forEach(([x, y]) => {
    const dx = x - meanX;
    const dy = y - meanY;
    xx += dx * dx;
    xy += dx * dy;
    yy += dy * dy;
  });

  const trace = xx + yy;
  const det = xx * yy - xy * xy;
  const eigen = trace / 2 + Math.sqrt(Math.max(0, (trace * trace) / 4 - det));
  let axisX = xy;
  let axisY = eigen - xx;
  if (Math.abs(axisX) < 1e-6 && Math.abs(axisY) < 1e-6) {
    axisX = xx >= yy ? 1 : 0;
    axisY = xx >= yy ? 0 : 1;
  }
  const axisNorm = Math.hypot(axisX, axisY) || 1;
  const ux = axisX / axisNorm;
  const uy = axisY / axisNorm;
  const vx = -uy;
  const vy = ux;

  const axisProjections = points.map(([x, y]) => x * ux + y * uy);
  const perpProjections = points.map(([x, y]) => x * vx + y * vy);
  const lengthPx = Math.max(...axisProjections) - Math.min(...axisProjections);
  const diameterPx = Math.max(...perpProjections) - Math.min(...perpProjections);

  if (lengthPx <= 0 || diameterPx <= 0) {
    return { diameterMm: null, lengthMm: null };
  }

  return {
    diameterMm: diameterPx * mmPerPx,
    lengthMm: lengthPx * mmPerPx,
  };
};

const normalizeBounds = (value: any) => {
  if (Array.isArray(value) && value.length >= 4) {
    const [x1, y1, x2, y2] = value.map(Number);
    if ([x1, y1, x2, y2].every(Number.isFinite)) {
      return {
        x1: Math.min(x1, x2),
        y1: Math.min(y1, y2),
        x2: Math.max(x1, x2),
        y2: Math.max(y1, y2),
      };
    }
  }

  if (value && typeof value === 'object') {
    const x = Number(value.x ?? value.left);
    const y = Number(value.y ?? value.top);
    const width = Number(value.width ?? value.w);
    const height = Number(value.height ?? value.h);
    if ([x, y, width, height].every(Number.isFinite)) {
      return { x1: x, y1: y, x2: x + width, y2: y + height };
    }
  }

  return null;
};

const flattenContourPoints = (contours: any): ImagePoint[] =>
  (Array.isArray(contours) ? contours : []).flatMap((contour: any) => normalizeContourPoints(contour));

const distanceBetweenPoints = (a: ImagePoint, b: ImagePoint) => Math.hypot(a.x - b.x, a.y - b.y);

const pointInPolygon = (point: ImagePoint, polygon: ImagePoint[]) => {
  let isInside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersect =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / Math.max(yj - yi, 1e-6) + xi;
    if (intersect) isInside = !isInside;
  }
  return isInside;
};

const squaredDistance = (a: ImagePoint, b: ImagePoint) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;

const boundsArea = (bounds: { x1: number; y1: number; x2: number; y2: number }) =>
  Math.max(0, bounds.x2 - bounds.x1) * Math.max(0, bounds.y2 - bounds.y1);

const intersectionArea = (
  a: { x1: number; y1: number; x2: number; y2: number },
  b: { x1: number; y1: number; x2: number; y2: number }
) => {
  const width = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const height = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  return width * height;
};

const distancePointToSegment = (point: ImagePoint, start: ImagePoint, end: ImagePoint) => {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) return distanceBetweenPoints(point, start);
  const t = clamp(((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy), 0, 1);
  const projection = { x: start.x + t * dx, y: start.y + t * dy };
  return distanceBetweenPoints(point, projection);
};

const getToothLabel = (value: any) => {
  const raw = String(
    value?.assigned_tooth ??
      value?.tooth_label ??
      value?.tooth ??
      value?.fdi ??
      value?.label ??
      ''
  ).trim();
  return /^\d{2}$/.test(raw) ? raw : null;
};

const getDetectionConfidence = (value: any) => {
  const confidence = Number(value?.conf ?? value?.confidence ?? value?.score ?? value?.probability ?? 0);
  return Number.isFinite(confidence) ? confidence : 0;
};

const getBestConfidence = (bestMap: any, rawMap: any, key: string) => {
  const item = bestMap?.[key] || rawMap?.[key];
  const confidence = Number(item?.conf ?? item?.confidence);
  return Number.isFinite(confidence) ? confidence : null;
};

const readOverlayDimension = (source: any, keys: string[]) => {
  for (const key of keys) {
    const value = Number(source?.[key]);
    if (Number.isFinite(value) && value > 0) return value;
  }
  return null;
};

const getToothId = (value: any) => {
  if (typeof value === 'number' || typeof value === 'string') {
    const raw = String(value).trim();
    return /^\d{2}$/.test(raw) ? raw : null;
  }

  if (value && typeof value === 'object') {
    return getToothLabel(value);
  }

  return null;
};

const inferToothFdi = (
  explicitTooth: string | null,
  center: ImagePoint,
  toothGeometries: ToothGeometry[],
  contour?: ImagePoint[] | null,
  bounds?: { x1: number; y1: number; x2: number; y2: number } | null
) => {
  if (explicitTooth) return explicitTooth;
  if (!toothGeometries.length) return null;

  if (contour && contour.length >= 3) {
    const bestOverlapTooth = toothGeometries.reduce(
      (best, tooth) => {
        const insideCount = contour.reduce((count, point) => count + (pointInPolygon(point, tooth.contour) ? 1 : 0), 0);
        const overlapRatio =
          bounds && tooth.bounds
            ? intersectionArea(bounds, tooth.bounds) / Math.max(1, boundsArea(bounds))
            : 0;
        if (insideCount <= 0 && overlapRatio <= 0) return best;
        const score = insideCount * 1000 + overlapRatio * 100 - squaredDistance(center, tooth.centroid) * 0.0001;
        if (!best || score > best.score) {
          return { tooth, score };
        }
        return best;
      },
      null as { tooth: ToothGeometry; score: number } | null
    );

    if (bestOverlapTooth) return bestOverlapTooth.tooth.fdi;
  }

  const containingTooth = toothGeometries.find((tooth) => pointInPolygon(center, tooth.contour));
  if (containingTooth) return containingTooth.fdi;

  if (bounds) {
    const bestBoundsTooth = toothGeometries.reduce(
      (best, tooth) => {
        const overlap = intersectionArea(bounds, tooth.bounds);
        if (overlap <= 0) return best;
        if (!best || overlap > best.overlap) {
          return { tooth, overlap };
        }
        return best;
      },
      null as { tooth: ToothGeometry; overlap: number } | null
    );

    if (bestBoundsTooth) return bestBoundsTooth.tooth.fdi;
  }

  return toothGeometries.reduce((closest, tooth) => {
    if (!closest) return tooth;
    return squaredDistance(center, tooth.centroid) < squaredDistance(center, closest.centroid) ? tooth : closest;
  }, null as ToothGeometry | null)?.fdi ?? null;
};

type ToolbarKey =
  | 'pointer'
  | 'pan'
  | 'wlww'
  | 'invert'
  | 'magnifier'
  | 'flip'
  | 'measure-length'
  | 'measure-draw'
  | 'measure-eraser'
  | 'measure-clear'
  | 'measure-rotate'
  | 'measure-reset'
  | 'output-capture'
  | 'output-save'
  | 'output-report'
  | 'task-original'
  | 'task-heatmap';

type MeasureMenuKey = 'measure' | 'annotate';

type MeasureSubtoolKey =
  | 'length'
  | 'bidirectional'
  | 'angle'
  | 'text'
  | 'arrow'
  | 'ellipse'
  | 'rect'
  | 'circle'
  | 'roi-free'
  | 'spline-roi'
  | 'livewire';

type MeasureShape = {
  type: MeasureSubtoolKey;
  points: ImagePoint[];
  text?: string;
  anchorLeft?: number;
  anchorTop?: number;
};

type OutputCaptureItem = {
  id: string;
  dataUrl: string;
  createdAt: number;
  label?: string;
  size?: string;
  note?: string;
};

function getToothAsset(tooth: number) {
  const index = (tooth % 10) - 1;
  const upper = tooth >= 11 && tooth <= 28;
  const prefix = upper ? 'U' : 'L';
  const size = upper ? upperSizes[index] : lowerSizes[index];
  return assetPath(`치아/건강치(ffffff)/${prefix}-${index + 1} (${size})_4.png`);
}

function ToolIcon({
  icon,
  activeIcon,
  left,
  top,
  active = false,
  onClick,
  label,
}: {
  icon: string;
  activeIcon?: string;
  left: number;
  top: number;
  active?: boolean;
  onClick?: () => void;
  label?: string;
}) {
  return (
    <button
      type="button"
      onPointerDown={(event) => {
        event.stopPropagation();
      }}
      onClick={onClick}
      aria-pressed={active}
      aria-label={label}
      title={label}
      style={{
        width: wp(TOOL_ICON_SIZE),
        height: hp(TOOL_ICON_SIZE),
        left: wp(left),
        top: hp(top),
        position: 'absolute',
        border: `${scalePx(1)} solid ${active ? '#00C0F3' : 'black'}`,
        background: active ? '#A8A8A8' : '#8D8D8D',
        boxShadow: active ? `0 0 ${scalePx(8)} rgba(0, 192, 243, 0.35)` : 'none',
        cursor: 'pointer',
        padding: 0,
      }}
    >
      <img
        src={active && activeIcon ? activeIcon : icon}
        alt=""
        draggable={false}
        style={{ width: '100%', height: '100%', position: 'absolute', left: 0, top: 0 }}
      />
    </button>
  );
}

function ToothImage({
  tooth,
  left,
  top,
  width,
  height,
  rotate = false,
}: {
  tooth: number;
  left: number;
  top: number;
  width: number;
  height: number;
  rotate?: boolean;
}) {
  return (
    <img
      src={getToothAsset(tooth)}
      alt=""
      draggable={false}
      style={{
        width: wp(width),
        height: hp(height),
        left: wp(left),
        top: hp(top),
        position: 'absolute',
        transform: rotate ? `translate(${width}px, ${height}px) rotate(180deg)` : undefined,
        transformOrigin: rotate ? 'top left' : undefined,
      }}
    />
  );
}

function getToothStatusAsset(arch: 'U' | 'L', order: number, status: ToothCondition) {
  const size = arch === 'U' ? upperSizes[order - 1] : lowerSizes[order - 1];
  if (status === 'implant') {
    return assetPath('teeth/implant (003dff)/28 54.png');
  }
  if (status === 'warning') {
    return assetPath(`teeth/warning (ff0037)/${arch}-${order} (${size})_2.png`);
  }
  if (status === 'requires') {
    return assetPath(`teeth/notice (fcff2a)/${arch}-${order} (${size})_3.png`);
  }
  if (status === 'missing' && arch === 'U') {
    return assetPath(`teeth/missing (3f3f3f)/U-${order}.png`);
  }
  if (status === 'missing' && arch === 'L') {
    return assetPath(`teeth/missing (3f3f3f)/${arch}-${order} (${size})@4x.png`);
  }
  return assetPath(`teeth/health(ffffff)/${arch}-${order} (${size})_4.png`);
}

function ToothSlotImage({
  arch,
  order,
  toothFdi,
  status,
  left,
  top,
  width,
  height,
  flipX = false,
  active = false,
  hasDetection = false,
  dimmed = false,
  onClick,
  onHoverChange,
}: {
  arch: 'U' | 'L';
  order: number;
  toothFdi: string;
  status: ToothCondition;
  left: number;
  top: number;
  width: number;
  height: number;
  flipX?: boolean;
  active?: boolean;
  hasDetection?: boolean;
  dimmed?: boolean;
  onClick?: (toothFdi: string) => void;
  onHoverChange?: (value: ToothHoverAnchor | null) => void;
}) {
  const imageSrc = getToothStatusAsset(arch, order, status);
  const shouldFlipImplantVertically = status === 'implant' && arch === 'L';
  const scaleX = flipX ? -1 : 1;
  const scaleY = shouldFlipImplantVertically ? -1 : 1;
  const imageTransform =
    scaleX === 1 && scaleY === 1
      ? undefined
      : `translate(${scaleX === -1 ? width : 0}px, ${scaleY === -1 ? height : 0}px) scale(${scaleX}, ${scaleY})`;
  const imageTransformOrigin = scaleX === -1 || scaleY === -1 ? 'top left' : undefined;
  const glowFilter = active
    ? 'drop-shadow(0 0 8px rgba(0, 192, 243, 0.95)) drop-shadow(0 0 14px rgba(0, 192, 243, 0.65))'
    : hasDetection
      ? 'drop-shadow(0 0 6px rgba(255, 182, 0, 0.45))'
      : 'none';
  return (
    <button
      type="button"
      onClick={() => onClick?.(toothFdi)}
      onMouseEnter={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        onHoverChange?.({
          toothFdi,
          x: rect.left + rect.width / 2,
          y: rect.top - 10,
        });
      }}
      onMouseLeave={() => onHoverChange?.(null)}
      aria-pressed={active}
      aria-label={`Tooth ${toothFdi}`}
      style={{
        width: wp(width),
        height: hp(height),
        left: wp(left),
        top: hp(top),
        position: 'absolute',
        background: 'transparent',
        border: 'none',
        padding: 0,
        cursor: 'pointer',
        zIndex: active ? 15 : 12,
        opacity: dimmed ? 0.36 : 1,
      }}
    >
      <img
        src={imageSrc}
        alt=""
        draggable={false}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          transform: imageTransform,
          transformOrigin: imageTransformOrigin,
          filter: glowFilter,
        }}
      />
    </button>
  );
}

function useViewportSize() {
  const [viewport, setViewport] = useState(() => ({
    width: typeof window === 'undefined' ? DESIGN_WIDTH : window.innerWidth,
    height: typeof window === 'undefined' ? DESIGN_HEIGHT : window.innerHeight,
  }));

  useEffect(() => {
    const updateViewport = () => {
      setViewport({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    updateViewport();
    window.addEventListener('resize', updateViewport);
    return () => window.removeEventListener('resize', updateViewport);
  }, []);

  return viewport;
}

export function RenewPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const locationState = (location.state as any) || {};
  const sidecarDicomInfo = (locationState?.dicomInfo || null) as DicomOverlayMetadata | null;
  const [result, setResult] = useState<any>(locationState?.result || null);
  const [jobId, setJobId] = useState<string | null>(locationState?.jobId || null);
  const [isProcessing, setIsProcessing] = useState(!locationState?.result && !!locationState?.jobId);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [activeFolderStudies, setActiveFolderStudies] = useState<FolderStudy[]>(() => {
    const raw = (locationState.originalFolderStudies as FolderStudy[] | undefined) || [];
    const seen = new Set<string>();
    return raw.filter((study) => {
      if (seen.has(study.id)) return false;
      seen.add(study.id);
      return true;
    });
  });
  const [serverStudies, setServerStudies] = useState<any[]>([]);
  const [serverImages, setServerImages] = useState<ServerFolderImage[]>([]);
  const [isReportActive, setIsReportActive] = useState(false);
  const [isChartVisible, setIsChartVisible] = useState(true);
  const [workspaceSection, setWorkspaceSection] = useState<'studies' | 'report' | 'none'>(
    locationState?.reportSessionId ? 'report' : 'none'
  );
  const [selectedToolbarButton, setSelectedToolbarButton] = useState<ToolbarKey>('pointer');
  const [flashToolbarButton, setFlashToolbarButton] = useState<ToolbarKey | null>(null);
  const [activeMeasureSubtool, setActiveMeasureSubtool] = useState<MeasureSubtoolKey>('length');
  const [toolSubmenu, setToolSubmenu] = useState<{
    menu: MeasureMenuKey;
    left: number;
    top: number;
  } | null>(null);
  const [measureShapes, setMeasureShapes] = useState<MeasureShape[]>([]);
  const [pendingMeasurePoints, setPendingMeasurePoints] = useState<ImagePoint[]>([]);
  const [measurePreviewPoint, setMeasurePreviewPoint] = useState<ImagePoint | null>(null);
  const [pendingTextAnnotation, setPendingTextAnnotation] = useState<{
    point: ImagePoint;
    text: string;
    anchorLeft: number;
    anchorTop: number;
  } | null>(null);
  const [capturedOutputs, setCapturedOutputs] = useState<OutputCaptureItem[]>([]);
  const [selectedReportCaptureIds, setSelectedReportCaptureIds] = useState<string[]>([]);
  const [isCapturePanelCollapsed, setIsCapturePanelCollapsed] = useState(true);
  const [reportSessionId, setReportSessionId] = useState<string | null>(locationState?.reportSessionId || null);
  const [reportDrawerOpen, setReportDrawerOpen] = useState(false);
  const [reportStartState, setReportStartState] = useState<'idle' | 'creating'>('idle');
  const [reportError, setReportError] = useState<string | null>(null);
  const [reportPreviewRefreshToken, setReportPreviewRefreshToken] = useState(0);
  const lastSyncedReportCapturesRef = useRef('');
  const lastRefreshedReportCaptureStructureRef = useRef('');
  const textAnnotationInputRef = useRef<HTMLTextAreaElement | null>(null);
  const textAnnotationWasOpenRef = useRef(false);
  const [inverted, setInverted] = useState(false);
  const [flipped, setFlipped] = useState(false);
  const [numberingSystem, setNumberingSystem] = useState<NumberingSystem>(() => readStoredNumberingSystem());
  const [viewMode, setViewMode] = useState<'original' | 'overlay' | 'heatmap'>(() => {
    const initialResult = locationState?.result;
    const hasInitialStructuredOverlay = Boolean(
      (Array.isArray(initialResult?.sinus_contours) && initialResult.sinus_contours.length > 0) ||
      (Array.isArray(initialResult?.nerve_contours) && initialResult.nerve_contours.length > 0) ||
      (Array.isArray(initialResult?.teeth) && initialResult.teeth.length > 0) ||
      (Array.isArray(initialResult?.teeth_objects) && initialResult.teeth_objects.length > 0)
    );
    return hasInitialStructuredOverlay ? 'overlay' : 'original';
  });
  const [panoZoom, setPanoZoom] = useState(1);
  const [panoOffset, setPanoOffset] = useState({ x: 0, y: 0 });
  const [panoBrightness, setPanoBrightness] = useState(PANO_DEFAULT_BRIGHTNESS);
  const [panoContrast, setPanoContrast] = useState(PANO_DEFAULT_CONTRAST);
  const [heatmapOpacity, setHeatmapOpacity] = useState(50);
  const [isHeatmapControlsExpanded, setIsHeatmapControlsExpanded] = useState(false);
  const [isHeatmapPresetMenuVisible, setIsHeatmapPresetMenuVisible] = useState(false);
  const [heatmapControlsPosition, setHeatmapControlsPosition] = useState({ x: 88, y: 458 });
  const [panoDisplaySize, setPanoDisplaySize] = useState({ width: 0, height: 0 });
  const [panoNaturalSize, setPanoNaturalSize] = useState({ width: 0, height: 0 });
  const [isOverlayPresetMenuVisible, setIsOverlayPresetMenuVisible] = useState(false);
  const [panoMagnifier, setPanoMagnifier] = useState({
    visible: false,
    clientX: 0,
    clientY: 0,
    viewerX: 0,
    viewerY: 0,
    imgX: 0,
    imgY: 0,
  });
  const [activeTooth, setActiveTooth] = useState<string | null>(null);
  const [activeDetectionId, setActiveDetectionId] = useState<string | null>(null);
  const [hoveredToothAnchor, setHoveredToothAnchor] = useState<ToothHoverAnchor | null>(null);
  const [activeLegendFilter, setActiveLegendFilter] = useState<ToothCondition | null>(null);
  const [overlayPreset, setOverlayPreset] = useState<OverlayPreset>('all');
  const [selectedFolderSeriesId, setSelectedFolderSeriesId] = useState<string | null>(
    locationState.folderSelectedSeriesId || activeFolderStudies.flatMap((study) => study.series)[0]?.id || null
  );
  const originalFolderMode = Boolean(locationState.originalFolderMode);
  const originalFile = locationState.originalFile as File | undefined;
  const selectedFolderSeries = useMemo(
    () => activeFolderStudies.flatMap((study) => study.series).find((series) => series.id === selectedFolderSeriesId) || null,
    [activeFolderStudies, selectedFolderSeriesId]
  );
  const originalIsDicom = Boolean(
    (originalFolderMode && selectedFolderSeries) ||
    locationState.originalIsDicom ||
    (originalFile && isDicomPath(originalFile.name)) ||
    isDicomPath(result?.image_url)
  );
  const panoViewportRef = useRef<HTMLDivElement | null>(null);
  const panoImageRef = useRef<HTMLImageElement | null>(null);
  const panoDisplayRef = useRef<HTMLDivElement | null>(null);
  const panoLensCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const panoOverlaySvgRef = useRef<SVGSVGElement | null>(null);
  const panoMeasureSvgRef = useRef<SVGSVGElement | null>(null);
  const autoAnalyzeTriggeredRef = useRef(false);
  const pollStartedAtRef = useRef<number | null>(null);
  const status404CountRef = useRef(0);
  const heatmapControlsTimeoutRef = useRef<number | null>(null);
  const autoWindowAppliedRef = useRef<string | null>(null);
  const previousSelectedFolderSeriesIdRef = useRef<string | null>(selectedFolderSeriesId);
  const panoDragRef = useRef<{
    mode: 'pan' | 'wlww' | null;
    startX: number;
    startY: number;
    startOffsetX: number;
    startOffsetY: number;
    startBrightness: number;
    startContrast: number;
  }>({
    mode: null,
    startX: 0,
    startY: 0,
    startOffsetX: 0,
    startOffsetY: 0,
    startBrightness: 100,
    startContrast: 100,
  });
  const viewport = useViewportSize();
  const cacheBuster = useMemo(() => Date.now(), []);
  const scale = Math.min(viewport.width / DESIGN_WIDTH, viewport.height / DESIGN_HEIGHT);
  const stageWidth = viewport.width;
  const stageHeight = DESIGN_HEIGHT * scale;
  const designCanvasWidth = stageWidth / scale;
  const viewerExtraWidth = Math.max(0, designCanvasWidth - DESIGN_WIDTH);
  const viewerLeft = 237;
  const topBarLeft = viewerLeft;
  const viewerWidth = 1676 + viewerExtraWidth;
  const topBarWidth = viewerWidth;
  const rightEdge = viewerLeft + viewerWidth;
  const chartSectionLeft = 238;
  const chartSectionWidth = rightEdge - chartSectionLeft;
  const chartLegendDividerX = chartSectionLeft + chartSectionWidth * 0.15;
  const chartLegendWidth = chartLegendDividerX - chartLegendLeft - 28;
  const reportLeft = rightEdge - 95;
  const odontoFrameLeft = chartLegendDividerX + 36;
  const odontoFrameRight = reportLeft - 58;
  const chartOdontoCenterX = (odontoFrameLeft + odontoFrameRight) / 2;
  const toothSlotCentersLeft = outerToInnerOffsets.map((offset) => chartOdontoCenterX - offset);
  const toothSlotCentersRight = innerToOuterOffsets.map((offset) => chartOdontoCenterX + offset);
  const chartOdontoFramePaddingY = 18;
  const chartOdontoFrameTop = chartContentTop + chartOdontoFramePaddingY;
  const chartOdontoFrameBottom = chartContentTop + chartContentHeight - chartOdontoFramePaddingY;
  const chartOdontoLineLeft = toothSlotCentersLeft[0] - 24;
  const chartOdontoLineRight = toothSlotCentersRight[toothSlotCentersRight.length - 1] + 24;
  const chartOdontoLineWidth = chartOdontoLineRight - chartOdontoLineLeft;
  const chartOdontoLineY = Math.round(chartContentTop + chartContentHeight / 2);
  const chartOdontoVerticalTop = chartOdontoFrameTop;
  const chartOdontoVerticalHeight = chartOdontoFrameBottom - chartOdontoFrameTop;
  const displayedUpperArch = getDisplayedArchTeeth(flipped, upperTeeth, upperRightTeeth);
  const displayedLowerArch = getDisplayedArchTeeth(flipped, lowerTeeth, lowerRightTeeth);
  const viewLabelTop = 75;
  const measureLabelTop = 195;
  const outputLabelTop = 315;
  const sectionArrowOffsetY = 6;
  const heatmapControlsTop = 458;
  const outputCaptureTop = viewMode === 'heatmap' ? 494 : 472;
  const panoFrameHeight = isChartVisible ? 755 : 1019;
  const reportTop = chartContentTop + 130;
  const fdiLeft = rightEdge - 82;
  const lLabelLeft = rightEdge - 29;
  const chartHeaderHideLeft = rightEdge - 28;
  const panoChartToggleLeft = viewerLeft + 8;
  const panoChartToggleTop = 49 + 1019 - 18;
  const panoLabelTop = Math.round(49 + panoFrameHeight / 2 - 7);
  const isChartBodyVisible = isChartVisible;
  const panoBodyTop = 68;
  const panoBodyHeight = panoFrameHeight - 19;
  const panoBodyWidth = viewerWidth - 2;
  const studiesPanelLeft = viewerLeft + 16;
  const studiesPanelTop = 74;
  const studiesPanelWidth = 252;
  const studiesPanelHeight = Math.max(420, panoFrameHeight - 36);
  const hasHeatmapAsset = Boolean(result?.heatmap_overlay_url);
  const [dicomHudMetadata, setDicomHudMetadata] = useState<DicomOverlayMetadata | null>(sidecarDicomInfo);
  const [dicomPreviewDataUrl, setDicomPreviewDataUrl] = useState<string | null>(null);
  const [dicomAutoWindow, setDicomAutoWindow] = useState<{ level: number; width: number } | null>(null);
  const hasStructuredOverlayData = Boolean(
    (Array.isArray(result?.sinus_contours) && result.sinus_contours.length > 0) ||
    (Array.isArray(result?.nerve_contours) && result.nerve_contours.length > 0) ||
    (Array.isArray(result?.teeth) && result.teeth.length > 0) ||
    (Array.isArray(result?.teeth_objects) && result.teeth_objects.length > 0)
  );
  const dicomHudFile =
    originalFolderMode && selectedFolderSeries?.files?.length
      ? selectedFolderSeries.files[0]
      : originalFile && originalIsDicom
        ? originalFile
        : null;
  const getUrlWithCacheBuster = (url?: string | null) => {
    if (!url) return null;
    if (url.startsWith('blob:') || url.startsWith('data:')) return url;
    return `${url}${url.includes('?') ? '&' : '?'}t=${cacheBuster}`;
  };

  const normalizeRuntimeAssetUrl = (url?: string | null) => {
    const normalized = String(url || '').trim().replace(/\\/g, '/');
    if (!normalized) return null;
    if (
      normalized.startsWith('blob:') ||
      normalized.startsWith('data:') ||
      /^https?:\/\//i.test(normalized)
    ) {
      return normalized;
    }
    if (normalized.startsWith('/')) return normalized;
    if (normalized.startsWith('temp/') || normalized.startsWith('api/') || normalized.includes('/')) {
      return `/${normalized}`;
    }
    return null;
  };

  const pickRuntimeAssetUrl = (...candidates: Array<string | null | undefined>) => {
    for (const candidate of candidates) {
      const normalized = normalizeRuntimeAssetUrl(candidate);
      if (normalized) return normalized;
    }
    return null;
  };

  const originalRasterSource = pickRuntimeAssetUrl(
    dicomPreviewDataUrl,
    result?.preview_url,
    locationState.previewUrl,
    locationState.imageUrl,
    result?.image_url
  );
  const originalAnalysisSource = pickRuntimeAssetUrl(
    result?.image_url,
    result?.preview_url,
    locationState.imageUrl,
    locationState.previewUrl
  );
  const heatmapPanoSource = pickRuntimeAssetUrl(
    result?.heatmap_overlay_url,
    result?.overlay_url,
    result?.preview_url,
    locationState.previewUrl,
    locationState.imageUrl,
    result?.image_url
  );
  const originalRasterUrl = getUrlWithCacheBuster(originalRasterSource);
  const originalAnalysisUrl = getUrlWithCacheBuster(originalAnalysisSource);
  const originalPanoUrl = originalRasterUrl || originalAnalysisUrl;
  const heatmapPanoUrl = getUrlWithCacheBuster(heatmapPanoSource);
  const realHeatmapOverlayUrl =
    viewMode === 'heatmap' && hasHeatmapAsset
      ? getUrlWithCacheBuster(pickRuntimeAssetUrl(result?.heatmap_overlay_url))
      : null;
  const panoUsesPreviewRaster = Boolean(
    originalIsDicom &&
    originalRasterUrl &&
    originalPanoUrl &&
    originalRasterUrl === originalPanoUrl
  );
  const panoViewerUrl = originalPanoUrl || heatmapPanoUrl;
  const effectiveScale = Math.max(
    ((panoDisplaySize.width || panoBodyWidth || 1) / Math.max(1, panoNaturalSize.width || panoBodyWidth || 1)) * panoZoom,
    0.001
  );
  const fitPanoImage = () => {
    const img = panoImageRef.current;
    if (!img?.naturalWidth || !img?.naturalHeight) return;
    setPanoNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
    const fitScale = Math.min(panoBodyWidth / img.naturalWidth, panoBodyHeight / img.naturalHeight);
    setPanoDisplaySize({
      width: img.naturalWidth * fitScale,
      height: img.naturalHeight * fitScale,
    });
  };

  const clearHeatmapControlsTimeout = () => {
    if (heatmapControlsTimeoutRef.current !== null) {
      window.clearTimeout(heatmapControlsTimeoutRef.current);
      heatmapControlsTimeoutRef.current = null;
    }
  };

  const scheduleHeatmapControlsCollapse = (_delay = 2600) => {
    clearHeatmapControlsTimeout();
  };

  const showHeatmapControls = (_delay = 2600) => {
    setIsHeatmapControlsExpanded(true);
    clearHeatmapControlsTimeout();
  };

  const handleHeatmapControlsDragStart = (event: React.PointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    clearHeatmapControlsTimeout();
    const startPointer = { x: event.clientX, y: event.clientY };
    const startPosition = heatmapControlsPosition;
    const panelWidth = 140;
    const panelHeight = isHeatmapControlsExpanded ? 40 : 20;

    const handlePointerMove = (moveEvent: PointerEvent) => {
      const deltaX = (moveEvent.clientX - startPointer.x) / Math.max(scale, 0.001);
      const deltaY = (moveEvent.clientY - startPointer.y) / Math.max(scale, 0.001);
      const nextX = clamp(startPosition.x + deltaX, 0, Math.max(0, designCanvasWidth - panelWidth));
      const nextY = clamp(startPosition.y + deltaY, 49, Math.max(49, DESIGN_HEIGHT - panelHeight - 18));
      setHeatmapControlsPosition({ x: nextX, y: nextY });
    };

    const handlePointerUp = () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
  };

  const drawSvgOverlayOnContext = async (
    context: CanvasRenderingContext2D,
    svgElement: SVGSVGElement | null,
    drawLeft: number,
    drawTop: number,
    drawWidth: number,
    drawHeight: number,
    useCenteredTransform = false
  ) => {
    if (!svgElement || !overlayCoordinateSize.width || !overlayCoordinateSize.height) return;
    const svgClone = svgElement.cloneNode(true) as SVGSVGElement;
    svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svgClone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
    svgClone.setAttribute('width', String(overlayCoordinateSize.width));
    svgClone.setAttribute('height', String(overlayCoordinateSize.height));
    svgClone.setAttribute('viewBox', `0 0 ${overlayCoordinateSize.width} ${overlayCoordinateSize.height}`);
    const serialized = new XMLSerializer().serializeToString(svgClone);
    const svgBlob = new Blob([`<?xml version="1.0" encoding="UTF-8"?>${serialized}`], { type: 'image/svg+xml;charset=utf-8' });
    const svgUrl = URL.createObjectURL(svgBlob);
    try {
      const overlayImage = await loadImageFromUrl(svgUrl);
      if (useCenteredTransform) {
        context.drawImage(overlayImage, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
      } else {
        context.drawImage(overlayImage, drawLeft, drawTop, drawWidth, drawHeight);
      }
    } finally {
      URL.revokeObjectURL(svgUrl);
    }
  };

  const drawRasterOverlayOnContext = async (
    context: CanvasRenderingContext2D,
    imageUrl: string | null,
    drawLeft: number,
    drawTop: number,
    drawWidth: number,
    drawHeight: number,
    opacity = 1,
    useCenteredTransform = false
  ) => {
    if (!imageUrl || opacity <= 0) return;

    const overlayImage = await loadImageFromUrl(imageUrl);
    const previousAlpha = context.globalAlpha;
    context.globalAlpha = previousAlpha * Math.min(1, Math.max(0, opacity));
    try {
      if (useCenteredTransform) {
        context.drawImage(overlayImage, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
      } else {
        context.drawImage(overlayImage, drawLeft, drawTop, drawWidth, drawHeight);
      }
    } finally {
      context.globalAlpha = previousAlpha;
    }
  };

  const buildPanoCaptureCanvas = async () => {
    const img = panoImageRef.current;
    if (!img || !panoDisplaySize.width || !panoDisplaySize.height) return null;

    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(panoBodyWidth));
    canvas.height = Math.max(1, Math.round(panoBodyHeight));
    const context = canvas.getContext('2d');
    if (!context) return null;

    context.fillStyle = '#000000';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.save();
    context.filter = `invert(${inverted ? 1 : 0}) brightness(${panoBrightness}%) contrast(${panoContrast}%)`;

    const drawWidth = panoDisplaySize.width * panoZoom;
    const drawHeight = panoDisplaySize.height * panoZoom;
    const drawLeft = (canvas.width - drawWidth) / 2 + panoOffset.x;
    const drawTop = (canvas.height - drawHeight) / 2 + panoOffset.y;

    if (flipped) {
      context.translate(drawLeft + drawWidth / 2, drawTop + drawHeight / 2);
      context.scale(-1, 1);
      context.drawImage(img, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
      await drawRasterOverlayOnContext(
        context,
        realHeatmapOverlayUrl,
        drawLeft,
        drawTop,
        drawWidth,
        drawHeight,
        heatmapOpacity / 100,
        true
      );
      if (viewMode !== 'original') {
        await drawSvgOverlayOnContext(context, panoOverlaySvgRef.current, drawLeft, drawTop, drawWidth, drawHeight, true);
      }
      await drawSvgOverlayOnContext(context, panoMeasureSvgRef.current, drawLeft, drawTop, drawWidth, drawHeight, true);
    } else {
      context.drawImage(img, drawLeft, drawTop, drawWidth, drawHeight);
      await drawRasterOverlayOnContext(
        context,
        realHeatmapOverlayUrl,
        drawLeft,
        drawTop,
        drawWidth,
        drawHeight,
        heatmapOpacity / 100
      );
      if (viewMode !== 'original') {
        await drawSvgOverlayOnContext(context, panoOverlaySvgRef.current, drawLeft, drawTop, drawWidth, drawHeight);
      }
      await drawSvgOverlayOnContext(context, panoMeasureSvgRef.current, drawLeft, drawTop, drawWidth, drawHeight);
    }
    context.restore();
    return canvas;
  };

  const trimCaptureCanvas = (sourceCanvas: HTMLCanvasElement) => {
    const context = sourceCanvas.getContext('2d');
    if (!context) return sourceCanvas;

    const { width, height } = sourceCanvas;
    const imageData = context.getImageData(0, 0, width, height);
    const pixels = imageData.data;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = (y * width + x) * 4;
        const r = pixels[index];
        const g = pixels[index + 1];
        const b = pixels[index + 2];
        const a = pixels[index + 3];
        if (a < 8) continue;

        const maxChannel = Math.max(r, g, b);
        const minChannel = Math.min(r, g, b);
        const luminance = (r + g + b) / 3;
        const chroma = maxChannel - minChannel;
        const isInformativePixel = (luminance > 14 && luminance < 246) || chroma > 18;
        if (!isInformativePixel) continue;

        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }

    if (maxX < minX || maxY < minY) {
      return sourceCanvas;
    }

    const padding = 8;
    const cropLeft = Math.max(0, minX - padding);
    const cropTop = Math.max(0, minY - padding);
    const cropRight = Math.min(width, maxX + padding + 1);
    const cropBottom = Math.min(height, maxY + padding + 1);
    const croppedWidth = Math.max(1, cropRight - cropLeft);
    const croppedHeight = Math.max(1, cropBottom - cropTop);

    if (croppedWidth >= width && croppedHeight >= height) {
      return sourceCanvas;
    }

    const trimmedCanvas = document.createElement('canvas');
    trimmedCanvas.width = croppedWidth;
    trimmedCanvas.height = croppedHeight;
    const trimmedContext = trimmedCanvas.getContext('2d');
    if (!trimmedContext) return sourceCanvas;

    trimmedContext.drawImage(
      sourceCanvas,
      cropLeft,
      cropTop,
      croppedWidth,
      croppedHeight,
      0,
      0,
      croppedWidth,
      croppedHeight
    );
    return trimmedCanvas;
  };

  const combinedStudies = useMemo(() => {
    const activeIds = new Set(activeFolderStudies.map((study) => study.id));
    const activeFingerprints = new Set(activeFolderStudies.map((study) => `${study.label}::${study.description}::${study.patientId}`));
    const additional = serverStudies.filter((study) => {
      if (activeIds.has(study.id)) return false;
      const fingerprint = `${study.label}::${study.description}::${study.patientId}`;
      if (activeFingerprints.has(fingerprint)) return false;
      return true;
    });
    const imageStudies = serverImages.map((image) => ({
      id: `${SERVER_IMAGE_SERIES_PREFIX}${image.relativePath}`,
      label: image.name,
      description: image.description || image.folderLabel || image.format || 'Image',
      patientId: image.patientId || '',
      studyDate: image.studyDate || '',
      modalities: image.modalities?.length ? image.modalities : [image.format || 'IMG'],
      totalFiles: 1,
      totalSeries: 1,
      series: [
        {
          id: `${SERVER_IMAGE_SERIES_PREFIX}${image.relativePath}`,
          studyId: `${SERVER_IMAGE_SERIES_PREFIX}${image.relativePath}`,
          label: image.name,
          description: image.description || image.folderLabel || 'Image',
          modality: image.format || 'IMG',
          files: [],
          orientation: 'Unknown' as const,
          sliceCount: 1,
          spacingLabel: '',
          compression: 'None',
          isCompressed: false,
          volumeEligible: false,
        },
      ],
    }));
    const merged = [...activeFolderStudies, ...additional, ...imageStudies];
    const seen = new Set<string>();
    return merged.filter((study) => {
      if (seen.has(study.id)) return false;
      seen.add(study.id);
      return true;
    });
  }, [activeFolderStudies, serverImages, serverStudies]);

  const toothGeometries = useMemo(() => {
    const toothList = Array.isArray(result?.teeth)
      ? result.teeth
      : Array.isArray(result?.teeth_objects)
        ? result.teeth_objects
        : [];

    return toothList
      .map((tooth: any) => {
        const fdi = getToothLabel(tooth);
        const contour = normalizeContourPoints(tooth?.contour);
        if (!fdi || contour.length < 3) return null;
        const bounds = boundsFromPoints(contour);
        if (!bounds) return null;
        return {
          fdi,
          contour,
          centroid: centroidFromPoints(contour),
          bounds,
        } satisfies ToothGeometry;
      })
      .filter((tooth): tooth is ToothGeometry => tooth !== null);
  }, [result?.teeth, result?.teeth_objects]);

  const normalizedDetections = useMemo(() => {
    if (!result) return [] as NormalizedDetection[];

    const items: NormalizedDetection[] = [];
    const seenIds = new Set<string>();
    const cariesBestMap = result?.caries_by_tooth_best || result?.analysis_result?.caries_by_tooth_best || {};
    const periapicalBestMap =
      result?.periapical_by_tooth_best || result?.analysis_result?.periapical_by_tooth_best || {};
    const pushDetection = (
      type: string,
      source: any,
      fallbackTooth?: string | null,
      fallbackLabel?: string,
      options?: { trustConfidence?: boolean }
    ) => {
      const contour = normalizeContourPoints(source?.contour ?? source?.polygon ?? source?.points);
      const bounds =
        normalizeBounds(source?.box ?? source?.bbox ?? source?.bounds) ??
        (contour.length >= 2 ? boundsFromPoints(contour) : null);
      if (!bounds) return;

      const confidence = getDetectionConfidence(source);
      if (!options?.trustConfidence && confidence < AI_CONFIDENCE_THRESHOLD) return;

      const center = contour.length ? centroidFromPoints(contour) : centerFromBounds(bounds);
      const explicitTooth = fallbackTooth ?? getToothLabel(source);
      const toothFdi = inferToothFdi(explicitTooth, center, toothGeometries, contour, bounds);
      const label = fallbackLabel ?? source?.class_name ?? source?.name ?? source?.label ?? type;
      const id = String(source?.id ?? `${type}:${label}:${toothFdi ?? 'none'}:${Math.round(center.x)}:${Math.round(center.y)}`);
      if (seenIds.has(id)) return;
      seenIds.add(id);

      items.push({
        id,
        type,
        label: String(label),
        confidence,
        toothFdi,
        contour: contour.length >= 3 ? contour : null,
        bounds,
        center,
        source,
      });
    };

    Object.entries(cariesBestMap).forEach(([tooth, data]) => {
      pushDetection('caries', data, String(tooth), 'Caries', { trustConfidence: true });
    });

    Object.entries(periapicalBestMap).forEach(([tooth, data]) => {
      pushDetection('periapical', data, String(tooth), 'Periapical', { trustConfidence: true });
    });

    toothGeometries.forEach((tooth) => {
      const boneLevel = Number(result?.bonelevel?.[tooth.fdi]?.percent ?? 0);
      if (boneLevel >= 15) {
        pushDetection(
          'bonelevel',
          {
            id: `bonelevel:${tooth.fdi}`,
            confidence: Math.min(0.99, boneLevel / 100),
            contour: tooth.contour.map((point) => [point.x, point.y]),
            label: 'Bone level',
          },
          tooth.fdi,
          'Bone level'
        );
      }
    });

    [
      ...(Array.isArray(result?.data) ? result.data : []),
      ...(Array.isArray(result?.analysis_result?.data) ? result.analysis_result.data : []),
    ].forEach((item: any) => {
      pushDetection(String(item?.class_name ?? item?.type ?? item?.label ?? 'finding').toLowerCase(), item);
    });

    return items.sort((left, right) => right.confidence - left.confidence);
  }, [
    result,
    result?.caries_by_tooth_best,
    result?.periapical_by_tooth_best,
    result?.analysis_result?.caries_by_tooth_best,
    result?.analysis_result?.periapical_by_tooth_best,
    result?.bonelevel,
    result?.data,
    result?.analysis_result?.data,
    toothGeometries,
  ]);

  const detectionsByTooth = useMemo(() => {
    return normalizedDetections.reduce<Record<string, NormalizedDetection[]>>((acc, detection) => {
      if (!detection.toothFdi) return acc;
      if (!acc[detection.toothFdi]) acc[detection.toothFdi] = [];
      acc[detection.toothFdi].push(detection);
      return acc;
    }, {});
  }, [normalizedDetections]);

  const primaryDetectionByTooth = useMemo(() => {
    return Object.entries(detectionsByTooth).reduce<Record<string, NormalizedDetection>>((acc, [tooth, detections]) => {
      const primary = [...detections].sort((left, right) => right.confidence - left.confidence)[0];
      if (primary) acc[tooth] = primary;
      return acc;
    }, {});
  }, [detectionsByTooth]);

  const findingSignalByTooth = useMemo(() => {
    const map: Record<string, boolean> = {};
    const markKeys = (record: any) => {
      Object.keys(record || {}).forEach((key) => {
        if (/^\d{2}$/.test(String(key))) map[String(key)] = true;
      });
    };
    const markList = (items: any) => {
      (Array.isArray(items) ? items : []).forEach((item: any) => {
        const key = getToothId(item);
        if (key) map[key] = true;
      });
    };

    markKeys(result?.caries_by_tooth_best);
    markKeys(result?.periapical_by_tooth_best);
    markKeys(result?.analysis_result?.caries_by_tooth_best);
    markKeys(result?.analysis_result?.periapical_by_tooth_best);
    markList(result?.caries);
    markList(result?.periapical);
    markList(result?.analysis_result?.caries);
    markList(result?.analysis_result?.periapical);

    return map;
  }, [
    result?.caries_by_tooth_best,
    result?.periapical_by_tooth_best,
    result?.analysis_result?.caries_by_tooth_best,
    result?.analysis_result?.periapical_by_tooth_best,
    result?.caries,
    result?.periapical,
    result?.analysis_result?.caries,
    result?.analysis_result?.periapical,
  ]);

  const toothRecords = useMemo(() => {
    const records: Record<string, any> = {};
    const sources = [
      Array.isArray(result?.data) ? result.data : [],
      Array.isArray(result?.teeth) ? result.teeth : Object.values(result?.teeth || {}),
      Array.isArray(result?.missing_teeth) ? result.missing_teeth : [],
      Array.isArray(result?.teeth_missing) ? result.teeth_missing : [],
      Array.isArray(result?.analysis_result?.data) ? result.analysis_result.data : [],
      Array.isArray(result?.analysis_result?.teeth)
        ? result.analysis_result.teeth
        : Object.values(result?.analysis_result?.teeth || {}),
      Array.isArray(result?.analysis_result?.missing_teeth) ? result.analysis_result.missing_teeth : [],
      Array.isArray(result?.analysis_result?.teeth_missing) ? result.analysis_result.teeth_missing : [],
    ];

    sources.forEach((items: any[]) => {
      items.forEach((item: any) => {
        const key = getToothId(item);
        if (!key) return;
        records[key] = { ...(records[key] || {}), ...item };
      });
    });

    return records;
  }, [result]);

  const implantMetricsByTooth = useMemo(() => {
    const map: Record<string, any> = {};
    const metricSources = [
      Array.isArray(result?.implant_metrics) ? result.implant_metrics : Object.values(result?.implant_metrics || {}),
      Array.isArray(result?.analysis_result?.implant_metrics)
        ? result.analysis_result.implant_metrics
        : Object.values(result?.analysis_result?.implant_metrics || {}),
    ];

    metricSources.forEach((metrics: any[]) => {
      metrics.forEach((item: any) => {
        const key = getToothId(item?.label || item?.tooth_label || item?.tooth || item?.id);
        if (key && key !== 'undefined') {
          map[key] = { ...(map[key] || {}), ...item };
        }
      });
    });

    return map;
  }, [result]);

  const overlayCoordinateInfo = useMemo(() => {
    const explicitWidth =
      readOverlayDimension(result, ['overlay_width', 'image_width', 'original_width', 'source_width', 'canvas_width', 'preview_width', 'width']) ??
      readOverlayDimension(result?.analysis_result, ['overlay_width', 'image_width', 'original_width', 'source_width', 'canvas_width', 'preview_width', 'width']);
    const explicitHeight =
      readOverlayDimension(result, ['overlay_height', 'image_height', 'original_height', 'source_height', 'canvas_height', 'preview_height', 'height']) ??
      readOverlayDimension(result?.analysis_result, ['overlay_height', 'image_height', 'original_height', 'source_height', 'canvas_height', 'preview_height', 'height']);

    const naturalWidth = panoNaturalSize.width || 1;
    const naturalHeight = panoNaturalSize.height || 1;

    let maxX = 0;
    let maxY = 0;
    const considerPoint = (point: ImagePoint | null) => {
      if (!point) return;
      if (Number.isFinite(point.x)) maxX = Math.max(maxX, point.x);
      if (Number.isFinite(point.y)) maxY = Math.max(maxY, point.y);
    };

    (Array.isArray(result?.sinus_contours) ? result.sinus_contours : []).forEach((contour: any) => {
      normalizeContourPoints(contour).forEach(considerPoint);
    });
    (Array.isArray(result?.nerve_contours) ? result.nerve_contours : []).forEach((contour: any) => {
      normalizeContourPoints(contour).forEach(considerPoint);
    });
    toothGeometries.forEach((tooth) => tooth.contour.forEach(considerPoint));
    normalizedDetections.forEach((detection) => {
      if (detection.contour?.length) {
        detection.contour.forEach(considerPoint);
      } else {
        considerPoint({ x: detection.bounds.x1, y: detection.bounds.y1 });
        considerPoint({ x: detection.bounds.x2, y: detection.bounds.y2 });
      }
    });

    const inferredWidth = maxX > 0 ? maxX + 16 : null;
    const inferredHeight = maxY > 0 ? maxY + 16 : null;
    const hasExplicitSize = Boolean(explicitWidth && explicitHeight);
    const hasNaturalSize = naturalWidth > 1 && naturalHeight > 1;

    if (hasExplicitSize) {
      return {
        width: Math.max(1, explicitWidth as number),
        height: Math.max(1, explicitHeight as number),
        source: 'backend-explicit' as const,
        inferredWidth,
        inferredHeight,
      };
    }

    if (hasNaturalSize) {
      return {
        width: Math.max(1, naturalWidth),
        height: Math.max(1, naturalHeight),
        source: 'image-natural' as const,
        inferredWidth,
        inferredHeight,
      };
    }

    return {
      width: Math.max(1, inferredWidth || naturalWidth),
      height: Math.max(1, inferredHeight || naturalHeight),
      source: 'inferred-fallback' as const,
      inferredWidth,
      inferredHeight,
    };
  }, [normalizedDetections, panoNaturalSize.height, panoNaturalSize.width, result, toothGeometries]);
  const overlayCoordinateSize = useMemo(
    () => ({ width: overlayCoordinateInfo.width, height: overlayCoordinateInfo.height }),
    [overlayCoordinateInfo.height, overlayCoordinateInfo.width]
  );

  const toothStatusByFdi = useMemo(() => {
    const map = odontogramTeeth.reduce<Record<string, ToothCondition>>((acc, tooth) => {
      acc[String(tooth)] = 'healthy';
      return acc;
    }, {});

    const markList = (items: any, status: ToothCondition) => {
      (Array.isArray(items) ? items : []).forEach((item: any) => {
        const key = getToothId(item);
        if (key) map[key] = status;
      });
    };

    const markKeys = (record: any, status: ToothCondition) => {
      Object.keys(record || {}).forEach((key) => {
        if (/^\d{2}$/.test(String(key))) {
          map[String(key)] = status;
        }
      });
    };

    markList(result?.missing_teeth || result?.teeth_missing || result?.analysis_result?.missing_teeth || [], 'missing');
    markKeys(result?.implant_by_tooth, 'implant');
    markKeys(result?.implant_by_tooth_best, 'implant');
    markKeys(result?.analysis_result?.implant_by_tooth, 'implant');
    markKeys(result?.analysis_result?.implant_by_tooth_best, 'implant');

    markList(result?.caries || result?.analysis_result?.caries || [], 'requires');
    markKeys(result?.caries_by_tooth, 'requires');
    markKeys(result?.caries_by_tooth_best, 'requires');
    markKeys(result?.analysis_result?.caries_by_tooth, 'requires');
    markKeys(result?.analysis_result?.caries_by_tooth_best, 'requires');

    markList(result?.periapical || result?.analysis_result?.periapical || [], 'requires');
    markKeys(result?.periapical_by_tooth, 'requires');
    markKeys(result?.periapical_by_tooth_best, 'requires');
    markKeys(result?.analysis_result?.periapical_by_tooth, 'requires');
    markKeys(result?.analysis_result?.periapical_by_tooth_best, 'requires');

    const boneLevelSources = [result?.bonelevel || {}, result?.analysis_result?.bonelevel || {}];
    boneLevelSources.forEach((source) => {
      Object.entries(source).forEach(([key, value]: any) => {
        if (!/^\d{2}$/.test(String(key))) return;
        const percent = Number(value?.percent ?? value?.bone_loss_pct ?? 0);
        if (percent >= 60) map[String(key)] = 'warning';
        else if (percent >= 15 && map[String(key)] === 'healthy') map[String(key)] = 'requires';
      });
    });

    const urgentSources = [
      result?.extraction_candidates || [],
      result?.analysis_result?.extraction_candidates || [],
      result?.implant_site_candidates || [],
      result?.analysis_result?.implant_site_candidates || [],
    ];

    urgentSources.forEach((items) => {
      (Array.isArray(items) ? items : []).forEach((item: any) => {
        const key = getToothId(item?.site_fdi || item?.tooth || item?.tooth_label || item);
        if (key && map[key] !== 'missing' && map[key] !== 'implant') {
          map[key] = 'warning';
        }
      });
    });

    Object.entries(toothRecords).forEach(([key, tooth]) => {
      const implantType = String(tooth?.type || '').toLowerCase();
      const boneLossLevel = Number(tooth?.bone_loss_level ?? 0);
      const boneLossPct = Number(tooth?.bone_loss_pct ?? 0);
      const isHopeless = Boolean(tooth?.hopeless) || boneLossLevel >= 4 || boneLossPct >= 60;
      const hasTreatmentFinding = Boolean(
        tooth?.caries || tooth?.periodontitis || tooth?.periapical || boneLossLevel >= 3 || boneLossPct >= 15
      );
      const isImplant = Boolean(tooth?.implant || implantType.includes('implant') || implantType.includes('fixture'));
      const isMissing = Boolean(tooth?.missing);

      if (isMissing) map[key] = 'missing';
      else if (isImplant) map[key] = 'implant';
      else if (isHopeless) map[key] = 'warning';
      else if (hasTreatmentFinding && map[key] === 'healthy') map[key] = 'requires';
    });

    return map;
  }, [result, toothRecords]);

  const nervePoints = useMemo(
    () => [
      ...flattenContourPoints(result?.nerve_contours),
      ...flattenContourPoints(result?.analysis_result?.nerve_contours),
    ],
    [result]
  );

  const toothHoverPanelByFdi = useMemo(() => {
    return odontogramTeeth.reduce<Record<string, ToothHoverPanelData>>((acc, tooth) => {
      const toothFdi = String(tooth);
      const record = toothRecords[toothFdi] || {};
      const status = toothStatusByFdi[toothFdi] || 'healthy';
      const implantMetric = implantMetricsByTooth[toothFdi] || {};
      const implantGuide = record?.implant_guide || {};
      const mmPerPx = Number(result?.mm_per_px || 0.1);
      const contourMetrics = estimateImplantMetricsFromContour(record?.contour, mmPerPx);
      const primaryDetection = primaryDetectionByTooth[toothFdi] || null;
      const boneLevelSource = result?.bonelevel?.[toothFdi] || result?.analysis_result?.bonelevel?.[toothFdi] || {};
      const boneLossLevel = Number(record?.bone_loss_level ?? boneLevelSource?.level ?? 0);
      const boneLossPct = Number(record?.bone_loss_pct ?? boneLevelSource?.percent ?? boneLevelSource?.bone_loss_pct ?? 0);
      const diameterMm = Number(
        implantMetric?.diameter_mm ??
        implantMetric?.diameter ??
        record?.implant_meta?.diameter ??
        contourMetrics.diameterMm ??
        0
      );
      const lengthMm = Number(
        implantMetric?.length_mm ??
        implantMetric?.length ??
        record?.implant_meta?.length ??
        contourMetrics.lengthMm ??
        0
      );
      const gapMm = Number(
        record?.mesiodistal_gap_mm ??
        implantGuide?.mesiodistal_gap_mm ??
        implantMetric?.mesiodistal_gap_mm ??
        0
      );
      const isUpperTooth = toothFdi.startsWith('1') || toothFdi.startsWith('2');
      const centerToNerveMmRaw = Number(
        implantGuide?.dist_mm ??
        record?.center_to_nerve_dist_mm ??
        implantMetric?.dist_mm ??
        record?.nerve_dist_mm ??
        0
      );
      const centerToNerveMm = isUpperTooth ? 0 : centerToNerveMmRaw;
      const toothNumber = Number(toothFdi);
      const lastDigit = toothNumber % 10;
      const isLowerPosteriorMissingPlanning =
        status === 'missing' &&
        (toothFdi.startsWith('3') || toothFdi.startsWith('4')) &&
        [6, 7, 8].includes(lastDigit);
      const missingBounds = normalizeBounds(record?.box);
      const fallbackGapMm =
        isLowerPosteriorMissingPlanning && !gapMm && missingBounds
          ? Math.max(0, (missingBounds.x2 - missingBounds.x1) * mmPerPx)
          : 0;
      const fallbackCenterToNerveMm =
        isLowerPosteriorMissingPlanning && !centerToNerveMmRaw && missingBounds && nervePoints.length
          ? Math.min(
              ...nervePoints.map((point) => distanceBetweenPoints(centerFromBounds(missingBounds), point))
            ) * mmPerPx
          : 0;
      const isExistingImplant = Boolean(status === 'implant' && (diameterMm > 0 || lengthMm > 0));
      const kind: ToothHoverPanelData['kind'] = isExistingImplant
        ? 'implant'
        : isLowerPosteriorMissingPlanning
          ? 'planning'
          : 'finding';
      let statusLabel = 'Healthy';
      if (status === 'missing') statusLabel = 'Missing';
      else if (status === 'implant') statusLabel = 'Implant';
      else if (status === 'warning') statusLabel = 'Urgent Priority';
      else if (status === 'requires') statusLabel = 'Treatment Required';
      const cariesProb = getBestConfidence(
        result?.caries_by_tooth_best || result?.analysis_result?.caries_by_tooth_best,
        result?.caries_by_tooth || result?.analysis_result?.caries_by_tooth,
        toothFdi
      );
      const periapicalProb = getBestConfidence(
        result?.periapical_by_tooth_best || result?.analysis_result?.periapical_by_tooth_best,
        result?.periapical_by_tooth || result?.analysis_result?.periapical_by_tooth,
        toothFdi
      );
      const primaryLabel = primaryDetection?.label || (status === 'healthy' ? 'Normal' : null);

      acc[toothFdi] = {
        toothFdi,
        title: `Tooth #${formatDisplayedToothNumber(toothFdi, numberingSystem, flipped)}`,
        kind,
        status: statusLabel,
        pblPct: Number.isFinite(boneLossPct) && boneLossPct > 0 ? boneLossPct : null,
        level: Number.isFinite(boneLossLevel) && boneLossLevel > 0 ? boneLossLevel : null,
        cariesProb,
        periapicalProb,
        primaryLabel,
        primaryProb: primaryDetection?.confidence ?? null,
        diameterMm: diameterMm > 0 ? diameterMm : null,
        lengthMm: lengthMm > 0 ? lengthMm : null,
        gapMm: isLowerPosteriorMissingPlanning && (gapMm > 0 || fallbackGapMm > 0) ? (gapMm > 0 ? gapMm : fallbackGapMm) : null,
        centerToNerveMm:
          isLowerPosteriorMissingPlanning && (centerToNerveMm > 0 || fallbackCenterToNerveMm > 0)
            ? (centerToNerveMm > 0 ? centerToNerveMm : fallbackCenterToNerveMm)
            : null,
      };
      return acc;
    }, {});
  }, [flipped, implantMetricsByTooth, nervePoints, numberingSystem, odontogramTeeth, primaryDetectionByTooth, result, toothRecords, toothStatusByFdi]);

  const hoveredToothPanel = hoveredToothAnchor ? toothHoverPanelByFdi[hoveredToothAnchor.toothFdi] || null : null;

  const legendCounts = useMemo(() => {
    return odontogramTeeth.reduce<Record<ToothCondition, number>>(
      (acc, tooth) => {
        const status = toothStatusByFdi[String(tooth)] || 'healthy';
        acc[status] += 1;
        return acc;
      },
      {
        healthy: 0,
        requires: 0,
        warning: 0,
        implant: 0,
        missing: 0,
      }
    );
  }, [toothStatusByFdi]);

  const activeDetection = useMemo(
    () => normalizedDetections.find((detection) => detection.id === activeDetectionId) ?? null,
    [activeDetectionId, normalizedDetections]
  );

  const mmPerPixel = Number(result?.mm_per_pixel ?? result?.mm_per_px ?? result?.analysis_result?.mm_per_pixel ?? result?.analysis_result?.mm_per_px ?? 0.1);
  const isCustomMeasureTool =
    selectedToolbarButton === 'measure-length' &&
    (activeMeasureSubtool === 'length' || activeMeasureSubtool === 'bidirectional' || activeMeasureSubtool === 'angle');
  const isCustomDrawTool =
    selectedToolbarButton === 'measure-draw' &&
    (activeMeasureSubtool === 'text' ||
      activeMeasureSubtool === 'arrow' ||
      activeMeasureSubtool === 'ellipse' ||
      activeMeasureSubtool === 'rect' ||
      activeMeasureSubtool === 'circle' ||
      activeMeasureSubtool === 'roi-free' ||
      activeMeasureSubtool === 'spline-roi' ||
      activeMeasureSubtool === 'livewire');
  const isCustomOverlayTool = isCustomMeasureTool || isCustomDrawTool;
  const isFreeformDrawTool =
    activeMeasureSubtool === 'roi-free' || activeMeasureSubtool === 'spline-roi' || activeMeasureSubtool === 'livewire';

  const requiredPointsForMeasureTool = (tool: MeasureSubtoolKey) => {
    if (tool === 'angle') return 3;
    if (
      tool === 'length' ||
      tool === 'bidirectional' ||
      tool === 'arrow' ||
      tool === 'ellipse' ||
      tool === 'rect' ||
      tool === 'circle'
    ) {
      return 2;
    }
    if (tool === 'text') return 1;
    return 0;
  };

  const clientToOverlayPoint = (clientX: number, clientY: number): ImagePoint | null => {
    const displayRect = panoDisplayRef.current?.getBoundingClientRect();
    if (!displayRect || !overlayCoordinateSize.width || !overlayCoordinateSize.height) return null;
    if (
      clientX < displayRect.left ||
      clientX > displayRect.right ||
      clientY < displayRect.top ||
      clientY > displayRect.bottom
    ) {
      return null;
    }

    const normalizedX = clamp((clientX - displayRect.left) / Math.max(1, displayRect.width), 0, 1);
    const normalizedY = clamp((clientY - displayRect.top) / Math.max(1, displayRect.height), 0, 1);

    return {
      x: (flipped ? 1 - normalizedX : normalizedX) * overlayCoordinateSize.width,
      y: normalizedY * overlayCoordinateSize.height,
    };
  };

  const overlayPointToViewportPosition = (point: ImagePoint) => {
    const displayRect = panoDisplayRef.current?.getBoundingClientRect();
    const viewportRect = panoViewportRef.current?.getBoundingClientRect();
    if (!displayRect || !viewportRect || !overlayCoordinateSize.width || !overlayCoordinateSize.height) return null;

    const normalizedX = clamp(point.x / Math.max(1, overlayCoordinateSize.width), 0, 1);
    const normalizedY = clamp(point.y / Math.max(1, overlayCoordinateSize.height), 0, 1);
    const renderedX = (flipped ? 1 - normalizedX : normalizedX) * displayRect.width;
    const renderedY = normalizedY * displayRect.height;

    return {
      left: displayRect.left - viewportRect.left + renderedX,
      top: displayRect.top - viewportRect.top + renderedY,
    };
  };

  const commitPendingTextAnnotation = () => {
    if (!pendingTextAnnotation) return;
    const nextText = pendingTextAnnotation.text.trim();
    if (nextText) {
      setMeasureShapes((current) => [
        ...current,
        {
          type: 'text',
          points: [pendingTextAnnotation.point],
          text: nextText,
          anchorLeft: pendingTextAnnotation.anchorLeft,
          anchorTop: pendingTextAnnotation.anchorTop,
        },
      ]);
    }
    setPendingTextAnnotation(null);
  };

  const cancelPendingTextAnnotation = () => {
    setPendingTextAnnotation(null);
  };

  const dismissPendingTextAnnotation = (event?: {
    preventDefault?: () => void;
    stopPropagation?: () => void;
  }) => {
    event?.preventDefault?.();
    event?.stopPropagation?.();
    cancelPendingTextAnnotation();
  };

  useEffect(() => {
    if (!pendingTextAnnotation) {
      textAnnotationWasOpenRef.current = false;
      return;
    }
    if (textAnnotationWasOpenRef.current) return;
    textAnnotationWasOpenRef.current = true;
    textAnnotationInputRef.current?.focus();
  }, [pendingTextAnnotation]);

  const focusedCondition = useMemo(() => {
    if (activeTooth) return toothStatusByFdi[activeTooth] || 'healthy';
    if (activeDetection?.toothFdi) return toothStatusByFdi[activeDetection.toothFdi] || 'healthy';
    return null;
  }, [activeDetection?.toothFdi, activeTooth, toothStatusByFdi]);

  const finalizeFreeformShape = (finalPoint?: ImagePoint | null) => {
    if (!isCustomDrawTool || !isFreeformDrawTool || pendingMeasurePoints.length === 0) return false;

    const nextPoints = [...pendingMeasurePoints];
    if (finalPoint) {
      const lastPoint = nextPoints[nextPoints.length - 1];
      if (!lastPoint || Math.hypot(finalPoint.x - lastPoint.x, finalPoint.y - lastPoint.y) > 1.5) {
        nextPoints.push(finalPoint);
      }
    }

    if (nextPoints.length >= 2) {
      setMeasureShapes((current) => [...current, { type: activeMeasureSubtool, points: nextPoints }]);
    }
    setPendingMeasurePoints([]);
    setMeasurePreviewPoint(null);
    return true;
  };

  const renderMeasureShape = (shape: MeasureShape, isTemp = false) => {
    const points = isTemp && measurePreviewPoint ? [...shape.points, measurePreviewPoint] : shape.points;
    if (!points.length) return null;

    const stroke = 'rgba(57, 255, 20, 0.98)';
    const glow = 'rgba(196, 255, 168, 0.92)';
    const fill = 'rgba(57, 255, 20, 0.14)';
    const strokeWidth = 1.2 / effectiveScale;
    const distance = (a: ImagePoint, b: ImagePoint) => Math.hypot(b.x - a.x, b.y - a.y);

    const renderLabel = (x: number, y: number, lines: string[]) => (
      <g>
        <rect
          x={x - 6 / effectiveScale}
          y={y - 14 / effectiveScale}
          width={Math.max(...lines.map((line) => line.length), 1) * (5.2 / effectiveScale) + 12 / effectiveScale}
          height={lines.length * (12 / effectiveScale) + 8 / effectiveScale}
          fill="rgba(0, 0, 0, 0.7)"
          rx={3 / effectiveScale}
        />
        {lines.map((line, index) => (
          <text
            key={`${line}-${index}`}
            x={x}
            y={y + index * (11 / effectiveScale)}
            fill="#FFFFFF"
            fontSize={9 / effectiveScale}
            fontWeight="700"
          >
            {line}
          </text>
        ))}
      </g>
    );

    if (shape.type === 'length' && points.length >= 2) {
      const value = `${(distance(points[0], points[1]) * mmPerPixel).toFixed(2)} mm`;
      return (
        <g>
          <line x1={points[0].x} y1={points[0].y} x2={points[1].x} y2={points[1].y} stroke={glow} strokeWidth={strokeWidth + 1 / effectiveScale} />
          <line x1={points[0].x} y1={points[0].y} x2={points[1].x} y2={points[1].y} stroke={stroke} strokeWidth={strokeWidth} />
          {!isTemp ? renderLabel((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2 - 8 / effectiveScale, [value]) : null}
        </g>
      );
    }

    if (shape.type === 'bidirectional' && points.length >= 2) {
      const x = Math.min(points[0].x, points[1].x);
      const y = Math.min(points[0].y, points[1].y);
      const w = Math.abs(points[1].x - points[0].x);
      const h = Math.abs(points[1].y - points[0].y);
      const cx = x + w / 2;
      const cy = y + h / 2;
      return (
        <g>
          <rect x={x} y={y} width={w} height={h} fill={fill} stroke={glow} strokeWidth={strokeWidth + 1 / effectiveScale} />
          <rect x={x} y={y} width={w} height={h} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />
          <line x1={x} y1={cy} x2={x + w} y2={cy} stroke={glow} strokeWidth={strokeWidth + 1 / effectiveScale} />
          <line x1={x} y1={cy} x2={x + w} y2={cy} stroke={stroke} strokeWidth={strokeWidth} />
          <line x1={cx} y1={y} x2={cx} y2={y + h} stroke={glow} strokeWidth={strokeWidth + 1 / effectiveScale} />
          <line x1={cx} y1={y} x2={cx} y2={y + h} stroke={stroke} strokeWidth={strokeWidth} />
          {!isTemp
            ? renderLabel(x + w + 12 / effectiveScale, y + h - 4 / effectiveScale, [
                `W ${(w * mmPerPixel).toFixed(2)} mm`,
                `H ${(h * mmPerPixel).toFixed(2)} mm`,
              ])
            : null}
        </g>
      );
    }

    if (shape.type === 'angle' && points.length >= 2) {
      if (points.length === 2) {
        return (
          <g>
            <line x1={points[0].x} y1={points[0].y} x2={points[1].x} y2={points[1].y} stroke={glow} strokeWidth={strokeWidth + 1 / effectiveScale} />
            <line x1={points[0].x} y1={points[0].y} x2={points[1].x} y2={points[1].y} stroke={stroke} strokeWidth={strokeWidth} />
          </g>
        );
      }
      const a = distance(points[1], points[0]);
      const b = distance(points[1], points[2]);
      const c = distance(points[0], points[2]);
      const val = (a * a + b * b - c * c) / Math.max(2 * a * b, 1e-6);
      const deg = Math.acos(Math.max(-1, Math.min(1, val))) * (180 / Math.PI);
      return (
        <g>
          <polyline
            points={points.map((point) => `${point.x},${point.y}`).join(' ')}
            fill="none"
            stroke={glow}
            strokeWidth={strokeWidth + 1 / effectiveScale}
          />
          <polyline
            points={points.map((point) => `${point.x},${point.y}`).join(' ')}
            fill="none"
            stroke={stroke}
            strokeWidth={strokeWidth}
          />
          {!isTemp ? renderLabel(points[1].x, points[1].y - 8 / effectiveScale, [`${deg.toFixed(1)}°`]) : null}
        </g>
      );
    }

    return null;
  };

  const renderCustomShape = (shape: MeasureShape, isTemp = false) => {
    if (
      shape.type === 'length' ||
      shape.type === 'bidirectional' ||
      shape.type === 'angle'
    ) {
      return renderMeasureShape(shape, isTemp);
    }

    const points = isTemp && measurePreviewPoint ? [...shape.points, measurePreviewPoint] : shape.points;
    if (!points.length) return null;

    const stroke = 'rgba(57, 255, 20, 0.98)';
    const glow = 'rgba(196, 255, 168, 0.92)';
    const strokeWidth = 1.2 / effectiveScale;
    const fill = 'rgba(57, 255, 20, 0.14)';
    const distance = (a: ImagePoint, b: ImagePoint) => Math.hypot(b.x - a.x, b.y - a.y);

    const renderLabel = (x: number, y: number, lines: string[]) => (
      <g>
        <rect
          x={x - 6 / effectiveScale}
          y={y - 14 / effectiveScale}
          width={Math.max(...lines.map((line) => line.length), 1) * (5.2 / effectiveScale) + 12 / effectiveScale}
          height={lines.length * (12 / effectiveScale) + 8 / effectiveScale}
          fill="rgba(0, 0, 0, 0.7)"
          rx={3 / effectiveScale}
        />
        {lines.map((line, index) => (
          <text
            key={`${line}-${index}`}
            x={x}
            y={y + index * (11 / effectiveScale)}
            fill="#FFFFFF"
            fontSize={9 / effectiveScale}
            fontWeight="700"
          >
            {line}
          </text>
        ))}
      </g>
    );

    if (shape.type === 'text') return null;

    if (shape.type === 'arrow' && points.length >= 2) {
      return (
        <g>
          <line
            x1={points[0].x}
            y1={points[0].y}
            x2={points[1].x}
            y2={points[1].y}
            stroke={glow}
            strokeWidth={strokeWidth + 1 / effectiveScale}
            markerEnd="url(#renewMeasureArrowGlow)"
          />
          <line
            x1={points[0].x}
            y1={points[0].y}
            x2={points[1].x}
            y2={points[1].y}
            stroke={stroke}
            strokeWidth={strokeWidth}
            markerEnd="url(#renewMeasureArrow)"
          />
        </g>
      );
    }

    if (shape.type === 'rect' && points.length >= 2) {
      const x = Math.min(points[0].x, points[1].x);
      const y = Math.min(points[0].y, points[1].y);
      const w = Math.abs(points[1].x - points[0].x);
      const h = Math.abs(points[1].y - points[0].y);
      return (
        <g>
          <rect x={x} y={y} width={w} height={h} fill={fill} stroke={glow} strokeWidth={strokeWidth + 1 / effectiveScale} />
          <rect x={x} y={y} width={w} height={h} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />
        </g>
      );
    }

    if (shape.type === 'ellipse' && points.length >= 2) {
      const x = Math.min(points[0].x, points[1].x);
      const y = Math.min(points[0].y, points[1].y);
      const w = Math.abs(points[1].x - points[0].x);
      const h = Math.abs(points[1].y - points[0].y);
      return (
        <g>
          <ellipse
            cx={x + w / 2}
            cy={y + h / 2}
            rx={w / 2}
            ry={h / 2}
            fill={fill}
            stroke={glow}
            strokeWidth={strokeWidth + 1 / effectiveScale}
          />
          <ellipse cx={x + w / 2} cy={y + h / 2} rx={w / 2} ry={h / 2} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />
        </g>
      );
    }

    if (shape.type === 'circle' && points.length >= 2) {
      const radius = distance(points[0], points[1]);
      return (
        <g>
          <circle cx={points[0].x} cy={points[0].y} r={radius} fill={fill} stroke={glow} strokeWidth={strokeWidth + 1 / effectiveScale} />
          <circle cx={points[0].x} cy={points[0].y} r={radius} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />
        </g>
      );
    }

    if ((shape.type === 'roi-free' || shape.type === 'livewire') && points.length >= 2) {
      const polygonPoints = points.map((point) => `${point.x},${point.y}`).join(' ');
      return isTemp || points.length < 3 ? (
        <g>
          <polyline points={polygonPoints} fill="none" stroke={glow} strokeWidth={strokeWidth + 1 / effectiveScale} />
          <polyline points={polygonPoints} fill="none" stroke={stroke} strokeWidth={strokeWidth} />
        </g>
      ) : (
        <g>
          <polygon points={polygonPoints} fill={fill} stroke={glow} strokeWidth={strokeWidth + 1 / effectiveScale} />
          <polygon points={polygonPoints} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />
        </g>
      );
    }

    if (shape.type === 'spline-roi' && points.length >= 2) {
      const path = buildSmoothPath(points, !isTemp);
      return (
        <g>
          <path d={path} fill={isTemp ? 'none' : fill} stroke={glow} strokeWidth={strokeWidth + 1 / effectiveScale} />
          <path d={path} fill={isTemp ? 'none' : fill} stroke={stroke} strokeWidth={strokeWidth} />
        </g>
      );
    }

    return null;
  };

  const isPointNearShape = (point: ImagePoint, shape: MeasureShape) => {
    const tolerance = 14 / effectiveScale;
    const points = shape.points;
    if (!points.length) return false;

    if (shape.type === 'text') {
      return distanceBetweenPoints(point, points[0]) <= 28 / effectiveScale;
    }

    if (shape.type === 'length' || shape.type === 'arrow') {
      return points.length >= 2 && distancePointToSegment(point, points[0], points[1]) <= tolerance;
    }

    if (shape.type === 'bidirectional' || shape.type === 'rect') {
      if (points.length < 2) return false;
      const x1 = Math.min(points[0].x, points[1].x);
      const y1 = Math.min(points[0].y, points[1].y);
      const x2 = Math.max(points[0].x, points[1].x);
      const y2 = Math.max(points[0].y, points[1].y);
      const expanded = { x1: x1 - tolerance, y1: y1 - tolerance, x2: x2 + tolerance, y2: y2 + tolerance };
      return point.x >= expanded.x1 && point.x <= expanded.x2 && point.y >= expanded.y1 && point.y <= expanded.y2;
    }

    if (shape.type === 'ellipse') {
      if (points.length < 2) return false;
      const cx = (points[0].x + points[1].x) / 2;
      const cy = (points[0].y + points[1].y) / 2;
      const rx = Math.abs(points[1].x - points[0].x) / 2;
      const ry = Math.abs(points[1].y - points[0].y) / 2;
      if (rx < 1 || ry < 1) return false;
      const normalized = ((point.x - cx) ** 2) / ((rx + tolerance) ** 2) + ((point.y - cy) ** 2) / ((ry + tolerance) ** 2);
      return normalized <= 1.05;
    }

    if (shape.type === 'circle') {
      if (points.length < 2) return false;
      return distanceBetweenPoints(point, points[0]) <= distanceBetweenPoints(points[0], points[1]) + tolerance;
    }

    if (shape.type === 'angle') {
      if (points.length < 2) return false;
      if (points.length === 2) return distancePointToSegment(point, points[0], points[1]) <= tolerance;
      return (
        distancePointToSegment(point, points[1], points[0]) <= tolerance ||
        distancePointToSegment(point, points[1], points[2]) <= tolerance
      );
    }

    if (shape.type === 'roi-free' || shape.type === 'livewire' || shape.type === 'spline-roi') {
      if (points.length >= 3 && pointInPolygon(point, points)) return true;
      for (let index = 1; index < points.length; index += 1) {
        if (distancePointToSegment(point, points[index - 1], points[index]) <= tolerance) return true;
      }
      if ((shape.type === 'livewire' || shape.type === 'spline-roi') && points.length >= 3) {
        if (distancePointToSegment(point, points[points.length - 1], points[0]) <= tolerance) return true;
      }
    }

    return false;
  };

  const renderAIDetections = () => {
    if (viewMode !== 'overlay' || !result) return null;

    const items: any[] = [];
    const shouldFocusSingleTooth = Boolean(activeTooth);
    const shouldShowSinus = !shouldFocusSingleTooth && (overlayPreset === 'all' || overlayPreset === 'sinus' || overlayPreset === 'sinus-upper-tooth');
    const shouldShowNerve = !shouldFocusSingleTooth && (overlayPreset === 'all' || overlayPreset === 'nerve' || overlayPreset === 'nerve-lower-tooth');
    const shouldShowUpperTooth = overlayPreset === 'all' || overlayPreset === 'tooth' || overlayPreset === 'sinus-upper-tooth';
    const shouldShowLowerTooth = overlayPreset === 'all' || overlayPreset === 'tooth' || overlayPreset === 'nerve-lower-tooth';
    const matchesFilter = (toothFdi: string | null | undefined) => {
      if (!activeLegendFilter) return true;
      if (!toothFdi) return false;
      return (toothStatusByFdi[toothFdi] || 'healthy') === activeLegendFilter;
    };
    const matchesToothSelection = (toothFdi: string | null | undefined) => {
      if (!activeTooth) return true;
      return toothFdi === activeTooth;
    };

    if (shouldShowSinus && Array.isArray(result.sinus_contours)) {
      result.sinus_contours.forEach((contour: any, idx: number) => {
        if (!Array.isArray(contour)) return;
        const runs = buildLowerContourRuns(contour, 0.25);
        runs.forEach((run, runIdx) => {
          const points = run.map((pt) => `${pt.x},${pt.y}`).join(' ');
          items.push(
            <polyline
              key={`renew-sinus-${idx}-${runIdx}`}
              points={points}
              fill="none"
              stroke="#ff4444"
              strokeWidth={1.2 / effectiveScale}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          );
        });
      });
    }

    if (shouldShowNerve && Array.isArray(result.nerve_contours)) {
      result.nerve_contours.forEach((contour: any, idx: number) => {
        if (!Array.isArray(contour) || contour.length < 2) return;
        const points = contour.map((pt: any) => `${pt[0]},${pt[1]}`).join(' ');
        items.push(
          <polyline
            key={`renew-nerve-${idx}`}
            points={points}
            fill="none"
            stroke="rgba(255, 0, 255, 0.32)"
            strokeWidth={1 / effectiveScale}
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        );
      });
    }

    toothGeometries.forEach((tooth, idx) => {
      if (!matchesFilter(tooth.fdi) || !matchesToothSelection(tooth.fdi)) return;
      const isUpperTooth = tooth.fdi.startsWith('1') || tooth.fdi.startsWith('2');
      if (isUpperTooth && !shouldShowUpperTooth) return;
      if (!isUpperTooth && !shouldShowLowerTooth) return;
      const paletteIndex = Number.isFinite(Number(tooth.fdi)) ? Math.abs(Number(tooth.fdi)) % warmPastelPalette.length : 0;
      const style = warmPastelPalette[paletteIndex];
      const toothStatus = toothStatusByFdi[tooth.fdi] || 'healthy';
      const isActive = activeTooth === tooth.fdi;
      const hasDetection = Boolean(detectionsByTooth[tooth.fdi]?.length);
      const points = tooth.contour.map((point) => `${point.x},${point.y}`).join(' ');
      const fillColor = isActive ? style.fill.replace('0.26', '0.45') : style.fill;
      const strokeColor = isActive
        ? 'rgba(0, 192, 243, 0.18)'
        : toothStatus === 'implant'
          ? 'rgba(0, 61, 255, 0.72)'
          : hasDetection
            ? 'rgba(255, 215, 102, 0.2)'
            : style.stroke;

      items.push(
        <g
          key={`renew-tooth-${tooth.fdi}-${idx}`}
          style={{ pointerEvents: 'auto', cursor: 'pointer' }}
          onClick={() => handleToothSelect(tooth.fdi)}
        >
          <polygon
            points={points}
            fill={fillColor}
            stroke={strokeColor}
            strokeWidth={(isActive ? 0.14 : hasDetection ? 0.8 : 2) / effectiveScale}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
          <g transform={`translate(${tooth.contour[0].x}, ${tooth.contour[0].y - 15 / effectiveScale})`}>
            <rect
              x={-10 / effectiveScale}
              y={-10 / effectiveScale}
              width={20 / effectiveScale}
              height={14 / effectiveScale}
              rx={4 / effectiveScale}
              fill={isActive ? 'rgba(0, 192, 243, 0.88)' : 'rgba(0,0,0,0.62)'}
            />
            <text
              textAnchor="middle"
              fill="#ffffff"
              fontSize={10 / effectiveScale}
              fontWeight="bold"
              transform={flipped ? 'scale(-1,1)' : undefined}
            >
              {formatDisplayedToothNumber(tooth.fdi, numberingSystem, flipped)}
            </text>
          </g>
        </g>
      );
    });

    normalizedDetections.forEach((detection) => {
      if (!matchesFilter(detection.toothFdi) || !matchesToothSelection(detection.toothFdi)) return;
      const isActive = activeDetectionId === detection.id;
      const matchesTooth = Boolean(activeTooth && detection.toothFdi === activeTooth);
      const strokeColor =
        detection.type === 'caries'
          ? '#ff4444'
          : detection.type === 'periapical'
            ? '#ff9800'
            : detection.type === 'implant'
              ? '#003DFF'
            : detection.type === 'bonelevel'
              ? '#FFD766'
              : '#FFD766';

      if (detection.contour && detection.contour.length >= 3) {
        items.push(
          <polygon
            key={detection.id}
            points={detection.contour.map((point) => `${point.x},${point.y}`).join(' ')}
            fill={isActive ? 'rgba(0, 192, 243, 0.02)' : matchesTooth ? 'rgba(255, 215, 102, 0.03)' : 'transparent'}
            stroke={isActive ? 'rgba(0, 192, 243, 0.55)' : strokeColor}
            strokeWidth={(isActive ? 1.4 : matchesTooth ? 1.4 : 1.0) / effectiveScale}
            vectorEffect="non-scaling-stroke"
            style={{ pointerEvents: 'auto', cursor: 'pointer' }}
            onClick={() => handleDetectionSelect(detection)}
          />
        );
        return;
      }

      const width = detection.bounds.x2 - detection.bounds.x1;
      const height = detection.bounds.y2 - detection.bounds.y1;
      const tagHeight = 14 / effectiveScale;
      const tagPadding = 6 / effectiveScale;
      const displayedToothNumber = detection.toothFdi
        ? formatDisplayedToothNumber(detection.toothFdi, numberingSystem, flipped)
        : '';
      const labelText = `${detection.label} ${displayedToothNumber}`.trim();
      const fontSize = 9 / effectiveScale;
      const tagWidth = labelText.length * (fontSize * 0.6) + tagPadding * 2;
      const textX = detection.bounds.x1 + tagPadding;
      const labelGroupTransform = flipped
        ? `translate(${2 * detection.bounds.x1 + tagWidth}, 0) scale(-1,1)`
        : undefined;

      items.push(
        <g
          key={detection.id}
          style={{ pointerEvents: 'auto', cursor: 'pointer' }}
          onClick={() => handleDetectionSelect(detection)}
        >
          <rect
            x={detection.bounds.x1}
            y={detection.bounds.y1}
            width={width}
            height={height}
            fill={isActive ? 'rgba(0, 192, 243, 0.03)' : 'transparent'}
            stroke={isActive ? '#00C0F3' : strokeColor}
            strokeWidth={(isActive ? 2.4 : matchesTooth ? 1.6 : 0.9) / effectiveScale}
            strokeDasharray={detection.type === 'bonelevel' ? undefined : `${3 / effectiveScale} ${2 / effectiveScale}`}
            vectorEffect="non-scaling-stroke"
          />
          <g transform={labelGroupTransform}>
            <rect
              x={detection.bounds.x1}
              y={detection.bounds.y1 - tagHeight}
              width={tagWidth}
              height={tagHeight}
              rx={3 / effectiveScale}
              ry={3 / effectiveScale}
              fill={isActive ? '#00C0F3' : strokeColor}
              fillOpacity={0.92}
            />
            <text
              x={textX}
              y={detection.bounds.y1 - tagHeight / 2 + fontSize / 3}
              fill="#fff"
              fontSize={fontSize}
              fontWeight="bold"
            >
              {labelText}
            </text>
          </g>
        </g>
      );
    });

    return <g id="renew-ai-overlay-layer">{items}</g>;
  };

  const renderRiskDetections = () => {
    if (viewMode !== 'heatmap' || !result) return null;

    const items: any[] = [];
    const heatmapOpacityScale = heatmapOpacity / 100;
    const shouldFocusSingleTooth = Boolean(activeTooth);
    const shouldShowNerve =
      !shouldFocusSingleTooth &&
      (overlayPreset === 'all' || overlayPreset === 'nerve' || overlayPreset === 'nerve-lower-tooth');
    const cariesEntries = Object.entries(result.caries_by_tooth_best || result.analysis_result?.caries_by_tooth_best || {});
    const periapicalEntries = Object.entries(result.periapical_by_tooth_best || result.analysis_result?.periapical_by_tooth_best || {});
    const teeth = Array.isArray(result.teeth)
      ? result.teeth
      : Array.isArray(result.teeth_objects)
        ? result.teeth_objects
        : [];
    const matchesFilter = (toothFdi: string | null | undefined) => {
      if (!activeLegendFilter) return true;
      if (!toothFdi) return false;
      return (toothStatusByFdi[toothFdi] || 'healthy') === activeLegendFilter;
    };
    const matchesToothSelection = (toothFdi: string | null | undefined) => {
      if (!activeTooth) return true;
      return toothFdi === activeTooth;
    };

    if (!hasHeatmapAsset) {
      cariesEntries.forEach(([tooth, data]: any, idx) => {
        const toothFdi = String(tooth);
        if (!matchesFilter(toothFdi) || !matchesToothSelection(toothFdi)) return;
        const box = data?.box;
        if (!box || box.length < 4) return;
        const conf = Number(data?.conf || 0.65);
        const [x1, y1, x2, y2] = box;
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        const rx = Math.max((x2 - x1) * 0.85, 18);
        const ry = Math.max((y2 - y1) * 0.85, 18);
        const opacity = Math.min(0.5, 0.18 + conf * 0.22);
        items.push(
          <g
            key={`renew-risk-caries-${tooth}-${idx}`}
            filter="url(#renewRiskBlurStrong)"
            style={{ mixBlendMode: 'normal', opacity: 0.62 * heatmapOpacityScale }}
          >
            <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="url(#renewCariesGlow)" fillOpacity={opacity} />
          </g>
        );
      });

      periapicalEntries.forEach(([tooth, data]: any, idx) => {
        const toothFdi = String(tooth);
        if (!matchesFilter(toothFdi) || !matchesToothSelection(toothFdi)) return;
        const box = data?.box;
        if (!box || box.length < 4) return;
        const conf = Number(data?.conf || 0.72);
        const [x1, y1, x2, y2] = box;
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        const rx = Math.max((x2 - x1) * 0.95, 22);
        const ry = Math.max((y2 - y1) * 0.95, 22);
        const opacity = Math.min(0.52, 0.2 + conf * 0.22);
        items.push(
          <g
            key={`renew-risk-peri-${tooth}-${idx}`}
            filter="url(#renewRiskBlurStrong)"
            style={{ mixBlendMode: 'normal', opacity: 0.6 * heatmapOpacityScale }}
          >
            <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="url(#renewPeriGlow)" fillOpacity={opacity} />
          </g>
        );
      });
    }

    if (shouldShowNerve && Array.isArray(result.nerve_contours)) {
      result.nerve_contours.forEach((contour: any, idx: number) => {
        if (!Array.isArray(contour) || contour.length < 2) return;
        const points = contour.map((pt: any) => `${pt[0]},${pt[1]}`).join(' ');
        items.push(
          <g
            key={`renew-risk-nerve-${idx}`}
            style={{ mixBlendMode: 'screen', opacity: (hasHeatmapAsset ? 0.5 : 0.42) * heatmapOpacityScale }}
          >
            {contour.length >= 3 ? (
              <polygon
                points={points}
                fill={hasHeatmapAsset ? 'rgba(255, 0, 255, 0.12)' : 'rgba(255, 0, 255, 0.08)'}
                filter="url(#renewRiskBlurSoft)"
              />
            ) : null}
            <polyline
              points={points}
              fill="none"
              stroke={hasHeatmapAsset ? 'rgba(255, 88, 255, 0.38)' : 'rgba(255, 88, 255, 0.28)'}
              strokeWidth={(hasHeatmapAsset ? 5 : 4) / effectiveScale}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
              filter="url(#renewRiskBlurStrong)"
            />
            <polyline
              points={points}
              fill="none"
              stroke={hasHeatmapAsset ? 'rgba(255, 240, 255, 0.72)' : 'rgba(255, 232, 255, 0.58)'}
              strokeWidth={(hasHeatmapAsset ? 1.2 : 1) / effectiveScale}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          </g>
        );
      });
    }

    if (!hasHeatmapAsset) {
      teeth.forEach((tooth: any, idx: number) => {
        const label = String(tooth?.tooth_label || tooth?.label || tooth?.tooth || '');
        if (!matchesFilter(label) || !matchesToothSelection(label)) return;
        const boneLossPct = Number(
          result?.bonelevel?.[label]?.percent ??
          tooth?.bone_loss_pct ??
          0
        );
        if (!label || boneLossPct < 15) return;
        const severity = Math.min(1, Math.max(0, (boneLossPct - 15) / 30));
        const opacity = Math.min(0.32, 0.12 + severity * 0.16);
        const green = Math.round(224 - severity * 140);
        const fill = `rgba(255,${green},71,${opacity})`;
        const contour = tooth?.contour;
        if (!Array.isArray(contour) || contour.length < 3) return;
        const points = contour.map((pt: any) => `${pt[0]},${pt[1]}`).join(' ');
        items.push(
          <g
            key={`renew-risk-bone-${label}-${idx}`}
            filter="url(#renewRiskBlurSoft)"
            style={{ mixBlendMode: 'normal', opacity: 0.48 * heatmapOpacityScale }}
          >
            <polygon points={points} fill={fill} />
          </g>
        );
      });
    }

    return <g id="renew-risk-overlay-layer">{items}</g>;
  };

  const handleOpenStudies = () => {
    setWorkspaceSection((current) => (current === 'studies' ? 'none' : 'studies'));
    setReportDrawerOpen(false);
    setIsReportActive(false);
    setReportError(null);
  };

  const handleOpenServerImage = async (image: ServerFolderImage) => {
    try {
      const imageUrl = resolveServerAssetUrl(image.downloadUrl);
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to load ${image.name}.`);
      }
      const blob = await response.blob();
      const originalFile = new File([blob], image.name, {
        type: blob.type || `image/${(image.format || 'png').toLowerCase()}`,
        lastModified: Date.now(),
      });

      navigate('/renew', {
        state: {
          previewUrl: imageUrl,
          imageUrl,
          originalFile,
          originalFileName: image.name,
          originalIsDicom: false,
          folderSource: 'server-image',
          imageRelativePath: image.relativePath,
          folderSelectedSeriesId: `${SERVER_IMAGE_SERIES_PREFIX}${image.relativePath}`,
          userName: image.patientName || locationState.userName || 'Patient',
          dicomInfo: image.dicomInfo || null,
          linkedStudyId: image.linkedStudyId || null,
        },
      });
    } catch (error) {
      console.error('Failed to open server image inside RenewPage', error);
    }
  };

  const handleSelectSeries = async (seriesId: string) => {
    if (seriesId.startsWith(SERVER_IMAGE_SERIES_PREFIX)) {
      const relativePath = seriesId.slice(SERVER_IMAGE_SERIES_PREFIX.length);
      const targetImage = serverImages.find((image) => image.relativePath === relativePath);
      if (targetImage) {
        await handleOpenServerImage(targetImage);
      }
      return;
    }

    const existingSeries = activeFolderStudies.flatMap((study) => study.series).find((series) => series.id === seriesId);
    if (existingSeries) {
      setSelectedFolderSeriesId(seriesId);
      handleOpenStudies();
      return;
    }

    const targetStudy = serverStudies.find((study) => study.series.some((series: any) => series.id === seriesId));
    if (!targetStudy) return;

    try {
      const materialized = await materializeServerStudy(targetStudy);
      if (!originalFolderMode) {
        navigate('/renew', {
          state: {
            originalFolderMode: true,
            originalFolderStudies: [materialized],
            folderSelectedSeriesId: seriesId,
            previewUrl: targetStudy.previewUrl ? resolveServerAssetUrl(targetStudy.previewUrl) : undefined,
            originalIsDicom: true,
            originalFileName:
              materialized.series.find((series) => series.id === seriesId)?.label ||
              materialized.label ||
              'DICOM Study',
            folderSource: 'server',
          },
        });
        return;
      }
      setActiveFolderStudies((current) => {
        if (current.some((study) => study.id === materialized.id)) return current;
        return [...current, materialized];
      });
      setSelectedFolderSeriesId(seriesId);
      handleOpenStudies();
    } catch (error) {
      console.error('Failed to materialize study inside RenewPage', error);
    }
  };

  const handleChartToggle = () => {
    setIsChartVisible((current) => !current);
  };

  useEffect(() => {
    if (locationState.folderSource === 'server' || locationState.folderSource === 'server-image') {
      fetchServerFolderIndex()
        .then((data) => {
          setServerStudies(data.studies || []);
          setServerImages(data.images || []);
        })
        .catch(console.error);
    }
  }, [locationState.folderSource]);

  useEffect(() => {
    if (!originalFolderMode) return;
    if (previousSelectedFolderSeriesIdRef.current === selectedFolderSeriesId) return;

    previousSelectedFolderSeriesIdRef.current = selectedFolderSeriesId;
    autoAnalyzeTriggeredRef.current = false;
    pollStartedAtRef.current = null;
    status404CountRef.current = 0;
    setResult(null);
    setJobId(null);
    setIsProcessing(false);
    setReportSessionId(null);
    setReportDrawerOpen(false);
    setIsReportActive(false);
    setReportError(null);
    setWorkspaceSection('none');
    setViewMode('original');
    setDicomHudMetadata(null);
    setDicomPreviewDataUrl(null);
    setDicomAutoWindow(null);
  }, [originalFolderMode, selectedFolderSeriesId]);

  useEffect(() => {
    let cancelled = false;

    if (!originalIsDicom || !dicomHudFile) {
      setDicomHudMetadata(sidecarDicomInfo && !originalIsDicom ? sidecarDicomInfo : null);
      setDicomPreviewDataUrl(null);
      setDicomAutoWindow(null);
      return;
    }

    const loadHudMetadata = async () => {
      try {
        const [arrayBuffer, inspection] = await Promise.all([
          dicomHudFile.arrayBuffer(),
          inspectLocalDicomFile(dicomHudFile).catch(() => null),
        ]);
        const byteArray = new Uint8Array(arrayBuffer);
        const dataSet = dicomParser.parseDicom(byteArray, { untilTag: 'x7fe00010' });
        const pixelElement = dataSet.elements.x7fe00010;
        const rows = dataSet.uint16('x00280010') || 0;
        const columns = dataSet.uint16('x00280011') || 0;
        const samplesPerPixel = dataSet.uint16('x00280002') || 1;
        const bitsAllocated = dataSet.uint16('x00280100') || 16;
        const pixelRepresentation = dataSet.uint16('x00280103') || 0;
        const rescaleSlope = Number(dataSet.string('x00281053') || '1') || 1;
        const rescaleIntercept = Number(dataSet.string('x00281052') || '0') || 0;
        let nextAutoWindow: { level: number; width: number } | null = null;

        if (pixelElement && rows > 0 && columns > 0 && samplesPerPixel === 1 && (bitsAllocated === 8 || bitsAllocated === 16)) {
          const bytesPerPixel = Math.max(1, bitsAllocated / 8);
          const singleFrameBytes = rows * columns * samplesPerPixel * bytesPerPixel;
          let frameOffset = pixelElement.dataOffset;
          if (pixelElement.length > singleFrameBytes && singleFrameBytes > 0) {
            const numberOfFrames = Math.floor(pixelElement.length / singleFrameBytes);
            frameOffset += Math.floor(numberOfFrames / 2) * singleFrameBytes;
          }

          const pixelBytes = dataSet.byteArray.slice(frameOffset, frameOffset + singleFrameBytes);
          let scalarData: Uint8Array | Int8Array | Uint16Array | Int16Array | null = null;
          if (bitsAllocated === 8) {
            scalarData = pixelRepresentation === 1
              ? new Int8Array(pixelBytes.buffer, pixelBytes.byteOffset, pixelBytes.byteLength)
              : pixelBytes;
          } else if (bitsAllocated === 16) {
            scalarData = pixelRepresentation === 1
              ? new Int16Array(pixelBytes.buffer, pixelBytes.byteOffset, pixelBytes.byteLength / 2)
              : new Uint16Array(pixelBytes.buffer, pixelBytes.byteOffset, pixelBytes.byteLength / 2);
          }

          const autoWindow = scalarData
            ? estimateAutoWindowFromPixelData(scalarData, rescaleSlope, rescaleIntercept)
            : null;
          nextAutoWindow = autoWindow
            ? { level: Math.round(autoWindow.level), width: Math.round(autoWindow.width) }
            : null;
        }

        if (!cancelled) {
          setDicomHudMetadata(parseDicomMetadataFromDataSet(dataSet, dicomHudFile.name));
          setDicomPreviewDataUrl(inspection?.previewDataUrl || null);
          setDicomAutoWindow(nextAutoWindow);
        }
      } catch (error) {
        console.warn('Failed to parse RenewPage DICOM HUD metadata', error);
        if (!cancelled) {
          setDicomHudMetadata(null);
          setDicomPreviewDataUrl(null);
          setDicomAutoWindow(null);
        }
      }
    };

    void loadHudMetadata();

    return () => {
      cancelled = true;
    };
  }, [dicomHudFile, originalIsDicom, sidecarDicomInfo]);

  useEffect(() => {
    if (autoAnalyzeTriggeredRef.current) return;
    if (result || jobId) return;

    const autoAnalyzeFile =
      originalFolderMode
        ? (selectedFolderSeries?.files?.[0] || null)
        : (originalFile || null);

    if (!autoAnalyzeFile) return;

    autoAnalyzeTriggeredRef.current = true;
    setIsProcessing(true);
    setReportError(null);

    void requestAsyncDetection(autoAnalyzeFile, [])
      .then((state) => {
        setJobId(state.jobId);
      })
      .catch((error) => {
        console.error('RenewPage auto analysis request failed', error);
        setIsProcessing(false);
        setReportError(error?.message || 'Failed to request analysis');
        autoAnalyzeTriggeredRef.current = false;
      });
  }, [
    jobId,
    locationState.originalIsDicom,
    originalFile,
    originalFolderMode,
    result,
    selectedFolderSeries,
  ]);

  useEffect(() => {
    let timer: number | undefined;

    if (jobId && !result) {
      setIsProcessing(true);
      if (pollStartedAtRef.current === null) {
        pollStartedAtRef.current = Date.now();
        status404CountRef.current = 0;
      }
      timer = window.setInterval(async () => {
        try {
          let response = await fetch(`/api/detect/status/${jobId}`);
          const contentType = response.headers.get('content-type') || '';
          if (!contentType.includes('application/json')) {
            response = await fetch(withDirectApiBase(`/api/detect/status/${jobId}`));
          }

          if (!response.ok) {
            if (response.status === 404) {
              status404CountRef.current += 1;
              const startedAt = pollStartedAtRef.current ?? Date.now();
              const elapsed = Date.now() - startedAt;
              if (status404CountRef.current >= 20 || elapsed > 90000) {
                window.clearInterval(timer);
                setIsProcessing(false);
                setReportError('Analysis status was not found. Please try again.');
                setJobId(null);
                pollStartedAtRef.current = null;
                status404CountRef.current = 0;
              }
            }
            return;
          }

          status404CountRef.current = 0;

          const data = await readJsonOrThrow<any>(response);
          if (data.success && data.status === 'done' && data.result) {
            window.clearInterval(timer);
            setResult(data.result);
            setJobId(null);
            setIsProcessing(false);
            pollStartedAtRef.current = null;
            status404CountRef.current = 0;
          } else if (data.status === 'failed') {
            window.clearInterval(timer);
            setJobId(null);
            setIsProcessing(false);
            setReportError(data.error || 'Analysis failed');
            pollStartedAtRef.current = null;
            status404CountRef.current = 0;
          }
        } catch (error) {
          console.error('RenewPage polling error:', error);
        }
      }, 1500);
    }

    return () => {
      if (timer) window.clearInterval(timer);
    };
  }, [jobId, result]);

  useEffect(() => {
    if (!isProcessing) {
      setLoadingProgress(0);
      return;
    }

    setLoadingProgress((current) => (current > 8 ? current : 8));
    const timer = window.setInterval(() => {
      setLoadingProgress((current) => {
        if (current < 42) return Math.min(42, current + 10);
        if (current < 68) return Math.min(68, current + 6);
        if (current < 86) return Math.min(86, current + 3);
        if (current < 94) return Math.min(94, current + 1);
        return current;
      });
    }, 260);

    return () => window.clearInterval(timer);
  }, [isProcessing]);

  useEffect(() => {
    writeStoredNumberingSystem(numberingSystem);
  }, [numberingSystem]);

  useEffect(() => {
    if (!result) return;

    const nextHasStructuredOverlay = Boolean(
      (Array.isArray(result?.sinus_contours) && result.sinus_contours.length > 0) ||
      (Array.isArray(result?.nerve_contours) && result.nerve_contours.length > 0) ||
      (Array.isArray(result?.teeth) && result.teeth.length > 0) ||
      (Array.isArray(result?.teeth_objects) && result.teeth_objects.length > 0)
    );

    if (nextHasStructuredOverlay) {
      setViewMode((current) => (current === 'original' ? 'overlay' : current));
    }
  }, [result]);

  useEffect(() => {
    clearHeatmapControlsTimeout();

    if (viewMode === 'heatmap') {
      setIsHeatmapControlsExpanded(true);
    } else {
      setIsHeatmapControlsExpanded(false);
      setIsHeatmapPresetMenuVisible(false);
    }

    return () => {
      clearHeatmapControlsTimeout();
    };
  }, [viewMode]);

  useEffect(() => {
    setPanoZoom(1);
    setPanoOffset({ x: 0, y: 0 });
    setPanoBrightness(PANO_DEFAULT_BRIGHTNESS);
    setPanoContrast(PANO_DEFAULT_CONTRAST);
    setPanoMagnifier({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });
    setActiveTooth(null);
    setActiveDetectionId(null);
    setActiveLegendFilter(null);
    autoWindowAppliedRef.current = null;
  }, [panoViewerUrl]);

  useEffect(() => {
    if (activeDetectionId && !normalizedDetections.some((detection) => detection.id === activeDetectionId)) {
      setActiveDetectionId(null);
    }
  }, [activeDetectionId, normalizedDetections]);

  useEffect(() => {
    if (activeTooth && activeDetectionId) {
      const activeDetection = normalizedDetections.find((detection) => detection.id === activeDetectionId);
      if (activeDetection?.toothFdi && activeDetection.toothFdi !== activeTooth) {
        setActiveTooth(activeDetection.toothFdi);
      }
    }
  }, [activeDetectionId, activeTooth, normalizedDetections]);

  useEffect(() => {
    fitPanoImage();
  }, [panoBodyWidth, panoBodyHeight, panoViewerUrl]);

  const overlayScaleX = overlayCoordinateSize.width / Math.max(1, panoNaturalSize.width || overlayCoordinateSize.width);
  const overlayScaleY = overlayCoordinateSize.height / Math.max(1, panoNaturalSize.height || overlayCoordinateSize.height);
  const displayDicomHudMetadata = (() => {
    if (!dicomHudMetadata) return null;
    if (!originalIsDicom) return dicomHudMetadata;
    const nextWindow = deriveDisplayWindowFromControls(
      dicomHudMetadata.windowCenter,
      dicomHudMetadata.windowWidth,
      panoBrightness,
      panoContrast
    );
    return {
      ...dicomHudMetadata,
      windowCenter: nextWindow.windowCenter,
      windowWidth: nextWindow.windowWidth,
    };
  })();

  const handleDetectionSelect = (detection: NormalizedDetection) => {
    const isSameDetection = activeDetectionId === detection.id;
    setActiveDetectionId(isSameDetection ? null : detection.id);
    setActiveTooth(isSameDetection ? null : detection.toothFdi ?? null);
    if (!isSameDetection) {
      setViewMode('overlay');
    }
  };

  const handleToothSelect = (toothFdi: string) => {
    const nextTooth = activeTooth === toothFdi ? null : toothFdi;
    setActiveTooth(nextTooth);
    setActiveDetectionId(nextTooth ? primaryDetectionByTooth[nextTooth]?.id ?? null : null);
    if (nextTooth && primaryDetectionByTooth[nextTooth]) {
      setViewMode('overlay');
    }
  };

  const handleLegendToggle = (condition: ToothCondition) => {
    setActiveLegendFilter((current) => (current === condition ? null : condition));
    setActiveTooth(null);
    setActiveDetectionId(null);
    setViewMode((current) => (current === 'original' ? 'overlay' : current));
  };

  const handlePanoWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    if (!panoViewerUrl) return;
    if (selectedToolbarButton !== 'pan') return;
    event.preventDefault();
    const delta = event.deltaY < 0 ? 0.12 : -0.12;
    setPanoZoom((current) => clamp(Number((current + delta).toFixed(2)), 0.35, 5));
  };

  const handlePanoPointerDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!panoViewerUrl) return;
    if (selectedToolbarButton === 'measure-eraser') {
      const point = clientToOverlayPoint(event.clientX, event.clientY);
      if (!point) return;
      event.preventDefault();
      setMeasureShapes((current) => {
        const targetIndex = [...current]
          .map((shape, index) => ({ shape, index }))
          .reverse()
          .find(({ shape }) => isPointNearShape(point, shape))?.index;
        if (targetIndex == null) return current;
        return current.filter((_, index) => index !== targetIndex);
      });
      setPendingMeasurePoints([]);
      setMeasurePreviewPoint(null);
      return;
    }
    if (isCustomOverlayTool) {
      if (event.button !== 0) return;
      const point = clientToOverlayPoint(event.clientX, event.clientY);
      if (!point) return;
      event.preventDefault();

      if (activeMeasureSubtool === 'text') {
        const viewportRect = panoViewportRef.current?.getBoundingClientRect();
        const viewportScaleX = viewportRect ? viewportRect.width / Math.max(1, panoBodyWidth) : 1;
        const viewportScaleY = viewportRect ? viewportRect.height / Math.max(1, panoBodyHeight) : 1;
        const anchorLeft = viewportRect ? (event.clientX - viewportRect.left) / Math.max(viewportScaleX, 0.0001) + 12 : 12;
        const anchorTop = viewportRect ? (event.clientY - viewportRect.top) / Math.max(viewportScaleY, 0.0001) - 12 : 0;
        setPendingTextAnnotation({ point, text: '', anchorLeft, anchorTop });
        setPendingMeasurePoints([]);
        setMeasurePreviewPoint(null);
        return;
      }

      if (isFreeformDrawTool) {
        const nextPoints = [...pendingMeasurePoints, point];
        setPendingMeasurePoints(nextPoints);
        setMeasurePreviewPoint(point);
        return;
      }

      const nextPoints = [...pendingMeasurePoints, point];
      const required = requiredPointsForMeasureTool(activeMeasureSubtool);
      if (required > 0 && nextPoints.length >= required) {
        setMeasureShapes((current) => [...current, { type: activeMeasureSubtool, points: nextPoints }]);
        setPendingMeasurePoints([]);
        setMeasurePreviewPoint(null);
      } else {
        setPendingMeasurePoints(nextPoints);
        setMeasurePreviewPoint(point);
      }
      return;
    }

    if (selectedToolbarButton === 'pan') {
      panoDragRef.current = {
        mode: 'pan',
        startX: event.clientX,
        startY: event.clientY,
        startOffsetX: panoOffset.x,
        startOffsetY: panoOffset.y,
        startBrightness: panoBrightness,
        startContrast: panoContrast,
      };
      return;
    }

    if (selectedToolbarButton === 'wlww') {
      panoDragRef.current = {
        mode: 'wlww',
        startX: event.clientX,
        startY: event.clientY,
        startOffsetX: panoOffset.x,
        startOffsetY: panoOffset.y,
        startBrightness: panoBrightness,
        startContrast: panoContrast,
      };
      return;
    }
  };

  const handlePanoPointerMove = (event: React.MouseEvent<HTMLDivElement>) => {
    const dragState = panoDragRef.current;
    if (isCustomOverlayTool && pendingMeasurePoints.length > 0) {
      const point = clientToOverlayPoint(event.clientX, event.clientY);
      setMeasurePreviewPoint(point);
      return;
    }

    if (dragState.mode === 'pan') {
      const dx = event.clientX - dragState.startX;
      const dy = event.clientY - dragState.startY;
      setPanoOffset({
        x: dragState.startOffsetX + dx,
        y: dragState.startOffsetY + dy,
      });
      return;
    }

    if (dragState.mode === 'wlww') {
      const dx = event.clientX - dragState.startX;
      const dy = event.clientY - dragState.startY;
      setPanoContrast(clamp(Math.round(dragState.startContrast + dx * 0.35), 40, 220));
      setPanoBrightness(clamp(Math.round(dragState.startBrightness - dy * 0.35), 40, 220));
      return;
    }

    if (selectedToolbarButton === 'magnifier') {
      const viewerRect = panoViewportRef.current?.getBoundingClientRect();
      const displayRect = panoDisplayRef.current?.getBoundingClientRect();
      if (!viewerRect || !displayRect || !panoNaturalSize.width || !panoNaturalSize.height) return;
      if (
        event.clientX < displayRect.left ||
        event.clientX > displayRect.right ||
        event.clientY < displayRect.top ||
        event.clientY > displayRect.bottom
      ) {
        setPanoMagnifier((current) =>
          current.visible
            ? { visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 }
            : current
        );
        return;
      }
      const normalizedX = clamp((event.clientX - displayRect.left) / Math.max(1, displayRect.width), 0, 1);
      const normalizedY = clamp((event.clientY - displayRect.top) / Math.max(1, displayRect.height), 0, 1);
      const imgX = (flipped ? 1 - normalizedX : normalizedX) * panoNaturalSize.width;
      const imgY = normalizedY * panoNaturalSize.height;
      setPanoMagnifier({
        visible: true,
        clientX: event.clientX,
        clientY: event.clientY,
        viewerX: clamp(event.clientX - viewerRect.left, 0, viewerRect.width),
        viewerY: clamp(event.clientY - viewerRect.top, 0, viewerRect.height),
        imgX,
        imgY,
      });
    }
  };

  const handlePanoPointerUp = () => {
    panoDragRef.current.mode = null;
  };

  const handlePanoPointerLeave = () => {
    panoDragRef.current.mode = null;
    setIsOverlayPresetMenuVisible(false);
    setPanoMagnifier((current) =>
      current.visible ? { visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 } : current
    );
    if (isCustomOverlayTool) {
      setMeasurePreviewPoint(null);
    }
  };

  const handlePanoDoubleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isCustomDrawTool || !isFreeformDrawTool) return;
    event.preventDefault();
    const point = clientToOverlayPoint(event.clientX, event.clientY);
    finalizeFreeformShape(point);
  };

  const handlePanoContextMenu = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isCustomDrawTool || !isFreeformDrawTool) return;
    event.preventDefault();
    const point = clientToOverlayPoint(event.clientX, event.clientY);
    finalizeFreeformShape(point);
  };

  const panoMagnifierViewport = (() => {
    if (
      selectedToolbarButton !== 'magnifier' ||
      !panoViewerUrl ||
      !panoMagnifier.visible ||
      !panoNaturalSize.width ||
      !panoNaturalSize.height
    ) {
      return null;
    }

    const viewerWidth = panoBodyWidth;
    const viewerHeight = panoBodyHeight;
    const sampleSize = Math.max(PANO_LENS_SIZE / PANO_LENS_ZOOM, 1);
    const sampleLeft = clamp(
      panoMagnifier.imgX - sampleSize / 2,
      0,
      Math.max(0, panoNaturalSize.width - sampleSize)
    );
    const sampleTop = clamp(
      panoMagnifier.imgY - sampleSize / 2,
      0,
      Math.max(0, panoNaturalSize.height - sampleSize)
    );

    let lensLeft = panoMagnifier.viewerX - PANO_LENS_SIZE / 2;
    let lensTop = panoMagnifier.viewerY - PANO_LENS_SIZE / 2;

    lensLeft = clamp(
      lensLeft,
      PANO_LENS_EDGE_PADDING,
      Math.max(PANO_LENS_EDGE_PADDING, viewerWidth - PANO_LENS_SIZE - PANO_LENS_EDGE_PADDING)
    );
    lensTop = clamp(
      lensTop,
      PANO_LENS_EDGE_PADDING,
      Math.max(PANO_LENS_EDGE_PADDING, viewerHeight - PANO_LENS_SIZE - PANO_LENS_EDGE_PADDING)
    );

    return {
      lensLeft,
      lensTop,
      sampleLeft,
      sampleTop,
      sampleSize,
    };
  })();

  useEffect(() => {
    if (!panoMagnifierViewport || !panoLensCanvasRef.current || !panoImageRef.current) return;

    let frameId = 0;
    frameId = window.requestAnimationFrame(() => {
      const canvas = panoLensCanvasRef.current;
      const image = panoImageRef.current;
      if (!canvas || !image) return;

      canvas.width = PANO_LENS_SIZE;
      canvas.height = PANO_LENS_SIZE;
      const context = canvas.getContext('2d');
      if (!context) return;

      context.clearRect(0, 0, PANO_LENS_SIZE, PANO_LENS_SIZE);
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = 'high';
      context.save();
      context.filter = `invert(${inverted ? 1 : 0}) brightness(${panoBrightness}%) contrast(${panoContrast}%)`;
      if (flipped) {
        context.translate(PANO_LENS_SIZE, 0);
        context.scale(-1, 1);
      }
      context.drawImage(
        image,
        panoMagnifierViewport.sampleLeft,
        panoMagnifierViewport.sampleTop,
        panoMagnifierViewport.sampleSize,
        panoMagnifierViewport.sampleSize,
        0,
        0,
        PANO_LENS_SIZE,
        PANO_LENS_SIZE
      );
      context.restore();
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [panoMagnifierViewport, panoBrightness, panoContrast, inverted, flipped]);

  const handleOutputCapture = async () => {
    const rawCanvas = await buildPanoCaptureCanvas();
    if (!rawCanvas) return;
    const canvas = trimCaptureCanvas(rawCanvas);
    const dataUrl = canvas.toDataURL('image/png');
    setIsCapturePanelCollapsed(false);
    setCapturedOutputs((current) => [
      {
        id: `capture-${Date.now()}`,
        dataUrl,
        createdAt: Date.now(),
      },
      ...current,
    ].slice(0, 8));

    try {
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('Failed to generate capture image');
      await navigator.clipboard.write([
        new ClipboardItem({
          'image/png': blob,
        }),
      ]);
    } catch (error) {
      console.error('Failed to copy pano capture', error);
    }
  };

  const toggleReportCaptureSelection = (captureId: string) => {
    setSelectedReportCaptureIds((current) =>
      current.includes(captureId) ? current.filter((id) => id !== captureId) : [...current, captureId]
    );
  };

  const updateReportCaptureNote = (captureId: string, note: string) => {
    setCapturedOutputs((current) =>
      current.map((capture) => (capture.id === captureId ? { ...capture, note } : capture))
    );
  };

  useEffect(() => {
    setSelectedReportCaptureIds((current) => current.filter((id) => capturedOutputs.some((capture) => capture.id === id)));
  }, [capturedOutputs]);

  useEffect(() => {
    lastSyncedReportCapturesRef.current = '';
    lastRefreshedReportCaptureStructureRef.current = '';
  }, [reportSessionId]);

  useEffect(() => {
    const handlePreviewMessage = (event: MessageEvent) => {
      const data = event.data;
      if (!data || data.source !== 'web-report-preview') return;
      if (data.action === 'capture-note-change') {
        const captureId = String(data.captureId || '');
        if (!captureId) return;
        updateReportCaptureNote(captureId, String(data.note || ''));
        return;
      }
      if (data.action === 'toggle-tooth') {
        const toothLabel = String(data.toothLabel || '');
        if (!reportSessionId || !toothLabel) return;
        void patchWebReportOverrides(reportSessionId, {
          tooth_overrides: {
            [toothLabel]: {
              included: Boolean(data.checked),
            },
          },
        })
          .catch((error) => {
            console.error('Failed to update report tooth visibility', error);
          });
      }
    };

    window.addEventListener('message', handlePreviewMessage);
    return () => {
      window.removeEventListener('message', handlePreviewMessage);
    };
  }, [reportSessionId]);

  useEffect(() => {
    if (!reportSessionId || reportStartState === 'creating') return;

    const selectedCaptures = capturedOutputs.filter((capture) => selectedReportCaptureIds.includes(capture.id));
    const mappedCaptures = selectedCaptures.map((capture, index) => ({
      id: capture.id || `capture-${index}`,
      dataUrl: capture.dataUrl,
      reportDataUrl: capture.dataUrl,
      createdAt: capture.createdAt,
      label: capture.label || `Capture ${index + 1}`,
      size: capture.size,
      note: capture.note || '',
    }));
    const nextPayload = JSON.stringify(mappedCaptures);
    const nextStructureSignature = JSON.stringify(
      mappedCaptures.map((capture) => ({
        id: capture.id,
        dataUrl: capture.dataUrl,
        createdAt: capture.createdAt,
        label: capture.label,
        size: capture.size,
      }))
    );
    if (nextPayload === lastSyncedReportCapturesRef.current) return;

    const timer = window.setTimeout(async () => {
      try {
        await patchWebReportOverrides(reportSessionId, {
          attached_captures: mappedCaptures,
        });
        lastSyncedReportCapturesRef.current = nextPayload;
        if (nextStructureSignature !== lastRefreshedReportCaptureStructureRef.current) {
          lastRefreshedReportCaptureStructureRef.current = nextStructureSignature;
          setReportPreviewRefreshToken((current) => current + 1);
        }
      } catch (error) {
        console.error('Failed to sync report captures', error);
      }
    }, 300);

    return () => {
      window.clearTimeout(timer);
    };
  }, [capturedOutputs, reportSessionId, reportStartState, selectedReportCaptureIds]);

  const handleOutputSave = async () => {
    const canvas = await buildPanoCaptureCanvas();
    if (!canvas) return;

    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = `renew-pano-${Date.now()}.png`;
    link.click();
  };

  const flashToolbarActive = (button: ToolbarKey) => {
    setFlashToolbarButton(button);
    window.setTimeout(() => {
      setFlashToolbarButton((current) => (current === button ? null : current));
    }, 180);
  };

  const closeToolSubmenu = () => {
    setToolSubmenu(null);
  };

  const activateMeasureSubtool = (tool: MeasureSubtoolKey) => {
    closeToolSubmenu();
    setActiveMeasureSubtool(tool);
    setPendingMeasurePoints([]);
    setMeasurePreviewPoint(null);
    setPendingTextAnnotation(null);

    switch (tool) {
      case 'length':
        setCornerstoneActiveTool('Length');
        setSelectedToolbarButton('measure-length');
        return;
      case 'bidirectional':
        setCornerstoneActiveTool('Bidirectional');
        setSelectedToolbarButton('measure-length');
        return;
      case 'angle':
        setCornerstoneActiveTool('Angle');
        setSelectedToolbarButton('measure-length');
        return;
      case 'text':
        setCornerstoneActiveTool('Label');
        setSelectedToolbarButton('measure-draw');
        return;
      case 'arrow':
        setCornerstoneActiveTool('ArrowAnnotate');
        setSelectedToolbarButton('measure-draw');
        return;
      case 'ellipse':
        setCornerstoneActiveTool('EllipticalROI');
        setSelectedToolbarButton('measure-draw');
        return;
      case 'rect':
        setCornerstoneActiveTool('RectangleROI');
        setSelectedToolbarButton('measure-draw');
        return;
      case 'circle':
        setCornerstoneActiveTool('CircleROI');
        setSelectedToolbarButton('measure-draw');
        return;
      case 'roi-free':
        setCornerstoneActiveTool('PlanarFreehandROI');
        setSelectedToolbarButton('measure-draw');
        return;
      case 'spline-roi':
        setCornerstoneActiveTool('SplineROI');
        setSelectedToolbarButton('measure-draw');
        return;
      case 'livewire':
        setCornerstoneActiveTool('LivewireContour');
        setSelectedToolbarButton('measure-draw');
        return;
    }
  };

  const toggleToolSubmenu = (menu: MeasureMenuKey, left: number, top: number) => {
    setToolSubmenu((current) =>
      current && current.menu === menu
        ? null
        : {
            menu,
            left,
            top,
          }
    );
  };

  const handleToolSelection = (button: ToolbarKey) => {
    if (button !== 'measure-length' && button !== 'measure-draw') {
      closeToolSubmenu();
      setPendingMeasurePoints([]);
      setMeasurePreviewPoint(null);
    }
    switch (button) {
      case 'pointer':
        setCornerstoneActiveTool('Pan');
        setPanoMagnifier({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });
        setSelectedToolbarButton('pointer');
        return;
      case 'pan':
        setCornerstoneActiveTool('Pan');
        setPanoMagnifier({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });
        setSelectedToolbarButton('pan');
        return;
      case 'wlww':
        setCornerstoneActiveTool('WindowLevel');
        setPanoMagnifier({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });
        setSelectedToolbarButton('wlww');
        return;
      case 'invert':
        setInverted((current) => !current);
        setSelectedToolbarButton((current) => (current === 'invert' ? 'pointer' : 'invert'));
        return;
      case 'magnifier':
        setCornerstoneActiveTool('Pan');
        setSelectedToolbarButton((current) => {
          const next = current === 'magnifier' ? 'pointer' : 'magnifier';
          if (next !== 'magnifier') {
            setPanoMagnifier({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });
          }
          return next;
        });
        return;
      case 'flip':
        setPanoMagnifier({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });
        setFlipped((current) => !current);
        setSelectedToolbarButton((current) => (current === 'flip' ? 'pointer' : 'flip'));
        return;
      case 'measure-length':
        return;
      case 'measure-draw':
        return;
      case 'measure-eraser':
        setCornerstoneActiveTool('Eraser');
        setSelectedToolbarButton('measure-eraser');
        return;
      case 'measure-clear':
        clearAllAnnotations();
        setMeasureShapes([]);
        setPendingMeasurePoints([]);
        setMeasurePreviewPoint(null);
        setPendingTextAnnotation(null);
        flashToolbarActive('measure-clear');
        setSelectedToolbarButton('pointer');
        setPanoMagnifier({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });
        return;
      case 'measure-rotate':
        flashToolbarActive('measure-rotate');
        return;
      case 'measure-reset':
        clearAllAnnotations();
        setMeasureShapes([]);
        setPendingMeasurePoints([]);
        setMeasurePreviewPoint(null);
        setPendingTextAnnotation(null);
        setInverted(false);
        setFlipped(false);
        setViewMode('original');
        setActiveTooth(null);
        setActiveDetectionId(null);
        setActiveLegendFilter(null);
        setPanoZoom(1);
        setPanoOffset({ x: 0, y: 0 });
        setPanoBrightness(PANO_DEFAULT_BRIGHTNESS);
        setPanoContrast(PANO_DEFAULT_CONTRAST);
        setPanoMagnifier({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });
        flashToolbarActive('measure-reset');
        setSelectedToolbarButton('pointer');
        setReportError(null);
        setCornerstoneActiveTool('Pan');
        return;
      case 'output-capture':
        flashToolbarActive('output-capture');
        void handleOutputCapture();
        return;
      case 'output-save':
        flashToolbarActive('output-save');
        void handleOutputSave();
        return;
      case 'output-report':
        void handleOpenReportPanel();
        return;
      case 'task-original':
        setPanoMagnifier({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });
        if (viewMode === 'overlay') {
          setViewMode('original');
          setActiveDetectionId(null);
          setSelectedToolbarButton('pointer');
        } else {
          setViewMode('overlay');
          setActiveDetectionId(null);
          setSelectedToolbarButton('task-original');
        }
        return;
      case 'task-heatmap':
        setPanoMagnifier({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });
        if (viewMode === 'heatmap') {
          setViewMode('original');
          setSelectedToolbarButton('pointer');
          setIsHeatmapControlsExpanded(false);
          setIsHeatmapPresetMenuVisible(false);
          clearHeatmapControlsTimeout();
        } else {
          setViewMode('heatmap');
          setSelectedToolbarButton('task-heatmap');
          showHeatmapControls();
        }
        return;
      default:
        return;
    }
  };

  useEffect(() => {
    if (!toolSubmenu) return;

    const handlePointerDown = () => {
      setToolSubmenu(null);
    };

    window.addEventListener('pointerdown', handlePointerDown);
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
    };
  }, [toolSubmenu]);

  useEffect(() => {
    if (!isCustomMeasureTool) {
      setPendingMeasurePoints([]);
      setMeasurePreviewPoint(null);
    }
  }, [isCustomMeasureTool]);

  const resolveReportPatientName = () => {
    const candidates = [
      locationState.userName,
      result?.patient_name,
      dicomHudMetadata?.patientName,
    ];
    const normalized = candidates
      .map((value) => String(value || '').trim())
      .find((value) => value && value.toLowerCase() !== 'patient');
    return normalized || 'Patient';
  };

  const ensureReportSession = async () => {
    if (reportStartState === 'creating') return;
    if (reportSessionId) return reportSessionId;

    if (!result) {
      setReportError('Analysis result is not ready yet.');
      return null;
    }

    setReportStartState('creating');
    setReportError(null);
    try {
      const sourceUrl = pickRuntimeAssetUrl(
        result?.image_url,
        result?.preview_url,
        locationState.imageUrl,
        locationState.previewUrl
      );
      const overlayUrl = pickRuntimeAssetUrl(result?.overlay_url);
      const response = await createWebReportFromChart({
        result,
        source_url: sourceUrl,
        overlay_url: overlayUrl,
        language: 'English',
        patient_name: resolveReportPatientName(),
      });
      setReportSessionId(response.session_id);
      return response.session_id;
    } catch (error: any) {
      setReportError(error?.message || 'Failed to start report workspace');
      return null;
    } finally {
      setReportStartState('idle');
    }
  };

  const handleStartReport = async () => {
    if (reportSessionId) {
      setWorkspaceSection((current) => {
        const next = current === 'report' ? 'none' : 'report';
        setIsReportActive(next === 'report');
        return next;
      });
      setReportDrawerOpen(false);
      setReportError(null);
      return;
    }

    const nextSessionId = await ensureReportSession();
    if (!nextSessionId) {
      setIsReportActive(false);
      return;
    }

    setWorkspaceSection('report');
    setReportDrawerOpen(false);
    setIsReportActive(true);
  };

  const handleOpenReportPanel = async () => {
    if (reportDrawerOpen) {
      setReportDrawerOpen(false);
      setIsReportActive(false);
      return;
    }

    const nextSessionId = await ensureReportSession();
    if (!nextSessionId) {
      setIsReportActive(false);
      return;
    }

    setReportDrawerOpen(true);
    setIsReportActive(true);
    setReportError(null);
  };

  const isReportWorkspaceVisible =
    workspaceSection === 'report' && (reportStartState === 'creating' || Boolean(reportSessionId) || Boolean(reportError));

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#000000',
        overflow: 'hidden',
        fontFamily: '"Noto Sans", "Noto Sans KR", sans-serif',
      }}
    >
      <div
        style={{
          width: `${stageWidth}px`,
          height: `${stageHeight}px`,
          position: 'relative',
          overflow: 'hidden',
          background: '#414950',
          boxShadow: 'inset 1px 0 0 rgba(0, 0, 0, 0.18), inset -1px 0 0 rgba(0, 0, 0, 0.18)',
        }}
      >
        <div
          style={{
            width: wp(designCanvasWidth),
            height: hp(DESIGN_HEIGHT),
            left: 0,
            top: 0,
            position: 'absolute',
            background: '#414950',
            transform: `scale(${scale})`,
            transformOrigin: 'top left',
          }}
        >
          <div style={{ width: wp(designCanvasWidth), height: hp(DESIGN_HEIGHT), left: 0, top: 0, position: 'absolute', background: '#414950' }} />

          <button
            type="button"
            onClick={() => navigate('/')}
            aria-label="Go to home"
            style={{
              width: wp(220),
              height: hp(48),
              left: wp(0),
              top: hp(6),
              position: 'absolute',
              border: 'none',
              background: 'transparent',
              padding: 0,
              cursor: 'pointer',
              zIndex: 12,
            }}
          >
            <img
              src={O3Logo}
              alt=""
              draggable={false}
              style={{
                width: wp(144),
                height: hp(38),
                left: wp(-20),
                top: hp(4),
                position: 'absolute',
                objectFit: 'contain',
              }}
            />
            <div
              style={{
                left: wp(88),
                top: hp(6),
                position: 'absolute',
                color: 'white',
                fontSize: scalePx(20),
                fontWeight: 700,
                letterSpacing: '0.08em',
              }}
            >
              SATURN
            </div>
          </button>

          {!isReportWorkspaceVisible && (
            <>
              <div style={{ width: wp(topBarWidth), height: hp(18), left: wp(topBarLeft), top: hp(49), position: 'absolute', background: '#5C5C5C', zIndex: 1 }} />
              {panoViewerUrl && (
                <button
                  type="button"
                  onClick={handleChartToggle}
                  aria-pressed={!isChartVisible}
                  aria-label={isChartVisible ? 'Maximize pano viewer' : 'Restore dental chart'}
                  title={isChartVisible ? 'Maximize pano viewer' : 'Restore dental chart'}
                  style={{
                    width: wp(18),
                    height: hp(18),
                    left: wp(viewerLeft + viewerWidth - 21),
                    top: hp(49),
                    position: 'absolute',
                    zIndex: 6,
                    background: '#5C5C5C',
                    borderLeft: `${scalePx(1)} solid #4C4C4C`,
                    padding: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                  }}
                >
                  <img
                    src={isChartVisible ? panoZoomIcons.maximize : panoZoomIcons.minimize}
                    alt=""
                    draggable={false}
                    style={{
                      width: wp(10),
                      height: hp(10),
                      objectFit: 'contain',
                      display: 'block',
                    }}
                  />
                </button>
              )}
              <div style={{ width: wp(viewerWidth), height: hp(1018), left: wp(viewerLeft), top: hp(50), position: 'absolute', background: 'black' }} />
              {isChartVisible && (
                <div style={{ width: wp(viewerWidth), height: hp(18), left: wp(viewerLeft), top: hp(804), position: 'absolute', background: '#5C5C5C', zIndex: 1 }} />
              )}
              <div style={{ width: wp(viewerWidth), height: hp(panoFrameHeight), left: wp(viewerLeft), top: hp(49), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
              {isChartVisible && (
                <div
                  style={{
                    width: wp(viewerWidth),
                    height: hp(264),
                    left: wp(viewerLeft),
                    top: hp(804),
                    position: 'absolute',
                    border: `${scalePx(1)} solid #4C4C4C`,
                    pointerEvents: 'none',
                  }}
                />
              )}
              <div style={{ width: scalePx(1), height: hp(1019), left: wp(viewerLeft), top: hp(49), position: 'absolute', background: '#4C4C4C' }} />
              <div
                style={{
                  width: wp(panoBodyWidth),
                  height: hp(panoBodyHeight),
                  left: wp(viewerLeft + 1),
                  top: hp(panoBodyTop),
                  position: 'absolute',
                  overflow: 'hidden',
                  background: '#000000',
                  cursor:
                    selectedToolbarButton === 'pan'
                      ? 'grab'
                      : selectedToolbarButton === 'wlww'
                        ? 'ew-resize'
                        : selectedToolbarButton === 'magnifier'
                          ? 'zoom-in'
                          : selectedToolbarButton === 'measure-eraser' || isCustomOverlayTool
                            ? 'crosshair'
                          : 'default',
                }}
                ref={panoViewportRef}
                onMouseDown={handlePanoPointerDown}
                onMouseMove={handlePanoPointerMove}
                onMouseUp={handlePanoPointerUp}
                onMouseLeave={handlePanoPointerLeave}
                onDoubleClick={handlePanoDoubleClick}
                onContextMenu={handlePanoContextMenu}
                onWheel={handlePanoWheel}
              >
                {panoViewerUrl ? (
                  <div
                    style={{
                      width: wp(panoDisplaySize.width || panoBodyWidth),
                      height: hp(panoDisplaySize.height || panoBodyHeight),
                      position: 'absolute',
                      left: wp((panoBodyWidth - (panoDisplaySize.width || panoBodyWidth)) / 2),
                      top: hp((panoBodyHeight - (panoDisplaySize.height || panoBodyHeight)) / 2),
                      transform: `translate(${panoOffset.x}px, ${panoOffset.y}px) scale(${panoZoom}) scaleX(${flipped ? -1 : 1})`,
                      transformOrigin: 'center',
                      isolation: 'isolate',
                      filter: isProcessing ? 'blur(1.8px)' : 'none',
                      opacity: isProcessing ? 0.82 : 1,
                      transition: 'filter 180ms ease, opacity 180ms ease',
                    }}
                    ref={panoDisplayRef}
                  >
                    <img
                      ref={panoImageRef}
                      src={panoViewerUrl}
                      alt="Panorama"
                      draggable={false}
                      onLoad={fitPanoImage}
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        filter: `invert(${inverted ? 1 : 0}) brightness(${panoBrightness}%) contrast(${panoContrast}%)`,
                        opacity: 1,
                        userSelect: 'none',
                        pointerEvents: 'none',
                        position: 'relative',
                        zIndex: 1,
                      }}
                    />
                    {realHeatmapOverlayUrl ? (
                      <img
                        src={realHeatmapOverlayUrl}
                        alt=""
                        aria-hidden="true"
                        draggable={false}
                        style={{
                          width: '100%',
                          height: '100%',
                          objectFit: 'contain',
                          filter: `invert(${inverted ? 1 : 0}) brightness(${panoBrightness}%) contrast(${panoContrast}%)`,
                          opacity: heatmapOpacity / 100,
                          userSelect: 'none',
                          pointerEvents: 'none',
                          position: 'absolute',
                          inset: 0,
                          zIndex: 2,
                        }}
                      />
                    ) : null}
                    {viewMode !== 'original' && overlayCoordinateSize.width > 0 && overlayCoordinateSize.height > 0 && (
                      <svg
                        ref={panoOverlaySvgRef}
                        viewBox={`0 0 ${overlayCoordinateSize.width} ${overlayCoordinateSize.height}`}
                        preserveAspectRatio="none"
                        style={{
                          position: 'absolute',
                          inset: 0,
                          width: '100%',
                          height: '100%',
                          pointerEvents: viewMode === 'overlay' ? 'auto' : 'none',
                          zIndex: 5,
                        }}
                      >
                        <defs>
                          <radialGradient id="renewCariesGlow" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stopColor="rgba(255, 60, 60, 0.95)" />
                            <stop offset="100%" stopColor="rgba(255, 60, 60, 0)" />
                          </radialGradient>
                          <radialGradient id="renewPeriGlow" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stopColor="rgba(255, 166, 0, 0.92)" />
                            <stop offset="100%" stopColor="rgba(255, 166, 0, 0)" />
                          </radialGradient>
                          <filter id="renewRiskBlurStrong" x="-25%" y="-25%" width="150%" height="150%">
                            <feGaussianBlur stdDeviation={10 / effectiveScale} />
                          </filter>
                          <filter id="renewRiskBlurSoft" x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation={5 / effectiveScale} />
                          </filter>
                        </defs>
                        {renderAIDetections()}
                        {renderRiskDetections()}
                      </svg>
                    )}
                    {overlayCoordinateSize.width > 0 && overlayCoordinateSize.height > 0 && (
                      <svg
                        ref={panoMeasureSvgRef}
                        viewBox={`0 0 ${overlayCoordinateSize.width} ${overlayCoordinateSize.height}`}
                        preserveAspectRatio="none"
                        style={{
                          position: 'absolute',
                          inset: 0,
                          width: '100%',
                          height: '100%',
                          pointerEvents: isCustomOverlayTool ? 'auto' : 'none',
                          zIndex: 6,
                        }}
                      >
                        <defs>
                          <marker id="renewMeasureArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(57, 255, 20, 0.98)" />
                          </marker>
                          <marker id="renewMeasureArrowGlow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
                            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(196, 255, 168, 0.92)" />
                          </marker>
                        </defs>
                        {measureShapes.map((shape, index) => (
                          <g key={`measure-shape-${index}`}>{renderCustomShape(shape)}</g>
                        ))}
                        {isCustomOverlayTool && pendingMeasurePoints.length > 0
                          ? renderCustomShape(
                              {
                                type: activeMeasureSubtool,
                                points: pendingMeasurePoints,
                              },
                              true
                            )
                          : null}
                      </svg>
                    )}
                  </div>
                ) : (
                  <div
                    style={{
                      color: '#7A7A7A',
                      fontSize: scalePx(14),
                      fontWeight: 700,
                      letterSpacing: '0.04em',
                    }}
                  >
                    No panorama source
                  </div>
                )}
                {measureShapes
                  .map((shape, index) => ({
                    shape,
                    index,
                    position:
                      shape.type === 'text'
                        ? shape.anchorLeft != null && shape.anchorTop != null
                          ? { left: shape.anchorLeft, top: shape.anchorTop }
                          : overlayPointToViewportPosition(shape.points[0])
                        : null,
                  }))
                  .filter(({ shape, position }) => shape.type === 'text' && position)
                  .map(({ shape, index, position }) => (
                    <div
                      key={`measure-text-${index}`}
                      style={{
                        position: 'absolute',
                        left: `${position!.left}px`,
                        top: `${position!.top}px`,
                        transform: 'translateY(-100%)',
                        zIndex: 20,
                        pointerEvents: 'none',
                        color: 'rgba(57, 255, 20, 0.98)',
                        fontSize: scalePx(12),
                        fontWeight: 700,
                        lineHeight: 1.4,
                        whiteSpace: 'pre-wrap',
                        textShadow: '0 0 1px rgba(0, 0, 0, 0.35)',
                      }}
                    >
                      {shape.text || 'Note'}
                    </div>
                  ))}
                {pendingTextAnnotation ? (
                  <div
                    style={{
                      position: 'absolute',
                      left: `${pendingTextAnnotation.anchorLeft}px`,
                      top: `${pendingTextAnnotation.anchorTop}px`,
                      transform: 'translateY(-100%)',
                      zIndex: 30,
                      width: wp(198),
                      border: `${scalePx(1)} solid #202020`,
                      borderRadius: 0,
                      background: '#2A2A2A',
                      boxShadow: '0 10px 18px rgba(0, 0, 0, 0.28)',
                      overflow: 'hidden',
                    }}
                    onPointerDown={(event) => {
                      event.stopPropagation();
                    }}
                    onMouseDown={(event) => {
                      event.stopPropagation();
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        height: hp(18),
                        padding: `0 ${wp(6)}`,
                        background: '#2A2A2A',
                        borderBottom: `${scalePx(1)} solid #202020`,
                      }}
                    >
                      <div
                        style={{
                          color: '#EDEDED',
                          fontSize: scalePx(8.5),
                          fontWeight: 700,
                          letterSpacing: '0.08em',
                          textTransform: 'uppercase',
                        }}
                      >
                        Saturn Note
                      </div>
                      <button
                        type="button"
                        onPointerDown={dismissPendingTextAnnotation}
                        onClick={dismissPendingTextAnnotation}
                        style={{
                          width: scalePx(13),
                          height: scalePx(13),
                          padding: 0,
                          border: 'none',
                          background: 'transparent',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: 'pointer',
                        }}
                      >
                        <img
                          src="/imgs/13 13.png"
                          alt="Close"
                          draggable={false}
                          style={{
                            width: scalePx(13),
                            height: scalePx(13),
                            display: 'block',
                            objectFit: 'contain',
                            pointerEvents: 'none',
                          }}
                        />
                      </button>
                    </div>
                    <div
                      style={{
                        padding: `${hp(5)} ${wp(5)} ${hp(5)}`,
                      }}
                    >
                    <textarea
                      ref={textAnnotationInputRef}
                      value={pendingTextAnnotation.text}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                      }}
                      onMouseDown={(event) => {
                        event.stopPropagation();
                      }}
                      onClick={(event) => {
                        event.stopPropagation();
                      }}
                      onChange={(event) => {
                        setPendingTextAnnotation((current) =>
                          current ? { ...current, text: event.target.value } : current
                        );
                      }}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' && !event.shiftKey) {
                          event.preventDefault();
                          commitPendingTextAnnotation();
                          return;
                        }
                        if (event.key === 'Escape') {
                          event.preventDefault();
                          cancelPendingTextAnnotation();
                        }
                      }}
                      rows={2}
                      style={{
                        width: '100%',
                        minHeight: hp(52),
                        resize: 'none',
                        border: `${scalePx(1)} solid #2A2A2A`,
                        borderRadius: 0,
                        background: '#6A6A6A',
                        color: 'rgba(57, 255, 20, 0.98)',
                        caretColor: 'rgba(57, 255, 20, 0.98)',
                        padding: `${hp(5)} ${wp(6)}`,
                        fontSize: scalePx(11),
                        lineHeight: 1.25,
                        outline: 'none',
                        boxShadow: 'none',
                        overflow: 'hidden',
                      }}
                    />
                    <div
                      style={{
                        marginTop: hp(5),
                        display: 'flex',
                        justifyContent: 'flex-end',
                        gap: wp(5),
                        paddingTop: hp(5),
                        borderTop: `${scalePx(1)} solid #202020`,
                      }}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          commitPendingTextAnnotation();
                        }}
                        onPointerDown={(event) => {
                          event.stopPropagation();
                        }}
                        onMouseDown={(event) => {
                          event.stopPropagation();
                        }}
                        style={{
                          minWidth: wp(46),
                          height: hp(20),
                          borderRadius: 0,
                          border: `${scalePx(1)} solid #202020`,
                          background: '#333333',
                          color: '#EAEAEA',
                          fontSize: scalePx(10),
                          fontWeight: 700,
                          cursor: 'pointer',
                          boxShadow: 'none',
                        }}
                      >
                        OK
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          cancelPendingTextAnnotation();
                        }}
                        onPointerDown={(event) => {
                          event.stopPropagation();
                        }}
                        onMouseDown={(event) => {
                          event.stopPropagation();
                        }}
                        style={{
                          minWidth: wp(54),
                          height: hp(20),
                          borderRadius: 0,
                          border: `${scalePx(1)} solid #202020`,
                          background: '#333333',
                          color: '#D9D9D9',
                          fontSize: scalePx(10),
                          fontWeight: 700,
                          cursor: 'pointer',
                          boxShadow: 'none',
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                    </div>
                  </div>
                ) : null}
                {isProcessing && (
                  <div
                    style={{
                      position: 'absolute',
                      inset: 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      pointerEvents: 'none',
                      zIndex: 18,
                      background: 'rgba(0, 0, 0, 0.12)',
                    }}
                  >
                    <div
                      style={{
                        width: wp(220),
                        padding: `${hp(14)} ${wp(16)}`,
                        borderRadius: scalePx(10),
                        border: `${scalePx(1)} solid rgba(255, 255, 255, 0.18)`,
                        background: 'rgba(30, 30, 30, 0.78)',
                        backdropFilter: 'blur(10px)',
                        boxShadow: '0 14px 40px rgba(0, 0, 0, 0.28)',
                      }}
                    >
                      <div
                        style={{
                          color: '#F3F4F6',
                          fontSize: scalePx(16),
                          fontWeight: 700,
                          letterSpacing: '0.03em',
                          marginBottom: hp(10),
                          textAlign: 'center',
                        }}
                      >
                        Loading ...
                      </div>
                      <div
                        style={{
                          width: '100%',
                          height: hp(8),
                          borderRadius: scalePx(999),
                          background: 'rgba(255, 255, 255, 0.12)',
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${loadingProgress}%`,
                            height: '100%',
                            borderRadius: 'inherit',
                            background: 'linear-gradient(90deg, rgba(180,180,180,0.85) 0%, rgba(240,240,240,0.96) 100%)',
                            boxShadow: '0 0 14px rgba(255, 255, 255, 0.24)',
                            transition: 'width 240ms ease',
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}
                {displayDicomHudMetadata && (
                  <DicomMetadataOverlay
                    metadata={displayDicomHudMetadata}
                    top={12}
                    left={12}
                    bottom={12}
                    leftPanelAlign="left"
                    rightPanelAlign="left"
                    leftPanelWidth={252}
                    rightPanelWidth={286}
                    leftPanelFontSize={11}
                    rightPanelFontSize={10}
                    headerFontSize={10}
                    panelPaddingX={12}
                    panelPaddingY={8}
                    panelRadius={12}
                  />
                )}
                {viewMode === 'overlay' && hasStructuredOverlayData && (
                  <div
                    style={{
                      position: 'absolute',
                      right: wp(18),
                      top: hp(18),
                      width: wp(190),
                      zIndex: 9,
                    }}
                  >
                    <button
                      type="button"
                      onClick={() => setIsOverlayPresetMenuVisible((current) => !current)}
                      style={{
                        width: wp(190),
                        minHeight: hp(28),
                        padding: `${hp(5)} ${wp(7)}`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        background: 'rgba(46, 46, 46, 0.92)',
                        color: '#FFFFFF',
                        border: `${scalePx(1)} solid rgba(255, 255, 255, 0.12)`,
                        borderRadius: isOverlayPresetMenuVisible ? `${scalePx(6)} ${scalePx(6)} 0 0` : scalePx(6),
                        backdropFilter: 'blur(8px)',
                        boxShadow: '0 8px 18px rgba(0, 0, 0, 0.22)',
                        cursor: 'pointer',
                        textAlign: 'left',
                      }}
                    >
                      <span
                        style={{
                          fontSize: scalePx(11),
                          fontWeight: 700,
                          letterSpacing: '0.02em',
                        }}
                      >
                        {overlayPresetOptions.find((option) => option.key === overlayPreset)?.menuLabel ?? '1. 전체'}
                      </span>
                      <span
                        aria-hidden="true"
                        style={{
                          fontSize: scalePx(14),
                          fontWeight: 700,
                          lineHeight: 1,
                          transform: isOverlayPresetMenuVisible ? 'rotate(180deg)' : 'rotate(0deg)',
                          transition: 'transform 120ms ease',
                          display: 'none',
                        }}
                      >
                        ˅
                      </span>
                      <span
                        aria-hidden="true"
                        style={{
                          transform: isOverlayPresetMenuVisible ? 'rotate(180deg)' : 'rotate(0deg)',
                          transition: 'transform 120ms ease',
                          width: 0,
                          height: 0,
                          borderLeft: `${scalePx(4)} solid transparent`,
                          borderRight: `${scalePx(4)} solid transparent`,
                          borderTop: `${scalePx(6)} solid rgba(255, 255, 255, 0.8)`,
                        }}
                      />
                    </button>
                    {isOverlayPresetMenuVisible && (
                      <div
                        style={{
                          width: wp(190),
                          background: 'rgba(34, 34, 34, 0.96)',
                          borderLeft: `${scalePx(1)} solid rgba(255, 255, 255, 0.12)`,
                          borderRight: `${scalePx(1)} solid rgba(255, 255, 255, 0.12)`,
                          borderBottom: `${scalePx(1)} solid rgba(255, 255, 255, 0.12)`,
                          borderRadius: `0 0 ${scalePx(6)} ${scalePx(6)}`,
                          overflow: 'hidden',
                          backdropFilter: 'blur(8px)',
                          boxShadow: '0 8px 18px rgba(0, 0, 0, 0.22)',
                        }}
                      >
                        {overlayPresetOptions.map((option) => {
                          const active = overlayPreset === option.key;
                          return (
                            <button
                              key={option.key}
                              type="button"
                              onClick={() => {
                                setOverlayPreset(option.key);
                                setIsOverlayPresetMenuVisible(false);
                              }}
                              style={{
                                width: '100%',
                                minHeight: hp(28),
                                padding: `${hp(5)} ${wp(7)}`,
                                background: active ? 'rgba(92, 92, 92, 0.96)' : 'transparent',
                                color: '#FFFFFF',
                                border: 'none',
                                fontSize: scalePx(11),
                                fontWeight: 700,
                                letterSpacing: '0.02em',
                                cursor: 'pointer',
                                whiteSpace: 'nowrap',
                                textAlign: 'left',
                              }}
                            >
                              {option.menuLabel}
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
                {selectedToolbarButton === 'magnifier' && panoMagnifierViewport && (
                  <div
                    style={{
                      width: wp(PANO_LENS_SIZE),
                      height: hp(PANO_LENS_SIZE),
                      left: wp(panoMagnifierViewport.lensLeft),
                      top: hp(panoMagnifierViewport.lensTop),
                      position: 'absolute',
                      overflow: 'hidden',
                      border: `${scalePx(2)} solid rgba(255, 255, 255, 0.75)`,
                      boxShadow: '0 14px 40px rgba(0, 0, 0, 0.4)',
                      pointerEvents: 'none',
                      zIndex: 8,
                      background: 'rgba(0,0,0,0.22)',
                    }}
                  >
                    <canvas ref={panoLensCanvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
                    {viewMode !== 'original' && overlayCoordinateSize.width > 0 && overlayCoordinateSize.height > 0 && (
                      <svg
                        viewBox={`${
                          panoMagnifierViewport.sampleLeft * overlayScaleX
                        } ${
                          panoMagnifierViewport.sampleTop * overlayScaleY
                        } ${
                          panoMagnifierViewport.sampleSize * overlayScaleX
                        } ${
                          panoMagnifierViewport.sampleSize * overlayScaleY
                        }`}
                        preserveAspectRatio="none"
                        style={{
                          position: 'absolute',
                          inset: 0,
                          width: '100%',
                          height: '100%',
                          transform: flipped ? 'scaleX(-1)' : undefined,
                          transformOrigin: 'center',
                        }}
                      >
                        <defs>
                          <radialGradient id="renewCariesGlow" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stopColor="rgba(255, 60, 60, 0.95)" />
                            <stop offset="100%" stopColor="rgba(255, 60, 60, 0)" />
                          </radialGradient>
                          <radialGradient id="renewPeriGlow" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stopColor="rgba(255, 166, 0, 0.92)" />
                            <stop offset="100%" stopColor="rgba(255, 166, 0, 0)" />
                          </radialGradient>
                          <filter id="renewRiskBlurStrong" x="-25%" y="-25%" width="150%" height="150%">
                            <feGaussianBlur stdDeviation={10 / effectiveScale} />
                          </filter>
                          <filter id="renewRiskBlurSoft" x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation={5 / effectiveScale} />
                          </filter>
                        </defs>
                        {renderAIDetections()}
                        {renderRiskDetections()}
                      </svg>
                    )}
                  </div>
                )}
              </div>
            </>
          )}

          <RenewReportWorkspacePanel
            visible={isReportWorkspaceVisible}
            left={wp(viewerLeft)}
            top={hp(49)}
            width={wp(viewerWidth)}
            height={hp(1019)}
            sessionId={reportSessionId}
            isLoading={reportStartState === 'creating'}
            error={reportError}
            availableCaptures={capturedOutputs}
            selectedCaptureIds={selectedReportCaptureIds}
            onToggleCaptureSelection={toggleReportCaptureSelection}
            previewRefreshToken={reportPreviewRefreshToken}
            onClose={() => {
              setWorkspaceSection('none');
              setIsReportActive(false);
              setReportDrawerOpen(false);
            }}
          />

        <div style={{ width: wp(70), height: hp(1019), left: wp(12), top: hp(49), position: 'absolute', background: '#2D2D2D' }} />
        <div style={{ width: wp(70), height: hp(1019), left: wp(12), top: hp(49), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
        {/* 먼 세로선이지 */} 
        {/*<div style={{ width: scalePx(1), height: hp(1019), left: wp(48), top: hp(49), position: 'absolute', background: '#3F3F3F' }} />  
        <div style={{ width: scalePx(1), height: hp(1019), left: wp(68), top: hp(49), position: 'absolute', background: '#5C5C5C' }} /> */}
        <button
          type="button"
          onClick={handleOpenStudies}
          aria-pressed={workspaceSection === 'studies'}
          aria-label="Open studies"
          title="Studies"
          style={{ width: wp(RAIL_ICON_WIDTH), height: hp(RAIL_ICON_HEIGHT), left: wp(16), top: hp(52), position: 'absolute', background: '#2D2D2D', padding: 0, cursor: 'pointer' }}
        >
          <img style={{ width: '100%', height: '100%', display: 'block' }} src={workspaceSection === 'studies' ? activeRailIcons.studies : displayRailIcons.studies} alt="" />
        </button>
        <button
          type="button"
          onClick={() => { void handleStartReport(); }}
          aria-pressed={workspaceSection === 'report' || reportDrawerOpen}
          aria-label="Open Dratf workspace"
          title="Report"
          style={{ width: wp(RAIL_ICON_WIDTH), height: hp(RAIL_ICON_HEIGHT), left: wp(16), top: hp(122), position: 'absolute', background: '#2D2D2D', padding: 0, cursor: 'pointer' }}
        >
          <img style={{ width: '100%', height: '100%', display: 'block' }} src={workspaceSection === 'report' || reportDrawerOpen ? activeRailIcons.captures : displayRailIcons.captures} alt="" />
        </button>
        <div style={{ width: wp(154), height: hp(1019), left: wp(82), top: hp(49), position: 'absolute', background: '#2D2D2D' }} />
        <div style={{ width: wp(154), height: hp(1019), left: wp(82), top: hp(49), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
        <div style={{ width: wp(148), height: hp(336), left: wp(86), top: hp(49), position: 'absolute', background: '#333333' }} />
        <div style={{ width: wp(148), height: hp(84), left: wp(86), top: hp(389), position: 'absolute', background: '#333333' }} />
        <div style={{ width: wp(140), height: hp(116), left: wp(88), top: hp(69), position: 'absolute', background: '#414141' }} />
        <div style={{ width: wp(140), height: hp(118), left: wp(88), top: hp(189), position: 'absolute', background: '#414141' }} />
        <div style={{ width: wp(140), height: hp(72), left: wp(88), top: hp(309), position: 'absolute', background: '#414141' }} />
        <div style={{ width: wp(140), height: hp(52), left: wp(88), top: hp(411), position: 'absolute', background: '#414141' }} />

        <div style={{ left: wp(109), top: hp(50), position: 'absolute', color: 'white', fontSize: scalePx(14), fontWeight: 700 }}>General</div>
        <div style={{ left: wp(109), top: hp(viewLabelTop), position: 'absolute', color: 'white', fontSize: scalePx(10), fontWeight: 700 }}>View</div>
        <div style={{ left: wp(109), top: hp(measureLabelTop), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700 }}>Measure</div>
        <div style={{ left: wp(109), top: hp(outputLabelTop), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700 }}>Output</div>
        <div style={{ left: wp(109), top: hp(390), position: 'absolute', color: 'white', fontSize: scalePx(14), fontWeight: 700 }}>Task</div>

        <img
          src={headerMarkerIcon}
          alt=""
          draggable={false}
          style={{ width: scalePx(7), height: scalePx(7), left: wp(96), top: hp(57), position: 'absolute' }}
        />
        <img
          src={headerMarkerIcon}
          alt=""
          draggable={false}
          style={{ width: scalePx(7), height: scalePx(7), left: wp(96), top: hp(397), position: 'absolute' }}
        />
        <div
          style={{
            width: 0,
            height: 0,
            left: wp(96),
            top: hp(viewLabelTop + sectionArrowOffsetY),
            position: 'absolute',
            borderLeft: `${scalePx(4)} solid transparent`,
            borderRight: `${scalePx(4)} solid transparent`,
            borderTop: `${scalePx(8)} solid #D9D9D9`,
          }}
        />
        <div
          style={{
            width: 0,
            height: 0,
            left: wp(96),
            top: hp(measureLabelTop + sectionArrowOffsetY),
            position: 'absolute',
            borderLeft: `${scalePx(4)} solid transparent`,
            borderRight: `${scalePx(4)} solid transparent`,
            borderTop: `${scalePx(8)} solid #D9D9D9`,
          }}
        />
        <div
          style={{
            width: 0,
            height: 0,
            left: wp(96),
            top: hp(outputLabelTop + sectionArrowOffsetY),
            position: 'absolute',
            borderLeft: `${scalePx(4)} solid transparent`,
            borderRight: `${scalePx(4)} solid transparent`,
            borderTop: `${scalePx(8)} solid #D9D9D9`,
          }}
        />

        {[
          { key: 'pointer' as ToolbarKey, x: 91.5, y: 96, label: 'Mouse', active: selectedToolbarButton === 'pointer' },
          { key: 'pan' as ToolbarKey, x: 136.5, y: 96, label: 'Zoom and drag', active: selectedToolbarButton === 'pan' },
          { key: 'wlww' as ToolbarKey, x: 181.5, y: 96, label: 'Window level', active: selectedToolbarButton === 'wlww' },
          { key: 'invert' as ToolbarKey, x: 91.5, y: 141, label: 'Invert', active: inverted },
          { key: 'magnifier' as ToolbarKey, x: 136.5, y: 141, label: 'Magnification', active: selectedToolbarButton === 'magnifier' },
          { key: 'flip' as ToolbarKey, x: 181.5, y: 141, label: 'Flip', active: flipped },
          {
            key: 'measure-length' as ToolbarKey,
            x: 91.5,
            y: 216,
            label: 'Ruler',
            active: toolSubmenu?.menu === 'measure' || selectedToolbarButton === 'measure-length',
          },
          {
            key: 'measure-draw' as ToolbarKey,
            x: 136.5,
            y: 216,
            label: 'Draw',
            active: toolSubmenu?.menu === 'annotate' || selectedToolbarButton === 'measure-draw',
          },
          { key: 'measure-eraser' as ToolbarKey, x: 181.5, y: 216, label: 'Eraser', active: selectedToolbarButton === 'measure-eraser' },
          { key: 'measure-clear' as ToolbarKey, x: 91.5, y: 261, label: 'Delete all measure', active: flashToolbarButton === 'measure-clear' },
          { key: 'measure-rotate' as ToolbarKey, x: 136.5, y: 261, label: 'Rotate', active: flashToolbarButton === 'measure-rotate' },
          { key: 'measure-reset' as ToolbarKey, x: 181.5, y: 261, label: 'Reset', active: flashToolbarButton === 'measure-reset' },
          { key: 'output-capture' as ToolbarKey, x: 92, y: 336, label: 'Capture', active: flashToolbarButton === 'output-capture' },
          { key: 'output-save' as ToolbarKey, x: 139, y: 336, label: 'Capture save', active: flashToolbarButton === 'output-save' },
          {
            key: 'output-report' as ToolbarKey,
            x: 186,
            y: 336,
            label: 'Report',
            active: isReportActive || reportDrawerOpen,
            icon: displayReportButtonIcons.inactive,
            activeIcon: displayReportButtonIcons.active,
          },
          {
            key: 'task-original' as ToolbarKey,
            x: 92,
            y: 416,
            label: 'Overlay',
            active: viewMode === 'overlay',
            icon: displayToolbarIcons[14],
            activeIcon: activeToolbarIcons[14],
          },
          {
            key: 'task-heatmap' as ToolbarKey,
            x: 139,
            y: 416,
            label: 'Heatmap',
            active: viewMode === 'heatmap',
            icon: displayToolbarIcons[15],
            activeIcon: activeToolbarIcons[15],
          },
        ].map((item, index) => (
          <ToolIcon
            key={item.key}
            icon={item.icon ?? displayToolbarIcons[index]}
            activeIcon={item.activeIcon ?? activeToolbarIcons[index]}
            left={item.x}
            top={item.y}
            active={item.active}
            onClick={() => {
              if (item.key === 'measure-length') {
                toggleToolSubmenu('measure', item.x + TOOL_ICON_SIZE + 4, item.y);
                return;
              }
              if (item.key === 'measure-draw') {
                toggleToolSubmenu('annotate', item.x + TOOL_ICON_SIZE + 4, item.y);
                return;
              }
              handleToolSelection(item.key);
            }}
            label={item.label}
          />
        ))}
        {viewMode === 'heatmap' && (
          <div
            style={{
              width: wp(140),
              left: wp(heatmapControlsPosition.x),
              top: hp(heatmapControlsPosition.y),
              position: 'absolute',
              zIndex: 3,
            }}
            onMouseEnter={() => clearHeatmapControlsTimeout()}
          >
            {isHeatmapControlsExpanded ? (
              <div
                style={{
                  width: wp(140),
                  minHeight: hp(40),
                  padding: `${hp(4)} ${wp(6)}`,
                  border: `${scalePx(1)} solid #5A5A5A`,
                  background: 'rgba(65, 65, 65, 0.8)',
                  boxSizing: 'border-box',
                  position: 'relative',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: wp(4),
                    marginBottom: hp(3),
                  }}
                >
                  <button
                    type="button"
                    onPointerDown={handleHeatmapControlsDragStart}
                    aria-label="Drag heatmap opacity panel"
                    style={{
                      width: wp(12),
                      height: hp(12),
                      border: 'none',
                      background: 'transparent',
                      padding: 0,
                      cursor: 'grab',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    <img
                      src={assetPath('7 7.png')}
                      alt=""
                      draggable={false}
                      style={{
                        width: scalePx(7),
                        height: scalePx(7),
                        display: 'block',
                        objectFit: 'contain',
                      }}
                    />
                  </button>
                  <span
                    style={{
                      color: '#D9D9D9',
                      fontSize: scalePx(10),
                      fontWeight: 700,
                      lineHeight: 1,
                      minWidth: wp(28),
                    }}
                  >
                    {heatmapOpacity}%
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      setIsHeatmapPresetMenuVisible((current) => !current);
                      showHeatmapControls(4200);
                    }}
                    style={{
                      width: wp(36),
                      height: hp(16),
                      border: `${scalePx(1)} solid #666666`,
                      background: 'rgba(75, 75, 75, 0.8)',
                      color: '#F1F1F1',
                      fontSize: scalePx(9),
                      fontWeight: 700,
                      cursor: 'pointer',
                      padding: 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: wp(3),
                      marginLeft: 'auto',
                    }}
                  >
                    <span>{heatmapOpacity}%</span>
                    <span
                      aria-hidden="true"
                      style={{
                        width: 0,
                        height: 0,
                        borderLeft: `${scalePx(3)} solid transparent`,
                        borderRight: `${scalePx(3)} solid transparent`,
                        borderTop: `${scalePx(5)} solid rgba(255, 255, 255, 0.82)`,
                      }}
                    />
                  </button>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={heatmapOpacity}
                  onChange={(event) => {
                    setHeatmapOpacity(Number(event.target.value));
                    showHeatmapControls();
                  }}
                  style={{
                    width: '100%',
                    height: hp(12),
                    accentColor: '#9A9A9A',
                    cursor: 'pointer',
                    margin: 0,
                    display: 'block',
                  }}
                />
                {isHeatmapPresetMenuVisible && (
                  <div
                    style={{
                      position: 'absolute',
                      right: wp(6),
                      top: hp(19),
                      width: wp(44),
                      border: `${scalePx(1)} solid #5A5A5A`,
                      background: 'rgba(65, 65, 65, 0.8)',
                      boxShadow: '0 8px 14px rgba(0, 0, 0, 0.24)',
                    }}
                  >
                    {heatmapOpacityPresets.map((preset) => {
                      const active = heatmapOpacity === preset;
                      return (
                        <button
                          key={preset}
                          type="button"
                          onClick={() => {
                            setHeatmapOpacity(preset);
                            setIsHeatmapPresetMenuVisible(false);
                            showHeatmapControls();
                          }}
                          style={{
                            width: '100%',
                            height: hp(18),
                            border: 'none',
                            borderBottom: preset === heatmapOpacityPresets[heatmapOpacityPresets.length - 1]
                              ? 'none'
                              : `${scalePx(1)} solid #575757`,
                            background: active ? 'rgba(106, 106, 106, 0.8)' : 'rgba(65, 65, 65, 0.8)',
                            color: '#F1F1F1',
                            fontSize: scalePx(9),
                            fontWeight: 700,
                            cursor: 'pointer',
                            padding: 0,
                          }}
                        >
                          {preset}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : (
              <button
                type="button"
                onClick={() => {
                  setIsHeatmapPresetMenuVisible(false);
                  showHeatmapControls();
                }}
                style={{
                  width: wp(140),
                  height: hp(20),
                  border: `${scalePx(1)} solid #5A5A5A`,
                  background: '#414141',
                  color: '#D9D9D9',
                  fontSize: scalePx(10),
                  fontWeight: 700,
                  textAlign: 'left',
                  padding: `0 ${wp(7)}`,
                  cursor: 'pointer',
                }}
              >
                {heatmapOpacity}%
              </button>
            )}
          </div>
        )}
        <RenewToolSubmenu
          visible={toolSubmenu?.menu === 'measure'}
          left={wp(toolSubmenu?.left ?? 0)}
          top={hp(toolSubmenu?.top ?? 0)}
          width={wp(104)}
          onClose={closeToolSubmenu}
          items={[
            { key: 'length', label: 'Length', active: activeMeasureSubtool === 'length', onClick: () => activateMeasureSubtool('length') },
            {
              key: 'bidirectional',
              label: 'Bidirectional',
              active: activeMeasureSubtool === 'bidirectional',
              onClick: () => activateMeasureSubtool('bidirectional'),
            },
            { key: 'angle', label: 'Angle', active: activeMeasureSubtool === 'angle', onClick: () => activateMeasureSubtool('angle') },
          ]}
        />
        <RenewToolSubmenu
          visible={toolSubmenu?.menu === 'annotate'}
          left={wp(toolSubmenu?.left ?? 0)}
          top={hp(toolSubmenu?.top ?? 0)}
          width={wp(118)}
          onClose={closeToolSubmenu}
          items={[
            { key: 'text', label: 'Annotation', active: activeMeasureSubtool === 'text', onClick: () => activateMeasureSubtool('text') },
            { key: 'arrow', label: 'Arrow', active: activeMeasureSubtool === 'arrow', onClick: () => activateMeasureSubtool('arrow') },
            { key: 'ellipse', label: 'Ellipse', active: activeMeasureSubtool === 'ellipse', onClick: () => activateMeasureSubtool('ellipse') },
            { key: 'rect', label: 'Rectangle', active: activeMeasureSubtool === 'rect', onClick: () => activateMeasureSubtool('rect') },
            { key: 'circle', label: 'Circle', active: activeMeasureSubtool === 'circle', onClick: () => activateMeasureSubtool('circle') },
            { key: 'roi-free', label: 'Freehand ROI', active: activeMeasureSubtool === 'roi-free', onClick: () => activateMeasureSubtool('roi-free') },
            { key: 'spline-roi', label: 'Spline ROI', active: activeMeasureSubtool === 'spline-roi', onClick: () => activateMeasureSubtool('spline-roi') },
            { key: 'livewire', label: 'Livewire Tool', active: activeMeasureSubtool === 'livewire', onClick: () => activateMeasureSubtool('livewire') },
          ]}
        />
        <OutputCapturePanel
          visible
          collapsed={isCapturePanelCollapsed}
          left={wp(88)}
          top={hp(outputCaptureTop)}
          width={wp(140)}
          height={hp(590)}
          captures={capturedOutputs}
          onToggle={() => {
            setIsCapturePanelCollapsed((current) => !current);
          }}
          onRemove={(id) => {
            setCapturedOutputs((current) => current.filter((item) => item.id !== id));
          }}
          onClear={() => {
            setCapturedOutputs([]);
          }}
        />

        {!isReportWorkspaceVisible && (
          <>
        <div style={{ left: wp(251), top: hp(49), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700, zIndex: 2 }}>Panorama</div>
        <img
          src={headerMarkerIcon}
          alt=""
          draggable={false}
          style={{ width: scalePx(7), height: scalePx(7), left: wp(240), top: hp(55), position: 'absolute', zIndex: 2 }}
        />
        {workspaceSection === 'studies' && (
          <RenewStudiesDock
            visible
            width={wp(studiesPanelWidth)}
            height={hp(studiesPanelHeight)}
            left={wp(studiesPanelLeft)}
            top={hp(studiesPanelTop)}
            studies={combinedStudies as FolderStudy[]}
            selectedSeriesId={selectedFolderSeriesId || locationState.folderSelectedSeriesId || null}
            onSelectSeries={(seriesId) => {
              void handleSelectSeries(seriesId);
            }}
          />
        )}
        {isChartVisible && (
          <>
            <div style={{ left: wp(251), top: hp(803), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700, zIndex: 2 }}>Dental Chart</div>
            <img
              src={headerMarkerIcon}
              alt=""
              draggable={false}
              style={{ width: scalePx(7), height: scalePx(7), left: wp(240), top: hp(809), position: 'absolute', zIndex: 2 }}
            />
          </>
        )}

        {isChartBodyVisible && (
          <>
            <button
              type="button"
              onClick={() => setNumberingSystem((current) => (current === 'fdi' ? 'univ' : 'fdi'))}
              aria-label="Toggle numbering system"
              style={{
                left: wp(fdiLeft),
                top: hp(845),
                position: 'absolute',
                display: 'inline-flex',
                alignItems: 'center',
                gap: wp(4),
                fontWeight: 700,
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                zIndex: 16,
                padding: 0,
              }}
            >
              <span style={{ color: numberingSystem === 'fdi' ? '#FFFFFF' : '#6F7A86', fontSize: scalePx(14) }}>FDI</span>
              <span style={{ color: '#9C9C9C', fontSize: scalePx(13) }}>/</span>
              <span style={{ color: numberingSystem === 'univ' ? '#FFFFFF' : '#6F7A86', fontSize: scalePx(13) }}>Univ</span>
            </button>
            <div style={{ left: wp(251), top: hp(panoLabelTop), position: 'absolute', color: '#D39C00', fontSize: scalePx(14), fontWeight: 700, zIndex: 14 }}>
              {flipped ? 'L' : 'R'}
            </div>
            <div style={{ left: wp(lLabelLeft), top: hp(panoLabelTop), position: 'absolute', color: '#D39C00', fontSize: scalePx(14), fontWeight: 700, zIndex: 14 }}>
              {flipped ? 'R' : 'L'}
            </div>
            <div
              style={{
                width: wp(chartOdontoLineWidth),
                height: scalePx(1),
                left: wp(chartOdontoLineLeft),
                top: hp(chartOdontoLineY),
                position: 'absolute',
                background: '#B4B4B4',
              }}
            />
            <div
              style={{
                width: scalePx(1),
                height: hp(chartOdontoVerticalHeight),
                left: wp(chartOdontoCenterX),
                top: hp(chartOdontoVerticalTop),
                position: 'absolute',
                background: '#B4B4B4',
              }}
            />
            <div style={{ width: scalePx(1), height: hp(264), left: wp(chartSectionLeft), top: hp(804), position: 'absolute', background: '#5C5C5C' }} />
            <div style={{ width: scalePx(1), height: hp(chartContentHeight), left: wp(chartLegendDividerX), top: hp(chartContentTop), position: 'absolute', background: '#5C5C5C' }} />
            <div style={{ width: scalePx(1), height: hp(250), left: wp(rightEdge - 1), top: hp(813), position: 'absolute', background: '#5C5C5C' }} />
            <div style={{ width: wp(chartLegendDividerX - chartSectionLeft), height: hp(chartContentHeight), left: wp(chartSectionLeft), top: hp(chartContentTop), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
          </>
        )}

        {isChartBodyVisible && upperLeftOrders.map((order, index) => {
          const width = Number(upperSizes[order - 1].split(' ')[0]);
          const height = Number(upperSizes[order - 1].split(' ')[1]);
          const toothFdi = String(displayedUpperArch.left[index]);
          const toothStatus = toothStatusByFdi[toothFdi] || 'healthy';
          const matchesLegendFilter = !activeLegendFilter || toothStatus === activeLegendFilter;
          return (
            <ToothSlotImage
              key={`upper-left-${order}`}
              arch="U"
              order={order}
              toothFdi={toothFdi}
              status={toothStatus}
              left={toothSlotCentersLeft[index] - width / 2}
              top={upperBaseline - height}
              width={width}
              height={height}
              flipX={!flipped}
              active={activeTooth === toothFdi}
              hasDetection={(Boolean(detectionsByTooth[toothFdi]?.length) || Boolean(findingSignalByTooth[toothFdi])) && matchesLegendFilter}
              dimmed={!matchesLegendFilter}
              onClick={handleToothSelect}
              onHoverChange={setHoveredToothAnchor}
            />
          );
        })}
        {isChartBodyVisible && upperRightOrders.map((order, index) => {
          const width = Number(upperSizes[order - 1].split(' ')[0]);
          const height = Number(upperSizes[order - 1].split(' ')[1]);
          const toothFdi = String(displayedUpperArch.right[index]);
          const toothStatus = toothStatusByFdi[toothFdi] || 'healthy';
          const matchesLegendFilter = !activeLegendFilter || toothStatus === activeLegendFilter;
          return (
            <ToothSlotImage
              key={`upper-right-${order}`}
              arch="U"
              order={order}
              toothFdi={toothFdi}
              status={toothStatus}
              left={toothSlotCentersRight[index] - width / 2}
              top={upperBaseline - height}
              width={width}
              height={height}
              flipX={flipped}
              active={activeTooth === toothFdi}
              hasDetection={(Boolean(detectionsByTooth[toothFdi]?.length) || Boolean(findingSignalByTooth[toothFdi])) && matchesLegendFilter}
              dimmed={!matchesLegendFilter}
              onClick={handleToothSelect}
              onHoverChange={setHoveredToothAnchor}
            />
          );
        })}
        {isChartBodyVisible && lowerLeftOrders.map((order, index) => {
          const width = Number(lowerSizes[order - 1].split(' ')[0]);
          const height = Number(lowerSizes[order - 1].split(' ')[1]);
          const toothFdi = String(displayedLowerArch.left[index]);
          const toothStatus = toothStatusByFdi[toothFdi] || 'healthy';
          const matchesLegendFilter = !activeLegendFilter || toothStatus === activeLegendFilter;
          return (
            <ToothSlotImage
              key={`lower-left-${order}`}
              arch="L"
              order={order}
              toothFdi={toothFdi}
              status={toothStatus}
              left={toothSlotCentersLeft[index] - width / 2}
              top={lowerTop}
              width={width}
              height={height}
              flipX={!flipped}
              active={activeTooth === toothFdi}
              hasDetection={(Boolean(detectionsByTooth[toothFdi]?.length) || Boolean(findingSignalByTooth[toothFdi])) && matchesLegendFilter}
              dimmed={!matchesLegendFilter}
              onClick={handleToothSelect}
              onHoverChange={setHoveredToothAnchor}
            />
          );
        })}
        {isChartBodyVisible && lowerRightOrders.map((order, index) => {
          const width = Number(lowerSizes[order - 1].split(' ')[0]);
          const height = Number(lowerSizes[order - 1].split(' ')[1]);
          const toothFdi = String(displayedLowerArch.right[index]);
          const toothStatus = toothStatusByFdi[toothFdi] || 'healthy';
          const matchesLegendFilter = !activeLegendFilter || toothStatus === activeLegendFilter;
          return (
            <ToothSlotImage
              key={`lower-right-${order}`}
              arch="L"
              order={order}
              toothFdi={toothFdi}
              status={toothStatus}
              left={toothSlotCentersRight[index] - width / 2}
              top={lowerTop}
              width={width}
              height={height}
              flipX={flipped}
              active={activeTooth === toothFdi}
              hasDetection={(Boolean(detectionsByTooth[toothFdi]?.length) || Boolean(findingSignalByTooth[toothFdi])) && matchesLegendFilter}
              dimmed={!matchesLegendFilter}
              onClick={handleToothSelect}
              onHoverChange={setHoveredToothAnchor}
            />
          );
        })}

        <DentalChartLegendOverlay
          visible={isChartBodyVisible}
          width={chartLegendWidth}
          height={chartLegendHeight}
          left={wp(chartLegendLeft + 22)}
          top={hp(chartLegendTop)}
          items={legendItems}
          activeKey={activeLegendFilter}
          focusedKey={focusedCondition}
          counts={legendCounts}
          activeBorderWidth={scalePx(2)}
          hoverBorderWidth={scalePx(1)}
          glowSize={scalePx(10)}
          badgeFontSize={scalePx(12)}
          labelFontSize={scalePx(13)}
          countGap={scalePx(6)}
          onToggle={handleLegendToggle}
        />

        <ReportWorkspaceControls
          showReportButton={false}
          reportButtonLeft={wp(reportLeft)}
          reportButtonTop={hp(reportTop)}
          reportButtonWidth={wp(77)}
          reportButtonHeight={hp(77)}
          reportActive={reportDrawerOpen}
          outlineWidth={scalePx(3)}
          outlineOffset={scalePx(2)}
          glowSize={scalePx(18)}
          activeIconSrc={displayReportButtonIcons.active}
          inactiveIconSrc={displayReportButtonIcons.inactive}
          onOpenReport={() => { void handleOpenReportPanel(); }}
          reportError={reportError || (!panoViewerUrl ? 'No panorama source available for viewer.' : null)}
          reportErrorLeft={wp(251)}
          reportErrorTop={hp(74)}
          reportErrorFontSize={scalePx(11)}
          showChartToggle={false}
          chartToggleLeft={wp(chartSectionLeft)}
          chartToggleTop={hp(panoChartToggleTop)}
          chartToggleWidth={wp(chartSectionWidth)}
          chartToggleHeight={hp(18)}
          chartToggleBorderWidth={scalePx(1)}
          chartToggleFontSize={scalePx(12)}
          markerIconSrc={headerMarkerIcon}
          markerIconWidth={scalePx(7)}
          markerIconHeight={scalePx(7)}
          markerIconLeft={wp(12)}
          markerIconTop={hp(6)}
          chartToggleLabelMarginLeft={wp(28)}
          onShowChart={handleChartToggle}
        />

        {false && isChartBodyVisible && (
          <div style={{ width: wp(chartLegendWidth), height: hp(chartLegendHeight), left: wp(chartLegendLeft + 22), top: hp(chartLegendTop), position: 'absolute' }}>
            {legendItems.map((item) => (
              <div key={item.label}>
                <button
                  type="button"
                  onClick={() => handleLegendToggle(item.key)}
                  aria-pressed={activeLegendFilter === item.key}
                  style={{
                    width: relativePercent(19, chartLegendWidth),
                    height: relativePercent(19, chartLegendHeight),
                    left: 0,
                    top: relativePercent(item.top - 834, chartLegendHeight),
                    position: 'absolute',
                    background: '#808181',
                    border: activeLegendFilter === item.key ? `${scalePx(2)} solid #00C0F3` : focusedCondition === item.key ? `${scalePx(1)} solid rgba(0, 192, 243, 0.7)` : 'none',
                    boxShadow: activeLegendFilter === item.key ? `0 0 ${scalePx(10)} rgba(0, 192, 243, 0.35)` : 'none',
                    cursor: 'pointer',
                    padding: 0,
                  }}
                >
                  {(activeLegendFilter === item.key || focusedCondition === item.key) && (
                    <div
                      style={{
                        position: 'absolute',
                        inset: 0,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: item.key === 'healthy' ? '#111111' : '#ffffff',
                        fontSize: scalePx(12),
                        fontWeight: 700,
                      }}
                    >
                      ✓
                    </div>
                  )}
                </button>
                <div
                  style={{
                    width: relativePercent(52, chartLegendWidth),
                    height: relativePercent(19, chartLegendHeight),
                    left: relativePercent(chartLegendWidth - 52 - 12, chartLegendWidth),
                    top: relativePercent(item.top - 834, chartLegendHeight),
                    position: 'absolute',
                    background: item.color,
                  }}
                />
                <div
                  style={{
                    left: relativePercent(30, chartLegendWidth),
                    top: relativePercent(item.top - 834, chartLegendHeight),
                    position: 'absolute',
                    color: 'white',
                    fontSize: scalePx(13),
                    fontWeight: 700,
                  }}
                >
                  {item.label}
                  <span style={{ color: '#B7B7B7', marginLeft: scalePx(6) }}>
                    {legendCounts[item.key]}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

          {false && isChartBodyVisible && (
            <button
              type="button"
              onClick={() => { void handleStartReport(); }}
              aria-pressed={isReportActive || reportDrawerOpen}
              aria-label="Open Dratf workspace"
              style={{
                width: wp(77),
                height: hp(77),
                left: wp(reportLeft),
                top: hp(reportTop),
                position: 'absolute',
                zIndex: 30,
                borderRadius: '50%',
                outline: isReportActive || reportDrawerOpen ? `${scalePx(3)} solid #00C0F3` : 'none',
                outlineOffset: scalePx(2),
                boxShadow: isReportActive || reportDrawerOpen ? `0 0 ${scalePx(18)} rgba(0, 192, 243, 0.45)` : 'none',
                cursor: 'pointer',
              }}
            >
              <img
                style={{
                  width: '100%',
                  height: '100%',
                  display: 'block',
                  filter: isReportActive || reportDrawerOpen ? 'drop-shadow(0 0 10px rgba(0, 192, 243, 0.7))' : 'none',
                }}
                src={isReportActive || reportDrawerOpen ? displayReportButtonIcons.active : displayReportButtonIcons.inactive}
                alt=""
              />
            </button>
          )}

          {false && reportError && (
            <div
              style={{
                left: wp(251),
                top: hp(74),
                position: 'absolute',
                color: '#FFB4B4',
                fontSize: scalePx(11),
                fontWeight: 700,
                zIndex: 20,
              }}
            >
              {reportError}
            </div>
          )}

          {false && !isChartVisible && (
            <button
              type="button"
              onClick={handleChartToggle}
              aria-pressed={false}
              aria-label="Show dental chart"
              style={{
                width: wp(chartSectionWidth),
                height: hp(18),
                left: wp(chartSectionLeft),
                top: hp(panoChartToggleTop),
                position: 'absolute',
                border: `${scalePx(1)} solid #4C4C4C`,
                background: '#5C5C5C',
                color: '#FFFFFF',
                fontSize: scalePx(12),
                fontWeight: 700,
                cursor: 'pointer',
                zIndex: 4,
                padding: 0,
                textAlign: 'left',
              }}
            >
              <img
                src={headerMarkerIcon}
                alt=""
                draggable={false}
                style={{ width: scalePx(7), height: scalePx(7), position: 'absolute', left: wp(12), top: hp(6) }}
              />
              <span style={{ marginLeft: wp(28), display: 'inline-block' }}>Dental Chart</span>
            </button>
          )}
          </>
        )}
        </div>
      </div>
      {reportSessionId && (
        <WebReportDrawer
          sessionId={reportSessionId}
          selectedToothId={activeTooth}
          open={reportDrawerOpen}
          layout="dock"
          positionMode="absolute"
          onClose={() => {
            setReportDrawerOpen(false);
            setIsReportActive(false);
            setWorkspaceSection((current) => (current === 'report' ? 'none' : current));
          }}
        />
      )}
      <ToothHoverHud
        anchor={isReportWorkspaceVisible ? null : hoveredToothAnchor}
        panel={hoveredToothPanel}
        minWidth={wp(228)}
        borderWidth={scalePx(1)}
        paddingY={hp(10)}
        paddingX={wp(12)}
        titleFontSize={scalePx(16)}
        labelFontSize={scalePx(10)}
        valueFontSize={scalePx(12)}
        titleGap={hp(10)}
        labelGap={hp(3)}
        columnGap={wp(18)}
        rowGap={hp(10)}
      />
    </div>
  );
}

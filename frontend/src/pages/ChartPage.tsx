import { useLocation } from 'react-router-dom';
import { useState, useRef, useEffect, useMemo } from 'react';
import { createPortal } from 'react-dom';
import dicomParser from 'dicom-parser';
import { BottomTeethChart } from '../components/BottomTeethChart';
import { WebReportDrawer } from '../components/WebReportDrawer';
import { CornerstoneViewer, CornerstoneGridViewer, MinimalCornerstoneDicomViewer } from '../viewer';
import {
  AlertTriangle, Activity, Zap, Layers, Image as ImageIcon,
  MousePointer, Hand, RotateCw, FlipHorizontal, Sliders,
  Ruler, PenLine, Loader2, RotateCcw, AlertCircle, Skull,
  ClipboardList, Quote, Sparkles, MousePointer2, Eraser, Rotate3d,
  Trash2, Monitor, ChevronsUpDown, Camera, Crop, Search
} from 'lucide-react';
import { TopHeader } from '../components/TopHeader';
import { StudiesWorkspacePanel } from '../components/chart/StudiesWorkspacePanel';
import { setActiveTool as setCornerstoneActiveTool, clearAllAnnotations } from '../viewer/cornerstone/tools';
import { DicomMetadataOverlay } from '../viewer/DicomMetadataOverlay';
import { parseDicomMetadataFromDataSet, type DicomOverlayMetadata } from '../viewer/cornerstone/dicomMetadata';
import { estimateAutoWindowFromPixelData } from '../viewer/cornerstone/autoWindow';
import { inspectLocalDicomFile } from '../viewer/cornerstone/dicomDebug';
import { createWebReportFromChart } from '../lib/webReportApi';
import { buildWebReportKeywords, countWebReportFindingTeeth } from '../lib/webReportKeywords';
import type { FolderStudy } from '../features/upload/dicomFolderStudies';
import { requestAsyncDetection } from '../features/upload/uploadApi';
import { fetchServerFolderIndex, materializeServerStudy, resolveServerAssetUrl } from '../lib/folderLeaderApi';
import { FileText } from 'lucide-react';

type ChartPageProps = {
  result?: any;
};

// Fallback data
const mockCounts = [
  { label: 'Teeth', value: '-' },
  { label: 'Caries', value: '-' },
  { label: 'Periapical', value: '-' },
  { label: 'CEJ masks', value: '-' },
  { label: 'Bone level masks', value: '-' },
];

const mockFindings = [
  { label: 'No Data', value: '', color: 'bg-gray-700' },
];

const padTimestampUnit = (value: number) => String(value).padStart(2, '0');

const formatCaptureTimeline = (date: Date) => {
  const hour = padTimestampUnit(date.getHours());
  const minute = padTimestampUnit(date.getMinutes());
  return `${hour}:${minute}`;
};

const formatCaptureFileTimestamp = (date: Date) => {
  const second = padTimestampUnit(date.getSeconds());
  return `${formatCaptureTimeline(date)}-${second}`;
};

const ODONTOGRAM_TEETH = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28, 48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38];
const MAGNIFIER_SIZE_PX = 200;
const MAGNIFIER_ZOOM_FACTOR = 1.5;
const MAGNIFIER_CURSOR_OFFSET_PX = 28;
const MAGNIFIER_EDGE_PADDING_PX = 12;
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

const clampNumber = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

const clampByte = (value: number) => clampNumber(Math.round(value), 0, 255);

const withDirectApiBase = (path: string) => {
  if (/^https?:\/\//i.test(path)) return path;
  return `${DIRECT_API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
};

const applyRoiHistogramBoost = (
  imageData: ImageData,
  blend = 0.62,
  clipFactor = 2.2
) => {
  const { data, width, height } = imageData;
  const pixelCount = width * height;
  if (!pixelCount) return imageData;

  const histogram = new Uint32Array(256);
  const luminance = new Uint8Array(pixelCount);

  for (let i = 0; i < pixelCount; i += 1) {
    const idx = i * 4;
    const lum = clampByte(data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114);
    luminance[i] = lum;
    histogram[lum] += 1;
  }

  const clipLimit = Math.max(1, Math.floor((pixelCount / 256) * clipFactor));
  let excess = 0;
  for (let i = 0; i < 256; i += 1) {
    if (histogram[i] > clipLimit) {
      excess += histogram[i] - clipLimit;
      histogram[i] = clipLimit;
    }
  }

  const redistributed = Math.floor(excess / 256);
  const remainder = excess - redistributed * 256;
  for (let i = 0; i < 256; i += 1) {
    histogram[i] += redistributed + (i < remainder ? 1 : 0);
  }

  const cdf = new Uint32Array(256);
  let cumulative = 0;
  for (let i = 0; i < 256; i += 1) {
    cumulative += histogram[i];
    cdf[i] = cumulative;
  }

  let cdfMin = 0;
  for (let i = 0; i < 256; i += 1) {
    if (cdf[i] > 0) {
      cdfMin = cdf[i];
      break;
    }
  }

  const denominator = Math.max(1, pixelCount - cdfMin);
  for (let i = 0; i < pixelCount; i += 1) {
    const idx = i * 4;
    const originalLum = luminance[i];
    const equalizedLum = clampByte(((cdf[originalLum] - cdfMin) / denominator) * 255);
    const targetLum = originalLum + (equalizedLum - originalLum) * blend;
    const gain = originalLum > 0 ? targetLum / originalLum : targetLum / 255;

    data[idx] = clampByte(data[idx] * gain);
    data[idx + 1] = clampByte(data[idx + 1] * gain);
    data[idx + 2] = clampByte(data[idx + 2] * gain);
  }

  return imageData;
};

const applyUnsharpMask = (imageData: ImageData, amount = 0.68) => {
  const { data, width, height } = imageData;
  if (!width || !height) return imageData;

  const pixelCount = width * height;
  const luminance = new Float32Array(pixelCount);
  const blurred = new Float32Array(pixelCount);

  for (let i = 0; i < pixelCount; i += 1) {
    const idx = i * 4;
    luminance[i] = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      let weightSum = 0;
      for (let ky = -1; ky <= 1; ky += 1) {
        for (let kx = -1; kx <= 1; kx += 1) {
          const sampleX = clampNumber(x + kx, 0, width - 1);
          const sampleY = clampNumber(y + ky, 0, height - 1);
          const weight = kx === 0 && ky === 0 ? 4 : kx === 0 || ky === 0 ? 2 : 1;
          sum += luminance[sampleY * width + sampleX] * weight;
          weightSum += weight;
        }
      }
      blurred[y * width + x] = sum / Math.max(1, weightSum);
    }
  }

  for (let i = 0; i < pixelCount; i += 1) {
    const idx = i * 4;
    const originalLum = luminance[i];
    const sharpenedLum = clampByte(originalLum + (originalLum - blurred[i]) * amount);
    const gain = originalLum > 0 ? sharpenedLum / originalLum : sharpenedLum / 255;

    data[idx] = clampByte(data[idx] * gain);
    data[idx + 1] = clampByte(data[idx + 1] * gain);
    data[idx + 2] = clampByte(data[idx + 2] * gain);
  }

  return imageData;
};

const applyEdgeAwareBoost = (imageData: ImageData, amount = 0.42) => {
  const { data, width, height } = imageData;
  if (width < 3 || height < 3) return imageData;

  const pixelCount = width * height;
  const luminance = new Float32Array(pixelCount);
  const edges = new Float32Array(pixelCount);

  for (let i = 0; i < pixelCount; i += 1) {
    const idx = i * 4;
    luminance[i] = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
  }

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const idx = y * width + x;
      const gx =
        -luminance[idx - width - 1] - 2 * luminance[idx - 1] - luminance[idx + width - 1] +
        luminance[idx - width + 1] + 2 * luminance[idx + 1] + luminance[idx + width + 1];
      const gy =
        -luminance[idx - width - 1] - 2 * luminance[idx - width] - luminance[idx - width + 1] +
        luminance[idx + width - 1] + 2 * luminance[idx + width] + luminance[idx + width + 1];
      edges[idx] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
    }
  }

  for (let i = 0; i < pixelCount; i += 1) {
    const edgeWeight = (edges[i] / 255) * amount;
    if (edgeWeight <= 0.02) continue;
    const idx = i * 4;
    const gain = 1 + edgeWeight;
    data[idx] = clampByte(data[idx] * gain);
    data[idx + 1] = clampByte(data[idx + 1] * gain);
    data[idx + 2] = clampByte(data[idx + 2] * gain);
  }

  return imageData;
};

const enhanceMagnifierImage = (imageData: ImageData) => {
  applyRoiHistogramBoost(imageData);
  applyUnsharpMask(imageData);
  applyEdgeAwareBoost(imageData);
  return imageData;
};

export function ChartPage(props?: ChartPageProps) {
  const location = useLocation();
  const locationState = (location.state as any) || {};
  const [activeFolderStudies, setActiveFolderStudies] = useState<FolderStudy[]>(() => {
    const raw = (locationState.originalFolderStudies as FolderStudy[] | undefined) || [];
    const seen = new Set<string>();
    return raw.filter(s => { if (seen.has(s.id)) return false; seen.add(s.id); return true; });
  });
  const [serverStudies, setServerStudies] = useState<any[]>([]);

  useEffect(() => {
    if (locationState.folderSource === 'server') {
      fetchServerFolderIndex().then(data => setServerStudies(data.studies || [])).catch(console.error);
    }
  }, [locationState.folderSource]);

  const combinedStudies = useMemo(() => {
    const activeIds = new Set(activeFolderStudies.map(s => s.id));
    const activeFingerprints = new Set(activeFolderStudies.map(s => `${s.label}::${s.description}::${s.patientId}`));
    const additional = serverStudies.filter(s => {
      if (activeIds.has(s.id)) return false;
      const fp = `${s.label}::${s.description}::${s.patientId}`;
      if (activeFingerprints.has(fp)) return false;
      return true;
    }).map(s => ({
      ...s,
      previewUrl: s.previewUrl ? resolveServerAssetUrl(s.previewUrl) : undefined
    }));
    // Final dedup pass to prevent any remaining duplicates
    const merged = [...activeFolderStudies, ...additional];
    const seen = new Set<string>();
    return merged.filter(s => { if (seen.has(s.id)) return false; seen.add(s.id); return true; });
  }, [activeFolderStudies, serverStudies]);

  const originalFolderStudies = activeFolderStudies;
  const originalFolderMode = Boolean(locationState.originalFolderMode && originalFolderStudies.length > 0);
  const findFolderSeriesById = (seriesId: string | null) =>
    originalFolderStudies
      .flatMap((study) => study.series)
      .find((series) => series.id === seriesId) || null;
  const initialFolderSeriesId =
    locationState.folderSelectedSeriesId ||
    originalFolderStudies.flatMap((study) => study.series)[0]?.id ||
    null;
  const initialFolderSeries = findFolderSeriesById(initialFolderSeriesId);
  const initialResult = props?.result ?? locationState?.result;
  const inferVolume = (res: any, folderSeries?: { volumeEligible?: boolean } | null) =>
    Boolean(res?.is_volume || folderSeries?.volumeEligible);
  const initialIsVolume = inferVolume(initialResult, initialFolderSeries);
  const initialViewMode =
    originalFolderMode || locationState.originalIsDicom
      ? 'original'
      : initialIsVolume
        ? 'original'
        : 'overlay';
  const [result, setResult] = useState<any>(initialResult);
  const [jobId, setJobId] = useState<string | null>(locationState?.jobId || null);
  const [isProcessing, setIsProcessing] = useState(!result && !!locationState?.jobId);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [timestamp, setTimestamp] = useState(Date.now());
  const [reportSessionId, setReportSessionId] = useState<string | null>(locationState?.reportSessionId || null);
  const [reportDrawerOpen, setReportDrawerOpen] = useState(false);
  const [reportStartState, setReportStartState] = useState<'idle' | 'creating'>('idle');
  const [reportError, setReportError] = useState<string | null>(null);
  const [captureNotice, setCaptureNotice] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [workspaceSection, setWorkspaceSection] = useState<'studies' | 'report'>('studies');
  const [selectedFolderSeriesId, setSelectedFolderSeriesId] = useState<string | null>(initialFolderSeriesId);
  const [captureGallery, setCaptureGallery] = useState<Array<{
    id: string;
    label: string;
    dataUrl: string;
    reportDataUrl: string;
    size: string;
    createdAt: string;
  }>>([]);
  const [draggingCaptureId, setDraggingCaptureId] = useState<string | null>(null);
  const [assignedCaptureSlots, setAssignedCaptureSlots] = useState<Record<string, {
    id: string;
    label: string;
    dataUrl: string;
    createdAt: string;
  }>>({});

  const [selectedTooth, setSelectedTooth] = useState<number | undefined>(undefined);
  const [activeLegendFilter, setActiveLegendFilter] = useState<'triage-3' | 'triage-2' | 'triage-1' | 'implant' | 'missing' | null>(null);
  const [viewMode, setViewMode] = useState<'overlay' | 'original' | 'heatmap'>(initialViewMode);
  const [overlayPreset, setOverlayPreset] = useState<'all' | 'sinus' | 'nerve' | 'tooth' | 'sinus-upper-tooth' | 'nerve-lower-tooth'>('all');
  const [dicomHudMetadata, setDicomHudMetadata] = useState<DicomOverlayMetadata | null>(null);
  const [dicomPreviewDataUrl, setDicomPreviewDataUrl] = useState<string | null>(null);
  const [dicomAutoWindow, setDicomAutoWindow] = useState<{ level: number; width: number } | null>(null);
  const [containerHeight] = useState(560);
  const [numberingSystem, setNumberingSystem] = useState<'fdi' | 'univ'>('fdi'); // [NEW]

  // Tools
  const [activeTool, setLocalActiveTool] = useState<string>('pointer');
  const [selectedToolbarButton, setSelectedToolbarButton] = useState<string>('pointer');
  const [activeSubTool, setActiveSubTool] = useState<string | null>(null);
  const [viewerMode, setViewerMode] = useState<'single' | 'grid'>(initialIsVolume ? 'grid' : 'single');

  // Viewport State
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [inverted, setInverted] = useState(false);
  const [rotation, setRotation] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [gridLayout, setGridLayout] = useState({ rows: 2, cols: 2 });
  const [tempGridLayout, setTempGridLayout] = useState({ rows: 2, cols: 2 });
  const [cornerstoneResetToken, setCornerstoneResetToken] = useState(0);
  const [cornerstoneAutoWindowToken, setCornerstoneAutoWindowToken] = useState(0);
  const [captureRect, setCaptureRect] = useState<{ x: number; y: number; w: number; h: number; active: boolean } | null>(null);
  const [magnifierState, setMagnifierState] = useState<{
    visible: boolean;
    clientX: number;
    clientY: number;
    viewerX: number;
    viewerY: number;
    imgX: number;
    imgY: number;
  }>({ visible: false, clientX: 0, clientY: 0, viewerX: 0, viewerY: 0, imgX: 0, imgY: 0 });

  const mmPerPixel = (result as any)?.mm_per_pixel || (result as any)?.mm_per_px || 0.1;
  const hasData = !!result;
  const reportKeywords = buildWebReportKeywords(result);
  const reportFindingCount = countWebReportFindingTeeth(result);
  const selectedFolderSeries = findFolderSeriesById(selectedFolderSeriesId);
  const isVolumeCase = inferVolume(result, selectedFolderSeries);

  useEffect(() => {
    let timer: any;
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

    if (jobId && !result) {
      setIsProcessing(true);
      timer = setInterval(async () => {
        try {
          // Increment progress slightly while waiting
          setLoadingProgress(p => Math.min(p + 5, 95));

          let res = await fetch(`/api/detect/status/${jobId}`);
          const contentType = res.headers.get('content-type') || '';
          if (!contentType.includes('application/json')) {
            res = await fetch(withDirectApiBase(`/api/detect/status/${jobId}`));
          }
          if (!res.ok) {
            if (res.status === 404) {
              clearInterval(timer);
              setIsProcessing(false);
              setLoadingProgress(0);
            }
            return;
          }
          const data = await readJsonOrThrow<any>(res);
          if (data.success && data.status === 'done' && data.result) {
            clearInterval(timer);
            setResult(data.result);
            setJobId(null);
            setTimestamp(Date.now());
            setIsProcessing(false);
            setLoadingProgress(100);
            if (inferVolume(data.result, selectedFolderSeries)) {
              setViewMode('original');
              setViewerMode('grid');
              setGridLayout({ rows: 2, cols: 2 });
              setTempGridLayout({ rows: 2, cols: 2 });
            } else {
              setViewMode('overlay');
            }
          } else if (data.status === 'failed') {
            clearInterval(timer);
            setJobId(null);
            setIsProcessing(false);
            alert('Analysis failed: ' + (data.error || 'unknown'));
          }
        } catch (e) {
          console.error('Polling error:', e);
        }
      }, 1500);
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [jobId, result, selectedFolderSeries]);
  // Interaction State
  const dragRef = useRef<{ active: boolean; mode: 'pan' | 'wl'; startX: number; startY: number }>({
    active: false, mode: 'pan', startX: 0, startY: 0
  });
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const rasterOverlaySvgRef = useRef<SVGSVGElement | null>(null);
  const magnifierCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const edgeMapRef = useRef<Float32Array | null>(null);
  const [imgRect, setImgRect] = useState<DOMRect | null>(null);
  const [fitScale, setFitScale] = useState(1);
  const [viewerHeight, setViewerHeight] = useState(containerHeight);
  const [displaySize, setDisplaySize] = useState({ width: 1000, height: 600 });


  // Drawing State
  const [shapes, setShapes] = useState<any[]>([]);
  const [pendingPoints, setPendingPoints] = useState<{ x: number; y: number }[]>([]);
  const [tempPoint, setTempPoint] = useState<{ x: number; y: number } | null>(null);
  const [contextMenu, setContextMenu] = useState<{ show: boolean; x: number; y: number; menu?: 'measure' | 'annotate' | 'grid' }>({ show: false, x: 0, y: 0, menu: undefined });

  const [zoom, setZoom] = useState(1);
  const [dimensions, setDimensions] = useState({ width: 1000, height: 600 });
  const containerRef = useRef<HTMLDivElement>(null);
  const [measurements, setMeasurements] = useState<any[]>([]);
  const [debugEvents, setDebugEvents] = useState<string[]>([]);
  const autoConfiguredCtRef = useRef(false);
  const autoAnalyzeTriggeredRef = useRef(false);

  // -- Missing Vars defined here --
  const today = new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  const handleZoom = (delta: number) => setZoom(z => Math.max(0.1, Math.min(5, z + delta)));
  const effectiveScale = Math.max(fitScale * scale * zoom, 0.001);
  // Alias for drawing:
  const currentMeasurement = pendingPoints.length > 0 ? { start: pendingPoints[0], current: tempPoint || pendingPoints[pendingPoints.length - 1] } : null;


  const originalFile = locationState.originalFile as File | undefined;
  const isDicomPath = (url?: string) => !!url && /\.(dcm|dicom)(?:$|[?#])/i.test(url);
  const originalIsDicom = Boolean(
    (originalFolderMode && selectedFolderSeries) ||
    locationState.originalIsDicom ||
    (originalFile && isDicomPath(originalFile.name)) ||
    isDicomPath(result?.image_url)
  );
  useEffect(() => {
    if (autoAnalyzeTriggeredRef.current) return;
    if (result || jobId) return;

    const autoAnalyzeFile =
      originalFolderMode
        ? (selectedFolderSeries?.files?.[0] || null)
        : ((locationState.originalFile as File | undefined) || null);

    if (!autoAnalyzeFile) return;

    autoAnalyzeTriggeredRef.current = true;
    setIsProcessing(true);
    setLoadingProgress(8);

    void requestAsyncDetection(autoAnalyzeFile, [])
      .then((state) => {
        setJobId(state.jobId);
        setLoadingProgress(12);
      })
      .catch((error) => {
        console.error('Auto analysis request failed', error);
        setIsProcessing(false);
        setLoadingProgress(0);
        autoAnalyzeTriggeredRef.current = false;
      });
  }, [jobId, locationState.originalFile, locationState.originalIsDicom, originalFolderMode, result, selectedFolderSeries]);
  const activeCaseName = (() => {
    if (selectedFolderSeries?.label) {
      return selectedFolderSeries.label;
    }
    const rawSource =
      String(result?.image_url || locationState?.previewUrl || locationState?.imageUrl || locationState?.jobId || 'Current Analysis');
    try {
      const normalized = rawSource.split('?')[0].replace(/\\/g, '/');
      const fileName = normalized.split('/').filter(Boolean).pop();
      return decodeURIComponent(fileName || rawSource);
    } catch {
      return rawSource;
    }
  })();
  const dicomHudFile =
    originalFolderMode && selectedFolderSeries?.files?.length
      ? selectedFolderSeries.files[0]
      : originalFile && originalIsDicom
        ? originalFile
        : null;

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

  useEffect(() => {
    let cancelled = false;

    if (!originalIsDicom || !dicomHudFile) {
      setDicomHudMetadata(null);
      setDicomPreviewDataUrl(null);
      setDicomAutoWindow(null);
      return;
    }

    const loadHudMetadata = async () => {
      try {
        const [arrayBuffer, inspection] = await Promise.all([
          dicomHudFile.arrayBuffer(),
          inspectLocalDicomFile(dicomHudFile),
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
          setDicomPreviewDataUrl(inspection.previewDataUrl);
          setDicomAutoWindow(nextAutoWindow);
        }
      } catch (error) {
        console.warn('Failed to parse DICOM HUD metadata', error);
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
  }, [dicomHudFile, originalIsDicom]);

  useEffect(() => {
    if (!isVolumeCase) {
      autoConfiguredCtRef.current = false;
      return;
    }

    if (autoConfiguredCtRef.current) return;

    setViewMode('original');
    setViewerMode('grid');
    setGridLayout({ rows: 2, cols: 2 });
    setTempGridLayout({ rows: 2, cols: 2 });
    autoConfiguredCtRef.current = true;
  }, [isVolumeCase]);

  const handleStartReport = async () => {
    if (reportStartState === 'creating') return;

    if (reportSessionId) {
      setReportDrawerOpen((prev) => !prev);
      setReportError(null);
      return;
    }

    if (!result) {
      setReportError('Analysis result is not ready yet.');
      return;
    }

    setReportStartState('creating');
    setReportError(null);
    try {
      const patientName = [locationState.userName, result?.patient_name, dicomHudMetadata?.patientName]
        .map((value) => String(value || '').trim())
        .find((value) => value && value.toLowerCase() !== 'patient') || 'Patient';
      const response = await createWebReportFromChart({
        result,
        source_url: result?.image_url,
        overlay_url: result?.overlay_url,
        language: 'English',
        patient_name: patientName,
      });
      setReportSessionId(response.session_id);
      setReportDrawerOpen(true);
    } catch (error: any) {
      console.error(error);
      setReportError(error?.message || 'Failed to start report workspace');
    } finally {
      setReportStartState('idle');
    }
  };

  // --- Data Mapping Logic ---
  const det = result?.det_counts || {};
  const counts = result
    ? [
      { label: 'Teeth', value: det.seg_teeth ?? 0 },
      { label: 'Caries', value: det.caries ?? 0 },
      { label: 'Periapical', value: det.periapical ?? 0 },
      { label: 'CEJ masks', value: det.cej_masks ?? 0 },
      { label: 'Bone level masks', value: det.bonelevel_masks ?? 0 },
    ]
    : mockCounts;

  const pblEntries = result?.pbl
    ? Object.entries(result.pbl).map(([tooth, val]) => `${tooth}: ${Number(val).toFixed(1)}%`)
    : [];

  const notifyCapture = (type: 'success' | 'error', text: string) => {
    setCaptureNotice({ type, text });
    window.setTimeout(() => {
      setCaptureNotice((prev) => (prev?.text === text ? null : prev));
    }, 2200);
  };

  const rememberCapturePreview = (sourceCanvas: HTMLCanvasElement, label: string, capturedAt: Date = new Date()) => {
    try {
      const previewCanvas = document.createElement('canvas');
      const maxWidth = 80;
      const maxHeight = 40;
      const scaleFactor = Math.min(maxWidth / sourceCanvas.width, maxHeight / sourceCanvas.height, 1);
      previewCanvas.width = Math.max(1, Math.round(sourceCanvas.width * scaleFactor));
      previewCanvas.height = Math.max(1, Math.round(sourceCanvas.height * scaleFactor));
      const previewCtx = previewCanvas.getContext('2d');
      if (!previewCtx) return;
      previewCtx.drawImage(sourceCanvas, 0, 0, previewCanvas.width, previewCanvas.height);
      const preview = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        label,
        dataUrl: previewCanvas.toDataURL('image/png'),
        reportDataUrl: sourceCanvas.toDataURL('image/png'),
        size: `${sourceCanvas.width}x${sourceCanvas.height}`,
        createdAt: formatCaptureTimeline(capturedAt),
      };
      setCaptureGallery((prev) => [preview, ...prev].slice(0, 12));
    } catch (error) {
      console.error('Failed to store capture preview', error);
    }
  };

  const moveCapturePreview = (fromId: string, toId: string) => {
    if (!fromId || !toId || fromId === toId) return;
    setCaptureGallery((prev) => {
      const fromIndex = prev.findIndex((item) => item.id === fromId);
      const toIndex = prev.findIndex((item) => item.id === toId);
      if (fromIndex < 0 || toIndex < 0 || fromIndex === toIndex) return prev;
      const next = [...prev];
      const [moved] = next.splice(fromIndex, 1);
      next.splice(toIndex, 0, moved);
      return next;
    });
  };

  const assignCaptureToViewport = (viewportId: string, captureId: string) => {
    const capture = captureGallery.find((item) => item.id === captureId);
    if (!capture) return;
    setAssignedCaptureSlots((prev) => ({
      ...prev,
      [viewportId]: {
        id: capture.id,
        label: capture.label,
        dataUrl: capture.dataUrl,
        createdAt: capture.createdAt,
      },
    }));
  };

  const clearAssignedCapture = (viewportId: string) => {
    setAssignedCaptureSlots((prev) => {
      if (!prev[viewportId]) return prev;
      const next = { ...prev };
      delete next[viewportId];
      return next;
    });
  };

  const getPrimaryCanvas = (root: ParentNode | null | undefined) => {
    const canvases = Array.from(root?.querySelectorAll?.('canvas') || []) as HTMLCanvasElement[];
    if (canvases.length === 0) return null;
    return canvases
      .filter((canvas) => canvas.width > 0 && canvas.height > 0)
      .sort((a, b) => {
        const aArea = Math.max(a.clientWidth, a.width) * Math.max(a.clientHeight, a.height);
        const bArea = Math.max(b.clientWidth, b.width) * Math.max(b.clientHeight, b.height);
        return bArea - aArea;
      })[0] || canvases[0];
  };

  const buildGridCompositeCanvas = () => {
    const cells = Array.from(
      viewerRef.current?.querySelectorAll?.('[data-grid-capture-cell="true"]') || []
    ) as HTMLElement[];
    if (cells.length === 0) return null;

    const cellCanvases = cells
      .map((cell) => getPrimaryCanvas(cell))
      .filter((canvas): canvas is HTMLCanvasElement => !!canvas);

    if (cellCanvases.length === 0) return null;

    const cellWidth = Math.max(...cellCanvases.map((canvas) => canvas.width));
    const cellHeight = Math.max(...cellCanvases.map((canvas) => canvas.height));
    const composite = document.createElement('canvas');
    composite.width = cellWidth * gridLayout.cols;
    composite.height = cellHeight * gridLayout.rows;
    const ctx = composite.getContext('2d');
    if (!ctx) return null;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, composite.width, composite.height);

    cellCanvases.forEach((canvas, idx) => {
      const row = Math.floor(idx / gridLayout.cols);
      const col = idx % gridLayout.cols;
      ctx.drawImage(canvas, col * cellWidth, row * cellHeight, cellWidth, cellHeight);
    });

    return composite;
  };

  const loadImageFromUrl = (src: string) =>
    new Promise<HTMLImageElement>((resolve, reject) => {
      const nextImage = new Image();
      nextImage.decoding = 'async';
      nextImage.onload = () => resolve(nextImage);
      nextImage.onerror = () => reject(new Error('Failed to load capture overlay image'));
      nextImage.src = src;
    });

  const buildRasterCaptureCanvas = async () => {
    const imageEl = imageRef.current;
    if (!imageEl?.naturalWidth || !imageEl?.naturalHeight) return null;

    const sourceWidth = imageEl.naturalWidth;
    const sourceHeight = imageEl.naturalHeight;
    const normalizedRotation = ((rotation % 360) + 360) % 360;
    const radians = (normalizedRotation * Math.PI) / 180;
    const absCos = Math.abs(Math.cos(radians));
    const absSin = Math.abs(Math.sin(radians));
    const exportWidth = Math.max(1, Math.round(sourceWidth * absCos + sourceHeight * absSin));
    const exportHeight = Math.max(1, Math.round(sourceWidth * absSin + sourceHeight * absCos));

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = exportWidth;
    exportCanvas.height = exportHeight;
    const exportCtx = exportCanvas.getContext('2d');
    if (!exportCtx) return null;

    exportCtx.imageSmoothingEnabled = true;
    exportCtx.imageSmoothingQuality = 'high';

    const drawLayer = (layer: CanvasImageSource, options?: { filter?: string }) => {
      exportCtx.save();
      exportCtx.translate(exportWidth / 2, exportHeight / 2);
      if (normalizedRotation) exportCtx.rotate(radians);
      if (flipped) exportCtx.scale(-1, 1);
      if (options?.filter) exportCtx.filter = options.filter;
      exportCtx.drawImage(layer, -sourceWidth / 2, -sourceHeight / 2, sourceWidth, sourceHeight);
      exportCtx.restore();
    };

    drawLayer(imageEl, {
      filter: `invert(${inverted ? 1 : 0}) brightness(${brightness}%) contrast(${contrast}%)`,
    });

    const overlaySvg = rasterOverlaySvgRef.current;
    if (!overlaySvg) return exportCanvas;

    const svgClone = overlaySvg.cloneNode(true) as SVGSVGElement;
    svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svgClone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
    svgClone.setAttribute('width', String(sourceWidth));
    svgClone.setAttribute('height', String(sourceHeight));
    svgClone.setAttribute('viewBox', `0 0 ${sourceWidth} ${sourceHeight}`);
    svgClone.style.width = `${sourceWidth}px`;
    svgClone.style.height = `${sourceHeight}px`;
    svgClone.style.pointerEvents = 'none';

    const serializedSvg = new XMLSerializer().serializeToString(svgClone);
    const svgBlob = new Blob(
      [`<?xml version="1.0" encoding="UTF-8"?>${serializedSvg}`],
      { type: 'image/svg+xml;charset=utf-8' }
    );
    const svgUrl = URL.createObjectURL(svgBlob);

    try {
      const overlayImage = await loadImageFromUrl(svgUrl);
      drawLayer(overlayImage);
    } finally {
      URL.revokeObjectURL(svgUrl);
    }

    return exportCanvas;
  };

  const downloadCanvasCapture = (sourceCanvas: HTMLCanvasElement, filePrefix = 'dental_capture') => {
    const capturedAt = new Date();
    const timestamp = formatCaptureFileTimestamp(capturedAt);
    const fileName = `${filePrefix}_${timestamp}.png`;
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = sourceCanvas.width;
    exportCanvas.height = sourceCanvas.height;
    const exportCtx = exportCanvas.getContext('2d');
    if (!exportCtx) {
      notifyCapture('error', 'Capture failed: export context unavailable');
      return;
    }
    exportCtx.drawImage(sourceCanvas, 0, 0);
    exportCanvas.toBlob((blob) => {
      if (!blob) {
        notifyCapture('error', 'Capture failed: empty image');
        return;
      }
      const link = document.createElement('a');
      const objectUrl = URL.createObjectURL(blob);
      link.download = fileName;
      link.href = objectUrl;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
      rememberCapturePreview(exportCanvas, fileName, capturedAt);
      notifyCapture('success', `Saved ${fileName}`);
    }, 'image/png');
  };

  const copyCanvasCapture = async (sourceCanvas: HTMLCanvasElement, successText = 'Copied capture to clipboard') => {
    try {
      const capturedAt = new Date();
      if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
        notifyCapture('error', 'Clipboard copy is not supported in this browser');
        return false;
      }

      const blob = await new Promise<Blob | null>((resolve) => sourceCanvas.toBlob(resolve, 'image/png'));
      if (!blob) {
        notifyCapture('error', 'Capture failed: empty image');
        return false;
      }

      await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
      rememberCapturePreview(sourceCanvas, successText, capturedAt);
      notifyCapture('success', successText);
      return true;
    } catch (error) {
      notifyCapture('error', 'Clipboard copy failed');
      return false;
    }
  };

  const handleCapture = async (
    cropRegion?: { x: number; y: number; w: number; h: number },
    options?: { destination?: 'download' | 'clipboard' }
  ) => {
    try {
      let sourceCanvas: HTMLCanvasElement | null = null;
      if (shouldUseCornerstone) {
        sourceCanvas = viewerMode === 'grid' && !cropRegion
          ? buildGridCompositeCanvas()
          : getPrimaryCanvas(containerRef.current || viewerRef.current);
      } else if (imageRef.current) {
        sourceCanvas = await buildRasterCaptureCanvas();
      }

      if (sourceCanvas) {
        let finalCanvas = sourceCanvas;
        if (cropRegion && (cropRegion.w !== 0 && cropRegion.h !== 0)) {
          const tempCanvas = document.createElement('canvas');
          let sx = Math.max(0, cropRegion.w > 0 ? cropRegion.x : cropRegion.x + cropRegion.w);
          let sy = Math.max(0, cropRegion.h > 0 ? cropRegion.y : cropRegion.y + cropRegion.h);
          let sw = Math.abs(cropRegion.w);
          let sh = Math.abs(cropRegion.h);

          if (shouldUseCornerstone && viewerRef.current) {
            const viewerRect = viewerRef.current.getBoundingClientRect();
            const canvasRect = sourceCanvas.getBoundingClientRect();
            const cropLeft = viewerRect.left + sx;
            const cropTop = viewerRect.top + sy;
            const cropRight = cropLeft + sw;
            const cropBottom = cropTop + sh;

            const intersectLeft = Math.max(cropLeft, canvasRect.left);
            const intersectTop = Math.max(cropTop, canvasRect.top);
            const intersectRight = Math.min(cropRight, canvasRect.right);
            const intersectBottom = Math.min(cropBottom, canvasRect.bottom);

            sw = Math.max(0, intersectRight - intersectLeft);
            sh = Math.max(0, intersectBottom - intersectTop);
            if (sw <= 1 || sh <= 1) {
              notifyCapture('error', 'Capture failed: select an area inside the viewport');
              return;
            }

            const scaleX = sourceCanvas.width / Math.max(1, canvasRect.width);
            const scaleY = sourceCanvas.height / Math.max(1, canvasRect.height);
            sx = Math.max(0, (intersectLeft - canvasRect.left) * scaleX);
            sy = Math.max(0, (intersectTop - canvasRect.top) * scaleY);
            sw *= scaleX;
            sh *= scaleY;
          } else if (viewerRef.current && imageRef.current) {
            const viewerRect = viewerRef.current.getBoundingClientRect();
            const imageRect = imageRef.current.getBoundingClientRect();
            const cropLeft = viewerRect.left + sx;
            const cropTop = viewerRect.top + sy;
            const cropRight = cropLeft + sw;
            const cropBottom = cropTop + sh;

            const intersectLeft = Math.max(cropLeft, imageRect.left);
            const intersectTop = Math.max(cropTop, imageRect.top);
            const intersectRight = Math.min(cropRight, imageRect.right);
            const intersectBottom = Math.min(cropBottom, imageRect.bottom);

            sw = Math.max(0, intersectRight - intersectLeft);
            sh = Math.max(0, intersectBottom - intersectTop);
            if (sw <= 1 || sh <= 1) {
              notifyCapture('error', 'Capture failed: select an area inside the image');
              return;
            }

            const scaleX = sourceCanvas.width / Math.max(1, imageRect.width);
            const scaleY = sourceCanvas.height / Math.max(1, imageRect.height);
            sx = Math.max(0, (intersectLeft - imageRect.left) * scaleX);
            sy = Math.max(0, (intersectTop - imageRect.top) * scaleY);
            sw *= scaleX;
            sh *= scaleY;
          } else {
            const vW = viewerRef.current?.clientWidth || 1;
            const vH = viewerRef.current?.clientHeight || 1;
            const scX = sourceCanvas.width / vW;
            const scY = sourceCanvas.height / vH;
            sx *= scX;
            sy *= scY;
            sw *= scX;
            sh *= scY;
          }

          tempCanvas.width = Math.max(1, Math.round(sw));
          tempCanvas.height = Math.max(1, Math.round(sh));
          const tempCtx = tempCanvas.getContext('2d');
          if (tempCtx) {
            tempCtx.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, tempCanvas.width, tempCanvas.height);
            finalCanvas = tempCanvas;
          }
        }
        const shouldCopyToClipboard = options?.destination === 'clipboard';
        if (shouldCopyToClipboard) {
          await copyCanvasCapture(finalCanvas, 'Copied area capture to clipboard');
        } else {
          downloadCanvasCapture(finalCanvas);
        }
        if (cropRegion) setCaptureRect(null);
      } else {
        notifyCapture('error', 'Capture failed: no render canvas found');
      }
    } catch (err) { notifyCapture('error', 'Capture failed'); }
  };

  const handleGridViewportCapture = async (sourceCanvas: HTMLCanvasElement, viewportLabel?: string) => {
    try {
      await copyCanvasCapture(
        sourceCanvas,
        viewportLabel ? `Copied ${viewportLabel} viewport to clipboard` : 'Copied viewport to clipboard'
      );
      handleToolChange('pointer');
    } catch (error) {
      notifyCapture('error', 'Capture failed');
    }
  };

  const findings = result
    ? [
      ...Object.entries(result.caries_by_tooth_best || {}).map(([tooth, data]: any) => ({
        label: 'Caries',
        value: `${tooth} (${(data.conf ?? 0).toFixed(2)})`,
        color: 'bg-red-500 text-white',
      })),
      ...Object.entries(result.periapical_by_tooth_best || {}).map(([tooth, data]: any) => ({
        label: 'Periapical',
        value: `${tooth} (${(data.conf ?? 0).toFixed(2)})`,
        color: 'bg-orange-500 text-white',
      })),
      ...Object.entries(result.filling_by_tooth_best || {}).map(([tooth, data]: any) => ({
        label: 'Filling',
        value: `${tooth} (${(data.conf ?? 0).toFixed(2)})`,
        color: 'bg-blue-500 text-white',
      })),
      ...Object.entries(result.implant_by_tooth_best || {}).map(([tooth, data]: any) => ({
        label: 'Implant',
        value: `${tooth} (${(data.conf ?? 0).toFixed(2)})`,
        color: 'bg-purple-600 text-white',
      })),
      ...Object.entries(result.crown_by_tooth_best || {}).map(([tooth, data]: any) => ({
        label: 'Crown',
        value: `${tooth} (${(data.conf ?? 0).toFixed(2)})`,
        color: 'bg-yellow-400 text-black',
      })),
    ]
    : mockFindings;

  // --- Tooth Status Logic ---
  // --- Tooth Status Logic (Synced with report_v2) ---
  const statuses: Record<string, any> = {};


  // Helper to extract tooth ID safely
  const getToothId = (t: any): string | null => {
    if (!t) return null;
    if (typeof t === 'object') {
      return String(t.assigned_tooth || t.tooth_label || t.tooth || t.fdi || t.label || '');
    }
    return String(t);
  };

  const toothRecords = (() => {
    const merged: Record<string, any> = {};
    const sources = [
      Array.isArray(result?.data) ? result.data : [],
      Array.isArray(result?.teeth) ? result.teeth : Object.values(result?.teeth || {}),
      Array.isArray(result?.missing_teeth) ? result.missing_teeth : [],
      Array.isArray(result?.implant_site_candidates) ? result.implant_site_candidates : [],
      Array.isArray(result?.implant_candidates) ? result.implant_candidates : [],
      Array.isArray(result?.analysis_result?.data) ? result.analysis_result.data : [],
      Array.isArray(result?.analysis_result?.teeth) ? result.analysis_result.teeth : Object.values(result?.analysis_result?.teeth || {}),
      Array.isArray(result?.analysis_result?.missing_teeth) ? result.analysis_result.missing_teeth : [],
      Array.isArray(result?.analysis_result?.implant_site_candidates) ? result.analysis_result.implant_site_candidates : [],
      Array.isArray(result?.analysis_result?.implant_candidates) ? result.analysis_result.implant_candidates : [],
    ];

    sources.forEach((items: any[]) => {
      items.forEach((item: any) => {
        const key = getToothId(item);
        if (key && key !== 'undefined') {
          merged[key] = { ...(merged[key] || {}), ...item };
        }
      });
    });

    return merged;
  })();

  // 1. Missing Teeth
  const missingList = result?.missing_teeth || result?.teeth_missing || [];
  missingList.forEach((t: any) => {
    const k = getToothId(t);
    if (k && k !== 'undefined') statuses[k] = { ...statuses[k], missing: true };
  });

  // 2. Caries
  const cariesList = result?.caries || [];
  cariesList.forEach((t: any) => {
    const k = getToothId(t);
    if (k && k !== 'undefined') statuses[k] = { ...statuses[k], caries: true };
  });

  // 3. Periapical
  const periList = result?.periapical || [];
  periList.forEach((t: any) => {
    const k = getToothId(t);
    if (k && k !== 'undefined') statuses[k] = { ...statuses[k], peri: true };
  });
  if (result?.periapical_by_tooth) {
    Object.keys(result.periapical_by_tooth).forEach(k => {
      statuses[k] = { ...statuses[k], peri: true };
    });
  }

  // 4. Implants (Dict keys)
  if (result?.implant_by_tooth) {
    Object.keys(result.implant_by_tooth).forEach(k => {
      statuses[k] = { ...statuses[k], implant: true };
    });
  }

  // 5. Crowns (Dict keys)
  if (result?.crown_by_tooth) {
    Object.keys(result.crown_by_tooth).forEach(k => {
      statuses[k] = { ...statuses[k], crown: true };
    });
  }

  // 6. Fillings (Dict keys)
  if (result?.filling_by_tooth) {
    Object.keys(result.filling_by_tooth).forEach(k => {
      statuses[k] = { ...statuses[k], filling: true };
    });
  }

  Object.entries(toothRecords).forEach(([key, tooth]: any) => {
    const implantType = String(tooth?.type || '').toLowerCase();
    statuses[key] = {
      ...(statuses[key] || {}),
      caries: !!(statuses[key]?.caries || tooth?.caries),
      peri: !!(statuses[key]?.peri || tooth?.periodontitis || tooth?.periapical),
      implant: !!(statuses[key]?.implant || tooth?.implant || implantType.includes('implant') || implantType.includes('fixture')),
      missing: !!(statuses[key]?.missing || tooth?.missing),
      crown: !!(statuses[key]?.crown || tooth?.crown),
      filling: !!(statuses[key]?.filling || tooth?.filling),
      bone_loss_level: Number(tooth?.bone_loss_level ?? result?.pbl_level?.[key] ?? 0),
      bone_loss_pct: Number(tooth?.bone_loss_pct ?? result?.pbl?.[key] ?? 0),
      hopeless: !!tooth?.hopeless,
      nerve_overlap: !!tooth?.nerve_overlap,
      sinus_overlap: !!tooth?.sinus_overlap,
      nerve_dist_mm: Number(tooth?.nerve_dist_mm ?? 0),
    };
  });

  Object.keys(statuses).forEach((key) => {
    const st = statuses[key] || {};
    const severeBl = Number(st.bone_loss_level || 0) >= 3;
    const hopeless = !!st.hopeless || Number(st.bone_loss_level || 0) >= 4 || Number(st.bone_loss_pct || 0) >= 60;
    const hasCaries = !!st.caries;
    const hasPerio = !!st.peri;
    const nerveOverlap = !!st.nerve_overlap;
    const sinusOverlap = !!st.sinus_overlap;
    const nerveDistMm = Number(st.nerve_dist_mm || 0);

    let triage: 'triage-1' | 'triage-2' | 'triage-3' | 'missing' | 'implant' = 'triage-3';
    if (st.missing) triage = 'missing';
    else if (st.implant) triage = 'implant';
    else if (hopeless || nerveOverlap || (sinusOverlap && (hasCaries || hasPerio || severeBl)) || (nerveDistMm > 0 && nerveDistMm < 2.0)) triage = 'triage-1';
    else if (severeBl || hasPerio || hasCaries) triage = 'triage-2';

    statuses[key] = { ...st, triage };
  });

  const getLegendCategory = (toothKey: string) => {
    const st = statuses[toothKey] || {};
    if (st.missing) return 'missing';
    if (st.implant) return 'implant';
    return st.triage || 'triage-3';
  };

  const legendCounts = ODONTOGRAM_TEETH.reduce((acc, tooth) => {
    const category = getLegendCategory(String(tooth));
    acc[category] = (acc[category] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const dentalLegendItems = [
    { key: 'triage-3', label: 'Routine Checkup (Triage 3)', color: '#16A34A', dashed: false, gap: 18, count: legendCounts['triage-3'] || 0 },
    { key: 'triage-2', label: 'Treatment Required (Triage 2)', color: '#D4A106', dashed: false, gap: 24, count: legendCounts['triage-2'] || 0 },
    { key: 'triage-1', label: 'Urgent Priority (Triage 1)', color: '#DC2626', dashed: false, gap: 18, count: legendCounts['triage-1'] || 0 },
    { key: 'implant', label: 'Implant', color: '#2563EB', dashed: false, gap: 18, count: legendCounts['implant'] || 0 },
    { key: 'missing', label: 'Missing Tooth', color: '#94A3B8', dashed: true, gap: 0, count: legendCounts['missing'] || 0 },
  ].filter((item) => item.count > 0);

  const baseStatuses = (() => {
    const map: Record<string, string> = {};
    const teethSeq = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28, 48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38];
    teethSeq.forEach(t => map[String(t)] = 'healthy');

    const mark = (arr: any[], label: string) => {
      (arr || []).forEach((v: any) => {
        const k = getToothId(v);
        if (k && k !== 'undefined') map[k] = label;
      });
    };
    // missing
    mark(result?.missing_teeth || result?.teeth_missing || [], 'missing');
    // implant
    Object.keys(result?.implant_by_tooth || {}).forEach(k => { map[k] = 'implant'; });
    Object.keys(result?.implant_by_tooth_best || {}).forEach(k => { map[k] = 'implant'; });
    // requires treatment (caries/periapical)
    mark(result?.caries || [], 'requires');
    Object.keys(result?.caries_by_tooth || {}).forEach(k => { map[k] = 'requires'; });
    Object.keys(result?.caries_by_tooth_best || {}).forEach(k => { map[k] = 'requires'; });
    mark(result?.periapical || [], 'requires');
    Object.keys(result?.periapical_by_tooth || {}).forEach(k => { map[k] = 'requires'; });
    Object.keys(result?.periapical_by_tooth_best || {}).forEach(k => { map[k] = 'requires'; });

    return map;
  })();

  const extractionCandidates = result?.extraction_candidates || [];
  const implantSiteCandidates = result?.implant_site_candidates || result?.implant_candidates || [];
  const perioHighlights = result?.periodontitis_highlight_teeth || [];
  const chartHighlights = [
    ...perioHighlights,
    ...implantSiteCandidates.map((c: any) => c?.site_fdi || c?.tooth || c),
  ];
  const implantMetricsByTooth = (() => {
    const map: Record<string, any> = {};
    const metrics = Array.isArray(result?.implant_metrics)
      ? result.implant_metrics
      : Object.values(result?.implant_metrics || {});
    metrics.forEach((item: any) => {
      const key = getToothId(item?.label || item?.tooth_label || item?.tooth || item?.id);
      if (key && key !== 'undefined') {
        map[key] = item;
      }
    });
    return map;
  })();
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
  const getBestConfidence = (bestMap: any, rawMap: any, key: string) => {
    const item = bestMap?.[key] || rawMap?.[key];
    const conf = Number(item?.conf);
    return Number.isFinite(conf) ? conf : null;
  };
  const odontogramTooltipData = Object.keys(statuses).reduce((acc, key) => {
    const tooth = toothRecords[key] || {};
    const status = statuses[key] || {};
    const implantMetric = implantMetricsByTooth[key] || {};
    const implantGuide = tooth?.implant_guide || {};
    const mmPerPx = Number(result?.mm_per_px || 0.1);
    const contourMetrics = estimateImplantMetricsFromContour(tooth?.contour, mmPerPx);
    const diameterMm = Number(
      implantMetric?.diameter_mm ??
      implantMetric?.diameter ??
      tooth?.implant_meta?.diameter ??
      contourMetrics.diameterMm ??
      0
    );
    const lengthMm = Number(
      implantMetric?.length_mm ??
      implantMetric?.length ??
      tooth?.implant_meta?.length ??
      contourMetrics.lengthMm ??
      0
    );
    const gapMm = Number(
      tooth?.mesiodistal_gap_mm ??
      implantGuide?.mesiodistal_gap_mm ??
      implantMetric?.mesiodistal_gap_mm ??
      0
    );
    const isUpperTooth = key.startsWith('1') || key.startsWith('2');
    const centerToNerveMmRaw = Number(
      implantGuide?.dist_mm ??
      tooth?.center_to_nerve_dist_mm ??
      implantMetric?.dist_mm ??
      tooth?.nerve_dist_mm ??
      status?.nerve_dist_mm ??
      0
    );
    const centerToNerveMm = isUpperTooth ? 0 : centerToNerveMmRaw;
    const cariesProb = getBestConfidence(result?.caries_by_tooth_best, result?.caries_by_tooth, key);
    const periapicalProb = getBestConfidence(result?.periapical_by_tooth_best, result?.periapical_by_tooth, key);
    const isExistingImplant = Boolean(status?.implant && (diameterMm > 0 || lengthMm > 0));
    const isPlanningLike = Boolean(status?.missing || gapMm > 0 || centerToNerveMm > 0);
    let statusLabel = 'Healthy';
    if (status?.missing && !status?.implant) statusLabel = 'Missing';
    else if (status?.implant) statusLabel = 'Implant';
    else if (status?.caries || status?.peri || Number(status?.bone_loss_level || 0) >= 3) statusLabel = 'Requires Tx';

    acc[key] = {
      kind: isExistingImplant ? 'implant' : isPlanningLike ? 'planning' : 'finding',
      status: statusLabel,
      pblPct: Number.isFinite(Number(status?.bone_loss_pct)) ? Number(status?.bone_loss_pct) : null,
      level: status?.bone_loss_level ?? null,
      cariesProb,
      periapicalProb,
      diameterMm: diameterMm > 0 ? diameterMm : null,
      lengthMm: lengthMm > 0 ? lengthMm : null,
      gapMm: gapMm > 0 ? gapMm : null,
      centerToNerveMm: centerToNerveMm > 0 ? centerToNerveMm : null,
    };
    return acc;
  }, {} as Record<string, any>);
  // --- Image Source Logic ---
  const getUrlWithCacheBuster = (url?: string) => {
    if (!url) return undefined;
    if (url.startsWith('blob:') || url.startsWith('data:')) return url;
    return `${url}${url.includes('?') ? '&' : '?'}t=${timestamp}`;
  };

  const originalRasterUrl = getUrlWithCacheBuster(dicomPreviewDataUrl || locationState.previewUrl || result?.preview_url);
  const originalUrl = getUrlWithCacheBuster(result?.image_url);
  const overlayUrl = getUrlWithCacheBuster(result?.overlay_url);
  const heatmapUrl = getUrlWithCacheBuster(result?.heatmap_overlay_url);
  
  const hasHeatmapAsset = Boolean(result?.heatmap_overlay_url);
  const hasStructuredOverlayData = Boolean(
    (Array.isArray(result?.sinus_contours) && result.sinus_contours.length > 0) ||
    (Array.isArray(result?.nerve_contours) && result.nerve_contours.length > 0) ||
    (Array.isArray(result?.teeth) && result.teeth.length > 0) ||
    (Array.isArray(result?.teeth_objects) && result.teeth_objects.length > 0)
  );
  const shouldUseStructuredAiOverlay = viewMode === 'overlay' && hasStructuredOverlayData;
  const overlayIsDicom = isDicomPath(result?.overlay_url);

  // If we are in 2D mode but it's a DICOM, we MUST use a raster preview for <img> tags.
  // Raw .dcm URLs will result in a black screen.
  const shouldUseCornerstone =
    viewMode === 'original'
      ? (originalIsDicom && isVolumeCase) || (originalFolderMode && Boolean(selectedFolderSeries))
      : viewMode === 'overlay'
        ? (overlayIsDicom && isVolumeCase)
        : false;

  const showSrc = shouldUseCornerstone
    ? null
    : (originalRasterUrl || originalUrl || result?.image_url);
    // Note: We always use the clear original raster even in heatmap/overlay mode 
    // to avoid server-side colored artifacts (like green sinus) from appearing in the background.

  // If no result yet but we have a preview, show that
  // if (!showSrc && locationState.previewUrl) {
  //   showSrc = locationState.previewUrl;
  // }


  const displayDicomHudMetadata = (() => {
    if (!dicomHudMetadata) return null;
    if (shouldUseCornerstone) return dicomHudMetadata;
    if (!originalIsDicom) return dicomHudMetadata;

    const nextWindow = deriveDisplayWindowFromControls(
      dicomHudMetadata.windowCenter,
      dicomHudMetadata.windowWidth,
      brightness,
      contrast
    );

    return {
      ...dicomHudMetadata,
      windowCenter: nextWindow.windowCenter,
      windowWidth: nextWindow.windowWidth,
    };
  })();
  const magnifierDisabled = shouldUseCornerstone && viewerMode === 'grid';
  const areaCaptureDisabled = false;
  const shouldShowFloatingDicomHud = !shouldUseCornerstone && Boolean(dicomHudMetadata);
  const shouldShowOverlayPresetSelector = viewMode === 'overlay' && hasStructuredOverlayData && !shouldUseCornerstone;
  const shouldCenterModeBadge = shouldShowFloatingDicomHud || shouldShowOverlayPresetSelector;
  const viewModeLabel =
    viewMode === 'overlay'
      ? 'AI Analysis Mode'
      : viewMode === 'heatmap'
        ? 'Risk Overlay Mode'
        : 'Original Source Mode';
  const viewModeDotClass =
    viewMode === 'overlay'
      ? 'bg-indigo-500 shadow-[0_0_10px_#6366f1] animate-pulse'
      : viewMode === 'heatmap'
        ? 'bg-orange-400 shadow-[0_0_12px_rgba(251,146,60,0.95)] animate-pulse'
        : 'bg-gray-500';
  const hasViewerShell = hasData || isProcessing || Boolean(showSrc) || shouldUseCornerstone;
  const cornerstoneSources = shouldUseCornerstone
    ? [
      viewMode === 'original'
        ? originalFolderMode && selectedFolderSeries
          ? {
            id: `chart-original-dicom-folder-${selectedFolderSeries.id}`,
            label: selectedFolderSeries.label,
            url: '',
            ...(selectedFolderSeries.volumeEligible
              ? {
                files: selectedFolderSeries.files,
                scheme: 'dicomfolder' as const,
              }
              : {
                file: selectedFolderSeries.files[0],
                scheme: 'dicomfile' as const,
              }),
          }
          : originalFile && originalIsDicom
          ? {
            id: 'chart-original-dicom-local',
            label: 'Original',
            url: locationState.previewUrl || '',
            file: originalFile,
            scheme: 'dicomfile' as const,
          }
          : {
            id: 'chart-original-dicom-remote',
            label: 'Original',
            url: result?.image_url || '',
            scheme: 'wadouri' as const,
          }
        : (originalFile && originalIsDicom && (result?.overlay_url === result?.image_url || !result?.overlay_url))
          ? {
            id: 'chart-overlay-dicom-local',
            label: 'Overlay',
            url: locationState.previewUrl || '',
            file: originalFile,
            scheme: 'dicomfile' as const,
          }
          : {
            id: 'chart-overlay-dicom',
            label: 'Overlay',
            url: result?.overlay_url || '',
            scheme: 'wadouri' as const,
          },
    ]
    : [];

  useEffect(() => {
    if (magnifierDisabled && activeTool === 'magnifier') {
      handleToolChange('pointer');
    }
  }, [magnifierDisabled, activeTool]);

  useEffect(() => {
    if (areaCaptureDisabled && activeTool === 'capture-area') {
      handleToolChange('pointer');
      setCaptureRect(null);
    }
  }, [areaCaptureDisabled, activeTool]);

  const calculateFitScale = (
    viewerWidth: number,
    viewerHeight: number,
    imageWidth: number,
    imageHeight: number
  ) => {
    if (!viewerWidth || !viewerHeight || !imageWidth || !imageHeight) return 1;

    const widthFit = viewerWidth / imageWidth;
    const heightFit = viewerHeight / imageHeight;

    return Math.min(widthFit, heightFit) * 0.995;
  };

  const fitImageToViewer = () => {
    if (!viewerRef.current || !imageRef.current) return;
    const img = imageRef.current;
    const viewerWidth = viewerRef.current.clientWidth;
    if (!img.naturalWidth || !img.naturalHeight || !viewerWidth) return;

    const nextViewerHeight = Math.max(
      320,
      Math.min(containerHeight, Math.round((viewerWidth / img.naturalWidth) * img.naturalHeight))
    );

    const s = calculateFitScale(
      viewerWidth,
      nextViewerHeight,
      img.naturalWidth,
      img.naturalHeight
    );
    setViewerHeight(nextViewerHeight);
    setFitScale(s);
    setDisplaySize({
      width: img.naturalWidth * s,
      height: img.naturalHeight * s,
    });
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  // Fit on source change and on window resize
  useEffect(() => {
    fitImageToViewer();
  }, [showSrc, timestamp]);

  useEffect(() => {
    const onResize = () => fitImageToViewer();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const buildLowerContourRuns = (contour: any[], lowerRatio = 0.25) => {
    if (!Array.isArray(contour) || contour.length < 3) return [];

    const points = contour
      .map((pt: any) => Array.isArray(pt) && pt.length >= 2
        ? { x: Number(pt[0]), y: Number(pt[1]) }
        : null)
      .filter((pt): pt is { x: number; y: number } =>
        pt !== null && Number.isFinite(pt.x) && Number.isFinite(pt.y));

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

      if (currentRun.length >= 2) {
        runs.push(currentRun);
      }
      currentRun = [];
    });

    if (currentRun.length >= 2) {
      runs.push(currentRun);
    }

    if (runs.length > 1) {
      const first = runs[0];
      const last = runs[runs.length - 1];
      const firstPoint = first[0];
      const lastPoint = last[last.length - 1];
      if (firstPoint && lastPoint && points[0].y >= cutoffY && points[points.length - 1].y >= cutoffY) {
        runs[0] = [...last, ...first];
        runs.pop();
      }
    }

    return runs;
  };

  const warmPastelPalette = [
    { fill: 'rgba(255, 216, 194, 0.26)', stroke: 'rgba(255, 182, 151, 0.94)' },
    { fill: 'rgba(255, 226, 203, 0.26)', stroke: 'rgba(241, 174, 129, 0.94)' },
    { fill: 'rgba(255, 212, 209, 0.26)', stroke: 'rgba(234, 153, 150, 0.94)' },
    { fill: 'rgba(246, 224, 198, 0.26)', stroke: 'rgba(221, 171, 118, 0.94)' },
    { fill: 'rgba(255, 229, 214, 0.26)', stroke: 'rgba(232, 163, 121, 0.94)' },
    { fill: 'rgba(255, 239, 221, 0.26)', stroke: 'rgba(227, 180, 126, 0.94)' },
  ];

  const selectedToothKey = selectedTooth ? String(selectedTooth) : null;

  // --- AI Overlay Rendering ---
    const renderAIDetections = () => {
      if (viewMode !== 'overlay' || !result) return null;
  
      const items: React.ReactNode[] = [];
      const shouldFocusSingleTooth = Boolean(selectedToothKey);
      
      const shouldShowSinus = !shouldFocusSingleTooth && (overlayPreset === 'all' || overlayPreset === 'sinus' || overlayPreset === 'sinus-upper-tooth');
      const shouldShowNerve = !shouldFocusSingleTooth && (overlayPreset === 'all' || overlayPreset === 'nerve' || overlayPreset === 'nerve-lower-tooth');
      const shouldShowUpperTooth = overlayPreset === 'all' || overlayPreset === 'tooth' || overlayPreset === 'sinus-upper-tooth';
      const shouldShowLowerTooth = overlayPreset === 'all' || overlayPreset === 'tooth' || overlayPreset === 'nerve-lower-tooth';
  
      const getToothOverlayStyle = (label: string) => {
        const numericLabel = Number(label);
        const paletteIndex = Number.isFinite(numericLabel) ? Math.abs(numericLabel) % warmPastelPalette.length : 0;
        const palette = warmPastelPalette[paletteIndex];
        const isSelected = selectedToothKey === label;
        
        return {
          fill: isSelected ? palette.fill.replace('0.26', '0.45') : palette.fill,
          stroke: isSelected ? '#ffffff' : palette.stroke,
          strokeWidth: isSelected ? 2.5 : 1.2
        };
      };
  
      // 1. Sinus Rendering (Lower 25% only, sharp red contour)
      if (shouldShowSinus && result.sinus_contours) {
        result.sinus_contours.forEach((contour: any, idx: number) => {
          if (!Array.isArray(contour)) return;
          const runs = buildLowerContourRuns(contour, 0.25);
          runs.forEach((run, ridx) => {
            const points = run.map((pt) => `${pt.x},${pt.y}`).join(' ');
            items.push(
              <polyline
                key={`sinus-lower-${idx}-${ridx}`}
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
  
      // 2. Nerve Rendering (Minimalist, 0.3 opacity, thin)
      if (shouldShowNerve && result.nerve_contours) {
        result.nerve_contours.forEach((contour: any, idx: number) => {
          if (!Array.isArray(contour) || contour.length < 2) return;
          const points = contour.map((pt: any) => `${pt[0]},${pt[1]}`).join(' ');
          items.push(
            <polyline
              key={`nerve-line-${idx}`}
              points={points}
              fill="none"
              stroke="rgba(255, 0, 255, 0.3)"
              strokeWidth={1 / effectiveScale}
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          );
        });
      }
  
      // 3. Tooth Contours
      const toothList = (Array.isArray(result?.teeth) ? result.teeth : (Array.isArray(result?.teeth_objects) ? result.teeth_objects : []));
      toothList.forEach((tooth: any, idx: number) => {
        const label = String(tooth?.tooth_label || tooth?.label || tooth?.tooth || '');
        if (!label || (selectedToothKey && label !== selectedToothKey)) return;
        
        const isUpper = label.startsWith('1') || label.startsWith('2');
        if (isUpper && !shouldShowUpperTooth) return;
        if (!isUpper && !shouldShowLowerTooth) return;
  
        const contour = tooth?.contour;
        if (!Array.isArray(contour) || contour.length < 3) return;
        
        const style = getToothOverlayStyle(label);
        const points = contour.map((pt: any) => `${pt[0]},${pt[1]}`).join(' ');
        
        items.push(
          <g key={`tooth-overlay-${label}-${idx}`}>
            <polygon
              points={points}
              fill={style.fill}
              stroke={style.stroke}
              strokeWidth={style.strokeWidth / effectiveScale}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
            {/* Tooth Label Tag */}
            {!selectedToothKey && (
                <g transform={`translate(${contour[0][0]}, ${contour[0][1] - 15 / effectiveScale})`}>
                    <rect x={-10 / effectiveScale} y={-10 / effectiveScale} width={20 / effectiveScale} height={14 / effectiveScale} rx={4 / effectiveScale} fill="rgba(0,0,0,0.6)" />
                    <text textAnchor="middle" fill="#ffffff" fontSize={10 / effectiveScale} fontWeight="bold">{label}</text>
                </g>
            )}
          </g>
        );
      });
  
      // 4. Labeled Pathology Boxes (Caries / Periapical)
      const addPathology = (dataMap: any, color: string, labelPrefix: string) => {
        if (!dataMap) return;
        Object.entries(dataMap).forEach(([tooth, data]: any, idx) => {
          if (selectedToothKey && String(tooth) !== selectedToothKey) return;
          const box = data?.box;
          if (!Array.isArray(box) || box.length < 4) return;
          const [x1, y1, x2, y2] = box;
          const w = x2 - x1;
          const h = y2 - y1;
          
          const tagHeight = 14 / effectiveScale;
          const tagPadding = 6 / effectiveScale;
          const labelText = `${labelPrefix} ${tooth}`;
          const fontSize = 9 / effectiveScale;
          // Rough estimate of width based on text length
          const tagWidth = (labelText.length * (fontSize * 0.6)) + (tagPadding * 2);

          items.push(
            <g key={`patho-${labelPrefix}-${tooth}-${idx}`}>
              <rect
                x={x1} y={y1} width={w} height={h}
                fill="none" stroke={color}
                strokeWidth={0.8 / effectiveScale}
                strokeDasharray={`${3 / effectiveScale} ${2 / effectiveScale}`}
                vectorEffect="non-scaling-stroke"
              />
              <rect
                x={x1} y={y1 - tagHeight}
                width={tagWidth} height={tagHeight}
                rx={3 / effectiveScale} ry={3 / effectiveScale}
                fill={color}
                fillOpacity={0.85}
              />
              <text 
                x={x1 + tagPadding} y={y1 - (tagHeight / 2) + (fontSize / 3)} 
                fill="#fff" fontSize={fontSize} fontWeight="bold"
              >
                {labelText}
              </text>
            </g>
          );
        });
      };
  
      addPathology(result.caries_by_tooth_best, "#ff4444", "Caries");
      addPathology(result.periapical_by_tooth_best, "#ff9800", "Periapical");
  
      return <g id="ai-overlay-layer">{items}</g>;
    };
  
  const renderRiskDetections = (options?: { emphasize?: boolean }) => {
    if (viewMode !== 'heatmap' || !result) return null;

    const emphasize = Boolean(options?.emphasize);
    const opacityBoost = emphasize ? 1.18 : 1;
    const items: React.ReactNode[] = [];
    const cariesEntries = Object.entries(result.caries_by_tooth_best || {}).filter(
      ([tooth]) => !selectedToothKey || String(tooth) === selectedToothKey
    );
    const periapicalEntries = Object.entries(result.periapical_by_tooth_best || {}).filter(
      ([tooth]) => !selectedToothKey || String(tooth) === selectedToothKey
    );
    const teeth = (Array.isArray(result.teeth) ? result.teeth : []).filter(
      (tooth: any) => {
        const label = String(tooth?.tooth_label || '');
        if (!label) return false; // Important: Skip non-tooth elements like sinus
        return !selectedToothKey || label === selectedToothKey;
      }
    );

    cariesEntries.forEach(([tooth, data]: any, idx) => {
      const box = data?.box;
      if (!box || box.length < 4) return;
      const conf = Number(data?.conf || 0.65);
      const [x1, y1, x2, y2] = box;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const rx = Math.max((x2 - x1) * 0.85, 18);
      const ry = Math.max((y2 - y1) * 0.85, 18);
      const opacity = Math.min(0.95, (0.45 + conf * 0.48) * opacityBoost);
      items.push(
        <g key={`risk-caries-${tooth}-${idx}`} filter="url(#riskBlurStrong)" style={{ mixBlendMode: 'screen' }}>
          <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="url(#cariesGlow)" fillOpacity={opacity} />
        </g>
      );
    });

    periapicalEntries.forEach(([tooth, data]: any, idx) => {
      const box = data?.box;
      if (!box || box.length < 4) return;
      const conf = Number(data?.conf || 0.72);
      const [x1, y1, x2, y2] = box;
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;
      const rx = Math.max((x2 - x1) * 0.95, 22);
      const ry = Math.max((y2 - y1) * 0.95, 22);
      const opacity = Math.min(0.98, (0.5 + conf * 0.45) * opacityBoost);
      items.push(
        <g key={`risk-peri-${tooth}-${idx}`} filter="url(#riskBlurStrong)" style={{ mixBlendMode: 'screen' }}>
          <ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="url(#periGlow)" fillOpacity={opacity} />
        </g>
      );
    });

    // Bone-loss heatmap restored but filtered tightly to tooth labels
    teeth.forEach((tooth: any, idx: number) => {
      const boneLossPct = Number(
        result?.bonelevel?.[String(tooth?.tooth_label || '')]?.percent ??
        tooth?.bone_loss_pct ??
        0
      );
      if (boneLossPct < 15) return; // Slightly higher threshold to reduce noise
      const severity = Math.min(1, Math.max(0, (boneLossPct - 15) / 30));
      const opacity = Math.min(0.7, (0.28 + severity * 0.35) * opacityBoost);
      const green = Math.round(224 - severity * 140);
      const fill = `rgba(255,${green},71,${opacity})`;
      const contour = tooth?.contour;
      
      if (Array.isArray(contour) && contour.length >= 3) {
        const points = contour.map((pt: any) => `${pt[0]},${pt[1]}`).join(' ');
        items.push(
          <g key={`risk-bone-contour-${tooth?.tooth_label || idx}`} filter="url(#riskBlurSoft)" style={{ mixBlendMode: 'screen' }}>
            <polygon points={points} fill={fill} />
          </g>
        );
      }
    });

    return <g id="risk-overlay-layer">{items}</g>;
  };

  // --- Handlers ---
  const resetView = () => {
    const defaultViewMode = isVolumeCase ? 'original' : 'overlay';
    const defaultViewerMode = isVolumeCase ? 'grid' : 'single';

    setScale(1);
    setZoom(1);
    setOffset({ x: 0, y: 0 });
    setRotation(0);
    setFlipped(false);
    setBrightness(100);
    setContrast(100);
    setInverted(false);
    setViewMode(defaultViewMode);
    setViewerMode(defaultViewerMode);
    setOverlayPreset('all');
    setSelectedTooth(undefined);
    setActiveLegendFilter(null);
    setCaptureRect(null);
    setMagnifierState({
      visible: false,
      clientX: 0,
      clientY: 0,
      viewerX: 0,
      viewerY: 0,
      imgX: 0,
      imgY: 0,
    });
    setShapes([]);
    setMeasurements([]);
    setPendingPoints([]);
    setTempPoint(null);
    setContextMenu({ show: false, x: 0, y: 0, menu: undefined });
    setReportError(null);
    setCaptureNotice(null);
    handleToolChange('pointer');
    setSelectedToolbarButton('pointer');
    setActiveSubTool(null);
    clearAllAnnotations();
    setCornerstoneResetToken((prev) => prev + 1);
  };

  const handleApplyAutoWindow = () => {
    setSelectedToolbarButton('auto-window');
    setReportError(null);

    if (!originalIsDicom) {
      setReportError('Auto Window is available only for DICOM views.');
      return;
    }

    if (!isVolumeCase) {
      if (!dicomHudMetadata || !dicomAutoWindow) {
        setReportError('Auto Window could not be derived for this DICOM.');
        return;
      }

      const controls = deriveControlsForWindow(
        dicomHudMetadata.windowCenter,
        dicomHudMetadata.windowWidth,
        dicomAutoWindow.level,
        dicomAutoWindow.width
      );
      setBrightness(controls.brightnessPercent);
      setContrast(controls.contrastPercent);
      return;
    }

    if (viewMode !== 'original') {
      setViewMode('original');
    }

    if (viewerMode !== (isVolumeCase ? 'grid' : 'single')) {
      setViewerMode(isVolumeCase ? 'grid' : 'single');
    }

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setCornerstoneAutoWindowToken((prev) => prev + 1);
      });
    });
  };

  // Image fit handled by CSS (width: 100%, height: auto) to avoid calculation errors
  const computeDisplayHeight = () => { };

  const toImgCoords = (clientX: number, clientY: number) => {
    if (!imageRef.current) return null;
    const rect = imageRef.current.getBoundingClientRect();
    if (
      clientX < rect.left ||
      clientX > rect.right ||
      clientY < rect.top ||
      clientY > rect.bottom
    ) {
      return null;
    }

    const dx = clientX - rect.left;
    const dy = clientY - rect.top;

    let x = (dx / rect.width) * imageRef.current.naturalWidth;
    let y = (dy / rect.height) * imageRef.current.naturalHeight;

    if (flipped) {
      x = imageRef.current.naturalWidth - x;
    }
    return clampToImage({ x, y });
  };

  const handleToolChange = (tool: string) => {
    const normalizedTool = tool === 'zoom' ? 'pan' : tool;
    const primaryTools = new Set(['pointer', 'pan', 'wlww', 'erase', 'rotate', 'scroll', 'capture-area', 'magnifier']);
    if (primaryTools.has(normalizedTool)) {
      setActiveSubTool(null);
      setPendingPoints([]);
      setTempPoint(null);
      setContextMenu({ show: false, x: 0, y: 0, menu: undefined });
    }
    setLocalActiveTool(normalizedTool);
    // Sync with Cornerstone global tool groups
    switch (normalizedTool) {
      case 'pan': setCornerstoneActiveTool('Pan'); break;
      case 'wlww': setCornerstoneActiveTool('WindowLevel'); break;
      case 'length': setCornerstoneActiveTool('Length'); break;
      case 'arrow': setCornerstoneActiveTool('ArrowAnnotate'); break;
      case 'rect': setCornerstoneActiveTool('RectangleROI'); break;
      case 'ellipse': setCornerstoneActiveTool('EllipticalROI'); break;
      case 'rotate': setCornerstoneActiveTool('TrackballRotate'); break;
      case 'erase': setCornerstoneActiveTool('Eraser'); break;
      case 'scroll': setCornerstoneActiveTool('StackScroll'); break;
      case 'magnifier': setCornerstoneActiveTool('Pan'); break;
      case 'pointer': setCornerstoneActiveTool('Pan'); break;
    }
  };

  const toggleToolOrPointer = (tool: string) => {
    if (activeTool === tool) {
      handleToolChange('pointer');
      setSelectedToolbarButton('pointer');
      return;
    }
    handleToolChange(tool);
    setSelectedToolbarButton(tool);
  };

  const toggleGroupedToolOrPointer = (
    event: React.MouseEvent,
    group: 'measure' | 'annotate',
    isActive: boolean,
  ) => {
    if (isActive) {
      event.stopPropagation();
      setContextMenu({ show: false, x: 0, y: 0, menu: undefined });
      setActiveSubTool(null);
      setPendingPoints([]);
      setTempPoint(null);
      handleToolChange('pointer');
      setSelectedToolbarButton('pointer');
      return;
    }
    setSelectedToolbarButton(group);
    openMenu(event, group);
  };

  const distance = (p1: any, p2: any) => Math.hypot(p2.x - p1.x, p2.y - p1.y);
  const drawingColor = activeTool === 'measure' ? '#22d3ee' : '#fbbf24';

  const pushDebug = (message: string) => {
    setDebugEvents(prev => [`${new Date().toLocaleTimeString()}: ${message}`, ...prev].slice(0, 8));
  };

  const pointCountForTool = (tool: string | null) => {
    if (!tool) return 0;
    if (['length', 'rect', 'circle', 'roi-circle', 'arrow', 'ellipse', 'bidirectional'].includes(tool)) return 2;
    if (tool === 'angle') return 3;
    return 0;
  };

  const buildSmoothPath = (pts: { x: number; y: number }[], close = false) => {
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

  const mapPointToDisplay = (pt: { x: number; y: number }) => {
    if (!imgRect || !imageRef.current?.naturalWidth || !imageRef.current?.naturalHeight) {
      return pt;
    }

    const rawX = (pt.x / imageRef.current.naturalWidth) * imgRect.width;
    return {
      x: flipped ? imgRect.width - rawX : rawX,
      y: (pt.y / imageRef.current.naturalHeight) * imgRect.height,
    };
  };

  const renderDisplayShape = (shape: any, isTemp = false, index?: number) => {
    const basePts = isTemp && tempPoint
      ? shape.type === 'livewire' && shape.points.length > 0
        ? [...shape.points, ...buildLivewireSegment(shape.points[shape.points.length - 1], tempPoint).slice(1)]
        : [...shape.points, tempPoint]
      : shape.points;

    const pts = basePts.map((p: any) => mapPointToDisplay(p));
    if (pts.length === 0) return null;

    const st = shape.type;
    const color = shape.color || drawingColor;
    const key = isTemp ? `display-temp-${st}` : `display-shp-${index ?? 0}`;
    const lw = isTemp ? 2 : 3;
    const dist = (p1: any, p2: any) => Math.hypot(p2.x - p1.x, p2.y - p1.y);
    const textProps = {
      fill: color,
      fontSize: 18,
      fontWeight: 'bold' as const,
      style: { textShadow: '0 0 4px rgba(0,0,0,0.95)' },
    };

    if (st === 'rect' && pts.length >= 2) {
      const x = Math.min(pts[0].x, pts[1].x);
      const y = Math.min(pts[0].y, pts[1].y);
      const w = Math.abs(pts[1].x - pts[0].x);
      const h = Math.abs(pts[1].y - pts[0].y);
      return (
        <g key={key}>
          <rect x={x} y={y} width={w} height={h} fill="rgba(251,191,36,0.22)" stroke="#ffffff" strokeWidth={lw + 2} />
          <rect x={x} y={y} width={w} height={h} fill="rgba(251,191,36,0.22)" stroke={color} strokeWidth={lw} />
        </g>
      );
    }

    if ((st === 'circle' || st === 'roi-circle') && pts.length >= 2) {
      const rad = dist(pts[0], pts[1]);
      return (
        <g key={key}>
          <circle cx={pts[0].x} cy={pts[0].y} r={rad} fill="rgba(251,191,36,0.22)" stroke="#ffffff" strokeWidth={lw + 2} />
          <circle cx={pts[0].x} cy={pts[0].y} r={rad} fill="rgba(251,191,36,0.22)" stroke={color} strokeWidth={lw} />
        </g>
      );
    }

    if (st === 'ellipse' && pts.length >= 2) {
      const x = Math.min(pts[0].x, pts[1].x);
      const y = Math.min(pts[0].y, pts[1].y);
      const w = Math.abs(pts[1].x - pts[0].x);
      const h = Math.abs(pts[1].y - pts[0].y);
      return (
        <g key={key}>
          <ellipse cx={x + w / 2} cy={y + h / 2} rx={w / 2} ry={h / 2} fill="rgba(251,191,36,0.22)" stroke="#ffffff" strokeWidth={lw + 2} />
          <ellipse cx={x + w / 2} cy={y + h / 2} rx={w / 2} ry={h / 2} fill="rgba(251,191,36,0.22)" stroke={color} strokeWidth={lw} />
        </g>
      );
    }

    if (st === 'length' && pts.length >= 2) {
      return (
        <g key={key}>
          <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke="#ffffff" strokeWidth={lw + 2} />
          <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={color} strokeWidth={lw} />
          {!isTemp && (
            <text x={(pts[0].x + pts[1].x) / 2} y={(pts[0].y + pts[1].y) / 2 - 10} textAnchor="middle" {...textProps}>
              {(dist(pts[0], pts[1]) * mmPerPixel).toFixed(2)} mm
            </text>
          )}
        </g>
      );
    }

    if (st === 'bidirectional' && pts.length >= 2) {
      const x = Math.min(pts[0].x, pts[1].x);
      const y = Math.min(pts[0].y, pts[1].y);
      const w = Math.abs(pts[1].x - pts[0].x);
      const h = Math.abs(pts[1].y - pts[0].y);
      const cx = x + w / 2;
      const cy = y + h / 2;
      return (
        <g key={key}>
          <rect x={x} y={y} width={w} height={h} fill="rgba(251,191,36,0.16)" stroke="#ffffff" strokeWidth={lw + 2} strokeDasharray="10 8" />
          <rect x={x} y={y} width={w} height={h} fill="rgba(251,191,36,0.16)" stroke={color} strokeWidth={lw} strokeDasharray="10 8" />
          <line x1={x} y1={cy} x2={x + w} y2={cy} stroke="#ffffff" strokeWidth={lw + 2} />
          <line x1={x} y1={cy} x2={x + w} y2={cy} stroke={color} strokeWidth={lw} />
          <line x1={cx} y1={y} x2={cx} y2={y + h} stroke="#ffffff" strokeWidth={lw + 2} />
          <line x1={cx} y1={y} x2={cx} y2={y + h} stroke={color} strokeWidth={lw} />
        </g>
      );
    }

    if (st === 'angle' && pts.length >= 2) {
      if (pts.length === 2) {
        return (
          <g key={key}>
            <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke="#ffffff" strokeWidth={lw + 2} strokeDasharray="6 6" />
            <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={color} strokeWidth={lw} strokeDasharray="6 6" />
          </g>
        );
      }

      const a = dist(pts[1], pts[0]);
      const b = dist(pts[1], pts[2]);
      const c = dist(pts[0], pts[2]);
      const val = (a * a + b * b - c * c) / (2 * a * b);
      const deg = Math.acos(Math.max(-1, Math.min(1, val))) * (180 / Math.PI);
      return (
        <g key={key}>
          <polyline points={pts.map((p: any) => `${p.x},${p.y}`).join(' ')} fill="none" stroke="#ffffff" strokeWidth={lw + 2} />
          <polyline points={pts.map((p: any) => `${p.x},${p.y}`).join(' ')} fill="none" stroke={color} strokeWidth={lw} />
          {!isTemp && (
            <text x={pts[1].x} y={pts[1].y - 12} textAnchor="middle" {...textProps}>
              {deg.toFixed(1)}{String.fromCharCode(176)}
            </text>
          )}
        </g>
      );
    }

    if (st === 'arrow' && pts.length >= 2) {
      return (
        <g key={key}>
          <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke="#ffffff" strokeWidth={lw + 2} markerEnd="url(#display-arrowhead-white)" />
          <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={color} strokeWidth={lw} markerEnd="url(#display-arrowhead)" />
        </g>
      );
    }

    if ((st === 'roi-free' || st === 'spline-roi' || st === 'livewire') && pts.length >= 2) {
      const points = pts.map((p: any) => `${p.x},${p.y}`).join(' ');
      const path = buildSmoothPath(pts, st !== 'livewire' && !isTemp);
      return (
        <g key={key}>
          {st === 'spline-roi' ? (
            <>
              <path d={path} fill={isTemp ? 'none' : 'rgba(251,191,36,0.18)'} stroke="#ffffff" strokeWidth={lw + 2} />
              <path d={path} fill={isTemp ? 'none' : 'rgba(251,191,36,0.18)'} stroke={color} strokeWidth={lw} />
            </>
          ) : st === 'livewire' && !isTemp ? (
            <>
              <polygon points={points} fill="rgba(251,191,36,0.18)" stroke="#ffffff" strokeWidth={lw + 2} />
              <polygon points={points} fill="rgba(251,191,36,0.18)" stroke={color} strokeWidth={lw} />
            </>
          ) : (
            <>
              <polyline points={points} fill={isTemp ? 'none' : 'rgba(251,191,36,0.18)'} stroke="#ffffff" strokeWidth={lw + 2} />
              <polyline points={points} fill={isTemp ? 'none' : 'rgba(251,191,36,0.18)'} stroke={color} strokeWidth={lw} />
            </>
          )}
        </g>
      );
    }

    if (st === 'text' && pts.length >= 1) {
      return (
        <text key={key} x={pts[0].x} y={pts[0].y} {...textProps}>
          {shape.text}
        </text>
      );
    }

    return null;
  };

  const clampToImage = (pt: { x: number; y: number }) => ({
    x: Math.max(0, Math.min(dimensions.width - 1, pt.x)),
    y: Math.max(0, Math.min(dimensions.height - 1, pt.y)),
  });

  const edgeStrengthAt = (x: number, y: number) => {
    const edgeMap = edgeMapRef.current;
    if (!edgeMap) return 0;

    const xi = Math.max(0, Math.min(dimensions.width - 1, Math.round(x)));
    const yi = Math.max(0, Math.min(dimensions.height - 1, Math.round(y)));
    return edgeMap[yi * dimensions.width + xi] || 0;
  };

  const buildEdgeMapFromImage = (img: HTMLImageElement) => {
    const w = img.naturalWidth || 0;
    const h = img.naturalHeight || 0;
    if (!w || !h) {
      edgeMapRef.current = null;
      return;
    }

    try {
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        edgeMapRef.current = null;
        return;
      }

      ctx.drawImage(img, 0, 0, w, h);
      const { data } = ctx.getImageData(0, 0, w, h);
      const gray = new Float32Array(w * h);
      const edge = new Float32Array(w * h);

      for (let i = 0; i < w * h; i += 1) {
        const idx = i * 4;
        gray[i] = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
      }

      for (let y = 1; y < h - 1; y += 1) {
        for (let x = 1; x < w - 1; x += 1) {
          const idx = y * w + x;
          const gx =
            -gray[idx - w - 1] - 2 * gray[idx - 1] - gray[idx + w - 1] +
            gray[idx - w + 1] + 2 * gray[idx + 1] + gray[idx + w + 1];
          const gy =
            -gray[idx - w - 1] - 2 * gray[idx - w] - gray[idx - w + 1] +
            gray[idx + w - 1] + 2 * gray[idx + w] + gray[idx + w + 1];
          edge[idx] = Math.sqrt(gx * gx + gy * gy);
        }
      }

      edgeMapRef.current = edge;
    } catch (error) {
      console.error('Failed to build edge map for livewire:', error);
      edgeMapRef.current = null;
    }
  };

  const buildLivewireSegment = (
    start: { x: number; y: number },
    end: { x: number; y: number }
  ) => {
    if (!edgeMapRef.current) return [start, end];

    const dist = distance(start, end);
    if (dist < 2) return [start, end];

    const searchRadius = Math.min(12, Math.max(4, Math.round(dist / 35)));
    const steps = Math.max(8, Math.min(180, Math.round(dist / 3)));
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const nx = -dy / dist;
    const ny = dx / dist;

    const segment = [start];
    let prev = start;

    for (let i = 1; i < steps; i += 1) {
      const t = i / steps;
      const baseX = start.x + dx * t;
      const baseY = start.y + dy * t;
      let bestPoint = { x: baseX, y: baseY };
      let bestScore = Number.NEGATIVE_INFINITY;

      for (let offset = -searchRadius; offset <= searchRadius; offset += 1) {
        const candidate = clampToImage({
          x: baseX + nx * offset,
          y: baseY + ny * offset,
        });
        const edgeScore = edgeStrengthAt(candidate.x, candidate.y);
        const smoothPenalty = distance(candidate, prev) * 0.12;
        const centerPenalty = Math.abs(offset) * 0.75;
        const score = edgeScore - smoothPenalty - centerPenalty;

        if (score > bestScore) {
          bestScore = score;
          bestPoint = candidate;
        }
      }

      segment.push(bestPoint);
      prev = bestPoint;
    }

    segment.push(end);
    return segment;
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!showSrc) return;
    if (e.button === 2) return; // Right click

    if (activeTool === 'magnifier') {
      return;
    }

    if (activeTool === 'pan') {
      dragRef.current = { active: true, mode: 'pan', startX: e.clientX, startY: e.clientY };
      return;
    }
    if (activeTool === 'wlww') {
      dragRef.current = { active: true, mode: 'wl', startX: e.clientX, startY: e.clientY };
      return;
    }

    if (activeTool === 'capture-area') {
      if (shouldUseCornerstone) {
        if (viewerMode !== 'grid') {
          e.preventDefault();
          void handleCapture(undefined, { destination: 'clipboard' });
        }
        return;
      }
      e.preventDefault();
      void handleCapture(undefined, { destination: 'clipboard' });
      return;
    }

    const isDrawing = activeTool === 'measure' || activeTool === 'annotate' || activeSubTool;
    if (isDrawing) {
      if (!activeSubTool) return;
      const pt = toImgCoords(e.clientX, e.clientY);
      if (!pt) {
        pushDebug(`mouseDown ignored outside-image tool=${activeTool}/${activeSubTool}`);
        return;
      }
      pushDebug(`mouseDown tool=${activeTool}/${activeSubTool} x=${pt.x.toFixed(1)} y=${pt.y.toFixed(1)}`);

      if (activeSubTool === 'text') {
        const text = prompt("Text Annotation:", "Note");
        if (text) {
          setShapes(prev => [...prev, { type: 'text', points: [pt], text, color: '#fbbf24' }]);
          pushDebug(`text added value="${text}"`);
        }
        return;
      }

      if (activeSubTool === 'livewire' && pendingPoints.length > 0) {
        const segment = buildLivewireSegment(pendingPoints[pendingPoints.length - 1], pt);
        setPendingPoints(prev => [...prev, ...segment.slice(1)]);
        return;
      }

      // Check for auto-finish
      const needed = pointCountForTool(activeSubTool);

      const newPts = [...pendingPoints, pt];
      if (needed > 0 && newPts.length >= needed) {
        finishShape(newPts);
      } else {
        setPendingPoints(newPts);
        pushDebug(`pendingPoints=${newPts.length}`);
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragRef.current.active) {
      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;
      dragRef.current.startX = e.clientX;
      dragRef.current.startY = e.clientY;

      if (dragRef.current.mode === 'pan') {
        setOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
      } else if (dragRef.current.mode === 'wl') {
        setBrightness(b => Math.max(0, Math.min(300, b - dy * 0.5)));
        setContrast(c => Math.max(0, Math.min(300, c + dx * 0.5)));
      }
      return;
    }

    if (activeTool === 'magnifier' && !shouldUseCornerstone) {
      const pt = toImgCoords(e.clientX, e.clientY);
      const vRect = viewerRef.current?.getBoundingClientRect();
      if (!pt || !vRect) {
        hideMagnifier();
      } else {
        setMagnifierState({
          visible: true,
          clientX: e.clientX,
          clientY: e.clientY,
          viewerX: e.clientX - vRect.left,
          viewerY: e.clientY - vRect.top,
          imgX: pt.x,
          imgY: pt.y,
        });
      }
      return;
    }

    const isDrawing = activeTool === 'measure' || activeTool === 'annotate' || activeSubTool;
    if (activeSubTool && isDrawing) {
      const pt = toImgCoords(e.clientX, e.clientY);
      setTempPoint(pt);
    }

    if (activeTool === 'capture-area' && captureRect?.active) {
      if (shouldUseCornerstone) return;
      e.preventDefault();
      const vRect = viewerRef.current?.getBoundingClientRect();
      if (!vRect) return;
      const currentX = e.clientX - vRect.left;
      const currentY = e.clientY - vRect.top;
      setCaptureRect(prev => prev ? ({ ...prev, w: currentX - prev.x, h: currentY - prev.y }) : null);
    }
  };

  const handleMouseUp = (e?: React.MouseEvent) => {
    if (activeTool === 'capture-area' && captureRect?.active) {
      if (shouldUseCornerstone) return;
      e?.preventDefault();
      setCaptureRect(prev => ({ ...prev!, active: false }));
      return;
    }
    dragRef.current.active = false;
  };

  const handleMouseLeave = () => {
    dragRef.current.active = false;
    if (activeTool === 'magnifier') {
      hideMagnifier();
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    pushDebug('contextMenu finish requested');
    // Finish shape if drawing
    if ((activeTool === 'measure' || activeTool === 'annotate') && pendingPoints.length > 0) {
      finishShape();
    }
  };

  const finishShape = (customPoints?: any[]) => {
    const pts = customPoints || pendingPoints;
    if (!activeSubTool || pts.length === 0) return;

    // Minimal validation
    const needed = pointCountForTool(activeSubTool);
    if (needed > 0 && pts.length < needed) return;

    setShapes(prev => [...prev, { type: activeSubTool, points: pts, color: drawingColor }]);
    setPendingPoints([]);
    setTempPoint(null);
    pushDebug(`shape committed type=${activeSubTool} points=${pts.length}`);
  };

  const hideMagnifier = () => {
    setMagnifierState(prev => (prev.visible ? { ...prev, visible: false } : prev));
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (activeTool !== 'pan') return;
    e.preventDefault();
    handleZoom(e.deltaY < 0 ? 0.1 : -0.1);
  };

  // Re-calc imgRect on change
  useEffect(() => {
    const updateRect = () => {
      if (imageRef.current) {
        setImgRect(imageRef.current.getBoundingClientRect());
      }
      computeDisplayHeight();
    };
    window.addEventListener('resize', updateRect);
    // backup poller
    const timer = setInterval(updateRect, 1000);
    return () => {
      window.removeEventListener('resize', updateRect);
      clearInterval(timer);
    };
  }, [scale, offset, rotation, flipped, showSrc]);

  useEffect(() => {
    if (activeTool !== 'magnifier' || shouldUseCornerstone || !showSrc) {
      hideMagnifier();
    }
  }, [activeTool, shouldUseCornerstone, showSrc]);

  // Handle Wheel Zoom (Non-passive to prevent scroll)
  useEffect(() => {
    const el = viewerRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      if (activeTool === 'pan') {
        e.preventDefault();
        e.stopPropagation();
        handleZoom(e.deltaY < 0 ? 0.1 : -0.1);
      }
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [activeTool]);

  // Menu Handling
  const openMenu = (e: React.MouseEvent, type: 'measure' | 'annotate' | 'grid') => {
    e.stopPropagation();
    if (contextMenu.show && contextMenu.menu === type) {
      setContextMenu({ show: false, x: 0, y: 0, menu: undefined });
      setActiveSubTool(null);
      setPendingPoints([]);
      setTempPoint(null);
      if (activeTool === type) handleToolChange('pointer');
      pushDebug(`menu closed type=${type}`);
      return;
    }

    handleToolChange(type);
    setActiveSubTool(null);
    setPendingPoints([]);
    setTempPoint(null);
    if (type === 'grid') setTempGridLayout(gridLayout);

    // Position menu near the button that was clicked
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    // We are inside a float absolute div, coordinate space might be tricky if we use fixed.
    // But fixed is safest.
    setContextMenu({ show: true, x: rect.right, y: rect.top, menu: type });
    pushDebug(`menu opened type=${type}`);
  };

  const selectSubTool = (sub: string) => {
    setActiveSubTool(sub);
    handleToolChange(sub); // Use synchronized handler
    setContextMenu(prev => ({ ...prev, show: false }));
    pushDebug(`subtool selected=${sub}`);
  };

  const magnifierViewport = (() => {
    if (
      shouldUseCornerstone ||
      activeTool !== 'magnifier' ||
      !showSrc ||
      !magnifierState.visible ||
      !dimensions.width ||
      !dimensions.height ||
      !viewerRef.current
    ) {
      return null;
    }

    const viewerWidth = viewerRef.current.clientWidth || 0;
    const viewerHeight = viewerRef.current.clientHeight || 0;
    if (!viewerWidth || !viewerHeight) return null;

    const sampleSize = Math.max(MAGNIFIER_SIZE_PX / MAGNIFIER_ZOOM_FACTOR, 1);
    const sampleLeft = clampNumber(
      magnifierState.imgX - sampleSize / 2,
      0,
      Math.max(0, dimensions.width - sampleSize)
    );
    const sampleTop = clampNumber(
      magnifierState.imgY - sampleSize / 2,
      0,
      Math.max(0, dimensions.height - sampleSize)
    );

    let lensLeft = magnifierState.viewerX + MAGNIFIER_CURSOR_OFFSET_PX;
    let lensTop = magnifierState.viewerY + MAGNIFIER_CURSOR_OFFSET_PX;

    if (lensLeft + MAGNIFIER_SIZE_PX + MAGNIFIER_EDGE_PADDING_PX > viewerWidth) {
      lensLeft = magnifierState.viewerX - MAGNIFIER_CURSOR_OFFSET_PX - MAGNIFIER_SIZE_PX;
    }
    if (lensTop + MAGNIFIER_SIZE_PX + MAGNIFIER_EDGE_PADDING_PX > viewerHeight) {
      lensTop = magnifierState.viewerY - MAGNIFIER_CURSOR_OFFSET_PX - MAGNIFIER_SIZE_PX;
    }

    lensLeft = clampNumber(
      lensLeft,
      MAGNIFIER_EDGE_PADDING_PX,
      Math.max(MAGNIFIER_EDGE_PADDING_PX, viewerWidth - MAGNIFIER_SIZE_PX - MAGNIFIER_EDGE_PADDING_PX)
    );
    lensTop = clampNumber(
      lensTop,
      MAGNIFIER_EDGE_PADDING_PX,
      Math.max(MAGNIFIER_EDGE_PADDING_PX, viewerHeight - MAGNIFIER_SIZE_PX - MAGNIFIER_EDGE_PADDING_PX)
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
    if (!magnifierViewport || !magnifierCanvasRef.current || !imageRef.current) {
      return;
    }

    let frameId = 0;
    frameId = window.requestAnimationFrame(() => {
      const canvas = magnifierCanvasRef.current;
      const image = imageRef.current;
      if (!canvas || !image) return;

      canvas.width = MAGNIFIER_SIZE_PX;
      canvas.height = MAGNIFIER_SIZE_PX;

      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      ctx.clearRect(0, 0, MAGNIFIER_SIZE_PX, MAGNIFIER_SIZE_PX);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.save();
      ctx.filter = `invert(${inverted ? 1 : 0}) brightness(${brightness}%) contrast(${contrast}%)`;
      if (flipped) {
        ctx.translate(MAGNIFIER_SIZE_PX, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(
        image,
        magnifierViewport.sampleLeft,
        magnifierViewport.sampleTop,
        magnifierViewport.sampleSize,
        magnifierViewport.sampleSize,
        0,
        0,
        MAGNIFIER_SIZE_PX,
        MAGNIFIER_SIZE_PX
      );
      ctx.restore();

      const enhanced = ctx.getImageData(0, 0, MAGNIFIER_SIZE_PX, MAGNIFIER_SIZE_PX);
      enhanceMagnifierImage(enhanced);
      ctx.putImageData(enhanced, 0, 0);
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [
    magnifierViewport,
    brightness,
    contrast,
    inverted,
    flipped,
    showSrc,
  ]);

  const renderMagnifier = () => {
    if (!magnifierViewport) return null;

    const overlayContent = (
      <>
        {renderAIDetections({ emphasize: true })}
        {renderRiskDetections({ emphasize: true })}
        {shapes.map((s: any, idx: number) => renderShape(s, false, idx))}
        {activeSubTool && renderShape({ type: activeSubTool, points: pendingPoints, color: drawingColor }, true)}
      </>
    );

    const transformedOverlay = (
      <g transform={`translate(${-magnifierViewport.sampleLeft * MAGNIFIER_ZOOM_FACTOR} ${-magnifierViewport.sampleTop * MAGNIFIER_ZOOM_FACTOR}) scale(${MAGNIFIER_ZOOM_FACTOR})`}>
        {overlayContent}
      </g>
    );

    return (
      <div
        className="pointer-events-none absolute overflow-hidden"
        style={{
          position: 'absolute',
          left: magnifierViewport.lensLeft,
          top: magnifierViewport.lensTop,
          width: MAGNIFIER_SIZE_PX,
          height: MAGNIFIER_SIZE_PX,
          isolation: 'isolate',
          zIndex: 20000,
          background: 'rgba(0,0,0,0.88)',
          border: '2px solid rgba(255,255,255,0.92)',
          boxShadow: '0 18px 42px rgba(0,0,0,0.42), inset 0 0 0 1px rgba(34,211,238,0.35)',
        }}
      >
        <canvas
          ref={magnifierCanvasRef}
          className="absolute inset-0 h-full w-full"
          style={{ imageRendering: 'auto' }}
        />
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox={`0 0 ${MAGNIFIER_SIZE_PX} ${MAGNIFIER_SIZE_PX}`}
          preserveAspectRatio="none"
        >
          <defs>
            <marker id="display-arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
              <polygon points="0 0, 6 2, 0 4" fill="#facc15" />
            </marker>
            <marker id="display-arrowhead-white" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
              <polygon points="0 0, 6 2, 0 4" fill="#ffffff" />
            </marker>
            <filter id="riskBlurStrong" x="-35%" y="-35%" width="170%" height="170%">
              <feGaussianBlur stdDeviation={22 / effectiveScale} />
            </filter>
            <filter id="riskBlurSoft" x="-25%" y="-25%" width="150%" height="150%">
              <feGaussianBlur stdDeviation={12 / effectiveScale} />
            </filter>
          </defs>
          <rect
            x={0}
            y={0}
            width={MAGNIFIER_SIZE_PX}
            height={MAGNIFIER_SIZE_PX}
            fill="rgba(255,255,255,0.04)"
          />
          {flipped ? (
            <g transform={`translate(${MAGNIFIER_SIZE_PX} 0) scale(-1 1)`}>
              {transformedOverlay}
            </g>
          ) : (
            transformedOverlay
          )}
          <rect
            x={1}
            y={1}
            width={MAGNIFIER_SIZE_PX - 2}
            height={MAGNIFIER_SIZE_PX - 2}
            fill="none"
            stroke="rgba(34,211,238,0.52)"
            strokeWidth={1}
          />
        </svg>
      </div>
    );
  };

  // Render SVG Helper
  const renderShape = (shape: any, isTemp = false, index?: number) => {
    const pts = isTemp && tempPoint
      ? shape.type === 'livewire' && shape.points.length > 0
        ? [...shape.points, ...buildLivewireSegment(shape.points[shape.points.length - 1], tempPoint).slice(1)]
        : [...shape.points, tempPoint]
      : shape.points;
    if (pts.length === 0) return null;
    const st = shape.type;
    const color = shape.color || drawingColor;
    const key = isTemp ? `temp-${st}` : `shp-${index ?? 0}`;

    // Dynamic sizing based on zoom scale
    const lw = 4 / effectiveScale;
    const txtSz = 16 / effectiveScale;
    const r = 6 / effectiveScale;
    const pad = 8 / effectiveScale;
    const dashedWidth = 1;
    const arrowColor = '#facc15';
    const shapeColor = '#22c55e';
    const dashedPattern = '6 4';

    const renderInfoLabel = (
      x: number,
      y: number,
      lines: string[],
      align: 'start' | 'middle' | 'end' = 'start',
      withBackground = false
    ) => {
      const fontSize = Math.max(10, txtSz * 0.72);
      const lineHeight = fontSize * 1.15;
      const width = Math.max(...lines.map(line => line.length), 4) * fontSize * 0.56 + pad * 2;
      const height = lines.length * lineHeight + pad * 2;
      const boxX = align === 'middle' ? x - width / 2 : align === 'end' ? x - width : x;
      const boxY = y - height;

      return (
        <g pointerEvents="none">
          {withBackground && (
            <rect
              x={boxX}
              y={boxY}
              width={width}
              height={height}
              rx={10 / effectiveScale}
              fill="rgba(3,7,18,0.82)"
              stroke="rgba(255,255,255,0.55)"
              strokeWidth={Math.max(0.8, 1 / effectiveScale)}
            />
          )}
          {lines.map((line, lineIndex) => (
            <text
              key={`${line}-${lineIndex}`}
              x={align === 'middle' ? x : align === 'end' ? boxX + width - pad : boxX + pad}
              y={withBackground ? boxY + pad + fontSize + lineIndex * lineHeight : y + lineIndex * lineHeight}
              fill={withBackground ? '#f8fafc' : color}
              fontSize={fontSize}
              fontWeight="700"
              textAnchor={align}
              stroke="rgba(0,0,0,0.95)"
              strokeWidth={2}
              paintOrder="stroke"
              style={{ textShadow: '0px 0px 2px rgba(0,0,0,0.95)' }}
            >
              {line}
            </text>
          ))}
        </g>
      );
    };

    const handleDelete = (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!isTemp && index !== undefined) {
        setShapes(prev => prev.filter((_, i) => i !== index));
        pushDebug(`shape removed index=${index} type=${st}`);
      }
    };

    // Add interaction props only for finalized shapes
    const commonProps = isTemp ? {} : {
      onClick: handleDelete,
      className: "cursor-pointer pointer-events-auto hover:opacity-80",
      style: {
        cursor: 'pointer',
        pointerEvents: 'visiblePainted' as const,
        filter: 'drop-shadow(0 0 2px rgba(0,0,0,0.95)) drop-shadow(0 0 8px rgba(255,255,255,0.45))'
      }
    };

    const dist = (p1: any, p2: any) => Math.hypot(p2.x - p1.x, p2.y - p1.y);

    if (st === 'length' && pts.length >= 2) {
      return (
        <g key={key} {...commonProps}>
          <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={shapeColor} strokeWidth={dashedWidth} strokeDasharray={dashedPattern} vectorEffect="non-scaling-stroke" />
          {!isTemp && (
            renderInfoLabel(
              (pts[0].x + pts[1].x) / 2,
              (pts[0].y + pts[1].y) / 2 - (8 / effectiveScale),
              [`Length ${(dist(pts[0], pts[1]) * mmPerPixel).toFixed(2)} mm`],
              'middle'
            )
          )}
        </g>
      );
    }
    if (st === 'angle' && pts.length >= 2) {
      if (pts.length === 2) {
        return (
          <g key={key}>
            <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={shapeColor} strokeWidth={dashedWidth} strokeDasharray={dashedPattern} vectorEffect="non-scaling-stroke" />
          </g>
        );
      }
      const a = dist(pts[1], pts[0]);
      const b = dist(pts[1], pts[2]);
      const c = dist(pts[0], pts[2]);
      const val = (a * a + b * b - c * c) / (2 * a * b);
      const deg = Math.acos(Math.max(-1, Math.min(1, val))) * (180 / Math.PI);
      return (
        <g key={key} {...commonProps}>
          <polyline points={pts.map((p: any) => `${p.x},${p.y}`).join(' ')} fill="none" stroke={shapeColor} strokeWidth={dashedWidth} strokeDasharray={dashedPattern} vectorEffect="non-scaling-stroke" />
          {!isTemp && renderInfoLabel(pts[1].x, pts[1].y - (8 / effectiveScale), [`Angle ${deg.toFixed(1)}${String.fromCharCode(176)}`], 'middle')}
        </g>
      );
    }
    if (st === 'bidirectional' && pts.length >= 2) {
      const x = Math.min(pts[0].x, pts[1].x);
      const y = Math.min(pts[0].y, pts[1].y);
      const w = Math.abs(pts[1].x - pts[0].x);
      const h = Math.abs(pts[1].y - pts[0].y);
      const cx = x + w / 2;
      const cy = y + h / 2;
      return (
        <g key={key} {...commonProps}>
          <rect x={x} y={y} width={w} height={h} fill="none" stroke={shapeColor} strokeWidth={dashedWidth} strokeDasharray={dashedPattern} vectorEffect="non-scaling-stroke" />
          <line x1={x} y1={cy} x2={x + w} y2={cy} stroke={shapeColor} strokeWidth={dashedWidth} strokeDasharray={dashedPattern} vectorEffect="non-scaling-stroke" />
          <line x1={cx} y1={y} x2={cx} y2={y + h} stroke={shapeColor} strokeWidth={dashedWidth} strokeDasharray={dashedPattern} vectorEffect="non-scaling-stroke" />
          {!isTemp && (
            <>
              {renderInfoLabel(x + w + (18 / effectiveScale), y + h - (6 / effectiveScale), [
                `Width ${(w * mmPerPixel).toFixed(2)} mm`,
                `Height ${(h * mmPerPixel).toFixed(2)} mm`
              ])}
            </>
          )}
        </g>
      );
    }
    if (st === 'roi-free') {
      if (pts.length < 2) return null;
      return (
        <polygon
          key={key}
          points={pts.map((p: any) => `${p.x},${p.y}`).join(' ')}
          fill="none"
          stroke={shapeColor}
          strokeWidth={dashedWidth}
          strokeDasharray={dashedPattern}
          vectorEffect="non-scaling-stroke"
          {...commonProps}
        />
      );
    }
    if (st === 'livewire' && pts.length >= 2) {
      const points = pts.map((p: any) => `${p.x},${p.y}`).join(' ');
      return isTemp ? (
        <polyline
          key={key}
          points={points}
          fill="none"
          stroke={shapeColor}
          strokeWidth={dashedWidth}
          strokeDasharray={dashedPattern}
          vectorEffect="non-scaling-stroke"
          {...commonProps}
        />
      ) : (
        <polygon
          key={key}
          points={points}
          fill="none"
          stroke={shapeColor}
          strokeWidth={dashedWidth}
          strokeDasharray={dashedPattern}
          vectorEffect="non-scaling-stroke"
          {...commonProps}
        />
      );
    }
    if (st === 'spline-roi' && pts.length >= 2) {
      const path = buildSmoothPath(pts, !isTemp);
      return (
        <path
          key={key}
          d={path}
          fill="none"
          stroke={shapeColor}
          strokeWidth={dashedWidth}
          strokeDasharray={dashedPattern}
          vectorEffect="non-scaling-stroke"
          {...commonProps}
        />
      );
    }
    if ((st === 'roi-circle' || st === 'circle') && pts.length >= 2) {
      const rad = dist(pts[0], pts[1]);
      const diameter = rad * 2;
      const area = Math.PI * rad * rad * mmPerPixel * mmPerPixel;
      return (
        <g key={key} {...commonProps}>
          <circle cx={pts[0].x} cy={pts[0].y} r={rad} fill="none" stroke={shapeColor} strokeWidth={dashedWidth} strokeDasharray={dashedPattern} vectorEffect="non-scaling-stroke" />
          {!isTemp && (
            renderInfoLabel(pts[0].x + rad + (16 / effectiveScale), pts[0].y - (12 / effectiveScale), [
              `Radius ${(rad * mmPerPixel).toFixed(2)} mm`,
              `Diameter ${(diameter * mmPerPixel).toFixed(2)} mm`,
              `Area ${area.toFixed(2)} mm2`
            ])
          )}
        </g>
      );
    }
    if (st === 'rect' && pts.length >= 2) {
      const x = Math.min(pts[0].x, pts[1].x);
      const y = Math.min(pts[0].y, pts[1].y);
      const w = Math.abs(pts[1].x - pts[0].x);
      const h = Math.abs(pts[1].y - pts[0].y);
      return (
        <g key={key} {...commonProps}>
          <rect x={x} y={y} width={w} height={h} fill="none" stroke={shapeColor} strokeWidth={dashedWidth} strokeDasharray={dashedPattern} vectorEffect="non-scaling-stroke" />
          {!isTemp && renderInfoLabel(x + w + (16 / effectiveScale), y + h - (4 / effectiveScale), [
            `Width ${(w * mmPerPixel).toFixed(2)} mm`,
            `Height ${(h * mmPerPixel).toFixed(2)} mm`,
            `Area ${(w * h * mmPerPixel * mmPerPixel).toFixed(2)} mm2`
          ])}
        </g>
      );
    }
    if (st === 'ellipse' && pts.length >= 2) {
      const x = Math.min(pts[0].x, pts[1].x);
      const y = Math.min(pts[0].y, pts[1].y);
      const w = Math.abs(pts[1].x - pts[0].x);
      const h = Math.abs(pts[1].y - pts[0].y);
      return (
        <g key={key} {...commonProps}>
          <ellipse
            cx={x + w / 2}
            cy={y + h / 2}
            rx={w / 2}
            ry={h / 2}
            fill="none"
            stroke={shapeColor}
            strokeWidth={dashedWidth}
            strokeDasharray={dashedPattern}
            vectorEffect="non-scaling-stroke"
          />
          {!isTemp && renderInfoLabel(x + w + (16 / effectiveScale), y + h - (4 / effectiveScale), [
            `Rx ${((w / 2) * mmPerPixel).toFixed(2)} mm`,
            `Ry ${((h / 2) * mmPerPixel).toFixed(2)} mm`,
            `Area ${(Math.PI * (w / 2) * (h / 2) * mmPerPixel * mmPerPixel).toFixed(2)} mm2`
          ])}
        </g>
      );
    }
    if (st === 'arrow' && pts.length >= 2) {
      return (
        <g key={key} stroke={arrowColor} strokeWidth={dashedWidth} {...commonProps}>
          <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={arrowColor} strokeWidth={dashedWidth} strokeDasharray={dashedPattern} markerEnd="url(#arrowhead)" vectorEffect="non-scaling-stroke" />
          {!isTemp && renderInfoLabel(pts[1].x + (12 / effectiveScale), pts[1].y - (8 / effectiveScale), ['Arrow'])}
        </g>
      );
    }
    if (st === 'text') {
      return (
        <g key={key} {...commonProps}>
          {renderInfoLabel(pts[0].x, pts[0].y, [shape.text || 'Annotation'], 'start', false)}
          <text x={pts[0].x} y={pts[0].y} fill={color} fontSize={txtSz} fontWeight="bold" style={{ textShadow: '0px 0px 4px black' }}>
            {shape.text}
          </text>
        </g>
      );
    }
    return null;
  };

  const toolbarPrimaryButtons = (
    <>
      <ToolBtn compact active={selectedToolbarButton === 'pointer'} onClick={() => { handleToolChange('pointer'); setSelectedToolbarButton('pointer'); }} icon={MousePointer} title="Select Tool" />
      <ToolBtn compact active={selectedToolbarButton === 'pan'} onClick={() => toggleToolOrPointer('pan')} icon={Hand} title="Pan + Wheel Zoom" />
      <ToolBtn compact active={selectedToolbarButton === 'wlww'} onClick={() => toggleToolOrPointer('wlww')} icon={WindowLevelIcon} title="Window / Level" />
      <ToolBtn compact active={selectedToolbarButton === 'invert'} onClick={() => { setInverted((value) => !value); setSelectedToolbarButton((value) => value === 'invert' ? 'pointer' : 'invert'); }} icon={InvertIcon} title="Invert" />
      <ToolBtn compact active={selectedToolbarButton === 'flip'} onClick={() => { setFlipped((value) => !value); setSelectedToolbarButton((value) => value === 'flip' ? 'pointer' : 'flip'); }} icon={FlipHorizontal} title="Flip Horizontal" />
      <ToolBtn compact active={false} onClick={() => setRotation((value) => value + 90)} icon={RotateCw} title="Rotate 90" />
      {viewerMode === 'grid' && (
        <>
          <ToolBtn compact active={selectedToolbarButton === 'rotate'} onClick={() => toggleToolOrPointer('rotate')} icon={Rotate3d} title="3D Rotate" />
          <ToolBtn compact active={selectedToolbarButton === 'scroll'} onClick={() => toggleToolOrPointer('scroll')} icon={ChevronsUpDown} title="Slice Scroll" />
        </>
      )}
      <ToolBtn compact active={selectedToolbarButton === 'magnifier'} disabled={magnifierDisabled} onClick={() => toggleToolOrPointer('magnifier')} icon={Search} title={magnifierDisabled ? 'Magnifier disabled in Grid/MPR view' : 'Magnifier'} />
      <ToolBtn compact active={selectedToolbarButton === 'measure'} onClick={(event: any) => toggleGroupedToolOrPointer(event, 'measure', selectedToolbarButton === 'measure')} icon={Ruler} title="Measure" />
    </>
  );

  const toolbarSecondaryButtons = (
    <>
      <ToolBtn compact active={selectedToolbarButton === 'annotate'} onClick={(event: any) => toggleGroupedToolOrPointer(event, 'annotate', selectedToolbarButton === 'annotate')} icon={PenLine} title="Annotate" />
      <ToolBtn compact active={selectedToolbarButton === 'erase'} onClick={() => toggleToolOrPointer('erase')} icon={Eraser} title="Eraser" />
      <ToolBtn compact active={selectedToolbarButton === 'capture-area'} disabled={areaCaptureDisabled} onClick={() => {
        toggleToolOrPointer('capture-area');
      }} icon={Crop} title={areaCaptureDisabled ? 'Area Capture disabled in CT/DICOM view' : 'Area Capture'} />
      <ToolBtn compact active={false} onClick={() => { void handleCapture(); }} icon={Camera} title="Full Capture" />
      <ToolBtn compact active={false} onClick={() => { if (window.confirm('Clear all?')) clearAllAnnotations(); }} icon={Trash2} title="Clear All" />
      <button
        onClick={resetView}
        className={getToolbarButtonClass({ compact: true })}
        title="Reset All Views"
      >
        <RotateCcw className="h-5 w-5" />
      </button>
      <ToolBtn
        compact
        active={selectedToolbarButton === 'auto-window'}
        onClick={handleApplyAutoWindow}
        icon={Sliders}
        title="Apply Auto Window"
      />
      <button
        onClick={() => { setViewMode(viewMode === 'overlay' ? 'original' : 'overlay'); setSelectedToolbarButton('view-toggle'); }}
        className={getToolbarButtonClass({ active: selectedToolbarButton === 'view-toggle', compact: true })}
        title={viewMode === 'overlay' ? 'Switch to Original Source' : 'Switch to AI Analysis'}
      >
        {viewMode === 'overlay' ? <Layers className="h-5 w-5" /> : <ImageIcon className="h-5 w-5" />}
      </button>
      <button
        onClick={() => { setViewMode('heatmap'); setSelectedToolbarButton('heatmap'); }}
        className={getToolbarButtonClass({ active: selectedToolbarButton === 'heatmap', compact: true })}
        title="Risk Overlay"
      >
        <Activity className="h-5 w-5" />
      </button>
    </>
  );

  const toolbarActions = (
    <div className="flex flex-col items-start gap-2" style={{ marginLeft: 100 }}>
      <div className="flex flex-nowrap items-center gap-2">
        {toolbarPrimaryButtons}
        {toolbarSecondaryButtons}
      </div>
    </div>
  );

  return (
    <div className="h-screen w-full text-gray-100 flex flex-col overflow-hidden font-sans" style={{ backgroundColor: '#06071A' }}>
      <TopHeader actions={toolbarActions} />
      {!hasViewerShell && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-3 text-gray-300">
            <p className="text-lg font-semibold">표시할 분석 화면이 아직 준비되지 않았습니다.</p>
            <p className="text-sm text-gray-400">업로드 또는 분석이 완료되면 뷰어와 차트가 이 화면에 표시됩니다.</p>
          </div>
        </div>
      )}
      {hasViewerShell && (
        <div className="flex flex-1 relative overflow-y-auto">
          <aside
            className={`flex-shrink-0 overflow-hidden border-r border-gray-800/80 transition-all duration-300 ${workspaceOpen ? 'w-[84px] min-w-[84px] max-w-[84px]' : 'w-[42px] min-w-[42px] max-w-[42px]'}`}
            style={{ backgroundColor: '#050816' }}
          >
            <div className={`flex h-full flex-col py-4 ${workspaceOpen ? 'px-3' : 'px-1.5'}`}>
              <div className={`flex items-center ${workspaceOpen ? 'justify-between' : 'justify-center'} gap-2`}>
                {workspaceOpen && (
                  <div>
                    <p className="text-[10px] font-black uppercase tracking-[0.32em] text-cyan-300/80">Workspace</p>
                  </div>
                )}
                <button
                  onClick={() => setWorkspaceOpen((prev) => !prev)}
                  className={`flex items-center justify-center border border-cyan-300/20 bg-cyan-400/10 text-cyan-100 transition hover:border-cyan-300/40 hover:bg-cyan-400/20 ${workspaceOpen ? 'h-10 w-10 rounded-2xl' : 'h-8 w-8 rounded-xl'}`}
                  title={workspaceOpen ? 'Collapse workspace' : 'Expand workspace'}
                >
                  <span className="text-lg font-bold leading-none">{workspaceOpen ? '‹' : '›'}</span>
                </button>
              </div>

              {workspaceOpen ? (
                <div className="mt-4 flex min-h-0 flex-1 flex-col gap-3">
                  <div className="space-y-2 rounded-[24px] border border-white/8 bg-black/35 p-2">
                    {[
                      { id: 'studies' as const, label: 'Studies', icon: ClipboardList },
                      { id: 'report' as const, label: 'Report', icon: FileText },
                    ].map((item) => (
                      <button
                        key={item.id}
                        onClick={() => setWorkspaceSection(item.id)}
                        className={`flex w-full items-center gap-2 rounded-2xl px-3 py-3 text-left text-sm font-semibold transition ${
                          workspaceSection === item.id
                            ? 'bg-cyan-400 text-slate-950 shadow-[0_12px_24px_rgba(34,211,238,0.18)]'
                            : 'bg-transparent text-slate-300 hover:bg-white/8 hover:text-white'
                        }`}
                      >
                        <item.icon className="h-4 w-4" />
                        <span>{item.label}</span>
                      </button>
                    ))}
                  </div>
                  <div className="min-h-0 flex-1 overflow-hidden rounded-[28px] border border-dashed border-white/8 bg-[linear-gradient(180deg,rgba(15,23,42,0.55),rgba(8,12,28,0.82))]">
                    <div className="flex h-full flex-row">
                      <div className="flex-1 min-w-0">
                        <StudiesWorkspacePanel
                          studies={combinedStudies as any}
                          selectedSeriesId={selectedFolderSeriesId}
                          isVisible={workspaceSection === 'studies'}
                          onSelectSeries={async (seriesId) => {
                            let nextSeries = findFolderSeriesById(seriesId);
                            if (!nextSeries) {
                              const targetStudy = serverStudies.find(s => s.series.some((ser: any) => ser.id === seriesId));
                              if (targetStudy) {
                                setIsProcessing(true);
                                try {
                                  const materialized = await materializeServerStudy(targetStudy);
                                  setActiveFolderStudies(prev => {
                                    if (prev.some(s => s.id === materialized.id)) return prev;
                                    return [...prev, materialized];
                                  });
                                  nextSeries = materialized.series.find((s: any) => s.id === seriesId) || null;
                                } catch (e) {
                                  console.error('Failed to materialize study inside ChartPage', e);
                                } finally {
                                  setIsProcessing(false);
                                }
                              }
                            }
                            if (nextSeries) {
                              setSelectedFolderSeriesId(seriesId);
                              setWorkspaceSection('studies');
                              setViewMode('original');
                              setViewerMode(nextSeries?.volumeEligible ? 'grid' : 'single');
                            }
                          }}
                        />
                      </div>
                      {captureGallery.length > 0 && (
                        <div className="w-10 shrink-0 border-l border-white/5 bg-black/10 flex flex-col items-center py-2 gap-2 overflow-y-auto">
                          {captureGallery.map(capture => (
                            <div 
                              key={capture.id}
                              className="w-7 h-7 rounded-md border border-white/10 overflow-hidden bg-black shrink-0 shadow-sm"
                              title={capture.createdAt}
                            >
                              <img src={capture.dataUrl} className="w-full h-full object-cover" />
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                  {workspaceSection === 'report' && (
                    <div className="rounded-[20px] border border-white/8 bg-white/5 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-200">
                      Drafting Report
                    </div>
                  )}
                </div>
              ) : (
                <div className="mt-4 flex flex-1 flex-col items-center gap-2">
                  {[
                    { id: 'studies' as const, icon: ClipboardList, label: 'Studies' },
                    { id: 'report' as const, icon: FileText, label: 'Report' },
                  ].map((item) => (
                    <button
                      key={item.id}
                      onClick={() => {
                        setWorkspaceSection(item.id);
                        setWorkspaceOpen(true);
                      }}
                      className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/8 bg-white/5 text-slate-300 transition hover:border-cyan-300/30 hover:bg-white/8 hover:text-white"
                      title={item.label}
                    >
                      <item.icon className="h-4 w-4" />
                    </button>
                  ))}
                </div>
              )}
            </div>
          </aside>

          {/* Center: report-style stacked layout on dark background */}
          <div className="min-w-0 flex-1 overflow-visible text-gray-100" style={{ backgroundColor: '#06071A' }}>
            <div className="max-w-7xl mx-auto py-6 px-4 lg:px-8 space-y-6">
              {workspaceSection === 'report' ? (
                <div className="bg-[#0f0f0f] border border-white/5 rounded-3xl shadow-2xl overflow-hidden" style={{ minHeight: '80vh' }}>
                  {reportSessionId ? (
                    <iframe 
                      src={`/api/web_report/session/${reportSessionId}/report`}
                      className="w-full h-full border-0"
                      style={{ height: '80vh' }}
                      title="Report Preview"
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full gap-5 p-20 text-center" style={{ minHeight: '80vh' }}>
                      <div className="h-20 w-20 rounded-full bg-cyan-400/10 flex items-center justify-center border border-cyan-400/20 shadow-xl">
                        <FileText className="h-10 w-10 text-cyan-400" />
                      </div>
                      <div className="space-y-2">
                        <h3 className="text-2xl font-bold text-white tracking-tight">Report Draft Not Found</h3>
                        <p className="max-w-md mx-auto text-base text-slate-400 leading-relaxed">
                          Please click the <span className="text-cyan-400 font-semibold px-2 py-0.5 rounded-md bg-cyan-400/10 border border-cyan-400/20 shadow-sm mx-1">Start AI Report</span> button in the bottom right corner to generate your first findings.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <>
                  {/* Hero card mimicking report_v2 */}
                  <div className="bg-[#0f0f0f] border border-white/5 rounded-3xl shadow-2xl overflow-hidden">
                <div className="flex flex-col lg:flex-row lg:items-start gap-6">
                  {/* Viewer column */}
                    <div className="w-full bg-black relative select-none">
                    {captureNotice?.type === 'success' && (
                      <div
                        className="pointer-events-none"
                        style={{
                          position: 'absolute',
                          inset: 0,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          zIndex: 20000,
                        }}
                      >
                        <div
                          className="flex max-w-[220px] items-start gap-2 rounded-xl px-3 py-2 text-white backdrop-blur-sm"
                          style={{
                            backgroundColor: 'rgba(115, 115, 115, 0.5)',
                            border: '1px solid rgba(255,255,255,0.1)',
                            boxShadow: '0 12px 28px rgba(0,0,0,0.24)',
                          }}
                        >
                          <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-white/10">
                            <Camera className="h-3 w-3 text-white/95" />
                          </div>
                          <div className="min-w-0">
                            <p className="font-semibold leading-none text-white" style={{ fontSize: 10 }}>
                              Copied
                            </p>
                            <p className="mt-0.5 break-words text-white/75" style={{ fontSize: 10, lineHeight: '14px' }}>
                              {captureNotice.text}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                    <div
                      className="absolute flex items-center gap-3 bg-black/60 backdrop-blur-xl px-5 py-2.5 rounded-full border border-white/10 shadow-2xl pointer-events-none"
                      style={shouldCenterModeBadge
                        ? { top: 16, left: '50%', transform: 'translateX(-50%)', zIndex: 70 }
                        : { top: 16, left: 16, zIndex: 70 }}
                    >
                      <div className={`w-2.5 h-2.5 rounded-full ${viewModeDotClass}`} />
                      <span className="text-xs font-bold uppercase tracking-wider text-gray-300">
                        {viewModeLabel}
                      </span>
                    </div>
                      {/* Magnifier debug panel kept for local troubleshooting; hidden in normal UI. */}

                    <div
                              className={`w-full h-full relative overflow-hidden ${shouldUseCornerstone ? '' : activeTool === 'magnifier' ? 'cursor-none' : activeTool === 'pan' ? 'cursor-grab active:cursor-grabbing' : 'cursor-crosshair'}`}
                      style={{ minHeight: `${shouldUseCornerstone ? containerHeight : viewerHeight}px`, height: `${shouldUseCornerstone ? containerHeight : viewerHeight}px` }}
                      ref={viewerRef}
                      onMouseDown={handleMouseDown}
                      onMouseMove={handleMouseMove}
                      onMouseUp={handleMouseUp}
                      onMouseLeave={handleMouseLeave}
                      onWheel={shouldUseCornerstone ? undefined : handleWheel}
                      onContextMenu={shouldUseCornerstone ? undefined : handleContextMenu}
                    >
                      {shouldShowOverlayPresetSelector && (
                        <div
                          className="pointer-events-auto"
                          style={{ position: 'absolute', top: 16, right: 16, zIndex: 90 }}
                        >
                          <select
                            value={overlayPreset}
                            onChange={(event) => setOverlayPreset(event.target.value as typeof overlayPreset)}
                            className="w-auto min-w-[240px] appearance-none rounded-xl border border-white/10 bg-black/60 px-3 py-2 text-xs font-semibold text-white shadow-2xl outline-none backdrop-blur-xl"
                            style={{
                              backgroundColor: 'rgba(0, 0, 0, 0.82)',
                              color: '#f8fafc',
                              colorScheme: 'dark',
                              WebkitTextFillColor: '#f8fafc',
                            }}
                            title="Select AI overlay preset"
                          >
                            <option value="all" style={{ backgroundColor: '#0b1120', color: '#f8fafc' }}>1. 전체</option>
                            <option value="sinus" style={{ backgroundColor: '#0b1120', color: '#f8fafc' }}>2. sinus</option>
                            <option value="nerve" style={{ backgroundColor: '#0b1120', color: '#f8fafc' }}>3. nerve</option>
                            <option value="tooth" style={{ backgroundColor: '#0b1120', color: '#f8fafc' }}>4. tooth</option>
                            <option value="sinus-upper-tooth" style={{ backgroundColor: '#0b1120', color: '#f8fafc' }}>5. sinus + upper tooth</option>
                            <option value="nerve-lower-tooth" style={{ backgroundColor: '#0b1120', color: '#f8fafc' }}>6. nerve + lower tooth</option>
                          </select>
                        </div>
                      )}
                      {shouldShowFloatingDicomHud && (
                        <DicomMetadataOverlay
                          metadata={displayDicomHudMetadata}
                          top={16}
                          bottom={16}
                          left={16}
                          right={16}
                        />
                      )}
                      {/* Area Capture Selection Overlay */}
                      {captureRect && (
                        <div
                          className="absolute border-2 border-dashed border-indigo-400 bg-indigo-500/10 z-[10000] pointer-events-none"
                          style={{
                            left: captureRect.w > 0 ? captureRect.x : captureRect.x + captureRect.w,
                            top: captureRect.h > 0 ? captureRect.y : captureRect.y + captureRect.h,
                            width: Math.abs(captureRect.w),
                            height: Math.abs(captureRect.h),
                          }}
                        >
                          {!captureRect.active && Math.abs(captureRect.w) > 5 && (
                            <div className="absolute -bottom-10 left-1/2 -translate-x-1/2 flex gap-1 pointer-events-auto">
                              <button
                                onClick={(e) => { e.stopPropagation(); void handleCapture(captureRect); }}
                                className="bg-indigo-600 text-white text-[10px] px-3 py-1.5 rounded-full font-bold shadow-lg flex items-center gap-1 hover:bg-indigo-700 whitespace-nowrap"
                              >
                                <Camera size={12} /> {shouldUseCornerstone && viewerMode !== 'grid' ? 'Copy' : 'Capture'}
                              </button>
                              <button
                                onClick={(e) => { e.stopPropagation(); setCaptureRect(null); }}
                                className="bg-gray-800 text-white text-[10px] px-3 py-1.5 rounded-full font-bold shadow-lg hover:bg-gray-700"
                              >
                                Cancel
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                      {viewerMode === 'grid' && shouldUseCornerstone ? (
                        <CornerstoneGridViewer
                          sources={cornerstoneSources}
                          title={viewMode === 'original' ? 'Original MPR' : 'Overlay MPR'}
                          maxHeight={containerHeight}
                          showToolbar={false} // Use ChartPage's sidebar instead
                          layout={gridLayout}
                          onLayoutChange={setGridLayout}
                          interactionMode={activeTool}
                          resetToken={cornerstoneResetToken}
                          autoWindowToken={cornerstoneAutoWindowToken}
                          onViewportCapture={handleGridViewportCapture}
                          invert={inverted}
                          brightness={brightness}
                          contrast={contrast}
                          rotation={rotation}
                          flipped={flipped}
                          assignedCaptureSlots={assignedCaptureSlots}
                          onAssignCaptureToViewport={assignCaptureToViewport}
                          onClearAssignedCapture={clearAssignedCapture}
                        />
                      ) : shouldUseCornerstone ? (
                        <div className="w-full h-full relative flex flex-col">
                          {!isVolumeCase && cornerstoneSources[0]?.file ? (
                            <MinimalCornerstoneDicomViewer
                              key={cornerstoneSources[0]?.id || 'minimal-cs'}
                              file={cornerstoneSources[0].file}
                              title={viewMode === 'original' ? 'Original DICOM' : 'Overlay DICOM'}
                              maxHeight={containerHeight}
                            />
                          ) : (
                            <CornerstoneViewer
                              title={viewMode === 'original' ? 'Original DICOM' : 'Overlay DICOM'}
                              sources={cornerstoneSources}
                              initialSourceId={cornerstoneSources[0]?.id}
                              maxHeight={containerHeight}
                              showToolbar={false}
                              interactionMode={activeTool}
                              resetToken={cornerstoneResetToken}
                              autoWindowToken={cornerstoneAutoWindowToken}
                              invert={inverted}
                              brightness={brightness}
                              contrast={contrast}
                              rotation={rotation}
                              flipped={flipped}
                            />
                          )}
                          {!isVolumeCase && dicomHudMetadata && (
                            <DicomMetadataOverlay
                              metadata={displayDicomHudMetadata}
                              top={16}
                              bottom={16}
                              left={16}
                              right={16}
                            />
                          )}
                          {result?.is_volume && (
                            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 bg-indigo-600/80 backdrop-blur-md px-4 py-2 rounded-full border border-indigo-400/50 shadow-lg flex items-center gap-2 pointer-events-none">
                              <Rotate3d className="w-4 h-4 text-white animate-pulse" />
                              <span className="text-[11px] font-bold text-white whitespace-nowrap">
                                3D Volume detected. Use 'Grid/MPR View' for cross-sections.
                              </span>
                            </div>
                          )}
                        </div>
                      ) : showSrc ? (
                        <div className="relative w-full h-full flex items-center justify-center overflow-hidden" ref={containerRef}>
                          <div
                            className={`relative transform-gpu will-change-transform shadow-2xl ring-1 ring-black/5 mx-auto ${activeTool === 'magnifier' ? 'cursor-none' : activeTool === 'pan' ? 'cursor-grab active:cursor-grabbing' : 'cursor-crosshair'}`}
                            style={{
                              width: displaySize.width,
                              height: displaySize.height,
                              isolation: 'isolate',
                              zIndex: 10,
                              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale * zoom}) rotate(${rotation}deg) scaleX(${flipped ? -1 : 1})`,
                              transition: activeTool === 'pan' ? 'none' : 'transform 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                              transformOrigin: 'center',
                              cursor: activeTool === 'magnifier' ? 'none' : activeTool === 'pan' ? 'grab' : 'crosshair',
                            }}
                          >
                            <img
                              src={showSrc}
                              alt="Analysis Result"
                              className="block w-full h-full select-none pointer-events-none"
                              style={{
                                position: 'relative',
                                zIndex: 1,
                                filter: `invert(${inverted ? 1 : 0}) brightness(${brightness}%) contrast(${contrast}%)`,
                              }}
                              draggable={false}
                              ref={imageRef}
                              onLoad={(e) => {
                                const img = e.currentTarget;
                                const w = img.naturalWidth || 1;
                                const h = img.naturalHeight || 1;
                                setDimensions({ width: w, height: h });
                                buildEdgeMapFromImage(img);
                                requestAnimationFrame(() => fitImageToViewer());
                              }}
                            />
                            <svg
                              ref={rasterOverlaySvgRef}
                              className="absolute inset-0 w-full h-full"
                              viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                              style={{
                                position: 'absolute',
                                inset: 0,
                                zIndex: 20,
                                overflow: 'visible',
                                pointerEvents: 'none',
                              }}
                            >
                              <defs>
                                <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
                                  <polygon points="0 0, 6 2, 0 4" fill="#facc15" />
                                </marker>
                                <marker id="display-arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
                                  <polygon points="0 0, 6 2, 0 4" fill="#facc15" />
                                </marker>
                                <marker id="display-arrowhead-white" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
                                  <polygon points="0 0, 6 2, 0 4" fill="#ffffff" />
                                </marker>
                                <filter id="riskBlurStrong" x="-50%" y="-50%" width="200%" height="200%">
                                  <feGaussianBlur stdDeviation={32 / effectiveScale} />
                                </filter>
                                <filter id="riskBlurSoft" x="-40%" y="-40%" width="180%" height="180%">
                                  <feGaussianBlur stdDeviation={18 / effectiveScale} />
                                </filter>
                                <radialGradient id="cariesGlow" cx="50%" cy="50%" r="50%">
                                  <stop offset="0%" stopColor="rgba(255, 255, 255, 0.9)" />
                                  <stop offset="30%" stopColor="rgba(255, 180, 0, 0.85)" />
                                  <stop offset="100%" stopColor="rgba(255, 100, 0, 0)" />
                                </radialGradient>
                                <radialGradient id="periGlow" cx="50%" cy="50%" r="50%">
                                  <stop offset="0%" stopColor="rgba(255, 255, 255, 0.95)" />
                                  <stop offset="30%" stopColor="rgba(255, 60, 60, 0.88)" />
                                  <stop offset="100%" stopColor="rgba(200, 0, 0, 0)" />
                                </radialGradient>
                                <radialGradient id="boneLossGlow" cx="50%" cy="50%" r="50%">
                                  <stop offset="0%" stopColor="rgba(255, 255, 255, 0.4)" />
                                  <stop offset="100%" stopColor="rgba(255, 180, 71, 0)" />
                                </radialGradient>
                                <radialGradient id="magnifierGloss" cx="35%" cy="30%" r="70%">
                                  <stop offset="0%" stopColor="rgba(255,255,255,0.9)" />
                                  <stop offset="35%" stopColor="rgba(255,255,255,0.18)" />
                                  <stop offset="100%" stopColor="rgba(255,255,255,0)" />
                                </radialGradient>
                              </defs>
                              {renderAIDetections()}
                              {renderRiskDetections()}
                              {shapes.map((s: any, idx: number) => renderShape(s, false, idx))}
                              {activeSubTool && renderShape({ type: activeSubTool, points: pendingPoints, color: drawingColor }, true)}
                            </svg>
                          </div>
                        </div>
                      ) : (
                        <div className="w-full h-full flex flex-col items-center justify-center gap-3 bg-gradient-to-b from-[#0c0c0c] to-black">
                          {isProcessing ? (
                            <>
                              <div className="flex items-center gap-3">
                                <Loader2 className="animate-spin text-indigo-400" />
                                <span className="text-sm text-gray-400">Analyzing image...</span>
                              </div>
                              <div className="w-48 h-2 bg-gray-800 rounded-full overflow-hidden">
                                <div className="h-full bg-indigo-500 transition-all duration-300" style={{ width: `${loadingProgress}%` }} />
                              </div>
                            </>
                          ) : (
                            <div className="text-gray-500 text-sm">Upload an image to begin</div>
                          )}
                        </div>
                      )}
                      {renderMagnifier()}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mb-4 rounded-2xl border shadow-sm" style={{ backgroundColor: '#0B0D26', borderColor: 'rgba(255,255,255,0.10)' }}>
                <div className="p-6">
                  <div className="space-y-6">
                    <div className="relative flex flex-col items-center gap-4 xl:pt-1">
                      <div className="min-w-0 flex justify-center">
                        <DentalChartLegend
                          items={dentalLegendItems}
                          activeKey={activeLegendFilter}
                          onToggle={(key) => setActiveLegendFilter((prev) => (prev === key ? null : key))}
                        />
                      </div>
                      <div className="flex justify-center">
                        {/* <span className="text-sm font-semibold tracking-[0.08em] text-[#CBD5E1]">
                          Odontogram
                        </span> */}
                      </div>
                    </div>

                    <section className="rounded-2xl border px-6 py-5 shadow-sm" style={{ backgroundColor: '#090B20', borderColor: 'rgba(255,255,255,0.10)' }}>
                      <div className="mb-5 flex items-start justify-between gap-4">
                        <h3 className="text-[30px] font-bold tracking-[-0.03em] text-white">Dental Chart</h3>
                        <button
                          onClick={() => setNumberingSystem(prev => prev === 'fdi' ? 'univ' : 'fdi')}
                          className="inline-flex shrink-0 items-center gap-2 whitespace-nowrap px-1 py-1 text-xs font-semibold leading-none transition-opacity hover:opacity-80"
                          style={{
                            display: 'inline-flex',
                            flexDirection: 'row',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: '10px',
                            whiteSpace: 'nowrap',
                          }}
                          title="Toggle numbering system"
                        >
                          <div
                            className="pointer-events-none whitespace-nowrap"
                            style={{ color: numberingSystem === 'fdi' ? '#FFFFFF' : '#475569' }}
                          >
                            FDI
                          </div>
                          <div
                            className="pointer-events-none whitespace-nowrap"
                            style={{ color: '#64748B' }}
                          >
                            /
                          </div>
                          <div
                            className="pointer-events-none whitespace-nowrap"
                            style={{ color: numberingSystem === 'univ' ? '#FFFFFF' : '#475569' }}
                          >
                            Univ
                          </div>
                        </button>
                      </div>
                      <div className="flex flex-col items-center">
                        <BottomTeethChart
                          onToothClick={(id) => setSelectedTooth((prev) => (prev === id ? undefined : id))}
                          selectedTooth={selectedTooth}
                          statuses={statuses}
                          highlightRing={chartHighlights}
                          extraction={extractionCandidates}
                          implantSites={implantSiteCandidates}
                          numberingSystem={numberingSystem}
                          tooltipData={odontogramTooltipData}
                          activeLegendFilter={activeLegendFilter}
                        />
                      </div>
                    </section>
                  </div>
                </div>
              </div>
              </>
              )}
            </div>
          </div>

          {/* Right Panel (Drill-down details)
          {selectedTooth && (
            <div className="absolute top-0 right-0 h-full z-50 shadow-xl border-l border-gray-100">
              <RightPanel
                selectedTooth={selectedTooth}
                result={result}
                statuses={statuses}
                onClose={() => setSelectedTooth(undefined)}
                numberingSystem={numberingSystem}
              />
            </div>
          )}
          */}

          {/* Floating Context Menu */}
          {contextMenu.show && createPortal(
            <>
              <div className="fixed inset-0 z-[999]" onClick={() => setContextMenu({ ...contextMenu, show: false })} />
              <div
                className="fixed z-[1000] w-[220px] rounded-2xl border border-white/10 p-1 flex flex-col gap-1 backdrop-blur-md isolate shadow-[0_24px_60px_rgba(0,0,0,0.42)]"
                style={{
                  position: 'fixed',
                  top: `${contextMenu.y}px`,
                  left: `${contextMenu.x + 12}px`,
                  background: 'linear-gradient(180deg, rgba(10,16,34,0.98), rgba(8,12,28,0.96))',
                  color: '#e5eefc',
                }}
              >
                {contextMenu.menu === 'measure' ? (
                  <>
                    <CtxBtn onClick={() => selectSubTool('length')} label="Length" />
                    <CtxBtn onClick={() => selectSubTool('bidirectional')} label="Bidirectional" />
                    <CtxBtn onClick={() => selectSubTool('angle')} label="Angle" />
                  </>
                ) : contextMenu.menu === 'annotate' ? (
                  <>
                    <CtxBtn onClick={() => selectSubTool('text')} label="Annotation" />
                    <CtxBtn onClick={() => selectSubTool('arrow')} label="Arrow" />
                    <CtxBtn onClick={() => selectSubTool('ellipse')} label="Ellipse" />
                    <CtxBtn onClick={() => selectSubTool('rect')} label="Rectangle" />
                    <CtxBtn onClick={() => selectSubTool('circle')} label="Circle" />
                    <CtxBtn onClick={() => selectSubTool('roi-free')} label="Freehand ROI" />
                    <CtxBtn onClick={() => selectSubTool('spline-roi')} label="Spline ROI" />
                    <CtxBtn onClick={() => selectSubTool('livewire')} label="Livewire Tool" />
                  </>
                ) : (
                  <div className="min-w-[220px] space-y-4 p-4">
                    <div className="mb-2 flex items-center justify-between border-b border-white/10 pb-2">
                      <span className="text-[11px] font-bold uppercase tracking-wider text-slate-100">Grid Layout</span>
                      <span className="text-[10px] font-black text-cyan-300">{tempGridLayout.rows} x {tempGridLayout.cols}</span>
                    </div>
                    <div className="flex gap-4">
                      <div className="flex-1 space-y-1.5">
                        <label className="text-[10px] font-bold uppercase tracking-tight text-slate-400">Rows</label>
                        <input
                          type="number"
                          min="1"
                          max="8"
                          value={tempGridLayout.rows}
                          onChange={(e) => {
                            const val = Math.max(1, Math.min(8, parseInt(e.target.value) || 1));
                            setTempGridLayout(prev => ({ ...prev, rows: val }));
                          }}
                          className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-slate-100 outline-none transition-all focus:border-cyan-400/70 focus:ring-2 focus:ring-cyan-400/30"
                        />
                      </div>
                      <div className="flex-1 space-y-1.5">
                        <label className="text-[10px] font-bold uppercase tracking-tight text-slate-400">Cols</label>
                        <input
                          type="number"
                          min="1"
                          max="8"
                          value={tempGridLayout.cols}
                          onChange={(e) => {
                            const val = Math.max(1, Math.min(8, parseInt(e.target.value) || 1));
                            setTempGridLayout(prev => ({ ...prev, cols: val }));
                          }}
                          className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-slate-100 outline-none transition-all focus:border-cyan-400/70 focus:ring-2 focus:ring-cyan-400/30"
                        />
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        setGridLayout(tempGridLayout);
                        setViewerMode('grid');
                        setContextMenu({ ...contextMenu, show: false });
                      }}
                      className="w-full rounded-xl border border-cyan-300/30 bg-cyan-400 px-3 py-2.5 text-[10px] font-bold uppercase tracking-widest text-slate-950 shadow-[0_12px_24px_rgba(34,211,238,0.18)] transition-colors hover:bg-cyan-300"
                    >
                      Apply & Close
                    </button>
                  </div>
                )}
              </div>
            </>,
            document.body
          )}

        </div>
      )}
      {reportSessionId && createPortal(
        <WebReportDrawer
          sessionId={reportSessionId}
          selectedToothId={selectedTooth ? String(selectedTooth) : null}
          availableCaptures={captureGallery}
          onClose={() => setReportDrawerOpen(false)}
          open={reportDrawerOpen}
          layout="dock"
          isInactive={workspaceSection === 'report'}
        />,
        document.body
      )}
      {reportError && (
        <div
          className="fixed rounded-2xl border border-red-500/30 bg-[rgba(60,10,16,0.92)] px-3 py-2 text-xs text-red-100 shadow-[0_18px_36px_rgba(0,0,0,0.35)] backdrop-blur-xl"
          style={{ right: 24, bottom: 128, zIndex: 2147483001, width: 280, maxWidth: 'calc(100vw - 2rem)' }}
        >
          {reportError}
        </div>
      )}
      <button
        type="button"
        disabled={workspaceSection === 'report'}
        className={`cursor-pointer bg-cyan-400 text-slate-950 transition hover:bg-cyan-300 ${workspaceSection === 'report' ? 'opacity-40 pointer-events-none' : ''}`}
        style={{
          position: 'fixed',
          right: 24,
          bottom: 24,
          zIndex: 2147483002,
          pointerEvents: workspaceSection === 'report' ? 'none' : 'auto',
          opacity: workspaceSection === 'report' ? 0.4 : (hasViewerShell ? 1 : 0.7),
          width: 80,
          height: 80,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          gap: 4,
          overflow: 'visible',
          borderRadius: '9999px',
          // border: '4px solid #ef4444',
          boxShadow: '0 18px 40px rgba(0,0,0,0.32)',
        }}
        onPointerDown={(event) => {
          event.stopPropagation();
        }}
        onMouseDown={(event) => {
          event.stopPropagation();
        }}
        onClick={(event) => {
          event.stopPropagation();
          void handleStartReport();
        }}
        aria-label="Open AI note report panel"
        title={reportStartState === 'creating' ? 'Loading report workspace' : reportSessionId ? 'Open report draft' : 'Open report workspace'}
        >
          <ClipboardList size={34} />
          {reportStartState === 'creating' && (
            <span
              className="pointer-events-none absolute -right-1 -top-1 flex h-5 min-w-5 items-center justify-center rounded-full border border-white bg-cyan-400 px-1 text-[9px] font-bold text-slate-950 shadow-[0_6px_16px_rgba(34,211,238,0.28)]"
              style={{ position: 'absolute' }}
            >
              ...
            </span>
          )}
        </button>
    </div>
  );
}

function getToolbarButtonClass({ active = false, disabled = false, compact = false }: { active?: boolean; disabled?: boolean; compact?: boolean }) {
  const sizeClass = compact ? 'h-9 w-9 rounded-xl' : 'w-full min-h-[44px] rounded-lg p-2';
  if (disabled) {
    return `${sizeClass} flex items-center justify-center border border-white/8 bg-white/5 text-gray-600 opacity-40 cursor-not-allowed transition-all`;
  }
  if (active) {
    return `${sizeClass} flex items-center justify-center border border-white/40 bg-[#0b1120] text-white ring-1 ring-white/25 shadow-[0_12px_28px_rgba(15,23,42,0.32)] transition-all`;
  }
  return `${sizeClass} flex items-center justify-center border border-slate-700 bg-[#0b1120] text-gray-600 transition-all hover:border-slate-500 hover:bg-[#111827] hover:text-gray-400`;
}

function ToolBtn({ active, onClick, icon: Icon, title, disabled = false, compact = false }: any) {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      title={title}
      disabled={disabled}
      aria-pressed={active}
      className={getToolbarButtonClass({ active, disabled, compact })}
    >
      <Icon size={20} />
    </button>
  );
}

function ToolRow({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-stretch gap-2 [&>*]:flex-1">
      {children}
    </div>
  );
}

function WindowLevelIcon({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="1.5" y="1.5" width="21" height="21" rx="5" fill="currentColor" opacity="0.18" />
      <circle cx="12" cy="12" r="6.5" stroke="currentColor" strokeWidth="1.8" />
      <path d="M12 5.5a6.5 6.5 0 0 1 0 13Z" fill="currentColor" opacity="0.95" />
      <path d="M7 17 17 7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function InvertIcon({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="20" height="20" rx="5" stroke="currentColor" strokeWidth="1.8" />
      <path d="M12 2v20" stroke="currentColor" strokeWidth="1.8" />
      <path d="M12 6a6 6 0 1 0 0 12Z" fill="currentColor" opacity="0.92" />
    </svg>
  );
}

function CtxBtn({ onClick, label }: any) {
  return (
    <button onClick={onClick} className="block w-full text-left px-3 py-2 text-xs text-gray-700 bg-white hover:bg-indigo-50 hover:text-indigo-700 rounded transition-colors">{label}</button>
  );
}

function DentalChartLegend({
  items,
  activeKey,
  onToggle,
}: {
  items: Array<{ key: string; label: string; color: string; dashed: boolean; gap: number }>;
  activeKey: string | null;
  onToggle: (key: 'triage-3' | 'triage-2' | 'triage-1' | 'implant' | 'missing') => void;
}) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-3 text-center text-[14px] font-semibold xl:max-w-none">
      {items.map((item) => (
        <button
          key={item.label}
          type="button"
          onClick={() => onToggle(item.key as 'triage-3' | 'triage-2' | 'triage-1' | 'implant' | 'missing')}
          className="inline-flex items-center justify-center gap-3 whitespace-nowrap rounded-full px-2 py-1 transition-all"
          style={{
            marginRight: item.gap,
            opacity: activeKey && activeKey !== item.key ? 0.35 : 1,
            filter: activeKey && activeKey !== item.key ? 'saturate(0.35)' : 'none',
            boxShadow: activeKey === item.key ? '0 0 0 1px rgba(255,255,255,0.2) inset, 0 8px 18px rgba(15,23,42,0.16)' : 'none',
          }}
        >
          <span
            className="block w-7"
            style={{
              borderTop: `3px ${item.dashed ? 'dashed' : 'solid'} ${item.color}`,
            }}
          />
          <span className="whitespace-nowrap" style={{ color: item.color }}>{item.label}</span>
        </button>
      ))}
    </div>
  );
}



import { useLocation } from 'react-router-dom';
import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { BottomTeethChart } from '../components/BottomTeethChart';
import {
  AlertTriangle, Activity, Zap, Layers, Image as ImageIcon,
  MousePointer, Hand, ZoomIn, RotateCw, FlipHorizontal,
  Ruler, PenLine, Loader2, ZoomOut, RotateCcw, AlertCircle, Skull,
  ClipboardList, Quote, Sparkles, Grid
} from 'lucide-react';
import { RightPanel } from '../components/RightPanel';
import { TopHeader } from '../components/TopHeader';

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

export function ChartPage(props?: ChartPageProps) {
  const location = useLocation();
  const [result, setResult] = useState<any>(props?.result ?? (location.state as any)?.result);
  const [isProcessing, setIsProcessing] = useState(!result && !!(location.state as any)?.jobId);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [timestamp, setTimestamp] = useState(Date.now());

  const [selectedTooth, setSelectedTooth] = useState<number | undefined>(undefined);
  const [viewMode, setViewMode] = useState<'overlay' | 'original'>('overlay');
  const [containerHeight] = useState(560);
  const [numberingSystem, setNumberingSystem] = useState<'fdi' | 'univ'>('fdi'); // [NEW]

  // Tools
  const [activeTool, setActiveTool] = useState<string>('pointer');
  const [activeSubTool, setActiveSubTool] = useState<string | null>(null);

  // Viewport State
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [rotation, setRotation] = useState(0);
  const [flipped, setFlipped] = useState(false);

  useEffect(() => {
    let timer: any;
    const jobId = (location.state as any)?.jobId;

    if (jobId && !result) {
      setIsProcessing(true);
      timer = setInterval(async () => {
        try {
          // Increment progress slightly while waiting
          setLoadingProgress(p => Math.min(p + 5, 95));

          const res = await fetch(`/api/detect/status/${jobId}`);
          const data = await res.json();
          if (data.success && data.status === 'done' && data.result) {
            clearInterval(timer);
            setResult(data.result);
            setTimestamp(Date.now());
            setIsProcessing(false);
            setLoadingProgress(100);
            setViewMode('overlay');
          } else if (data.status === 'failed') {
            clearInterval(timer);
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
  }, [location.state, result]);
  // Interaction State
  const dragRef = useRef<{ active: boolean; mode: 'pan' | 'wl'; startX: number; startY: number }>({
    active: false, mode: 'pan', startX: 0, startY: 0
  });
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const edgeMapRef = useRef<Float32Array | null>(null);
  const [imgRect, setImgRect] = useState<DOMRect | null>(null);
  const [fitScale, setFitScale] = useState(1);


  // Drawing State
  const [shapes, setShapes] = useState<any[]>([]);
  const [pendingPoints, setPendingPoints] = useState<{ x: number; y: number }[]>([]);
  const [tempPoint, setTempPoint] = useState<{ x: number; y: number } | null>(null);
  const [contextMenu, setContextMenu] = useState<{ show: boolean; x: number; y: number; menu?: 'measure' | 'annotate' }>({ show: false, x: 0, y: 0, menu: undefined });

  const [zoom, setZoom] = useState(1);
  const [dimensions, setDimensions] = useState({ width: 1000, height: 600 });
  const containerRef = useRef<HTMLDivElement>(null);
  const [measurements, setMeasurements] = useState<any[]>([]);
  const [debugEvents, setDebugEvents] = useState<string[]>([]);

  // -- Missing Vars defined here --
  const today = new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  const handleZoom = (delta: number) => setZoom(z => Math.max(0.1, Math.min(5, z + delta)));
  // Alias for drawing:
  const currentMeasurement = pendingPoints.length > 0 ? { start: pendingPoints[0], current: tempPoint || pendingPoints[pendingPoints.length - 1] } : null;

  const mmPerPixel = (result as any)?.mm_per_pixel || (result as any)?.mm_per_px || 0.1;
  const hasData = !!result;

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
      Array.isArray(result?.analysis_result?.data) ? result.analysis_result.data : [],
      Array.isArray(result?.analysis_result?.teeth) ? result.analysis_result.teeth : Object.values(result?.analysis_result?.teeth || {}),
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
  const odontoUrl = '/imgs/Odonto/odontogram.png';

  // --- Image Source Logic ---
  const getUrlWithCacheBuster = (url?: string) => {
    if (!url) return undefined;
    if (url.startsWith('blob:')) return url;
    return `${url}${url.includes('?') ? '&' : '?'}t=${timestamp}`;
  };

  let showSrc =
    viewMode === 'original'
      ? getUrlWithCacheBuster(result?.image_url || result?.overlay_url)
      : getUrlWithCacheBuster(result?.overlay_url || result?.image_url);

  // If no result yet but we have a preview, show that
  if (!showSrc && (location.state as any)?.previewUrl) {
    showSrc = (location.state as any).previewUrl;
  }

  const calculateFitScale = (
    viewerWidth: number,
    viewerHeight: number,
    imageWidth: number,
    imageHeight: number
  ) => {
    if (!viewerWidth || !viewerHeight || !imageWidth || !imageHeight) return 1;

    const widthFit = viewerWidth / imageWidth;
    const heightFit = viewerHeight / imageHeight;

    // Keep the full pano visible but allow mild upscale so side gutters stay smaller.
    return Math.min(1.18, Math.min(widthFit, heightFit) * 0.98);
  };

  const fitImageToViewer = () => {
    if (!viewerRef.current || !imageRef.current) return;
    const viewer = viewerRef.current.getBoundingClientRect();
    const img = imageRef.current;
    if (!img.naturalWidth || !img.naturalHeight || !viewer.width || !viewer.height) return;

    const s = calculateFitScale(
      viewer.width,
      viewer.height,
      img.naturalWidth,
      img.naturalHeight
    );
    setFitScale(s);
    setScale(s);
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

  // --- AI Overlay Rendering ---
  const renderAIDetections = () => {
    if (viewMode !== 'overlay' || !result) return null;

    const items: React.ReactNode[] = [];

    const addBoxes = (list: any[], color: string, label: string) => {
      // Handle the fact that some lists might be empty or undefined
      if (!list) return;
      list.forEach((item, idx) => {
        // Backend format: [tooth_number, confidence, [x1, y1, x2, y2]]
        const [tooth, conf, box] = item;
        if (!box || box.length < 4) return;
        const [x1, y1, x2, y2] = box;

        items.push(
          <g key={`${label}-${tooth}-${idx}`}>
            <rect
              x={x1} y={y1} width={x2 - x1} height={y2 - y1}
              fill="none"
              stroke={color}
              strokeWidth={3 / scale}
              className="transition-opacity duration-300"
            />
            <text
              x={x1} y={y1 - (5 / scale)}
              fill={color}
              fontSize={12 / scale}
              fontWeight="bold"
              style={{ textShadow: '0 0 4px black' }}
            >
              {label} {tooth}
            </text>
          </g>
        );
      });
    };

    addBoxes(Object.entries(result.caries_by_tooth_best || {}).map(([t, d]: any) => [t, d.conf, d.box]), '#ef4444', 'Caries');
    addBoxes(Object.entries(result.periapical_by_tooth_best || {}).map(([t, d]: any) => [t, d.conf, d.box]), '#f97316', 'Periapical');
    addBoxes(Object.entries(result.filling_by_tooth_best || {}).map(([t, d]: any) => [t, d.conf, d.box]), '#3B82F6', 'Filling');

    return <g id="ai-overlay-layer">{items}</g>;
  };

  // --- Handlers ---
  const resetView = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
    setRotation(0);
    setFlipped(false);
    setBrightness(100);
    setContrast(100);
    setActiveTool('pointer');
    setActiveSubTool(null);
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

    if (activeTool === 'pan') {
      dragRef.current = { active: true, mode: 'pan', startX: e.clientX, startY: e.clientY };
      return;
    }
    if (activeTool === 'wlww') {
      dragRef.current = { active: true, mode: 'wl', startX: e.clientX, startY: e.clientY };
      return;
    }

    if (activeTool === 'measure' || activeTool === 'annotate') {
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

    if (activeSubTool && (activeTool === 'measure' || activeTool === 'annotate')) {
      const pt = toImgCoords(e.clientX, e.clientY);
      setTempPoint(pt);
    }
  };

  const handleMouseUp = () => {
    dragRef.current.active = false;
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

  const handleWheel = (e: React.WheelEvent) => {
    if (activeTool !== 'zoom') return;
    const delta = e.deltaY < 0 ? 0.1 : -0.1;
    setScale(s => Math.min(5, Math.max(0.2, s + delta)));
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

  // Handle Wheel Zoom (Non-passive to prevent scroll)
  useEffect(() => {
    const el = viewerRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      if (activeTool === 'zoom') {
        e.preventDefault();
        e.stopPropagation();
        const delta = e.deltaY < 0 ? 0.1 : -0.1;
        setScale(s => Math.min(5, Math.max(0.2, s + delta)));
      }
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [activeTool]);

  // Menu Handling
  const openMenu = (e: React.MouseEvent, type: 'measure' | 'annotate') => {
    e.stopPropagation();
    if (contextMenu.show && contextMenu.menu === type) {
      setContextMenu({ show: false, x: 0, y: 0, menu: undefined });
      setActiveSubTool(null);
      setPendingPoints([]);
      setTempPoint(null);
      if (activeTool === type) setActiveTool('pointer');
      pushDebug(`menu closed type=${type}`);
      return;
    }

    setActiveTool(type);
    setActiveSubTool(null);
    setPendingPoints([]);
    setTempPoint(null);

    // Position menu near the button that was clicked
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    // We are inside a float absolute div, coordinate space might be tricky if we use fixed.
    // But fixed is safest.
    setContextMenu({ show: true, x: rect.right, y: rect.top, menu: type });
    pushDebug(`menu opened type=${type}`);
  };

  const selectSubTool = (sub: string) => {
    setActiveSubTool(sub);
    setContextMenu(prev => ({ ...prev, show: false }));
    pushDebug(`subtool selected=${sub}`);
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
    const lw = 4 / scale;
    const txtSz = 16 / scale;
    const r = 6 / scale;
    const pad = 8 / scale;
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
              rx={10 / scale}
              fill="rgba(3,7,18,0.82)"
              stroke="rgba(255,255,255,0.55)"
              strokeWidth={Math.max(0.8, 1 / scale)}
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
              (pts[0].y + pts[1].y) / 2 - (8 / scale),
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
            {!isTemp && renderInfoLabel(pts[1].x, pts[1].y - (8 / scale), [`Angle ${deg.toFixed(1)}${String.fromCharCode(176)}`], 'middle')}
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
              {renderInfoLabel(x + w + (18 / scale), y + h - (6 / scale), [
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
            renderInfoLabel(pts[0].x + rad + (16 / scale), pts[0].y - (12 / scale), [
              `Radius ${(rad * mmPerPixel).toFixed(2)} mm`,
              `Diameter ${(diameter * mmPerPixel).toFixed(2)} mm`,
              `Area ${area.toFixed(2)} mm²`
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
          {!isTemp && renderInfoLabel(x + w + (16 / scale), y + h - (4 / scale), [
            `Width ${(w * mmPerPixel).toFixed(2)} mm`,
            `Height ${(h * mmPerPixel).toFixed(2)} mm`,
            `Area ${(w * h * mmPerPixel * mmPerPixel).toFixed(2)} mm²`
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
          {!isTemp && renderInfoLabel(x + w + (16 / scale), y + h - (4 / scale), [
            `Rx ${((w / 2) * mmPerPixel).toFixed(2)} mm`,
            `Ry ${((h / 2) * mmPerPixel).toFixed(2)} mm`,
            `Area ${(Math.PI * (w / 2) * (h / 2) * mmPerPixel * mmPerPixel).toFixed(2)} mm²`
          ])}
        </g>
      );
    }
    if (st === 'arrow' && pts.length >= 2) {
      return (
        <g key={key} stroke={arrowColor} strokeWidth={dashedWidth} {...commonProps}>
          <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={arrowColor} strokeWidth={dashedWidth} strokeDasharray={dashedPattern} markerEnd="url(#arrowhead)" vectorEffect="non-scaling-stroke" />
          {!isTemp && renderInfoLabel(pts[1].x + (12 / scale), pts[1].y - (8 / scale), ['Arrow'])}
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

  return (
    <div className="h-screen w-full text-gray-100 flex flex-col overflow-hidden font-sans" style={{ backgroundColor: '#06071A' }}>
      <TopHeader />
      {!hasData && !isProcessing && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-3 text-gray-300">
            <p className="text-lg font-semibold">분석 결과가 없습니다.</p>
            <p className="text-sm text-gray-400">이미지를 업로드하거나 처리 완료된 세션에서 진입해 주세요.</p>
          </div>
        </div>
      )}
      {hasData && (
        <div className="flex flex-1 relative overflow-y-auto">
          {/* Left Toolbar remains vertical & functional */}
          <aside className="w-32 min-w-32 flex-shrink-0 border-r border-gray-800 flex flex-col py-4 px-3 gap-4 z-50" style={{ backgroundColor: '#0A0B22' }}>
            <div className="flex gap-2 [&>*]:flex-1">
              <ToolBtn active={activeTool === 'pointer'} onClick={resetView} icon={MousePointer} title="Reset / Pointer" />
              <ToolBtn active={activeTool === 'pan'} onClick={() => setActiveTool('pan')} icon={Hand} title="Pan" />
            </div>
            <div className="flex gap-2 [&>*]:flex-1">
              <ToolBtn active={activeTool === 'wlww'} onClick={() => setActiveTool('wlww')} icon={WindowLevelIcon} title="Window / Level" />
              <ToolBtn active={false} onClick={() => setRotation(r => r + 90)} icon={RotateCw} title="Rotate 90" />
            </div>
            <div className="h-px w-full bg-dashed bg-gray-700 opacity-50" />
            <div className="flex gap-2 [&>*]:flex-1">
              <ToolBtn active={false} onClick={() => setFlipped(f => !f)} icon={FlipHorizontal} title="Flip Horizontal" />
              <ToolBtn active={activeTool === 'measure'} onClick={(e: any) => openMenu(e, 'measure')} icon={Ruler} title="Measure" />
            </div>
            <div className="flex gap-2 [&>*]:flex-1">
              <ToolBtn active={activeTool === 'annotate'} onClick={(e: any) => openMenu(e, 'annotate')} icon={PenLine} title="Annotate" />
              <div className="min-h-[44px]" aria-hidden="true" />
            </div>

            {/* Zoom Controls at Bottom */}
            <div className="flex flex-col gap-2 mt-auto pb-2">
              <div className="flex gap-2 [&>*]:flex-1">
                <button onClick={() => handleZoom(0.1)} className="w-full p-2 bg-gray-800/50 rounded-xl hover:bg-gray-700 text-gray-400 transition-colors flex items-center justify-center">
                  <ZoomIn className="w-5 h-5" />
                </button>
                <button onClick={() => handleZoom(-0.1)} className="w-full p-2 bg-gray-800/50 rounded-xl hover:bg-gray-700 text-gray-400 transition-colors flex items-center justify-center">
                  <ZoomOut className="w-5 h-5" />
                </button>
              </div>
              <div className="flex gap-2 [&>*]:flex-1">
                <button onClick={() => { setZoom(1); fitImageToViewer(); }} className="w-full p-2 bg-gray-800/50 rounded-xl hover:bg-gray-700 text-gray-400 transition-colors group flex items-center justify-center">
                  <RotateCcw className="w-5 h-5 group-hover:-rotate-180 transition-transform duration-500" />
                </button>
                <div className="min-h-[44px]" aria-hidden="true" />
              </div>
              <button
                onClick={() => setViewMode(viewMode === 'original' ? 'overlay' : 'original')}
                className={`w-full p-2 rounded-xl transition-all duration-200 flex items-center justify-center ${viewMode === 'overlay'
                  ? 'bg-indigo-600 text-white shadow-[0_0_15px_rgba(79,70,229,0.4)] ring-1 ring-indigo-400'
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                  }`}
                title="Toggle AI Overlay"
              >
                <Layers className="w-5 h-5" />
              </button>
            </div>
          </aside>

          {/* Center: report-style stacked layout on dark background */}
          <div className="flex-1 overflow-visible text-gray-100" style={{ backgroundColor: '#06071A' }}>
            <div className="max-w-7xl mx-auto py-6 px-4 lg:px-8 space-y-6">
              {/* Hero card mimicking report_v2 */}
              <div className="bg-[#0f0f0f] border border-white/5 rounded-3xl shadow-2xl overflow-hidden">
                <div className="flex flex-col lg:flex-row lg:items-start gap-6">
                  {/* Viewer column */}
                  <div className="w-full bg-black relative">
                    <div className="absolute top-4 left-4 z-10 flex items-center gap-3 bg-black/60 backdrop-blur-xl px-5 py-2.5 rounded-full border border-white/10 shadow-2xl pointer-events-none">
                      <div className={`w-2.5 h-2.5 rounded-full ${viewMode === 'overlay' ? 'bg-indigo-500 shadow-[0_0_10px_#6366f1] animate-pulse' : 'bg-gray-500'}`} />
                      <span className="text-xs font-bold uppercase tracking-wider text-gray-300">
                        {viewMode === 'overlay' ? 'AI Analysis Mode' : 'Original Source Mode'}
                      </span>
                    </div>

                    <div
                      className="w-full h-full relative overflow-hidden cursor-grab active:cursor-grabbing"
                      style={{ minHeight: `${containerHeight}px`, height: 'clamp(420px, 58vh, 620px)' }}
                      ref={viewerRef}
                      onMouseDown={handleMouseDown}
                      onMouseMove={handleMouseMove}
                      onMouseUp={handleMouseUp}
                      onWheel={handleWheel}
                      onContextMenu={handleContextMenu}
                    >
                      {!isProcessing && (result?.image || result?.image_path || result?.overlay_path || result?.image_url || result?.overlay_url) ? (
                        <div className="relative w-full h-full flex items-center justify-center overflow-hidden" ref={containerRef}>
                          <div
                            className="relative cursor-crosshair transform-gpu will-change-transform shadow-2xl ring-1 ring-black/5 mx-auto"
                            style={{
                              width: dimensions.width,
                              height: dimensions.height,
                              isolation: 'isolate',
                              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale * zoom}) rotate(${rotation}deg) scaleX(${flipped ? -1 : 1})`,
                              transition: activeTool === 'pan' ? 'none' : 'transform 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                              transformOrigin: 'center',
                            }}
                          >
                            <img
                              src={showSrc}
                              alt="Analysis Result"
                              className="block w-full h-full select-none pointer-events-none"
                              style={{
                                position: 'relative',
                                zIndex: 1,
                                filter: `brightness(${brightness}%) contrast(${contrast}%)`,
                              }}
                              draggable={false}
                              ref={imageRef}
                              onLoad={(e) => {
                                const img = e.currentTarget;
                                const w = img.naturalWidth || 1;
                                const h = img.naturalHeight || 1;
                                setDimensions({ width: w, height: h });
                                buildEdgeMapFromImage(img);

                                if (containerRef.current) {
                                  const cw = containerRef.current.clientWidth;
                                  const ch = containerRef.current.clientHeight;
                                  const fit = calculateFitScale(cw, ch, w, h);
                                  setScale(fit);
                                  setFitScale(fit);
                                  setOffset({ x: 0, y: 0 });
                                }
                              }}
                            />
                            <svg
                              className="absolute inset-0 w-full h-full"
                              viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                              style={{
                                position: 'absolute',
                                inset: 0,
                                zIndex: 9999,
                                overflow: 'visible',
                                pointerEvents: 'none',
                              }}
                            >
                              <defs>
                                <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
                                  <polygon points="0 0, 6 2, 0 4" fill="#facc15" />
                                </marker>
                              </defs>
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
                    </div>
                  </div>

                </div>
              </div>

              <div className="mb-4 rounded-2xl border shadow-sm" style={{ backgroundColor: '#0B0D26', borderColor: 'rgba(255,255,255,0.10)' }}>
                <div className="p-6">
                  <div className="space-y-6">
                    <div className="relative flex flex-col items-center gap-4 xl:pt-1">
                      <div className="min-w-0 flex justify-center">
                        <DentalChartLegend />
                      </div>
                      <div className="flex justify-center">
                        <span className="text-sm font-semibold tracking-[0.08em] text-[#CBD5E1]">
                          Odontogram
                        </span>
                      </div>
                    </div>

                    <section className="rounded-2xl border px-6 py-5 shadow-sm" style={{ backgroundColor: '#090B20', borderColor: 'rgba(255,255,255,0.10)' }}>
                      {/* <h3 className="mb-5 text-[28px] font-bold tracking-[-0.03em] text-[#334155]">Dental Chart</h3> */}
                      <div className="flex flex-col items-center">
                        <BottomTeethChart
                          onToothClick={(id) => setSelectedTooth(id)}
                          selectedTooth={selectedTooth}
                          statuses={statuses}
                          highlightRing={chartHighlights}
                          extraction={extractionCandidates}
                          implantSites={implantSiteCandidates}
                          numberingSystem={numberingSystem}
                        />
                        <div className="mt-5 flex justify-center">
                          <button
                            onClick={() => setNumberingSystem(prev => prev === 'fdi' ? 'univ' : 'fdi')}
                            className="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold shadow-sm transition-all"
                            style={{ backgroundColor: '#10132F', borderColor: '#2A3166', color: '#E5E7EB' }}
                          >
                            <RotateCw size={12} className="text-[#2563EB]" />
                            Notation: <span className="text-[#2563EB] whitespace-nowrap">{numberingSystem === 'fdi' ? 'FDI (11-48)' : 'Univ (1-32)'}</span>
                          </button>
                        </div>
                      </div>
                    </section>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel (Drill-down details) */}

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

          {/* Floating Context Menu */}
          {contextMenu.show && createPortal(
            <>
              <div className="fixed inset-0 z-[999]" onClick={() => setContextMenu({ ...contextMenu, show: false })} />
              <div
                className="fixed z-[1000] w-[220px] border border-gray-200 rounded-lg shadow-2xl p-1 flex flex-col gap-1 backdrop-blur-sm isolate"
                style={{
                  position: 'fixed',
                  top: `${contextMenu.y}px`,
                  left: `${contextMenu.x + 12}px`,
                  backgroundColor: 'rgba(255, 255, 255, 0.98)',
                  color: '#111827',
                }}
              >
                <div className="px-3 py-1.5 text-[10px] font-bold text-gray-500 uppercase tracking-wider border-b border-gray-200 mb-1 bg-white rounded-t-md">
                  {contextMenu.menu === 'measure' ? 'Measurement Tools' : 'Annotation Tools'}
                </div>
                {contextMenu.menu === 'measure' ? (
                  <>
                    <CtxBtn onClick={() => selectSubTool('length')} label="Length" />
                    <CtxBtn onClick={() => selectSubTool('bidirectional')} label="Bidirectional" />
                    <CtxBtn onClick={() => selectSubTool('angle')} label="Angle" />
                  </>
                ) : (
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
                )}
              </div>
            </>,
            document.body
          )}

        </div>
      )}
    </div>
  );
}

function ToolBtn({ active, onClick, icon: Icon, title }: any) {
  return (
    <button onClick={onClick} title={title} className={`w-full min-h-[44px] p-2 rounded-lg transition-all flex items-center justify-center ${active ? 'bg-indigo-600 text-white shadow-lg' : 'text-gray-400 hover:text-white hover:bg-white/10'}`}>
      <Icon size={20} />
    </button>
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

function CtxBtn({ onClick, label }: any) {
  return (
    <button onClick={onClick} className="block w-full text-left px-3 py-2 text-xs text-gray-700 bg-white hover:bg-indigo-50 hover:text-indigo-700 rounded transition-colors">{label}</button>
  );
}

function DentalChartLegend() {
  const items = [
    { label: 'Routine Checkup (Triage 3)', color: '#16A34A', dashed: false, gap: 18 },
    { label: 'Treatment Required (Triage 2)', color: '#D4A106', dashed: false, gap: 24 },
    { label: 'Urgent Priority (Triage 1)', color: '#DC2626', dashed: false, gap: 18 },
    { label: 'Implant', color: '#2563EB', dashed: false, gap: 18 },
    { label: 'Missing Tooth', color: '#94A3B8', dashed: true, gap: 0 },
  ];

  return (
    <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-3 text-center text-[14px] font-semibold xl:max-w-none">
      {items.map((item) => (
        <div
          key={item.label}
          className="inline-flex items-center justify-center gap-3 whitespace-nowrap text-[#334155]"
          style={{ marginRight: item.gap }}
        >
          <span
            className="block w-7"
            style={{
              borderTop: `3px ${item.dashed ? 'dashed' : 'solid'} ${item.color}`,
            }}
          />
          <span className="whitespace-nowrap" style={{ color: item.color }}>{item.label}</span>
        </div>
      ))}
    </div>
  );
}




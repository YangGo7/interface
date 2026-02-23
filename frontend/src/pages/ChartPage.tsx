import { useLocation } from 'react-router-dom';
import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { BottomTeethChart } from '../components/BottomTeethChart';
import {
  AlertTriangle, Activity, Zap, Layers, Image as ImageIcon,
  MousePointer, Hand, ZoomIn, Move, RotateCw, FlipHorizontal,
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
  const [activeTab, setActiveTab] = useState<'overview' | 'periodontitis' | 'implant'>('overview');
  const [containerHeight, setContainerHeight] = useState(520);
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

  const fitImageToViewer = () => {
    if (!viewerRef.current || !imageRef.current) return;
    const viewer = viewerRef.current.getBoundingClientRect();
    const img = imageRef.current;
    if (!img.naturalWidth || !img.naturalHeight || !viewer.width || !viewer.height) return;

    const widthFit = viewer.width / img.naturalWidth;
    const heightFit = viewer.height / img.naturalHeight;

    // Slightly overshoot both fits so it fills more of the viewport, but clamp to avoid heavy cropping
    // Favor width fill a bit, but keep height close to fit to avoid clipping bottom
    // Fill 모드: 여백 최소화를 위해 두 축 중 큰 값으로 스케일을 결정(잘림 가능)
    const s = Math.max(widthFit, heightFit);
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
    if (!imageRef.current) return { x: 0, y: 0 };
    const rect = imageRef.current.getBoundingClientRect();
    const dx = clientX - rect.left;
    const dy = clientY - rect.top;

    let x = (dx / rect.width) * imageRef.current.naturalWidth;
    let y = (dy / rect.height) * imageRef.current.naturalHeight;

    if (flipped) {
      x = imageRef.current.naturalWidth - x;
    }
    return { x, y };
  };

  const distance = (p1: any, p2: any) => Math.hypot(p2.x - p1.x, p2.y - p1.y);

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

      if (activeSubTool === 'text') {
        const text = prompt("Text Annotation:", "Note");
        if (text) {
          setShapes(prev => [...prev, { type: 'text', points: [pt], text, color: '#fbbf24' }]);
        }
        return;
      }

      // Check for auto-finish
      let needed = 0;
      if (['length', 'rect', 'circle', 'roi-circle', 'arrow'].includes(activeSubTool)) needed = 2;
      else if (['angle'].includes(activeSubTool)) needed = 3;

      const newPts = [...pendingPoints, pt];
      if (needed > 0 && newPts.length >= needed) {
        finishShape(newPts);
      } else {
        setPendingPoints(newPts);
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
    // Finish shape if drawing
    if ((activeTool === 'measure' || activeTool === 'annotate') && pendingPoints.length > 0) {
      finishShape();
    }
  };

  const finishShape = (customPoints?: any[]) => {
    const pts = customPoints || pendingPoints;
    if (!activeSubTool || pts.length === 0) return;

    // Minimal validation
    if (['length', 'roi-circle', 'rect', 'arrow'].includes(activeSubTool) && pts.length < 2) return;
    if (['angle'].includes(activeSubTool) && pts.length < 3) return;

    const color = activeTool === 'measure' ? '#22d3ee' : '#fbbf24'; // Cyan or Amber
    setShapes(prev => [...prev, { type: activeSubTool, points: pts, color }]);
    setPendingPoints([]);
    setTempPoint(null);
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
    setActiveTool(type);
    setActiveSubTool(null);
    setPendingPoints([]);

    // Position menu near the button that was clicked
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    // We are inside a float absolute div, coordinate space might be tricky if we use fixed.
    // But fixed is safest.
    setContextMenu({ show: true, x: rect.right, y: rect.top, menu: type });
  };

  const selectSubTool = (sub: string) => {
    setActiveSubTool(sub);
    setContextMenu(prev => ({ ...prev, show: false }));
  };

  // Render SVG Helper
  const renderShape = (shape: any, isTemp = false, index?: number) => {
    const pts = isTemp && tempPoint ? [...shape.points, tempPoint] : shape.points;
    if (pts.length === 0) return null;
    const st = shape.type;
    const color = shape.color || (activeTool === 'measure' ? '#22d3ee' : '#fbbf24');
    const key = isTemp ? 'temp' : `shp-${Math.random()}`;

    // Dynamic sizing based on zoom scale
    const lw = 4 / scale;
    const txtSz = 24 / scale;
    const r = 6 / scale;

    const handleDelete = (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!isTemp && index !== undefined) {
        if (confirm("Delete this annotation?")) {
          setShapes(prev => prev.filter((_, i) => i !== index));
        }
      }
    };

    // Add interaction props only for finalized shapes
    const commonProps = isTemp ? {} : {
      onClick: handleDelete,
      className: "cursor-pointer pointer-events-auto hover:opacity-80",
      style: { cursor: 'pointer' } // Force pointer
    };

    const dist = (p1: any, p2: any) => Math.hypot(p2.x - p1.x, p2.y - p1.y);

    if (st === 'length' && pts.length >= 2) {
      return (
        <g key={key} {...commonProps}>
          <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={color} strokeWidth={lw} />
          {!isTemp && (
            <text x={(pts[0].x + pts[1].x) / 2} y={(pts[0].y + pts[1].y) / 2 - (15 / scale)} fill={color} fontSize={txtSz} fontWeight="bold" textAnchor="middle" style={{ textShadow: '0px 0px 4px black' }}>
              {(dist(pts[0], pts[1]) * mmPerPixel).toFixed(2)} mm
            </text>
          )}
        </g>
      );
    }
    if (st === 'angle' && pts.length >= 2) {
      if (pts.length === 2) return <line key={key} x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={color} strokeWidth={lw} strokeDasharray={`${5 / scale},${5 / scale}`} />;
      const a = dist(pts[1], pts[0]);
      const b = dist(pts[1], pts[2]);
      const c = dist(pts[0], pts[2]);
      const val = (a * a + b * b - c * c) / (2 * a * b);
      const deg = Math.acos(Math.max(-1, Math.min(1, val))) * (180 / Math.PI);
      return (
        <g key={key} {...commonProps}>
          <polyline points={pts.map((p: any) => `${p.x},${p.y}`).join(' ')} fill="none" stroke={color} strokeWidth={lw} />
          {!isTemp && <text x={pts[1].x} y={pts[1].y - (20 / scale)} fill={color} fontSize={txtSz} fontWeight="bold" textAnchor="middle" style={{ textShadow: '0px 0px 4px black' }}>{deg.toFixed(1)}{String.fromCharCode(176)}</text>}
        </g>
      );
    }
    if (st === 'roi-free') {
      const d = pts.map((p: any) => `${p.x},${p.y}`).join(' ');
      return <polyline key={key} points={d} fill={isTemp ? "none" : "rgba(34,211,238,0.2)"} stroke={color} strokeWidth={lw} {...commonProps} />;
    }
    if ((st === 'roi-circle' || st === 'circle') && pts.length >= 2) {
      const rad = dist(pts[0], pts[1]);
      return (
        <g key={key} {...commonProps}>
          <circle cx={pts[0].x} cy={pts[0].y} r={rad} fill="rgba(34,211,238,0.2)" stroke={color} strokeWidth={lw} />
          {(st === 'roi-circle' && !isTemp) && (
            <text x={pts[0].x} y={pts[0].y} fill={color} fontSize={txtSz} fontWeight="bold" textAnchor="middle" style={{ textShadow: '0px 0px 4px black' }}>A: {(Math.PI * rad * rad * mmPerPixel * mmPerPixel).toFixed(1)}</text>
          )}
        </g>
      );
    }
    if (st === 'rect' && pts.length >= 2) {
      const x = Math.min(pts[0].x, pts[1].x);
      const y = Math.min(pts[0].y, pts[1].y);
      const w = Math.abs(pts[1].x - pts[0].x);
      const h = Math.abs(pts[1].y - pts[0].y);
      return <rect key={key} x={x} y={y} width={w} height={h} fill="rgba(34,211,238,0.2)" stroke={color} strokeWidth={lw} {...commonProps} />;
    }
    if (st === 'arrow' && pts.length >= 2) {
      return (
        <g key={key} stroke={color} strokeWidth={lw} {...commonProps}>
          <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} markerEnd="url(#arrowhead)" />
        </g>
      );
    }
    if (st === 'text') {
      return <text key={key} x={pts[0].x} y={pts[0].y} fill={color} fontSize={txtSz * 1.5} fontWeight="bold" style={{ textShadow: '0px 0px 4px black' }} {...commonProps}>{shape.text}</text>;
    }
    return null;
  };





  return (
    <div className="h-screen w-full bg-[#0a0a0a] text-gray-100 flex flex-col overflow-hidden font-sans">
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
          <aside className="w-16 flex-shrink-0 border-r border-gray-800 flex flex-col items-center py-4 gap-4 z-50 bg-[#111111]">
            <ToolBtn active={activeTool === 'pointer'} onClick={resetView} icon={MousePointer} title="Reset / Pointer" />
            <div className="h-px w-8 bg-dashed bg-gray-700 opacity-50" />
            <ToolBtn active={activeTool === 'pan'} onClick={() => setActiveTool('pan')} icon={Hand} title="Pan" />
            <ToolBtn active={activeTool === 'zoom'} onClick={() => setActiveTool('zoom')} icon={ZoomIn} title="Zoom" />
            <div className="h-px w-8 bg-dashed bg-gray-700 opacity-50" />
            <button
              onClick={() => setViewMode(viewMode === 'original' ? 'overlay' : 'original')}
              className={`p-3 rounded-2xl transition-all duration-200 ${viewMode === 'overlay'
                ? 'bg-indigo-600 text-white shadow-[0_0_15px_rgba(79,70,229,0.4)] ring-1 ring-indigo-400'
                : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                }`}
              title="Toggle AI Overlay"
            >
              <Layers className="w-6 h-6" />
            </button>
            <div className="h-px w-8 bg-dashed bg-gray-700 opacity-50" />
            <ToolBtn active={activeTool === 'measure'} onClick={(e: any) => openMenu(e, 'measure')} icon={Ruler} title="Measure" />
            <ToolBtn active={activeTool === 'annotate'} onClick={(e: any) => openMenu(e, 'annotate')} icon={PenLine} title="Annotate" />

            {/* Zoom Controls at Bottom */}
            <div className="flex flex-col gap-2 mt-auto pb-4">
              <button onClick={() => handleZoom(0.1)} className="p-2 bg-gray-800/50 rounded-xl hover:bg-gray-700 text-gray-400 transition-colors">
                <ZoomIn className="w-5 h-5" />
              </button>
              <button onClick={() => { setZoom(1); fitImageToViewer(); }} className="p-2 bg-gray-800/50 rounded-xl hover:bg-gray-700 text-gray-400 transition-colors group">
                <RotateCcw className="w-5 h-5 group-hover:-rotate-180 transition-transform duration-500" />
              </button>
              <button onClick={() => handleZoom(-0.1)} className="p-2 bg-gray-800/50 rounded-xl hover:bg-gray-700 text-gray-400 transition-colors">
                <ZoomOut className="w-5 h-5" />
              </button>
            </div>
          </aside>

          {/* Center: report-style stacked layout on dark background */}
          <div className="flex-1 overflow-visible bg-[#f8fafc] text-[#0f172a]">
            <div className="max-w-7xl mx-auto py-6 px-4 lg:px-8 space-y-6">
              {/* Hero card mimicking report_v2 */}
              <div className="bg-[#0f0f0f] border border-white/5 rounded-3xl shadow-2xl overflow-hidden">
                <div className="flex flex-col lg:flex-row lg:items-start gap-6">
                  {/* Viewer column */}
                  <div className="w-full lg:w-[60%] max-w-[1200px] bg-black relative">
                    <div className="absolute top-4 left-4 z-10 flex items-center gap-3 bg-black/60 backdrop-blur-xl px-5 py-2.5 rounded-full border border-white/10 shadow-2xl pointer-events-none">
                      <div className={`w-2.5 h-2.5 rounded-full ${viewMode === 'overlay' ? 'bg-indigo-500 shadow-[0_0_10px_#6366f1] animate-pulse' : 'bg-gray-500'}`} />
                      <span className="text-xs font-bold uppercase tracking-wider text-gray-300">
                        {viewMode === 'overlay' ? 'AI Analysis Mode' : 'Original Source Mode'}
                      </span>
                    </div>

                    <div
                      className="w-full h-full relative overflow-hidden cursor-grab active:cursor-grabbing"
                      ref={viewerRef}
                      onMouseDown={handleMouseDown}
                      onMouseMove={handleMouseMove}
                      onMouseUp={handleMouseUp}
                      onWheel={handleWheel}
                    >
                      {!isProcessing && (result?.image || result?.image_path || result?.overlay_path || result?.image_url || result?.overlay_url) ? (
                        <div className="relative w-full h-full flex items-center justify-center overflow-hidden" ref={containerRef}>
                          <div
                            className="relative cursor-crosshair transform-gpu will-change-transform shadow-2xl ring-1 ring-black/5 mx-auto"
                            style={{
                              width: '100%',
                              height: '100%',
                              maxWidth: '90vw',
                              maxHeight: '70vh',
                              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale * zoom}) rotate(${rotation}deg) scaleX(${flipped ? -1 : 1})`,
                              transition: activeTool === 'pan' ? 'none' : 'transform 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                              transformOrigin: 'center',
                            }}
                          >
                            <img
                              src={viewMode === 'overlay' ? result.overlay_path || result.overlay_url : result.image_path || result.image_url}
                              alt="Analysis Result"
                              className="w-full h-full object-contain select-none pointer-events-none"
                              style={{ maxHeight: '100%', maxWidth: '100%', margin: '0 auto' }}
                              draggable={false}
                              ref={imageRef}
                              onLoad={(e) => {
                                const img = e.currentTarget;
                                const w = img.naturalWidth || 1;
                                const h = img.naturalHeight || 1;
                                setDimensions({ width: w, height: h });

                                // Auto-fit Logic
                                if (containerRef.current) {
                                  const cw = containerRef.current.clientWidth;
                                  const ch = containerRef.current.clientHeight;
                                  const scaleX = cw / w;
                                  const scaleY = ch / h;
                                  // Viewport 맞춤: 두 축 중 작은 값으로 꽉 차게, 살짝 여유(2%)
                                  const fit = Math.min(scaleX, scaleY) * 0.98;
                                  setScale(fit);
                                  setFitScale(fit);
                                  setOffset({ x: 0, y: 0 });
                                }
                              }}
                            />

                            {/* SVG Overlay for Measurements/Annotations */}
                            <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} style={{ transform: `scaleX(${flipped ? -1 : 1})` }}>
                              {shapes.map((s: any) => renderShape(s))}
                              {currentMeasurement && (
                                <>
                                  <line
                                    x1={currentMeasurement.start.x}
                                    y1={currentMeasurement.start.y}
                                    x2={currentMeasurement.current.x}
                                    y2={currentMeasurement.current.y}
                                    stroke="#22d3ee"
                                    strokeWidth={2}
                                  />
                                  <text
                                    x={(currentMeasurement.start.x + currentMeasurement.current.x) / 2}
                                    y={(currentMeasurement.start.y + currentMeasurement.current.y) / 2 - 8}
                                    fill="#22d3ee"
                                    fontSize="14"
                                    fontWeight="bold"
                                    textAnchor="middle"
                                  >
                                    {mmPerPixel ? `${(distance(currentMeasurement.start, currentMeasurement.current) * mmPerPixel).toFixed(1)} mm` : '...'}
                                  </text>
                                </>
                              )}
                              {pendingPoints.map((p, idx) => (
                                <circle key={idx} cx={p.x} cy={p.y} r={4} fill="#22d3ee" />
                              ))}
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

              {/* Tabs below viewer */}
              <div className="bg-[#ffffff] border border-[#E2E8F0] rounded-2xl shadow-sm">
                <div className="px-4 pt-3 pb-2 border-b border-[#E2E8F0] bg-[#F8FAFC] rounded-t-2xl">
                  <div role="tablist" className="flex items-center gap-2 text-sm font-semibold text-[#475569]">
                    {[
                      { key: 'overview', label: 'Overview' },
                      { key: 'periodontitis', label: 'Periodontitis' },
                      { key: 'implant', label: 'Implant Option' },
                    ].map(tab => (
                      <button
                        key={tab.key}
                        role="tab"
                        aria-selected={activeTab === tab.key}
                        onClick={() => setActiveTab(tab.key as any)}
                        className={`px-4 py-2 rounded-[10px] border transition-all ${activeTab === tab.key
                          ? 'bg-white text-[#0f172a] border-[#E2E8F0] shadow-sm'
                          : 'bg-transparent text-[#475569] border-transparent hover:bg-[#EEF2FF] hover:text-[#0f172a]'
                          }`}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="p-4">
                  {/* Overview Panel */}
                  {activeTab === 'overview' && (
                    <div role="tabpanel" className="space-y-4">

                      {/* Controls Row */}
                      <div className="flex justify-center flex-wrap gap-4 items-center">
                        <OverviewLegend />
                        <button
                          onClick={() => setNumberingSystem(prev => prev === 'fdi' ? 'univ' : 'fdi')}
                          className="px-3 py-1.5 rounded-full text-xs font-semibold bg-white border border-[#E2E8F0] shadow-sm hover:bg-gray-50 flex items-center gap-2 transition-all"
                        >
                          <RotateCw size={12} className="text-blue-500" />
                          Notation: <span className="text-blue-600">{numberingSystem === 'fdi' ? 'FDI (11-48)' : 'Univ (1-32)'}</span>
                        </button>
                      </div>

                      <BottomTeethChart
                        onToothClick={(id) => setSelectedTooth(id)}
                        selectedTooth={selectedTooth}
                        statuses={statuses}
                        highlightRing={[]}
                        extraction={[]}
                        implantSites={[]}
                        numberingSystem={numberingSystem} // [NEW] Prop
                      />
                      <MetricsAndAlerts />
                    </div>
                  )}

                  {/* Periodontitis Panel */}
                  {activeTab === 'periodontitis' && (
                    <div role="tabpanel" className="space-y-4">
                      <LegendRow extra={false} odontoUrl={odontoUrl} />
                      <BottomTeethChart
                        onToothClick={(id) => setSelectedTooth(id)}
                        selectedTooth={selectedTooth}
                        statuses={statuses}
                        highlightRing={perioHighlights}
                        extraction={[]}
                        implantSites={[]}
                        numberingSystem={numberingSystem}
                      />
                      <div className="bg-white border border-[#E2E8F0] rounded-xl p-4 text-sm text-[#0f172a] shadow-sm">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-base font-semibold">Periodontitis Staging</h3>
                          <span className="text-xs text-[#475569]">Stage: {result?.perio_stage || 'Unknown'}</span>
                        </div>
                        <ul className="list-disc list-inside text-[#475569] space-y-1">
                          {(result?.perio_reasons || ['추가 검사가 필요합니다.']).slice(0, 3).map((t: any, i: number) => (
                            <li key={i}>{t}</li>
                          ))}
                        </ul>
                        <p className="mt-2 text-xs text-[#ef4444]">※ 추정 결과이므로 임상 검증이 필요합니다.</p>
                      </div>
                    </div>
                  )}

                  {/* Implant Option Panel */}
                  {activeTab === 'implant' && (
                    <div role="tabpanel" className="space-y-4">
                      <LegendRow extra odontoUrl={odontoUrl} />
                      <BottomTeethChart
                        onToothClick={(id) => setSelectedTooth(id)}
                        selectedTooth={selectedTooth}
                        statuses={statuses}
                        highlightRing={implantSiteCandidates.map((c: any) => c?.tooth || c?.site_fdi || c)}
                        extraction={extractionCandidates}
                        implantSites={implantSiteCandidates}
                        numberingSystem={numberingSystem}
                      />
                      <div className="grid lg:grid-cols-2 gap-4">
                        <div className="bg-white border border-[#E2E8F0] rounded-xl p-4 text-sm text-[#0f172a] shadow-sm">
                          <h3 className="text-base font-semibold mb-2">Extraction Candidates</h3>
                          <ul className="space-y-1 text-[#475569]">
                            {(extractionCandidates || []).map((c: any, idx: number) => (
                              <li key={idx} className="flex items-center justify-between border-b border-[#E2E8F0] py-1">
                                <span className="font-semibold text-[#0f172a]">FDI {c?.tooth || c?.fdi || '-'}</span>
                                <span className="text-xs text-[#ef4444]">{c?.reason || c?.note || ''}</span>
                              </li>
                            ))}
                            {(extractionCandidates || []).length === 0 && <li className="text-xs text-[#94A3B8]">데이터 없음</li>}
                          </ul>
                        </div>
                        <div className="bg-white border border-[#E2E8F0] rounded-xl p-4 text-sm text-[#0f172a] shadow-sm">
                          <h3 className="text-base font-semibold mb-2">Implant Site Candidates</h3>
                          <ul className="space-y-1 text-[#475569]">
                            {(implantSiteCandidates || []).map((c: any, idx: number) => (
                              <li key={idx} className="flex items-center justify-between border-b border-[#E2E8F0] py-1">
                                <span className="font-semibold text-[#0f172a]">FDI {c?.site_fdi || c?.tooth || '-'}</span>
                                <span className="text-xs text-[#10B981]">{c?.safety_notes || c?.reason || ''}</span>
                              </li>
                            ))}
                            {(implantSiteCandidates || []).length === 0 && <li className="text-xs text-[#94A3B8]">데이터 없음</li>}
                          </ul>
                        </div>
                      </div>
                    </div>
                  )}
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
                className="fixed z-[1000] bg-white border border-gray-200 rounded-lg shadow-xl p-1 flex flex-col gap-1 min-w-[140px]"
                style={{ top: `${contextMenu.y - 10}px`, left: `${contextMenu.x + 15}px` }}
              >
                {contextMenu.menu === 'measure' ? (
                  <>
                    <CtxBtn onClick={() => selectSubTool('length')} label="Length (2pts)" />
                    <CtxBtn onClick={() => selectSubTool('angle')} label="Angle (3pts)" />
                  </>
                ) : (
                  <>
                    <CtxBtn onClick={() => selectSubTool('text')} label="Text" />
                    <CtxBtn onClick={() => selectSubTool('arrow')} label="Arrow" />
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
    <button onClick={onClick} title={title} className={`p-2 rounded-lg transition-all ${active ? 'bg-indigo-600 text-white shadow-lg' : 'text-gray-400 hover:text-white hover:bg-white/10'}`}>
      <Icon size={20} />
    </button>
  );
}

function CtxBtn({ onClick, label }: any) {
  return (
    <button onClick={onClick} className="text-left px-3 py-2 text-xs text-gray-500 hover:bg-indigo-50 hover:text-indigo-700 rounded transition-colors">{label}</button>
  );
}

// Legend chips
function LegendRow({ extra, odontoUrl }: { extra: boolean; odontoUrl?: string }) {
  const chips = [
    { label: 'Healthy', fill: '#D1FAE5', border: '#34D399' },
    { label: 'Findings', fill: '#FEE2E2', border: '#EF4444' },
    { label: 'Implant', fill: '#DBEAFE', border: '#3B82F6' },
    { label: 'Missing', fill: '#F1F5F9', border: '#94A3B8', dashed: true },
  ];
  if (extra) {
    chips.push({ label: 'Extraction', fill: '#FFEDD5', border: '#F97316' });
    chips.push({ label: 'Implant Site', fill: 'transparent', border: '#DC2626' });
  }
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex flex-wrap justify-center gap-2">
        {chips.map((c, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs"
            style={{
              background: c.fill,
              border: `1px ${c.dashed ? 'dashed' : 'solid'} ${c.border}`,
              color: '#0f172a'
            }}
          >
            <span className="w-3 h-3 rounded-full" style={{ background: c.fill === 'transparent' ? c.border : c.border, opacity: 0.9 }} />
            {c.label}
          </span>
        ))}
      </div>
      {odontoUrl && (
        <a
          href={odontoUrl}
          target="_blank"
          rel="noreferrer"
          className="text-xs px-3 py-1 rounded-full border border-[#E2E8F0] bg-white text-[#0f172a] shadow-sm hover:bg-[#EEF2FF]"
        >
          Odontogram (참조 이미지)
        </a>
      )}
    </div>
  );
}

function OverviewLegend() {
  const chips = [
    { label: 'Implant', fill: '#DBEAFE', border: '#3B82F6' },
    { label: 'Missing', fill: '#F1F5F9', border: '#94A3B8', dashed: true },
    { label: 'Findings', fill: '#FEE2E2', border: '#EF4444' },
  ];
  return (
    <div className="flex flex-wrap justify-center gap-2">
      {chips.map((c, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs"
          style={{
            background: c.fill,
            border: `1px ${c.dashed ? 'dashed' : 'solid'} ${c.border}`,
            color: '#0f172a'
          }}
        >
          {c.label}
        </span>
      ))}
    </div>
  );
}

// Tile board for odontogram-style grid
function TileBoard({
  baseStatuses,
  highlightRing,
  extraction,
  implantSites,
  selected,
  onSelect,
}: {
  baseStatuses: Record<string, string>;
  highlightRing: any[];
  extraction: any[];
  implantSites: any[];
  selected?: number;
  onSelect: (t: number) => void;
}) {
  const rows = [
    [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
  ];
  const ringSet = new Set((highlightRing || []).map((t: any) => String(t)));
  const exSet = new Set((extraction || []).map((t: any) => String(t?.tooth || t?.fdi || t)));
  const impSet = new Set((implantSites || []).map((t: any) => String(t?.site_fdi || t?.tooth || t)));

  const getStyle = (tooth: number) => {
    const key = String(tooth);
    const status = baseStatuses[key] || 'healthy';
    let fill = '#D1FAE5'; let border = '#34D399';
    if (status === 'requires') { fill = '#FEE2E2'; border = '#EF4444'; }
    if (status === 'implant') { fill = '#DBEAFE'; border = '#3B82F6'; }
    if (status === 'missing') { fill = '#F1F5F9'; border = '#94A3B8'; }
    // overlays
    if (exSet.has(key)) { fill = '#FFEDD5'; border = '#F97316'; }
    const ring = ringSet.has(key) || impSet.has(key);
    return { fill, border, ring };
  };

  return (
    <div className="bg-white border border-[#E2E8F0] rounded-xl p-4 shadow-sm">
      {rows.map((row, idx) => (
        <div key={idx} className="flex gap-2 justify-center mb-2">
          {row.map(tooth => {
            const { fill, border, ring } = getStyle(tooth);
            const isSelected = selected === tooth;
            return (
              <button
                key={tooth}
                onClick={() => onSelect(tooth)}
                className="relative w-11 h-11 rounded-[10px] text-xs font-semibold transition-all"
                style={{
                  background: fill,
                  border: `2px ${statusDashed(baseStatuses[String(tooth)]) ? 'dashed' : 'solid'} ${border}`,
                  color: '#0f172a',
                  boxShadow: isSelected ? '0 0 0 2px #6366F1' : undefined,
                }}
              >
                {ring && <span className="absolute inset-[-4px] rounded-[12px] border-2 border-[#DC2626] pointer-events-none" />}
                <span>{tooth}</span>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function statusDashed(status?: string) {
  return status === 'missing';
}

function MetricsAndAlerts() {
  return (
    <div className="grid lg:grid-cols-3 gap-4">
      <div className="lg:col-span-2 bg-white border border-[#E2E8F0] rounded-xl p-4 shadow-sm">
        <h4 className="text-sm font-semibold text-[#0f172a] mb-2">Notes</h4>
        <p className="text-sm text-[#475569]">상세 노트와 요약을 여기에 표시하세요.</p>
      </div>
      <div className="space-y-3">
        <div className="bg-white border border-[#E2E8F0] rounded-xl p-3 shadow-sm">
          <h4 className="text-sm font-bold text-[#0f172a] mb-2">Key Metrics</h4>
          {/* placeholders; real values bound in caller if needed */}
          <div className="grid grid-cols-2 gap-2 text-xs text-[#475569]">
            <div><span className="font-semibold">Scale:</span> —</div>
            <div><span className="font-semibold">Mode:</span> —</div>
            <div><span className="font-semibold">Annotations:</span> —</div>
            <div><span className="font-semibold">Zoom:</span> —</div>
          </div>
        </div>
        <div className="bg-white border border-[#E2E8F0] rounded-xl p-3 shadow-sm">
          <h4 className="text-sm font-bold text-[#0f172a] mb-2">Alerts</h4>
          <div className="space-y-1 text-sm text-[#475569]">
            <div className="flex items-center gap-2 text-[#d97706]"><AlertTriangle size={16} /> Caries flags</div>
            <div className="flex items-center gap-2 text-[#ea580c]"><Zap size={16} /> Periapical flags</div>
            <div className="flex items-center gap-2 text-[#0ea5e9]"><Activity size={16} /> Missing</div>
          </div>
        </div>
      </div>
    </div>
  );
}

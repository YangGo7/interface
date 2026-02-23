import { useLocation } from 'react-router-dom';
import { useState, useRef, useEffect } from 'react';
import { BottomTeethChart } from '../components/BottomTeethChart';
import {
    AlertTriangle, Activity, Zap, Layers, Image as ImageIcon,
    MousePointer, Hand, ZoomIn, Move, RotateCw, FlipHorizontal,
    Ruler, PenLine
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
    const result = props?.result ?? (location.state as any)?.result;
    const [selectedTooth, setSelectedTooth] = useState<number | undefined>(undefined);
    const [viewMode, setViewMode] = useState<'overlay' | 'original'>('overlay');

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

    // Interaction State
    const dragRef = useRef<{ active: boolean; mode: 'pan' | 'wl'; startX: number; startY: number }>({
        active: false, mode: 'pan', startX: 0, startY: 0
    });
    const viewerRef = useRef<HTMLDivElement | null>(null);
    const imageRef = useRef<HTMLImageElement | null>(null);
    const [imgRect, setImgRect] = useState<DOMRect | null>(null);
    const [imgDisplayHeight, setImgDisplayHeight] = useState<number | null>(null);

    // Drawing State
    const [shapes, setShapes] = useState<any[]>([]);
    const [pendingPoints, setPendingPoints] = useState<{ x: number; y: number }[]>([]);
    const [tempPoint, setTempPoint] = useState<{ x: number; y: number } | null>(null);
    const [contextMenu, setContextMenu] = useState<{ show: boolean; x: number; y: number; menu?: 'measure' | 'annotate' }>({ show: false, x: 0, y: 0, menu: undefined });

    const mmPerPixel = (result as any)?.mm_per_pixel || (result as any)?.mm_per_px || 0.1;

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
            ...(result.caries_by_tooth || []).map((c: any) => ({
                label: 'Caries',
                value: `${c[0]} (${(c[1] ?? 0).toFixed ? c[1].toFixed(2) : c[1]})`,
                color: 'bg-rose-500',
            })),
            ...(result.periapical_by_tooth || []).map((p: any) => ({
                label: 'Periapical',
                value: `${p[0]} (${(p[1] ?? 0).toFixed ? p[1].toFixed(2) : p[1]})`,
                color: 'bg-emerald-500',
            })),
        ]
        : mockFindings;

    // --- Tooth Status Logic ---
    const statuses: Record<string, any> = {};
    const missing = result?.teeth_missing || [];
    missing.forEach((t: string) => (statuses[t] = { ...(statuses[t] || {}), missing: true }));
    const carBest = result?.caries_by_tooth_best || {};
    Object.keys(carBest).forEach((t) => (statuses[t] = { ...(statuses[t] || {}), caries: true }));
    const periBest = result?.periapical_by_tooth_best || {};
    Object.keys(periBest).forEach((t) => (statuses[t] = { ...(statuses[t] || {}), peri: true }));

    const markStatus = (field: any, prop: 'implant' | 'crown' | 'filling') => {
        if (!field) return;
        if (Array.isArray(field)) {
            field.forEach((item: any) => {
                const key = Array.isArray(item) ? item[0] : item?.toString();
                if (!key) return;
                statuses[key] = { ...(statuses[key] || {}), [prop]: true };
            });
        } else if (typeof field === 'object') {
            Object.keys(field).forEach((key) => {
                statuses[key] = { ...(statuses[key] || {}), [prop]: true };
            });
        }
    };
    markStatus(result?.implant_by_tooth_best || result?.implant_by_tooth, 'implant');
    markStatus(result?.crown_by_tooth_best || result?.crown_by_tooth, 'crown');
    markStatus(result?.filling_by_tooth_best || result?.filling_by_tooth, 'filling');

    // --- Image Source Logic ---
    const showSrc =
        viewMode === 'original'
            ? result?.image_url || result?.overlay_url
            : result?.overlay_url || result?.image_url;

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

    // 이미지 로드 시 표시 높이 계산 (화면 너비 기반)
    const computeDisplayHeight = () => {
        if (!imageRef.current || !viewerRef.current) return;
        const naturalW = imageRef.current.naturalWidth || 1;
        const naturalH = imageRef.current.naturalHeight || 1;
        const availW = viewerRef.current.clientWidth || naturalW;
        const displayH = (naturalH / naturalW) * availW;
        const capped = Math.min(displayH, window.innerHeight * 0.85);
        setImgDisplayHeight(Math.max(400, capped));
    };

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

            setPendingPoints(prev => [...prev, pt]);
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

    const finishShape = () => {
        if (!activeSubTool || pendingPoints.length === 0) return;
        const pts = [...pendingPoints];
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
        setContextMenu({ show: true, x: rect.right + 10, y: rect.top, menu: type });
    };

    const selectSubTool = (sub: string) => {
        setActiveSubTool(sub);
        setContextMenu(prev => ({ ...prev, show: false }));
    };

    // Render SVG Helper
    const renderShape = (shape: any, isTemp = false) => {
        const pts = isTemp && tempPoint ? [...shape.points, tempPoint] : shape.points;
        if (pts.length === 0) return null;
        const st = shape.type;
        const color = shape.color || (activeTool === 'measure' ? '#22d3ee' : '#fbbf24');
        const key = isTemp ? 'temp' : `shp-${Math.random()}`;

        const dist = (p1: any, p2: any) => Math.hypot(p2.x - p1.x, p2.y - p1.y);

        if (st === 'length' && pts.length >= 2) {
            return (
                <g key={key}>
                    <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={color} strokeWidth="4" />
                    {!isTemp && (
                        <text x={(pts[0].x + pts[1].x) / 2} y={(pts[0].y + pts[1].y) / 2 - 15} fill={color} fontSize="32" fontWeight="bold" textAnchor="middle" style={{ textShadow: '0px 0px 4px black' }}>
                            {(dist(pts[0], pts[1]) * mmPerPixel).toFixed(2)} mm
                        </text>
                    )}
                </g>
            );
        }
        if (st === 'angle' && pts.length >= 2) {
            if (pts.length === 2) return <line key={key} x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} stroke={color} strokeWidth="4" strokeDasharray="5,5" />;
            const a = dist(pts[1], pts[0]);
            const b = dist(pts[1], pts[2]);
            const c = dist(pts[0], pts[2]);
            const val = (a * a + b * b - c * c) / (2 * a * b);
            const deg = Math.acos(Math.max(-1, Math.min(1, val))) * (180 / Math.PI);
            return (
                <g key={key}>
                    <polyline points={pts.map((p: any) => `${p.x},${p.y}`).join(' ')} fill="none" stroke={color} strokeWidth="4" />
                    {!isTemp && <text x={pts[1].x} y={pts[1].y - 20} fill={color} fontSize="32" fontWeight="bold" textAnchor="middle" style={{ textShadow: '0px 0px 4px black' }}>{deg.toFixed(1)}°</text>}
                </g>
            );
        }
        if (st === 'roi-free') {
            const d = pts.map((p: any) => `${p.x},${p.y}`).join(' ');
            return <polyline key={key} points={d} fill={isTemp ? "none" : "rgba(34,211,238,0.2)"} stroke={color} strokeWidth="4" />;
        }
        if ((st === 'roi-circle' || st === 'circle') && pts.length >= 2) {
            const r = dist(pts[0], pts[1]);
            return (
                <g key={key}>
                    <circle cx={pts[0].x} cy={pts[0].y} r={r} fill="rgba(34,211,238,0.2)" stroke={color} strokeWidth="4" />
                    {(st === 'roi-circle' && !isTemp) && (
                        <text x={pts[0].x} y={pts[0].y} fill={color} fontSize="32" fontWeight="bold" textAnchor="middle" style={{ textShadow: '0px 0px 4px black' }}>A: {(Math.PI * r * r * mmPerPixel * mmPerPixel).toFixed(1)}</text>
                    )}
                </g>
            );
        }
        if (st === 'rect' && pts.length >= 2) {
            const x = Math.min(pts[0].x, pts[1].x);
            const y = Math.min(pts[0].y, pts[1].y);
            const w = Math.abs(pts[1].x - pts[0].x);
            const h = Math.abs(pts[1].y - pts[0].y);
            return <rect key={key} x={x} y={y} width={w} height={h} fill="rgba(34,211,238,0.2)" stroke={color} strokeWidth="4" />;
        }
        if (st === 'arrow' && pts.length >= 2) {
            return (
                <g key={key} stroke={color} strokeWidth="4">
                    <line x1={pts[0].x} y1={pts[0].y} x2={pts[1].x} y2={pts[1].y} markerEnd="url(#arrowhead)" />
                </g>
            );
        }
        if (st === 'text') {
            return <text key={key} x={pts[0].x} y={pts[0].y} fill={color} fontSize="40" fontWeight="bold" style={{ textShadow: '0px 0px 4px black' }}>{shape.text}</text>;
        }
        return null;
    };

    return (
        <div className="h-screen w-full bg-[#0a0a0a] text-gray-100 flex flex-col overflow-hidden">
            <TopHeader />

            <div className="flex flex-1 overflow-hidden bg-[#0a0a0a] relative">

                {/* Left Toolbar - Sidebar Style */}
                <aside className="w-16 flex-shrink-0 border-r border-gray-800 flex flex-col items-center py-4 gap-4 z-50 bg-[#111111]">
                    <ToolBtn active={activeTool === 'pointer'} onClick={resetView} icon={MousePointer} title="Reset / Pointer" />
                    <div className="h-px w-8 bg-gray-800" />
                    <ToolBtn active={activeTool === 'pan'} onClick={() => setActiveTool('pan')} icon={Hand} title="Pan" />
                    <ToolBtn active={activeTool === 'zoom'} onClick={() => setActiveTool('zoom')} icon={ZoomIn} title="Zoom" />
                    <ToolBtn active={activeTool === 'wlww'} onClick={() => setActiveTool('wlww')} icon={Move} title="WL/WW" />
                    <div className="h-px w-8 bg-gray-800" />
                    <ToolBtn active={activeTool === 'rotate'} onClick={() => setRotation(r => r + 90)} icon={RotateCw} title="Rotate" />
                    <ToolBtn active={activeTool === 'flip'} onClick={() => setFlipped(f => !f)} icon={FlipHorizontal} title="Flip" />
                    <div className="h-px w-8 bg-gray-800" />
                    <ToolBtn active={activeTool === 'measure'} onClick={(e: any) => openMenu(e, 'measure')} icon={Ruler} title="Measure" />
                    <ToolBtn active={activeTool === 'annotate'} onClick={(e: any) => openMenu(e, 'annotate')} icon={PenLine} title="Annotate" />
                </aside>

                {/* Main Dashboard Area */}
                <div className="flex-1 overflow-hidden relative flex flex-col">
                    <main className="flex-1 overflow-y-auto px-6 py-4 flex flex-col gap-4 w-full max-w-[1600px] mx-auto custom-scrollbar">

                        {/* A. Viewer Section */}
                        <section
                            className="bg-[#050505] border border-gray-800 rounded-xl flex flex-col relative group w-full overflow-hidden shadow-sm flex-1"
                            ref={viewerRef}
                            style={{ minHeight: '400px', height: imgDisplayHeight ? `${imgDisplayHeight}px` : 'auto' }}
                        >

                            {showSrc ? (
                                <div
                                    className="w-full h-full relative cursor-crosshair overflow-hidden flex items-center justify-center p-0 bg-black"
                                    onMouseDown={handleMouseDown}
                                    onMouseMove={handleMouseMove}
                                    onMouseUp={handleMouseUp}
                                    onContextMenu={handleContextMenu}
                                >
                                    <div
                                        className="relative inline-flex items-center justify-center transition-transform duration-75 ease-out"
                                        style={{
                                            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
                                            transformOrigin: 'center'
                                        }}
                                    >
                                        <img
                                            ref={imageRef}
                                            src={showSrc}
                                            alt="Medical Imaging"
                                            className="pointer-events-none select-none max-w-full max-h-full object-contain block"
                                            style={{
                                                transform: `rotate(${rotation}deg) scaleX(${flipped ? -1 : 1})`,
                                                filter: `brightness(${brightness}%) contrast(${contrast}%)`
                                            }}
                                            onLoad={computeDisplayHeight}
                                            onError={(e) => {
                                                (e.target as HTMLImageElement).style.opacity = '0.5';
                                                console.error("Image load failed", showSrc);
                                            }}
                                        />
                                        <svg
                                            className="absolute inset-0 w-full h-full pointer-events-none"
                                            viewBox={`0 0 ${imageRef.current?.naturalWidth || 100} ${imageRef.current?.naturalHeight || 100}`}
                                            style={{
                                                transform: `rotate(${rotation}deg) scaleX(${flipped ? -1 : 1})`
                                            }}
                                        >
                                            <defs>
                                                <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
                                                    <polygon points="0 0, 6 2, 0 4" fill={activeTool === 'measure' ? '#22d3ee' : '#fbbf24'} />
                                                </marker>
                                            </defs>
                                            {shapes.map((s, i) => renderShape(s))}
                                            {activeSubTool && renderShape({ type: activeSubTool, points: pendingPoints }, true)}

                                            {[...shapes, { points: pendingPoints }].flatMap(s => s.points || []).map((p: any, i) => (
                                                <circle key={`pt-${i}`} cx={p.x} cy={p.y} r={4} fill="#22c55e" stroke="none" />
                                            ))}
                                            {activeSubTool && tempPoint && <circle cx={tempPoint.x} cy={tempPoint.y} r={4} fill="#22c55e" opacity={0.5} />}
                                        </svg>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-2">
                                    <ImageIcon size={48} className="opacity-20" />
                                    <span>No Image Data Available</span>
                                </div>
                            )}
                        </section>

                        {/* View Mode Toggle */}
                        <div className="flex justify-center gap-3 shrink-0">
                            <button
                                onClick={() => setViewMode('overlay')}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${viewMode === 'overlay' ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-500/20' : 'bg-[#111827] border-gray-700 text-gray-400 hover:text-white hover:bg-gray-800'}`}
                            >
                                <Layers size={16} />
                                <span>Overlay Analysis</span>
                            </button>
                            <button
                                onClick={() => setViewMode('original')}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${viewMode === 'original' ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-500/20' : 'bg-[#111827] border-gray-700 text-gray-400 hover:text-white hover:bg-gray-800'}`}
                            >
                                <ImageIcon size={16} />
                                <span>Original Image</span>
                            </button>
                        </div>

                        {/* B. Stats Section - Original Grid Layout */}
                        <section className="bg-[#0f0f0f] border border-gray-800 rounded-xl p-4 flex flex-col gap-3 shrink-0">
                            <div className="flex items-center gap-2 text-sm text-gray-400 font-semibold">
                                <Activity className="w-4 h-4 text-cyan-400" />
                                <span>Detection Counts</span>
                            </div>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                                {counts.map(c => (
                                    <div key={c.label} className="bg-[#111827] border border-gray-800 rounded-lg px-4 py-3 flex justify-between items-center shadow-sm">
                                        <span className="text-gray-400 text-sm">{c.label}</span>
                                        <span className="text-white font-bold text-lg">{c.value}</span>
                                    </div>
                                ))}
                            </div>
                        </section>

                        {/* C. Findings & Odontogram */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 shrink-0">
                            {/* Findings */}
                            <section className="col-span-1 bg-[#0f0f0f] border border-gray-800 rounded-xl p-4 flex flex-col gap-4">
                                <div className="flex items-center gap-2 text-sm text-gray-400 font-semibold">
                                    <AlertTriangle className="w-4 h-4 text-rose-400" />
                                    <span>Findings</span>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                    {findings.length > 0 && findings[0].label !== 'No Data' ? findings.map((f, i) => (
                                        <span key={i} className={`px-2 py-1 rounded text-xs text-white ${f.color} flex items-center gap-1`}>
                                            {f.label}: <span className="font-mono">{f.value}</span>
                                        </span>
                                    )) : <span className="text-gray-500 text-sm">No findings</span>}
                                </div>
                                {/* PBL Section Hidden */}
                                {/* <div className="border-t border-gray-800 pt-3">
                <div className="flex items-center gap-2 text-sm text-gray-400 font-semibold mb-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <span>PBL %</span>
                </div>
                <div className="flex flex-wrap gap-1 max-h-[150px] overflow-y-auto custom-scrollbar">
                  {pblEntries.length > 0 ? pblEntries.map((item, idx) => (
                    <span key={idx} className="px-2 py-0.5 rounded bg-[#1f2937] border border-gray-700 text-xs text-gray-300 font-mono">{item}</span>
                  )) : <span className="text-gray-500 text-sm">No data</span>}
                </div>
              </div> */}
                            </section>

                            {/* Odontogram */}
                            <section className="col-span-1 lg:col-span-2 bg-[#0f0f0f] border border-gray-800 rounded-xl p-4">
                                <div className="text-sm text-gray-400 font-semibold mb-3">Odontogram</div>
                                <BottomTeethChart onToothClick={setSelectedTooth} selectedTooth={selectedTooth} statuses={statuses} />
                            </section>
                        </div>

                    </main>
                </div>

                {/* Right Panel */}
                <RightPanel selectedTooth={selectedTooth} result={result} onClose={() => setSelectedTooth(undefined)} />

            </div>

            {/* Floating Context Menu (Portal-like behavior at root) */}
            {contextMenu.show && (
                <>
                    <div className="fixed inset-0 z-[999]" onClick={() => setContextMenu({ show: false, x: 0, y: 0 })} />
                    <div
                        className="fixed z-[1000] bg-[#1a1a1a] border border-gray-700 rounded-lg shadow-xl p-1 flex flex-col gap-1 min-w-[140px]"
                        style={{ top: contextMenu.y - 10, left: contextMenu.x + 5 }}
                    >
                        <div className="px-3 py-1.5 text-[10px] font-bold text-gray-500 uppercase tracking-wider border-b border-gray-800 mb-1">{contextMenu.menu} Tools</div>
                        {contextMenu.menu === 'measure' ? (
                            <>
                                <CtxBtn onClick={() => selectSubTool('length')} label="Length (2pts)" />
                                <CtxBtn onClick={() => selectSubTool('angle')} label="Angle (3pts)" />
                                <CtxBtn onClick={() => selectSubTool('roi-free')} label="ROI (Free)" />
                                <CtxBtn onClick={() => selectSubTool('roi-circle')} label="ROI (Circle)" />
                            </>
                        ) : (
                            <>
                                <CtxBtn onClick={() => selectSubTool('text')} label="Text Note" />
                                <CtxBtn onClick={() => selectSubTool('arrow')} label="Arrow" />
                                <CtxBtn onClick={() => selectSubTool('rect')} label="Rectangle" />
                                <CtxBtn onClick={() => selectSubTool('circle')} label="Circle" />
                            </>
                        )}
                    </div>
                </>
            )}
        </div>
    );
}

const ToolBtn = ({ active, onClick, icon: Icon, title }: any) => (
    <button onClick={onClick} title={title} className={`p-2 rounded-lg transition-all ${active ? 'bg-indigo-600 text-white shadow-lg' : 'text-gray-400 hover:text-white hover:bg-white/10'}`}>
        <Icon size={20} />
    </button>
);

const CtxBtn = ({ onClick, label }: any) => (
    <button onClick={onClick} className="text-left px-3 py-2 text-xs text-gray-300 hover:bg-indigo-600 hover:text-white rounded transition-colors">{label}</button>
);

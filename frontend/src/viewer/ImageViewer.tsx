import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Bug,
  Eye,
  Hand,
  Image as ImageIcon,
  RotateCcw,
  Ruler,
  Square,
  Trash2,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';

export type ViewerSource = {
  id: string;
  label: string;
  url: string;
};

type ImageViewerProps = {
  sources: ViewerSource[];
  initialSourceId?: string;
  title?: string;
  maxHeight?: number;
};

type ToolMode = 'pan' | 'length' | 'arrow' | 'rect' | 'ellipse';

type ViewerShape = {
  id: string;
  type: Exclude<ToolMode, 'pan'>;
  points: { x: number; y: number }[];
};

type DebugSnapshot = {
  clientX: number;
  clientY: number;
  viewerX: number;
  viewerY: number;
  canvasX: number;
  canvasY: number;
  imageX: number;
  imageY: number;
  absoluteBackX: number;
  absoluteBackY: number;
  backToScreenX: number;
  backToScreenY: number;
};

export function ImageViewer({
  sources,
  initialSourceId,
  title = 'Viewer',
  maxHeight = 560,
}: ImageViewerProps) {
  const validSources = useMemo(
    () => sources.filter((source) => Boolean(source?.url)),
    [sources]
  );
  const defaultSourceId = initialSourceId && validSources.some((source) => source.id === initialSourceId)
    ? initialSourceId
    : validSources[0]?.id;

  const [activeSourceId, setActiveSourceId] = useState<string | undefined>(defaultSourceId);
  const [naturalSize, setNaturalSize] = useState({ width: 1, height: 1 });
  const [fitScale, setFitScale] = useState(1);
  const [displaySize, setDisplaySize] = useState({ width: 1000, height: 560 });
  const [viewerHeight, setViewerHeight] = useState(maxHeight);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [activeTool, setActiveTool] = useState<ToolMode>('pan');
  const [shapes, setShapes] = useState<ViewerShape[]>([]);
  const [pendingPoints, setPendingPoints] = useState<{ x: number; y: number }[]>([]);
  const [tempPoint, setTempPoint] = useState<{ x: number; y: number } | null>(null);
  const [debugEnabled, setDebugEnabled] = useState(false);
  const [debugSnapshot, setDebugSnapshot] = useState<DebugSnapshot | null>(null);

  const viewerRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const displayRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ active: boolean; startX: number; startY: number }>({
    active: false,
    startX: 0,
    startY: 0,
  });

  const activeSource = validSources.find((source) => source.id === activeSourceId) ?? validSources[0];

  useEffect(() => {
    if (!activeSourceId && validSources[0]?.id) {
      setActiveSourceId(validSources[0].id);
    }
  }, [activeSourceId, validSources]);

  useEffect(() => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
    setBrightness(100);
    setContrast(100);
    setShapes([]);
    setPendingPoints([]);
    setTempPoint(null);
  }, [activeSourceId]);

  const fitImage = () => {
    const viewer = viewerRef.current;
    const image = imageRef.current;
    if (!viewer || !image?.naturalWidth || !image?.naturalHeight) {
      return;
    }

    const viewerWidth = viewer.clientWidth;
    if (!viewerWidth) {
      return;
    }

    const nextHeight = Math.max(
      320,
      Math.min(maxHeight, Math.round((viewerWidth / image.naturalWidth) * image.naturalHeight))
    );
    const scale = Math.min(viewerWidth / image.naturalWidth, nextHeight / image.naturalHeight) * 0.995;

    setNaturalSize({ width: image.naturalWidth, height: image.naturalHeight });
    setViewerHeight(nextHeight);
    setFitScale(scale);
    setDisplaySize({
      width: image.naturalWidth * scale,
      height: image.naturalHeight * scale,
    });
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  };

  useEffect(() => {
    const handleResize = () => {
      fitImage();
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    const debugData = getDebugSnapshot(event.clientX, event.clientY);
    if (debugEnabled && debugData) {
      setDebugSnapshot(debugData);
    }

    if (activeTool !== 'pan') {
      const point = debugData
        ? { x: debugData.imageX, y: debugData.imageY }
        : toImagePoint(event.clientX, event.clientY);
      if (!point) {
        return;
      }

      const nextPoints = [...pendingPoints, point];
      if (nextPoints.length >= 2) {
        setShapes((prev) => [
          ...prev,
          {
            id: `${activeTool}-${Date.now()}-${prev.length}`,
            type: activeTool,
            points: nextPoints.slice(0, 2),
          },
        ]);
        setPendingPoints([]);
        setTempPoint(null);
      } else {
        setPendingPoints(nextPoints);
        setTempPoint(point);
      }
      return;
    }

    dragRef.current = {
      active: true,
      startX: event.clientX,
      startY: event.clientY,
    };
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (activeTool !== 'pan') {
      if (pendingPoints.length > 0) {
        const point = toImagePoint(event.clientX, event.clientY);
        if (point) {
          setTempPoint(point);
        }
      }
      return;
    }

    if (!dragRef.current.active) {
      return;
    }

    const dx = event.clientX - dragRef.current.startX;
    const dy = event.clientY - dragRef.current.startY;

    dragRef.current.startX = event.clientX;
    dragRef.current.startY = event.clientY;
    setOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
  };

  const handleMouseUp = () => {
    dragRef.current.active = false;
  };

  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const delta = event.deltaY < 0 ? 0.1 : -0.1;
    setZoom((prev) => Math.max(0.2, Math.min(5, prev + delta)));
  };

  const resetView = () => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
    setBrightness(100);
    setContrast(100);
    fitImage();
  };

  const clearShapes = () => {
    setShapes([]);
    setPendingPoints([]);
    setTempPoint(null);
  };

  const toImagePoint = (clientX: number, clientY: number) => {
    const snapshot = getDebugSnapshot(clientX, clientY);
    if (!snapshot) {
      return null;
    }

    return {
      x: snapshot.imageX,
      y: snapshot.imageY,
    };
  };

  const getDebugSnapshot = (clientX: number, clientY: number) => {
    const frame = imageRef.current;
    const viewer = viewerRef.current;
    if (!frame || !naturalSize.width || !naturalSize.height) {
      return null;
    }
    if (!viewer) {
      return null;
    }

    const rect = frame.getBoundingClientRect();
    const viewerRect = viewer.getBoundingClientRect();

    if (
      clientX < rect.left ||
      clientX > rect.right ||
      clientY < rect.top ||
      clientY > rect.bottom
    ) {
      return null;
    }

    const localX = clientX - rect.left;
    const localY = clientY - rect.top;

    const imageX = (localX / rect.width) * naturalSize.width;
    const imageY = (localY / rect.height) * naturalSize.height;
    const backToScreenX = rect.left + (imageX / naturalSize.width) * rect.width;
    const backToScreenY = rect.top + (imageY / naturalSize.height) * rect.height;

    return {
      clientX,
      clientY,
      viewerX: clientX - viewerRect.left,
      viewerY: clientY - viewerRect.top,
      canvasX: localX,
      canvasY: localY,
      imageX,
      imageY,
      absoluteBackX: backToScreenX,
      absoluteBackY: backToScreenY,
      backToScreenX: backToScreenX - viewerRect.left,
      backToScreenY: backToScreenY - viewerRect.top,
    };
  };

  const renderShape = (shape: ViewerShape | { type: ToolMode; points: { x: number; y: number }[] }) => {
    const previewPoints =
      shape.points.length === 1 && tempPoint ? [shape.points[0], tempPoint] : shape.points;
    if (previewPoints.length < 2 || shape.type === 'pan') {
      return null;
    }

    const [p1, p2] = previewPoints;
    const stroke = shape.type === 'arrow' ? '#facc15' : '#22c55e';
    const dash = shape.type === 'arrow' ? '6 4' : '5 5';
    const x = Math.min(p1.x, p2.x);
    const y = Math.min(p1.y, p2.y);
    const w = Math.abs(p2.x - p1.x);
    const h = Math.abs(p2.y - p1.y);
    const length = Math.hypot(p2.x - p1.x, p2.y - p1.y);

    if (shape.type === 'length') {
      return (
        <g key={'id' in shape ? shape.id : 'preview-length'}>
          <line
            x1={p1.x}
            y1={p1.y}
            x2={p2.x}
            y2={p2.y}
            stroke={stroke}
            strokeWidth={2}
            strokeDasharray={dash}
            vectorEffect="non-scaling-stroke"
          />
          <text
            x={(p1.x + p2.x) / 2}
            y={(p1.y + p2.y) / 2 - 8}
            fill={stroke}
            fontSize={14}
            fontWeight="bold"
            textAnchor="middle"
            style={{ textShadow: '0 0 4px rgba(0,0,0,0.9)' }}
          >
            {length.toFixed(1)} px
          </text>
        </g>
      );
    }

    if (shape.type === 'arrow') {
      return (
        <g key={'id' in shape ? shape.id : 'preview-arrow'}>
          <line
            x1={p1.x}
            y1={p1.y}
            x2={p2.x}
            y2={p2.y}
            stroke={stroke}
            strokeWidth={2}
            strokeDasharray={dash}
            markerEnd="url(#viewer-arrowhead)"
            vectorEffect="non-scaling-stroke"
          />
        </g>
      );
    }

    if (shape.type === 'rect') {
      return (
        <rect
          key={'id' in shape ? shape.id : 'preview-rect'}
          x={x}
          y={y}
          width={w}
          height={h}
          fill="none"
          stroke={stroke}
          strokeWidth={2}
          strokeDasharray={dash}
          vectorEffect="non-scaling-stroke"
        />
      );
    }

    if (shape.type === 'ellipse') {
      return (
        <ellipse
          key={'id' in shape ? shape.id : 'preview-ellipse'}
          cx={x + w / 2}
          cy={y + h / 2}
          rx={w / 2}
          ry={h / 2}
          fill="none"
          stroke={stroke}
          strokeWidth={2}
          strokeDasharray={dash}
          vectorEffect="non-scaling-stroke"
        />
      );
    }

    return null;
  };

  if (!activeSource) {
    return null;
  }

  return (
    <div className="overflow-hidden rounded-3xl border border-white/10 bg-[#0B0D26] shadow-2xl">
      <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-white">
          <Eye className="h-4 w-4 text-cyan-300" />
          {title}
        </div>
        <div className="flex items-center gap-2">
          {validSources.map((source) => (
            <button
              key={source.id}
              onClick={() => setActiveSourceId(source.id)}
              className={`rounded-full px-3 py-1 text-xs font-semibold transition-colors ${source.id === activeSource.id
                ? 'bg-cyan-500 text-[#04111E]'
                : 'bg-white/5 text-gray-300 hover:bg-white/10'
                }`}
            >
              {source.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
        <div className="flex items-center gap-2 text-xs text-gray-300">
          <ImageIcon className="h-4 w-4 text-cyan-300" />
          {activeSource.label}
          <span className="text-gray-500">
            {naturalSize.width} x {naturalSize.height}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom((prev) => Math.max(0.2, prev - 0.1))}
            className="rounded-lg bg-white/5 p-2 text-gray-300 transition-colors hover:bg-white/10 hover:text-white"
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <button
            onClick={() => setZoom((prev) => Math.min(5, prev + 0.1))}
            className="rounded-lg bg-white/5 p-2 text-gray-300 transition-colors hover:bg-white/10 hover:text-white"
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            onClick={resetView}
            className="rounded-lg bg-white/5 p-2 text-gray-300 transition-colors hover:bg-white/10 hover:text-white"
            title="Reset view"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="grid gap-3 border-b border-white/10 px-4 py-3 md:grid-cols-2">
        <label className="flex items-center gap-3 text-xs text-gray-300">
          <span className="w-20 shrink-0">Brightness</span>
          <input
            type="range"
            min="30"
            max="200"
            value={brightness}
            onChange={(event) => setBrightness(Number(event.target.value))}
            className="w-full"
          />
          <span className="w-10 text-right">{brightness}</span>
        </label>
        <label className="flex items-center gap-3 text-xs text-gray-300">
          <span className="w-20 shrink-0">Contrast</span>
          <input
            type="range"
            min="30"
            max="200"
            value={contrast}
            onChange={(event) => setContrast(Number(event.target.value))}
            className="w-full"
          />
          <span className="w-10 text-right">{contrast}</span>
        </label>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-b border-white/10 px-4 py-3">
        <ToolChip
          label="Pan"
          active={activeTool === 'pan'}
          icon={<Hand className="h-4 w-4" />}
          onClick={() => {
            setActiveTool('pan');
            setPendingPoints([]);
            setTempPoint(null);
          }}
        />
        <ToolChip
          label="Length"
          active={activeTool === 'length'}
          icon={<Ruler className="h-4 w-4" />}
          onClick={() => {
            setActiveTool('length');
            setPendingPoints([]);
            setTempPoint(null);
          }}
        />
        <ToolChip
          label="Arrow"
          active={activeTool === 'arrow'}
          icon={<Eye className="h-4 w-4" />}
          onClick={() => {
            setActiveTool('arrow');
            setPendingPoints([]);
            setTempPoint(null);
          }}
        />
        <ToolChip
          label="Rect"
          active={activeTool === 'rect'}
          icon={<Square className="h-4 w-4" />}
          onClick={() => {
            setActiveTool('rect');
            setPendingPoints([]);
            setTempPoint(null);
          }}
        />
        <ToolChip
          label="Ellipse"
          active={activeTool === 'ellipse'}
          icon={<Eye className="h-4 w-4" />}
          onClick={() => {
            setActiveTool('ellipse');
            setPendingPoints([]);
            setTempPoint(null);
          }}
        />
        <ToolChip
          label="Clear"
          active={false}
          icon={<Trash2 className="h-4 w-4" />}
          onClick={clearShapes}
        />
        <ToolChip
          label="Debug"
          active={debugEnabled}
          icon={<Bug className="h-4 w-4" />}
          onClick={() => setDebugEnabled((prev) => !prev)}
        />
      </div>

      <div
        ref={viewerRef}
        className="relative overflow-hidden bg-black"
        style={{ minHeight: `${viewerHeight}px`, height: `${viewerHeight}px` }}
      >
        {debugEnabled && debugSnapshot && (
          <>
            <div
              className="pointer-events-none absolute z-20 h-3 w-3 rounded-full border-2 border-red-400 bg-red-400/30"
              style={{
                left: debugSnapshot.viewerX - 6,
                top: debugSnapshot.viewerY - 6,
              }}
            />
            <div
              className="pointer-events-none absolute z-20 h-3 w-3 rounded-full border-2 border-cyan-300 bg-cyan-300/30"
              style={{
                left: debugSnapshot.backToScreenX - 6,
                top: debugSnapshot.backToScreenY - 6,
              }}
            />
          </>
        )}
        <div className="flex h-full w-full items-center justify-center overflow-hidden">
          <div
            ref={displayRef}
            className="relative cursor-grab shadow-2xl active:cursor-grabbing"
            style={{
              width: displaySize.width,
              height: displaySize.height,
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
              transformOrigin: 'center',
              transition: dragRef.current.active ? 'none' : 'transform 0.15s ease-out',
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onWheel={handleWheel}
          >
            <img
              ref={imageRef}
              src={activeSource.url}
              alt={activeSource.label}
              className="absolute inset-0 block h-full w-full select-none"
              draggable={false}
              style={{
                zIndex: 1,
                width: '100%',
                height: '100%',
                filter: `brightness(${brightness}%) contrast(${contrast}%)`,
              }}
              onLoad={() => requestAnimationFrame(() => fitImage())}
            />
            <svg
              className="pointer-events-none absolute inset-0 block h-full w-full"
              viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
              preserveAspectRatio="none"
              style={{ zIndex: 10 }}
            >
              <defs>
                <marker
                  id="viewer-arrowhead"
                  markerWidth="6"
                  markerHeight="4"
                  refX="6"
                  refY="2"
                  orient="auto"
                >
                  <polygon points="0 0, 6 2, 0 4" fill="#facc15" />
                </marker>
              </defs>
              {shapes.map((shape) => renderShape(shape))}
              {pendingPoints.length > 0 && renderShape({ type: activeTool, points: pendingPoints })}
            </svg>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between px-4 py-3 text-xs text-gray-400">
        <span>Zoom {(fitScale * zoom).toFixed(2)}x</span>
        <span>
          {activeTool === 'pan'
            ? 'Drag to pan, wheel to zoom'
            : 'Click two points to draw'}
        </span>
      </div>

      {debugEnabled && debugSnapshot && (
        <div className="grid gap-2 border-t border-white/10 bg-black/20 px-4 py-3 text-xs text-gray-300 md:grid-cols-2">
          <div>
            <div className="font-semibold text-cyan-300">Page</div>
            <div>clientX: {debugSnapshot.clientX.toFixed(1)}</div>
            <div>clientY: {debugSnapshot.clientY.toFixed(1)}</div>
          </div>
          <div>
            <div className="font-semibold text-cyan-300">Viewer Local</div>
            <div>x: {debugSnapshot.viewerX.toFixed(1)}</div>
            <div>y: {debugSnapshot.viewerY.toFixed(1)}</div>
          </div>
          <div>
            <div className="font-semibold text-cyan-300">Canvas</div>
            <div>x: {debugSnapshot.canvasX.toFixed(1)}</div>
            <div>y: {debugSnapshot.canvasY.toFixed(1)}</div>
          </div>
          <div>
            <div className="font-semibold text-cyan-300">Image Pixel</div>
            <div>x: {debugSnapshot.imageX.toFixed(1)}</div>
            <div>y: {debugSnapshot.imageY.toFixed(1)}</div>
          </div>
          <div>
            <div className="font-semibold text-cyan-300">Back To Screen</div>
            <div>x: {debugSnapshot.backToScreenX.toFixed(1)}</div>
            <div>y: {debugSnapshot.backToScreenY.toFixed(1)}</div>
          </div>
          <div>
            <div className="font-semibold text-cyan-300">Back To Page</div>
            <div>x: {debugSnapshot.absoluteBackX.toFixed(1)}</div>
            <div>y: {debugSnapshot.absoluteBackY.toFixed(1)}</div>
          </div>
        </div>
      )}
    </div>
  );
}

function ToolChip({
  active,
  label,
  icon,
  onClick,
}: {
  active: boolean;
  label: string;
  icon: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold transition-colors ${active
        ? 'border-cyan-400 bg-cyan-400 text-[#04111E]'
        : 'border-white/10 bg-white/5 text-gray-300 hover:bg-white/10'
        }`}
    >
      {icon}
      {label}
    </button>
  );
}

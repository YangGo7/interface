import React, { useEffect, useRef, useState } from 'react';
import * as cornerstone from '@cornerstonejs/core';
import { PanTool, ZoomTool, WindowLevelTool, LengthTool, ArrowAnnotateTool, RectangleROITool, EllipticalROITool } from '@cornerstonejs/tools';
import {
    Bug,
    Eye,
    Hand,
    Image as ImageIcon,
    RotateCcw,
    Ruler,
    Square,
    ZoomIn,
    ZoomOut,
    Sliders
} from 'lucide-react';
import {
    getRegisteredDicomMetadata,
    initCornerstone,
    registerLocalDicomFileWithMetadata,
    registerNativeDicomFileWithMetadata,
} from './cornerstone/init';
import type { DicomOverlayMetadata } from './cornerstone/dicomMetadata';
import { getAutoWindowForCornerstoneImage } from './cornerstone/autoWindow';
import { addAndGroupTools, createOrGetToolGroup, setActiveTool, TOOL_GROUP_ID } from './cornerstone/tools';
import { DicomMetadataOverlay } from './DicomMetadataOverlay';

export type ViewerSource = {
    id: string;
    label: string;
    url: string;
    file?: File;
    files?: File[];
    scheme?: 'web' | 'wadouri' | 'dicomlocal' | 'dicomfile' | 'dicomfolder';
};

type CornerstoneViewerProps = {
    sources: ViewerSource[];
    initialSourceId?: string;
    title?: string;
    maxHeight?: number;
    is3D?: boolean;
    showToolbar?: boolean;
    invert?: boolean;
    interactionMode?: string;
    resetToken?: number;
    autoWindowToken?: number;
    brightness?: number;
    contrast?: number;
    rotation?: number;
    flipped?: boolean;
};

type ToolMode = 'pan' | 'length' | 'arrow' | 'rect' | 'ellipse' | 'wl';

export function CornerstoneViewer({
    sources,
    initialSourceId,
    title = 'Viewer',
    maxHeight = 560,
    is3D = false,
    showToolbar = true,
    invert = false,
    interactionMode,
    resetToken = 0,
    autoWindowToken = 0,
    brightness = 100,
    contrast = 100,
    rotation = 0,
    flipped = false,
}: CornerstoneViewerProps) {
    const viewerRef = useRef<HTMLDivElement>(null);
    const dicomImageIdsRef = useRef<Record<string, string>>({});
    const renderingEngineIdRef = useRef(`cs-engine-${Math.random().toString(36).slice(2)}`);
    const viewportIdRef = useRef(`cs-viewport-${Math.random().toString(36).slice(2)}`);
    const renderingEngineRef = useRef<cornerstone.RenderingEngine | null>(null);
    const [isInit, setIsInit] = useState(false);
    const [loadState, setLoadState] = useState<'initializing' | 'loading' | 'ready' | 'error'>('initializing');
    const [statusMessage, setStatusMessage] = useState('Initializing Cornerstone...');
    const [lastImageId, setLastImageId] = useState('');
    const [activeSourceId, setActiveSourceId] = useState<string | undefined>(
        initialSourceId || sources[0]?.id
    );
    const [activeTool, setActiveToolState] = useState<ToolMode>('pan');
    const [debugEnabled, setDebugEnabled] = useState(false);
    const lensCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const [magnifier, setMagnifier] = useState({ visible: false, x: 0, y: 0 });
    const [overlayMetadata, setOverlayMetadata] = useState<DicomOverlayMetadata | null>(null);

    const activeSource = sources.find((s) => s.id === activeSourceId) || sources[0];

    function syncOverlayWindow(windowCenter: number, windowWidth: number) {
        setOverlayMetadata((prev) => prev
            ? {
                ...prev,
                windowCenter,
                windowWidth,
            }
            : prev);
    }

    function applyViewportDisplayState(
        viewport: cornerstone.Types.IStackViewport | undefined,
        imageId?: string
    ) {
        if (!viewport || !imageId) return;

        const image = cornerstone.cache.getImage(imageId);
        if (!image) return;

        const props: cornerstone.Types.ViewportProperties = {};
        if (image.minPixelValue !== undefined && image.maxPixelValue !== undefined) {
            props.voiRange = { lower: image.minPixelValue, upper: image.maxPixelValue };
        } else if (image.windowCenter != null && image.windowWidth != null) {
            const wc = Array.isArray(image.windowCenter) ? image.windowCenter[0] : image.windowCenter;
            const ww = Array.isArray(image.windowWidth) ? image.windowWidth[0] : image.windowWidth;
            if (typeof wc === 'number' && typeof ww === 'number' && ww > 0) {
                props.voiRange = { lower: wc - ww / 2, upper: wc + ww / 2 };
            }
        }

        props.invert = Boolean(image.invert);
        viewport.setProperties(props);
    }

    // Initialize Cornerstone once
    useEffect(() => {
        initCornerstone()
            .then(() => {
                addAndGroupTools();
                createOrGetToolGroup();
                setIsInit(true);
                setLoadState('loading');
                setStatusMessage('Cornerstone initialized. Loading image...');
            })
            .catch((error) => {
                console.error('Cornerstone init failed', error);
                setLoadState('error');
                setStatusMessage(error instanceof Error ? error.message : 'Cornerstone initialization failed.');
            });
    }, []);

    // Setup RenderingEngine and Viewport when ref is ready
    useEffect(() => {
        if (!isInit || !viewerRef.current || !activeSource) return;

        const element = viewerRef.current;
        const scheme = activeSource.scheme || 'web';
        const renderingEngineId = renderingEngineIdRef.current;
        const viewportId = viewportIdRef.current;
        console.log('Cornerstone viewport host size', {
            source: activeSource.label,
            scheme,
            width: element.clientWidth,
            height: element.clientHeight,
        });

        try {
            renderingEngineRef.current?.destroy();
        } catch (error) {
            console.warn('Previous rendering engine destroy failed', error);
        }

        // We rely on instance-specific useCPURendering configuration.
        const renderingEngine = new cornerstone.RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;

        // Enable element
        const viewportInput = {
            viewportId,
            element: element,
            type: cornerstone.Enums.ViewportType.STACK,
        };
        renderingEngine.enableElement(viewportInput);

        const viewport = renderingEngine.getViewport(viewportId) as cornerstone.Types.IStackViewport;
        const toolGroup = createOrGetToolGroup();


        // Add viewport to ToolGroup if not already added
        if (toolGroup && !toolGroup.getViewportIds().includes(viewportId)) {
            toolGroup.addViewport(viewportId, renderingEngineId);
        }

        let isCancelled = false;
        setOverlayMetadata(null);
        setLoadState('loading');
        setStatusMessage(`Loading ${activeSource.label} automatically on GPU...`);

        const loadSource = async () => {
            try {
                const imageId =
                    scheme === 'dicomfile' && activeSource.file
                        ? (dicomImageIdsRef.current[activeSource.id] ||= await registerNativeDicomFileWithMetadata(activeSource.file))
                        : scheme === 'dicomlocal' && activeSource.file
                            ? (dicomImageIdsRef.current[activeSource.id] ||= await registerLocalDicomFileWithMetadata(activeSource.id, activeSource.file))
                            : `${scheme}:${activeSource.url}`;
                if (isCancelled) return;

                console.log('Cornerstone: Setting stack for', imageId);
                await cornerstone.imageLoader.loadAndCacheImage(imageId);
                if (isCancelled) return;

                setLastImageId(imageId);
                setOverlayMetadata(getRegisteredDicomMetadata(imageId));

                await viewport.setStack([imageId]);
                if (isCancelled) return;

                applyViewportDisplayState(viewport, imageId);

                console.log('Cornerstone: Stack set, resetting camera...');
                renderingEngine.resize(true, false);
                viewport.resetCamera();
                applyViewportDisplayState(viewport, imageId);
                viewport.render();
                renderingEngine.render();
                setOverlayMetadata(getRegisteredDicomMetadata(imageId));
                setLoadState('ready');
                setStatusMessage(`Rendered ${activeSource.label}`);
            } catch (err) {
                console.error('Cornerstone: setStack failed', err);
                setLoadState('error');
                setStatusMessage(err instanceof Error ? err.message : 'setStack failed');
            }
        };

        void loadSource();

        // Initial tool
        handleToolChange('pan');

        const handleResize = () => {
            renderingEngine?.resize(true, false);
        };
        window.addEventListener('resize', handleResize);

        return () => {
            isCancelled = true;
            window.removeEventListener('resize', handleResize);
            // CRITICAL: Must disable element on unmount/remount so CS3D doesn't hold onto an orphaned Canvas
            try {
                renderingEngine.disableElement(viewportId);
                renderingEngine.destroy();
                renderingEngineRef.current = null;
            } catch (e) {
                console.warn('Silent fail on disableElement', e);
            }
        };
    }, [isInit, activeSource?.id, activeSource?.url, activeSource?.scheme]);

    useEffect(() => {
        const renderingEngine = renderingEngineRef.current;
        const viewport = renderingEngine?.getViewport(viewportIdRef.current) as cornerstone.Types.IStackViewport | undefined;
        applyViewportDisplayState(viewport, lastImageId);
        setOverlayMetadata(getRegisteredDicomMetadata(lastImageId));
        viewport?.render();
    }, [lastImageId]);

    useEffect(() => {
        const root = viewerRef.current;
        if (!root) return;

        const visualFilter = `${invert ? 'invert(1) ' : ''}brightness(${brightness}%) contrast(${contrast}%)`;
        const visualTransform = `rotate(${rotation}deg) scaleX(${flipped ? -1 : 1})`;
        const canvases = Array.from(root.querySelectorAll('canvas')) as HTMLCanvasElement[];
        canvases.forEach((canvas) => {
            canvas.style.filter = visualFilter;
            canvas.style.transform = visualTransform;
            canvas.style.transformOrigin = 'center center';
        });
    }, [invert, brightness, contrast, rotation, flipped, loadState, lastImageId, activeSourceId]);

    useEffect(() => {
        if (interactionMode !== 'magnifier') {
            setMagnifier((prev) => (prev.visible ? { ...prev, visible: false } : prev));
        }
    }, [interactionMode]);

    const drawMagnifier = (host: HTMLDivElement, localX: number, localY: number) => {
        const lensCanvas = lensCanvasRef.current;
        const sourceCanvas = host.querySelector('canvas') as HTMLCanvasElement | null;
        if (!lensCanvas || !sourceCanvas) return;

        const rect = host.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0 || sourceCanvas.width <= 0 || sourceCanvas.height <= 0) return;

        const lensSize = 126;
        const zoomFactor = 1.5;
        const dpr = window.devicePixelRatio || 1;
        if (lensCanvas.width !== Math.round(lensSize * dpr) || lensCanvas.height !== Math.round(lensSize * dpr)) {
            lensCanvas.width = Math.round(lensSize * dpr);
            lensCanvas.height = Math.round(lensSize * dpr);
            lensCanvas.style.width = `${lensSize}px`;
            lensCanvas.style.height = `${lensSize}px`;
        }

        const ctx = lensCanvas.getContext('2d');
        if (!ctx) return;

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, lensSize, lensSize);
        ctx.imageSmoothingEnabled = false;

        const sampleWidth = (lensSize / zoomFactor) * (sourceCanvas.width / rect.width);
        const sampleHeight = (lensSize / zoomFactor) * (sourceCanvas.height / rect.height);
        const centerX = (localX / rect.width) * sourceCanvas.width;
        const centerY = (localY / rect.height) * sourceCanvas.height;
        const sx = Math.max(0, Math.min(sourceCanvas.width - sampleWidth, centerX - sampleWidth / 2));
        const sy = Math.max(0, Math.min(sourceCanvas.height - sampleHeight, centerY - sampleHeight / 2));

        ctx.save();
        ctx.beginPath();
        ctx.arc(lensSize / 2, lensSize / 2, lensSize / 2 - 2, 0, Math.PI * 2);
        ctx.clip();
        ctx.drawImage(sourceCanvas, sx, sy, sampleWidth, sampleHeight, 0, 0, lensSize, lensSize);
        ctx.restore();
    };

    const handleMagnifierMove = (event: React.MouseEvent<HTMLDivElement>) => {
        if (interactionMode !== 'magnifier') return;
        const host = viewerRef.current;
        if (!host) return;
        const rect = host.getBoundingClientRect();
        const localX = event.clientX - rect.left;
        const localY = event.clientY - rect.top;
        if (localX < 0 || localY < 0 || localX > rect.width || localY > rect.height) {
            setMagnifier((prev) => (prev.visible ? { ...prev, visible: false } : prev));
            return;
        }
        setMagnifier({ visible: true, x: localX, y: localY });
        drawMagnifier(host, localX, localY);
    };

    const handleToolChange = (tool: ToolMode) => {
        setActiveToolState(tool);
        switch (tool) {
            case 'pan':
                setActiveTool(PanTool.toolName);
                break;
            case 'length':
                setActiveTool(LengthTool.toolName);
                break;
            case 'arrow':
                setActiveTool(ArrowAnnotateTool.toolName);
                break;
            case 'rect':
                setActiveTool(RectangleROITool.toolName);
                break;
            case 'ellipse':
                setActiveTool(EllipticalROITool.toolName);
                break;
            case 'wl':
                setActiveTool(WindowLevelTool.toolName);
                break;
        }
    };

    const resetView = () => {
        const renderingEngine = renderingEngineRef.current;
        const viewport = renderingEngine?.getViewport(viewportIdRef.current) as cornerstone.Types.IStackViewport;
        if (viewport) {
            viewport.resetCamera();
            viewport.resetProperties();
            applyViewportDisplayState(viewport, lastImageId);
            viewport.render();
        }
    };

    function applyAutoWindow() {
        const renderingEngine = renderingEngineRef.current;
        const viewport = renderingEngine?.getViewport(viewportIdRef.current) as cornerstone.Types.IStackViewport | undefined;
        if (!viewport || !lastImageId) return;

        const image = cornerstone.cache.getImage(lastImageId);
        const autoWindow = getAutoWindowForCornerstoneImage(image as any);
        if (!image || !autoWindow) return;

        viewport.setProperties({
            voiRange: {
                lower: autoWindow.level - autoWindow.width / 2,
                upper: autoWindow.level + autoWindow.width / 2,
            },
            invert: Boolean(image.invert),
        });
        syncOverlayWindow(autoWindow.level, autoWindow.width);
        viewport.render();
    }

    useEffect(() => {
        if (resetToken > 0) {
            resetView();
        }
    }, [resetToken]);

    useEffect(() => {
        if (autoWindowToken > 0) {
            applyAutoWindow();
        }
    }, [autoWindowToken, lastImageId]);

    const zoomInOut = (delta: number) => {
        const renderingEngine = renderingEngineRef.current;
        const viewport = renderingEngine?.getViewport(viewportIdRef.current) as cornerstone.Types.IStackViewport;
        if (viewport) {
            const camera = viewport.getCamera();
            // Zooming in CS3D modifies the parallelScale (smaller = zoomed in)
            if (camera.parallelScale) {
                const zoomFactor = delta < 0 ? 0.8 : 1.25; // Simple scale
                viewport.setCamera({ parallelScale: camera.parallelScale * zoomFactor });
                viewport.render();
            }
        }
    };

    if (!activeSource) return null;

    return (
        <div className="w-full h-full overflow-hidden rounded-3xl border border-white/10 bg-[#0B0D26] shadow-2xl flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-3 shrink-0">
                <div className="flex items-center gap-2 text-sm font-semibold text-white">
                    <Eye className="h-4 w-4 text-cyan-300" />
                    {title} <span className="text-xs text-gray-500">(Cornerstone3D)</span>
                </div>
                <div className="flex items-center gap-2">
                    {sources.map((source) => (
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

            {/* Info & View Controls */}
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-3 shrink-0">
                <div className="flex items-center gap-2 text-xs text-gray-300">
                    <ImageIcon className="h-4 w-4 text-cyan-300" />
                    {activeSource.label}
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={() => zoomInOut(1)}
                        className="rounded-lg bg-white/5 p-2 text-gray-300 transition-colors hover:bg-white/10 hover:text-white"
                        title="Zoom out"
                    >
                        <ZoomOut className="h-4 w-4" />
                    </button>
                    <button
                        onClick={() => zoomInOut(-1)}
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

            {/* Tool Controls */}
            {showToolbar && (
                <div className="flex flex-wrap items-center gap-2 border-b border-white/10 px-4 py-3 shrink-0">
                    <ToolChip
                        label="Pan"
                        active={activeTool === 'pan'}
                        icon={<Hand className="h-4 w-4" />}
                        onClick={() => handleToolChange('pan')}
                    />
                    <ToolChip
                        label="WL/WW"
                        active={activeTool === 'wl'}
                        icon={<Sliders className="h-4 w-4" />}
                        onClick={() => handleToolChange('wl')}
                    />
                    <ToolChip
                        label="Length"
                        active={activeTool === 'length'}
                        icon={<Ruler className="h-4 w-4" />}
                        onClick={() => handleToolChange('length')}
                    />
                    <ToolChip
                        label="Arrow"
                        active={activeTool === 'arrow'}
                        icon={<Eye className="h-4 w-4" />}
                        onClick={() => handleToolChange('arrow')}
                    />
                    <ToolChip
                        label="Rect ROI"
                        active={activeTool === 'rect'}
                        icon={<Square className="h-4 w-4" />}
                        onClick={() => handleToolChange('rect')}
                    />
                    <ToolChip
                        label="Ellipse ROI"
                        active={activeTool === 'ellipse'}
                        icon={<Eye className="h-4 w-4" />}
                        onClick={() => handleToolChange('ellipse')}
                    />
                    <ToolChip
                        label="Debug"
                        active={debugEnabled}
                        icon={<Bug className="h-4 w-4" />}
                        onClick={() => setDebugEnabled((prev) => !prev)}
                    />
                </div>
            )}

            <div className="flex-1 min-h-0 relative p-4">
                <div
                    ref={viewerRef}
                    className="w-full h-full bg-black rounded-xl overflow-hidden relative"
                    style={{ maxHeight: maxHeight ? `${maxHeight}px` : undefined, minHeight: '400px' }}
                    onMouseMove={handleMagnifierMove}
                    onMouseLeave={() => setMagnifier((prev) => (prev.visible ? { ...prev, visible: false } : prev))}
                    onContextMenu={(e) => e.preventDefault()} // prevent context menu for tools right click
                >
                    {!isInit && (
                        <div className="absolute inset-0 flex items-center justify-center text-white/50 z-10">
                            Initializing Cornerstone...
                        </div>
                    )}

                    {debugEnabled && (
                        <div className="absolute top-4 left-4 max-w-[calc(100%-2rem)] p-3 bg-black/60 text-emerald-400 text-xs pointer-events-none rounded z-10 space-y-1">
                            <div>CS3D Debug On</div>
                            <div>State: {loadState}</div>
                            <div>Source: {activeSource.label}</div>
                            <div>Scheme: {activeSource.scheme || 'web'}</div>
                            <div>ImageId: {lastImageId || '-'}</div>
                            <div>Status: {statusMessage}</div>
                        </div>
                    )}

                    <DicomMetadataOverlay
                        metadata={overlayMetadata}
                        top={debugEnabled ? 116 : 12}
                    />

                    {loadState === 'error' && (
                        <div className="absolute inset-x-4 bottom-4 rounded-xl border border-red-500/40 bg-black/70 px-4 py-3 text-sm text-red-200 z-10">
                            {statusMessage}
                        </div>
                    )}

                    {interactionMode === 'magnifier' && magnifier.visible && (
                        <div
                            className="pointer-events-none absolute z-20 overflow-hidden rounded-full border border-cyan-200/80 shadow-[0_14px_40px_rgba(0,0,0,0.38)]"
                            style={{
                                width: 126,
                                height: 126,
                                left: magnifier.x,
                                top: magnifier.y,
                                transform: 'translate(-50%, -50%)',
                                background: 'rgba(9, 17, 40, 0.18)',
                                boxShadow: '0 0 0 2px rgba(255,255,255,0.22), 0 10px 30px rgba(0,0,0,0.35)',
                            }}
                        >
                            <canvas ref={lensCanvasRef} className="block h-full w-full" />
                        </div>
                    )}
                </div>
            </div>
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

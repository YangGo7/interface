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
import { initCornerstone, registerLocalDicomFile, registerNativeDicomFile } from './cornerstone/init';
import { addAndGroupTools, createOrGetToolGroup, setActiveTool, TOOL_GROUP_ID } from './cornerstone/tools';

export type ViewerSource = {
    id: string;
    label: string;
    url: string;
    file?: File;
    scheme?: 'web' | 'wadouri' | 'dicomlocal' | 'dicomfile';
};

type CornerstoneViewerProps = {
    sources: ViewerSource[];
    initialSourceId?: string;
    title?: string;
    maxHeight?: number;
    is3D?: boolean;
    showToolbar?: boolean;
};

type ToolMode = 'pan' | 'length' | 'arrow' | 'rect' | 'ellipse' | 'wl';

export function CornerstoneViewer({
    sources,
    initialSourceId,
    title = 'Viewer',
    maxHeight = 560,
    is3D = false,
    showToolbar = true,
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

    const activeSource = sources.find((s) => s.id === activeSourceId) || sources[0];

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

        // Load image
        const imageId =
            scheme === 'dicomfile' && activeSource.file
                ? (dicomImageIdsRef.current[activeSource.id] ||= registerNativeDicomFile(activeSource.file))
                : scheme === 'dicomlocal' && activeSource.file
                    ? (dicomImageIdsRef.current[activeSource.id] ||= registerLocalDicomFile(activeSource.id, activeSource.file))
                    : `${scheme}:${activeSource.url}`;
        console.log('Cornerstone: Setting stack for', imageId);
        setLastImageId(imageId);
        setLoadState('loading');
        setStatusMessage(`Loading ${activeSource.label} automatically on GPU...`);

        let isCancelled = false;

        viewport.setStack([imageId])
            .then(() => {
                if (isCancelled) return;

                // Explicitly fall back to image's native min/max to prevent black screen on GPU
                const image = cornerstone.cache.getImage(imageId);
                if (image) {
                    const props: cornerstone.Types.ViewportProperties = {};
                    if (image.minPixelValue !== undefined && image.maxPixelValue !== undefined) {
                        props.voiRange = { lower: image.minPixelValue, upper: image.maxPixelValue };
                    }
                    if (image.invert) {
                        props.invert = true;
                    }
                    viewport.setProperties(props);
                }

                console.log('Cornerstone: Stack set, resetting camera...');
                renderingEngine.resize(true, false);
                viewport.resetCamera();
                viewport.render();
                setLoadState('ready');
                setStatusMessage(`Rendered ${activeSource.label}`);
            })
            .catch((err) => {
                console.error('Cornerstone: setStack failed', err);
                setLoadState('error');
                setStatusMessage(err instanceof Error ? err.message : 'setStack failed');
            });

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
            viewport.render();
        }
    };

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

                    {loadState === 'error' && (
                        <div className="absolute inset-x-4 bottom-4 rounded-xl border border-red-500/40 bg-black/70 px-4 py-3 text-sm text-red-200 z-10">
                            {statusMessage}
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

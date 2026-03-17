import React, { useEffect, useRef, useState } from 'react';
import * as cornerstone from '@cornerstonejs/core';
import * as cornerstoneTools from '@cornerstonejs/tools';
import {
    PanTool, LengthTool, ArrowAnnotateTool, RectangleROITool, EllipticalROITool,
    WindowLevelTool, TrackballRotateTool, EraserTool, StackScrollTool
} from '@cornerstonejs/tools';
import {
    Bug, Hand, RotateCcw, Ruler, Square, LayoutGrid,
    Rotate3d, Eraser, MousePointer2, Sliders, Trash2
} from 'lucide-react';
import { initCornerstone, registerNativeDicomFile, registerLocalDicomFile } from './cornerstone/init';
import { registerMultiframeDicomFile, generateMultiframeImageIds } from './cornerstone/multiframeLoader';
import { addAndGroupTools, createOrGetToolGroup, createOrGet3DToolGroup, setActiveTool, clearAllAnnotations, initToolEventListeners } from './cornerstone/tools';
import { ViewerSource } from './CornerstoneViewer';

type ToolMode = 'pan' | 'length' | 'arrow' | 'rect' | 'ellipse' | 'wl' | 'rotate' | 'erase' | 'scroll';
type ViewportOrientation = 'axial' | 'sagittal' | 'coronal' | 'acquisition' | 'volume3d';

type GridControl = {
    id: string;
    orientation: ViewportOrientation;
};

type CornerstoneGridViewerProps = {
    sources: ViewerSource[];
    title?: string;
    maxHeight?: number;
    showToolbar?: boolean;
    hideLayoutControls?: boolean;
    onToolChange?: (tool: ToolMode) => void;
    layout?: { rows: number; cols: number };
    onLayoutChange?: (layout: { rows: number; cols: number }) => void;
    invert?: boolean;
};

export function CornerstoneGridViewer({
    sources,
    title = 'MPR Grid Viewer',
    maxHeight = 600,
    showToolbar = true,
    hideLayoutControls = false,
    onToolChange,
    layout: externalLayout,
    onLayoutChange,
    invert = false,
}: CornerstoneGridViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const viewportRefs = useRef<(HTMLDivElement | null)[]>([]);
    const renderingEngineIdRef = useRef('grid-engine-' + Math.random().toString(36).slice(2));
    const volumeIdRef = useRef('volume-' + Math.random().toString(36).slice(2));
    const renderingEngineRef = useRef<cornerstone.RenderingEngine | null>(null);

    const [isInit, setIsInit] = useState(false);
    const [internalLayout, setInternalLayout] = useState({ rows: 2, cols: 2 });
    const layout = externalLayout || internalLayout;
    const setLayout = onLayoutChange || setInternalLayout;
    const [loadState, setLoadState] = useState<'initializing' | 'loading' | 'ready' | 'error'>('initializing');
    const [statusMessage, setStatusMessage] = useState('Initializing Cornerstone...');
    const [activeTool, setActiveToolState] = useState<ToolMode>('pan');
    const [debugEnabled, setDebugEnabled] = useState(false);
    const [currentImageId, setCurrentImageId] = useState('');

    const [viewportsConfig, setViewportsConfig] = useState<GridControl[]>([]);
    const [preset3D, setPreset3D] = useState<Record<string, string>>({});

    const PRESETS_3D = [
        { value: 'CT-Bone', label: 'CT Bone' },
        { value: 'CT-Soft-Tissue', label: 'CT Soft Tissue' },
        { value: 'CT-Lung', label: 'CT Lung' },
        { value: 'CT-Fat', label: 'CT Fat' },
        { value: 'CT-Muscle', label: 'CT Muscle' },
        { value: 'CT-Cardiac', label: 'CT Cardiac' },
        { value: 'CT-MIP', label: 'CT MIP' },
        { value: 'CT-Chest-Contrast-Enhanced', label: 'CT Chest CE' },
        { value: 'CT-Chest-Vessels', label: 'CT Vessels' },
        { value: 'CT-Air', label: 'CT Air' },
        { value: 'MR-Default', label: 'MR Default' },
        { value: 'MR-MIP', label: 'MR MIP' },
    ];

    const activeSource = sources[0];

    function applyViewportDisplayState(renderingEngine: cornerstone.RenderingEngine | null, imageId?: string) {
        if (!renderingEngine || !imageId) return;

        const image = cornerstone.cache.getImage(imageId);
        if (!image) return;

        const props: cornerstone.Types.ViewportProperties = {};
        if (image.windowCenter != null && image.windowWidth != null) {
            const wc = Array.isArray(image.windowCenter) ? image.windowCenter[0] : image.windowCenter;
            const ww = Array.isArray(image.windowWidth) ? image.windowWidth[0] : image.windowWidth;
            if (typeof wc === 'number' && typeof ww === 'number' && ww > 0) {
                props.voiRange = { lower: wc - ww / 2, upper: wc + ww / 2 };
            }
        } else if (image.minPixelValue !== undefined && image.maxPixelValue !== undefined) {
            props.voiRange = { lower: image.minPixelValue, upper: image.maxPixelValue };
        }

        props.invert = Boolean(image.invert) !== invert;
        viewportsConfig.forEach(config => {
            if (config.orientation === 'volume3d') return;
            const vp = renderingEngine.getViewport(config.id) as any;
            if (vp?.setProperties) {
                vp.setProperties(props);
                vp.render();
            }
        });
    }

    useEffect(() => {
        initCornerstone()
            .then(() => {
                addAndGroupTools();
                createOrGetToolGroup();
                initToolEventListeners();
                setIsInit(true);
                setLoadState('loading');
                setStatusMessage('Cornerstone initialized. Constructing grid...');
            })
            .catch((error) => {
                console.error('Cornerstone init failed', error);
                setLoadState('error');
                setStatusMessage(error instanceof Error ? error.message : 'Cornerstone initialization failed.');
            });
    }, []);

    useEffect(() => {
        const numViewports = layout.rows * layout.cols;
        const initialOrientations: ViewportOrientation[] = ['axial', 'sagittal', 'coronal', 'volume3d'];
        const newConfigs: GridControl[] = [];
        for (let i = 0; i < numViewports; i++) {
            newConfigs.push({
                id: 'viewport-' + i,
                orientation: initialOrientations[i % initialOrientations.length]
            });
        }
        setViewportsConfig(newConfigs);
    }, [layout]);

    useEffect(() => {
        if (!isInit || !activeSource || viewportsConfig.length === 0) return;

        const scheme = activeSource.scheme || 'web';
        const imageId = scheme === 'dicomfile' && activeSource.file
            ? (registerNativeDicomFile(activeSource.file))
            : scheme === 'dicomlocal' && activeSource.file
                ? (registerLocalDicomFile(activeSource.id, activeSource.file))
                : scheme + ':' + activeSource.url;
        setCurrentImageId(imageId);

        const renderingEngineId = renderingEngineIdRef.current;
        const volumeId = 'streaming-volume:' + volumeIdRef.current;

        try { renderingEngineRef.current?.destroy(); } catch (e) { }

        const renderingEngine = new cornerstone.RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;

        const validRefs = viewportRefs.current.filter(ref => ref !== null);
        if (validRefs.length !== viewportsConfig.length) return;

        const isVolume = scheme !== 'web';

        const viewportInputs = viewportsConfig.map((config, i) => {
            const is3D = config.orientation === 'volume3d';
            return {
                viewportId: config.id,
                type: isVolume
                    ? (is3D ? cornerstone.Enums.ViewportType.VOLUME_3D : cornerstone.Enums.ViewportType.ORTHOGRAPHIC)
                    : cornerstone.Enums.ViewportType.STACK,
                element: validRefs[i] as HTMLDivElement,
                defaultOptions: {
                    orientation: (isVolume && !is3D)
                        ? cornerstone.Enums.OrientationAxis[config.orientation.toUpperCase() as keyof typeof cornerstone.Enums.OrientationAxis]
                        : undefined,
                    background: (is3D ? [0.1, 0.1, 0.15] : [0, 0, 0]) as cornerstone.Types.Point3,
                },
            };
        });

        renderingEngine.setViewports(viewportInputs);

        const voiSynchronizerId = 'voi-sync-' + renderingEngineId;
        cornerstoneTools.SynchronizerManager.destroySynchronizer(voiSynchronizerId);
        const voiSync = cornerstoneTools.synchronizers.createVOISynchronizer(voiSynchronizerId, { syncInvertState: false, syncColormap: false });

        const toolGroup = createOrGetToolGroup();
        const toolGroup3D = createOrGet3DToolGroup();
        viewportsConfig.forEach(config => {
            if (config.orientation === 'volume3d') {
                if (toolGroup3D && !toolGroup3D.getViewportIds().includes(config.id)) {
                    toolGroup3D.addViewport(config.id, renderingEngineId);
                }
            } else {
                if (toolGroup && !toolGroup.getViewportIds().includes(config.id)) {
                    toolGroup.addViewport(config.id, renderingEngineId);
                }
                voiSync.add({ renderingEngineId, viewportId: config.id });
            }
        });

        let isCancelled = false;
        const loadVolume = async () => {
            setLoadState('loading');
            setStatusMessage('Loading ' + (isVolume ? 'Volume' : 'Stack') + '...');
            try {
                if (isVolume) {
                    let allImageIds: string[] = [];
                    if (activeSource.file) {
                        const reg = await registerMultiframeDicomFile(activeSource.file);
                        if (isCancelled) return;
                        if (reg.numberOfFrames > 1) { allImageIds = generateMultiframeImageIds(reg); }
                        else { await cornerstone.imageLoader.loadAndCacheImage(imageId); if (isCancelled) return; allImageIds = [imageId]; }
                    } else { await cornerstone.imageLoader.loadAndCacheImage(imageId); if (isCancelled) return; allImageIds = [imageId]; }
                    const volume = await cornerstone.volumeLoader.createAndCacheVolume(volumeId, { imageIds: allImageIds });
                    if (isCancelled) return;
                    await volume.load();
                    if (isCancelled) return;
                    await cornerstone.setVolumesForViewports(renderingEngine, [{ volumeId }], viewportsConfig.map(v => v.id));
                } else {
                    for (const config of viewportsConfig) {
                        const vp = renderingEngine.getViewport(config.id) as cornerstone.Types.IStackViewport;
                        await vp.setStack([imageId]);
                    }
                }
                if (isCancelled) return;
                applyViewportDisplayState(renderingEngine, imageId);
                // Post-load: Restore presets and reset cameras
                viewportsConfig.forEach(config => {
                    const vp = renderingEngine.getViewport(config.id) as any;
                    if (!vp) return;

                    if (config.orientation === 'volume3d') {
                        try {
                            vp.setProperties({ preset: preset3D[config.id] || 'CT-Bone' });
                        } catch (e) {
                            console.warn('Failed to set 3D preset', e);
                        }
                    }

                    vp.resetCamera();
                    vp.render();
                });

                // Final global render pass after a short delay for stability
                setTimeout(() => {
                    if (!isCancelled) {
                        renderingEngine.render();
                    }
                }, 100);

                setLoadState('ready');
                setStatusMessage('Ready');
            } catch (err) {
                if (isCancelled) return;
                setLoadState('error');
                setStatusMessage(err instanceof Error ? err.message : 'Rendering failed');
            }
        };

        handleToolChange('pan');
        loadVolume();

        const handleResize = () => { renderingEngine.resize(true, false); };
        window.addEventListener('resize', handleResize);

        return () => {
            isCancelled = true;
            window.removeEventListener('resize', handleResize);
            try {
                viewportsConfig.forEach(config => renderingEngine.disableElement(config.id));
                renderingEngine.destroy();
                renderingEngineRef.current = null;
                cornerstoneTools.SynchronizerManager.destroySynchronizer(voiSynchronizerId);
                cornerstone.cache.removeVolumeLoadObject(volumeId);
            } catch (e) { }
        };
    }, [isInit, activeSource?.id, activeSource?.url, activeSource?.scheme, JSON.stringify(viewportsConfig)]);

    useEffect(() => {
        applyViewportDisplayState(renderingEngineRef.current, currentImageId);
    }, [invert, currentImageId, viewportsConfig]);

    const handleToolChange = (tool: ToolMode) => {
        setActiveToolState(tool);
        onToolChange?.(tool);
        switch (tool) {
            case 'pan': setActiveTool(PanTool.toolName); break;
            case 'length': setActiveTool(LengthTool.toolName); break;
            case 'arrow': setActiveTool(ArrowAnnotateTool.toolName); break;
            case 'rect': setActiveTool(RectangleROITool.toolName); break;
            case 'ellipse': setActiveTool(EllipticalROITool.toolName); break;
            case 'wl': setActiveTool(WindowLevelTool.toolName); break;
            case 'rotate': setActiveTool(TrackballRotateTool.toolName); break;
            case 'erase': setActiveTool(EraserTool.toolName); break;
            case 'scroll': setActiveTool(StackScrollTool.toolName); break;
        }
    };

    const handleOrientationChange = (viewportId: string, newOrientation: ViewportOrientation) => {
        setViewportsConfig(prev => prev.map(c => c.id === viewportId ? { ...c, orientation: newOrientation } : c));
    };

    const handlePresetChange = (viewportId: string, presetName: string) => {
        const renderingEngine = renderingEngineRef.current;
        if (!renderingEngine) return;
        const vp = renderingEngine.getViewport(viewportId) as any;
        if (vp?.setProperties) { try { vp.setProperties({ preset: presetName }); vp.render(); setPreset3D(prev => ({ ...prev, [viewportId]: presetName })); } catch (e) { } }
    };

    const resetView = () => {
        const renderingEngine = renderingEngineRef.current;
        if (!renderingEngine) return;
        viewportsConfig.forEach(config => {
            const vp = renderingEngine.getViewport(config.id) as any;
            if (vp) {
                vp.resetCamera();
                vp.resetProperties?.();
                if (config.orientation === 'volume3d') {
                    vp.render?.();
                }
            }
        });
        applyViewportDisplayState(renderingEngine, currentImageId);
    };

    return (
        <div className="w-full h-full flex flex-col rounded-2xl border border-gray-200 bg-white overflow-hidden shadow-sm">
            {showToolbar && (
                <div className="flex items-center justify-between border-b border-gray-100 bg-gray-50/80 px-4 py-3 backdrop-blur-sm">
                    <div className="flex items-center gap-3">
                        <h3 className="font-semibold text-gray-800">{title}</h3>
                        <div className="flex items-center gap-1 rounded-lg bg-white p-1 shadow-sm border border-gray-100 ml-4">
                            {[
                                { id: 'pan', icon: Hand, label: 'Pan' },
                                { id: 'wl', icon: Sliders, label: 'W/L' },
                                { id: 'length', icon: Ruler, label: 'Length' },
                                { id: 'rect', icon: Square, label: 'Rect' },
                                { id: 'rotate', icon: Rotate3d, label: 'Rotate' },
                                { id: 'scroll', icon: MousePointer2, label: 'Scroll' },
                                { id: 'erase', icon: Eraser, label: 'Eraser' },
                            ].map((tool) => (
                                <button key={tool.id} onClick={() => handleToolChange(tool.id as ToolMode)} className={`p-2 rounded-md transition-all ${activeTool === tool.id ? 'bg-blue-50 text-blue-600 shadow-sm' : 'text-gray-500 hover:bg-gray-50 hover:text-gray-900'}`} title={tool.label}>
                                    <tool.icon className="h-4 w-4" />
                                </button>
                            ))}
                            <button
                                onClick={() => {
                                    if (window.confirm('모든 드로잉을 삭제하시겠습니까?')) {
                                        clearAllAnnotations();
                                    }
                                }}
                                className="p-2 rounded-md transition-all text-red-400 hover:bg-red-50 hover:text-red-600 border-l border-gray-100 ml-1"
                                title="Clear All Drawings"
                            >
                                <Trash2 className="h-4 w-4" />
                            </button>
                        </div>

                        {/* Integrated Orientation Selector for 1x1 layout */}
                        {viewportsConfig.length === 1 && (
                            <div className="flex items-center gap-2 ml-4">
                                <span className="text-xs font-medium text-gray-500">View:</span>
                                <select
                                    className="text-xs font-semibold bg-white border border-gray-200 rounded-md px-2 py-1 outline-none text-blue-600 shadow-sm"
                                    value={viewportsConfig[0].orientation}
                                    onChange={(e) => handleOrientationChange(viewportsConfig[0].id, e.target.value as ViewportOrientation)}
                                >
                                    <option value="axial">Axial</option>
                                    <option value="sagittal">Sagittal</option>
                                    <option value="coronal">Coronal</option>
                                    <option value="acquisition">Original</option>
                                    <option value="volume3d">3D Volume</option>
                                </select>
                                {viewportsConfig[0].orientation === 'volume3d' && (
                                    <select
                                        className="text-xs font-semibold bg-blue-50 border border-blue-200 rounded-md px-2 py-1 outline-none text-blue-700 shadow-sm"
                                        value={preset3D[viewportsConfig[0].id] || 'CT-Bone'}
                                        onChange={(e) => handlePresetChange(viewportsConfig[0].id, e.target.value)}
                                    >
                                        {PRESETS_3D.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                                    </select>
                                )}
                            </div>
                        )}
                    </div>
                    <div className="flex items-center gap-2">
                        {!hideLayoutControls && (
                            <div className="flex items-center bg-white rounded-lg shadow-sm border border-gray-100 p-1 mr-4">
                                <button onClick={() => setLayout({ rows: 1, cols: 1 })} className={`p-1.5 rounded ${layout.rows === 1 && layout.cols === 1 ? 'bg-blue-50 text-blue-600' : 'text-gray-500'}`} title="1x1"><LayoutGrid className="w-4 h-4" /></button>
                                <button onClick={() => setLayout({ rows: 2, cols: 2 })} className={`p-1.5 rounded ${layout.rows === 2 && layout.cols === 2 ? 'bg-blue-50 text-blue-600' : 'text-gray-500'}`} title="2x2"><LayoutGrid className="w-4 h-4" /></button>
                            </div>
                        )}
                        <button onClick={resetView} className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-gray-600 hover:bg-gray-100"><RotateCcw className="h-4 w-4" /> Reset</button>
                        <button onClick={() => setDebugEnabled(!debugEnabled)} className={`p-2 rounded-lg transition-colors ${debugEnabled ? 'bg-emerald-100 text-emerald-700' : 'text-gray-400 hover:bg-gray-100'}`} title="Toggle Debug Info"><Bug className="h-4 w-4" /></button>
                    </div>
                </div>
            )}

            <div ref={containerRef} style={{ position: 'relative', backgroundColor: '#000', width: '100%', height: maxHeight + 'px', overflow: 'hidden' }} onContextMenu={(e) => e.preventDefault()}>
                {!isInit && <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#9ca3af', backgroundColor: '#111827', zIndex: 50 }}>Initializing...</div>}
                <div style={{ position: 'absolute', inset: 0, display: 'grid', gap: '2px', backgroundColor: '#4b5563', gridTemplateColumns: 'repeat(' + layout.cols + ', 1fr)', gridTemplateRows: 'repeat(' + layout.rows + ', 1fr)' }}>
                    {viewportsConfig.map((config, idx) => (
                        <div key={config.id} style={{ position: 'relative', backgroundColor: '#000', width: '100%', height: '100%', overflow: 'hidden' }}>
                            <div ref={(el) => { viewportRefs.current[idx] = el; }} style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, width: '100%', height: '100%' }} />
                            <div
                                style={{
                                    position: 'absolute', top: 8, right: 8, zIndex: 10,
                                    opacity: viewportsConfig.length === 1 ? 1 : 0,
                                    transition: 'opacity 0.2s'
                                }}
                                onMouseEnter={(e) => { if (viewportsConfig.length > 1) e.currentTarget.style.opacity = '1'; }}
                                onMouseLeave={(e) => { if (viewportsConfig.length > 1) e.currentTarget.style.opacity = '0'; }}
                            >
                                <select style={{ backgroundColor: 'rgba(0,0,0,0.6)', color: '#fff', fontSize: '12px', padding: '2px 8px', borderRadius: '4px', border: '1px solid #4b5563', outline: 'none' }} value={config.orientation} onChange={(e) => handleOrientationChange(config.id, e.target.value as ViewportOrientation)}>
                                    <option value="axial">Axial</option><option value="sagittal">Sagittal</option><option value="coronal">Coronal</option><option value="acquisition">Original</option><option value="volume3d">3D Volume</option>
                                </select>
                                {config.orientation === 'volume3d' && (
                                    <select style={{ backgroundColor: 'rgba(0,0,0,0.6)', color: '#60a5fa', fontSize: '11px', padding: '2px 6px', borderRadius: '4px', border: '1px solid #3b82f6', outline: 'none', marginLeft: '4px' }} value={preset3D[config.id] || 'CT-Bone'} onChange={(e) => handlePresetChange(config.id, e.target.value)}>
                                        {PRESETS_3D.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                                    </select>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

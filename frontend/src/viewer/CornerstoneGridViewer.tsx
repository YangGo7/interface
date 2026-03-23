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

type VolumePresetDefinition = {
    name: string;
    gradientOpacity: string;
    specularPower: string;
    scalarOpacity: string;
    specular: string;
    shade: string;
    ambient: string;
    colorTransfer: string;
    diffuse: string;
    interpolation: string;
};

type VolumePresetOption = {
    value: string;
    label: string;
    preset: VolumePresetDefinition;
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
    interactionMode?: string;
    resetToken?: number;
    onViewportCapture?: (canvas: HTMLCanvasElement, viewportLabel?: string) => void;
    brightness?: number;
    contrast?: number;
    rotation?: number;
    flipped?: boolean;
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
    interactionMode,
    resetToken = 0,
    onViewportCapture,
    brightness = 100,
    contrast = 100,
    rotation = 0,
    flipped = false,
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
    const magnifierCanvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({});
    const [magnifier, setMagnifier] = useState<{ visible: boolean; viewportId: string | null; x: number; y: number }>({
        visible: false,
        viewportId: null,
        x: 0,
        y: 0,
    });

    const [viewportsConfig, setViewportsConfig] = useState<GridControl[]>([]);
    const [preset3D, setPreset3D] = useState<Record<string, string>>({});

    const PRESETS_3D: VolumePresetOption[] = [
        {
            value: 'dental-bone',
            label: 'Dental Bone',
            preset: {
                name: 'Dental Bone',
                gradientOpacity: '4 0 1 255 1',
                specularPower: '10',
                scalarOpacity: '8 -3024 0 -16.4458 0 641.385 0.715686 3071 0.705882',
                specular: '0.2',
                shade: '1',
                ambient: '0.1',
                colorTransfer: '16 -3024 0 0 0 -16.4458 0.729412 0.254902 0.301961 641.385 0.905882 0.815686 0.552941 3071 1 1 1',
                diffuse: '0.9',
                interpolation: '1',
            },
        },
        {
            value: 'dental-surface',
            label: 'Dental Surface',
            preset: {
                name: 'Dental Surface',
                gradientOpacity: '4 0 1 255 1',
                specularPower: '1',
                scalarOpacity: '10 -2048 0 -451 0 -450 1 1050 1 3661 1',
                specular: '0',
                shade: '0',
                ambient: '0.2',
                colorTransfer: '20 -2048 0 0 0 -451 0 0 0 -450 0.0556356 0.0556356 0.0556356 1050 1 1 1 3661 1 1 1',
                diffuse: '1',
                interpolation: '1',
            },
        },
        {
            value: 'dental-soft-tissue',
            label: 'Dental Soft Tissue',
            preset: {
                name: 'Dental Soft Tissue',
                gradientOpacity: '4 0 1 255 1',
                specularPower: '1',
                scalarOpacity: '10 -2048 0 -167.01 0 -160 1 240 1 3661 1',
                specular: '0',
                shade: '0',
                ambient: '0.2',
                colorTransfer: '20 -2048 0 0 0 -167.01 0 0 0 -160 0.0556356 0.0556356 0.0556356 240 1 1 1 3661 1 1 1',
                diffuse: '1',
                interpolation: '1',
            },
        },
        {
            value: 'dental-mip',
            label: 'Dental MIP',
            preset: {
                name: 'Dental MIP',
                gradientOpacity: '4 0 1 255 1',
                specularPower: '10',
                scalarOpacity: '8 -3024 0 -637.62 0 700 1 3071 1',
                specular: '0.2',
                shade: '1',
                ambient: '0.1',
                colorTransfer: '16 -3024 0 0 0 -637.62 1 1 1 700 1 1 1 3071 1 1 1',
                diffuse: '0.9',
                interpolation: '1',
            },
        },
    ];
    const DEFAULT_3D_PRESET = PRESETS_3D[0].value;

    const getPresetOption = (presetValue?: string) =>
        PRESETS_3D.find((preset) => preset.value === presetValue) || PRESETS_3D[0];

    const activeSource = sources[0];

    const getViewportCanvas = (viewportId: string, fallbackRoot?: ParentNode | null) => {
        const viewport = renderingEngineRef.current?.getViewport(viewportId) as any;
        const directCanvas = viewport?.getCanvas?.() || viewport?.canvas;
        if (directCanvas instanceof HTMLCanvasElement && directCanvas.width > 0 && directCanvas.height > 0) {
            return directCanvas;
        }
        return getPrimaryCanvas(fallbackRoot);
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

        props.invert = Boolean(image.invert);
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
                            vp.setProperties({ preset: getPresetOption(preset3D[config.id] || DEFAULT_3D_PRESET).preset });
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
    }, [currentImageId, viewportsConfig]);

    useEffect(() => {
        const visualFilter = `${invert ? 'invert(1) ' : ''}brightness(${brightness}%) contrast(${contrast}%)`;
        const visualTransform = `rotate(${rotation}deg) scaleX(${flipped ? -1 : 1})`;

        viewportRefs.current.forEach((host) => {
            if (!host) return;
            const canvases = Array.from(host.querySelectorAll('canvas')) as HTMLCanvasElement[];
            canvases.forEach((canvas) => {
                canvas.style.filter = visualFilter;
                canvas.style.transform = visualTransform;
                canvas.style.transformOrigin = 'center center';
            });
        });
    }, [invert, brightness, contrast, rotation, flipped, loadState, currentImageId, viewportsConfig]);

    useEffect(() => {
        if (interactionMode !== 'magnifier') {
            setMagnifier((prev) => (prev.visible ? { ...prev, visible: false } : prev));
        }
    }, [interactionMode]);

    const drawMagnifier = (viewportId: string, host: HTMLDivElement, localX: number, localY: number) => {
        const lensCanvas = magnifierCanvasRefs.current[viewportId];
        const sourceCanvas = getViewportCanvas(viewportId, host);
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

    const handleMagnifierMove = (viewportId: string, orientation: ViewportOrientation, event: React.MouseEvent<HTMLDivElement>) => {
        if (interactionMode !== 'magnifier' || orientation === 'volume3d') return;
        const host = viewportRefs.current[viewportsConfig.findIndex((c) => c.id === viewportId)];
        if (!host) return;
        const rect = host.getBoundingClientRect();
        const localX = event.clientX - rect.left;
        const localY = event.clientY - rect.top;
        if (localX < 0 || localY < 0 || localX > rect.width || localY > rect.height) {
            setMagnifier((prev) => (prev.visible && prev.viewportId === viewportId ? { ...prev, visible: false } : prev));
            return;
        }
        setMagnifier({ visible: true, viewportId, x: localX, y: localY });
        drawMagnifier(viewportId, host, localX, localY);
    };

    const handleViewportCaptureClick = (viewportId: string, orientation: ViewportOrientation, idx: number, event: React.MouseEvent<HTMLElement>) => {
        if (interactionMode !== 'capture-area' || orientation === 'volume3d') return;
        event.preventDefault();
        event.stopPropagation();
        const host = viewportRefs.current[idx];
        const sourceCanvas = getViewportCanvas(viewportId, host) || getPrimaryCanvas(containerRef.current);
        if (sourceCanvas && onViewportCapture && host) {
            const hostRect = host.getBoundingClientRect();
            const canvasRect = sourceCanvas.getBoundingClientRect();
            const intersectLeft = Math.max(hostRect.left, canvasRect.left);
            const intersectTop = Math.max(hostRect.top, canvasRect.top);
            const intersectRight = Math.min(hostRect.right, canvasRect.right);
            const intersectBottom = Math.min(hostRect.bottom, canvasRect.bottom);
            const intersectWidth = intersectRight - intersectLeft;
            const intersectHeight = intersectBottom - intersectTop;

            if (intersectWidth <= 1 || intersectHeight <= 1) {
                onViewportCapture(sourceCanvas, orientation);
                return;
            }

            const scaleX = sourceCanvas.width / canvasRect.width;
            const scaleY = sourceCanvas.height / canvasRect.height;
            const sx = Math.max(0, (intersectLeft - canvasRect.left) * scaleX);
            const sy = Math.max(0, (intersectTop - canvasRect.top) * scaleY);
            const sw = Math.max(1, intersectWidth * scaleX);
            const sh = Math.max(1, intersectHeight * scaleY);

            const croppedCanvas = document.createElement('canvas');
            croppedCanvas.width = Math.round(sw);
            croppedCanvas.height = Math.round(sh);
            const croppedCtx = croppedCanvas.getContext('2d');
            if (!croppedCtx) {
                onViewportCapture(sourceCanvas, orientation);
                return;
            }

            croppedCtx.drawImage(
                sourceCanvas,
                sx,
                sy,
                sw,
                sh,
                0,
                0,
                croppedCanvas.width,
                croppedCanvas.height
            );
            onViewportCapture(croppedCanvas, orientation);
        }
    };

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
        const preset = getPresetOption(presetName);
        if (vp?.setProperties) {
            try {
                vp.setProperties({ preset: preset.preset });
                vp.render();
                setPreset3D((prev) => ({ ...prev, [viewportId]: preset.value }));
            } catch (e) { }
        }
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

    useEffect(() => {
        if (resetToken > 0) {
            resetView();
        }
    }, [resetToken]);

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
                                        value={preset3D[viewportsConfig[0].id] || DEFAULT_3D_PRESET}
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
                        <div
                            key={config.id}
                            data-grid-capture-cell="true"
                            style={{ position: 'relative', backgroundColor: '#000', width: '100%', height: '100%', overflow: 'hidden', cursor: interactionMode === 'magnifier' && config.orientation !== 'volume3d' ? 'zoom-in' : interactionMode === 'capture-area' && config.orientation !== 'volume3d' ? 'crosshair' : 'default' }}
                            onMouseDown={(event) => {
                                if (interactionMode !== 'capture-area' || config.orientation === 'volume3d') return;
                                event.preventDefault();
                                event.stopPropagation();
                            }}
                            onClick={(event) => {
                                if (interactionMode !== 'capture-area' || config.orientation === 'volume3d') return;
                                handleViewportCaptureClick(config.id, config.orientation, idx, event);
                            }}
                            onMouseMove={(event) => handleMagnifierMove(config.id, config.orientation, event)}
                            onMouseLeave={() => setMagnifier((prev) => (prev.visible && prev.viewportId === config.id ? { ...prev, visible: false } : prev))}
                        >
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
                                    <select style={{ backgroundColor: 'rgba(0,0,0,0.6)', color: '#60a5fa', fontSize: '11px', padding: '2px 6px', borderRadius: '4px', border: '1px solid #3b82f6', outline: 'none', marginLeft: '4px' }} value={preset3D[config.id] || DEFAULT_3D_PRESET} onChange={(e) => handlePresetChange(config.id, e.target.value)}>
                                        {PRESETS_3D.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                                    </select>
                                )}
                            </div>
                            {interactionMode === 'capture-area' && config.orientation !== 'volume3d' && (
                                <div
                                    className="pointer-events-none absolute inset-0 z-[11] border border-cyan-300/70 bg-cyan-400/5 shadow-[inset_0_0_0_1px_rgba(103,232,249,0.35)] transition-colors"
                                    aria-hidden="true"
                                >
                                    <span className="pointer-events-none absolute left-3 top-3 rounded-full border border-cyan-200/60 bg-slate-950/70 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-100">
                                        Click To Capture
                                    </span>
                                </div>
                            )}
                            {interactionMode === 'magnifier' && config.orientation !== 'volume3d' && magnifier.visible && magnifier.viewportId === config.id && (
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
                                    <canvas ref={(el) => { magnifierCanvasRefs.current[config.id] = el; }} className="block h-full w-full" />
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

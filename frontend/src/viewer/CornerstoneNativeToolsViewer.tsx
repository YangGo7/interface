import React, { useEffect, useRef, useState } from 'react';
import * as cornerstone from '@cornerstonejs/core';
import {
    addTool,
    ToolGroupManager,
    PanTool,
    ZoomTool,
    WindowLevelTool,
    LengthTool,
    ArrowAnnotateTool,
    RectangleROITool,
    EllipticalROITool,
    Enums as ToolEnums,
} from '@cornerstonejs/tools';
import { initCornerstone, registerNativeDicomFile } from './cornerstone/init';
import type { LocalDicomInspection } from './cornerstone/dicomDebug';

type CornerstoneNativeToolsViewerProps = {
    file?: File;
    imageUrl?: string;
    imageLabel?: string;
    scheme?: 'dicomfile' | 'web';
    inspection?: LocalDicomInspection | null;
    title?: string;
    maxHeight?: number;
};

type NativeToolMode = 'pan' | 'zoom' | 'wl' | 'length' | 'arrow' | 'rect' | 'ellipse';

let nativeToolsAdded = false;

const ensureNativeToolsAdded = () => {
    if (nativeToolsAdded) {
        return;
    }

    addTool(PanTool);
    addTool(ZoomTool);
    addTool(WindowLevelTool);
    addTool(LengthTool);
    addTool(ArrowAnnotateTool);
    addTool(RectangleROITool);
    addTool(EllipticalROITool);

    nativeToolsAdded = true;
};

const buildVoiRange = (inspection?: LocalDicomInspection | null) => {
    if (inspection?.windowCenter == null || inspection?.windowWidth == null) {
        return null;
    }

    const lower = inspection.windowCenter - inspection.windowWidth / 2;
    const upper = inspection.windowCenter + inspection.windowWidth / 2;

    if (!Number.isFinite(lower) || !Number.isFinite(upper) || lower === upper) {
        return null;
    }

    return { lower, upper };
};

const TOOL_NAME_MAP: Record<NativeToolMode, string> = {
    pan: PanTool.toolName,
    zoom: ZoomTool.toolName,
    wl: WindowLevelTool.toolName,
    length: LengthTool.toolName,
    arrow: ArrowAnnotateTool.toolName,
    rect: RectangleROITool.toolName,
    ellipse: EllipticalROITool.toolName,
};

const TOOL_BUTTONS: Array<{ id: NativeToolMode; label: string }> = [
    { id: 'pan', label: 'Pan' },
    { id: 'zoom', label: 'Zoom' },
    { id: 'wl', label: 'W/L' },
    { id: 'length', label: 'Length' },
    { id: 'arrow', label: 'Arrow' },
    { id: 'rect', label: 'Rect ROI' },
    { id: 'ellipse', label: 'Ellipse ROI' },
];

const setPrimaryTool = (toolGroupId: string, mode: NativeToolMode) => {
    const toolGroup = ToolGroupManager.getToolGroup(toolGroupId);
    if (!toolGroup) {
        return;
    }

    Object.values(TOOL_NAME_MAP).forEach((toolName) => {
        if (toolGroup.hasTool(toolName)) {
            toolGroup.setToolPassive(toolName);
        }
    });

    toolGroup.setToolActive(TOOL_NAME_MAP[mode], {
        bindings: [{ mouseButton: ToolEnums.MouseBindings.Primary }],
    });
    toolGroup.setToolActive(ZoomTool.toolName, {
        bindings: [{ mouseButton: ToolEnums.MouseBindings.Secondary }],
    });
    toolGroup.setToolActive(WindowLevelTool.toolName, {
        bindings: [{ mouseButton: ToolEnums.MouseBindings.Auxiliary }],
    });
};

export function CornerstoneNativeToolsViewer({
    file,
    imageUrl,
    imageLabel,
    scheme = 'dicomfile',
    inspection,
    title = 'Cornerstone Native Tools Viewer',
    maxHeight = 520,
}: CornerstoneNativeToolsViewerProps) {
    const viewerRef = useRef<HTMLDivElement>(null);
    const renderingEngineRef = useRef<cornerstone.RenderingEngine | null>(null);
    const toolGroupIdRef = useRef(`native-tool-group-${Math.random().toString(36).slice(2)}`);
    const renderingEngineIdRef = useRef(`native-tools-engine-${Math.random().toString(36).slice(2)}`);
    const viewportIdRef = useRef(`native-tools-viewport-${Math.random().toString(36).slice(2)}`);
    const imageIdRef = useRef('');
    const [state, setState] = useState<'initializing' | 'loading' | 'ready' | 'error'>('initializing');
    const [status, setStatus] = useState('Initializing Cornerstone...');
    const [activeTool, setActiveTool] = useState<NativeToolMode>('pan');

    useEffect(() => {
        imageIdRef.current = '';
    }, [file, imageUrl, scheme]);

    useEffect(() => {
        let isCancelled = false;

        initCornerstone()
            .then(() => {
                if (isCancelled) {
                    return;
                }

                ensureNativeToolsAdded();
                setState('loading');
                setStatus('Cornerstone initialized. Native tools viewer is preparing...');
            })
            .catch((error) => {
                if (isCancelled) {
                    return;
                }
                console.error('Native tools viewer init failed', error);
                setState('error');
                setStatus(error instanceof Error ? error.message : 'Cornerstone initialization failed.');
            });

        return () => {
            isCancelled = true;
        };
    }, []);

    useEffect(() => {
        if (state === 'initializing' || state === 'error' || !viewerRef.current) {
            return;
        }

        const element = viewerRef.current;
        const renderingEngineId = renderingEngineIdRef.current;
        const viewportId = viewportIdRef.current;
        const toolGroupId = toolGroupIdRef.current;
        if (scheme === 'dicomfile' && !file) {
            setState('error');
            setStatus('DICOM file is required for dicomfile scheme.');
            return;
        }

        if (scheme === 'web' && !imageUrl) {
            setState('error');
            setStatus('Image URL is required for web scheme.');
            return;
        }

        const imageId = imageIdRef.current || (
            scheme === 'dicomfile' && file
                ? registerNativeDicomFile(file)
                : `web:${imageUrl}`
        );
        imageIdRef.current = imageId;

        try {
            renderingEngineRef.current?.destroy();
        } catch (error) {
            console.warn('Native tools viewer: previous engine destroy failed', error);
        }

        const existingToolGroup = ToolGroupManager.getToolGroup(toolGroupId);
        if (existingToolGroup) {
            ToolGroupManager.destroyToolGroup(toolGroupId);
        }

        // Rely on viewport.setUseCPURendering(true) instead of global namespace

        const renderingEngine = new cornerstone.RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;
        renderingEngine.enableElement({
            viewportId,
            element,
            type: cornerstone.Enums.ViewportType.STACK,
        });

        const viewport = renderingEngine.getViewport(viewportId) as cornerstone.StackViewport;

        const toolGroup = ToolGroupManager.createToolGroup(toolGroupId);
        if (!toolGroup) {
            setState('error');
            setStatus('Failed to create Cornerstone tool group.');
            return;
        }

        toolGroup.addTool(PanTool.toolName);
        toolGroup.addTool(ZoomTool.toolName);
        toolGroup.addTool(WindowLevelTool.toolName);
        toolGroup.addTool(LengthTool.toolName);
        toolGroup.addTool(ArrowAnnotateTool.toolName);
        toolGroup.addTool(RectangleROITool.toolName);
        toolGroup.addTool(EllipticalROITool.toolName);
        toolGroup.addViewport(viewportId, renderingEngineId);

        let isCancelled = false;
        const sourceLabel = imageLabel || file?.name || imageUrl || 'image';

        setState('loading');
        setStatus(`Loading ${sourceLabel} with Cornerstone native tools...`);
        console.log('Native tools viewer: Setting stack for', imageId);

        viewport.setStack([imageId])
            .then(() => {
                if (isCancelled) {
                    return;
                }

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

                setPrimaryTool(toolGroupId, activeTool);
                renderingEngine.resize(true, false);
                viewport.resetCamera();
                viewport.render();

                setState('ready');
                setStatus(`Rendered ${sourceLabel} with native Cornerstone tools.`);
            })
            .catch((error) => {
                if (isCancelled) {
                    return;
                }
                console.error('Native tools viewer: setStack failed', error);
                setState('error');
                setStatus(error instanceof Error ? error.message : 'Native tools viewer setStack failed.');
            });

        const handleResize = () => {
            renderingEngine.resize(true, false);
            viewport.render();
        };

        window.addEventListener('resize', handleResize);

        return () => {
            isCancelled = true;
            window.removeEventListener('resize', handleResize);
            try {
                toolGroup.removeViewports(renderingEngineId, viewportId);
            } catch (error) {
                console.warn('Native tools viewer: removeViewports failed', error);
            }
            try {
                ToolGroupManager.destroyToolGroup(toolGroupId);
            } catch (error) {
                console.warn('Native tools viewer: destroyToolGroup failed', error);
            }
            try {
                renderingEngine.disableElement(viewportId);
                renderingEngine.destroy();
                renderingEngineRef.current = null;
            } catch (error) {
                console.warn('Native tools viewer cleanup failed', error);
            }
        };
    }, [file, imageUrl, imageLabel, inspection, scheme, state]);

    useEffect(() => {
        if (state !== 'ready') {
            return;
        }

        setPrimaryTool(toolGroupIdRef.current, activeTool);
    }, [activeTool, state]);

    const resetView = () => {
        const renderingEngine = renderingEngineRef.current;
        const viewport = renderingEngine?.getViewport(viewportIdRef.current) as cornerstone.StackViewport | undefined;
        if (!viewport) {
            return;
        }

        viewport.resetCamera();
        viewport.resetProperties();
        viewport.render();
    };

    return (
        <div className="border border-slate-300 bg-white">
            <div className="border-b border-slate-300 bg-slate-100 px-3 py-2 text-sm font-medium text-slate-800">
                {title}
            </div>
            <div className="border-b border-slate-300 bg-slate-50 px-2 py-2">
                <div className="flex flex-wrap gap-2">
                    {TOOL_BUTTONS.map((tool) => (
                        <button
                            key={tool.id}
                            type="button"
                            onClick={() => setActiveTool(tool.id)}
                            className={`border px-3 py-1 text-sm ${activeTool === tool.id
                                ? 'border-slate-700 bg-slate-700 text-white'
                                : 'border-slate-300 bg-white text-slate-700'
                                }`}
                        >
                            {tool.label}
                        </button>
                    ))}
                    <button
                        type="button"
                        onClick={resetView}
                        className="border border-slate-300 bg-white px-3 py-1 text-sm text-slate-700"
                    >
                        Reset
                    </button>
                </div>
                <div className="mt-2 text-xs text-slate-500">
                    Source: {scheme === 'dicomfile' ? 'dicomfile:' : 'web:'} | Left: selected tool, Right: Zoom, Middle: Window/Level
                </div>
            </div>
            <div
                ref={viewerRef}
                className="relative bg-black"
                style={{ height: `${maxHeight}px`, touchAction: 'none' }}
            >
                <div className="absolute left-2 top-2 z-10 bg-black/60 px-2 py-1 text-xs text-white">
                    <div>State: {state}</div>
                    <div>{status}</div>
                </div>
                {state === 'error' && (
                    <div className="absolute inset-x-2 bottom-2 z-10 border border-red-400 bg-black/80 px-3 py-2 text-sm text-red-200">
                        {status}
                    </div>
                )}
            </div>
        </div>
    );
}

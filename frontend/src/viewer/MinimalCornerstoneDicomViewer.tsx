import React, { useEffect, useRef, useState } from 'react';
import * as cornerstone from '@cornerstonejs/core';
import { initCornerstone, registerNativeDicomFile } from './cornerstone/init';
import type { LocalDicomInspection } from './cornerstone/dicomDebug';

type MinimalCornerstoneDicomViewerProps = {
    file: File;
    inspection?: LocalDicomInspection | null;
    title?: string;
    maxHeight?: number;
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

export function MinimalCornerstoneDicomViewer({
    file,
    inspection,
    title = 'Minimal Cornerstone DICOM Viewer',
    maxHeight = 420,
}: MinimalCornerstoneDicomViewerProps) {
    const viewerRef = useRef<HTMLDivElement>(null);
    const renderingEngineRef = useRef<cornerstone.RenderingEngine | null>(null);
    const renderingEngineIdRef = useRef(`minimal-cs-engine-${Math.random().toString(36).slice(2)}`);
    const viewportIdRef = useRef(`minimal-cs-viewport-${Math.random().toString(36).slice(2)}`);
    const imageIdRef = useRef<string>('');
    const [state, setState] = useState<'initializing' | 'loading' | 'ready' | 'error'>('initializing');
    const [status, setStatus] = useState('Initializing Cornerstone...');

    useEffect(() => {
        imageIdRef.current = '';
    }, [file]);

    useEffect(() => {
        let isCancelled = false;

        initCornerstone()
            .then(() => {
                if (isCancelled) {
                    return;
                }
                setState('loading');
                setStatus('Cornerstone initialized. Preparing minimal DICOM viewport...');
            })
            .catch((error) => {
                if (isCancelled) {
                    return;
                }
                console.error('Minimal Cornerstone init failed', error);
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
        const imageId = imageIdRef.current || registerNativeDicomFile(file);
        imageIdRef.current = imageId;

        try {
            renderingEngineRef.current?.destroy();
        } catch (error) {
            console.warn('Minimal viewer: previous engine destroy failed', error);
        }

        // Use per-viewport CPU rendering flag instead of global

        const renderingEngine = new cornerstone.RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;
        renderingEngine.enableElement({
            viewportId,
            element,
            type: cornerstone.Enums.ViewportType.STACK,
        });

        const viewport = renderingEngine.getViewport(viewportId) as cornerstone.StackViewport;

        let isCancelled = false;

        setState('loading');
        setStatus(`Loading ${file.name} with native dicomfile loader...`);
        console.log('Minimal Cornerstone: Setting stack for', imageId);

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

                renderingEngine.resize(true, false);
                viewport.resetCamera();
                viewport.render();

                setState('ready');
                setStatus(`Rendered ${file.name} automatically on GPU`);
            })
            .catch((error) => {
                if (isCancelled) {
                    return;
                }
                console.error('Minimal Cornerstone: setStack failed', error);
                setState('error');
                setStatus(error instanceof Error ? error.message : 'Minimal viewer setStack failed.');
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
                renderingEngine.disableElement(viewportId);
                renderingEngine.destroy();
                renderingEngineRef.current = null;
            } catch (error) {
                console.warn('Minimal viewer cleanup failed', error);
            }
        };
    }, [file, inspection, state]);

    return (
        <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 px-4 py-3">
                <div className="text-sm font-semibold text-slate-800">{title}</div>
                <div className="mt-1 text-xs text-slate-500">Native `dicomfile:` + `STACK viewport` only. No custom tool wrapper.</div>
            </div>
            <div
                ref={viewerRef}
                className="relative bg-black"
                style={{ height: `${maxHeight}px`, touchAction: 'none' }}
            >
                <div className="absolute left-3 top-3 rounded-lg bg-black/70 px-3 py-2 text-xs text-slate-100">
                    <div>State: {state}</div>
                    <div>{status}</div>
                    <div>ImageId: {imageIdRef.current || 'pending'}</div>
                </div>
                {state === 'error' && (
                    <div className="absolute inset-x-3 bottom-3 rounded-lg border border-red-400/40 bg-black/80 px-3 py-2 text-sm text-red-200">
                        {status}
                    </div>
                )}
            </div>
        </div>
    );
}

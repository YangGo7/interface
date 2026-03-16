import React, { useEffect, useRef, useState } from 'react';
import * as cornerstone from '@cornerstonejs/core';
import cornerstoneDICOMImageLoader from '@cornerstonejs/dicom-image-loader';
import { initCornerstone, registerNativeDicomFile } from './cornerstone/init';

type CornerstoneCanvasProbeProps = {
    file: File;
    title?: string;
    maxHeight?: number;
};

type ProbeState = {
    status: 'initializing' | 'loading' | 'ready' | 'error';
    message: string;
    imageId: string;
};

export function CornerstoneCanvasProbe({
    file,
    title = 'Cornerstone Native Canvas Probe',
    maxHeight = 420,
}: CornerstoneCanvasProbeProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imageIdRef = useRef('');
    const [probeState, setProbeState] = useState<ProbeState>({
        status: 'initializing',
        message: 'Initializing Cornerstone...',
        imageId: '',
    });

    useEffect(() => {
        imageIdRef.current = '';
    }, [file]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) {
            return;
        }

        let isCancelled = false;
        const imageId = imageIdRef.current || registerNativeDicomFile(file);
        imageIdRef.current = imageId;

        setProbeState({
            status: 'loading',
            message: `Loading ${file.name} via loadImageToCanvas...`,
            imageId,
        });

        initCornerstone()
            .then(async () => {
                if (isCancelled) {
                    return;
                }

                const imageLoadObject = cornerstoneDICOMImageLoader.wadouri.loadImage(imageId, {
                    useRGBA: true,
                });
                const image = await imageLoadObject.promise;

                console.log('Canvas probe: native image stats', {
                    imageId,
                    rows: image.rows,
                    columns: image.columns,
                    minPixelValue: image.minPixelValue,
                    maxPixelValue: image.maxPixelValue,
                    windowCenter: image.windowCenter,
                    windowWidth: image.windowWidth,
                    color: image.color,
                    rgba: image.rgba,
                    sizeInBytes: image.sizeInBytes,
                });

                canvas.width = image.columns;
                canvas.height = image.rows;

                if (typeof image.getCanvas === 'function') {
                    const sourceCanvas = image.getCanvas();
                    const context = canvas.getContext('2d');

                    if (!context) {
                        throw new Error('Canvas probe could not get 2D context.');
                    }

                    context.clearRect(0, 0, canvas.width, canvas.height);
                    context.drawImage(sourceCanvas, 0, 0, canvas.width, canvas.height);
                } else {
                    await cornerstone.utilities.renderToCanvasCPU(canvas, image, 'OT');
                }

                if (isCancelled) {
                    return;
                }

                setProbeState({
                    status: 'ready',
                    message: `Rendered ${file.name} directly to canvas through Cornerstone.`,
                    imageId,
                });
            })
            .catch((error) => {
                if (isCancelled) {
                    return;
                }
                console.error('Canvas probe failed', error);
                setProbeState({
                    status: 'error',
                    message: error instanceof Error ? error.message : 'Canvas probe failed.',
                    imageId,
                });
            });

        return () => {
            isCancelled = true;
        };
    }, [file]);

    return (
        <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 px-4 py-3">
                <div className="text-sm font-semibold text-slate-800">{title}</div>
                <div className="mt-1 text-xs text-slate-500">Cornerstone loader to plain canvas. No RenderingEngine or StackViewport.</div>
            </div>
            <div className="relative bg-black" style={{ minHeight: `${maxHeight}px` }}>
                <div className="absolute left-3 top-3 z-10 rounded-lg bg-black/70 px-3 py-2 text-xs text-slate-100">
                    <div>State: {probeState.status}</div>
                    <div>{probeState.message}</div>
                    <div>ImageId: {probeState.imageId || 'pending'}</div>
                </div>
                <div className="flex h-full min-h-[inherit] items-center justify-center overflow-auto p-4">
                    <canvas
                        ref={canvasRef}
                        className="max-h-full max-w-full object-contain"
                        style={{ maxHeight: `${maxHeight - 32}px` }}
                    />
                </div>
                {probeState.status === 'error' && (
                    <div className="absolute inset-x-3 bottom-3 rounded-lg border border-red-400/40 bg-black/80 px-3 py-2 text-sm text-red-200">
                        {probeState.message}
                    </div>
                )}
            </div>
        </div>
    );
}

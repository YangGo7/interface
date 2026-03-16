import React, { useState, useEffect, useCallback } from 'react';
import * as cornerstone from '@cornerstonejs/core';
import { inspectLocalDicomFile, type LocalDicomInspection } from '../viewer/cornerstone/dicomDebug';
import { initCornerstone, registerNativeDicomFile } from '../viewer/cornerstone/init';
import { registerMultiframeDicomFile, generateMultiframeImageIds } from '../viewer/cornerstone/multiframeLoader';
import { CornerstoneGridViewer } from '../viewer/CornerstoneGridViewer';
import type { ViewerSource } from '../viewer/CornerstoneViewer';

/**
 * Isolated MPR Grid Viewer test page.
 * Shows detailed DICOM metadata debug info alongside the grid viewer
 * to diagnose rendering issues.
 */
export function MprTestPage() {
    const [file, setFile] = useState<File | null>(null);
    const [inspection, setInspection] = useState<LocalDicomInspection | null>(null);
    const [csMetadata, setCsMetadata] = useState<Record<string, any> | null>(null);
    const [cachedImageInfo, setCachedImageInfo] = useState<Record<string, any> | null>(null);
    const [volumeInfo, setVolumeInfo] = useState<Record<string, any> | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0]);
            setCsMetadata(null);
            setCachedImageInfo(null);
            setVolumeInfo(null);
            setError(null);
        }
    };

    // 1) Run our own dicom-parser inspection
    useEffect(() => {
        if (!file) {
            setInspection(null);
            return;
        }
        inspectLocalDicomFile(file)
            .then(setInspection)
            .catch((err) => {
                console.error('DICOM inspection failed', err);
                setInspection(null);
            });
    }, [file]);

    // 2) After file is selected, pre-load with CS3D and extract all metadata
    const probeMetadata = useCallback(async () => {
        if (!file) return;
        setIsLoading(true);
        setError(null);

        try {
            await initCornerstone();

            // Register and load the image through Cornerstone
            const imageId = registerNativeDicomFile(file);
            console.log('[MPR Test] Registered imageId:', imageId);

            const image = await cornerstone.imageLoader.loadAndCacheImage(imageId);
            console.log('[MPR Test] Loaded image object:', image);

            // Extract all the metadata that CS3D volume pipeline reads
            const imagePixelModule = cornerstone.metaData.get('imagePixelModule', imageId);
            const imagePlaneModule = cornerstone.metaData.get('imagePlaneModule', imageId);
            const voiLutModule = cornerstone.metaData.get('voiLutModule', imageId);
            const modalityLutModule = cornerstone.metaData.get('modalityLutModule', imageId);
            const generalSeriesModule = cornerstone.metaData.get('generalSeriesModule', imageId);
            const transferSyntax = cornerstone.metaData.get('transferSyntax', imageId);

            setCsMetadata({
                imagePixelModule,
                imagePlaneModule,
                voiLutModule,
                modalityLutModule,
                generalSeriesModule,
                transferSyntax,
            });

            // Extract cached image info
            const cachedImage = cornerstone.cache.getImage(imageId);
            let cachedImageInfoObj: Record<string, any> = {};
            if (cachedImage) {
                cachedImageInfoObj = {
                    imageId: cachedImage.imageId,
                    rows: cachedImage.rows,
                    columns: cachedImage.columns,
                    width: cachedImage.width,
                    height: cachedImage.height,
                    color: cachedImage.color,
                    rgba: cachedImage.rgba,
                    minPixelValue: cachedImage.minPixelValue,
                    maxPixelValue: cachedImage.maxPixelValue,
                    slope: cachedImage.slope,
                    intercept: cachedImage.intercept,
                    windowCenter: cachedImage.windowCenter,
                    windowWidth: cachedImage.windowWidth,
                    invert: cachedImage.invert,
                    sizeInBytes: cachedImage.sizeInBytes,
                    numberOfComponents: (cachedImage as any).numberOfComponents,
                    photometricInterpretation: (cachedImage as any).photometricInterpretation,
                    dataType: (cachedImage as any).dataType,
                    pixelDataConstructor: cachedImage.getPixelData?.()?.constructor?.name,
                };
                setCachedImageInfo(cachedImageInfoObj);
            }

            // Detect multi-frame using our custom multiframe loader
            const reg = await registerMultiframeDicomFile(file);
            const numberOfFrames = reg.numberOfFrames;

            setCachedImageInfo({
                ...cachedImageInfoObj,
                '🔢 detectedFrames': numberOfFrames,
                '📐 calculatedVolumeSlices': numberOfFrames,
            });

            // Generate all frame imageIds using custom multiframe scheme
            let allImageIds: string[];
            if (numberOfFrames > 1) {
                allImageIds = generateMultiframeImageIds(reg);
                console.log(`[MPR Test] Multi-frame: ${numberOfFrames} frames via custom loader`);
            } else {
                allImageIds = [imageId];
            }

            // Try to create a test volume
            try {
                const volumeId = `cornerstoneStreamingImageVolume:mpr-test-vol-${Date.now()}`;
                const volume = await cornerstone.volumeLoader.createAndCacheVolume(volumeId, {
                    imageIds: allImageIds,
                });

                setVolumeInfo({
                    volumeId: volume.volumeId,
                    dimensions: volume.dimensions ? [...volume.dimensions] : null,
                    spacing: volume.spacing ? [...volume.spacing] : null,
                    origin: volume.origin ? [...volume.origin] : null,
                    direction: volume.direction ? [...volume.direction] : null,
                    dataType: volume.dataType,
                    detectedFrames: numberOfFrames,
                    imageIdsCount: volume.imageIds?.length,
                    metadata: volume.metadata,
                });

                // Clean up test volume
                cornerstone.cache.removeVolumeLoadObject(volumeId);
            } catch (volErr) {
                console.error('[MPR Test] Volume creation failed:', volErr);
                setVolumeInfo({ error: volErr instanceof Error ? volErr.message : String(volErr) });
            }
        } catch (err) {
            console.error('[MPR Test] Probe failed:', err);
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setIsLoading(false);
        }
    }, [file]);

    useEffect(() => {
        if (file) {
            probeMetadata();
        }
    }, [file, probeMetadata]);

    // Build source array for GridViewer
    const gridSources: ViewerSource[] = file
        ? [{
            id: 'mpr-test-dicom',
            label: file.name,
            url: '',
            scheme: 'dicomfile' as const,
            file,
        }]
        : [];

    const isDicom = !!file && (file.name.toLowerCase().endsWith('.dcm') || file.name.toLowerCase().endsWith('.dicom'));

    return (
        <div className="min-h-screen bg-gray-950 text-gray-100">
            {/* Top Bar */}
            <div className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-md sticky top-0 z-50">
                <div className="max-w-[1800px] mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                            MPR Grid Viewer — Debug
                        </h1>
                        <span className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded">isolated test page</span>
                    </div>
                    <div>
                        <label className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium cursor-pointer transition-colors">
                            Upload DICOM
                            <input
                                type="file"
                                accept=".dcm,.dicom"
                                onChange={onFileChange}
                                className="hidden"
                            />
                        </label>
                    </div>
                </div>
            </div>

            <div className="max-w-[1800px] mx-auto px-6 py-6">
                {!file && (
                    <div className="flex items-center justify-center h-[60vh]">
                        <div className="text-center space-y-4">
                            <div className="text-6xl">🩻</div>
                            <h2 className="text-2xl font-semibold text-gray-300">Upload a DICOM file to begin</h2>
                            <p className="text-sm text-gray-500 max-w-md">
                                This page will isolate the MPR Grid Viewer and show detailed DICOM metadata
                                to diagnose rendering problems.
                            </p>
                        </div>
                    </div>
                )}

                {file && (
                    <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-6">
                        {/* LEFT: Grid Viewer */}
                        <div className="space-y-4">
                            <CornerstoneGridViewer
                                sources={gridSources}
                                title={`MPR: ${file.name}`}
                                maxHeight={700}
                            />
                        </div>

                        {/* RIGHT: Debug Panel */}
                        <div className="space-y-4 overflow-y-auto max-h-[calc(100vh-120px)]">
                            {/* File Info */}
                            <DebugCard title="📄 File Info">
                                <DebugRow label="Name" value={file.name} />
                                <DebugRow label="Size" value={`${(file.size / 1024).toFixed(1)} KB`} />
                                <DebugRow label="Type" value={file.type || 'unknown'} />
                                <DebugRow label="Is DICOM" value={isDicom ? '✅ Yes' : '❌ No'} />
                            </DebugCard>

                            {/* Our Own Parser Inspection */}
                            <DebugCard title="🔬 dicom-parser Inspection">
                                {inspection ? (
                                    <>
                                        <DebugRow label="Transfer Syntax" value={inspection.transferSyntaxUid} />
                                        <DebugRow label="Rows × Cols" value={`${inspection.rows} × ${inspection.columns}`} />
                                        <DebugRow label="Bits Allocated" value={inspection.bitsAllocated} />
                                        <DebugRow label="Bits Stored" value={inspection.bitsStored} />
                                        <DebugRow label="Pixel Representation" value={inspection.pixelRepresentation} />
                                        <DebugRow label="Samples/Pixel" value={inspection.samplesPerPixel} />
                                        <DebugRow label="Photometric" value={inspection.photometricInterpretation} />
                                        <DebugRow label="🔢 Number of Frames" value={
                                            inspection.numberOfFrames > 1
                                                ? `✅ ${inspection.numberOfFrames} (multi-frame volume!)`
                                                : `1 (single frame)`
                                        } />
                                        <DebugRow label="Window Center" value={inspection.windowCenter ?? 'N/A'} />
                                        <DebugRow label="Window Width" value={inspection.windowWidth ?? 'N/A'} />
                                        <DebugRow label="Pixel Bytes" value={
                                            inspection.numberOfFrames > 1
                                                ? `${inspection.pixelBytes} total (${inspection.expectedBytes}/frame × ${inspection.numberOfFrames} frames)`
                                                : `${inspection.pixelBytes} (expected: ${inspection.expectedBytes})`
                                        } />
                                        {inspection.issues.length > 0 && (
                                            <div className="mt-2 text-xs text-amber-400 space-y-1">
                                                <div className="font-semibold text-amber-300">⚠️ Issues:</div>
                                                {inspection.issues.map((issue, i) => (
                                                    <div key={i}>• {issue}</div>
                                                ))}
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <div className="text-gray-500 text-xs">Loading...</div>
                                )}
                            </DebugCard>

                            {/* CS3D Metadata */}
                            <DebugCard title="🏗️ CS3D Metadata Provider">
                                {isLoading && <div className="text-gray-500 text-xs">Loading via CS3D...</div>}
                                {error && <div className="text-red-400 text-xs">❌ {error}</div>}
                                {csMetadata && (
                                    <>
                                        <DebugSection label="imagePixelModule" data={csMetadata.imagePixelModule} />
                                        <DebugSection label="imagePlaneModule" data={csMetadata.imagePlaneModule} />
                                        <DebugSection label="voiLutModule" data={csMetadata.voiLutModule} />
                                        <DebugSection label="modalityLutModule" data={csMetadata.modalityLutModule} />
                                        <DebugSection label="generalSeriesModule" data={csMetadata.generalSeriesModule} />
                                        <DebugSection label="transferSyntax" data={csMetadata.transferSyntax} />
                                    </>
                                )}
                            </DebugCard>

                            {/* Cached Image */}
                            <DebugCard title="🖼️ Cached Image Object">
                                {cachedImageInfo ? (
                                    Object.entries(cachedImageInfo).map(([key, val]) => (
                                        <DebugRow key={key} label={key} value={val} />
                                    ))
                                ) : (
                                    <div className="text-gray-500 text-xs">{isLoading ? 'Loading...' : 'Not available'}</div>
                                )}
                            </DebugCard>

                            {/* Volume Info */}
                            <DebugCard title="📦 Volume Creation Result">
                                {volumeInfo ? (
                                    volumeInfo.error ? (
                                        <div className="text-red-400 text-xs">❌ {volumeInfo.error}</div>
                                    ) : (
                                        Object.entries(volumeInfo).map(([key, val]) => (
                                            <DebugRow key={key} label={key} value={
                                                Array.isArray(val)
                                                    ? `[${val.map(v => typeof v === 'number' ? v.toFixed(3) : v).join(', ')}]`
                                                    : typeof val === 'object' && val !== null
                                                        ? JSON.stringify(val, null, 1)
                                                        : val
                                            } />
                                        ))
                                    )
                                ) : (
                                    <div className="text-gray-500 text-xs">{isLoading ? 'Loading...' : 'Not available'}</div>
                                )}
                            </DebugCard>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

/* ----- Helper Components ----- */

function DebugCard({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 overflow-hidden">
            <div className="px-4 py-2.5 bg-gray-800/50 border-b border-gray-800">
                <h3 className="text-sm font-semibold text-gray-200">{title}</h3>
            </div>
            <div className="px-4 py-3 space-y-1.5 text-xs font-mono">
                {children}
            </div>
        </div>
    );
}

function DebugRow({ label, value }: { label: string; value: any }) {
    const displayValue = value === null || value === undefined
        ? '—'
        : typeof value === 'boolean'
            ? value ? '✅ true' : '❌ false'
            : String(value);

    const isWarning = displayValue === '—' || displayValue.includes('undefined') || displayValue.includes('null');

    return (
        <div className="flex items-start gap-2">
            <span className="text-gray-500 shrink-0 w-40 text-right">{label}:</span>
            <span className={`break-all ${isWarning ? 'text-amber-400' : 'text-cyan-300'}`}>
                {displayValue}
            </span>
        </div>
    );
}

function DebugSection({ label, data }: { label: string; data: any }) {
    if (!data) {
        return (
            <div className="mb-2">
                <div className="text-gray-500 text-[10px] uppercase tracking-wider mb-1">{label}</div>
                <div className="text-amber-400">⚠️ undefined (not provided by metadata provider)</div>
            </div>
        );
    }
    return (
        <div className="mb-3">
            <div className="text-gray-500 text-[10px] uppercase tracking-wider mb-1">{label}</div>
            {typeof data === 'object' ? (
                Object.entries(data).map(([key, val]) => (
                    <DebugRow key={key} label={key} value={
                        Array.isArray(val)
                            ? `[${(val as any[]).map(v => typeof v === 'number' ? v.toFixed(3) : v).join(', ')}]`
                            : val
                    } />
                ))
            ) : (
                <DebugRow label="value" value={data} />
            )}
        </div>
    );
}

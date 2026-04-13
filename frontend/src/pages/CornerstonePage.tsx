import React, { useEffect, useMemo, useState } from 'react';
import { CornerstoneViewer, MinimalCornerstoneDicomViewer, CornerstoneCanvasProbe, CornerstoneNativeToolsViewer, ImageViewer, CornerstoneGridViewer } from '../viewer';
import { inspectLocalDicomFile, type LocalDicomInspection } from '../viewer/cornerstone/dicomDebug';
import { buildDicomFolderStudies, type FolderStudy } from '../features/upload/dicomFolderStudies';
import { buildDicomPreviewDataUrl, generateDentalArchCprPano, type ArchPoint } from '../features/cpr/cprPano';

function buildTemporaryCorticalArch(columns: number, rows: number): ArchPoint[] {
    const baseY = rows * 0.6; // 40% up from the bottom
    const midLift = rows * 0.08;
    return [
        { x: columns * 0.16, y: baseY + rows * 0.03 },
        { x: columns * 0.32, y: baseY - rows * 0.02 },
        { x: columns * 0.5, y: baseY - midLift },
        { x: columns * 0.68, y: baseY - rows * 0.02 },
        { x: columns * 0.84, y: baseY + rows * 0.03 },
    ];
}

export function CornerstonePage() {
    const [file, setFile] = useState<File | null>(null);
    const [folderStudies, setFolderStudies] = useState<FolderStudy[]>([]);
    const [selectedFolderSeriesId, setSelectedFolderSeriesId] = useState<string | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [dicomInspection, setDicomInspection] = useState<LocalDicomInspection | null>(null);
    const [cprPreview, setCprPreview] = useState<LocalDicomInspection | null>(null);
    const [cprSliceIndex, setCprSliceIndex] = useState(0);
    const [archPoints, setArchPoints] = useState<ArchPoint[]>([]);
    const [cprImageUrl, setCprImageUrl] = useState<string | null>(null);
    const [cprStatus, setCprStatus] = useState<string>('Select at least 3 arch points on the axial preview.');
    const [cprBusy, setCprBusy] = useState(false);

    const selectedFolderSeries = useMemo(
        () => folderStudies.flatMap((study) => study.series).find((series) => series.id === selectedFolderSeriesId) || null,
        [folderStudies, selectedFolderSeriesId]
    );
    const folderMode = Boolean(selectedFolderSeries);
    const cprSliceCount = selectedFolderSeries?.files.length || 0;

    useEffect(() => {
        if (!file || folderMode) {
            setPreviewUrl(null);
            setDicomInspection(null);
            return;
        }

        const isDicom = file.name.toLowerCase().endsWith('.dcm') || file.name.toLowerCase().endsWith('.dicom');
        if (isDicom) {
            setPreviewUrl('');
            inspectLocalDicomFile(file)
                .then((inspection) => {
                    setDicomInspection(inspection);
                    console.log('Local DICOM inspection', inspection);
                })
                .catch((error) => {
                    console.error('DICOM inspection failed', error);
                    setDicomInspection(null);
                });
            return;
        }

        setDicomInspection(null);
        let isCancelled = false;
        const reader = new FileReader();
        reader.onload = () => {
            if (!isCancelled) {
                setPreviewUrl(typeof reader.result === 'string' ? reader.result : null);
            }
        };
        reader.onerror = () => {
            if (!isCancelled) {
                setPreviewUrl(null);
            }
        };
        reader.readAsDataURL(file);

        return () => {
            isCancelled = true;
        };
    }, [file, folderMode]);

    const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setFolderStudies([]);
            setSelectedFolderSeriesId(null);
            setFile(e.target.files[0]);
        }
    };

    const onFolderChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        if (!files.length) return;
        setFile(null);
        setPreviewUrl(null);
        setDicomInspection(null);
        const studies = await buildDicomFolderStudies(files);
        setFolderStudies(studies);
        setSelectedFolderSeriesId(studies.flatMap((study) => study.series)[0]?.id || null);
    };

    useEffect(() => {
        setArchPoints([]);
        setCprImageUrl(null);
        if (!selectedFolderSeries || !selectedFolderSeries.files.length) {
            setCprSliceIndex(0);
            return;
        }
        setCprSliceIndex(Math.floor(selectedFolderSeries.files.length / 2));
    }, [selectedFolderSeries]);

    useEffect(() => {
        let cancelled = false;
        if (!selectedFolderSeries || !selectedFolderSeries.files.length || selectedFolderSeries.orientation !== 'Axial') {
            setCprPreview(null);
            setCprStatus(selectedFolderSeries ? 'CPR sandbox currently supports axial CT series.' : 'Select an axial CT series for CPR.');
            return;
        }

        const clampedIndex = Math.max(0, Math.min(cprSliceIndex, selectedFolderSeries.files.length - 1));
        const selectedFile = selectedFolderSeries.files[clampedIndex];
        buildDicomPreviewDataUrl(selectedFile)
            .then((preview) => {
                if (!cancelled) {
                    const defaultArch = buildTemporaryCorticalArch(preview.columns, preview.rows);
                    setCprPreview({
                        fileName: selectedFile.name,
                        transferSyntaxUid: '',
                        rows: preview.rows,
                        columns: preview.columns,
                        samplesPerPixel: 1,
                        photometricInterpretation: 'MONOCHROME2',
                        bitsAllocated: 16,
                        bitsStored: 16,
                        pixelRepresentation: 0,
                        pixelBytes: 0,
                        expectedBytes: 0,
                        numberOfFrames: 1,
                        windowCenter: null,
                        windowWidth: null,
                        canRenderPreview: true,
                        previewDataUrl: preview.dataUrl,
                        issues: [],
                    });
                    setArchPoints(defaultArch);
                    setCprStatus(`Temporary cortical arch applied on axial slice ${clampedIndex + 1}/${selectedFolderSeries.files.length}.`);
                }
            })
            .catch((error) => {
                console.error('CPR preview inspection failed', error);
                if (!cancelled) {
                    setCprPreview(null);
                    setCprStatus('Could not prepare axial preview for CPR.');
                }
            });

        return () => {
            cancelled = true;
        };
    }, [selectedFolderSeries, cprSliceIndex]);

    const handleCprPreviewClick = (event: React.MouseEvent<HTMLImageElement>) => {
        if (!cprPreview?.previewDataUrl || !selectedFolderSeries || cprBusy) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const scaleX = cprPreview.columns / rect.width;
        const scaleY = cprPreview.rows / rect.height;
        const x = (event.clientX - rect.left) * scaleX;
        const y = (event.clientY - rect.top) * scaleY;
        setArchPoints((prev) => [...prev, { x, y }]);
    };

    const handleGenerateCpr = async () => {
        if (!selectedFolderSeries || archPoints.length < 3) {
            setCprStatus('Arch points are not ready yet.');
            return;
        }
        setCprBusy(true);
        setCprStatus('Generating curved pano from axial stack...');
        try {
            const result = await generateDentalArchCprPano({
                files: selectedFolderSeries.files,
                archPoints,
            });
            setCprImageUrl(result.dataUrl);
            setCprStatus(`Generated pano ${result.width} x ${result.height} from ${result.sliceCount} slices.`);
        } catch (error: any) {
            console.error('CPR generation failed', error);
            setCprStatus(error?.message || 'CPR generation failed.');
        } finally {
            setCprBusy(false);
        }
    };

    const isDicomFile = !!file && (file.name.toLowerCase().endsWith('.dcm') || file.name.toLowerCase().endsWith('.dicom'));
    const originalViewerUrl = isDicomFile ? dicomInspection?.previewDataUrl || '' : previewUrl || '';
    const viewerSources = folderMode
        ? selectedFolderSeries
            ? [{
                id: `folder-series-${selectedFolderSeries.id}`,
                label: selectedFolderSeries.label,
                url: '',
                files: selectedFolderSeries.files,
                scheme: 'dicomfolder' as const,
            }]
            : []
        : file
            ? [{
                id: 'local-file',
                label: file?.name || 'Local File',
                url: previewUrl || '',
                file: file || undefined,
                scheme: isDicomFile ? 'dicomfile' as const : 'web' as const,
            }]
            : [];

    return (
        <div className="h-screen overflow-y-auto bg-gray-100 flex flex-col p-8">
            <div className="mx-auto w-full max-w-6xl flex flex-col gap-6">
                <div className="bg-white p-6 rounded-2xl shadow-sm">
                    <h1 className="text-2xl font-bold text-gray-800 mb-2">Cornerstone Sandbox</h1>
                    <p className="text-sm text-gray-500 mb-4">
                        Upload a local image (PNG/JPG) or DICOM (.dcm) to test the Cornerstone3D Viewer.
                    </p>

                    <input
                        type="file"
                        id="cornerstoneFileInput"
                        accept="image/*,.dcm,.dicom,application/dicom"
                        className="hidden"
                        onChange={onFileChange}
                    />
                    <input
                        type="file"
                        id="cornerstoneFolderInput"
                        className="hidden"
                        multiple
                        ref={(node) => {
                            if (!node) return;
                            node.setAttribute('webkitdirectory', '');
                            node.setAttribute('directory', '');
                        }}
                        onChange={onFolderChange}
                    />
                    <div className="flex flex-wrap gap-3">
                    <label
                        htmlFor="cornerstoneFileInput"
                        className="inline-block px-6 py-3 rounded-full border border-gray-300 bg-gray-50 text-gray-700 font-medium cursor-pointer hover:bg-gray-100 transition-colors"
                    >
                        {folderMode ? 'Replace Folder Selection' : file ? file.name : 'Select Image or DICOM'}
                    </label>
                    <label
                        htmlFor="cornerstoneFolderInput"
                        className="inline-block px-6 py-3 rounded-full border border-cyan-300 bg-cyan-50 text-cyan-700 font-medium cursor-pointer hover:bg-cyan-100 transition-colors"
                    >
                        Select DICOM Folder
                    </label>
                    </div>
                    {folderMode && (
                        <p className="mt-3 text-sm text-cyan-700">
                            Folder mode active. Study count: {folderStudies.length} / selected series files: {selectedFolderSeries?.files.length || 0}
                        </p>
                    )}
                </div>

                {folderMode && (
                    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
                        <div className="bg-white p-6 rounded-2xl shadow-sm">
                            <h2 className="text-lg font-semibold text-gray-800">Studies / Series</h2>
                            <div className="mt-4 space-y-3">
                                {folderStudies.map((study) => (
                                    <section key={study.id} className="rounded-2xl border border-gray-200 bg-gray-50 p-3">
                                        <div className="mb-2">
                                            <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-700">{study.label || 'Study'}</p>
                                            <p className="mt-1 text-xs text-gray-500">{study.description}</p>
                                        </div>
                                        <div className="space-y-2">
                                            {study.series.map((series) => {
                                                const isActive = selectedFolderSeriesId === series.id;
                                                return (
                                                    <button
                                                        key={series.id}
                                                        type="button"
                                                        onClick={() => setSelectedFolderSeriesId(series.id)}
                                                        className={`w-full rounded-xl border px-3 py-2 text-left transition ${isActive ? 'border-cyan-400 bg-cyan-50 text-cyan-900' : 'border-gray-200 bg-white text-gray-700 hover:border-cyan-200 hover:bg-cyan-50/50'}`}
                                                    >
                                                        <div className="font-semibold">{series.label || 'Unnamed Series'}</div>
                                                        <div className="mt-1 text-xs text-gray-500">{series.modality} / {series.files.length} files</div>
                                                    </button>
                                                );
                                            })}
                                        </div>
                                    </section>
                                ))}
                            </div>
                        </div>
                        <div className="bg-white p-6 rounded-2xl shadow-sm flex flex-col gap-4">
                            <h2 className="text-lg font-semibold text-gray-800">Folder Series Viewer</h2>
                            <p className="text-sm text-gray-500">
                                Use this sandbox to verify local study/series grouping and volume loading for folder-based DICOM input.
                            </p>
                            <CornerstoneGridViewer
                                title={selectedFolderSeries?.label || 'Folder Series'}
                                sources={viewerSources}
                                maxHeight={720}
                            />
                        </div>
                    </div>
                )}

                {folderMode && (
                    <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
                        <div className="bg-white p-6 rounded-2xl shadow-sm flex flex-col gap-4">
                            <div>
                                <h2 className="text-lg font-semibold text-gray-800">CPR Pano Sandbox</h2>
                                <p className="mt-1 text-sm text-gray-500">
                                    Temporary MVP uses a cortical guide curve at 40% from the lower border of the axial preview.
                                </p>
                            </div>
                            <div className="flex flex-wrap gap-2">
                                <button
                                    type="button"
                                    onClick={() => setArchPoints([])}
                                    className="rounded-full border border-gray-300 px-4 py-2 text-xs font-medium text-gray-700 hover:bg-gray-50"
                                >
                                    Clear Points
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setArchPoints((prev) => prev.slice(0, -1))}
                                    disabled={!archPoints.length || cprBusy}
                                    className="rounded-full border border-gray-300 px-4 py-2 text-xs font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
                                >
                                    Undo
                                </button>
                                <button
                                    type="button"
                                    onClick={handleGenerateCpr}
                                    disabled={archPoints.length < 3 || cprBusy || !selectedFolderSeries || selectedFolderSeries.orientation !== 'Axial'}
                                    className="rounded-full bg-cyan-600 px-4 py-2 text-xs font-semibold text-white hover:bg-cyan-500 disabled:opacity-50"
                                >
                                    {cprBusy ? 'Generating...' : 'Generate Pano'}
                                </button>
                            </div>
                            <div className="text-xs text-gray-500">
                                {cprStatus} {archPoints.length > 0 ? `(${archPoints.length} points)` : ''}
                            </div>
                            <div className="rounded-2xl border border-gray-200 bg-gray-50 p-3">
                                {cprPreview?.previewDataUrl ? (
                                    <div className="relative">
                                        <img
                                            src={cprPreview.previewDataUrl}
                                            alt="Axial preview for CPR"
                                            className="block w-full rounded-xl bg-black"
                                        />
                                        <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${cprPreview.columns} ${cprPreview.rows}`} preserveAspectRatio="none">
                                            {archPoints.length > 1 && (
                                                <polyline
                                                    points={archPoints.map((point) => `${point.x},${point.y}`).join(' ')}
                                                    fill="none"
                                                    stroke="#22d3ee"
                                                    strokeWidth={2}
                                                    strokeLinejoin="round"
                                                    strokeLinecap="round"
                                                />
                                            )}
                                            {archPoints.map((point, index) => (
                                                <g key={`${point.x}-${point.y}-${index}`}>
                                                    <circle cx={point.x} cy={point.y} r={4} fill="#06b6d4" stroke="white" strokeWidth={1.5} />
                                                    <text x={point.x + 6} y={point.y - 6} fontSize="10" fill="white">
                                                        {index + 1}
                                                    </text>
                                                </g>
                                            ))}
                                        </svg>
                                    </div>
                                ) : (
                                    <div className="flex h-56 items-center justify-center rounded-xl bg-gray-100 text-sm text-gray-500">
                                        Axial preview unavailable for this series.
                                    </div>
                                )}
                                {cprSliceCount > 1 && (
                                    <div className="mt-3 space-y-2">
                                        <div className="flex items-center justify-between text-[11px] text-gray-500">
                                            <span>Axial Slice</span>
                                            <span>{cprSliceIndex + 1} / {cprSliceCount}</span>
                                        </div>
                                        <input
                                            type="range"
                                            min={0}
                                            max={Math.max(0, cprSliceCount - 1)}
                                            step={1}
                                            value={Math.min(cprSliceIndex, Math.max(0, cprSliceCount - 1))}
                                            onChange={(event) => setCprSliceIndex(Number(event.target.value))}
                                            className="w-full accent-cyan-500"
                                        />
                                    </div>
                                )}
                            </div>
                        </div>
                        <div className="bg-white p-6 rounded-2xl shadow-sm flex flex-col gap-4">
                            <div>
                                <h2 className="text-lg font-semibold text-gray-800">Generated Curved Pano</h2>
                                <p className="mt-1 text-sm text-gray-500">
                                    MVP output using manual arch points and slice-wise sampling across the CT volume.
                                </p>
                            </div>
                            <div className="rounded-2xl border border-gray-200 bg-gray-50 p-4">
                                {cprImageUrl ? (
                                    <img
                                        src={cprImageUrl}
                                        alt="Curved panoramic reconstruction"
                                        className="max-h-[520px] w-full rounded-xl bg-black object-contain"
                                    />
                                ) : (
                                    <div className="flex h-[320px] items-center justify-center rounded-xl bg-gray-100 text-sm text-gray-500">
                                        Generate the pano after selecting the arch points.
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {!folderMode && file && (previewUrl !== null || isDicomFile) && (
                    <div className="bg-white p-6 rounded-2xl shadow-sm flex flex-col gap-4">
                        <h2 className="text-lg font-semibold text-gray-800">Viewer Comparison</h2>
                        <p className="text-sm text-gray-500">
                            Compare the current custom wrapper against a minimal Cornerstone-only DICOM viewport.
                        </p>
                        <CornerstoneViewer
                            title="Custom Cornerstone Wrapper"
                            sources={viewerSources}
                            initialSourceId="local-file"
                            maxHeight={600}
                        />
                        <CornerstoneGridViewer
                            title="MPR Grid Viewer (Volume)"
                            sources={viewerSources}
                            maxHeight={600}
                        />
                        {isDicomFile && file && (
                            <MinimalCornerstoneDicomViewer
                                file={file}
                                inspection={dicomInspection}
                                maxHeight={520}
                            />
                        )}
                        <CornerstoneNativeToolsViewer
                            file={isDicomFile ? file || undefined : undefined}
                            imageUrl={!isDicomFile ? previewUrl || undefined : undefined}
                            imageLabel={file?.name || undefined}
                            scheme={isDicomFile ? 'dicomfile' : 'web'}
                            inspection={dicomInspection}
                            maxHeight={520}
                        />
                        {isDicomFile && file && (
                            <CornerstoneCanvasProbe
                                file={file}
                                maxHeight={520}
                            />
                        )}
                        {originalViewerUrl && (
                            <ImageViewer
                                title="Original Viewer"
                                sources={[
                                    {
                                        id: 'original-viewer',
                                        label: isDicomFile ? `${file.name} (Raw Preview)` : file?.name || 'Original',
                                        url: originalViewerUrl,
                                    }
                                ]}
                                initialSourceId="original-viewer"
                                maxHeight={520}
                            />
                        )}
                    </div>
                )}

                {!folderMode && file && (
                    <div className="bg-white p-6 rounded-2xl shadow-sm flex flex-col gap-4">
                        <h2 className="text-lg font-semibold text-gray-800">Diagnostic Preview</h2>
                        <div className="grid gap-6 lg:grid-cols-2">
                            <div className="rounded-2xl border border-gray-200 bg-gray-50 p-4">
                                <h3 className="mb-3 text-sm font-semibold text-gray-700">File Info</h3>
                                <div className="space-y-2 text-sm text-gray-600">
                                    <div>Name: {file.name}</div>
                                    <div>Size: {file.size.toLocaleString()} bytes</div>
                                    <div>Type: {file.type || 'unknown'}</div>
                                    <div>Kind: {isDicomFile ? 'DICOM' : 'Image'}</div>
                                </div>
                            </div>

                            {dicomInspection ? (
                                <div className="rounded-2xl border border-gray-200 bg-gray-50 p-4">
                                    <h3 className="mb-3 text-sm font-semibold text-gray-700">DICOM Parse Result</h3>
                                    <div className="space-y-2 text-sm text-gray-600">
                                        <div>Transfer Syntax: {dicomInspection.transferSyntaxUid}</div>
                                        <div>Size: {dicomInspection.columns} x {dicomInspection.rows}</div>
                                        <div>Bits: {dicomInspection.bitsAllocated} / stored {dicomInspection.bitsStored}</div>
                                        <div>Samples Per Pixel: {dicomInspection.samplesPerPixel}</div>
                                        <div>Photometric: {dicomInspection.photometricInterpretation}</div>
                                        <div>Pixel Bytes: {dicomInspection.pixelBytes} / expected {dicomInspection.expectedBytes}</div>
                                        <div>Preview Ready: {dicomInspection.canRenderPreview ? 'yes' : 'no'}</div>
                                    </div>
                                    {dicomInspection.issues.length > 0 && (
                                        <div className="mt-4 rounded-xl border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800">
                                            {dicomInspection.issues.join(' | ')}
                                        </div>
                                    )}
                                </div>
                            ) : previewUrl ? (
                                <div className="rounded-2xl border border-gray-200 bg-gray-50 p-4">
                                    <h3 className="mb-3 text-sm font-semibold text-gray-700">Native Image Preview</h3>
                                    <img src={previewUrl} alt="Native preview" className="max-h-[320px] w-full object-contain rounded-xl bg-black" />
                                </div>
                            ) : null}
                        </div>

                        {dicomInspection?.previewDataUrl && (
                            <div className="rounded-2xl border border-gray-200 bg-gray-50 p-4">
                                <h3 className="mb-3 text-sm font-semibold text-gray-700">Raw DICOM Preview</h3>
                                <img src={dicomInspection.previewDataUrl} alt="Raw DICOM preview" className="max-h-[420px] w-full object-contain rounded-xl bg-black" />
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

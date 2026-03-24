import React, { useEffect, useMemo, useState } from 'react';
import { CornerstoneViewer, MinimalCornerstoneDicomViewer, CornerstoneCanvasProbe, CornerstoneNativeToolsViewer, ImageViewer, CornerstoneGridViewer } from '../viewer';
import { inspectLocalDicomFile, type LocalDicomInspection } from '../viewer/cornerstone/dicomDebug';
import { buildDicomFolderStudies, type FolderStudy } from '../features/upload/dicomFolderStudies';

export function CornerstonePage() {
    const [file, setFile] = useState<File | null>(null);
    const [folderStudies, setFolderStudies] = useState<FolderStudy[]>([]);
    const [selectedFolderSeriesId, setSelectedFolderSeriesId] = useState<string | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [dicomInspection, setDicomInspection] = useState<LocalDicomInspection | null>(null);

    const selectedFolderSeries = useMemo(
        () => folderStudies.flatMap((study) => study.series).find((series) => series.id === selectedFolderSeriesId) || null,
        [folderStudies, selectedFolderSeriesId]
    );
    const folderMode = Boolean(selectedFolderSeries);

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

    const isDicomFile = !!file && (file.name.toLowerCase().endsWith('.dcm') || file.name.toLowerCase().endsWith('.dicom'));
    const originalViewerUrl = isDicomFile ? dicomInspection?.previewDataUrl || '' : previewUrl || '';
    const viewerSources = folderMode
        ? [{
            id: `folder-series-${selectedFolderSeries.id}`,
            label: selectedFolderSeries.label,
            url: '',
            files: selectedFolderSeries.files,
            scheme: 'dicomfolder' as const,
        }]
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
        <div className="min-h-screen bg-gray-100 flex flex-col p-8">
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

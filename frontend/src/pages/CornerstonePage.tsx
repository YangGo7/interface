import React, { useState, useEffect } from 'react';
import { CornerstoneViewer, MinimalCornerstoneDicomViewer, CornerstoneCanvasProbe, CornerstoneNativeToolsViewer, ImageViewer, CornerstoneGridViewer } from '../viewer';
import { inspectLocalDicomFile, type LocalDicomInspection } from '../viewer/cornerstone/dicomDebug';

export function CornerstonePage() {
    const [file, setFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [dicomInspection, setDicomInspection] = useState<LocalDicomInspection | null>(null);

    useEffect(() => {
        if (!file) {
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
    }, [file]);

    const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0]);
        }
    };

    const isDicomFile = !!file && (file.name.toLowerCase().endsWith('.dcm') || file.name.toLowerCase().endsWith('.dicom'));
    const originalViewerUrl = isDicomFile ? dicomInspection?.previewDataUrl || '' : previewUrl || '';

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
                    <label
                        htmlFor="cornerstoneFileInput"
                        className="inline-block px-6 py-3 rounded-full border border-gray-300 bg-gray-50 text-gray-700 font-medium cursor-pointer hover:bg-gray-100 transition-colors"
                    >
                        {file ? file.name : 'Select Image or DICOM'}
                    </label>
                </div>

                {file && (previewUrl !== null || isDicomFile) && (
                    <div className="bg-white p-6 rounded-2xl shadow-sm flex flex-col gap-4">
                        <h2 className="text-lg font-semibold text-gray-800">Viewer Comparison</h2>
                        <p className="text-sm text-gray-500">
                            Compare the current custom wrapper against a minimal Cornerstone-only DICOM viewport.
                        </p>
                        <CornerstoneViewer
                            title="Custom Cornerstone Wrapper"
                            sources={[
                                {
                                    id: 'local-file',
                                    label: file?.name || 'Local File',
                                    url: previewUrl || '',
                                    file: file || undefined,
                                    scheme: isDicomFile
                                        ? 'dicomfile'
                                        : 'web'
                                }
                            ]}
                            initialSourceId="local-file"
                            maxHeight={600}
                        />
                        <CornerstoneGridViewer
                            title="MPR Grid Viewer (Volume)"
                            sources={[
                                {
                                    id: 'local-file',
                                    label: file?.name || 'Local File',
                                    url: previewUrl || '',
                                    file: file || undefined,
                                    scheme: isDicomFile
                                        ? 'dicomfile'
                                        : 'web'
                                }
                            ]}
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

                {file && (
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

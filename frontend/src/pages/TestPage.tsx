import React, { useEffect, useState } from 'react';
import { ImageViewer, CornerstoneViewer, CornerstoneNativeToolsViewer } from '../viewer';

const generateSessionId = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
};

export function TestPage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>('');
  const [statusType, setStatusType] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [reportHtml, setReportHtml] = useState<string | null>(null);
  const [reportPdfUrl, setReportPdfUrl] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>('');

  useEffect(() => {
    setSessionId(generateSessionId());
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleUpload = async () => {
    if (!file || !sessionId) {
      return;
    }

    setStatusType('loading');
    setStatus('Step 1: uploading file and starting analysis...');
    setResultUrl(null);
    setReportHtml(null);
    setReportPdfUrl(null);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`/api/v2/upload_process/${sessionId}`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        setStatus('Step 2: AI analysis is running...');
        pollStatus();
      } else {
        const errorText = await res.text();
        setStatusType('error');
        setStatus(errorText || 'Upload failed. Please try again.');
      }
    } catch (error) {
      console.error(error);
      setStatusType('error');
      setStatus('Network error while uploading.');
    }
  };

  const pollStatus = () => {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/v2/session/${sessionId}`);
        const data = await res.json();

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          setStatusType('success');
          setStatus('Analysis completed.');
          if (data.result?.overlay_url) {
            setResultUrl(data.result.overlay_url);
            setReportHtml(data.result.report_html || null);
            setReportPdfUrl(data.result.report_url || null);
          } else {
            setStatus((prev) => `${prev} Overlay URL was not returned.`);
          }
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          setStatusType('error');
          setStatus(`Analysis failed: ${data.error || 'Unknown error'}`);
        }
      } catch (error) {
        console.error('Poll error', error);
      }
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-100 px-4 py-10">
      <div className="mx-auto w-full max-w-5xl rounded-3xl bg-white p-8 shadow-xl">
        <h2 className="mb-2 text-2xl font-bold text-gray-900">Viewer Sandbox</h2>
        <p className="mb-6 text-sm text-gray-500">
          Upload an image, wait for the overlay result, then test the reusable viewer module here first.
        </p>

        <div className="mx-auto mb-6 flex max-w-md flex-col gap-4">
          <input
            type="file"
            id="fileInput"
            accept="image/*,.dcm,.dicom,application/dicom"
            className="hidden"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />

          <label
            htmlFor="fileInput"
            className="inline-block w-full rounded-full border border-gray-300 bg-gray-100 px-6 py-3 text-center font-medium text-gray-700 transition-colors hover:bg-gray-200"
          >
            {file ? file.name : 'Select image'}
          </label>

          <button
            onClick={handleUpload}
            disabled={!file || statusType === 'loading'}
            className={`w-full rounded-full px-6 py-3 font-medium text-white transition-all ${!file || statusType === 'loading'
              ? 'cursor-not-allowed bg-gray-400'
              : 'bg-blue-600 shadow-md hover:bg-blue-700'
              }`}
          >
            {statusType === 'loading' ? 'Analyzing...' : 'Start analysis'}
          </button>

          {status && (
            <div
              className={`rounded-lg p-4 text-sm font-medium ${statusType === 'loading'
                ? 'bg-blue-50 text-blue-700'
                : statusType === 'success'
                  ? 'bg-green-50 text-green-700'
                  : statusType === 'error'
                    ? 'bg-red-50 text-red-700'
                    : 'bg-gray-50 text-gray-700'
                }`}
            >
              {status}
            </div>
          )}
        </div>

        {previewUrl && file && (
          <div className="mb-6 space-y-4">
            <div className="border-t border-dashed border-gray-300 pt-6">
              <h3 className="mb-2 text-lg font-semibold text-gray-800">Cornerstone Native Tools Viewer</h3>
              <p className="mb-4 text-sm text-gray-500">
                Original source rendered through the native Cornerstone tools path for comparison against the reusable viewer wrapper.
              </p>
              <CornerstoneNativeToolsViewer
                file={(file.name.toLowerCase().endsWith('.dcm') || file.name.toLowerCase().endsWith('.dicom')) ? file : undefined}
                imageUrl={!(file.name.toLowerCase().endsWith('.dcm') || file.name.toLowerCase().endsWith('.dicom')) ? previewUrl : undefined}
                imageLabel={file.name}
                scheme={(file.name.toLowerCase().endsWith('.dcm') || file.name.toLowerCase().endsWith('.dicom')) ? 'dicomfile' : 'web'}
                maxHeight={520}
              />
            </div>
          </div>
        )}

        {resultUrl && (
          <div className="space-y-6">
            {((file?.name.toLowerCase().endsWith('.dcm') || file?.name.toLowerCase().endsWith('.dicom')) ||
              resultUrl.toLowerCase().endsWith('.dcm') ||
              resultUrl.toLowerCase().endsWith('.dicom')) ? (
              <CornerstoneViewer
                title="Cornerstone Prototype"
                sources={[
                  ...(previewUrl
                    ? [
                      {
                        id: 'original',
                        label: 'Original',
                        url: previewUrl,
                        file: file ?? undefined,
                        scheme: (file?.name.toLowerCase().endsWith('.dcm') || file?.name.toLowerCase().endsWith('.dicom')
                          ? 'dicomfile'
                          : 'web') as 'dicomfile' | 'web',
                      },
                    ]
                    : []),
                  {
                    id: 'overlay',
                    label: 'Overlay',
                    url: resultUrl,
                    scheme: (resultUrl.toLowerCase().endsWith('.dcm') || resultUrl.toLowerCase().endsWith('.dicom')
                      ? 'wadouri'
                      : 'web') as 'wadouri' | 'web',
                  },
                ]}
                initialSourceId={previewUrl ? 'original' : 'overlay'}
                maxHeight={560}
              />
            ) : (
              <ImageViewer
                title="2D Prototype"
                sources={[
                  ...(previewUrl
                    ? [
                      {
                        id: 'original',
                        label: 'Original',
                        url: previewUrl,
                      },
                    ]
                    : []),
                  {
                    id: 'overlay',
                    label: 'Overlay',
                    url: resultUrl,
                  },
                ]}
                initialSourceId={previewUrl ? 'original' : 'overlay'}
                maxHeight={560}
              />
            )}

            {reportHtml && (
              <div className="border-t-2 border-dashed border-gray-300 pt-6">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="text-xl font-bold text-gray-800">Report Preview</h3>
                  {reportPdfUrl && (
                    <a
                      href={reportPdfUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-lg bg-green-600 px-4 py-2 text-sm text-white transition-colors hover:bg-green-700"
                    >
                      Open report
                    </a>
                  )}
                </div>

                <div
                  className="max-h-[800px] overflow-auto rounded-lg border border-gray-200 bg-white p-6 shadow-sm"
                  dangerouslySetInnerHTML={{ __html: reportHtml }}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

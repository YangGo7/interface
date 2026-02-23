import React, { useState, useEffect } from 'react';

// Simple session ID generator (avoids 'uuid' dependency issues without restart)
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
  const [reportHtml, setReportHtml] = useState<string | null>(null);
  const [reportPdfUrl, setReportPdfUrl] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>('');

  useEffect(() => {
    // Generate session ID on mount
    setSessionId(generateSessionId());
  }, []);

  const handleUpload = async () => {
    if (!file || !sessionId) return;

    setStatusType('loading');
    setStatus('1단계: 파일 전송 및 분석 시작...');
    setResultUrl(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`/api/v2/upload_process/${sessionId}`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        setStatus('2단계: AI 분석 진행 중 (약 10~20초)...');
        pollStatus();
      } else {
        setStatusType('error');
        setStatus('❌ 업로드 실패. 다시 시도해주세요.');
      }
    } catch (e) {
      console.error(e);
      setStatusType('error');
      setStatus('❌ 네트워크 오류.');
    }
  };

  const pollStatus = () => {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/v2/session/${sessionId}`);
        const data = await res.json();
        console.log("Polling...", data);

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          setStatusType('success');
          setStatus('✅ 분석 완료!');
          if (data.result && data.result.overlay_url) {
            setResultUrl(data.result.overlay_url);
            setReportHtml(data.result.report_html || null);
            setReportPdfUrl(data.result.report_url || null); // Use report_url as link (HTML view)
          } else {
            setStatus(prev => prev + ' (이미지 URL 없음)');
          }
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          setStatusType('error');
          setStatus('❌ 분석 실패: ' + (data.error || "Unknown error"));
        }
      } catch (e) {
        console.error("Poll Error", e);
      }
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <div className="bg-white p-8 rounded-2xl shadow-xl w-full max-w-md text-center transition-all" style={{ maxWidth: resultUrl ? '800px' : '400px' }}>
        <h2 className="text-2xl font-bold mb-2 text-gray-900">🔭 실시간 분석 테스트 (React)</h2>
        <p className="text-gray-500 mb-6">이미지를 업로드하면 결과 오버레이를<br />이 페이지에서 즉시 확인합니다.</p>

        <input
          type="file"
          id="fileInput"
          accept="image/*"
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />

        <label
          htmlFor="fileInput"
          className="inline-block w-full py-3 px-6 rounded-full bg-gray-100 text-gray-700 font-medium cursor-pointer hover:bg-gray-200 transition-colors mb-4 border border-gray-300"
        >
          {file ? file.name : '사진 선택하기'}
        </label>

        <button
          onClick={handleUpload}
          disabled={!file || statusType === 'loading'}
          className={`w-full py-3 px-6 rounded-full text-white font-medium transition-all ${!file || statusType === 'loading'
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700 shadow-md'
            }`}
        >
          {statusType === 'loading' ? '분석 중...' : '분석 시작'}
        </button>

        {status && (
          <div className={`mt-4 p-4 rounded-lg font-medium text-sm ${statusType === 'loading' ? 'bg-blue-50 text-blue-700' :
            statusType === 'success' ? 'bg-green-50 text-green-700' :
              statusType === 'error' ? 'bg-red-50 text-red-700' : 'bg-gray-50'
            }`}>
            {status}
          </div>
        )}

        {resultUrl && (
          <div className="mt-6 animate-fade-in">
            <h3 className="text-lg font-bold mb-2 text-gray-800">분석 결과 (Overlay)</h3>
            <div className="border border-gray-200 rounded-lg overflow-hidden mb-6">
              <img src={resultUrl} alt="Overlay Result" className="w-full h-auto object-contain" />
            </div>

            {/* Report Injection */}
            {reportHtml && (
              <div className="mt-8 border-t-2 border-dashed border-gray-300 pt-6 text-left">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-bold text-gray-800">📄 상세 분석 리포트</h3>
                  {reportPdfUrl && (
                    <a href={reportPdfUrl} target="_blank" rel="noreferrer" className="bg-green-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-green-700 transition-colors">
                      새 창에서 열기
                    </a>
                  )}
                </div>

                <div
                  className="border border-gray-200 rounded-lg p-6 bg-white shadow-sm overflow-auto max-h-[800px]"
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

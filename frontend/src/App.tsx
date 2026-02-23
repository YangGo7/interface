import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './index.css';
import logo from './assets/O3_logo_only.png';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleDetect = async () => {
    if (!file) return alert('이미지나 DICOM을 선택해 주세요!');
    setLoading(true);
    try {
      const form = new FormData();
      form.append('image', file);

      // Start async job
      const res = await fetch(`/api/detect_async`, { method: 'POST', body: form });
      const data = await res.json();

      if (!res.ok || !data.success || !data.job_id) {
        alert(`추론 요청 실패: ${data.message || 'unknown'}`);
        return;
      }

      // Navigate to chart page immediately with jobId
      // We also pass a local preview URL so the user can see their image immediately
      const previewUrl = URL.createObjectURL(file);
      navigate('/chart', {
        state: {
          jobId: data.job_id,
          previewUrl: previewUrl
        }
      });
    } catch (e: any) {
      console.error(e);
      alert('에러 발생: ' + e?.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0a] text-gray-100">
      <main className="flex-1 flex flex-col items-center justify-center gap-8 px-4">
        <div className="flex items-center gap-3">
          <div style={{ width: '24px', height: '24px' }}>
            <img src={logo} alt="E2DW logo" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
          </div>
          <div className="flex flex-col leading-tight">
            <div className="font-extrabold text-2xl">E2DW</div>
            <div className="text-sm text-gray-400">AI panoramic insights</div>
          </div>
        </div>

        <div className="w-1/2 max-w-md flex flex-col items-center gap-3">
          <input
            type="file"
            id="imageInput"
            accept="image/*,.dcm"
            hidden
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <label
            htmlFor="imageInput"
            className="w-full py-4 px-6 rounded-full border border-gray-800 bg-[#0f0f0f] text-center cursor-pointer hover:border-blue-500 hover:text-blue-400 transition-colors shadow-lg"
          >
            {file ? file.name : '이미지 / DICOM input'}
          </label>
          <button
            className="w-full py-4 px-4 rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 text-white font-bold hover:opacity-90 transition-opacity disabled:opacity-50 shadow-lg max-w-[200px]"
            onClick={handleDetect}
            disabled={loading || !file}
          >
            {loading ? 'Starting...' : 'Start'}
          </button>

          <div className="w-full max-w-[200px] border-t border-gray-800 my-2"></div>

          <input
            type="text"
            placeholder="Patient Name (for Report)"
            className="w-full p-3 rounded-lg bg-[#222] border border-gray-700 text-white text-center"
            id="patientNameInput"
          />

          <button
            className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-green-600 to-green-700 text-white font-bold hover:opacity-90 transition-opacity disabled:opacity-50 shadow-lg max-w-[200px]"
            onClick={async () => {
              if (!file) return alert('Select an image first!');
              const nameInput = document.getElementById('patientNameInput') as HTMLInputElement;
              const userName = nameInput?.value || "Patient";

              setLoading(true);
              try {
                const form = new FormData();
                form.append('image', file);
                form.append('user_name', userName);

                const res = await fetch('/api/v2/analyze', { method: 'POST', body: form });
                const data = await res.json();

                if (data.report_url) {
                  // Navigate to internal Report Page with Data
                  navigate('/report', {
                    state: {
                      reportHtml: data.report_html,
                      reportUrl: data.report_url, // [NEW] Link to generated HTML file
                      analysisResult: data.analysis_result,
                      aiCommentary: data.ai_commentary,
                      overlayUrl: data.overlay_url,
                      userName: userName
                    }
                  });
                } else {
                  alert("Report generation failed: " + (data.error || "Unknown error"));
                }
              } catch (e: any) {
                alert("Error: " + e.message);
              } finally {
                setLoading(false);
              }
            }}
            disabled={loading || !file}
          >
            {loading ? 'Generating...' : 'Generate Patient Report'}
          </button>
        </div>
      </main>

      <footer className="w-full border-t border-gray-800 bg-[#0f0f0f] py-4 px-6 text-center text-sm text-gray-500">
        안내: 모델 결과는 의료머시기에 결정이 아닙니다. (참고용)
      </footer>
    </div>
  );
}

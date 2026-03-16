import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './index.css';
import logo from './assets/O3_logo_only.png';

const isDicomFile = (inputFile: File) => /\.(dcm|dicom)$/i.test(inputFile.name);

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('English');
  const navigate = useNavigate();

  const handleDetect = async () => {
    if (!file) {
      alert('Select an image or DICOM file first.');
      return;
    }

    setLoading(true);

    try {
      const form = new FormData();
      form.append('image', file);

      const res = await fetch('/api/detect_async', { method: 'POST', body: form });
      const data = await res.json();

      if (!res.ok || !data.success || !data.job_id) {
        alert(`Detection request failed: ${data.message || 'unknown error'}`);
        return;
      }

      const dicom = isDicomFile(file);
      const previewUrl = data.preview_url || (!dicom ? URL.createObjectURL(file) : undefined);

      navigate('/chart', {
        state: {
          jobId: data.job_id,
          previewUrl,
          originalFile: dicom ? file : undefined,
          originalIsDicom: dicom,
          originalFileName: file.name,
        },
      });
    } catch (error: any) {
      console.error(error);
      alert(`Error: ${error?.message || 'unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!file) {
      alert('Select an image first.');
      return;
    }

    const nameInput = document.getElementById('patientNameInput') as HTMLInputElement | null;
    const userName = nameInput?.value || 'Patient';

    setLoading(true);

    try {
      const form = new FormData();
      form.append('image', file);
      form.append('user_name', userName);
      form.append('language', language);

      const res = await fetch('/api/v2/analyze', { method: 'POST', body: form });
      const data = await res.json();

      if (!data.report_url) {
        alert(`Report generation failed: ${data.error || 'Unknown error'}`);
        return;
      }

      navigate('/report', {
        state: {
          reportHtml: data.report_html,
          reportUrl: data.report_url,
          analysisResult: data.analysis_result,
          aiCommentary: data.ai_commentary,
          overlayUrl: data.overlay_url,
          userName,
        },
      });
    } catch (error: any) {
      console.error(error);
      alert(`Error: ${error?.message || 'unknown error'}`);
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
            accept="image/*,.dcm,.dicom,application/dicom"
            hidden
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
          <label
            htmlFor="imageInput"
            className="w-full py-4 px-6 rounded-full border border-gray-800 bg-[#0f0f0f] text-center cursor-pointer hover:border-blue-500 hover:text-blue-400 transition-colors shadow-lg"
          >
            {file ? file.name : 'Image / DICOM input'}
          </label>

          <button
            className="w-full py-4 px-4 rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 text-white font-bold hover:opacity-90 transition-opacity disabled:opacity-50 shadow-lg max-w-[200px]"
            onClick={handleDetect}
            disabled={loading || !file}
          >
            {loading ? 'Starting...' : 'Start'}
          </button>

          <div className="w-full max-w-[200px] border-t border-gray-800 my-2" />

          <input
            type="text"
            placeholder="Patient Name (for Report)"
            className="w-full p-3 rounded-lg bg-[#222] border border-gray-700 text-white text-center"
            id="patientNameInput"
          />

          <select
            className="w-full p-3 rounded-lg bg-[#222] border border-gray-700 text-white text-center mb-2"
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
          >
            <option value="English">English</option>
            <option value="Korean">Korean</option>
            <option value="Spanish">Spanish</option>
            <option value="Japanese">Japanese</option>
            <option value="Vietnamese">Vietnamese</option>
          </select>

          <button
            className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-green-600 to-green-700 text-white font-bold hover:opacity-90 transition-opacity disabled:opacity-50 shadow-lg max-w-[200px]"
            onClick={handleGenerateReport}
            disabled={loading || !file}
          >
            {loading ? 'Generating...' : 'Generate Patient Report'}
          </button>
        </div>
      </main>

      <footer className="w-full border-t border-gray-800 bg-[#0f0f0f] py-4 px-6 text-center text-sm text-gray-500">
        Notice: model output is for reference only and is not a medical decision.
      </footer>
    </div>
  );
}

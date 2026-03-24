import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/O3_logo_only.png';
import { UploadPicker } from '../components/upload/UploadPicker';
import { buildDicomFolderStudies, type FolderStudy } from '../features/upload/dicomFolderStudies';
import { requestAsyncDetection, requestPatientReport } from '../features/upload/uploadApi';
import { buildUploadSelection } from '../features/upload/uploadSelection';

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [folderFiles, setFolderFiles] = useState<File[]>([]);
  const [folderStudies, setFolderStudies] = useState<FolderStudy[]>([]);
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('English');
  const navigate = useNavigate();
  const selection = useMemo(() => buildUploadSelection(file, folderFiles), [file, folderFiles]);

  const handleDetect = async () => {
    if (!selection.primaryFile) return alert('Select an image or DICOM file first.');
    if (selection.folderMode) {
      const firstSeries = folderStudies.flatMap((study) => study.series)[0];
      navigate('/chart', {
        state: {
          originalFolderMode: true,
          originalFolderStudies: folderStudies,
          folderSelectedSeriesId: firstSeries?.id,
          originalIsDicom: true,
          originalFileName: firstSeries?.label || selection.primaryFile.name,
        },
      });
      return;
    }
    setLoading(true);
    try {
      const state = await requestAsyncDetection(selection.primaryFile, folderFiles);
      navigate('/chart', { state });
    } catch (error: any) {
      console.error(error);
      alert(`Error: ${error?.message || 'unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!selection.primaryFile) return alert('Select an image first.');
    const nameInput = document.getElementById('patientNameInput') as HTMLInputElement | null;
    const userName = nameInput?.value || 'Patient';
    setLoading(true);
    try {
      const data = await requestPatientReport(selection.primaryFile, userName, language);
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
          <div style={{ width: 24, height: 24 }}>
            <img src={logo} alt="E2DW logo" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
          </div>
          <div className="flex flex-col leading-tight">
            <div className="font-extrabold text-2xl">E2DW</div>
            <div className="text-sm text-gray-400">AI panoramic insights</div>
          </div>
        </div>

        <div className="w-1/2 max-w-md flex flex-col items-center gap-3">
          <UploadPicker
            inputSummary={selection.inputSummary}
            activeFileCount={selection.activeFiles.length}
            primaryFileName={selection.primaryFile?.name}
            folderMode={selection.folderMode}
            onSingleFileChange={(nextFile) => {
              setFolderStudies([]);
              setFolderFiles([]);
              setFile(nextFile);
            }}
            onFolderFilesChange={async (files) => {
              setFile(null);
              setFolderFiles(files);
              setFolderStudies(await buildDicomFolderStudies(files));
            }}
          />
          <button className="w-full py-4 px-4 rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 text-white font-bold hover:opacity-90 transition-opacity disabled:opacity-50 shadow-lg max-w-[200px]" onClick={handleDetect} disabled={loading || !selection.primaryFile}>
            {loading ? 'Starting...' : 'Start'}
          </button>
          <div className="w-full max-w-[200px] border-t border-gray-800 my-2" />
          <input type="text" placeholder="Patient Name (for Report)" className="w-full p-3 rounded-lg bg-[#222] border border-gray-700 text-white text-center" id="patientNameInput" />
          <select className="w-full p-3 rounded-lg bg-[#222] border border-gray-700 text-white text-center mb-2" value={language} onChange={(event) => setLanguage(event.target.value)}>
            <option value="English">English</option>
            <option value="Korean">Korean</option>
            <option value="Spanish">Spanish</option>
            <option value="Japanese">Japanese</option>
            <option value="Vietnamese">Vietnamese</option>
          </select>
          <button className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-green-600 to-green-700 text-white font-bold hover:opacity-90 transition-opacity disabled:opacity-50 shadow-lg max-w-[200px]" onClick={handleGenerateReport} disabled={loading || !selection.primaryFile}>
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

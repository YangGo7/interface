import { useLocation, useNavigate } from 'react-router-dom';
import logo from '../assets/O3_logo_only.png';

export function ReportPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { reportUrl } = location.state || {};

  if (!reportUrl) {
    return (
      <div className="h-screen flex flex-col items-center justify-center bg-[#0a0a0a] text-white">
        <h2 className="text-xl mb-4">No report URL found.</h2>
        <button
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition"
        >
          Go Home
        </button>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50 overflow-hidden">
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between shadow-sm z-10 flex-shrink-0 h-16">
        <div className="flex items-center cursor-pointer gap-3" onClick={() => navigate('/')}>
          <img src={logo} alt="Logo" className="h-8 w-auto object-contain" />
          <span className="font-bold text-lg text-gray-800 tracking-tight">E2DW Original Report</span>
        </div>

        <div className="flex items-center gap-3">
          <a
            href={reportUrl}
            target="_blank"
            rel="noreferrer"
            className="text-sm px-4 py-2 rounded-lg text-blue-600 font-medium hover:bg-blue-50 transition-colors"
          >
            Open in New Tab
          </a>
          <button
            onClick={() => navigate('/')}
            className="text-sm px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-white font-medium transition-colors"
          >
            Close Viewer
          </button>
        </div>
      </header>

      <div className="flex-1 w-full relative bg-gray-200">
        <iframe
          title="Patient Report"
          src={reportUrl}
          className="w-full h-full border-none block"
          sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
        />
      </div>
    </div>
  );
}

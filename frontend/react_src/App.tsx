import { useEffect, useState } from 'react';
import { ChartPage } from './ChartPage';
import { TopHeader } from './components/TopHeader';

type ModelItem = { name: string; size?: string };

function LandingPage({
  models,
  onDetect,
  loading,
}: {
  models: ModelItem[];
  onDetect: (file: File, model?: string) => void;
  loading: boolean;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<string | undefined>(undefined);

  useEffect(() => {
    if (models.length && !model) setModel(models[0].name);
  }, [models, model]);

  return (
    <div className="h-screen w-full bg-[#06070a] text-gray-100 flex flex-col">
      <TopHeader />
      <div className="flex-1 flex flex-col items-center justify-center gap-8 px-4">
        <div className="flex flex-col items-center gap-2">
          <div className="text-2xl font-bold">system name</div>
          <div className="text-sm text-gray-400">Easy to Dentist Web</div>
        </div>
        <div className="w-full max-w-3xl flex items-center gap-3">
          <input
            id="landing-file"
            type="file"
            accept="image/*,.dcm"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0] || null;
              setFile(f);
            }}
          />
          <label
            htmlFor="landing-file"
            className="flex-1 px-4 py-3 rounded-full border border-gray-700 bg-[#0f1320] text-center cursor-pointer shadow-lg"
          >
            {file ? file.name : '이미지 / DICOM input'}
          </label>
          <select
            className="px-3 py-3 rounded-lg border border-gray-700 bg-[#0f1320] min-w-[180px]"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {models.length === 0 ? <option>Loading...</option> :
              models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}{m.size ? ` (${m.size})` : ''}
                </option>
              ))
            }
          </select>
          <button
            className="px-4 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-blue-700 text-black font-semibold min-w-[110px] shadow-lg disabled:opacity-50"
            onClick={() => file && onDetect(file, model)}
            disabled={loading || !file}
          >
            {loading ? 'Loading...' : 'Start'}
          </button>
        </div>
        <div className="text-sm text-gray-500">
          안내: 모델 결과는 의료머시기에 결정이 아닙니다. (참고용)
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<'landing' | 'chart'>('landing');
  const [models, setModels] = useState<ModelItem[]>([]);
  const [detecting, setDetecting] = useState(false);
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${window.location.origin}/api/models`);
        const data = await res.json();
        const list: ModelItem[] = data.models || [];
        setModels(list);
      } catch (e) {
        console.error(e);
      }
    })();
  }, []);

  const handleDetect = async (file: File, model?: string) => {
    setDetecting(true);
    try {
      const form = new FormData();
      form.append('image', file);
      if (model) form.append('model', model);

      const res = await fetch(`${window.location.origin}/api/detect`, { method: 'POST', body: form });
      const data = await res.json();
      if (!data.success) {
        alert('제출 실패: ' + (data.message || 'unknown'));
        return;
      }
      const statusUrlRaw = data.status_url || data.statusUrl || (data.job_id ? `/api/detect/status/${data.job_id}` : null);
      if (!statusUrlRaw) {
        alert('서버 응답에 status_url이 없습니다.');
        return;
      }
      const statusUrl = statusUrlRaw.startsWith('http') ? statusUrlRaw : `${window.location.origin}${statusUrlRaw}`;
      pollStatus(statusUrl);
    } catch (e: any) {
      alert('에러 발생: ' + e?.message);
    } finally {
      setDetecting(false);
    }
  };

  const pollStatus = (url: string) => {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(url);
        const data = await res.json();
        if (!data.success) return;
        if (data.status === 'done' && data.result) {
          clearInterval(timer);
          setResult(data.result);
          setPage('chart');
        } else if (data.status === 'failed') {
          clearInterval(timer);
          alert('작업 실패: ' + (data.error || 'unknown'));
        }
      } catch (e) {
        console.error(e);
      }
    }, 1200);
  };

  if (page === 'chart') {
    return <ChartPage result={result} />;
  }

  return (
    <LandingPage
      models={models}
      onDetect={handleDetect}
      loading={detecting}
    />
  );
}

import { useState } from 'react';
import { TopHeader } from './components/TopHeader';
import { BottomTeethChart } from './components/BottomTeethChart';
import { AlertTriangle, Activity, Zap } from 'lucide-react';
import { RightPanel } from './components/RightPanel';

const mockCounts = [
  { label: 'Teeth', value: 36 },
  { label: 'Caries', value: 2 },
  { label: 'Periapical', value: 0 },
  { label: 'CEJ masks', value: 5 },
  { label: 'Bone level masks', value: 1 },
];

const mockFindings = [
  { label: 'Caries', value: '35 (0.52)', color: 'bg-rose-500' },
  { label: 'Caries', value: '27 (0.37)', color: 'bg-rose-400' },
];

type ChartPageProps = {
  result?: any;
};

export function ChartPage({ result }: ChartPageProps) {
  const [selectedTooth, setSelectedTooth] = useState<number | undefined>(undefined);

  const det = result?.det_counts || {};
  const counts = result
    ? [
        { label: 'Teeth', value: det.seg_teeth ?? 0 },
        { label: 'Caries', value: det.caries ?? 0 },
        { label: 'Periapical', value: det.periapical ?? 0 },
        { label: 'CEJ masks', value: det.cej_masks ?? 0 },
        { label: 'Bone level masks', value: det.bonelevel_masks ?? 0 },
      ]
    : mockCounts;

  const pblEntries = result?.pbl
    ? Object.entries(result.pbl).map(([tooth, val]) => `${tooth}: ${Number(val).toFixed(1)}%`)
    : ['11: 84.4%', '12: 82.6%', '13: 84.4%'];

  const findings = result
    ? [
        ...(result.caries_by_tooth || []).map((c: any) => ({
          label: 'Caries',
          value: `${c[0]} (${(c[1] ?? 0).toFixed ? c[1].toFixed(2) : c[1]})`,
          color: 'bg-rose-500',
        })),
        ...(result.periapical_by_tooth || []).map((p: any) => ({
          label: 'Periapical',
          value: `${p[0]} (${(p[1] ?? 0).toFixed ? p[1].toFixed(2) : p[1]})`,
          color: 'bg-emerald-500',
        })),
      ]
    : mockFindings;

  // tooth statuses for 3분할 색상
  const statuses: Record<string, any> = {};
  const missing = result?.teeth_missing || [];
  missing.forEach((t: string) => (statuses[t] = { ...(statuses[t] || {}), missing: true }));
  const carBest = result?.caries_by_tooth_best || {};
  Object.keys(carBest).forEach((t) => (statuses[t] = { ...(statuses[t] || {}), caries: true }));
  const periBest = result?.periapical_by_tooth_best || {};
  Object.keys(periBest).forEach((t) => (statuses[t] = { ...(statuses[t] || {}), peri: true }));
  // crown/filling/implant도 있으면 반영
  const markStatus = (field: any, prop: 'implant' | 'crown' | 'filling') => {
    if (!field) return;
    if (Array.isArray(field)) {
      field.forEach((item: any) => {
        const key = Array.isArray(item) ? item[0] : item?.toString();
        if (!key) return;
        statuses[key] = { ...(statuses[key] || {}), [prop]: true };
      });
    } else if (typeof field === 'object') {
      Object.keys(field).forEach((key) => {
        statuses[key] = { ...(statuses[key] || {}), [prop]: true };
      });
    }
  };
  markStatus(result?.implant_by_tooth_best || result?.implant_by_tooth, 'implant');
  markStatus(result?.crown_by_tooth_best || result?.crown_by_tooth, 'crown');
  markStatus(result?.filling_by_tooth_best || result?.filling_by_tooth, 'filling');

  return (
    <div className="h-screen w-full bg-[#0a0a0a] text-gray-100 flex flex-col overflow-hidden">
      <TopHeader />

      <div className="flex flex-1 overflow-hidden">
        <main className="flex-1 overflow-auto px-4 py-3 flex flex-col gap-3">
          <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            <div className="bg-[#0f0f0f] border border-gray-800 rounded-xl p-4 flex flex-col gap-3">
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <Activity className="w-4 h-4 text-cyan-400" />
                <span>Detection Counts</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {counts.map((item) => (
                  <div key={item.label} className="bg-[#111827] border border-gray-800 rounded-lg px-3 py-2 flex justify-between text-sm">
                    <span className="text-gray-400">{item.label}</span>
                    <span className="text-white font-semibold">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-[#0f0f0f] border border-gray-800 rounded-xl p-4 flex flex-col gap-2">
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <Zap className="w-4 h-4 text-amber-400" />
                <span>PBL (%) 샘플</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {pblEntries.map((item, idx) => (
                  <span key={idx} className="px-3 py-2 rounded-full bg-[#111827] border border-gray-800 text-sm text-white">
                    {item}
                  </span>
                ))}
              </div>
            </div>

            <div className="bg-[#0f0f0f] border border-gray-800 rounded-xl p-4 flex flex-col gap-3">
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <AlertTriangle className="w-4 h-4 text-rose-400" />
                <span>Findings</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {findings.map((f, i) => (
                  <span key={i} className={`px-3 py-2 rounded-full text-sm text-white ${f.color}`}>
                    {f.label}: {f.value}
                  </span>
                ))}
              </div>
            </div>
          </section>

          <section className="bg-[#0f0f0f] border border-gray-800 rounded-xl p-4 flex flex-col gap-3">
            <div className="text-sm text-gray-400">Odontogram (click a tooth to select)</div>
            <BottomTeethChart
              onToothClick={setSelectedTooth}
              selectedTooth={selectedTooth}
              statuses={statuses}
            />
          </section>
        </main>

        <RightPanel
          selectedTooth={selectedTooth}
          result={result}
          onClose={() => setSelectedTooth(undefined)}
        />
      </div>
    </div>
  );
}

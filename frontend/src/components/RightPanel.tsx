import { useState, useEffect } from 'react';
import { X } from 'lucide-react';

type BestDet = { conf?: number; box?: number[] };

interface RightPanelProps {
  selectedTooth?: number;
  result?: any;
  statuses?: Record<string, any>;
  onClose?: () => void;
  numberingSystem?: 'fdi' | 'univ';
}

const fmt = (v?: any, digits = 2) => {
  const n = Number(v);
  return !Number.isNaN(n) && v !== null && v !== undefined ? n.toFixed(digits) : '-';
};

const toUniv = (fdi: number): number | string => {
  const map: Record<number, number> = {
    18: 1, 17: 2, 16: 3, 15: 4, 14: 5, 13: 6, 12: 7, 11: 8,
    21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
    38: 17, 37: 18, 36: 19, 35: 20, 34: 21, 33: 22, 32: 23, 31: 24,
    41: 25, 42: 26, 43: 27, 44: 28, 45: 29, 46: 30, 47: 31, 48: 32
  };
  return map[fdi] || fdi;
};

export function RightPanel({ selectedTooth, result, statuses = {}, onClose, numberingSystem = 'fdi' }: RightPanelProps) {
  const [lastTooth, setLastTooth] = useState<number | undefined>(selectedTooth);
  const [impDia, setImpDia] = useState('');
  const [impLen, setImpLen] = useState('');

  useEffect(() => {
    if (selectedTooth) setLastTooth(selectedTooth);
  }, [selectedTooth]);

  // Sync metrics to local state
  useEffect(() => {
    const key = selectedTooth ? String(selectedTooth) : undefined;
    const metrics = key ? result?.implant_metrics?.[key] : undefined;
    if (metrics) {
      const d = metrics.diameter_mm ?? metrics.diameter ?? metrics.diameter_bucket;
      const l = metrics.length_mm ?? metrics.length ?? metrics.length_bucket;
      setImpDia(d ? String(Number(d).toFixed(1)) : '');
      setImpLen(l ? String(Number(l).toFixed(1)) : '');
    } else {
      setImpDia('');
      setImpLen('');
    }
  }, [selectedTooth, result]);


  const displayTooth = selectedTooth ?? lastTooth;
  const key = displayTooth ? String(displayTooth) : undefined;
  const missingList: string[] = Array.isArray(result?.teeth_missing)
    ? result.teeth_missing.map((t: any) => String(t))
    : [];
  const statusMap = statuses || {};
  const pbl = key ? result?.pbl?.[key] : undefined;
  const pblLevel = key ? result?.pbl_level?.[key] : undefined;

  const carBest: BestDet | undefined = key
    ? (result?.caries_by_tooth_best?.[key] || result?.caries_by_tooth?.[key])
    : undefined;

  const periBest: BestDet | undefined = key
    ? (result?.periapical_by_tooth_best?.[key] || result?.periapical_by_tooth?.[key])
    : undefined;

  const implantData = key ? (result?.implant_by_tooth_best?.[key] || result?.implant_by_tooth?.[key]) : undefined;
  const implant = key ? (statusMap[key]?.implant || !!implantData) : false;
  const crown = key ? !!(result?.crown_by_tooth_best?.[key] || result?.crown_by_tooth?.[key]) : false;
  const filling = key ? !!(result?.filling_by_tooth_best?.[key] || result?.filling_by_tooth?.[key]) : false;
  const iMetrics = key ? result?.implant_metrics?.[key] : undefined;
  const missing = key ? (statusMap[key]?.missing || missingList.includes(key)) : false;
  const requires = key ? (!!statusMap[key]?.caries || !!statusMap[key]?.peri) : false;
  const triage = key ? statusMap[key]?.triage : undefined;
  const toothStatus = key ? (statusMap[key] || {}) : {};
  const triageReasons: string[] = [];
  if (missing) triageReasons.push('Missing tooth');
  if (implant) triageReasons.push('Implant / fixture');
  if (toothStatus.hopeless) triageReasons.push('Hopeless prognosis');
  if (toothStatus.nerve_overlap) triageReasons.push('Nerve overlap');
  if (toothStatus.sinus_overlap) triageReasons.push('Sinus overlap');
  if (Number(toothStatus.nerve_dist_mm || 0) > 0 && Number(toothStatus.nerve_dist_mm || 0) < 2) {
    triageReasons.push(`Nerve distance ${fmt(toothStatus.nerve_dist_mm, 1)} mm`);
  }
  if (Number(toothStatus.bone_loss_level || 0) >= 3) {
    triageReasons.push(`Bone loss level ${toothStatus.bone_loss_level}`);
  }
  if (Number(toothStatus.bone_loss_pct || 0) >= 60) {
    triageReasons.push(`Bone loss ${fmt(toothStatus.bone_loss_pct, 1)}%`);
  }
  if (carBest) triageReasons.push(`Caries ${fmt(carBest.conf)}`);
  if (periBest) triageReasons.push(`Periapical ${fmt(periBest.conf)}`);

  // Dynamic Status Logic
  let statusText = 'Present';
  if (missing && !implant) statusText = 'Missing';
  else if (implant) statusText = 'Implant Site';
  else if (carBest?.conf || periBest?.conf || requires) statusText = 'Requires Tx';
  else if (crown) statusText = 'Restored (Crown)';

  // Panel is hidden (0 width) by default, expands when a tooth is selected
  // Added min-w-0 and border-l-0 for smoother close transition
  const widthCls = selectedTooth ? "w-[15%] min-w-[320px] border-l" : "w-0 min-w-0 border-l-0 overflow-hidden";
  const containerCls = `${widthCls} flex-shrink-0 border-gray-200 bg-white overflow-y-auto no-scrollbar hover:custom-scrollbar transition-all duration-300 [direction:ltr]`;

  if (!displayTooth) {
    return <div className={containerCls} />;
  }

  return (
    <div className={containerCls} style={{ direction: 'ltr', scrollbarGutter: 'stable' }}>
      <div className="p-4 text-gray-800">
        <div className="text-2xl font-black mb-6 tracking-tight text-gray-900 flex items-center justify-between border-b border-gray-200 pb-4">
          <div className="flex items-center gap-3">
            <span className="bg-indigo-600 text-[10px] px-2 py-0.5 rounded-md uppercase text-white">Tooth</span>
            #{displayTooth ? (numberingSystem === 'univ' ? toUniv(displayTooth) : displayTooth) : ''}
          </div>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors text-gray-400 hover:text-gray-900 group"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="panel-section mb-6">
          <div className="text-[10px] uppercase font-black tracking-widest text-indigo-600 mb-3">Basics</div>
          <ul className="space-y-2.5">
            <li className="flex justify-between items-center text-sm">
              <span className="text-gray-500 font-medium">Status</span>
              <span className={`font-bold ${statusText.includes('Site') || statusText.includes('Lesion') ? 'text-rose-600' : 'text-emerald-600'}`}>{statusText}</span>
            </li>
            <li className="flex justify-between items-center text-sm">
              <span className="text-gray-500 font-medium">PBL (%)</span>
              <span className="text-gray-900 font-bold">{pbl !== undefined ? `${fmt(pbl, 1)}%` : '-'}</span>
            </li>
            <li className="flex justify-between items-center text-sm">
              <span className="text-gray-500 font-medium">PBL level</span>
              <span className="text-gray-900 font-bold">{pblLevel ?? '-'}</span>
            </li>
            <li className="flex justify-between items-center text-sm">
              <span className="text-gray-500 font-medium">Triage</span>
              <span className="text-gray-900 font-bold">{triage || '-'}</span>
            </li>
          </ul>
          <div className="mt-4 rounded-xl border border-gray-200 bg-gray-50 px-3 py-3">
            <div className="text-[10px] uppercase font-black tracking-widest text-indigo-600 mb-2">Triage Reason</div>
            {triageReasons.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {triageReasons.map((reason) => (
                  <span
                    key={reason}
                    className="rounded-full border border-indigo-200 bg-white px-2.5 py-1 text-[11px] font-semibold text-gray-700"
                  >
                    {reason}
                  </span>
                ))}
              </div>
            ) : (
              <div className="text-xs text-gray-500">No active triage trigger</div>
            )}
          </div>
        </div>

        <div className="panel-section mb-6">
          <div className="text-[10px] uppercase font-black tracking-widest text-indigo-600 mb-3">Lesions</div>
          <ul className="space-y-2.5">
            <li className="flex justify-between items-center text-sm">
              <span className="text-gray-500 font-medium">Caries</span>
              <span className="text-gray-900 font-bold">{carBest ? fmt(carBest.conf) : '-'}</span>
            </li>
            <li className="flex justify-between items-center text-sm">
              <span className="text-gray-500 font-medium">Periapical</span>
              <span className="text-gray-900 font-bold">{periBest ? fmt(periBest.conf) : '-'}</span>
            </li>
          </ul>
        </div>

        <div className="panel-section mb-6">
          <div className="text-[10px] uppercase font-black tracking-widest text-indigo-600 mb-3">Prosthetic / Implant</div>
          <ul className="space-y-2.5">
            <li className="flex justify-between items-center text-sm">
              <span className="text-gray-500 font-medium">Implant</span>
              <span className="text-gray-900 font-bold">{implant ? 'Yes' : 'No'}</span>
            </li>
            {implant && (
              <>
                <li className="flex justify-between items-center text-sm pl-4 border-l border-gray-200">
                  <span className="text-gray-500 font-medium italic">Diameter</span>
                  <div className="flex items-center gap-1">
                    <input
                      type="text"
                      value={impDia}
                      onChange={(e) => setImpDia(e.target.value)}
                      placeholder="-"
                      className="w-16 bg-white border border-gray-300 rounded px-1.5 py-0.5 text-right text-sm font-bold text-blue-600 focus:border-blue-500 focus:outline-none transition-colors"
                    />
                    <span className="text-gray-500 text-xs">mm</span>
                  </div>
                </li>
                <li className="flex justify-between items-center text-sm pl-4 border-l border-gray-200">
                  <span className="text-gray-500 font-medium italic">Length</span>
                  <div className="flex items-center gap-1">
                    <input
                      type="text"
                      value={impLen}
                      onChange={(e) => setImpLen(e.target.value)}
                      placeholder="-"
                      className="w-16 bg-white border border-gray-300 rounded px-1.5 py-0.5 text-right text-sm font-bold text-blue-600 focus:border-blue-500 focus:outline-none transition-colors"
                    />
                    <span className="text-gray-500 text-xs">mm</span>
                  </div>
                </li>
              </>
            )}
            <li className="flex justify-between items-center text-sm">
              <span className="text-gray-500 font-medium">Crown</span>
              <span className="text-gray-900 font-bold">{crown ? 'Yes' : 'No'}</span>
            </li>
            <li className="flex justify-between items-center text-sm">
              <span className="text-gray-500 font-medium">Filling</span>
              <span className="text-gray-900 font-bold">{filling ? 'Yes' : 'No'}</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}

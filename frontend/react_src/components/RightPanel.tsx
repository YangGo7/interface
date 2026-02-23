import { X } from 'lucide-react';

type BestDet = { conf?: number; box?: number[] };

interface RightPanelProps {
  selectedTooth?: number;
  result?: any;
  onClose?: () => void;
}

const fmt = (v?: number, digits = 2) =>
  typeof v === 'number' && !Number.isNaN(v) ? v.toFixed(digits) : '-';

export function RightPanel({ selectedTooth, result, onClose }: RightPanelProps) {
  if (!selectedTooth) {
    return (
      <aside className="w-80 bg-[#0f0f0f] border-l border-gray-800 flex items-center justify-center">
        <div className="text-center text-gray-500 px-6">
          <div className="text-4xl mb-3">· · ·</div>
          <p className="text-sm">치아를 선택하세요</p>
        </div>
      </aside>
    );
  }

  const key = String(selectedTooth);
  const pbl = result?.pbl?.[key];
  const pblLevel = result?.pbl_level?.[key];
  const carBest: BestDet | undefined = result?.caries_by_tooth_best?.[key];
  const periBest: BestDet | undefined = result?.periapical_by_tooth_best?.[key];
  const implant = !!(result?.implant_by_tooth_best?.[key] || result?.implant_by_tooth?.[key]);
  const crown = !!(result?.crown_by_tooth_best?.[key] || result?.crown_by_tooth?.[key]);
  const filling = !!(result?.filling_by_tooth_best?.[key] || result?.filling_by_tooth?.[key]);
  const missing = (result?.teeth_missing || []).includes(key);

  return (
    <aside className="w-80 bg-[#0f0f0f] border-l border-gray-800 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <h2 className="text-lg">Tooth #{selectedTooth}</h2>
        <button className="p-1 hover:bg-[#1a1a1a] rounded" onClick={onClose}>
          <X className="w-5 h-5 text-gray-400" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Basic Info */}
        <div>
          <h3 className="text-sm text-gray-400 mb-2">기본 정보</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">상태</span>
              <span className={missing ? 'text-gray-400' : 'text-green-400'}>
                {missing ? 'Missing' : 'Present'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">PBL (%)</span>
              <span className="text-white">{pbl !== undefined ? fmt(pbl, 1) + '%' : '-'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">PBL level</span>
              <span className="text-white">{pblLevel ?? '-'}</span>
            </div>
          </div>
        </div>

        {/* Lesions */}
        <div>
          <h3 className="text-sm text-gray-400 mb-2">병소</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-rose-300">Caries</span>
              <span className="text-white">{carBest?.conf ? fmt(carBest.conf) : '-'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-emerald-300">Periapical</span>
              <span className="text-white">{periBest?.conf ? fmt(periBest.conf) : '-'}</span>
            </div>
          </div>
        </div>

        {/* Prosthetics */}
        <div>
          <h3 className="text-sm text-gray-400 mb-2">보철/임플란트</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Implant</span>
              <span className={implant ? 'text-purple-300' : 'text-white'}>{implant ? 'Yes' : 'No'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Crown</span>
              <span className={crown ? 'text-amber-300' : 'text-white'}>{crown ? 'Yes' : 'No'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Filling</span>
              <span className={filling ? 'text-blue-300' : 'text-white'}>{filling ? 'Yes' : 'No'}</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}

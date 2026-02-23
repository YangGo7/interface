interface ToothStatus {
  missing?: boolean;
  caries?: boolean;
  peri?: boolean; // periapical/root issue
  implant?: boolean;
  crown?: boolean;
  filling?: boolean;
}

interface BottomTeethChartProps {
  onToothClick: (tooth: number) => void;
  selectedTooth?: number;
  statuses?: Record<string, ToothStatus>;
}

export function BottomTeethChart({ onToothClick, selectedTooth, statuses = {} }: BottomTeethChartProps) {
  const upperTeeth = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28];
  const lowerTeeth = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38];

  // 우선순위: missing > caries > implant > crown > filling > peri > default
  const getPartColor = (tooth: number, part: 'root' | 'middle' | 'crown') => {
    const st = statuses[String(tooth)] || {};
    if (st.missing) return 'bg-gray-700';
    if (st.caries && part === 'crown') return 'bg-rose-500';
    if (st.implant) return 'bg-purple-500';
    if (st.crown && part === 'crown') return 'bg-amber-400';
    if (st.filling && part === 'crown') return 'bg-blue-500';
    if (st.peri && part === 'root') return 'bg-emerald-500';

    const neutral: Record<'root' | 'middle' | 'crown', string> = {
      root: 'bg-gray-700',
      middle: 'bg-gray-600',
      crown: 'bg-gray-500',
    };
    return neutral[part];
  };

  return (
    <div className="bg-[#0f0f0f] border-t border-gray-800 px-4 py-3">
      <div className="max-w-4xl mx-auto">
        {/* Upper Teeth */}
        <div className="flex justify-center gap-1 mb-4">
          {upperTeeth.map((tooth) => (
            <button
              key={tooth}
              onClick={() => onToothClick(tooth)}
              className={`flex flex-col items-center gap-1 transition-transform hover:scale-110 ${
                selectedTooth === tooth ? 'scale-110 opacity-100' : 'opacity-80'
              }`}
            >
              <div className="text-xs text-gray-500">{tooth}</div>
              <div
                className={`w-6 h-12 rounded-t-lg overflow-hidden flex flex-col ${
                  selectedTooth === tooth ? 'ring-2 ring-white' : ''
                }`}
              >
                <div className={`h-1/3 ${getPartColor(tooth, 'root')}`}></div>
                <div className={`h-1/3 ${getPartColor(tooth, 'middle')}`}></div>
                <div className={`h-1/3 ${getPartColor(tooth, 'crown')}`}></div>
              </div>
            </button>
          ))}
        </div>

        {/* Lower Teeth */}
        <div className="flex justify-center gap-1">
          {lowerTeeth.map((tooth) => (
            <button
              key={tooth}
              onClick={() => onToothClick(tooth)}
              className={`flex flex-col items-center gap-1 transition-transform hover:scale-110 ${
                selectedTooth === tooth ? 'scale-110 opacity-100' : 'opacity-80'
              }`}
            >
              <div
                className={`w-6 h-12 rounded-b-lg overflow-hidden flex flex-col ${
                  selectedTooth === tooth ? 'ring-2 ring-white' : ''
                }`}
              >
                <div className={`h-1/3 ${getPartColor(tooth, 'crown')}`}></div>
                <div className={`h-1/3 ${getPartColor(tooth, 'middle')}`}></div>
                <div className={`h-1/3 ${getPartColor(tooth, 'root')}`}></div>
              </div>
              <div className="text-xs text-gray-500">{tooth}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

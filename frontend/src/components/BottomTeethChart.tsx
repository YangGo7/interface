interface ToothStatus {
  missing?: boolean;
  caries?: boolean;
  peri?: boolean;
  implant?: boolean;
  crown?: boolean;
  filling?: boolean;
}

interface BottomTeethChartProps {
  onToothClick: (tooth: number) => void;
  selectedTooth?: number;
  statuses?: Record<string, ToothStatus>;
  highlightRing?: (number | string)[];
  extraction?: (number | string)[];
  implantSites?: (number | string)[];
  numberingSystem?: 'fdi' | 'univ'; // [NEW]
}

export function BottomTeethChart({
  onToothClick,
  selectedTooth,
  statuses = {},
  highlightRing = [],
  extraction = [],
  implantSites = [],
  numberingSystem = 'fdi',
}: BottomTeethChartProps) {
  const upperTeeth = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28];
  const lowerTeeth = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38];

  const getTileStyle = (tooth: number) => {
    const toothKey = String(tooth);
    const st = statuses[toothKey] || {};
    const isExtraction = extraction.map(String).includes(toothKey);
    const isImplantSite = implantSites.map(String).includes(toothKey);
    const ring = highlightRing.map(String).includes(toothKey) || isImplantSite;
    let fill = '#D1FAE5';
    let border = '#34D399';
    if (st.missing) { fill = '#F1F5F9'; border = '#94A3B8'; }
    else if (st.implant) { fill = '#DBEAFE'; border = '#3B82F6'; }
    else if (st.caries || st.peri) { fill = '#FEE2E2'; border = '#EF4444'; }
    if (isExtraction) { fill = '#FFEDD5'; border = '#F97316'; }
    const dashed = !!st.missing;
    return { fill, border, dashed, ring };
  };

  const getOdontoImg = (tooth: number) => {
    const n = tooth % 10; // 1..8
    const upper = tooth >= 11 && tooth <= 28;
    const base = upper ? 'up' : 'down';
    const file = `${base}-${n}.png`;
    const origin = (typeof window !== 'undefined' && window.location?.origin) ? window.location.origin : '';
    return `${origin}/imgs/odonto/${file}`;
  };

  // Universal Mapping
  const toUniv = (fdi: number): string => {
    const map: Record<number, number> = {
      18: 1, 17: 2, 16: 3, 15: 4, 14: 5, 13: 6, 12: 7, 11: 8,
      21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
      38: 17, 37: 18, 36: 19, 35: 20, 34: 21, 33: 22, 32: 23, 31: 24,
      41: 25, 42: 26, 43: 27, 44: 28, 45: 29, 46: 30, 47: 31, 48: 32
    };
    return String(map[fdi] || fdi);
  };

  const renderRow = (teeth: number[], isUpper: boolean) => (
    <div className="flex justify-center gap-3 flex-wrap sm:flex-nowrap">
      {teeth.map((tooth) => {
        const { fill, border, dashed, ring } = getTileStyle(tooth);
        const numColor = dashed ? '#94A3B8' : border;
        const displayNum = numberingSystem === 'univ' ? toUniv(tooth) : tooth;

        return (
          <div key={tooth} className="flex flex-col items-center gap-1.5">

            <button
              onClick={() => onToothClick(tooth)}
              className={`w-14 h-[72px] rounded-full border-2 flex items-center justify-center transition-all ${selectedTooth === tooth ? 'ring-2 ring-indigo-500 ring-offset-2 ring-offset-black/40' : ''
                }`}
              style={{
                background: fill,
                borderColor: border,
                borderStyle: dashed ? 'dashed' : 'solid',
                boxShadow: ring ? '0 0 0 2px #DC2626 inset, 0 1px 2px rgba(0,0,0,0.12)' : '0 1px 2px rgba(0,0,0,0.08)'
              }}
            >
              <img
                src={getOdontoImg(tooth)}
                alt={`Tooth ${tooth}`}
                className="w-12 h-14 object-contain mix-blend-multiply pointer-events-none"
                style={{ filter: 'contrast(1.4) saturate(0.9)' }}
                onError={(e) => {
                  const svg = encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="50"><rect width="40" height="50" rx="10" ry="10" fill="white" stroke="#94A3B8" stroke-width="2"/><line x1="10" y1="10" x2="30" y2="40" stroke="#94A3B8" stroke-width="2"/></svg>');
                  (e.currentTarget as HTMLImageElement).src = `data:image/svg+xml,${svg}`;
                }}
              />
            </button>
            <span className="text-[12px] font-semibold leading-none" style={{ color: numColor }}>
              {displayNum}
            </span>
          </div>
        );
      })}
    </div>
  );

  return (
    <div className="bg-white border-t border-gray-200 px-4 py-6">
      <div className="max-w-4xl mx-auto flex flex-col gap-4">
        {renderRow(upperTeeth, true)}
        <div className="h-px w-full bg-gray-200" />
        {renderRow(lowerTeeth, false)}
      </div>
    </div>
  );
}

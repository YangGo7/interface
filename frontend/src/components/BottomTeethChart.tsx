import { useState } from 'react';
import { createPortal } from 'react-dom';

interface ToothStatus {
  missing?: boolean;
  caries?: boolean;
  peri?: boolean;
  implant?: boolean;
  crown?: boolean;
  filling?: boolean;
  triage?: 'triage-1' | 'triage-2' | 'triage-3' | 'missing' | 'implant';
}

interface ToothTooltipData {
  kind?: 'finding' | 'implant';
  status?: string;
  pblPct?: number | null;
  level?: number | string | null;
  caries?: boolean;
  periapical?: boolean;
  gapMm?: number | null;
  nerveDistMm?: number | null;
}

interface BottomTeethChartProps {
  onToothClick: (tooth: number) => void;
  selectedTooth?: number;
  statuses?: Record<string, ToothStatus>;
  highlightRing?: (number | string)[];
  extraction?: (number | string)[];
  implantSites?: (number | string)[];
  numberingSystem?: 'fdi' | 'univ'; // [NEW]
  tooltipData?: Record<string, ToothTooltipData>;
}

export function BottomTeethChart({
  onToothClick,
  selectedTooth,
  statuses = {},
  highlightRing = [],
  extraction = [],
  implantSites = [],
  numberingSystem = 'fdi',
  tooltipData = {},
}: BottomTeethChartProps) {
  const tileWidth = 52;
  const tileHeight = 117;
  const imageWidth = 41;
  const imageHeight = 88;
  const tileGap = 5;
  const rowGap = 32;
  const [hoveredTooth, setHoveredTooth] = useState<number | null>(null);
  const [tooltipAnchor, setTooltipAnchor] = useState({ x: 0, y: 0 });

  const upperTeeth = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28];
  const lowerTeeth = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38];

  const getTileStyle = (tooth: number) => {
    const toothKey = String(tooth);
    const st = statuses[toothKey] || {};
    const isExtraction = extraction.map(String).includes(toothKey);
    const isImplantSite = implantSites.map(String).includes(toothKey);
    const ring = highlightRing.map(String).includes(toothKey) || isImplantSite;
    const reportStatus = st.triage || (st.missing ? 'missing' : st.implant ? 'implant' : st.caries || st.peri ? 'triage-2' : 'triage-3');
    let fill = 'rgba(34, 197, 94, 0.15)';
    let border = '#22c55e';
    if (reportStatus === 'triage-1') { fill = 'rgba(220, 38, 38, 0.2)'; border = '#dc2626'; }
    else if (reportStatus === 'triage-2') { fill = 'rgba(234, 179, 8, 0.2)'; border = '#eab308'; }
    else if (reportStatus === 'missing') { fill = '#D1D5DB'; border = '#9CA3AF'; }
    else if (reportStatus === 'implant') { fill = 'rgba(37, 99, 235, 0.15)'; border = '#2563EB'; }
    if (isExtraction) { fill = '#FFEDD5'; border = '#F97316'; }
    const dashed = reportStatus === 'missing';
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

  const shouldMirrorTooth = (tooth: number) => {
    const quadrant = Math.floor(tooth / 10);
    return quadrant === 2 || quadrant === 3;
  };

  const formatValue = (value?: number | string | null, suffix = '') => {
    if (value === null || value === undefined || value === '') return '-';
    if (typeof value === 'number') return `${value.toFixed(1)}${suffix}`;
    return `${value}${suffix}`;
  };

  const hoveredTooltip = hoveredTooth ? tooltipData[String(hoveredTooth)] : null;

  const renderRow = (teeth: number[]) => (
    <div
      className="grid items-center"
      style={{
        gridTemplateColumns: `minmax(0, max-content)`,
      }}
    >
      <div className="flex justify-center flex-nowrap" style={{ gap: `${tileGap}px` }}>
      {teeth.map((tooth) => {
        const { fill, border, dashed, ring } = getTileStyle(tooth);
        const numColor = dashed ? '#94A3B8' : border;
        const displayNum = numberingSystem === 'univ' ? toUniv(tooth) : tooth;

        return (
          <div key={tooth} className="relative flex flex-col items-center gap-1.5 overflow-visible">
            <button
              onClick={() => onToothClick(tooth)}
              onMouseEnter={(event) => {
                const rect = event.currentTarget.getBoundingClientRect();
                setHoveredTooth(tooth);
                setTooltipAnchor({
                  x: rect.left + rect.width / 2,
                  y: rect.top - 22,
                });
              }}
              onMouseLeave={() => setHoveredTooth((current) => (current === tooth ? null : current))}
              className={`flex items-center justify-center border-2 transition-all ${selectedTooth === tooth ? 'ring-2 ring-[#2563EB]/35 ring-offset-2 ring-offset-white' : ''
                }`}
              style={{
                width: `${tileWidth}px`,
                height: `${tileHeight}px`,
                background: fill,
                borderColor: border,
                borderStyle: dashed ? 'dashed' : 'solid',
                boxShadow: ring ? '0 0 0 1.5px #DC2626 inset, 0 4px 10px rgba(15,23,42,0.06)' : '0 4px 10px rgba(15,23,42,0.05)',
                borderRadius: 18,
              }}
            >
              <img
                src={getOdontoImg(tooth)}
                alt={`Tooth ${tooth}`}
                className="object-contain pointer-events-none"
                style={{
                  width: `${imageWidth}px`,
                  height: `${imageHeight}px`,
                  transform: shouldMirrorTooth(tooth) ? 'scaleX(-1)' : 'none',
                  filter: dashed
                    ? 'brightness(0) saturate(0)'
                    : 'brightness(0) invert(1) drop-shadow(0 0 0.55px rgba(255,255,255,1)) drop-shadow(0 0 0.55px rgba(255,255,255,1))',
                  opacity: dashed ? 0.9 : 0.98,
                }}
                onError={(e) => {
                  const svg = encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="50"><rect width="40" height="50" rx="10" ry="10" fill="white" stroke="#94A3B8" stroke-width="2"/><line x1="10" y1="10" x2="30" y2="40" stroke="#94A3B8" stroke-width="2"/></svg>');
                  (e.currentTarget as HTMLImageElement).src = `data:image/svg+xml,${svg}`;
                }}
              />
            </button>
            <span className="text-[25px] font-semibold leading-none" style={{ color: numColor }}>
              {displayNum}
            </span>
          </div>
        );
      })}
      </div>
    </div>
  );

  return (
    <div className="px-2 py-2">
      <div className="mx-auto flex w-fit max-w-full items-stretch gap-3">
        <div className="flex w-[44px] shrink-0 items-center justify-center rounded-2xl border border-[#1E293B] bg-[#0B1220] shadow-[inset_0_0_0_1px_rgba(148,163,184,0.08)]">
          <span className="text-center text-[27px] font-bold tracking-[0.2em] text-[#94A3B8]">
            R
          </span>
        </div>
        <div className="flex flex-col" style={{ gap: `${rowGap}px` }}>
          {renderRow(upperTeeth)}
          {renderRow(lowerTeeth)}
        </div>
        <div className="flex w-[44px] shrink-0 items-center justify-center rounded-2xl border border-[#1E293B] bg-[#0B1220] shadow-[inset_0_0_0_1px_rgba(148,163,184,0.08)]">
          <span className="text-center text-[27px] font-bold tracking-[0.2em] text-[#94A3B8]">
            L
          </span>
        </div>
      </div>
      {hoveredTooth && hoveredTooltip && typeof document !== 'undefined' ? createPortal(
        <div
          style={{
            position: 'fixed',
            pointerEvents: 'none',
            left: tooltipAnchor.x,
            top: tooltipAnchor.y,
            zIndex: 5000,
            minWidth: 220,
            padding: '12px',
            textAlign: 'left',
            transform: 'translate(-50%, -100%)',
            borderRadius: 18,
            border: '1px solid rgba(103, 232, 249, 0.35)',
            backgroundColor: '#16203f',
            boxShadow: '0 22px 46px rgba(2, 6, 23, 0.52)',
          }}
        >
          <div
            style={{
              marginBottom: 8,
              fontSize: 10,
              fontWeight: 900,
              letterSpacing: '0.18em',
              textTransform: 'uppercase',
              color: '#67E8F9',
            }}
          >
            Tooth #{numberingSystem === 'univ' ? toUniv(hoveredTooth) : hoveredTooth}
          </div>
          <div className="space-y-1.5 text-[11px] text-slate-200">
            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-400">Status</span>
              <span className="font-semibold text-white">{hoveredTooltip.status || '-'}</span>
            </div>
            {hoveredTooltip.kind === 'implant' ? (
              <>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">Space X</span>
                  <span className="font-semibold text-white">{formatValue(hoveredTooltip.gapMm, ' mm')}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">Axis To Nerve</span>
                  <span className="font-semibold text-white">{formatValue(hoveredTooltip.nerveDistMm, ' mm')}</span>
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">PBL / Level</span>
                  <span className="font-semibold text-white">
                    {hoveredTooltip.pblPct !== null && hoveredTooltip.pblPct !== undefined
                      ? `${formatValue(hoveredTooltip.pblPct, '%')} / ${hoveredTooltip.level ?? '-'}`
                      : '-'}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">Caries</span>
                  <span className="font-semibold text-white">{hoveredTooltip.caries ? 'Yes' : 'No'}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">Periapical</span>
                  <span className="font-semibold text-white">{hoveredTooltip.periapical ? 'Yes' : 'No'}</span>
                </div>
              </>
            )}
          </div>
          <div
            style={{
              position: 'absolute',
              left: '50%',
              bottom: 0,
              width: 12,
              height: 12,
              transform: 'translate(-50%, 50%) rotate(45deg)',
              borderRight: '1px solid rgba(103, 232, 249, 0.35)',
              borderBottom: '1px solid rgba(103, 232, 249, 0.35)',
              backgroundColor: '#16203f',
            }}
          />
        </div>,
        document.body
      ) : null}
    </div>
  );
}

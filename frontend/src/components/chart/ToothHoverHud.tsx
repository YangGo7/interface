import { createPortal } from 'react-dom';

type ToothHoverAnchor = {
  toothFdi: string;
  x: number;
  y: number;
};

type ToothHoverPanelData = {
  toothFdi: string;
  title: string;
  kind: 'finding' | 'implant' | 'planning';
  status: string;
  pblPct?: number | null;
  level?: number | string | null;
  cariesProb?: number | null;
  periapicalProb?: number | null;
  primaryLabel?: string | null;
  primaryProb?: number | null;
  diameterMm?: number | null;
  lengthMm?: number | null;
  gapMm?: number | null;
  centerToNerveMm?: number | null;
};

type ToothHoverHudProps = {
  anchor: ToothHoverAnchor | null;
  panel: ToothHoverPanelData | null;
  minWidth: string;
  borderWidth: string;
  paddingY: string;
  paddingX: string;
  titleFontSize: string;
  labelFontSize: string;
  valueFontSize: string;
  titleGap: string;
  labelGap: string;
  columnGap: string;
  rowGap: string;
};

const formatProbability = (value?: number | null) => {
  if (value === null || value === undefined) return '-';
  const normalized = value <= 1 ? value * 100 : value;
  return `${normalized.toFixed(1)}%`;
};

const formatMetric = (value?: number | null, unit = ' mm') => {
  if (value === null || value === undefined) return `-${unit}`;
  return `${value.toFixed(1)}${unit}`;
};

export function ToothHoverHud({
  anchor,
  panel,
  minWidth,
  borderWidth,
  paddingY,
  paddingX,
  titleFontSize,
  labelFontSize,
  valueFontSize,
  titleGap,
  labelGap,
  columnGap,
  rowGap,
}: ToothHoverHudProps) {
  if (!anchor || !panel || typeof document === 'undefined') return null;

  return createPortal(
    <div
      style={{
        position: 'fixed',
        left: anchor.x,
        top: anchor.y,
        transform: 'translate(-50%, -100%)',
        pointerEvents: 'none',
        zIndex: 120,
      }}
    >
      <div
        style={{
          minWidth,
          background: '#2F2F2F',
          border: `${borderWidth} solid #3D3D3D`,
          padding: `${paddingY} ${paddingX}`,
          boxShadow: '0 12px 28px rgba(0, 0, 0, 0.32)',
        }}
      >
        <div
          style={{
            color: '#67E8F9',
            fontSize: titleFontSize,
            fontWeight: 900,
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            marginBottom: titleGap,
          }}
        >
          {panel.title}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: rowGap }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: columnGap }}>
            <div style={{ color: '#FFFFFF', fontSize: labelFontSize }}>Status</div>
            <div style={{ color: '#FFFFFF', fontSize: valueFontSize, fontWeight: 700 }}>{panel.status}</div>
          </div>
          {panel.kind === 'implant' ? (
            <>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: columnGap }}>
                <div style={{ color: '#FFFFFF', fontSize: labelFontSize }}>Diameter</div>
                <div style={{ color: '#FFFFFF', fontSize: valueFontSize, fontWeight: 700 }}>
                  {formatMetric(panel.diameterMm)}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: columnGap }}>
                <div style={{ color: '#FFFFFF', fontSize: labelFontSize }}>Length</div>
                <div style={{ color: '#FFFFFF', fontSize: valueFontSize, fontWeight: 700 }}>
                  {formatMetric(panel.lengthMm)}
                </div>
              </div>
            </>
          ) : panel.kind === 'planning' ? (
            <>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: columnGap }}>
                <div style={{ color: '#FFFFFF', fontSize: labelFontSize }}>Space Width</div>
                <div style={{ color: '#FFFFFF', fontSize: valueFontSize, fontWeight: 700 }}>
                  {formatMetric(panel.gapMm)}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: columnGap }}>
                <div style={{ color: '#FFFFFF', fontSize: labelFontSize }}>Center To Nerve</div>
                <div style={{ color: '#FFFFFF', fontSize: valueFontSize, fontWeight: 700 }}>
                  {formatMetric(panel.centerToNerveMm)}
                </div>
              </div>
            </>
          ) : (
            <>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: columnGap }}>
                <div style={{ color: '#FFFFFF', fontSize: labelFontSize }}>PBL / Level</div>
                <div style={{ color: '#FFFFFF', fontSize: valueFontSize, fontWeight: 700 }}>
                  {panel.pblPct !== null && panel.pblPct !== undefined
                    ? `${panel.pblPct.toFixed(1)}% / ${panel.level ?? '-'}`
                    : '-'}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: columnGap }}>
                <div style={{ color: '#FFFFFF', fontSize: labelFontSize }}>Caries</div>
                <div style={{ color: '#FFFFFF', fontSize: valueFontSize, fontWeight: 700 }}>
                  {formatProbability(panel.cariesProb)}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: columnGap }}>
                <div style={{ color: '#FFFFFF', fontSize: labelFontSize }}>Periapical</div>
                <div style={{ color: '#FFFFFF', fontSize: valueFontSize, fontWeight: 700 }}>
                  {formatProbability(panel.periapicalProb)}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: columnGap }}>
                <div style={{ color: '#FFFFFF', fontSize: labelFontSize }}>Primary</div>
                <div style={{ color: '#FFFFFF', fontSize: valueFontSize, fontWeight: 700 }}>
                  {panel.primaryLabel
                    ? `${panel.primaryLabel}${panel.primaryProb !== null && panel.primaryProb !== undefined ? ` / ${formatProbability(panel.primaryProb)}` : ''}`
                    : '-'}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
}

type LegendKey = 'healthy' | 'requires' | 'warning' | 'implant' | 'missing';

type LegendItem = {
  key: LegendKey;
  label: string;
  color: string;
  top: number;
};

type DentalChartLegendOverlayProps = {
  visible: boolean;
  left: string;
  top: string;
  width: number;
  height: number;
  items: readonly LegendItem[];
  activeKey: LegendKey | null;
  focusedKey: LegendKey | null;
  counts: Record<LegendKey, number>;
  activeBorderWidth: string;
  hoverBorderWidth: string;
  glowSize: string;
  badgeFontSize: string;
  labelFontSize: string;
  countGap: string;
  onToggle: (key: LegendKey) => void;
};

const relativePercent = (value: number, total: number) => `${(value / total) * 100}%`;

export function DentalChartLegendOverlay({
  visible,
  left,
  top,
  width,
  height,
  items,
  activeKey,
  focusedKey,
  counts,
  activeBorderWidth,
  hoverBorderWidth,
  glowSize,
  badgeFontSize,
  labelFontSize,
  countGap,
  onToggle,
}: DentalChartLegendOverlayProps) {
  if (!visible) return null;

  return (
    <div style={{ width: `${width}px`, height: `${height}px`, left, top, position: 'absolute' }}>
      {items.map((item) => (
        <div key={item.label}>
          <button
            type="button"
            onClick={() => onToggle(item.key)}
            aria-pressed={activeKey === item.key}
            style={{
              width: relativePercent(19, width),
              height: relativePercent(19, height),
              left: 0,
              top: relativePercent(item.top - 834, height),
              position: 'absolute',
              background: '#808181',
              border: activeKey === item.key
                ? `${activeBorderWidth} solid #00C0F3`
                : focusedKey === item.key
                  ? `${hoverBorderWidth} solid rgba(0, 192, 243, 0.7)`
                  : 'none',
              boxShadow: activeKey === item.key ? `0 0 ${glowSize} rgba(0, 192, 243, 0.35)` : 'none',
              cursor: 'pointer',
              padding: 0,
            }}
          >
            {(activeKey === item.key || focusedKey === item.key) && (
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: item.key === 'healthy' ? '#111111' : '#ffffff',
                  fontSize: badgeFontSize,
                  fontWeight: 700,
                }}
              >
                ??
              </div>
            )}
          </button>
          <div
            style={{
              width: relativePercent(52, width),
              height: relativePercent(19, height),
              left: relativePercent(width - 52 - 12, width),
              top: relativePercent(item.top - 834, height),
              position: 'absolute',
              background: item.color,
            }}
          />
          <div
            style={{
              left: relativePercent(30, width),
              top: relativePercent(item.top - 834, height),
              position: 'absolute',
              color: 'white',
              fontSize: labelFontSize,
              fontWeight: 700,
            }}
          >
            {item.label}
            <span style={{ color: '#B7B7B7', marginLeft: countGap }}>{counts[item.key]}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

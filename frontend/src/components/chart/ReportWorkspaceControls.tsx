type ReportWorkspaceControlsProps = {
  showReportButton: boolean;
  reportButtonLeft: string;
  reportButtonTop: string;
  reportButtonWidth: string;
  reportButtonHeight: string;
  reportActive: boolean;
  outlineWidth: string;
  outlineOffset: string;
  glowSize: string;
  activeIconSrc: string;
  inactiveIconSrc: string;
  onOpenReport: () => void;
  reportError: string | null;
  reportErrorLeft: string;
  reportErrorTop: string;
  reportErrorFontSize: string;
  showChartToggle: boolean;
  chartToggleLeft: string;
  chartToggleTop: string;
  chartToggleWidth: string;
  chartToggleHeight: string;
  chartToggleBorderWidth: string;
  chartToggleFontSize: string;
  markerIconSrc: string;
  markerIconWidth: string;
  markerIconHeight: string;
  markerIconLeft: string;
  markerIconTop: string;
  chartToggleLabelMarginLeft: string;
  onShowChart: () => void;
};

export function ReportWorkspaceControls({
  showReportButton,
  reportButtonLeft,
  reportButtonTop,
  reportButtonWidth,
  reportButtonHeight,
  reportActive,
  outlineWidth,
  outlineOffset,
  glowSize,
  activeIconSrc,
  inactiveIconSrc,
  onOpenReport,
  reportError,
  reportErrorLeft,
  reportErrorTop,
  reportErrorFontSize,
  showChartToggle,
  chartToggleLeft,
  chartToggleTop,
  chartToggleWidth,
  chartToggleHeight,
  chartToggleBorderWidth,
  chartToggleFontSize,
  markerIconSrc,
  markerIconWidth,
  markerIconHeight,
  markerIconLeft,
  markerIconTop,
  chartToggleLabelMarginLeft,
  onShowChart,
}: ReportWorkspaceControlsProps) {
  return (
    <>
      {showReportButton && (
        <button
          type="button"
          onClick={onOpenReport}
          aria-pressed={reportActive}
          aria-label="Open report panel"
          title="Report panel"
          style={{
            width: reportButtonWidth,
            height: reportButtonHeight,
            left: reportButtonLeft,
            top: reportButtonTop,
            position: 'absolute',
            zIndex: 30,
            borderRadius: '50%',
            outline: reportActive ? `${outlineWidth} solid #00C0F3` : 'none',
            outlineOffset,
            boxShadow: reportActive ? `0 0 ${glowSize} rgba(0, 192, 243, 0.45)` : 'none',
            cursor: 'pointer',
          }}
        >
          <img
            style={{
              width: '100%',
              height: '100%',
              display: 'block',
              filter: reportActive ? 'drop-shadow(0 0 10px rgba(0, 192, 243, 0.7))' : 'none',
            }}
            src={reportActive ? activeIconSrc : inactiveIconSrc}
            alt=""
          />
        </button>
      )}

      {reportError && (
        <div
          style={{
            left: reportErrorLeft,
            top: reportErrorTop,
            position: 'absolute',
            zIndex: 95,
            maxWidth: 'min(420px, calc(100% - 32px))',
            padding: '10px 12px',
            borderRadius: '12px',
            border: '1px solid rgba(255, 104, 104, 0.42)',
            background: 'rgba(37, 10, 14, 0.88)',
            boxShadow: '0 14px 30px rgba(0, 0, 0, 0.35)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
          }}
        >
          <div
            style={{
              color: '#FF8E8E',
              fontSize: '10px',
              fontWeight: 800,
              letterSpacing: '0.16em',
              textTransform: 'uppercase',
            }}
          >
            Viewer HUD
          </div>
          <div
            style={{
              marginTop: '4px',
              color: '#FFE0E0',
              fontSize: reportErrorFontSize,
              fontWeight: 700,
              lineHeight: 1.4,
            }}
          >
            {reportError}
          </div>
        </div>
      )}

      {showChartToggle && (
        <button
          type="button"
          onClick={onShowChart}
          aria-pressed={false}
          aria-label="Show dental chart"
          title="Show dental chart"
          style={{
            width: chartToggleWidth,
            height: chartToggleHeight,
            left: chartToggleLeft,
            top: chartToggleTop,
            position: 'absolute',
            border: `${chartToggleBorderWidth} solid #4C4C4C`,
            background: '#5C5C5C',
            color: '#FFFFFF',
            fontSize: chartToggleFontSize,
            fontWeight: 700,
            cursor: 'pointer',
            zIndex: 4,
            padding: 0,
            textAlign: 'left',
          }}
        >
          <img
            src={markerIconSrc}
            alt=""
            draggable={false}
            style={{
              width: markerIconWidth,
              height: markerIconHeight,
              position: 'absolute',
              left: markerIconLeft,
              top: markerIconTop,
            }}
          />
          <span style={{ marginLeft: chartToggleLabelMarginLeft, display: 'inline-block' }}>Dental Chart</span>
        </button>
      )}
    </>
  );
}

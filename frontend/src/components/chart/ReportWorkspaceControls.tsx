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
            color: '#FFB4B4',
            fontSize: reportErrorFontSize,
            fontWeight: 700,
            zIndex: 20,
          }}
        >
          {reportError}
        </div>
      )}

      {showChartToggle && (
        <button
          type="button"
          onClick={onShowChart}
          aria-pressed={false}
          aria-label="Show dental chart"
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
